# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Differentiable joint vision/action forwards for Cosmos WAM policies."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass
class WAMContext:
    """Frozen conditioning and tensor layout for one uniform WAM batch."""

    sequence_plans: list[Any]
    clean: Any
    text_tokens: list[list[int]]
    state: torch.Tensor
    reference: torch.Tensor
    condition_mask: torch.Tensor
    free_mask: torch.Tensor
    action_mask: torch.Tensor
    vision_counts: tuple[int, ...]
    vision_shapes: tuple[torch.Size, ...]
    action_shape: torch.Size
    action_offset: int
    hidden_size: int


@torch.no_grad()
def prepare_context(
    omni: torch.nn.Module, batch: dict[str, Any], seeds: list[int]
) -> WAMContext:
    """Encode fixed observations using the native Cosmos inference preparation.

    The returned layout is ``[vision | action]`` for each sample. Only uniform
    batches without sound or LiDAR are accepted by the WAM policy interface.
    """
    (
        plans,
        clean,
        text_tokens,
        _unconditional_tokens,
        initial,
        reference,
        condition,
        has_noisy_actions,
    ) = omni._prepare_inference_data(batch, seeds, False)
    batch_size = len(initial)
    if not has_noisy_actions or not all(plan.has_action for plan in plans):
        raise ValueError("WAM PPO requires generated action tokens in every sample.")
    if getattr(clean, "x0_tokens_lidar", None) is not None or any(
        getattr(plan, "has_sound", False) for plan in plans
    ):
        raise ValueError(
            "WAM PPO supports joint vision/action batches without LiDAR or sound."
        )
    if batch_size == 0 or len(seeds) != batch_size:
        raise ValueError(
            "Expected one seed per nonempty WAM batch sample, "
            f"got {len(seeds)} seeds and {batch_size} samples."
        )
    if clean.x0_tokens_vision is None or clean.x0_tokens_action is None:
        raise ValueError("WAM PPO requires both encoded vision and action tensors.")
    if len(clean.x0_tokens_action) != batch_size:
        raise ValueError("WAM PPO requires one action tensor per sample.")
    action_shape = clean.x0_tokens_action[0].shape
    if len(action_shape) != 2 or any(
        x.shape != action_shape for x in clean.x0_tokens_action
    ):
        raise ValueError(
            "WAM PPO requires uniform action tensors with shape [horizon, dimension]."
        )
    counts = tuple(clean.num_vision_items_per_sample or [1] * batch_size)
    vision_shapes = tuple(x.shape for x in clean.x0_tokens_vision)
    offsets = []
    vision_index = 0
    for count in counts:
        if count < 1:
            raise ValueError("WAM PPO requires at least one vision item per sample.")
        offsets.append(
            sum(
                clean.x0_tokens_vision[j].numel()
                for j in range(vision_index, vision_index + count)
            )
        )
        vision_index += count
    if vision_index != len(vision_shapes) or len(set(offsets)) != 1:
        raise ValueError(
            "WAM PPO requires the same flattened vision size in every sample."
        )
    action_offset = offsets[0]
    expected_size = action_offset + clean.x0_tokens_action[0].numel()
    if any(x.numel() != expected_size for x in initial):
        raise ValueError(
            "Expected native WAM states to contain only vision and action tensors."
        )

    state = torch.stack(initial).float()
    reference_tensor = torch.stack(reference).to(state)
    condition_tensor = torch.stack(condition).to(state)
    action_mask = torch.zeros_like(state, dtype=torch.bool)
    horizon, action_dim = action_shape
    for i in range(batch_size):
        active = torch.ones(
            (horizon, action_dim), dtype=torch.bool, device=state.device
        )
        if clean.raw_action_dim is not None and clean.raw_action_dim[i] is not None:
            raw_dim = int(clean.raw_action_dim[i])
            if not 0 < raw_dim <= action_dim:
                raise ValueError(
                    f"Expected raw action dimension in [1, {action_dim}], got {raw_dim}."
                )
            active[:, raw_dim:] = False
        valid_masks = getattr(clean, "action_valid_mask", None)
        if valid_masks is not None and valid_masks[i] is not None:
            valid = valid_masks[i].to(device=state.device, dtype=torch.bool).reshape(-1)
            if valid.numel() > action_dim:
                raise ValueError(
                    "Action validity mask is wider than the generated action tensor."
                )
            active[:, : valid.numel()] &= valid
        active &= ~condition_tensor[i, action_offset:].bool().reshape(action_shape)
        action_mask[i, action_offset:] = active.reshape(-1)
    free_mask = ~condition_tensor.bool()
    free_mask[:, action_offset:] &= action_mask[:, action_offset:]
    return WAMContext(
        sequence_plans=plans,
        clean=clean,
        text_tokens=text_tokens,
        state=state,
        reference=reference_tensor,
        condition_mask=condition_tensor,
        free_mask=free_mask,
        action_mask=action_mask,
        vision_counts=counts,
        vision_shapes=vision_shapes,
        action_shape=action_shape,
        action_offset=action_offset,
        hidden_size=int(omni.net.hidden_size),
    )


def enforce_condition(context: WAMContext, state: torch.Tensor) -> torch.Tensor:
    """Keep observed latents fixed and invalid action coordinates equal to zero."""
    _check_state(context, state)
    result = state * context.free_mask.to(state.dtype)
    return result + context.condition_mask.to(state.dtype) * context.reference.to(state)


def action_latents(context: WAMContext, state: torch.Tensor) -> torch.Tensor:
    """Return the padded normalized action portion of a flat WAM state."""
    _check_state(context, state)
    return state[:, context.action_offset :].reshape(
        state.shape[0], *context.action_shape
    )


def predict_velocity(
    omni: torch.nn.Module,
    context: WAMContext,
    state: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict joint velocity and pooled action hidden states without sampling.

    ``timestep`` uses normalized Cosmos time: one is noise and zero is data.
    The cached ``state`` is consumed unchanged, including observed vision
    latents, so actor replay evaluates the rollout's exact internal state.
    """
    _check_state(context, state)
    batch_size = state.shape[0]
    times = timestep.to(device=state.device, dtype=torch.float32).reshape(-1)
    if times.numel() == 1:
        times = times.expand(batch_size)
    if times.numel() != batch_size:
        raise ValueError(f"Expected {batch_size} sampling times, got {times.numel()}.")
    vision = []
    actions = []
    vision_index = 0
    for i, count in enumerate(context.vision_counts):
        offset = 0
        for _ in range(count):
            shape = context.vision_shapes[vision_index]
            size = context.clean.x0_tokens_vision[vision_index].numel()
            vision.append(state[i, offset : offset + size].reshape(shape))
            offset += size
            vision_index += 1
        actions.append(state[i, offset:].reshape(context.action_shape))
    clean = replace(context.clean, x0_tokens_vision=vision, x0_tokens_action=actions)
    train_steps = float(omni.config.rectified_flow_inference_config.num_train_timesteps)
    packed = omni._pack_input_sequence(
        context.sequence_plans,
        context.text_tokens,
        clean,
        (times * train_steps).cpu(),
        include_end_of_generation_token=omni._derive_include_end_of_generation_token(),
        skip_text_tokens=False,
    )
    # Match the native inference boundary: packing owns metadata, while noisy
    # token values are explicitly cast to the denoiser's configured precision.
    packed.vision.tokens = [x.to(**omni.tensor_kwargs) for x in vision]
    packed.action.tokens = [x.to(**omni.tensor_kwargs) for x in actions]
    packed.action.domain_id = context.clean.action_domain_id
    packed.to_cuda()
    output = omni.net(
        packed_seq=packed,
        memory=None,
        video_temporal_causal=None,
        correct_cp_gradients=omni.config.correct_cp_gradients,
    )
    flat_velocity = []
    vision_index = 0
    for i, count in enumerate(context.vision_counts):
        parts = [
            output["preds_vision"][j].reshape(-1)
            for j in range(vision_index, vision_index + count)
        ]
        parts.append(output["preds_action"][i].reshape(-1))
        flat_velocity.append(torch.cat(parts))
        vision_index += count
    velocity = torch.stack(flat_velocity).float() * context.free_mask.to(state.dtype)

    hidden = output["last_hidden_state"][packed.action.mse_loss_indexes]
    counts = [len(indexes) for indexes in packed.action.noisy_frame_indexes]
    if len(counts) != batch_size or any(count == 0 for count in counts):
        raise ValueError("Expected generated action hidden states in every WAM sample.")
    features = torch.stack([part.mean(dim=0) for part in hidden.split(counts)])
    return velocity, features


def _check_state(context: WAMContext, state: torch.Tensor) -> None:
    if state.shape != context.state.shape:
        raise ValueError(
            f"Expected WAM state shape {context.state.shape}, got {state.shape}."
        )
