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

import json
import random
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Literal, Optional, Union
from unittest.mock import patch

import numpy as np
import torch
from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7, Gr00tN1d7ActionHead
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor
from torch.distributions import Normal
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.gr00t.simulation_io import (
    ACTION_CONVERSION_N1D7,
    OBS_CONVERSION,
)
from rlinf.models.embodiment.gr00t.utils import (
    squeeze_dict_values,
    unsqueeze_dict_values,
)
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

logger = get_logger()


@contextmanager
def redirect_qwen3_backbone_to_local(canonical_name: str, local_path: str | None):
    if local_path is None:
        yield
        return

    local_path = Path(local_path).expanduser().resolve()
    if not local_path.is_dir():
        raise FileNotFoundError(f"Backbone model path does not exist: {local_path}")

    original_model_from_pretrained = Qwen3VLForConditionalGeneration.from_pretrained
    original_processor_from_pretrained = Qwen3VLProcessor.from_pretrained

    def _make_local_redirect(original):
        def from_pretrained_with_local_redirect(model_name, *args, **kwargs):
            if str(model_name) == canonical_name:
                model_name = str(local_path)
                kwargs["local_files_only"] = True
            return original(model_name, *args, **kwargs)

        return from_pretrained_with_local_redirect

    # Redirect both the backbone weights and its processor (image processor +
    # tokenizer): GR00T's Gr00tN1d7DataCollator builds the processor via the
    # canonical hub name, which would otherwise hit the Hub and fail offline.
    with (
        patch.object(
            Qwen3VLForConditionalGeneration,
            "from_pretrained",
            side_effect=_make_local_redirect(original_model_from_pretrained),
        ),
        patch.object(
            Qwen3VLProcessor,
            "from_pretrained",
            side_effect=_make_local_redirect(original_processor_from_pretrained),
        ),
    ):
        yield


def _find_processor_dir(model_path: Path) -> Path | None:
    """Find the local directory containing GR00T N1.7 processor files."""
    processor_required_files = (
        "processor_config.json",
        "statistics.json",
        "embodiment_id.json",
    )
    for candidate in (model_path / "processor", model_path):
        if candidate.is_dir() and all(
            (candidate / f).is_file() for f in processor_required_files
        ):
            return candidate
    return None


# Keys produced during rollout that must never be replayed as backbone inputs by
# the actor (they are RL bookkeeping rather than model inputs).
_FORWARD_INPUT_SKIP_KEYS = {
    "advantages",
    "returns",
    "values",
    "prev_values",
    "prev_logprobs",
    "old_values",
    "loss_mask",
    "loss_mask_sum",
    "chains",
    "denoise_inds",
}

_FORWARD_INPUT_MODEL_KEYS = {
    "state",
    "state_mask",
    "action",
    "action_mask",
    "embodiment_id",
    "input_ids",
    "attention_mask",
    "pixel_values",
    "image_grid_thw",
    "image_sizes",
}


def _resolve_env_action_dim(action_dim: int | None, valid_action_dim: int) -> int:
    """Resolve the environment-facing action dim, clamped by ``valid_action_dim``."""
    env_action_dim = valid_action_dim if action_dim is None else int(action_dim)
    assert env_action_dim <= int(valid_action_dim), (
        f"Configured action_dim ({env_action_dim}) exceeds valid_action_dim "
        f"({valid_action_dim})."
    )
    return env_action_dim


def _reshape_forward_tensor(key: str, value: Any) -> Any:
    """Normalize rollout-stashed tensors back to backbone-friendly shapes."""
    if not torch.is_tensor(value):
        return value

    if key == "pixel_values" and value.ndim > 4:
        return value.reshape(-1, *value.shape[-3:])
    if key in {"image_grid_thw", "image_sizes"} and value.ndim > 2:
        return value.reshape(-1, value.shape[-1])
    return value


def _canonicalize_gr00t_text_forward_inputs(
    forward_inputs: dict[str, Any],
    padding_value: int,
) -> dict[str, Any]:
    """Right-pad ``input_ids`` and ``attention_mask`` to ``padding_value``."""
    canonicalized = dict(forward_inputs)

    for key in ("input_ids", "attention_mask"):
        tensor = canonicalized.get(key)
        if tensor is None:
            continue
        if not torch.is_tensor(tensor):
            raise TypeError(
                f"Expected GR00T text field '{key}' to be a tensor, "
                f"got {type(tensor).__name__}."
            )
        if tensor.ndim < 2:
            raise ValueError(
                f"Expected GR00T text field '{key}' to be at least 2D, "
                f"got shape {tuple(tensor.shape)}."
            )
        if padding_value > 0 and tensor.shape[-1] > padding_value:
            raise ValueError(
                f"GR00T text field '{key}' length {tensor.shape[-1]} exceeds "
                f"padding_value={padding_value}."
            )
        if padding_value > 0 and tensor.shape[-1] < padding_value:
            tensor = torch.nn.functional.pad(
                tensor,
                pad=(0, padding_value - tensor.shape[-1]),
                mode="constant",
                value=0,
            )
        canonicalized[key] = tensor

    return canonicalized


def _normalize_gr00t_forward_inputs(forward_inputs: dict[str, Any]) -> dict[str, Any]:
    """Convert cached actor ``forward_inputs`` back into backbone inputs.

    Drops RL bookkeeping keys, restores flattened visual shapes, and synthesizes
    a default ``state_mask`` when missing.
    """
    normalized_input = {}
    for key, value in forward_inputs.items():
        if key in _FORWARD_INPUT_SKIP_KEYS:
            continue
        if key not in _FORWARD_INPUT_MODEL_KEYS:
            continue
        normalized_input[key] = _reshape_forward_tensor(key, value)

    state = normalized_input.get("state")
    if "state_mask" not in normalized_input and torch.is_tensor(state):
        normalized_input["state_mask"] = torch.ones(
            state.shape[:-1], dtype=torch.bool, device=state.device
        )

    return {key: value for key, value in normalized_input.items() if value is not None}


def _batchify_gr00t_forward_input(
    key: str,
    value: Any,
    batch_size: int,
) -> Any:
    """Store rollout forward inputs with an explicit batch dimension.

    Some GR00T processor outputs, especially visual fields such as
    ``pixel_values`` and ``image_grid_thw``, are emitted in flattened
    backbone-friendly shapes like ``[num_patches, hidden]`` or
    ``[num_images, 3]``. Those shapes are correct for immediate inference, but
    once cached into trajectory buffers they get treated as if dim-0 were the
    env batch dimension and are later sliced incorrectly. We therefore restore a
    leading batch axis before stashing them, and flatten them back inside
    :func:`_normalize_gr00t_forward_inputs` when the actor consumes them.
    """
    if not torch.is_tensor(value) or batch_size <= 0:
        return value

    if key == "pixel_values" and value.ndim >= 2 and value.shape[0] != batch_size:
        if value.shape[0] % batch_size != 0:
            raise ValueError(
                f"{key} leading dim {value.shape[0]} is not divisible by batch size {batch_size}"
            )
        return value.reshape(batch_size, value.shape[0] // batch_size, *value.shape[1:])

    if key in {"image_grid_thw", "image_sizes"} and value.ndim >= 2:
        if value.shape[0] != batch_size:
            if value.shape[0] % batch_size != 0:
                raise ValueError(
                    f"{key} leading dim {value.shape[0]} is not divisible by batch size {batch_size}"
                )
            return value.reshape(
                batch_size, value.shape[0] // batch_size, value.shape[-1]
            )

    return value


def _tensorize_forward_input(value: Any) -> Any:
    """Convert list-valued cached inputs into tensors."""
    if not isinstance(value, list):
        return value
    if len(value) == 0:
        return torch.tensor(value)
    if torch.is_tensor(value[0]):
        return torch.stack(value)
    return torch.tensor(value)


class FlowMatchingActionHeadForRLActionPrediction(Gr00tN1d7ActionHead):
    """Flow-matching action head with RL extensions for GR00T N1.7.

    Extends the upstream :class:`Gr00tN1d7ActionHead` with:

    * stochastic (flow-SDE) denoising for exploration,
    * per-denoising-step Gaussian log-probabilities, and
    * an optional value head for actor-critic style training.
    """

    def __init__(
        self,
        config: Any,  # Gr00tN1d7Config
        rl_head_config: dict[str, Any],
        output_action_chunks: int = 1,
    ):
        super().__init__(config)
        self.config = config
        self.rl_config = rl_head_config
        # Only set defaults if not already specified in config.
        if "noise_method" not in self.rl_config:
            self.rl_config["noise_method"] = "flow_sde"
        if "noise_level" not in self.rl_config:
            self.rl_config["noise_level"] = 0.5
        if "noise_anneal" not in self.rl_config:
            self.rl_config["noise_anneal"] = False
        self.padding_value = rl_head_config.get("padding_value", 0)
        self.output_action_chunks = output_action_chunks
        # Keep the upstream diffusion/action-head width separate from the
        # environment-facing action width inferred from modality metadata.
        self.model_action_dim = getattr(
            config, "max_action_dim", getattr(config, "action_dim", 7)
        )
        self.valid_action_dim = self.model_action_dim
        self.env_action_dim = _resolve_env_action_dim(
            getattr(config, "action_dim", self.valid_action_dim),
            self.valid_action_dim,
        )
        self.action_chunk = output_action_chunks
        self.hidden_size = getattr(
            config, "hidden_size", getattr(self, "hidden_size", 1024)
        )
        self.action_horizon = getattr(
            config, "action_horizon", getattr(self, "action_horizon", 16)
        )
        self.num_timestep_buckets = getattr(
            config, "num_timestep_buckets", getattr(self, "num_timestep_buckets", 1000)
        )
        self.num_inference_timesteps = getattr(
            config,
            "num_inference_timesteps",
            getattr(self, "num_inference_timesteps", 4),
        )

        vlm_width = getattr(config, "backbone_embedding_dim", 2048)
        state_width = getattr(config, "input_embedding_dim", 1536)
        if self.rl_config.get("use_vlm_value", False):
            proj_width = vlm_width
        else:
            proj_width = vlm_width + state_width

        if self.rl_config.get("add_value_head", False):
            self.value_head = ValueHead(
                input_dim=proj_width,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )

        if self.rl_config.get("noise_method") == "reinflow":
            self.reinflow_explore_noise_net = ExploreNoiseNet(
                in_dim=self.hidden_size,
                out_dim=getattr(config, "max_action_dim", 7),
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[0.08, 0.16],
                noise_scheduler_type="learn",
            )

    def _get_component(self, name: str):
        """Return a named submodule of the head, or ``None`` if absent."""
        return getattr(self, name, None)

    def _process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        """Apply the optional VL layer-norm and self-attention refinements."""
        if not hasattr(backbone_output, "backbone_features"):
            return backbone_output

        backbone_features = backbone_output.backbone_features
        vlln = self._get_component("vlln")
        if vlln is not None:
            backbone_features = vlln(backbone_features)

        vl_self_attention = self._get_component("vl_self_attention")
        if vl_self_attention is not None:
            backbone_features = vl_self_attention(backbone_features)

        backbone_output.backbone_features = backbone_features
        return backbone_output

    def _encode_state_features(
        self, action_input: BatchFeature, embodiment_id: int | torch.Tensor
    ) -> torch.Tensor:
        """Encode the proprioceptive state into the action-head feature space."""
        state = action_input.state
        # Match the official GR00T N1.7 _encode_features assertion to catch
        # shape mismatches early (e.g. state_history_length != 1).
        current_T = state.shape[1] if state.ndim >= 3 else 1
        assert current_T == self.config.state_history_length, (
            f"State time dimension {current_T} != "
            f"config.state_history_length {self.config.state_history_length}"
        )
        if state.ndim == 2:
            state = state[:, None, :]
        elif state.ndim >= 3:
            state = state.reshape(state.shape[0], 1, -1)

        state_encoder = self._get_component("state_encoder")
        if state_encoder is None:
            return state
        return state_encoder(state, embodiment_id)

    def prepare_input(self, inputs: dict) -> BatchFeature:
        """Collect the action-head relevant fields into a ``BatchFeature``."""
        action_inputs = {}
        for k in ["state", "action", "action_mask", "embodiment_id"]:
            if k in inputs:
                action_inputs[k] = inputs[k]

        return BatchFeature(data=action_inputs)

    def get_logprob_norm(self, sample, mu, sigma):
        """Gaussian log-probability of ``sample`` under ``Normal(mu, sigma)``.

        Deterministic (``sigma == 0``) coordinates contribute zero log-prob.
        """
        if self.rl_config.get("safe_get_logprob", False):
            dist = Normal(loc=mu, scale=sigma)
            return dist.log_prob(sample)
        else:
            mask = sigma == 0
            sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
            constant_term = -torch.log(sigma_safe) - 0.5 * torch.log(
                2 * torch.pi * torch.ones_like(sample)
            )
            exponent_term = -0.5 * torch.pow((sample - mu) / sigma_safe, 2)
            log_prob = constant_term + exponent_term
            log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
            return log_prob

    def sample_noise(self, shape, device, dtype=None):
        """Sample standard Gaussian exploration noise in bf16."""
        return torch.normal(mean=0.0, std=1.0, size=shape, dtype=dtype, device=device)

    def sample_mean_var_val(
        self,
        vl_embs: torch.Tensor,
        denoise_steps: int,
        x_t: torch.Tensor,
        embodiment_id: int,
        state_features: torch.Tensor,
        idx: Optional[int | torch.Tensor],
        mode: Literal["train", "eval"] = "train",
        compute_values=False,
        backbone_output: Optional[BatchFeature] = None,
    ):
        """Compute the mean and std of the posterior over the next denoising state.

        In ``eval`` mode the transition is deterministic (zero std). In ``train``
        mode with ``noise_method == "flow_sde"`` an SDE-consistent Gaussian
        perturbation is injected to enable exploration.
        """
        bsize = vl_embs.shape[0]
        device = vl_embs.device
        if isinstance(idx, int):
            idx = torch.tensor(idx).expand(bsize)

        if self.rl_config.get("noise_anneal"):
            noise_start, noise_end, anneal_steps = self.rl_config.get(
                "noise_params", (0.0, 0.0, 100)
            )
            noise_level = (
                noise_start
                + (noise_end - noise_start)
                * min(getattr(self, "global_step", 0), anneal_steps)
                / anneal_steps
            )
            noise_level = torch.tensor(noise_level).to(device)
        else:
            noise_level = torch.tensor(self.rl_config.get("noise_level", 0.5)).to(
                device
            )

        t_cont = idx / float(denoise_steps)
        timesteps_tensor = (
            (t_cont * self.num_timestep_buckets).to(torch.int64).to(device)
        )
        action_encoder = self._get_component("action_encoder")
        action_features = (
            action_encoder(x_t, timesteps_tensor, embodiment_id)
            if action_encoder is not None
            else x_t
        )
        position_embedding = self._get_component("position_embedding")
        if (
            getattr(self.config, "add_pos_embed", False)
            and position_embedding is not None
        ):
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=device
            )
            pos_embs = position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        sa_embs = torch.cat((state_features, action_features), dim=1)

        denoising_model = self._get_component("model")
        if denoising_model is not None:
            if (
                getattr(self.config, "use_alternate_vl_dit", False)
                and backbone_output is not None
            ):
                model_output = denoising_model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                    image_mask=backbone_output.image_mask,
                    backbone_attention_mask=backbone_output.backbone_attention_mask,
                )
            else:
                model_output = denoising_model(
                    hidden_states=sa_embs,
                    encoder_hidden_states=vl_embs,
                    timestep=timesteps_tensor,
                )
        else:
            model_output = sa_embs
        model_output = model_output[:, -self.action_horizon :]

        action_decoder = self._get_component("action_decoder")
        v_t = (
            action_decoder(model_output, embodiment_id)
            if action_decoder is not None
            else torch.zeros_like(model_output)
        )

        timesteps = torch.linspace(
            0, 1, denoise_steps + 1, device=device, dtype=vl_embs.dtype
        )
        t_input = timesteps[idx]
        delta = timesteps[idx + 1] - timesteps[idx]
        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)

        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input + delta)
            x1_weight = t_input + delta
            x_t_std = torch.zeros_like(t_input)
        else:
            if self.rl_config.get("noise_method") == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        (1 - timesteps)
                        / torch.where(timesteps == 0, timesteps[1], timesteps)
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = (
                    torch.ones_like(t_input)
                    - (t_input + delta)
                    - sigma_i**2 * delta / (2 * (1 - t_input))
                )
                x1_weight = t_input + delta
                x_t_std = torch.sqrt(delta) * sigma_i
            else:
                x0_weight = 1 - (t_input + delta)
                x1_weight = t_input + delta
                x_t_std = torch.zeros_like(t_input)

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std

    def get_value(self, vl_embs, state_features):
        """Estimate the state value from pooled VL and state features."""
        bsize = vl_embs.shape[0]
        mask_length = vl_embs.shape[1]
        if self.rl_config.get("value_vlm_mode") == "mean_token":
            prefix_mask = [True] * mask_length
        elif self.rl_config.get("value_vlm_mode") == "last_token":
            prefix_mask = [False] * (mask_length - 1) + [True] * 1
        elif self.rl_config.get("value_vlm_mode") == "first_token":
            prefix_mask = [True] * 1 + [False] * (mask_length - 1)
        vl_embs_value = vl_embs[:, prefix_mask, :].mean(dim=1, keepdim=False)
        state_features_value = state_features.reshape(bsize, -1)
        if self.rl_config.get("use_vlm_value", False):
            value_embs = vl_embs_value
        else:
            value_embs = torch.cat((vl_embs_value, state_features_value), dim=1)

        values_vlm = self.value_head(value_embs)[:, 0]
        return values_vlm

    def get_rl_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        mode: Literal["train", "eval"] = "train",
        compute_values=True,
    ) -> BatchFeature:
        """Sample an action chunk via stochastic denoising (rollout path).

        Returns the predicted action together with the full denoising ``chains``,
        per-step log-probabilities, value estimate and the sampled denoising
        indices, which the actor later replays in :meth:`forward`.
        """
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = (
            backbone_output.backbone_features
            if hasattr(backbone_output, "backbone_features")
            else backbone_output
        )
        embodiment_id = (
            action_input.embodiment_id if hasattr(action_input, "embodiment_id") else 0
        )
        state_features = self._encode_state_features(action_input, embodiment_id)
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        x_t = torch.randn(
            size=(batch_size, self.action_horizon, self.model_action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        chains = [x_t]
        log_probs = []

        if self.rl_config.get("joint_logprob"):
            initial_log_prob = self.get_logprob_norm(
                x_t, torch.zeros_like(x_t), torch.ones_like(x_t)
            )
            log_probs.append(initial_log_prob)

        num_steps = self.num_inference_timesteps
        if mode == "train":
            if self.rl_config.get("joint_logprob"):
                denoise_inds = torch.arange(num_steps, device=device)
            else:
                rand_idx = random.randint(0, num_steps - 1)
                denoise_inds = torch.full(
                    (num_steps,), rand_idx, dtype=torch.long, device=device
                )
        else:
            denoise_inds = torch.full((num_steps,), -1, dtype=torch.long, device=device)
        denoise_inds = denoise_inds.unsqueeze(0).repeat(batch_size, 1)

        for idx in range(num_steps):
            # Stochastic noise is injected only on the sampled denoising index;
            # all other steps follow the deterministic ("eval") transition.
            step_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=idx,
                x_t=x_t,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode=step_mode,
                denoise_steps=num_steps,
                compute_values=compute_values,
                backbone_output=backbone_output,
            )

            x_t = (
                x_t_mean
                + self.sample_noise(x_t.shape, device, dtype=x_t.dtype) * x_t_std
            )
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            chains.append(x_t)
            log_probs.append(log_prob)

        x_0 = x_t
        chains = torch.stack(chains, dim=1)
        log_probs = torch.stack(log_probs, dim=1)[
            :, :, : self.action_chunk, : self.env_action_dim
        ]
        if compute_values:
            values = self.get_value(vl_embs, state_features)
            values = values[:, None]
        else:
            values = torch.zeros((batch_size, 1), device=device, dtype=vl_embs.dtype)

        return BatchFeature(data={"action_pred": x_0}), {
            "actions": x_0,
            "action_pred": x_0,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
        }

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        chains,
        denoise_inds,
        compute_values=True,
    ):
        """Recompute log-probabilities and values for cached denoising chains.

        This is the actor-side counterpart of :meth:`get_rl_action`: given the
        denoising chain sampled during rollout it evaluates the current policy's
        log-probability of each recorded transition.
        """
        if hasattr(backbone_output, "backbone_features"):
            backbone_output = self._process_backbone_output(backbone_output)
        vl_embs = (
            backbone_output.backbone_features
            if hasattr(backbone_output, "backbone_features")
            else backbone_output
        )
        embodiment_id = (
            action_input.embodiment_id if hasattr(action_input, "embodiment_id") else 0
        )
        state_features = self._encode_state_features(action_input, embodiment_id)
        batch_size = vl_embs.shape[0]

        chains_log_probs = []
        if self.rl_config.get("joint_logprob"):
            num_steps = getattr(self.config, "num_steps", 1)
            initial_log_prob = self.get_logprob_norm(
                chains[:, 0],
                torch.zeros_like(chains[:, 0]),
                torch.ones_like(chains[:, 0]),
            )
            chains_log_probs.append(initial_log_prob)
        else:
            num_steps = 1

        denoise_inds = denoise_inds.to(chains.device)
        batch_indices = torch.arange(batch_size, device=chains.device)
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[batch_indices, denoise_ind]
            chains_next = chains[batch_indices, denoise_ind + 1]
            x_t_mean, x_t_std = self.sample_mean_var_val(
                vl_embs=vl_embs,
                idx=denoise_ind,
                x_t=chains_pre,
                embodiment_id=embodiment_id,
                state_features=state_features,
                mode="train",
                denoise_steps=self.num_inference_timesteps,
                compute_values=compute_values,
                backbone_output=backbone_output,
            )
            log_probs = self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            chains_log_probs.append(log_probs)

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        if compute_values:
            chains_values = self.get_value(vl_embs, state_features)
            chains_values = chains_values[:, None]
        else:
            chains_values = torch.zeros(
                (batch_size, 1), device=chains_log_probs.device, dtype=vl_embs.dtype
            )

        return chains_log_probs, chains_values


class GR00T_N1_7_ForRLActionPrediction(Gr00tN1d7, BasePolicy):
    """GR00T N1.7 model for reinforcement-learning action prediction."""

    _no_split_modules = [
        "Qwen3VLTextDecoderLayer",
        "Qwen3VLVisionBlock",
        "BasicTransformerBlock",
    ]

    def __init__(
        self,
        config: Gr00tN1d7Config,
        rl_head_config: dict[str, Any],
        embodiment_tag: Union[str, EmbodimentTag],
        local_model_path: str,
        modality_config: Optional[Any] = None,
        modality_transform: Optional[Any] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        denoising_steps: Optional[int] = None,
        obs_converter_type: str = "libero",
        output_action_chunks: int = 1,
        **kwargs,
    ):
        if isinstance(embodiment_tag, str):
            self.embodiment_tag = EmbodimentTag(embodiment_tag)
        else:
            self.embodiment_tag = embodiment_tag

        loading_kwargs = kwargs.pop(
            "transformers_loading_kwargs", {"trust_remote_code": True}
        )

        backbone_model_path = kwargs.pop("backbone_model_path", None)
        if backbone_model_path is not None:
            backbone_model_path = str(Path(backbone_model_path).expanduser().resolve())
            if not Path(backbone_model_path).is_dir():
                raise FileNotFoundError(
                    f"Backbone model path does not exist: {backbone_model_path}"
                )
            loading_kwargs["local_files_only"] = True

        original_model_name = str(config.model_name)

        if backbone_model_path is not None:
            logger.info(
                "Loading backbone locally from %s with canonical model_name=%s",
                backbone_model_path,
                original_model_name,
            )
        else:
            logger.info("Loading backbone from HuggingFace: %s", original_model_name)

        for key in list(kwargs.keys()):
            if hasattr(config, key):
                setattr(config, key, kwargs.pop(key))
        if kwargs:
            logger.warning("Ignoring unexpected kwargs: %s", sorted(kwargs))

        with redirect_qwen3_backbone_to_local(original_model_name, backbone_model_path):
            super().__init__(config, transformers_loading_kwargs=loading_kwargs)

            self._modality_config, self._modality_transform = (
                self._load_modality_processor(
                    modality_config=modality_config,
                    modality_transform=modality_transform,
                    local_model_path=local_model_path,
                    backbone_model_path=backbone_model_path,
                )
            )

        self.padding_value = rl_head_config.get("padding_value", 0)
        self.model_path = Path(local_model_path)
        self.compute_dtype = compute_dtype
        self.output_action_chunks = output_action_chunks

        self.action_head = FlowMatchingActionHeadForRLActionPrediction(
            config, rl_head_config, output_action_chunks
        )

        if denoising_steps is not None and hasattr(
            self.action_head, "num_inference_timesteps"
        ):
            self.action_head.num_inference_timesteps = denoising_steps

        self.obs_converter_type = obs_converter_type
        self.obs_convert_fn = OBS_CONVERSION[obs_converter_type]
        self.action_convert_fn = ACTION_CONVERSION_N1D7[obs_converter_type]
        exp_cfg_path = self.model_path / "experiment_cfg"
        self._load_metadata(exp_cfg_path)
        self.action_dim = _resolve_env_action_dim(
            getattr(config, "action_dim", self.valid_action_dim),
            self.valid_action_dim,
        )
        self.action_head.env_action_dim = self.action_dim
        self.action_head.valid_action_dim = self.valid_action_dim

        self._no_split_modules = self.__class__._no_split_modules
        if hasattr(self, "config"):
            self.config.no_split_modules = self._no_split_modules
            self.config._no_split_modules = self._no_split_modules
        logger.info(
            "Forced FSDP _no_split_modules into config: %s",
            self.config.no_split_modules,
        )

    def _load_modality_processor(
        self,
        modality_config: Optional[Any],
        modality_transform: Optional[Any],
        local_model_path: str,
        backbone_model_path: Optional[str],
    ) -> tuple[Any, Any]:
        """Resolve the modality config and transform (processor)."""
        if modality_config is not None and modality_transform is not None:
            return modality_config, modality_transform

        processor_dir = _find_processor_dir(Path(local_model_path))
        if processor_dir is not None:
            logger.info("Loading processor from local dir: %s", processor_dir)
            modality_transform, modality_config = self._load_processor_from_dir(
                processor_dir,
                backbone_model_path=backbone_model_path,
            )
        else:
            from transformers import AutoProcessor

            logger.info(
                "Loading processor via AutoProcessor from: %s", local_model_path
            )
            processor = AutoProcessor.from_pretrained(
                local_model_path,
                trust_remote_code=True,
                local_files_only=Path(local_model_path).is_dir(),
            )
            modality_transform = processor
            modality_config = getattr(
                processor,
                "modality_configs",
                getattr(processor, "modality_config", None),
            )

        return modality_config, modality_transform

    @staticmethod
    def _load_processor_from_dir(
        processor_dir: Path,
        *,
        backbone_model_path: Optional[str],
    ) -> tuple[Gr00tN1d7Processor, Any]:
        """Load the official GR00T N1.7 processor from a local directory."""
        with open(processor_dir / "processor_config.json", "r") as f:
            processor_cfg = json.load(f)["processor_kwargs"]
        with open(processor_dir / "statistics.json", "r") as f:
            processor_cfg["statistics"] = json.load(f)
        with open(processor_dir / "embodiment_id.json", "r") as f:
            processor_cfg["embodiment_id_mapping"] = json.load(f)
        if backbone_model_path is not None:
            processor_cfg.setdefault("transformers_loading_kwargs", {})
            processor_cfg["transformers_loading_kwargs"]["local_files_only"] = True
        modality_transform = Gr00tN1d7Processor(**processor_cfg)
        modality_config = getattr(modality_transform, "modality_configs", None)
        return modality_transform, modality_config

    def eval(self):
        self._modality_transform.eval()
        super().eval()

    @staticmethod
    def _check_state_is_batched(obs: dict[str, Any]) -> bool:
        """Return whether observation state tensors already carry a batch dim."""
        for k, v in obs.items():
            if "state" in k and len(v.shape) < 3:
                return False
        return True

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        compute_logprobs: bool = True,
        compute_entropy: bool = False,
        compute_values: bool = True,
        use_cache: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Actor forward pass: recompute log-probs/values from cached rollouts."""
        normalized_input = _normalize_gr00t_forward_inputs(forward_inputs)
        normalized_input = _canonicalize_gr00t_text_forward_inputs(
            normalized_input,
            getattr(self, "padding_value", 0),
        )

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)

        chains = forward_inputs["chains"]
        denoise_inds = forward_inputs["denoise_inds"]
        log_probs, value_t = self.action_head(
            backbone_output=backbone_outputs,
            action_input=action_inputs,
            chains=chains,
            denoise_inds=denoise_inds,
            compute_values=compute_values,
        )

        log_probs = log_probs[
            :,
            :,
            : self.action_head.action_chunk,
            : self.valid_action_dim,
        ]
        if self.action_head.rl_config.get("joint_logprob"):
            log_probs = log_probs.mean(dim=1)
            prev_logprobs = kwargs["prev_logprobs"].mean(dim=1)
        else:
            bsize = log_probs.shape[0]
            log_probs = log_probs[:, 0]
            prev_logprobs = kwargs["prev_logprobs"]
            prev_logprobs = prev_logprobs[
                torch.arange(bsize, device=prev_logprobs.device),
                denoise_inds[:, 0].to(device=prev_logprobs.device),
                : self.action_head.action_chunk,
                : self.valid_action_dim,
            ]
        value_t = value_t.mean(dim=-1, keepdim=False)

        env_action_dim = self.action_dim
        log_probs = log_probs[..., :env_action_dim]
        prev_logprobs = prev_logprobs[..., :env_action_dim]

        return {
            "logprobs": log_probs.float(),
            "prev_logprobs": prev_logprobs.float(),
            "values": value_t,
            "entropy": None,
        }

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ):
        """Rollout entry point: produce env-ready actions and RL bookkeeping."""
        del kwargs
        observations, obs_copy, is_batch = self._prepare_rollout_observation(env_obs)
        normalized_action, result = self._predict_normalized_action(obs_copy, mode)
        unnormalized_action = self._get_unnormalized_action(
            normalized_action,
            state=observations,
        )

        if not is_batch:
            unnormalized_action = squeeze_dict_values(unnormalized_action)

        raw_action = self.action_convert_fn(
            unnormalized_action,
            chunk_size=self.output_action_chunks,
        )
        raw_action = self._apply_exploration_noise(raw_action, mode)
        return raw_action, result

    @staticmethod
    def _coerce_observation_values_to_numpy(
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure every observation value is a numpy array."""
        coerced = {}
        for key, value in observation.items():
            coerced[key] = value if isinstance(value, np.ndarray) else np.array(value)
        return coerced

    @staticmethod
    def _cast_float_tensors_to_compute_dtype(
        normalized_input: dict[str, Any],
        compute_dtype: torch.dtype,
    ) -> dict[str, Any]:
        """Cast float32 tensors to the model compute dtype, leaving others intact."""
        casted = {}
        for key, value in normalized_input.items():
            if isinstance(value, torch.Tensor) and value.dtype == torch.float32:
                casted[key] = value.to(compute_dtype)
            else:
                casted[key] = value
        return casted

    def _prepare_rollout_observation(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Convert raw env observations into batched GR00T processor inputs."""
        env_obs = dict(env_obs)
        # Here we have a source causing tiny inference-training inconsistency,
        # force convert the state to bf16 then back to float32 to reproduce the info loss in training.
        env_obs["states"] = env_obs["states"].to(torch.bfloat16)
        env_obs["states"] = env_obs["states"].cpu().float()

        observations = self.obs_convert_fn(env_obs)
        obs_copy = observations.copy()
        is_batch = self._check_state_is_batched(obs_copy)
        if not is_batch:
            obs_copy = unsqueeze_dict_values(obs_copy)
        obs_copy = self._coerce_observation_values_to_numpy(obs_copy)
        return observations, obs_copy, is_batch

    def _predict_normalized_action(
        self,
        obs_copy: dict[str, Any],
        mode: Literal["train", "eval"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run the policy and return normalized actions plus RL bookkeeping."""
        normalized_input = self.apply_transforms(obs_copy)
        normalized_input = self._cast_float_tensors_to_compute_dtype(
            normalized_input,
            self.compute_dtype,
        )
        normalized_input = _canonicalize_gr00t_text_forward_inputs(
            normalized_input,
            getattr(self, "padding_value", 0),
        )

        if mode == "eval":
            normalized_action = self._get_action_from_normalized_input(normalized_input)
            result = {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {},
            }
        else:
            normalized_action, result = self._get_rl_action(
                normalized_input,
                mode=mode,
            )
        return normalized_action, result

    def _apply_exploration_noise(
        self,
        raw_action: np.ndarray | torch.Tensor,
        mode: Literal["train", "eval"],
    ) -> np.ndarray | torch.Tensor:
        """Optionally perturb actions with clipped Gaussian noise during training."""
        if mode != "train":
            return raw_action

        noise_scale = float(self.action_head.rl_config.get("action_noise_scale", 0.1))
        if noise_scale <= 0:
            return raw_action

        is_numpy = isinstance(raw_action, np.ndarray)
        raw_tensor = torch.from_numpy(raw_action) if is_numpy else raw_action
        noise = torch.randn_like(raw_tensor) * noise_scale
        raw_tensor = (raw_tensor + noise).clamp(-1.0, 1.0)
        return raw_tensor.numpy() if is_numpy else raw_tensor

    def apply_transforms(self, obs: dict) -> dict:
        """Tokenize/normalize a batched observation via the GR00T processor."""
        return self._modality_transform.process_observation(obs, self.embodiment_tag)

    def unapply_transforms(
        self,
        action: dict[str, Any],
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Decode (unnormalize, relative->absolute) a normalized action chunk."""
        raw_action_tensor = action["action"]

        if isinstance(raw_action_tensor, torch.Tensor):
            raw_action_tensor = raw_action_tensor.detach().cpu().numpy()

        decoded_state = None
        if state is not None:
            decoded_state = {}
            for key, value in state.items():
                if not key.startswith("state."):
                    continue
                if isinstance(value, torch.Tensor):
                    value = value.detach().cpu().numpy()
                decoded_state[key.split(".", 1)[1]] = value

        decoded = self._modality_transform.decode_action(
            action=raw_action_tensor,
            embodiment_tag=self.embodiment_tag,
            state=decoded_state,
        )
        return decoded

    def _get_rl_action(
        self,
        normalized_input: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ):
        """Sample an action and assemble the ``forward_inputs`` cached for the actor."""
        normalized_input = _normalize_gr00t_forward_inputs(normalized_input)

        backbone_inputs, action_inputs = self.prepare_input(normalized_input)
        backbone_outputs = self.backbone(backbone_inputs)
        action_head_outputs, rlinf_outputs = self.action_head.get_rl_action(
            backbone_outputs, action_inputs, mode=mode
        )
        actions = rlinf_outputs["actions"]
        if hasattr(self, "validate_data"):
            self.validate_data(action_head_outputs, backbone_outputs, is_training=False)
        actions = actions.float()

        batch_size = actions.shape[0]
        stashed_forward_inputs = {
            key: _batchify_gr00t_forward_input(key, value, batch_size)
            for key, value in normalized_input.items()
        }
        forward_inputs = {
            "chains": rlinf_outputs["chains"],
            "denoise_inds": rlinf_outputs["denoise_inds"],
            **stashed_forward_inputs,
        }
        result = {
            "prev_logprobs": rlinf_outputs["prev_logprobs"],
            "prev_values": rlinf_outputs["prev_values"],
            "forward_inputs": self._finalize_rollout_forward_inputs(forward_inputs),
        }
        return actions, result

    def _finalize_rollout_forward_inputs(
        self,
        forward_inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Ensure cached rollout inputs are batch-splittable tensors."""
        finalized = {}
        batch_size = int(forward_inputs["chains"].shape[0])
        for key, value in forward_inputs.items():
            value = _tensorize_forward_input(value)
            if key in _FORWARD_INPUT_MODEL_KEYS:
                value = _batchify_gr00t_forward_input(key, value, batch_size)
            finalized[key] = value
        return finalized

    def _get_action_from_normalized_input(
        self, normalized_input: dict[str, Any]
    ) -> torch.Tensor:
        """Deterministic action prediction (eval path) without RL bookkeeping."""
        device_type = getattr(self.device, "type", "cpu")
        autocast_context = (
            torch.autocast(device_type=device_type, dtype=self.compute_dtype)
            if device_type == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            model_pred = self.get_action(normalized_input)

        normalized_action = model_pred["action_pred"].float()
        return normalized_action

    def _get_unnormalized_action(
        self,
        normalized_action: torch.Tensor,
        state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        return self.unapply_transforms({"action": normalized_action.cpu()}, state=state)

    def _load_metadata(self, exp_cfg_dir: Path):
        """Populate ``valid_action_dim`` and ``image_nums`` from checkpoint metadata.

        Falls back to inferring these from the modality config (and finally from
        the model config) when ``metadata.json`` is absent.
        """
        metadata_path = exp_cfg_dir / "metadata.json"
        if not metadata_path.exists():
            logger.info(
                "Metadata file not found at %s. Inferring from modality_config...",
                metadata_path,
            )

            tag_value = self.embodiment_tag.value
            if self._modality_config is None or tag_value not in self._modality_config:
                logger.info(
                    "Modality config is missing or does not contain tag %s. "
                    "Attempting to infer valid_action_dim and image_nums from "
                    "config attributes.",
                    self.embodiment_tag.value,
                )
                self.valid_action_dim = getattr(
                    self.config, "max_action_dim", getattr(self.config, "action_dim", 7)
                )
                self.image_nums = getattr(self.config, "image_nums", 1)
                logger.info(
                    "Inferred fallback: valid_action_dim=%s, image_nums=%s",
                    self.valid_action_dim,
                    self.image_nums,
                )
                return

            current_modality = self._modality_config[tag_value]

            valid_action_dim = 0
            if "action" in current_modality:
                action_modality_cfg = current_modality["action"]
                if (
                    hasattr(action_modality_cfg, "dim_map")
                    and action_modality_cfg.dim_map
                ):
                    for dim_val in action_modality_cfg.dim_map.values():
                        valid_action_dim += dim_val
                elif (
                    hasattr(action_modality_cfg, "modality_keys")
                    and action_modality_cfg.modality_keys
                ):
                    norm_params = getattr(
                        getattr(
                            self._modality_transform, "state_action_processor", None
                        ),
                        "norm_params",
                        {},
                    )
                    action_norm_params = (
                        norm_params.get(tag_value, {}).get("action", {})
                        if isinstance(norm_params, dict)
                        else {}
                    )
                    if action_norm_params:
                        valid_action_dim = sum(
                            int(action_norm_params[key]["dim"].item())
                            for key in action_modality_cfg.modality_keys
                            if key in action_norm_params
                        )
                    else:
                        valid_action_dim = getattr(self.config, "max_action_dim", 29)
                elif isinstance(action_modality_cfg, dict):
                    if action_modality_cfg.get("dim_map"):
                        for dim_val in action_modality_cfg["dim_map"].values():
                            valid_action_dim += dim_val
                    elif "dim" in action_modality_cfg:
                        valid_action_dim = action_modality_cfg["dim"]
                    else:
                        valid_action_dim = getattr(self.config, "max_action_dim", 29)
                else:
                    valid_action_dim = getattr(self.config, "max_action_dim", 29)

            self.valid_action_dim = valid_action_dim

            if "video" in current_modality:
                video_modality_cfg = current_modality["video"]
                if (
                    hasattr(video_modality_cfg, "modality_keys")
                    and video_modality_cfg.modality_keys
                ):
                    self.image_nums = len(video_modality_cfg.modality_keys)
                elif (
                    isinstance(video_modality_cfg, dict)
                    and "modality_keys" in video_modality_cfg
                ):
                    self.image_nums = len(video_modality_cfg["modality_keys"])
                else:
                    self.image_nums = 1
            else:
                self.image_nums = 1

            logger.info(
                "Inferred from modality_config: valid_action_dim=%s, image_nums=%s",
                self.valid_action_dim,
                self.image_nums,
            )
            return

        with open(metadata_path, "r") as f:
            metadatas = json.load(f)

        metadata_dict = metadatas.get(self.embodiment_tag.value)
        if metadata_dict is None:
            raise ValueError(
                f"No metadata found for embodiment tag: {self.embodiment_tag.value}"
            )

        self.metadata = metadata_dict
        if hasattr(self._modality_transform, "set_metadata"):
            self._modality_transform.set_metadata(metadata_dict)

        valid_action_dim = 0
        action_mods = self.metadata.get("modalities", {}).get("action", {})
        for v in action_mods.values():
            shape = v.get("shape", [0]) if isinstance(v, dict) else [0]
            valid_action_dim += shape[0] if len(shape) > 0 else 0
        self.valid_action_dim = valid_action_dim

        video_mods = self.metadata.get("modalities", {}).get("video", {})
        self.image_nums = len(video_mods)
