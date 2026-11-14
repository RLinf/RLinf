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

import numpy as np
import torch

from rlinf.algorithms.diffusion.art import (
    compute_art_loss,
    prepare_art_training_inputs,
    prompt_disjoint_windows,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_rollout_metrics
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedARTFSDPPolicy(EmbodiedFSDPActor):
    """FSDP policy worker for DiffusionART."""

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, float]:
        """Report rollout metrics; ART builds guidance directly from rewards."""
        metrics = compute_rollout_metrics(self.rollout_batch)
        metrics.update(self._group_success_metrics)
        return metrics

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        """Prepare frozen ART targets and apply prompt-balanced updates."""
        endpoints = self.rollout_batch["forward_inputs"]["art_endpoint"]
        rollout_size = endpoints.shape[0] * endpoints.shape[1]
        self.rollout_batch = process_nested_dict_for_train(
            self.rollout_batch,
            torch.arange(rollout_size),
        )

        generator = torch.Generator().manual_seed(
            self.cfg.actor.seed + self.version + self._rank
        )
        preparation_metrics, has_guidance = prepare_art_training_inputs(
            self.rollout_batch["forward_inputs"],
            self.rollout_batch["rewards"],
            config=self.cfg.algorithm,
            generator=generator,
        )
        active = torch.tensor(float(has_guidance), device=self.device)
        torch.distributed.all_reduce(active, op=torch.distributed.ReduceOp.MAX)
        has_guidance = bool(active.item())
        preparation_metrics["actor/skipped_zero_guidance"] = float(not has_guidance)
        if not has_guidance:
            clear_memory()
            return all_reduce_dict(
                preparation_metrics,
                op=torch.distributed.ReduceOp.AVG,
            )

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
        self._precompute_old_velocities()

        group_size = int(self.cfg.algorithm.group_size)
        updates = int(self.cfg.algorithm.get("art_updates_per_rollout", 2))
        groups = torch.arange(rollout_size) // group_size
        order = torch.randperm(rollout_size, generator=generator).tolist()
        row_windows = prompt_disjoint_windows(
            groups,
            updates=updates,
            order=order,
        )
        windows = tuple(torch.tensor(rows, dtype=torch.long) for rows in row_windows)

        self.model.train()
        metrics: dict[str, list[float]] = {}
        for key, value in preparation_metrics.items():
            append_to_dict(metrics, {key: value})

        for window_rows in windows:
            window_batch = self._index_nested(self.rollout_batch, window_rows)
            window_size = window_rows.numel()
            micro_batch_size = int(self.cfg.actor.micro_batch_size)
            expected_window_size = self.cfg.actor.global_batch_size // self._world_size
            assert window_size == expected_window_size, (
                f"ART window has {window_size} rows on this rank; expected "
                f"{expected_window_size} from actor.global_batch_size."
            )
            assert window_size % micro_batch_size == 0, (
                f"ART window size {window_size} must be divisible by "
                f"actor.micro_batch_size {micro_batch_size}."
            )
            micro_batches = split_dict_to_chunk(
                window_batch,
                window_size // micro_batch_size,
            )
            num_train_timesteps = window_batch["forward_inputs"]["art_states"].shape[1]
            self.gradient_accumulation = len(micro_batches) * num_train_timesteps

            self.optimizer.zero_grad()
            backward_index = 0
            for micro_batch in micro_batches:
                forward_inputs = micro_batch["forward_inputs"]
                model_inputs = put_tensor_device(
                    self._model_forward_inputs(forward_inputs),
                    self.device,
                )
                guidance_direction = forward_inputs["art_guidance_direction"].to(
                    self.device
                )
                guidance_magnitude = forward_inputs["art_guidance_magnitude"].to(
                    self.device
                )
                for timestep_index in range(num_train_timesteps):
                    backward_index += 1
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(
                            backward_index == self.gradient_accumulation
                        ),
                    )
                    loss, step_metrics = self._art_forward_and_loss(
                        forward_inputs,
                        model_inputs,
                        guidance_direction,
                        guidance_magnitude,
                        timestep_index,
                    )
                    scaled_loss = loss / self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(scaled_loss).backward()
                    append_to_dict(metrics, step_metrics)
                    del loss, scaled_loss, step_metrics

            self.torch_platform.empty_cache()
            grad_norm, lr_list = self.optimizer_step()
            append_to_dict(
                metrics,
                {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                },
            )

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        return self._reduce_training_metrics(metrics)

    @staticmethod
    def _reduce_training_metrics(metrics: dict[str, list[float]]) -> dict[str, float]:
        """Weight active-only metrics across micro-batches and actor ranks."""
        active_counts = metrics.pop("actor/active_count")
        active_metric_sums = {
            key: sum(
                value * count
                for value, count in zip(metrics.pop(key), active_counts, strict=True)
            )
            for key in (
                "actor/unmasked_loss",
                "actor/mask_frac",
                "actor/outward_frac",
            )
        }
        local_metrics = {key: np.mean(values) for key, values in metrics.items()}
        local_metrics.update(active_metric_sums)
        local_metrics["actor/active_count"] = sum(active_counts)
        reduced_metrics = all_reduce_dict(
            local_metrics, op=torch.distributed.ReduceOp.SUM
        )
        active_count = max(reduced_metrics.pop("actor/active_count"), 1)
        world_size = torch.distributed.get_world_size()
        return {
            key: value / (active_count if key in active_metric_sums else world_size)
            for key, value in reduced_metrics.items()
        }

    @torch.no_grad()
    def _precompute_old_velocities(self) -> None:
        forward_inputs = self.rollout_batch["forward_inputs"]
        states = forward_inputs["art_states"]
        sigmas = forward_inputs["art_sigmas"]
        old_velocities = torch.empty_like(states)
        micro_batch_size = int(self.cfg.actor.micro_batch_size)
        row_count, timestep_count = states.shape[:2]

        self.model.eval()
        for start in range(0, row_count, micro_batch_size):
            end = min(start + micro_batch_size, row_count)
            rows = torch.arange(start, end)
            model_inputs = put_tensor_device(
                self._model_forward_inputs(forward_inputs, rows),
                self.device,
            )
            for timestep_index in range(timestep_count):
                with self.amp_context:
                    output = self.model(
                        forward_type=ForwardType.NFT,
                        forward_inputs=model_inputs,
                        nft_inputs={
                            "x_t": states[start:end, timestep_index].to(self.device),
                            "timesteps": sigmas[start:end, timestep_index].to(
                                self.device
                            ),
                        },
                        compute_values=False,
                    )
                old_velocities[start:end, timestep_index] = (
                    output["v_theta"].detach().to(old_velocities.device)
                )
        forward_inputs["art_old_velocity"] = old_velocities
        self.torch_platform.empty_cache()

    def _art_forward_and_loss(
        self,
        forward_inputs: dict[str, torch.Tensor],
        model_inputs: dict[str, torch.Tensor],
        guidance_direction: torch.Tensor,
        guidance_magnitude: torch.Tensor,
        timestep_index: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        states = forward_inputs["art_states"][:, timestep_index].to(self.device)
        sigmas = forward_inputs["art_sigmas"][:, timestep_index].to(self.device)
        sigmas_next = forward_inputs["art_sigmas_next"][:, timestep_index].to(
            self.device
        )
        old_velocity = forward_inputs["art_old_velocity"][:, timestep_index].to(
            self.device
        )

        with self.amp_context:
            output = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=model_inputs,
                nft_inputs={"x_t": states, "timesteps": sigmas},
                compute_values=False,
            )
        return compute_art_loss(
            output["v_theta"],
            old_velocity,
            sigmas,
            sigmas_next,
            guidance_direction,
            guidance_magnitude,
            config=self.cfg.algorithm,
        )

    @classmethod
    def _index_nested(cls, value, indices: torch.Tensor):
        if isinstance(value, torch.Tensor):
            return value.index_select(0, indices.to(value.device))
        if isinstance(value, dict):
            return {
                key: cls._index_nested(nested_value, indices)
                for key, nested_value in value.items()
            }
        return value

    @classmethod
    def _model_forward_inputs(
        cls,
        forward_inputs: dict,
        indices: torch.Tensor | None = None,
    ) -> dict:
        model_input_keys = {
            "prompt_embeds",
            "pooled_prompt_embeds",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "latent_condition",
            "first_frame_mask",
        }
        inputs = {
            key: value
            for key, value in forward_inputs.items()
            if key in model_input_keys
        }
        return inputs if indices is None else cls._index_nested(inputs, indices)
