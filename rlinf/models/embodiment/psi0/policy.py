# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-process Psi0 policy for SIMPLE evaluation and PPO training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.psi0.processing import Psi0ProcessorAdapter
from rlinf.models.embodiment.psi0.sampler import Psi0StochasticTransitionSampler


@dataclass(frozen=True)
class _Psi0PlanRecord:
    """Self-contained statistic for one sampled 30-step plan."""

    hidden_states: torch.Tensor
    states: torch.Tensor
    transition_x: torch.Tensor
    transition_next: torch.Tensor
    timestep: torch.Tensor
    sigma: torch.Tensor
    sigma_next: torch.Tensor
    sample_mask: torch.Tensor
    logprobs: torch.Tensor


class _Psi0ValueHead(ValueHead):
    """Value head that follows FSDP's mixed-precision parameter dtype."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        head_dtype = next(self.parameters()).dtype
        return super().forward(features.to(head_dtype))


class Psi0Policy(torch.nn.Module, BasePolicy):
    """Wrap the fixed upstream Psi0 checkpoint with RLinf's policy contract."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: Psi0ProcessorAdapter,
        num_inference_steps: int = 10,
        plan_horizon: int = 30,
        execution_horizon: int = 24,
        overlap_horizon: int = 6,
        rtc_max_delay: int = 10,
        stochastic_noise_scale: float = 1.0,
        add_value_head: bool = False,
        value_input_dim: int | None = None,
    ) -> None:
        super().__init__()
        if plan_horizon != execution_horizon + overlap_horizon:
            raise ValueError("Psi0 requires plan = execution + overlap horizon.")
        if overlap_horizon >= rtc_max_delay:
            raise ValueError("Psi0 RTC overlap must be smaller than max_delay.")
        self.psi0_model = model
        self.processor = processor
        self.num_inference_steps = num_inference_steps
        self.plan_horizon = plan_horizon
        self.execution_horizon = execution_horizon
        self.overlap_horizon = overlap_horizon
        self.rtc_max_delay = rtc_max_delay
        self._previous_plan: torch.Tensor | None = None
        self._previous_plan_record: _Psi0PlanRecord | None = None
        self._train_sampler = Psi0StochasticTransitionSampler(
            noise_scale=stochastic_noise_scale
        )
        if add_value_head:
            if value_input_dim is None:
                text_config = getattr(model.vlm_model.config, "text_config", None)
                value_input_dim = int(text_config.hidden_size)
            self.value_head = _Psi0ValueHead(
                input_dim=value_input_dim,
                hidden_sizes=(1024, 512, 256),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
        self._set_trainable_boundary()
        self.train(False)

    @classmethod
    def from_config(cls, cfg: DictConfig, torch_dtype=None) -> "Psi0Policy":
        """Load the joint S2+S1 checkpoint and checkpoint-owned transforms."""
        run_dir = Path(str(cfg.model_path)).expanduser()
        checkpoint_step = cfg.get("checkpoint_step", 40000)
        if not run_dir.exists():
            raise ValueError(f"Psi0 run directory does not exist: {run_dir}.")

        from psi.config.config import LaunchConfig
        from psi.models.psi0 import Psi0Model
        from psi.utils import parse_args_to_tyro_config, seed_everything

        config_template: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")
        launch_config = config_template.model_validate_json(
            (run_dir / "run_config.json").read_text()
        )
        seed_everything(launch_config.seed or 42)
        model = Psi0Model.from_pretrained(
            run_dir,
            checkpoint_step,
            launch_config,
            device="cpu",
        )
        processor = Psi0ProcessorAdapter.from_upstream_transform(
            launch_config.data.transform.field,
            launch_config.data.transform.model,
        )
        psi0_cfg = cfg.psi0
        policy = cls(
            model=model,
            processor=processor,
            num_inference_steps=int(psi0_cfg.num_inference_steps),
            plan_horizon=int(psi0_cfg.plan_horizon),
            execution_horizon=int(cfg.num_action_chunks),
            overlap_horizon=int(psi0_cfg.overlap_horizon),
            rtc_max_delay=int(launch_config.model.max_delay),
            stochastic_noise_scale=float(psi0_cfg.get("stochastic_noise_scale", 1.0)),
            add_value_head=bool(cfg.get("add_value_head", False)),
        )
        if hasattr(policy, "value_head") and torch_dtype is not None:
            policy.to(dtype=torch_dtype)
        return policy

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "Qwen3VLVisionBlock",
            "Qwen3VLTextDecoderLayer",
            "VLATransformerBlock",
        ]

    def _set_trainable_boundary(self) -> None:
        self.psi0_model.vlm_model.requires_grad_(False)
        self.psi0_model.action_header.requires_grad_(True)

    def train(self, mode: bool = True) -> "Psi0Policy":
        super().train(mode)
        self.psi0_model.vlm_model.eval()
        self.psi0_model.action_header.eval()
        return self

    def to(self, *args, **kwargs):
        result = super().to(*args, **kwargs)
        try:
            self.psi0_model.device = next(self.parameters()).device
        except StopIteration:
            pass
        return result

    @staticmethod
    def _reset_requested(env_obs: dict[str, Any]) -> bool:
        reset_mask = env_obs.get("reset_mask")
        if reset_mask is None:
            return False
        reset_mask = torch.as_tensor(reset_mask, dtype=torch.bool).reshape(-1)
        if reset_mask.numel() != 1:
            raise ValueError("Psi0 SIMPLE currently supports exactly one environment.")
        return bool(reset_mask[0])

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict one 24-action execution chunk."""
        del kwargs
        processed = self.processor.process(env_obs)
        if len(processed.instructions) != 1:
            raise ValueError("Psi0 SIMPLE currently supports exactly one environment.")
        if self._reset_requested(env_obs):
            self._previous_plan = None
            self._previous_plan_record = None
        if mode == "train":
            return self._predict_train(processed)
        if mode != "eval":
            raise ValueError(f"Unsupported Psi0 policy mode: {mode!r}.")
        return self._predict_eval(processed)

    def _predict_eval(self, processed) -> tuple[torch.Tensor, dict[str, Any]]:
        predict_kwargs = {
            "observations": processed.observations,
            "states": processed.states.to(self.psi0_model.device),
            "instructions": processed.instructions,
            "num_inference_steps": self.num_inference_steps,
            "traj2ds": None,
        }
        if self._previous_plan is None:
            normalized_plan = self.psi0_model.predict_action(**predict_kwargs)
        else:
            conditioned_plan = torch.zeros(
                (1, self.plan_horizon, self.processor.action_dim),
                dtype=torch.float32,
                device=self.psi0_model.device,
            )
            conditioned_plan[:, : self.overlap_horizon] = self._previous_plan[
                :, self.execution_horizon :
            ].to(self.psi0_model.device)
            normalized_plan = self.psi0_model.predict_action_with_training_rtc_flow(
                **predict_kwargs,
                prev_actions=conditioned_plan,
                inference_delay=self.overlap_horizon,
                max_delay=self.rtc_max_delay,
            )

        self._previous_plan = normalized_plan.detach().cpu().contiguous()
        actions = self.processor.denormalize_action(normalized_plan)[
            :, : self.execution_horizon
        ]
        return actions.detach().cpu().contiguous(), {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {},
        }

    def _encode_system2(self, processed) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode image/language observations with frozen System-2."""
        from qwen_vl_utils import process_vision_info

        input_ids = []
        attention_masks = []
        pixel_values = []
        image_grids = []
        for observations, instruction in zip(
            processed.observations, processed.instructions
        ):
            content = [{"type": "image", "image": image} for image in observations]
            content.append({"type": "text", "text": instruction})
            messages = [[{"role": "user", "content": content}]]
            texts = [
                self.psi0_model.vlm_processor.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=True
                )
                for message in messages
            ]
            images, videos = process_vision_info(messages, image_patch_size=16)
            inputs = self.psi0_model.vlm_processor(
                text=texts,
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            ).to(self.psi0_model.device)
            input_ids.append(inputs["input_ids"].squeeze(0))
            attention_masks.append(inputs["attention_mask"].squeeze(0))
            pixel_values.append(inputs["pixel_values"])
            image_grids.append(inputs["image_grid_thw"].squeeze(0))

        device_type = str(self.psi0_model.device).split(":")[0]
        with (
            torch.no_grad(),
            torch.autocast(
                device_type,
                dtype=torch.bfloat16,
                enabled=device_type in {"cuda", "xpu"},
            ),
        ):
            output = self.psi0_model.vlm_model(
                input_ids=torch.stack(input_ids),
                attention_mask=torch.stack(attention_masks),
                pixel_values=torch.stack(pixel_values),
                image_grid_thw=torch.stack(image_grids),
                output_hidden_states=True,
                return_dict=True,
            )
        return output.hidden_states[-1].unsqueeze(1), processed.states.to(
            self.psi0_model.device
        )

    def _predict_velocity(
        self,
        hidden_states: torch.Tensor,
        states: torch.Tensor,
        action_samples: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        device_type = action_samples.device.type
        with torch.autocast(
            device_type,
            dtype=torch.bfloat16,
            enabled=device_type in {"cuda", "xpu"},
        ):
            return self.psi0_model.action_header(
                hidden_states=None,
                timestep=timestep,
                joint_attention_kwargs={
                    "action_hidden_embeds": action_samples,
                    "views": hidden_states,
                    "obs": states,
                    "traj2ds": None,
                },
                return_dict=True,
            ).action

    def _value_features(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Pool frozen System-2 tokens for the Psi0-local PPO critic."""
        return hidden_states[:, 0].mean(dim=1).to(torch.float32)

    def _predict_value(self, features: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "value_head"):
            raise RuntimeError("Psi0 PPO requires actor.model.add_value_head=true.")
        return self.value_head(features).to(torch.float32)

    def _execution_statistics(
        self,
        current: _Psi0PlanRecord,
        previous: _Psi0PlanRecord | None,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        batch_size = actions.shape[0]
        records = [current] if previous is None else [previous, current]

        def stack_slots(name: str) -> torch.Tensor:
            values = [getattr(record, name) for record in records]
            while len(values) < 2:
                values.append(torch.zeros_like(values[0]))
            return torch.stack(values, dim=1)

        valid = torch.zeros((batch_size, 2), dtype=torch.bool, device=actions.device)
        valid[:, : len(records)] = True
        if previous is None:
            source_slots = torch.zeros(
                (batch_size, self.execution_horizon),
                dtype=torch.long,
                device=actions.device,
            )
            source_indices = torch.arange(
                self.execution_horizon, device=actions.device
            ).expand(batch_size, -1)
        else:
            source_slots = torch.cat(
                [
                    torch.zeros(
                        (batch_size, self.overlap_horizon),
                        dtype=torch.long,
                        device=actions.device,
                    ),
                    torch.ones(
                        (batch_size, self.execution_horizon - self.overlap_horizon),
                        dtype=torch.long,
                        device=actions.device,
                    ),
                ],
                dim=1,
            )
            source_indices = torch.cat(
                [
                    torch.arange(
                        self.execution_horizon,
                        self.plan_horizon,
                        device=actions.device,
                    ),
                    torch.arange(
                        self.overlap_horizon,
                        self.execution_horizon,
                        device=actions.device,
                    ),
                ]
            ).expand(batch_size, -1)

        slot_logprobs = stack_slots("logprobs")
        batch_indices = torch.arange(batch_size, device=actions.device)[:, None]
        execution_logprobs = slot_logprobs[batch_indices, source_slots, source_indices]
        forward_inputs = {
            "action": actions.reshape(batch_size, -1).contiguous(),
            "psi0_slot_valid": valid,
            "psi0_slot_hidden_states": stack_slots("hidden_states"),
            "psi0_slot_states": stack_slots("states"),
            "psi0_slot_transition_x": stack_slots("transition_x"),
            "psi0_slot_transition_next": stack_slots("transition_next"),
            "psi0_slot_timestep": stack_slots("timestep"),
            "psi0_slot_sigma": stack_slots("sigma"),
            "psi0_slot_sigma_next": stack_slots("sigma_next"),
            "psi0_slot_sample_mask": stack_slots("sample_mask"),
            "psi0_value_features": self._value_features(current.hidden_states),
            "psi0_execution_source_slots": source_slots,
            "psi0_execution_source_indices": source_indices,
            "psi0_execution_mask": torch.ones(
                (batch_size, self.execution_horizon),
                dtype=torch.bool,
                device=actions.device,
            ),
        }
        return execution_logprobs, forward_inputs

    def _predict_train(self, processed) -> tuple[torch.Tensor, dict[str, Any]]:
        hidden_states, states = self._encode_system2(processed)
        batch_size = states.shape[0]
        initial_noise = torch.randn(
            (batch_size, self.plan_horizon, self.processor.action_dim),
            device=self.psi0_model.device,
            dtype=torch.float32,
        )
        condition_actions = None
        condition_mask = torch.zeros(
            (batch_size, self.plan_horizon),
            dtype=torch.bool,
            device=self.psi0_model.device,
        )
        previous_record = self._previous_plan_record
        if self._previous_plan is not None:
            condition_actions = torch.zeros_like(initial_noise)
            condition_actions[:, : self.overlap_horizon] = self._previous_plan[
                :, self.execution_horizon :
            ].to(self.psi0_model.device)
            condition_mask[:, : self.overlap_horizon] = True

        normalized_plan, transition = self._train_sampler.sample(
            scheduler=self.psi0_model.noise_scheduler,
            num_inference_steps=self.num_inference_steps,
            initial_noise=initial_noise,
            velocity_fn=lambda samples, timestep: self._predict_velocity(
                hidden_states, states, samples, timestep
            ),
            condition_actions=condition_actions,
            condition_mask=condition_mask,
        )
        record = _Psi0PlanRecord(
            hidden_states=hidden_states,
            states=states,
            transition_x=transition.x,
            transition_next=transition.next_x,
            timestep=transition.timestep,
            sigma=transition.sigma,
            sigma_next=transition.sigma_next,
            sample_mask=transition.sample_mask,
            logprobs=transition.logprobs,
        )
        self._previous_plan = normalized_plan.detach().cpu().contiguous()
        self._previous_plan_record = record
        actions = self.processor.denormalize_action(normalized_plan)[
            :, : self.execution_horizon
        ].float()
        prev_logprobs, forward_inputs = self._execution_statistics(
            record, previous_record, actions
        )
        prev_values = self._predict_value(forward_inputs["psi0_value_features"])
        return actions.detach().cpu().contiguous(), {
            "prev_logprobs": prev_logprobs.detach().cpu().contiguous(),
            "prev_values": prev_values.detach().cpu().contiguous(),
            "forward_inputs": {
                key: value.detach().cpu().contiguous()
                for key, value in forward_inputs.items()
            },
        }

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        *,
        compute_values: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Recompute two RTC source transitions with System-1 gradients."""
        del kwargs
        valid = forward_inputs["psi0_slot_valid"].bool()
        batch_size, slot_count = valid.shape
        flat_valid = valid.reshape(-1)
        if not flat_valid.any():
            raise ValueError("Psi0 actor batch contains no valid source transition.")

        def valid_rows(name: str) -> torch.Tensor:
            value = forward_inputs[name].reshape(
                batch_size * slot_count, *forward_inputs[name].shape[2:]
            )
            return value[flat_valid]

        hidden_states = valid_rows("psi0_slot_hidden_states")
        states = valid_rows("psi0_slot_states")
        transition_x = valid_rows("psi0_slot_transition_x")
        transition_next = valid_rows("psi0_slot_transition_next")
        timestep = valid_rows("psi0_slot_timestep")
        sigma = valid_rows("psi0_slot_sigma")
        sigma_next = valid_rows("psi0_slot_sigma_next")
        sample_mask = valid_rows("psi0_slot_sample_mask")
        velocity = self._predict_velocity(hidden_states, states, transition_x, timestep)
        valid_logprobs, valid_entropy = self._train_sampler.recompute(
            transition_x=transition_x,
            transition_next=transition_next,
            velocity=velocity,
            sigma=sigma,
            sigma_next=sigma_next,
            sample_mask=sample_mask,
        )
        slot_shape = (
            batch_size,
            slot_count,
            self.plan_horizon,
            self.processor.action_dim,
        )
        slot_logprobs = torch.zeros(
            slot_shape, device=velocity.device, dtype=torch.float32
        )
        slot_entropy = torch.zeros_like(slot_logprobs)
        slot_logprobs.reshape(-1, self.plan_horizon, self.processor.action_dim)[
            flat_valid
        ] = valid_logprobs
        slot_entropy.reshape(-1, self.plan_horizon, self.processor.action_dim)[
            flat_valid
        ] = valid_entropy

        source_slots = forward_inputs["psi0_execution_source_slots"].long()
        source_indices = forward_inputs["psi0_execution_source_indices"].long()
        batch_indices = torch.arange(batch_size, device=velocity.device)[:, None]
        logprobs = slot_logprobs[batch_indices, source_slots, source_indices]
        entropy = slot_entropy[batch_indices, source_slots, source_indices]
        execution_mask = forward_inputs["psi0_execution_mask"].bool()[..., None]
        result = {
            "logprobs": logprobs * execution_mask,
            "entropy": entropy * execution_mask,
        }
        if compute_values:
            result["values"] = self._predict_value(
                forward_inputs["psi0_value_features"]
            )
        return result

    def forward(
        self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs
    ) -> dict[str, torch.Tensor]:
        """Dispatch RLinf actor forwards through the policy interface."""
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)
