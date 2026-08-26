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

"""Eval-only in-process Psi0 policy for the SIMPLE benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.psi0.processing import Psi0ProcessorAdapter


class Psi0Policy(torch.nn.Module, BasePolicy):
    """Wrap the fixed upstream Psi0 checkpoint with RLinf's eval contract."""

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
        self._set_trainable_boundary()
        self.train(False)

    @classmethod
    def from_config(cls, cfg: DictConfig, torch_dtype=None) -> "Psi0Policy":
        """Load the joint S2+S1 checkpoint and its checkpoint-owned transforms."""
        del torch_dtype
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
        return cls(
            model=model,
            processor=processor,
            num_inference_steps=int(psi0_cfg.num_inference_steps),
            plan_horizon=int(psi0_cfg.plan_horizon),
            execution_horizon=int(cfg.num_action_chunks),
            overlap_horizon=int(psi0_cfg.overlap_horizon),
            rtc_max_delay=int(launch_config.model.max_delay),
        )

    @property
    def _no_split_modules(self) -> list[str]:
        return ["Qwen3VLDecoderLayer", "ActionTransformerBlock"]

    def _set_trainable_boundary(self) -> None:
        self.psi0_model.vlm_model.requires_grad_(False)
        self.psi0_model.action_header.requires_grad_(True)

    def train(self, mode: bool = True) -> "Psi0Policy":
        super().train(mode)
        self.psi0_model.vlm_model.eval()
        self.psi0_model.action_header.train(mode)
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
            raise ValueError("The Psi0 Eval MVP supports exactly one environment.")
        return bool(reset_mask[0])

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict one official 24-action execution chunk."""
        del kwargs
        if mode != "eval":
            raise RuntimeError(
                "Psi0 RL sampling is locked in P3; only mode='eval' is supported."
            )
        processed = self.processor.process(env_obs)
        if len(processed.instructions) != 1:
            raise ValueError("The Psi0 Eval MVP supports exactly one environment.")
        if self._reset_requested(env_obs):
            self._previous_plan = None

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

    def default_forward(self, **kwargs):
        raise RuntimeError(
            "Psi0 actor recompute is locked in P3 and will be implemented in P5."
        )
