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

"""Configuration for OpenPI's stochastic RL policy."""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any, Literal


@dataclasses.dataclass(frozen=True)
class Pi0RLConfig:
    """RL options from ``actor.model.openpi``."""

    add_value_head: bool = False
    noise_method: str = "flow_ode"
    noise_level: float = 0.0
    noise_logvar_range: tuple[float, float] = (0.08, 0.16)
    joint_logprob: bool = False
    ignore_last: bool = False
    stochastic_step_range: tuple[int, int] | None = None
    frozen_vision_mode: Literal["follow_model", "eval"] = "follow_model"
    value_after_vlm: bool = False
    value_vlm_mode: str = "mean_token"
    value_head_hidden_sizes: tuple[int, ...] | None = None
    detach_critic_input: bool = False
    chunk_critic_input: bool = False
    train_expert_only: bool = False
    is_nft: bool = False
    config_name: str = ""

    @classmethod
    def from_model_config(cls, cfg: Any) -> Pi0RLConfig:
        """Read the model configuration used by both actor and rollout."""
        model_cfg = cfg.openpi
        step_range = model_cfg.get("stochastic_step_range")
        if step_range is not None:
            if not isinstance(step_range, Sequence) or isinstance(step_range, str):
                raise ValueError(
                    "Expected stochastic_step_range=[start, end] or null, "
                    f"got {step_range!r}"
                )
            step_range = tuple(step_range)
        sizes = model_cfg.get("value_head_hidden_sizes")
        return cls(
            add_value_head=bool(cfg.get("add_value_head", False)),
            noise_method=str(model_cfg.get("noise_method", "flow_ode")),
            noise_level=float(model_cfg.get("noise_level", 0.0)),
            noise_logvar_range=tuple(
                float(x) for x in model_cfg.get("noise_logvar_range", [0.08, 0.16])
            ),
            joint_logprob=bool(model_cfg.get("joint_logprob", False)),
            ignore_last=bool(model_cfg.get("ignore_last", False)),
            stochastic_step_range=step_range,
            frozen_vision_mode=model_cfg.get("frozen_vision_mode", "follow_model"),
            value_after_vlm=bool(model_cfg.get("value_after_vlm", False)),
            value_vlm_mode=str(model_cfg.get("value_vlm_mode", "mean_token")),
            value_head_hidden_sizes=(
                None if sizes is None else tuple(int(x) for x in sizes)
            ),
            detach_critic_input=bool(model_cfg.get("detach_critic_input", False)),
            chunk_critic_input=bool(model_cfg.get("chunk_critic_input", False)),
            train_expert_only=bool(model_cfg.get("train_expert_only", False)),
            is_nft=bool(model_cfg.get("is_nft", False)),
            config_name=str(model_cfg.get("config_name", "")),
        )

    def validate(self, num_steps: int) -> None:
        """Validate options before allocating the policy's modules."""
        if (
            isinstance(num_steps, bool)
            or not isinstance(num_steps, int)
            or num_steps < 1
        ):
            raise ValueError(f"Expected positive integer num_steps, got {num_steps!r}")
        if self.frozen_vision_mode not in ("follow_model", "eval"):
            raise ValueError(
                "Expected frozen_vision_mode='follow_model' or 'eval', "
                f"got {self.frozen_vision_mode!r}"
            )
        if self.frozen_vision_mode == "eval" and not self.train_expert_only:
            raise ValueError(
                "frozen_vision_mode='eval' requires train_expert_only=true"
            )
        if self.stochastic_step_range is None:
            return
        bounds = self.stochastic_step_range
        if len(bounds) != 2 or any(
            isinstance(x, bool) or not isinstance(x, int) for x in bounds
        ):
            raise ValueError(
                "Expected stochastic_step_range=[start, end] with integer bounds, "
                f"got {bounds!r}"
            )
        start, end = bounds
        if not 0 <= start < end <= num_steps:
            raise ValueError(
                "Expected 0 <= start < end <= num_steps for stochastic_step_range, "
                f"got {bounds!r} with num_steps={num_steps}"
            )
        if self.joint_logprob or self.is_nft:
            raise ValueError(
                "stochastic_step_range requires single-transition sampling "
                "with joint_logprob=false and is_nft=false"
            )
        if self.noise_method == "flow_ode":
            raise ValueError("stochastic_step_range requires a stochastic noise_method")
        if self.ignore_last and start >= min(end, num_steps - 1):
            raise ValueError(
                "Expected nonempty stochastic_step_range after ignore_last, "
                f"got {bounds!r} with num_steps={num_steps}"
            )

    def stochastic_step_bounds(self, num_steps: int) -> tuple[int, int]:
        """Return the validated half-open candidate range."""
        if self.stochastic_step_range is None:
            return 0, max(num_steps - int(self.ignore_last), 1)
        start, end = self.stochastic_step_range
        return start, min(end, num_steps - int(self.ignore_last))
