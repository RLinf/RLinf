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

"""Static configuration and derived requirements for language distillation."""

import math
from dataclasses import dataclass, field
from typing import Mapping

ALIASES = {"kl": "k1", "mse": "k2", "low_var_kl": "k3"}
SAMPLED_MODES = frozenset({"k1", "k2", "k3", "abs"})
DISTRIBUTION_MODES = frozenset(
    {
        "forward_kl_topk",
        "forward_kl_topk_tail",
        "reverse_kl_topk",
        "reverse_kl_topk_tail",
    }
)


@dataclass(frozen=True)
class DistillationLossConfig:
    """Loss parameters; aliases are resolved without changing user configuration."""

    loss_mode: str = "k3"
    use_policy_gradient: bool = True
    use_task_rewards: bool = False
    distillation_loss_coef: float = 1.0
    topk: int = 64
    topk_source: str = "teacher"
    policy_loss_mode: str = "vanilla"
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    clip_ratio_c: float = 3.0
    loss_max_clamp: float | None = None
    log_prob_min_clamp: float | None = None
    tail_mass_eps: float = 1e-6

    @property
    def canonical_loss_mode(self) -> str:
        """Return the registered name for this loss."""
        return ALIASES.get(self.loss_mode, self.loss_mode)

    def __post_init__(self) -> None:
        mode = self.canonical_loss_mode
        if mode not in SAMPLED_MODES | DISTRIBUTION_MODES:
            raise ValueError(f"Unknown distillation loss_mode: {self.loss_mode!r}")
        if mode == "k1" and not self.use_policy_gradient:
            raise ValueError("k1/kl requires use_policy_gradient=True")
        if self.policy_loss_mode != "vanilla":
            raise ValueError("Distillation requires policy_loss_mode='vanilla'")
        if self.topk_source not in {"teacher", "student"}:
            raise ValueError(
                f"Expected teacher/student topk_source, got {self.topk_source!r}"
            )
        if mode in DISTRIBUTION_MODES:
            if type(self.topk) is not int or (self.topk != -1 and self.topk <= 0):
                raise ValueError(f"Expected positive topk or -1, got {self.topk!r}")
            if (
                mode.startswith("reverse")
                and self.topk != -1
                and self.topk_source != "student"
            ):
                raise ValueError("Finite reverse KL requires topk_source='student'")
            if (
                mode.endswith("_tail") or mode.startswith("reverse") or self.topk == -1
            ) and self.log_prob_min_clamp is not None:
                raise ValueError(
                    "Tail, reverse and full KL require log_prob_min_clamp=None"
                )
        for name in ("clip_ratio_low", "clip_ratio_high", "distillation_loss_coef"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Expected finite nonnegative {name}, got {value}")
        if self.clip_ratio_low > 1:
            raise ValueError("clip_ratio_low must not exceed 1")
        if not math.isfinite(self.clip_ratio_c) or self.clip_ratio_c <= 1:
            raise ValueError("clip_ratio_c must be finite and greater than 1")
        if not 0 < self.tail_mass_eps < 1:
            raise ValueError("tail_mass_eps must be in (0, 1)")
        if self.loss_max_clamp is not None and (
            not math.isfinite(self.loss_max_clamp) or self.loss_max_clamp <= 0
        ):
            raise ValueError("loss_max_clamp must be finite and positive")
        if self.log_prob_min_clamp is not None and (
            not math.isfinite(self.log_prob_min_clamp) or self.log_prob_min_clamp > 0
        ):
            raise ValueError("log_prob_min_clamp must be finite and nonpositive")


@dataclass(frozen=True)
class TeacherModelConfig:
    """Frozen model and inference topology, independent of Student placement."""

    model_path: str
    num_replicas: int = 1
    inference: Mapping = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_path:
            raise ValueError("Teacher model_path must be nonempty")
        for name, value in (
            ("num_replicas", self.num_replicas),
            ("tensor_parallel_size", self.inference.get("tensor_parallel_size", 1)),
            ("pipeline_parallel_size", self.inference.get("pipeline_parallel_size", 1)),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"Expected positive integer {name}, got {value!r}")
        if self.inference.get("backend", "sglang") not in {"sglang", "vllm"}:
            raise ValueError("Teacher backend must be sglang or vllm")


@dataclass(frozen=True)
class DistillationConfig:
    """Sample-routed synchronous distillation configuration."""

    enabled: bool = False
    teacher_key: str = "data_source"
    teacher_models: Mapping[str, TeacherModelConfig] = field(default_factory=dict)
    distillation_loss: DistillationLossConfig = field(
        default_factory=DistillationLossConfig
    )


@dataclass(frozen=True)
class DistillationPlan:
    """Requirements shared by rollout, inference and training workers."""

    enabled: bool = False
    canonical_loss_mode: str = ""
    topk: int | None = None
    support_source: str = "sampled"
    target_format: str = "sampled"
    scoring_stage: str = "rollout"
    need_rollout_topk: bool = False
    need_current_logprobs: bool = True
    need_old_logprobs: bool = True
    need_task_rewards: bool = True
    need_task_advantages: bool = True
    need_entropy: bool = False
    use_policy_gradient: bool = False
    teacher_keys: tuple[str, ...] = ()


def build_distillation_plan(cfg: Mapping) -> DistillationPlan:
    """Derive data dependencies without mutating the Hydra configuration."""
    section = cfg.get("distillation")
    if not section or not section.get("enabled", False):
        return DistillationPlan()
    runner = cfg.get("runner", {})
    if (
        runner.get("async_mode", False)
        or runner.get("execution_mode", "sync") != "sync"
    ):
        raise ValueError(
            "Synchronous distillation requires runner.execution_mode='sync' and runner.async_mode=False"
        )
    teachers = section.get("teacher_models", {})
    if not teachers or any(
        not isinstance(key, str) or not key.strip() for key in teachers
    ):
        raise ValueError(
            "Expected at least one Teacher model with a nonempty string key"
        )
    route_field = section.get("teacher_key", "data_source")
    if not isinstance(route_field, str) or not route_field.strip():
        raise ValueError("Expected a nonempty distillation.teacher_key")
    for model in teachers.values():
        TeacherModelConfig(**model)
    loss = DistillationLossConfig(**section.get("distillation_loss", {}))
    mode = loss.canonical_loss_mode
    distribution = mode in DISTRIBUTION_MODES
    source = (
        ("full" if loss.topk == -1 else loss.topk_source) if distribution else "sampled"
    )
    algorithm = cfg.get("algorithm", {})
    if cfg.get("cluster", {}).get("auto_scheduler", False):
        raise ValueError(
            "Synchronous distillation requires fixed placement, not the migration scheduler"
        )
    if loss.use_policy_gradient:
        rollout = cfg.get("rollout", {})
        sampling = (
            rollout.get("sampling_params")
            if rollout.get("rollout_backend", "sglang") == "sglang"
            else None
        )
        if sampling is None:
            sampling = algorithm.get("sampling_params", {})
        if (
            not sampling.get("do_sample", True)
            or sampling.get("temperature", 1.0) != 1.0
            or sampling.get("top_p", 1.0) != 1.0
            or sampling.get("top_k", -1) not in (-1, 0)
            or sampling.get("repetition_penalty", 1.0) != 1.0
        ):
            raise ValueError(
                "Distillation PG currently requires do_sample=True, temperature=1, top_p=1 and disabled sampling top_k; truncated behavior probabilities are not verified"
            )
    if cfg.get("data", {}).get("type") in {"vlm", "vision_language"}:
        raise ValueError("Synchronous distillation supports text-only inputs")
    need_ref = algorithm.get("kl_beta", 0) > 0
    return DistillationPlan(
        enabled=True,
        teacher_keys=tuple(sorted(teachers)),
        canonical_loss_mode=mode,
        topk=loss.topk if distribution else None,
        support_source=source,
        target_format="full"
        if source == "full"
        else "sparse"
        if distribution
        else "sampled",
        scoring_stage="rollout",
        need_rollout_topk=source == "student",
        need_current_logprobs=not distribution
        or loss.use_policy_gradient
        or loss.use_task_rewards
        or need_ref,
        need_old_logprobs=loss.use_policy_gradient or loss.use_task_rewards,
        need_task_rewards=loss.use_task_rewards,
        need_task_advantages=loss.use_task_rewards,
        need_entropy=algorithm.get("entropy_bonus", 0) != 0
        or algorithm.get("calculate_entropy", False),
        use_policy_gradient=loss.use_policy_gradient,
    )


def validate_distillation_cfg(cfg: Mapping) -> DistillationPlan:
    """Validate static distillation settings and return their requirements."""
    return build_distillation_plan(cfg)
