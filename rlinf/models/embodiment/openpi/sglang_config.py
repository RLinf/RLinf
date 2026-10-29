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

"""Configuration checks for the native SGLang Pi0.5 policy."""

from omegaconf import DictConfig, OmegaConf


def validate_sglang_pi05_cfg(cfg: DictConfig, *, only_eval: bool) -> None:
    """Validate the in-process Pi0.5 backend before launching workers."""
    model_cfg = cfg.rollout.model if only_eval else cfg.actor.model
    native = OmegaConf.select(model_cfg, "openpi.backend", default="openpi") == "sglang"
    if not native:
        if (
            not only_eval
            and model_cfg.model_type == "openpi"
            and cfg.rollout.get("rollout_backend") == "sglang"
        ):
            raise ValueError(
                "SGLang embodied training requires actor.model.openpi.backend=sglang"
            )
        return
    if cfg.rollout.get("rollout_backend", "huggingface") != "sglang":
        raise ValueError(
            "openpi.backend=sglang requires rollout.rollout_backend=sglang"
        )
    if not only_eval:
        rollout_backend = OmegaConf.select(
            cfg, "rollout.model.openpi.backend", default="sglang"
        )
        if rollout_backend != "sglang":
            raise ValueError("Actor and rollout must both use openpi.backend=sglang")
        if cfg.algorithm.get("loss_type") != "actor_critic":
            raise ValueError(
                "SGLang Pi0.5 training currently supports actor_critic loss"
            )
        if cfg.algorithm.get("adv_type") not in ("gae", "grpo"):
            raise ValueError("SGLang Pi0.5 training requires gae or grpo advantages")
        if cfg.actor.model.get("is_lora", False) or cfg.actor.get(
            "enable_sft_co_train", False
        ):
            raise ValueError(
                "SGLang Pi0.5 training does not support LoRA or SFT co-training"
            )
        if cfg.actor.get("sync_weight_no_wait", False):
            raise ValueError("SGLang Pi0.5 requires synchronous weight updates")
    if cfg.rollout.get("enable_cuda_graph", False) or cfg.rollout.get(
        "enable_torch_compile", False
    ):
        raise ValueError(
            "SGLang Pi0.5 RL uses eager execution; disable rollout graphs/compile"
        )
    if cfg.runner.get("rtc", {}).get("enabled", False):
        raise ValueError("SGLang Pi0.5 RL does not support RTC")
