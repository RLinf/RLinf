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

"""Validation for the evaluation-only upstream HumanoidArena bridge."""

from omegaconf import DictConfig


def validate_humanoidarena_cfg(
    cfg: DictConfig, model_cfg: DictConfig, only_eval: bool
) -> None:
    """Reject configurations requiring trajectories or trainable model weights."""
    if not only_eval or cfg.env.get("train") is not None:
        raise ValueError(
            "HumanoidArena/WB-WAM HTTP integration supports standalone evaluation only"
        )
    env = cfg.env.eval
    if env.env_type != "humanoidarena" or model_cfg.model_type != "wbwam":
        raise ValueError(
            "HumanoidArena evaluation requires model_type=wbwam and env_type=humanoidarena"
        )
    if cfg.rollout.get("rollout_backend", "huggingface") != "huggingface":
        raise ValueError("WB-WAM requires rollout_backend=huggingface")
    if cfg.rollout.pipeline_stage_num != 1 or env.total_num_envs != 1:
        raise ValueError(
            "HumanoidArena currently requires one environment and one pipeline stage"
        )
    if env.auto_reset or env.ignore_terminations or not env.early_stop_on_done:
        raise ValueError(
            "HumanoidArena requires auto_reset=false, ignore_terminations=false, early_stop_on_done=true"
        )
    if cfg.rollout.get("enable_offload", False) or cfg.env.get("enable_offload", False):
        raise ValueError("HumanoidArena/WB-WAM runtime offload is not supported")
    if cfg.runner.get("enable_decoupled_mode", False) or cfg.runner.get("rtc", {}).get(
        "enabled", False
    ):
        raise ValueError(
            "HumanoidArena evaluation requires synchronous execution without RTC"
        )
    if cfg.runner.get("ckpt_path") or model_cfg.get("is_lora", False):
        raise ValueError(
            "Select upstream WB-WAM weights with model_path and wbwam.weight_file"
        )
    if any(
        cfg.rollout.get(key, False)
        for key in (
            "collect_prev_infos",
            "recompute_logprobs",
            "enable_cuda_graph",
            "enable_torch_compile",
        )
    ):
        raise ValueError(
            "WB-WAM HTTP evaluation does not expose logprobs, CUDA graphs or compiled inference"
        )
    chunk_size = int(model_cfg.num_action_chunks)
    if (
        not 1 <= chunk_size <= 32
        or int(env.chunk_size) != chunk_size
        or model_cfg.action_dim != 40
    ):
        raise ValueError(
            "Expected matching chunk sizes in 1..32 and native action_dim=40"
        )
    if str(env.prompt) != str(model_cfg.wbwam.prompt):
        raise ValueError("Environment and WB-WAM prompts must match")
    if not 1 <= int(env.rollout_epoch) <= 20:
        raise ValueError("HumanoidArena requires 1..20 repeats per task/seed batch")
    post_steps = int(env.get("native_post_termination_steps", 0))
    if post_steps < 0 or (post_steps and not env.get("native_record_video", False)):
        raise ValueError("Nonnegative post-termination steps require native recording")
    required_steps = int(env.max_episode_steps) + post_steps
    if (
        env.max_steps_per_rollout_epoch < required_steps
        or env.max_steps_per_rollout_epoch % chunk_size
    ):
        raise ValueError(
            "Rollout horizon must cover the episode and native recording tail, "
            f"rounded up to a multiple of {chunk_size}; expected >= {required_steps}, "
            f"got {env.max_steps_per_rollout_epoch}"
        )
    if env.video_cfg.save_video or env.get("data_collection", {}).get("enabled", False):
        raise ValueError(
            "Arena HTTP exposes sparse policy observations; video/data collection wrappers are unsupported"
        )
