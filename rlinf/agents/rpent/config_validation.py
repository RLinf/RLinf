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

"""Validate the RPent simulator and learner contracts before scheduling."""

from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.scheduler import Cluster
from rlinf.utils.placement import ModelParallelComponentPlacement


def _validate_rpent_common(cfg: DictConfig) -> None:
    model_type = OmegaConf.select(cfg, "rollout.model.model_type", default=None)
    if model_type != "qwen3_5":
        raise ValueError(
            f"RPent requires rollout.model.model_type='qwen3_5', found {model_type!r}."
        )
    env = cfg.tools.rpent
    if env.workflow != "episode":
        raise ValueError(
            f"Unsupported tools.rpent.workflow {env.workflow!r}; expected 'episode'."
        )
    if (
        not env.env_gpus
        or min(env.episodes_per_worker, env.max_env_initializations_per_worker) < 1
    ):
        raise ValueError(
            "RPent needs a non-empty tools.rpent.env_gpus and positive "
            "episodes_per_worker / max_env_initializations_per_worker; "
            f"found env_gpus={list(env.env_gpus)}, "
            f"episodes_per_worker={env.episodes_per_worker}, "
            f"max_env_initializations_per_worker={env.max_env_initializations_per_worker}."
        )
    if env.keep_episodes not in ("all", "failures", "none"):
        raise ValueError(
            "tools.rpent.keep_episodes must be 'all', 'failures', or 'none', "
            f"found {env.keep_episodes!r}."
        )


def validate_rpent_eval_cfg(cfg: DictConfig) -> DictConfig:
    """Validate RPent evaluation settings before launching workers."""
    _validate_rpent_common(cfg)
    if OmegaConf.select(cfg, "rollout.sglang.backend_type", default=None) != "server":
        raise ValueError(
            "RPent evaluation requires rollout.sglang.backend_type='server', "
            f"found {OmegaConf.select(cfg, 'rollout.sglang.backend_type')!r}."
        )
    if not cfg.rollout.get("detokenize", False):
        raise ValueError("RPent evaluation requires rollout.detokenize=True.")
    parser = OmegaConf.select(cfg, "rollout.sglang.tool_call_parser", default=None)
    if parser != "qwen3_coder":
        raise ValueError(
            "RPent evaluation requires rollout.sglang.tool_call_parser='qwen3_coder', "
            f"found {parser!r}."
        )
    if OmegaConf.select(
        cfg, "agentloop.chat_template_kwargs.enable_thinking", default=False
    ):
        reasoning_parser = OmegaConf.select(
            cfg, "rollout.sglang.reasoning_parser", default=None
        )
        if reasoning_parser != "qwen3":
            raise ValueError(
                "RPent evaluation with thinking enabled requires "
                "rollout.sglang.reasoning_parser='qwen3', "
                f"found {reasoning_parser!r}."
            )
    return cfg


def _validate_rpent_mode_cfg(cfg: DictConfig) -> None:
    if cfg.runner.synchronous:
        if cfg.runner.get("async_mode", False):
            raise ValueError(
                "Synchronous RPent requires runner.async_mode=False when "
                "runner.synchronous=True."
            )
        if cfg.data.sampling_batch_size < cfg.data.rollout_batch_size:
            raise ValueError(
                "Synchronous RPent requires data.sampling_batch_size >= "
                f"data.rollout_batch_size; found {cfg.data.sampling_batch_size} < "
                f"{cfg.data.rollout_batch_size}."
            )
        return
    if not cfg.runner.get("async_mode", False):
        raise ValueError(
            "Asynchronous RPent requires runner.async_mode=True when "
            "runner.synchronous=False."
        )
    if cfg.algorithm.recompute_logprobs:
        raise ValueError(
            "Asynchronous RPent requires algorithm.recompute_logprobs=False; "
            "the actor does not recompute behavior logprobs in async mode."
        )


def validate_rpent_cfg(cfg: DictConfig) -> DictConfig:
    """Validate RPent training contracts for synchronous and asynchronous GRPO."""
    _validate_rpent_common(cfg)
    if not OmegaConf.select(cfg, "agentloop.task_name", default=None):
        raise ValueError("RPent training requires agentloop.task_name.")
    tokenizer_type = OmegaConf.select(cfg, "agentloop.agentic_tokenizer", default=None)
    if tokenizer_type != "qwen3_5":
        raise ValueError(
            "RPent training requires agentloop.agentic_tokenizer='qwen3_5', "
            f"found {tokenizer_type!r}."
        )
    if not OmegaConf.select(cfg, "session_server.enable", default=False):
        raise ValueError("RPent training requires session_server.enable=True.")
    min_output_tokens = OmegaConf.select(
        cfg, "session_server.minimum_output_tokens", default=0
    )
    if not 0 < min_output_tokens <= cfg.algorithm.sampling_params.max_new_tokens:
        raise ValueError(
            "RPent training requires 0 < session_server.minimum_output_tokens <= "
            "algorithm.sampling_params.max_new_tokens; found "
            f"{min_output_tokens} and {cfg.algorithm.sampling_params.max_new_tokens}."
        )
    if OmegaConf.select(cfg, "session_server.verify_assembly", default=False):
        raise ValueError(
            "RPent uses the OpenAI chat route with server-side assembly; "
            "set session_server.verify_assembly=False."
        )
    if not OmegaConf.select(
        cfg, "rollout.sglang.align_multimodal_inputs", default=False
    ):
        raise ValueError(
            "RPent image observations require "
            "rollout.sglang.align_multimodal_inputs=True."
        )
    if cfg.critic.get("use_critic_model", False):
        raise ValueError(
            "RPent GRPO runs without a critic model; set critic.use_critic_model=False."
        )
    if cfg.algorithm.adv_type != "grpo_dynamic":
        raise ValueError(
            "RPent training requires algorithm.adv_type='grpo_dynamic', "
            f"found {cfg.algorithm.adv_type!r}."
        )
    if not (
        cfg.rollout.get("return_logprobs", False) or cfg.algorithm.recompute_logprobs
    ):
        raise ValueError(
            "RPent training requires rollout.return_logprobs=True or "
            "algorithm.recompute_logprobs=True."
        )
    _validate_rpent_mode_cfg(cfg)
    if cfg.rollout.get("pause_mode") != "retract":
        raise ValueError(
            "RPent requires rollout.pause_mode='retract' to flush rollout KV cache "
            f"after weight sync; found {cfg.rollout.get('pause_mode')!r}."
        )
    if cfg.algorithm.recompute_logprobs and not (
        cfg.rollout.get("return_logprobs", False)
        and cfg.algorithm.get("importance_sampling_fix", False)
        and cfg.algorithm.get("importance_sampling_clip", 0) > 0
    ):
        raise ValueError(
            "Actor logprob recomputation requires rollout.return_logprobs=True, "
            "algorithm.importance_sampling_fix=True, and positive "
            "algorithm.importance_sampling_clip."
        )
    with open_dict(cfg):
        if cfg.get("rollout_buffer", None) is None:
            cfg.rollout_buffer = {}
        if cfg.rollout_buffer.get("max_dropped_groups_per_step", None) is None:
            cfg.rollout_buffer.max_dropped_groups_per_step = (
                10 * cfg.data.rollout_batch_size
            )
    if cfg.rollout_buffer.max_dropped_groups_per_step < 1:
        raise ValueError(
            "rollout_buffer.max_dropped_groups_per_step must be >= 1, found "
            f"{cfg.rollout_buffer.max_dropped_groups_per_step}."
        )
    if cfg.rollout_buffer.max_inflight_groups < 1:
        raise ValueError(
            "rollout_buffer.max_inflight_groups must be >= 1, found "
            f"{cfg.rollout_buffer.max_inflight_groups}."
        )
    if cfg.rollout_buffer.max_buffered_groups < cfg.data.rollout_batch_size:
        raise ValueError(
            "rollout_buffer.max_buffered_groups must be >= data.rollout_batch_size; "
            f"found {cfg.rollout_buffer.max_buffered_groups} < "
            f"{cfg.data.rollout_batch_size}."
        )
    placement = ModelParallelComponentPlacement(cfg, Cluster())
    if not placement.is_disaggregated:
        raise ValueError(
            "RPent training requires disaggregated placement between actor and "
            f"rollout, found {placement.placement_mode.name}."
        )
    if (
        placement.rollout_num_model_instances != 1
        or placement.rollout_nnodes_per_model_instance != 1
    ):
        raise ValueError(
            "Aligned RPent multimodal training requires a single single-node SGLang rollout "
            f"instance, found {placement.rollout_num_model_instances} instances "
            f"with {placement.rollout_nnodes_per_model_instance} nodes each."
        )
    if cfg.data.rollout_batch_size % placement.actor_dp_size != 0:
        raise ValueError(
            f"data.rollout_batch_size ({cfg.data.rollout_batch_size}) must be "
            f"divisible by the actor data-parallel size ({placement.actor_dp_size})."
        )
    return cfg
