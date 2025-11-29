# Copyright 2025 The RLinf Authors.
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

import dataclasses
import importlib.util
import logging
import os
from dataclasses import asdict
from typing import TYPE_CHECKING, Callable, Optional, Union

import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf, open_dict
from omegaconf.dictconfig import DictConfig
from transformers import AutoConfig

from rlinf.scheduler.cluster import Cluster
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode

if TYPE_CHECKING:
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.transformer.transformer_config import TransformerConfig

logging.getLogger().setLevel(logging.INFO)

SUPPORTED_MODEL_ARCHS = [
    "qwen2.5",
    "qwen2.5_vl",
    "openvla",
    "openvla_oft",
    "qwen3_moe",
    "openpi",
    "mlp_policy",
    "gr00t",
]
SUPPORTED_ROLLOUT_BACKENDS = ["sglang", "vllm"]
SUPPORTED_TASK_TYPE = ["embodied", "reasoning", "coding_online_rl"]
SUPPORTED_TRAINING_BACKENDS = ["megatron", "fsdp"]
__all__ = ["build_config"]


def torch_dtype_from_precision(precision: Union[int, str]) -> torch.dtype:
    if precision in ["bf16", "bf16-mixed"]:
        return torch.bfloat16
    elif precision in [16, "16", "fp16", "16-mixed"]:
        return torch.float16
    elif precision in [32, "32", "fp32", "32-true"]:
        return torch.float32
    elif precision in [None]:
        return None
    else:
        raise ValueError(
            f"Could not parse the precision of `{precision}` to a valid torch.dtype"
        )


@torch.jit.script
def gelu_impl(x):
    """
    OpenAI's gelu implementation.
    """
    return (
        0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x * (1.0 + 0.044715 * x * x)))
    )


def openai_gelu(x):
    return gelu_impl(x)


try:
    jit_fuser = torch.compile
except Exception:
    jit_fuser = torch.jit.script


@jit_fuser
def squared_relu(x):
    return torch.pow(torch.nn.functional.relu(x), 2)


# This is actually Python equivalent of torch.nn.functional.gelu(), also with type hints for ONNX exporter
@torch.jit.script
def erf_gelu(x):
    return (
        x
        * 0.5
        * (
            torch.erf(x / 1.41421).to(dtype=x.dtype)
            + torch.ones_like(x).to(dtype=x.dtype)
        )
    )


def activation_to_func(
    activation: str, openai_gelu: bool = False, onnx_safe: bool = False
) -> Callable:
    """
    Converts an activation function represented as a string to a function.

    Args:
        activation (str): string representation of an activation function, typically gotten from the model config.
        openai_gelu (bool): whether to use the OpenAI GELU implementation. Used with HF compatibility.
        onnx_safe (bool): whether to use the ONNX-compatible implementation of GELU.

    Returns:
        Callable: the activation function.
    """

    supported_activations = [
        "gelu",
        "geglu",
        "reglu",
        "swiglu",
        "squared-relu",
        "fast-geglu",
        "fast-swiglu",
        "fast-reglu",
        "approx-gelu",
    ]

    if activation not in supported_activations:
        raise ValueError(
            f"Unsupported activation {activation}. Supported activations: {supported_activations} "
        )

    # Give openai_gelu precedence over other activations if set, for HF compatibility.
    # Normally this is off and shouldn't affect regular model training.
    if openai_gelu:
        activation_func = openai_gelu
    elif activation in ["gelu", "geglu", "fast-geglu"]:
        activation_func = F.gelu
    elif onnx_safe:
        activation_func = erf_gelu
    elif activation in ["reglu", "fast-reglu"]:
        activation_func = F.relu
    elif activation in ["swiglu", "fast-swiglu"]:
        # SiLU or sigmoid linear unit is the same as swish with beta = 1 (which is what https://arxiv.org/pdf/2002.05202.pdf uses.)
        activation_func = F.silu
    elif activation == "squared-relu":
        activation_func = squared_relu

    return activation_func


def validate_rollout_cfg(cfg, algorithm_cfg):
    def validate_sglang_cfg(cfg):
        assert cfg is not None, (
            "sglang config must be specified if rollout_backend is sglang."
        )
        cfg.attention_backend = cfg.get("attention_backend", "triton")
        cfg.decode_log_interval = cfg.get("decode_log_interval", 500000)
        cfg.use_torch_compile = cfg.get("use_torch_compile", False)
        cfg.torch_compile_max_bs = cfg.get("torch_compile_max_bs", 128)
        return cfg

    def validate_vllm_cfg(cfg):
        assert cfg is not None, (
            "vllm config must be specified if rollout_backend is vllm."
        )
        cfg.attention_backend = cfg.get("attention_backend", "FLASH_ATTN")
        cfg.enable_chunked_prefill = cfg.get("enable_chunked_prefill", True)
        cfg.enable_prefix_caching = cfg.get("enable_prefix_caching", True)
        cfg.enable_flash_infer_sampler = cfg.get("enable_flash_infer_sampler", True)
        cfg.max_num_batched_tokens = cfg.get("max_num_batched_tokens", None)
        cfg.torch_profiler_dir = cfg.get("torch_profiler_dir", None)
        return cfg

    with open_dict(cfg):
        cfg.gpu_memory_utilization = cfg.get("gpu_memory_utilization", 0.65)
        assert cfg.model_dir is not None, "model_dir must be specified for rollout."
        assert cfg.model_arch in SUPPORTED_MODEL_ARCHS, (
            f"model_arch must be one of {SUPPORTED_MODEL_ARCHS}"
        )
        cfg.disable_log_stats = cfg.get("disable_log_stats", False)
        cfg.detokenize = cfg.get("detokenize", False)
        cfg.rollout_backend = cfg.get("rollout_backend", "sglang")
        assert cfg.rollout_backend in SUPPORTED_ROLLOUT_BACKENDS, (
            f"rollout_backend must be one of {SUPPORTED_ROLLOUT_BACKENDS}."
        )
        cfg.return_logprobs = cfg.return_logprobs or algorithm_cfg.get(
            "importance_sampling_fix", False
        )
        cfg.sglang = validate_sglang_cfg(cfg.sglang)
        cfg.vllm = validate_vllm_cfg(cfg.vllm)

    return cfg


def validate_model_cfg_by_hf_config(cfg, hf_model_path):
    # validate by hf config
    hf_config = AutoConfig.from_pretrained(hf_model_path, trust_remote_code=True)

    if "Qwen2ForCausalLM" in hf_config.architectures:
        qkv_bias = True
    else:
        qkv_bias = getattr(hf_config, "attention_bias", False)

    if (
        "Qwen3ForCausalLM" in hf_config.architectures
        or "Qwen3MoeForCausalLM" in hf_config.architectures
    ):
        qk_layernorm = True
    else:
        qk_layernorm = getattr(cfg.model, "qk_layernorm", False)

    with open_dict(cfg):
        rs = getattr(hf_config, "rope_scaling", None)
        if isinstance(rs, dict):
            rtype = rs.get("type", "")
            if rtype in {"linear", "dynamic", "ntk", "yarn"}:
                f = rs.get("factor")
                if f is not None:
                    cfg.model.seq_len_interpolation_factor = float(f)
            else:
                # mrope
                cfg.model.seq_len_interpolation_factor = None
        cfg.model.padded_vocab_size = hf_config.vocab_size
        cfg.model.max_position_embeddings = hf_config.max_position_embeddings
        cfg.model.rotary_base = hf_config.rope_theta
        cfg.model.share_embeddings_and_output_weights = getattr(
            hf_config, "tie_word_embeddings", False
        )
        cfg.model.num_layers = hf_config.num_hidden_layers
        cfg.model.hidden_size = hf_config.hidden_size
        cfg.model.num_attention_heads = hf_config.num_attention_heads
        cfg.model.num_query_groups = hf_config.num_key_value_heads
        cfg.model.ffn_hidden_size = hf_config.intermediate_size
        cfg.model.attention_dropout = hf_config.attention_dropout
        cfg.model.hidden_dropout = getattr(hf_config, "hidden_dropout", 0.0)
        cfg.model.add_qkv_bias = qkv_bias
        cfg.model.qk_layernorm = qk_layernorm
        cfg.model.layernorm_epsilon = hf_config.rms_norm_eps
        cfg.model.head_dim = getattr(
            hf_config,
            "head_dim",
            cfg.model.hidden_size // cfg.model.num_attention_heads,
        )
        if cfg.model.head_dim is not None:
            cfg.model.kv_channels = cfg.model.head_dim

        # MoE model
        cfg.model.num_moe_experts = getattr(hf_config, "num_experts", None)
        cfg.model.moe_ffn_hidden_size = getattr(
            hf_config, "moe_intermediate_size", None
        )
        cfg.model.moe_router_topk = getattr(hf_config, "num_experts_per_tok", 2)

    return cfg


def validate_fsdp_cfg(cfg: DictConfig, resume_dir: Optional[str] = None) -> DictConfig:
    def validate_amp_cfg(config: DictConfig) -> DictConfig:
        if "amp" not in config:
            config.amp = {}
        config.amp.enabled = config.amp.get("enabled", False)
        config.amp.precision = config.amp.get("precision", "bf16")
        assert config.amp.precision in ["fp16", "bf16", "fp32"], (
            "fsdp.amp.precision must be one of ['fp16', 'bf16', 'fp32']"
        )
        config.amp.use_grad_scaler = config.amp.get("use_grad_scaler", False)
        return config

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.fsdp_config.strategy = cfg.fsdp_config.get("strategy", "fsdp")

        cfg.fsdp_config.sharding_strategy = cfg.fsdp_config.get(
            "sharding_strategy", "full_shard"
        )

        cfg.fsdp_config.forward_prefetch = cfg.fsdp_config.get(
            "forward_prefetch", False
        )
        cfg.fsdp_config.limit_all_gathers = cfg.fsdp_config.get(
            "limit_all_gathers", False
        )
        cfg.fsdp_config.backward_prefetch = cfg.fsdp_config.get(
            "backward_prefetch", None
        )
        cfg.fsdp_config.use_orig_params = cfg.fsdp_config.get("use_orig_params", False)
        cfg.fsdp_config.use_liger_kernel = cfg.fsdp_config.get(
            "use_liger_kernel", False
        )
        cfg.fsdp_config = validate_amp_cfg(cfg.fsdp_config)

        cfg.fsdp_config.cpu_offload = cfg.fsdp_config.get("cpu_offload", False)
        cfg.fsdp_config.offload_pin_memory = cfg.fsdp_config.get(
            "offload_pin_memory", False
        )
        cfg.fsdp_config.reshard_after_forward = cfg.fsdp_config.get(
            "reshard_after_forward", True
        )
        cfg.fsdp_config.enable_gradient_accumulation = cfg.fsdp_config.get(
            "enable_gradient_accumulation", False
        )

        if resume_dir is not None:
            cfg.fsdp_config.use_orig_params = True

        assert cfg.fsdp_config.backward_prefetch in [
            None,
            "pre",
            "post",
        ], "fsdp_config.backward_prefetch must be one of [None, 'pre', 'post']"

        # validate mixed precision config
        assert hasattr(cfg.fsdp_config, "mixed_precision"), (
            "fsdp_config.mixed_precision is required in FSDP actor configuration."
        )

        mixed_precision_config = cfg.fsdp_config.mixed_precision
        mixed_precision_config.param_dtype = mixed_precision_config.get(
            "param_dtype", "bf16"
        )
        mixed_precision_config.reduce_dtype = mixed_precision_config.get(
            "reduce_dtype", "bf16"
        )
        mixed_precision_config.buffer_dtype = mixed_precision_config.get(
            "buffer_dtype", "fp32"
        )

    return cfg


def validate_megatron_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.mcore_gpt = cfg.get("mcore_gpt", True)
        spec_name = cfg.get("spec_name", "local_gpt")
        cfg.spec_name = spec_name

        # Pad the vocab size to be divisible by this value.
        cfg.model.make_vocab_size_divisible_by = cfg.model.get(
            "make_vocab_size_divisible_by", 8
        )
        cfg.use_torch_fsdp2 = False

        # training args for megatron
        cfg.megatron.load = cfg.get("checkpoint_load_path", None)
        use_hf_ckpt = cfg.megatron.get("use_hf_ckpt", False)
        if cfg.megatron.load is None:
            assert use_hf_ckpt, (
                "checkpoint_load_path is required if use_hf_ckpt is False"
            )
        else:
            assert not use_hf_ckpt, (
                "checkpoint_load_path should be None if use_hf_ckpt is True"
            )
        cfg.megatron.pretrained_checkpoint = cfg.get("pretrained_checkpoint", None)
        cfg.megatron.save = None
        cfg.megatron.micro_batch_size = cfg.get("micro_batch_size", 1)
        cfg.megatron.global_batch_size = cfg.get("global_batch_size", 1)
        cfg.megatron.tp_comm_overlap_cfg = cfg.megatron.get("tp_comm_overlap_cfg", None)
        cfg.megatron.decoder_tp_comm_overlap = cfg.megatron.get(
            "decoder_tp_comm_overlap", False
        )
        cfg.megatron.timing_log_level = cfg.megatron.get(
            "timing_log_level", 0
        )  # choices=range(0,3)
        cfg.megatron.timing_log_option = cfg.megatron.get(
            "timing_log_option", "minmax"
        )  # choices=['max', 'minmax', 'all']

        # Megatron >= 0.12.0
        cfg.megatron.init_model_with_meta_device = cfg.megatron.get(
            "init_model_with_meta_device", False
        )
        cfg.megatron.use_torch_fsdp2 = cfg.megatron.get("use_torch_fsdp2", False)
        cfg.megatron.use_custom_fsdp = cfg.megatron.get("use_custom_fsdp", False)
        cfg.megatron.check_for_large_grads = cfg.megatron.get(
            "check_for_large_grads", False
        )
        cfg.megatron.ddp_num_buckets = cfg.megatron.get("ddp_num_buckets", None)
        cfg.megatron.ddp_pad_buckets_for_high_nccl_busbw = cfg.megatron.get(
            "ddp_pad_buckets_for_high_nccl_busbw", False
        )
        cfg.megatron.enable_gloo_process_groups = cfg.megatron.get(
            "enable_gloo_process_groups", True
        )

        # ddp config
        cfg.megatron.check_for_nan_in_loss_and_grad = cfg.megatron.get(
            "check_for_nan_in_loss_and_grad", False
        )
        cfg.megatron.ddp_bucket_size = cfg.megatron.get("ddp_bucket_size", None)
        cfg.megatron.ddp_average_in_collective = cfg.megatron.get(
            "ddp_average_in_collective", False
        )
        cfg.megatron.accumulate_allreduce_grads_in_fp32 = cfg.megatron.get(
            "accumulate_allreduce_grads_in_fp32", True
        )

        # profiler
        cfg.megatron.use_profiler = cfg.megatron.get("use_profiler", False)
        if cfg.megatron.use_profiler:
            cfg.megatron.profiler.schedule_warmup = cfg.megatron.profiler.get(
                "schedule_warmup", 3
            )
            cfg.megatron.profiler.schedule_active = cfg.megatron.profiler.get(
                "schedule_active", 1
            )

        # distributed
        # If set, distributed ranks initialize order is changed from tp-cp-ep-dp-pp to tp-cp-ep-pp-dp.
        cfg.megatron.use_tp_pp_dp_mapping = cfg.megatron.get(
            "use_tp_pp_dp_mapping", False
        )
        # Which backend to use for distributed training. Support 'nccl' and 'gloo'
        cfg.megatron.distributed_backend = cfg.megatron.get(
            "distributed_backend", "nccl"
        )
        cfg.megatron.distributed_timeout_minutes = cfg.megatron.get(
            "distributed_timeout_minutes", 10
        )
        cfg.megatron.num_distributed_optimizer_instances = cfg.megatron.get(
            "num_distributed_optimizer_instances", 1
        )
        cfg.megatron.nccl_communicator_config_path = cfg.megatron.get(
            "nccl_communicator_config_path", None
        )
        cfg.megatron.encoder_tensor_model_parallel_size = cfg.megatron.get(
            "encoder_tensor_model_parallel_size", 0
        )
        cfg.megatron.encoder_pipeline_model_parallel_size = cfg.megatron.get(
            "encoder_pipeline_model_parallel_size", 0
        )

        # checkpoint
        cfg.megatron.rerun_mode = cfg.megatron.get(
            "rerun_mode", "disabled"
        )  # choices=['disabled', 'validate_results', 'report_stats']
        cfg.megatron.error_injection_rate = cfg.megatron.get(
            "error_injection_rate", 0
        )  # Rate at which to inject unexpected results, e.g. 1000 means once every 1000 result validations
        cfg.megatron.error_injection_type = cfg.megatron.get(
            "error_injection_type", "transient_error"
        )  # choices=['correct_result', 'transient_error', 'persistent_error']

        cfg.megatron.moe_use_upcycling = cfg.megatron.get("moe_use_upcycling", False)
        cfg.megatron.async_save = cfg.megatron.get("async_save", False)
        cfg.megatron.use_dist_ckpt = cfg.megatron.get("use_dist_ckpt", False)
        cfg.megatron.no_load_optim = cfg.megatron.get("no_load_optim", False)
        cfg.megatron.no_load_rng = cfg.megatron.get("no_load_rng", False)
        cfg.megatron.no_save_optim = cfg.megatron.get("no_save_optim", False)
        cfg.megatron.no_save_rng = cfg.megatron.get("no_save_rng", False)
        cfg.megatron.ckpt_fully_parallel_save = cfg.megatron.get(
            "ckpt_fully_parallel_save", False
        )
        cfg.megatron.ckpt_format = cfg.megatron.get("ckpt_format", "torch")
        cfg.megatron.ckpt_convert_format = cfg.megatron.get(
            "ckpt_convert_format", None
        )  # choices=[None, 'torch', 'torch_dist', 'zarr']
        cfg.megatron.auto_detect_ckpt_format = cfg.megatron.get(
            "auto_detect_ckpt_format", False
        )
        cfg.megatron.non_persistent_save_interval = cfg.megatron.get(
            "non_persistent_save_interval", None
        )
        cfg.megatron.non_persistent_ckpt_type = cfg.megatron.get(
            "non_persistent_ckpt_type", None
        )
        cfg.megatron.non_persistent_local_ckpt_dir = cfg.megatron.get(
            "non_persistent_local_ckpt_dir", None
        )
        cfg.megatron.non_persistent_global_ckpt_dir = cfg.megatron.get(
            "non_persistent_global_ckpt_dir", None
        )
        cfg.megatron.non_persistent_local_ckpt_algo = cfg.megatron.get(
            "non_persistent_local_ckpt_algo", "fully_parallel"
        )  # choices=['fully_parallel', 'atomic']
        cfg.megatron.ckpt_convert_update_legacy_dist_opt_format = cfg.megatron.get(
            "ckpt_convert_update_legacy_dist_opt_format", False
        )
        cfg.megatron.finetune = cfg.megatron.get("finetune", False)
        cfg.megatron.ckpt_assume_constant_structure = cfg.megatron.get(
            "ckpt_assume_constant_structure", False
        )
        cfg.megatron.log_progress = cfg.megatron.get("log_progress", False)
        cfg.megatron.exit_on_missing_checkpoint = cfg.megatron.get(
            "exit_on_missing_checkpoint", True
        )
        cfg.megatron.retro_add_retriever = cfg.megatron.get(
            "retro_add_retriever", False
        )
        cfg.megatron.data_parallel_random_init = cfg.megatron.get(
            "data_parallel_random_init", False
        )
        cfg.megatron.use_tokenizer_model_from_checkpoint_args = cfg.megatron.get(
            "use_tokenizer_model_from_checkpoint_args", False
        )

        cfg.model.tensor_model_parallel_size = cfg.model.get(
            "tensor_model_parallel_size", 1
        )
        cfg.model.pipeline_model_parallel_size = cfg.model.get(
            "pipeline_model_parallel_size", 1
        )
        cfg.model.virtual_pipeline_model_parallel_size = cfg.model.get(
            "virtual_pipeline_model_parallel_size", None
        )
        cfg.model.pipeline_model_parallel_split_rank = cfg.model.get(
            "pipeline_model_parallel_split_rank", None
        )
        cfg.model.context_parallel_size = cfg.model.get("context_parallel_size", 1)

        cfg.model.expert_model_parallel_size = cfg.model.get(
            "expert_model_parallel_size", 1
        )

        cfg.model.expert_tensor_parallel_size = cfg.model.get(
            "expert_tensor_parallel_size", None
        )

        cfg.model.moe_grouped_gemm = cfg.model.get("moe_grouped_gemm", None)
        assert cfg.model.moe_grouped_gemm in [None, "te"], (
            f"grouped_gemm type only avail in [null, te]. get value ({cfg.model.moe_grouped_gemm})"
        )

        if cfg.model.expert_tensor_parallel_size is not None:
            assert (
                cfg.model.expert_tensor_parallel_size
                <= cfg.model.tensor_model_parallel_size
            ), (
                f"expert_tensor_parallel_size ({cfg.model.expert_tensor_parallel_size}) must be less than or equal to tensor_model_parallel_size ({cfg.model.tensor_model_parallel_size})"
            )

        cfg.model.position_embedding_type = cfg.model.get(
            "position_embedding_type", "learned_absolute"
        )
        cfg.model.rotary_percentage = cfg.model.get("rotary_percentage", 1.0)
        cfg.model.seq_len_interpolation_factor = cfg.model.get(
            "seq_len_interpolation_factor", None
        )
        cfg.model.rotary_base = cfg.model.get("rotary_base", 10000)
        cfg.model.share_embeddings_and_output_weights = cfg.model.get(
            "share_embeddings_and_output_weights", False
        )

        cfg.model.gradient_accumulation_fusion = cfg.model.get(
            "gradient_accumulation_fusion", True
        )
        cfg.model.masked_softmax_fusion = cfg.model.get("masked_softmax_fusion", True)
        cfg.model.persist_layer_norm = cfg.model.get("persist_layer_norm", True)

        cfg.model.padded_vocab_size = cfg.model.get("padded_vocab_size", None)
        cfg.model.use_cpu_initialization = cfg.model.get(
            "use_cpu_initialization", False
        )
        cfg.model.add_position_embedding = cfg.model.get("add_position_embedding", True)

        cfg.model.variable_seq_lengths = cfg.model.get("variable_seq_lengths", True)
        cfg.model.add_bias_linear = cfg.model.get("add_bias_linear", False)

        cfg.optim.fp16 = (
            torch_dtype_from_precision(cfg.model.precision) == torch.float16
        )
        cfg.optim.bf16 = (
            torch_dtype_from_precision(cfg.model.precision) == torch.bfloat16
        )
        cfg.optim.weight_decay = cfg.optim.get("weight_decay", 0.01)
        cfg.optim.overlap_param_gather_with_optimizer_step = cfg.optim.get(
            "overlap_param_gather_with_optimizer_step", False
        )

        # learning rate
        cfg.lr_sched.lr = cfg.optim.get("lr", None)
        cfg.lr_sched.min_lr = cfg.lr_sched.get("min_lr", 0.0)
        # lr_decay_style choices=['constant', 'linear', 'cosine', 'inverse-square-root', 'WSD']
        cfg.lr_sched.lr_decay_style = cfg.lr_sched.get("lr_decay_style", "constant")
        # weight_decay_incr_style: Weight decay increment function. choices=['constant', 'linear', 'cosine']
        cfg.lr_sched.weight_decay_incr_style = cfg.lr_sched.get(
            "weight_decay_incr_style", "constant"
        )
        # lr_wsd_decay_style choices=['exponential', 'linear', 'cosine']
        cfg.lr_sched.lr_wsd_decay_style = cfg.lr_sched.get(
            "lr_wsd_decay_style", "exponential"
        )

        # TODO fix this
        cfg.megatron.train_iters = 100000
        # lr_decay_iters: number of iterations to decay learning rate over, defaults to train_iters
        cfg.lr_sched.lr_decay_iters = cfg.lr_sched.get("lr_decay_iters", None)
        cfg.lr_sched.lr_wsd_decay_iters = cfg.lr_sched.get("lr_wsd_decay_iters", None)
        cfg.lr_sched.lr_warmup_init = cfg.lr_sched.get("lr_warmup_init", 0.0)
        cfg.lr_sched.lr_warmup_iters = cfg.lr_sched.get("lr_warmup_iters", 0)
        cfg.lr_sched.lr_warmup_fraction = cfg.lr_sched.get("lr_warmup_fraction", None)
        cfg.lr_sched.use_checkpoint_opt_param_scheduler = cfg.lr_sched.get(
            "use_checkpoint_opt_param_scheduler", True
        )
        cfg.lr_sched.override_opt_param_scheduler = cfg.lr_sched.get(
            "override_opt_param_scheduler", False
        )

        if cfg.lr_sched.lr_decay_style == "constant":
            assert cfg.lr_sched.get("start_weight_decay") is None
            assert cfg.lr_sched.get("end_weight_decay") is None
            cfg.lr_sched.start_weight_decay = cfg.optim.weight_decay
            cfg.lr_sched.end_weight_decay = cfg.optim.weight_decay
        else:
            if not hasattr(cfg.lr_sched, "start_weight_decay"):
                raise ValueError(
                    "Error: 'start_weight_decay' is missing from 'cfg.lr_sched'"
                )
            if not hasattr(cfg.lr_sched, "end_weight_decay"):
                raise ValueError(
                    "Error: 'end_weight_decay' is missing from 'cfg.lr_sched'"
                )
            assert cfg.lr_sched.start_weight_decay is not None
            assert cfg.lr_sched.end_weight_decay is not None

        # TODO. Following args are needed for AUTO mode now, but will be removed in the future.
        cfg.megatron.transformer_impl = getattr(
            cfg.megatron, "transformer_impl", "transformer_engine"
        )
        cfg.megatron.swiglu = cfg.model.activation in ["swiglu", "fast-swiglu"]
        cfg.megatron.untie_embeddings_and_output_weights = (
            not cfg.model.share_embeddings_and_output_weights
        )
        # In RLinf, padded_vocab_size is set to hf_config.vocab_size, so make_vocab_size_divisible_by=1
        cfg.megatron.make_vocab_size_divisible_by = 1
        if cfg.model.normalization == "rmsnorm":
            cfg.megatron.normalization = "RMSNorm"

    return cfg


def validate_embodied_cfg(cfg):
    assert cfg.actor.model.model_name in SUPPORTED_MODEL_ARCHS, (
        f"Model {cfg.actor.model.model_name} is not supported"
    )

    # process num-envs
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement

    component_placement = HybridComponentPlacement(
        cfg, Cluster(num_nodes=cfg.cluster.num_nodes)
    )
    stage_num = cfg.rollout.pipeline_stage_num
    env_world_size = component_placement.get_world_size("env")
    cfg.algorithm.num_group_envs = (
        cfg.algorithm.num_group_envs // stage_num // env_world_size
    )
    cfg.env.eval.num_envs = cfg.env.eval.num_envs // stage_num // env_world_size

    with open_dict(cfg):
        if cfg.env.train.simulator_type == "maniskill":

            def get_robot_control_mode(robot: str):
                if "google_robot_static" in robot:
                    return "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
                elif "widowx" in robot:
                    return "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"
                elif "panda-qpos" in robot:
                    return None
                else:
                    raise NotImplementedError(f"Robot {robot} not supported")

            cfg.env.train.init_params.control_mode = get_robot_control_mode(
                cfg.actor.model.policy_setup
            )
            cfg.env.eval.init_params.control_mode = get_robot_control_mode(
                cfg.actor.model.policy_setup
            )
        elif cfg.env.train.simulator_type == "libero":
            if cfg.actor.model.get("num_images_in_input", 1) > 1:
                assert cfg.actor.model.get("use_wrist_image", False), (
                    "Invalid config: Multiple input images are enabled "
                    "(num_images_in_input > 1) but 'use_wrist_image' is set to False. "
                    "Please enable wrist images by setting 'use_wrist_image=True'."
                )
        elif cfg.env.train.simulator_type == "behavior":
            import omnigibson as og

            assert cfg.env.train.base_config_name == "r1pro_behavior", (
                f"Only r1pro_behavior is supported for omnigibson, got {cfg.env.train.base_config_name}"
            )
            # Load the pre-selected configuration and set the online_sampling flag
            config_filename = os.path.join(
                og.example_config_path, "r1pro_behavior.yaml"
            )
            omnigibson_cfg = yaml.load(
                open(config_filename, "r"), Loader=yaml.FullLoader
            )
            omnigibson_cfg = OmegaConf.create(omnigibson_cfg)
            cfg.env.train.omnigibson_cfg = omnigibson_cfg
            cfg.env.eval.omnigibson_cfg = omnigibson_cfg

            # Also accepts int or list/tuple of tokens (ints or range strings)
            def parse_activity_ids(activity_ids) -> list[int]:
                if activity_ids is None:
                    return []
                out: list[int] = []

                def _add_token(tok: str):
                    tok = tok.strip()
                    if not tok:
                        return
                    if "-" in tok:
                        start, end = tok.split("-", 1)
                        start_i, end_i = int(start.strip()), int(end.strip())
                        if end_i < start_i:
                            start_i, end_i = end_i, start_i
                        out.extend(range(start_i, end_i + 1))
                    else:
                        out.append(int(tok))

                if isinstance(activity_ids, int):
                    out.append(int(activity_ids))
                elif isinstance(activity_ids, (list, tuple)):
                    for item in activity_ids:
                        if isinstance(item, int):
                            out.append(int(item))
                        else:
                            for tok in str(item).split(","):
                                _add_token(tok)
                else:
                    for tok in str(activity_ids).split(","):
                        _add_token(tok)
                return out

            cfg.env.train.tasks.activity_task_indices = parse_activity_ids(
                cfg.env.train.tasks.activity_task_indices
            )
            cfg.env.eval.tasks.activity_task_indices = parse_activity_ids(
                cfg.env.eval.tasks.activity_task_indices
            )
            assert (
                len(cfg.env.train.tasks.activity_task_indices) > 0
                and len(cfg.env.eval.tasks.activity_task_indices) > 0
            ), "No activity IDs provided"

    return cfg


def validate_reasoning_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.rollout.model_arch in SUPPORTED_MODEL_ARCHS, (
        f"Model {cfg.rollout.model_arch} is not supported"
    )

    assert cfg.algorithm.recompute_logprobs or cfg.rollout.return_logprobs, (
        "One of `algorithm.recompute_logprobs` or `rollout.return_logprobs` must be True to compute `prev_logprobs`."
    )

    if cfg.algorithm.recompute_logprobs and cfg.rollout.return_logprobs:
        assert cfg.algorithm.get("importance_sampling_fix", False), (
            "Importance sampling fix must be enabled if both `algorithm.recompute_logprobs` and `rollout.return_logprobs` are True."
        )

    with open_dict(cfg):
        cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get(
            "training_batch_size_per_gpu", 1
        )
        cfg.algorithm.n_minibatches = cfg.algorithm.get("n_minibatches", 1)
        cfg.algorithm.max_num_gen_batches = cfg.algorithm.get("max_num_gen_batches", 1)
        cfg.actor.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
        cfg.actor.global_batch_size = (
            cfg.data.rollout_batch_size
            * cfg.algorithm.group_size
            // cfg.algorithm.n_minibatches
        )
        assert cfg.actor.micro_batch_size >= 1
        assert cfg.actor.global_batch_size >= 1
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )

        # add configs for importance sampling fix
        cfg.algorithm.recompute_logprobs = (
            cfg.algorithm.recompute_logprobs
            or cfg.algorithm.get("importance_sampling_fix", False)
        )

        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_coding_online_rl_cfg(cfg: DictConfig) -> DictConfig:
    assert cfg.rollout.model_arch == "qwen2.5", (
        f"Model {cfg.rollout.model_arch} is not supported"
    )

    assert cfg.algorithm.recompute_logprobs or cfg.rollout.return_logprobs, (
        "One of `algorithm.recompute_logprobs` or `rollout.return_logprobs` must be True to compute `prev_logprobs`."
    )

    if cfg.algorithm.recompute_logprobs and cfg.rollout.return_logprobs:
        assert cfg.algorithm.get("importance_sampling_fix", False), (
            "Importance sampling fix must be enabled if both `algorithm.recompute_logprobs` and `rollout.return_logprobs` are True."
        )

    assert cfg.algorithm.recompute_logprobs, (
        "Online coding task must use recompute_logprobs"
    )

    assert cfg.actor.training_backend == "megatron", (
        "Online coding task must use megatron training backend"
    )

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)
    assert component_placement.placement_mode == PlacementMode.DISAGGREGATED, (
        "Online coding task must use disaggregated placement mode"
    )

    with open_dict(cfg):
        cfg.algorithm.training_batch_size_per_gpu = cfg.algorithm.get(
            "training_batch_size_per_gpu", 1
        )
        cfg.algorithm.n_minibatches = cfg.algorithm.get("n_minibatches", 1)
        cfg.algorithm.max_num_gen_batches = cfg.algorithm.get("max_num_gen_batches", 1)
        cfg.actor.micro_batch_size = cfg.algorithm.training_batch_size_per_gpu
        cfg.actor.global_batch_size = (
            cfg.data.rollout_batch_size
            * cfg.algorithm.group_size
            // cfg.algorithm.n_minibatches
        )
        assert cfg.actor.micro_batch_size >= 1
        assert cfg.actor.global_batch_size >= 1
        assert cfg.runner.seq_length > cfg.data.max_prompt_length, (
            f"runner.seq_length ({cfg.runner.seq_length}) must be greater than data.max_prompt_length ({cfg.data.max_prompt_length})"
        )

        # add configs for importance sampling fix
        cfg.algorithm.recompute_logprobs = (
            cfg.algorithm.recompute_logprobs
            or cfg.algorithm.get("importance_sampling_fix", False)
        )

        cfg.rollout = validate_rollout_cfg(cfg.rollout, cfg.algorithm)
    return cfg


def validate_cfg(cfg: DictConfig) -> DictConfig:
    OmegaConf.set_struct(cfg, True)

    assert cfg.runner.task_type in SUPPORTED_TASK_TYPE, (
        f"task_type must be one of {SUPPORTED_TASK_TYPE}"
    )
    if cfg.runner.task_type == "embodied":
        cfg = validate_embodied_cfg(cfg)
    elif cfg.runner.task_type == "reasoning":
        cfg = validate_reasoning_cfg(cfg)
    elif cfg.runner.task_type == "coding_online_rl":
        cfg = validate_coding_online_rl_cfg(cfg)

    if cfg.algorithm.adv_type in ("grpo", "reinpp_baseline"):
        assert cfg.algorithm.group_size > 1

    assert cfg.actor.training_backend in SUPPORTED_TRAINING_BACKENDS, (
        f"Unsupported training_backend {cfg.actor.training_backend}. Supported training backends are {SUPPORTED_TRAINING_BACKENDS}."
    )

    if cfg.actor.training_backend == "megatron":
        cfg.actor = validate_megatron_cfg(cfg.actor)
        cfg.actor = validate_model_cfg_by_hf_config(cfg.actor, cfg.rollout.model_dir)
        # TODO. Need actually pad padded_vocab_size.
        assert (
            cfg.actor.model.padded_vocab_size
            % cfg.actor.model.tensor_model_parallel_size
            == 0
        ), (
            f"padded_vocab_size ({cfg.actor.model.padded_vocab_size}) must be divisible by tensor_model_parallel_size ({cfg.actor.model.tensor_model_parallel_size})"
        )
    elif cfg.actor.training_backend == "fsdp":
        cfg.actor = validate_fsdp_cfg(cfg.actor, cfg.runner.get("resume_dir", None))

    if cfg.critic.use_critic_model and cfg.critic.training_backend == "megatron":
        cfg.critic = validate_megatron_cfg(cfg.critic)
        cfg.critic = validate_model_cfg_by_hf_config(cfg.critic, cfg.rollout.model_dir)
    elif cfg.critic.use_critic_model and cfg.critic.training_backend == "fsdp":
        cfg.critic = validate_fsdp_cfg(cfg.critic)

    return cfg


def build_config(cls, cfg):
    if not isinstance(cfg, (dict, DictConfig)):
        cfg = asdict(cfg)

    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name in cfg:
            kwargs[f.name] = cfg.get(f.name)

    return cls(**kwargs)


def build_transformer_config(cfg) -> "TransformerConfig":
    """
    Builds the megatron core transformer config for the model.
    For attributes in the RLinf model config that are the same
    as the megatron core TransformerConfig, we will use the value from the RLinf model config.
    For attributes in TransformerConfig that are not in the RLinf model config, we add custom logic.
    """
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.utils import (
        init_method_normal,
        scaled_init_method_normal,
    )

    # get model parallel configs
    model_parallel_config = _build_model_parallel_config(cfg)

    # create a dictionary copy of the model config
    cfg = OmegaConf.to_container(cfg, resolve=True)

    # create a dict to store the transformer config arguments
    transformer_config_dict = {}

    num_layers = cfg.get("num_layers", 1)
    if num_layers % cfg.get("pipeline_model_parallel_size", 1) != 0:
        raise ValueError(
            f"num_layers ({cfg.num_layers}) should be divisible by "
            f"pipeline_model_parallel_size ({cfg.get('pipeline_model_parallel_size', 1)})"
        )

    add_bias_linear = cfg.get("add_bias_linear", True)
    add_qkv_bias = cfg.get("add_qkv_bias", False)

    activation = cfg.get("activation", "gelu")
    gated_linear_unit = activation.endswith("glu")
    # TODO: need to check which activation functions are supported in mcore
    activation_func = activation_to_func(
        activation, openai_gelu=cfg.get("openai_gelu", False)
    )

    normalization = cfg.get("normalization", "layernorm").lower()
    layernorm_zero_centered_gamma = cfg.get(
        "normalization", "layernorm"
    ) == "layernorm1p" or cfg.get("layernorm_zero_centered_gamma", False)
    if normalization == "layernorm":
        normalization = "LayerNorm"
    elif normalization == "rmsnorm":
        normalization = "RMSNorm"
    elif normalization == "layernorm1p":
        normalization = "LayerNorm"
        layernorm_zero_centered_gamma = True
    else:
        logging.warning(
            f"The normalization type: {normalization} might not be supported in megatron core."
            f"Supported types are LayerNorm and RMSNorm."
        )

    tp_comm_overlap = cfg.get("tp_comm_overlap", False)

    if not cfg.get("fp8", False):
        fp8 = None
    elif cfg.get("fp8_e4m3", False):
        fp8 = "e4m3"
    elif cfg.get("fp8_hybrid", False):
        fp8 = "hybrid"
    else:
        raise ValueError(
            "fp8 enabled but fp8_format (fp8_e4m3 | fp8_hybrid) is not set."
        )

    init_method_std = cfg.get("init_method_std", 0.02)
    # default used in mcore
    init_method = init_method_normal(init_method_std)

    output_layer_init_method = init_method

    use_scaled_init_method = cfg.get("use_scaled_init_method", True)
    if use_scaled_init_method:
        output_layer_init_method = scaled_init_method_normal(
            init_method_std, num_layers=num_layers
        )

    attention_softmax_in_fp32 = cfg.get("attention_softmax_in_fp32", True)
    apply_query_key_layer_scaling = cfg.get("apply_query_key_layer_scaling", False)

    rotary_interleaved = cfg.get("rotary_interleaved", False)

    if apply_query_key_layer_scaling:
        if model_parallel_config.fp16:
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "1"
        else:
            logging.warning(
                "apply_query_key_layer_scaling is only enabled when using FP16, setting it to False "
                "and setting NVTE_APPLY_QK_LAYER_SCALING=0"
            )
            os.environ["NVTE_APPLY_QK_LAYER_SCALING"] = "0"
            apply_query_key_layer_scaling = False

    if apply_query_key_layer_scaling:
        attention_softmax_in_fp32 = True

    bias_activation_fusion = cfg.get("bias_activation_fusion", True)

    bias_dropout_fusion = cfg.get("bias_dropout_fusion", True)

    apply_rope_fusion = cfg.get("apply_rope_fusion", False)

    # TODO: need to check if recompute APIs are matching up properly
    recompute_granularity = cfg.get("recompute_granularity", None)
    recompute_method = cfg.get("recompute_method", None)
    recompute_num_layers = cfg.get("recompute_num_layers", None)

    tp_only_amax_red = cfg.get("tp_only_amax_red", False)

    if cfg.get("enable_cuda_graph", False):
        if importlib.util.find_spec("transformer_engine") is None:
            raise ImportError(
                "Can not import transformer_engine, which is required for cudagraphs."
            )
        assert cfg.get("use_te_rng_tracker", False), (
            "Transformer engine's RNG tracker is required for cudagraphs, this can be enabled with \
            'use_te_rng_tracker=True'."
        )

    # any configs that are not in the RLinf model config will be added here
    config_mapping = {
        "apply_query_key_layer_scaling": apply_query_key_layer_scaling,
        "apply_residual_connection_post_layernorm": False,  # we don't use this in NeMo
        "add_bias_linear": add_bias_linear,
        "add_qkv_bias": add_qkv_bias,
        "gated_linear_unit": gated_linear_unit,
        "activation_func": activation_func,
        "normalization": normalization,
        "layernorm_zero_centered_gamma": layernorm_zero_centered_gamma,
        "init_method": init_method,
        "output_layer_init_method": output_layer_init_method,
        "attention_softmax_in_fp32": attention_softmax_in_fp32,
        "bias_activation_fusion": bias_activation_fusion,
        "bias_dropout_fusion": bias_dropout_fusion,
        "apply_rope_fusion": apply_rope_fusion,
        "recompute_granularity": recompute_granularity,
        "recompute_method": recompute_method,
        "recompute_num_layers": recompute_num_layers,
        "distribute_saved_activations": False,  # not currently used in NeMo
        "fp8": fp8,
        "tp_comm_overlap": tp_comm_overlap,
        "rotary_interleaved": rotary_interleaved,
        "deallocate_pipeline_outputs": False,
        "tp_only_amax_red": tp_only_amax_red,
        "qk_layernorm": cfg.get("qk_layernorm", False),
        "kv_channels": cfg.get("head_dim", None),
        # MoE related
        "num_moe_experts": cfg.get("num_moe_experts", None),
        "moe_ffn_hidden_size": cfg.get("moe_ffn_hidden_size", None),
        # now the sequential mlp should ffn hidden size == moe_ffn_hidden_size
        "ffn_hidden_size": cfg.get("moe_ffn_hidden_size", None)
        or cfg.get("ffn_hidden_size", None),
        "moe_router_load_balancing_type": cfg.get(
            "moe_router_load_balancing_type", "aux_loss"
        ),
        "moe_router_topk": cfg.get("moe_router_topk", 2),
        "moe_grouped_gemm": cfg.get("moe_grouped_gemm", False),
        "moe_aux_loss_coeff": cfg.get(
            "moe_aux_loss_coeff", 0
        ),  # 1e-2 would be a good start value for load balance loss.
        "moe_z_loss_coeff": cfg.get(
            "moe_z_loss_coeff", None
        ),  # 1e-3 would be a good start value for z-loss
        "moe_input_jitter_eps": cfg.get("moe_input_jitter_eps", None),
        "moe_token_dropping": cfg.get("moe_token_dropping", False),
        "enable_cuda_graph": cfg.get("enable_cuda_graph", False),
    }

    # populate the transformer config dict
    for field in dataclasses.fields(TransformerConfig):
        # config mapping has second highest priority
        if field.name in config_mapping:
            transformer_config_dict[field.name] = config_mapping[field.name]
        # then config
        elif field.name in cfg:
            transformer_config_dict[field.name] = cfg[field.name]
        # then model parallel config
        elif field in dataclasses.fields(model_parallel_config):
            transformer_config_dict[field.name] = getattr(
                model_parallel_config, field.name
            )

    transformer_config = TransformerConfig(**transformer_config_dict)

    # pass mcore customization configs directly to mcore
    mcore_customization_config_dict = cfg.get("mcore_customization_config", {})
    for key, value in mcore_customization_config_dict.items():
        setattr(transformer_config, key, value)

    return transformer_config


def _build_model_parallel_config(cfg: DictConfig) -> "ModelParallelConfig":
    """
    For attributes in the RLinf model config that are the same as the
    megatron core ModelParallelConfig we will use the value from the RLinf config.
    For attributes in ModelParallelConfig that are not in the RLinf model config, we add custom logic.
    """
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.training.global_vars import get_timers
    # cfg = OmegaConf.to_container(cfg, resolve=True)

    # dtype used in p2p communication
    if cfg.get("precision", None) is None:
        raise f"precision not found in {cfg}."
    torch_dtype = torch_dtype_from_precision(cfg.precision)
    params_dtype = (
        torch_dtype if torch_dtype in [torch.bfloat16, torch.float16] else torch.float32
    )
    pipeline_dtype = cfg.get("pipeline_dtype", params_dtype)
    autocast_dtype = cfg.get("autocast_dtype", params_dtype)

    timers = get_timers()
    # maps NeMo model configs to ModelParallelConfig from megatron core
    config_mapping = {
        "perform_initialization": True,  # initailize weights when constructing the module
        "fp16": torch_dtype == torch.float16,
        "bf16": torch_dtype == torch.bfloat16,
        "params_dtype": params_dtype,
        "timers": timers,
        "async_tensor_model_parallel_allreduce": False,  # Deprecated in megatron
        "pipeline_dtype": pipeline_dtype,
        "grad_scale_func": None,
        "enable_autocast": False,  # torch_dtype in [torch.bfloat16, torch.float16],
        "autocast_dtype": autocast_dtype,
        "num_microbatches_with_partial_activation_checkpoints": cfg.get(
            "num_microbatches_with_partial_activation_checkpoints", None
        ),
        "batch_p2p_sync": True,  # call torch.cuda.synchronize() after batch isend/rcv
        "use_ring_exchange_p2p": False,
        "deallocate_pipeline_outputs": False,
        "no_sync_func": None,  # set dynamically during training
        "grad_sync_func": None,  # set dynamically during training
        "param_sync_func": None,  # set dynamically during training
        "tp_comm_overlap": cfg.get("tp_comm_overlap", False),
        "tp_comm_bootstrap_backend": cfg.get("tp_comm_bootstrap_backend", "nccl"),
    }

    # instantitate ModelParallelConfig from this dict
    mp_config_dict = {}

    for field in dataclasses.fields(ModelParallelConfig):
        # model config has priority
        if field.name in cfg:
            mp_config_dict[field.name] = cfg[field.name]
        # then config_mapping
        elif field.name in config_mapping:
            mp_config_dict[field.name] = config_mapping[field.name]

    model_parallel_config = ModelParallelConfig(**mp_config_dict)

    return model_parallel_config
