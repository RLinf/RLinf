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

import os
import random
import time
import warnings
from datetime import timedelta

import numpy as np
import torch
import torch.distributed
from megatron.core import mpu, tensor_parallel
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.rerun_state_machine import (
    RerunDiagnostic,
    RerunErrorInjector,
    RerunMode,
    initialize_rerun_state_machine,
)
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.legacy import fused_kernels
from megatron.training.global_vars import _set_timers, set_args
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from omegaconf.omegaconf import OmegaConf

from rlinf.config import torch_dtype_from_precision


def extract_selected_fields(cfg: DictConfig) -> DictConfig:
    result = {}

    keys_to_extract = [
        ("model", ""),
        ("optim", ""),
        ("lr_sched", ""),
        ("megatron", ""),
    ]
    for path, prefix in keys_to_extract:
        sub_cfg = OmegaConf.select(cfg, path)
        if sub_cfg is not None:
            sub_cfg = OmegaConf.to_container(sub_cfg, resolve=True)
            for k, v in sub_cfg.items():
                result[f"{prefix}{k}"] = v

    return OmegaConf.create(result)


def set_megatron_args(cfg):
    args = extract_selected_fields(cfg)

    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0

    args.use_mp_args_from_checkpoint_args = False
    args.fp16 = torch_dtype_from_precision(cfg.model.precision) == torch.float16
    args.bf16 = torch_dtype_from_precision(cfg.model.precision) == torch.bfloat16
    params_dtype = "${torch.dtype:float32}"
    if cfg.optim.fp16:
        params_dtype = "${torch.dtype:half}"
    elif cfg.optim.bf16:
        params_dtype = "${torch.dtype:bfloat16}"
    args.params_dtype = params_dtype

    args.vocab_file = None

    args.iteration = 0

    set_args(args)

    return args


def initialize_megatron(cfg: DictConfig):
    args = set_megatron_args(cfg)
    _set_timers(args)

    # init rerun state
    def state_save_func():
        return {
            "rng_tracker_states": tensor_parallel.get_cuda_rng_tracker().get_states()
        }

    def state_restore_func(state_dict):
        if state_dict["rng_tracker_states"]:
            tensor_parallel.get_cuda_rng_tracker().set_states(
                state_dict["rng_tracker_states"]
            )

    initialize_rerun_state_machine(
        state_save_func=state_save_func,
        state_restore_func=state_restore_func,
        mode=RerunMode(cfg.megatron.rerun_mode),
        error_injector=RerunErrorInjector(
            error_injection_rate=cfg.megatron.error_injection_rate,
            error_injection_type=RerunDiagnostic(cfg.megatron.error_injection_type),
        ),
    )

    # Megatron's MPU is the master. Complete initialization right away.
    # Pytorch distributed.
    _initialize_distributed(cfg)

    tensor_parallel.random.initialize_rng_tracker(use_te_rng_tracker=False)

    # # Random seeds for reproducibility.
    _set_random_seed(cfg.seed, cfg.megatron.data_parallel_random_init)

    init_num_microbatches_calculator(
        rank=torch.distributed.get_rank(),
        global_batch_size=cfg.global_batch_size,
        micro_batch_size=cfg.micro_batch_size,
        data_parallel_size=mpu.get_data_parallel_world_size(),
        rampup_batch_size=None,
    )

    with open_dict(cfg):
        cfg.rank = torch.distributed.get_rank()

    # Compile dependencies.
    _compile_dependencies(cfg)

    if cfg.megatron.tp_comm_overlap_cfg is not None:
        _initialize_tp_communicators(cfg)


def _set_random_seed(seed_, data_parallel_random_init=False):
    """Set random seed for reproducability."""
    if seed_ is not None and seed_ > 0:
        # Ensure that different pipeline MP stages get different seeds.
        seed = seed_ + (100 * mpu.get_pipeline_model_parallel_rank())
        # Ensure different data parallel ranks get different seeds
        if data_parallel_random_init:
            seed = seed + (10 * mpu.get_data_parallel_rank())
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.device_count() > 0:
            tensor_parallel.model_parallel_cuda_manual_seed(seed)
    else:
        raise ValueError("Seed ({}) should be a positive integer.".format(seed_))


def _compile_dependencies(cfg: DictConfig):
    # ==================
    # Load fused kernels
    # ==================

    # Custom kernel constraints check.
    seq_len = cfg.model.seq_length
    attn_batch_size = (
        cfg.model.num_attention_heads / cfg.model.tensor_model_parallel_size
    ) * cfg.micro_batch_size
    # Constraints on sequence length and attn_batch_size to enable warp based
    # optimization and upper triangular optimization (for causal mask)
    custom_kernel_constraint = (
        seq_len > 16
        and seq_len <= 16384
        and seq_len % 4 == 0
        and attn_batch_size % 4 == 0
    )
    # Print a warning.
    if not (
        (cfg.model.precision == "fp16" or cfg.model.precision == "bf16")
        and custom_kernel_constraint
        and cfg.model.get("masked_softmax_fusion", False)
    ):
        if cfg.rank == 0:
            print(
                "WARNING: constraints for invoking optimized"
                " fused softmax kernel are not met. We default"
                " back to unfused kernel invocations.",
                flush=True,
            )

    # Always build on rank zero first.
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print("> compiling and loading fused kernels ...", flush=True)
        fused_kernels.load(cfg)
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        fused_kernels.load(cfg)
    # Simple barrier to make sure all ranks have passed the
    # compilation phase successfully before moving on to the
    # rest of the program. We think this might ensure that
    # the lock is released.
    torch.distributed.barrier()
    if torch.distributed.get_rank() == 0:
        print(
            ">>> done with compiling and loading fused kernels. "
            "Compilation time: {:.3f} seconds".format(time.time() - start_time),
            flush=True,
        )


def _initialize_tp_communicators(cfg: DictConfig):
    """initializing the communicators with user buffers for high-performance tensor-model-parallel
    communication overlap"""

    try:
        import yaml
        from transformer_engine.pytorch import module as te_module

    except ImportError:
        raise RuntimeError(
            "Tensor Parallel Communication/GEMM Overlap optimization needs 'yaml' and "
            "'transformer_engine' packages"
        )

    if cfg.megatron.tp_comm_overlap_cfg is not None:
        with open(cfg.megatron.tp_comm_overlap_cfg, "r") as stream:
            ub_cfgs = yaml.safe_load(stream)
    else:
        ub_cfgs = {}

    if getattr(cfg, "decoder_tp_comm_overlap", False):
        input_shape = [
            (cfg.model.decoder_seq_length * cfg.micro_batch_size)
            // cfg.model.context_parallel_size,
            cfg.model.hidden_size,
        ]
    else:
        input_shape = [
            (cfg.model.seq_length * cfg.micro_batch_size)
            // cfg.model.context_parallel_size,
            cfg.model.hidden_size,
        ]

    if is_te_min_version("1.9.0"):
        # The process group with the target bootstrap backend is created in Transformer Engine.
        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=cfg.model.tensor_model_parallel_size,
            use_fp8=cfg.megatron.get("fp8", False),
            dtype=torch_dtype_from_precision(cfg.model.precision),
            ub_cfgs=ub_cfgs,
            bootstrap_backend=cfg.megatron.tp_comm_bootstrap_backend,
        )
    else:
        if cfg.megatron.tp_comm_bootstrap_backend != "mpi":
            warnings.warn(
                f"Transformer Engine v{get_te_version()} supports only MPI bootstrap backend."
            )
        # Create a MPI process group to help with TP communication overlap bootstrap.
        torch.distributed.new_group(backend="mpi")

        te_module.base.initialize_ub(
            shape=input_shape,
            tp_size=cfg.model.tensor_model_parallel_size,
            use_fp8=cfg.megatron.get("fp8", False),
            ub_cfgs=ub_cfgs,
        )


def _initialize_distributed(cfg: DictConfig):
    node_rank = int(os.getenv("NODE_RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    global_rank = int(os.getenv("RANK", str(node_rank)))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "29500")

    """Initialize torch.distributed and core model parallel."""
    device_count = torch.cuda.device_count()

    if global_rank == 0:
        print(
            f"> Initializing torch.distributed with:\n"
            f"  MASTER_ADDR={master_addr}, MASTER_PORT={master_port}\n"
            f"  RANK={global_rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}",
            flush=True,
        )

    if not torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend=cfg.megatron.distributed_backend,
            rank=global_rank,
            world_size=world_size,
            timeout=timedelta(minutes=cfg.megatron.distributed_timeout_minutes),
        )
    else:
        if global_rank == 0:
            print("torch.distributed already initialized, skipping init.", flush=True)
        local_rank = torch.distributed.get_rank()

    # Set the tensor model-parallel, pipeline model-parallel, and
    # data-parallel communicators.
    if device_count > 0:
        if mpu.model_parallel_is_initialized():
            print("model parallel is already initialized")
        else:
            mpu.initialize_model_parallel(
                cfg.model.tensor_model_parallel_size,
                cfg.model.pipeline_model_parallel_size,
                cfg.model.virtual_pipeline_model_parallel_size,
                cfg.model.pipeline_model_parallel_split_rank,
                context_parallel_size=cfg.model.context_parallel_size,
                expert_model_parallel_size=cfg.model.expert_model_parallel_size,
                expert_tensor_parallel_size=cfg.model.expert_tensor_parallel_size,
                distributed_timeout_minutes=cfg.megatron.distributed_timeout_minutes,
                nccl_communicator_config_path=cfg.megatron.nccl_communicator_config_path,
                order="tp-cp-ep-dp-pp"
                if not cfg.megatron.use_tp_pp_dp_mapping
                else "tp-pp-dp",
            )
            if local_rank == 0:
                print(
                    f"> initialized tensor model parallel with size "
                    f"{mpu.get_tensor_model_parallel_world_size()}"
                )
                print(
                    f"> initialized pipeline model parallel with size "
                    f"{mpu.get_pipeline_model_parallel_world_size()}"
                )
