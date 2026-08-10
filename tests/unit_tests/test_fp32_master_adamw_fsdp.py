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

import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW
from rlinf.utils.utils import warmup_optimizer_state

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FSDP_GPU_TESTS") != "1",
    reason="Set RUN_FSDP_GPU_TESTS=1 and launch with torchrun.",
)


def _build_model_and_optimizer(local_rank):
    torch.manual_seed(1234)
    model = torch.nn.Linear(4, 4, bias=False, device=local_rank, dtype=torch.bfloat16)
    model = FSDP(model, device_id=local_rank, sync_module_states=True)
    optimizer = FP32MasterAdamW(
        model.parameters(),
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    warmup_optimizer_state(optimizer)
    return model, optimizer


def _train_step(model, optimizer, local_rank):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(16, device=local_rank, dtype=torch.bfloat16).view(4, 4)
    model(inputs).float().sum().backward()
    optimizer.step()


def test_fp32_master_adamw_restores_with_fsdp_distributed_state_dict():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    try:
        model, optimizer = _build_model_and_optimizer(local_rank)
        _train_step(model, optimizer, local_rank)

        options = StateDictOptions(full_state_dict=False, cpu_offload=True)
        model_state, optimizer_state = get_state_dict(
            model=model, optimizers=optimizer, options=options
        )

        restored_model, restored_optimizer = _build_model_and_optimizer(local_rank)
        set_state_dict(
            model=restored_model,
            optimizers=restored_optimizer,
            model_state_dict=model_state,
            optim_state_dict=optimizer_state,
            options=options,
        )

        original_state = next(iter(optimizer.state.values()))
        restored_state = next(iter(restored_optimizer.state.values()))
        for key in ("fp32_master_param", "exp_avg", "exp_avg_sq"):
            assert restored_state[key].dtype == torch.float32
            assert torch.equal(restored_state[key], original_state[key])

        _train_step(model, optimizer, local_rank)
        _train_step(restored_model, restored_optimizer, local_rank)
        assert torch.equal(next(model.parameters()), next(restored_model.parameters()))
        assert torch.equal(
            optimizer.state[next(model.parameters())]["fp32_master_param"],
            restored_optimizer.state[next(restored_model.parameters())][
                "fp32_master_param"
            ],
        )
    finally:
        dist.destroy_process_group()
