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

import functools
import os

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_state_dict,
    set_state_dict,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW
from rlinf.utils.utils import warmup_optimizer_state

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_FSDP_DISTRIBUTED_TESTS") != "1"
    and os.environ.get("RUN_FSDP_GPU_TESTS") != "1",
    reason="Set RUN_FSDP_DISTRIBUTED_TESTS=1 and launch with torchrun.",
)


@pytest.fixture(scope="module", autouse=True)
def _distributed_process_group():
    use_cuda = torch.cuda.is_available()
    dist.init_process_group(backend="nccl" if use_cuda else "gloo")
    if use_cuda:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    try:
        yield
    finally:
        dist.destroy_process_group()


class _LoRALikeLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(4, 4, bias=False)
        self.lora_a = torch.nn.Linear(4, 2, bias=False)
        self.lora_b = torch.nn.Linear(2, 4, bias=False)
        self.base.requires_grad_(False)

    def forward(self, inputs):
        return self.base(inputs) + self.lora_b(self.lora_a(inputs))


def _trainable_leaf_policy(module):
    return bool(
        not list(module.named_children())
        and getattr(module, "weight", None) is not None
        and module.weight.requires_grad
    )


def _build_model_and_optimizer(device, *, use_orig_params):
    torch.manual_seed(1234)
    model = _LoRALikeLinear().to(device=device, dtype=torch.bfloat16)
    auto_wrap_policy = functools.partial(
        lambda_auto_wrap_policy, lambda_fn=_trainable_leaf_policy
    )
    model = FullyShardedDataParallel(
        model,
        auto_wrap_policy=auto_wrap_policy,
        device_id=device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        sync_module_states=device.type == "cuda",
        use_orig_params=use_orig_params,
    )
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    assert trainable_params
    optimizer = FP32MasterAdamW(
        trainable_params,
        lr=1e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    warmup_optimizer_state(optimizer)
    return model, optimizer


def _train_step(model, optimizer, device):
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.arange(16, device=device, dtype=torch.bfloat16).view(4, 4)
    model(inputs).float().sum().backward()
    optimizer.step()


@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fp32_master_adamw_restores_with_fsdp_distributed_state_dict(
    use_orig_params,
):
    use_cuda = torch.cuda.is_available()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank) if use_cuda else torch.device("cpu")
    assert dist.get_world_size() >= 2, "FULL_SHARD test requires at least 2 ranks"
    model, optimizer = _build_model_and_optimizer(
        device, use_orig_params=use_orig_params
    )
    _train_step(model, optimizer, device)

    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state, optimizer_state = get_state_dict(
        model=model, optimizers=optimizer, options=options
    )

    restored_model, restored_optimizer = _build_model_and_optimizer(
        device, use_orig_params=use_orig_params
    )
    set_state_dict(
        model=restored_model,
        optimizers=restored_optimizer,
        model_state_dict=model_state,
        optim_state_dict=optimizer_state,
        options=options,
    )

    assert len(optimizer.state) == len(restored_optimizer.state)
    for original_state, restored_state in zip(
        optimizer.state.values(), restored_optimizer.state.values(), strict=True
    ):
        for key in ("fp32_master_param", "exp_avg", "exp_avg_sq"):
            assert restored_state[key].dtype == torch.float32
            assert torch.equal(restored_state[key], original_state[key])

    _train_step(model, optimizer, device)
    _train_step(restored_model, restored_optimizer, device)
    for original_param, restored_param in zip(
        model.parameters(), restored_model.parameters(), strict=True
    ):
        assert torch.equal(original_param, restored_param)
    for original_state, restored_state in zip(
        optimizer.state.values(), restored_optimizer.state.values(), strict=True
    ):
        for key in ("fp32_master_param", "exp_avg", "exp_avg_sq"):
            assert torch.equal(restored_state[key], original_state[key])
