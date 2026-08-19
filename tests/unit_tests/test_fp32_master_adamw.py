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

import pytest
import torch

from rlinf.hybrid_engines.fsdp.optim import (
    FP32MasterAdamW,
    build_adamw,
    validate_fp32_master_adamw_config,
)


def _run_constant_gradient_steps(optimizer, parameter, count):
    for _ in range(count):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_fp32_master_adamw_accumulates_sub_bf16_updates():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW(
        [parameter],
        lr=1e-4,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
    )

    _run_constant_gradient_steps(optimizer, parameter, count=40)

    state = optimizer.state[parameter]
    assert parameter.dtype == torch.bfloat16
    assert parameter.item() < 1.0
    assert state["fp32_master_param"].dtype == torch.float32
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32
    assert torch.allclose(
        state["fp32_master_param"], torch.tensor([0.9960]), atol=1e-6, rtol=0
    )


def test_fp32_master_adamw_state_dict_round_trip_preserves_master_weight():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW(
        [parameter],
        lr=1e-4,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
    )
    _run_constant_gradient_steps(optimizer, parameter, count=20)

    restored_parameter = torch.nn.Parameter(parameter.detach().clone())
    restored_optimizer = FP32MasterAdamW(
        [restored_parameter],
        lr=1e-4,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
    )
    restored_optimizer.load_state_dict(optimizer.state_dict())

    original_master = optimizer.state[parameter]["fp32_master_param"]
    restored_master = restored_optimizer.state[restored_parameter]["fp32_master_param"]
    assert torch.equal(restored_master, original_master)

    _run_constant_gradient_steps(optimizer, parameter, count=20)
    _run_constant_gradient_steps(restored_optimizer, restored_parameter, count=20)

    assert torch.equal(restored_parameter, parameter)
    assert torch.equal(
        restored_optimizer.state[restored_parameter]["fp32_master_param"],
        optimizer.state[parameter]["fp32_master_param"],
    )


def test_fp32_master_adamw_rejects_incompatible_state_topology():
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    restored = FP32MasterAdamW(
        [
            torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)),
            torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16)),
        ]
    )

    with pytest.raises(ValueError, match="parameter topology"):
        restored.load_state_dict(optimizer.state_dict())


def test_build_adamw_selects_fp32_master_implementation_when_enabled():
    optimizer = build_adamw(
        [{"params": list(torch.nn.Linear(2, 2).parameters()), "lr": 1e-4}],
        eps=1e-8,
        weight_decay=0.01,
        use_fp32_master_params=True,
    )

    assert isinstance(optimizer, FP32MasterAdamW)


def test_build_adamw_preserves_native_adamw_by_default():
    optimizer = build_adamw(
        torch.nn.Linear(2, 2).parameters(),
        eps=1e-8,
        weight_decay=0.01,
    )

    assert type(optimizer) is torch.optim.AdamW


def test_fp32_master_adamw_uses_scheduler_updated_learning_rate():
    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
    optimizer = FP32MasterAdamW(
        [parameter],
        lr=1e-3,
        betas=(0.0, 0.0),
        eps=1e-8,
        weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    _run_constant_gradient_steps(optimizer, parameter, count=1)
    first_master = optimizer.state[parameter]["fp32_master_param"].clone()
    scheduler.step()
    _run_constant_gradient_steps(optimizer, parameter, count=1)
    second_master = optimizer.state[parameter]["fp32_master_param"].clone()

    assert torch.allclose(first_master, torch.tensor([0.999]), atol=1e-7, rtol=0)
    assert torch.allclose(second_master, torch.tensor([0.9989]), atol=1e-7, rtol=0)


def test_fp32_master_adamw_matches_torch_adamw_for_fp32_parameters():
    actual = torch.nn.Parameter(torch.tensor([0.5, -0.25], dtype=torch.float32))
    expected = torch.nn.Parameter(actual.detach().clone())
    actual_optimizer = FP32MasterAdamW(
        [actual], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )
    expected_optimizer = torch.optim.AdamW(
        [expected], lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01
    )

    for grad in (
        torch.tensor([0.1, -0.2]),
        torch.tensor([-0.3, 0.4]),
        torch.tensor([0.05, 0.25]),
    ):
        actual.grad = grad.clone()
        expected.grad = grad.clone()
        actual_optimizer.step()
        expected_optimizer.step()

    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert "fp32_master_param" not in actual_optimizer.state[actual]


@pytest.mark.parametrize("sharding_strategy", ["no_shard", "full_shard"])
def test_fp32_master_config_accepts_fsdp1_lora(sharding_strategy):
    validate_fp32_master_adamw_config(
        strategy="fsdp",
        sharding_strategy=sharding_strategy,
        is_lora=True,
    )


@pytest.mark.parametrize(
    ("strategy", "sharding_strategy", "is_lora"),
    [
        ("fsdp2", "no_shard", True),
        ("fsdp2", "full_shard", True),
        ("fsdp", "shard_grad_op", True),
        ("fsdp", "no_shard", False),
        ("fsdp", "full_shard", False),
    ],
)
def test_fp32_master_config_rejects_unsupported_combinations(
    strategy, sharding_strategy, is_lora
):
    with pytest.raises(ValueError, match="FSDP1.*LoRA"):
        validate_fp32_master_adamw_config(
            strategy=strategy,
            sharding_strategy=sharding_strategy,
            is_lora=is_lora,
        )
