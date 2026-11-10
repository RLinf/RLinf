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

"""Numerical contracts of the multi-agent Megatron value and policy losses."""

import copy
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.utils.utils import masked_mean


class TokenModel(torch.nn.Module):
    pre_process = True
    post_process = True

    def __init__(self, role="critic"):
        super().__init__()
        self.role = role
        self.weight = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, input_ids, attention_mask, position_ids, packed_seq_params):
        output = (input_ids.float() * self.weight).unsqueeze(-1)
        if self.role == "actor":
            output = output * torch.arange(8, dtype=torch.float32)
        return output


@pytest.fixture
def megatron_cpu(monkeypatch, tmp_path):
    pytest.importorskip("megatron.training")
    from megatron.core import parallel_state

    from rlinf.scheduler import Worker
    from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
    from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic

    monkeypatch.setattr(Worker, "torch_device_type", "cpu")
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: "cpu")
    torch.distributed.init_process_group(
        "gloo", init_method=(tmp_path / "megatron-group").as_uri(), rank=0, world_size=1
    )
    for name in (
        "_TENSOR_MODEL_PARALLEL_GROUP",
        "_CONTEXT_PARALLEL_GROUP",
        "_DATA_PARALLEL_GROUP",
    ):
        monkeypatch.setattr(parallel_state, name, torch.distributed.group.WORLD)
    monkeypatch.setattr(parallel_state, "get_context_parallel_rank", lambda: 0)

    def make_worker(role, valid_token_scale=False):
        worker = object.__new__(
            MAMegatronActor if role == "actor" else MAMegatronCritic
        )
        worker.cfg = OmegaConf.create(
            {
                "runner": {"task_type": "reasoning"},
                "algorithm": {
                    "loss_type": "actor",
                    "use_valid_token_scale": valid_token_scale,
                    "sampling_params": {"temperature": 1.0},
                    "early_stop_imp_ratio": None,
                },
            }
        )
        worker.transformer_config = SimpleNamespace(sequence_parallel=False)
        worker.return_loss = True
        worker.loss_agg_func = masked_mean
        worker.calculate_entropy = False
        worker.kl_beta = 0
        worker.clip_ratio_c = None
        worker.clip_ratio_low = worker.clip_ratio_high = 0.2
        worker.value_clip = 0.5
        worker.num_microbatches = 1
        return worker

    try:
        yield make_worker, parallel_state
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("role", ["actor", "critic"])
def test_valid_token_scaling_preserves_unsplit_gradient(
    megatron_cpu, monkeypatch, role
):
    make_worker, parallel_state = megatron_cpu
    worker = make_worker(role)
    input_ids = torch.arange(64).reshape(16, 4) % 7 + 1
    positions = torch.arange(4).expand_as(input_ids).clone()
    mask = (positions > 0) & (positions <= (torch.arange(16) % 3 + 1)[:, None])
    mask[[0, 2, 3, 15]] = False
    batch = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(mask),
        "position_ids": positions,
        "response_mask": mask,
        "loss_scales": mask.float(),
        "advantages": input_ids.float() / 7 - 0.5,
        "recomputed_logprobs": torch.zeros_like(mask, dtype=torch.float32),
        "values": torch.zeros_like(mask, dtype=torch.float32),
        "returns": torch.full_like(mask, 0.7, dtype=torch.float32),
    }
    forward = worker.get_forward_step_func()
    expected_model = TokenModel(role)
    output, loss_func = forward(iter([copy.deepcopy(batch)]), expected_model)
    expected_loss, _ = loss_func(output)
    expected_loss.backward()
    assert expected_model.weight.grad.abs() > 0

    worker.cfg.algorithm.use_valid_token_scale = True
    worker._setup_valid_token_scale(batch)
    dp_size, micro_batch_size = 4, 2
    worker.num_microbatches = len(input_ids) // dp_size // micro_batch_size
    monkeypatch.setattr(parallel_state, "get_data_parallel_world_size", lambda: dp_size)
    model = TokenModel(role)
    for rank_rows in torch.chunk(torch.arange(len(input_ids)), dp_size):
        for rows in rank_rows.split(micro_batch_size):
            microbatch = {key: value[rows] for key, value in batch.items()}
            output, loss_func = forward(iter([microbatch]), model)
            loss, _ = loss_func(output)
            assert loss.requires_grad and torch.isfinite(loss)
            # Megatron averages microbatch losses and then DP gradients.
            (loss / worker.num_microbatches / dp_size).backward()
    torch.testing.assert_close(
        model.weight.grad, expected_model.weight.grad, rtol=1e-5, atol=1e-7
    )


def test_critic_packing_preserves_values_loss_and_optimizer_update(megatron_cpu):
    make_worker, _ = megatron_cpu
    worker = make_worker("critic", valid_token_scale=True)
    result = DynamicRolloutResult(
        num_sequence=3,
        group_size=2,
        idx_to_traj=[0, 0, 1],
        input_ids=[[10, 11, 12], [10, 11, 12, 50, 51, 13], [20, 21, 22]],
        prompt_lengths=[1, 5, 1],
        response_lengths=[2, 1, 2],
        is_end=[False, True, True],
        rewards=[1.0, 1.0, 0.0],
        values=torch.tensor(
            [[0, 0.2, 0.3, 0, 0, 0], [0, 0, 0, 0, 0, 0.4], [0, 0.1, 0.2, 0, 0, 0]]
        ),
        returns=torch.tensor(
            [[0, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0]],
            dtype=torch.float32,
        ),
    )
    batch = result.to_actor_batch(6, 0)
    frozen = copy.deepcopy(batch)
    outcomes = []
    for pack in (False, True):
        minibatch = copy.deepcopy(batch)
        if pack:
            minibatch = DynamicRolloutResult.pack_traj_batch({}, minibatch)
        assert minibatch["input_ids"].shape[0] == (2 if pack else 3)
        mask = minibatch["response_mask"]
        model = TokenModel()
        worker._setup_valid_token_scale(minibatch)
        forward = worker.get_forward_step_func()
        worker.return_loss = False
        values, _ = forward(iter([minibatch]), model)
        expected_values = torch.nn.functional.pad(
            minibatch["input_ids"].float()[:, :-1] * model.weight, (1, 0)
        )
        torch.testing.assert_close(values, expected_values * mask)

        worker.return_loss = True
        output, loss_func = forward(iter([minibatch]), model)
        loss, _ = loss_func(output)
        loss.backward()
        assert torch.isfinite(model.weight.grad) and model.weight.grad.abs() > 0
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        outcomes.append((values[mask].detach(), loss.detach(), model.weight.detach()))
    for unpacked, packed in zip(*outcomes, strict=True):
        torch.testing.assert_close(unpacked, packed)
    for key in ("values", "returns"):
        torch.testing.assert_close(batch[key], frozen[key])
