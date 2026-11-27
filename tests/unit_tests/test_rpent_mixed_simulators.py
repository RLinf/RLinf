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


import asyncio
import json
from copy import deepcopy

import pytest
from omegaconf import OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader

from rlinf.agents.rpent.simulators import SimulatorBatchSampler, mixed_simulators


def mixed_config(names=("libero", "robotwin", "robocasa", "robodojo")):
    return OmegaConf.create(
        {
            "runner": {"task_type": "reasoning", "synchronous": True},
            "algorithm": {"adv_type": "grpo_dynamic", "group_size": 8},
            "rollout": {"rollout_backend": "sglang"},
            "data": {"rollout_batch_size": 32, "sampling_batch_size": 40},
            "env": {
                "rpent": {
                    name: {
                        "endpoint": f"http://localhost:{index + 9000}",
                        "train_data_paths": f"/tasks/{name}.jsonl",
                    }
                    for index, name in enumerate(names)
                },
            },
        }
    )


def test_mixed_quotas_count_full_groups():
    cfg = mixed_config()
    assert [
        (sim.rollout_groups, sim.sampling_groups)
        for sim in mixed_simulators(cfg).values()
    ] == [
        (8, 10),
        (8, 10),
        (8, 10),
        (8, 10),
    ]
    for entry, effective, candidates in zip(
        cfg.env.rpent.values(), (96, 64, 64, 32), (112, 80, 80, 48), strict=True
    ):
        entry.rollout_episodes = effective
        entry.sampling_episodes = candidates
    assert [
        (sim.rollout_groups, sim.sampling_groups)
        for sim in mixed_simulators(cfg).values()
    ] == [
        (12, 14),
        (8, 10),
        (8, 10),
        (4, 6),
    ]
    cfg.env.rpent.libero.rollout_episodes = 97
    with pytest.raises(ValueError, match="multiple of group_size"):
        mixed_simulators(cfg)
    cfg.env.rpent.libero.rollout_episodes = 88
    with pytest.raises(ValueError, match="must total 256"):
        mixed_simulators(cfg)
    cfg.env.rpent.libero.rollout_episodes = 96
    cfg.env.rpent.libero.sampling_episodes = 88
    cfg.env.rpent.robotwin.sampling_episodes = 104
    with pytest.raises(ValueError, match="at least 12 candidate groups"):
        mixed_simulators(cfg)


def test_uneven_default_requires_explicit_whole_groups():
    with pytest.raises(ValueError, match="cannot be split equally"):
        mixed_simulators(mixed_config(("libero", "robotwin", "robocasa")))


@pytest.mark.parametrize("num_workers", [0, 2])
def test_mixed_sampler_restores_next_batch_and_epoch(num_workers):
    def loader():
        return StatefulDataLoader(
            list(range(47)),
            batch_sampler=SimulatorBatchSampler(
                [15, 12, 20], [3, 2, 4], shuffle=True, seed=13
            ),
            num_workers=num_workers,
        )

    source = loader()
    iterator = iter(source)
    first = next(iterator).tolist()
    assert [0 if i < 15 else 1 if i < 27 else 2 for i in first] == [
        0,
        1,
        2,
        0,
        1,
        2,
        0,
        2,
        2,
    ]
    state = deepcopy(source.state_dict())
    remaining = [batch.tolist() for batch in iterator]
    next_epoch = [batch.tolist() for batch in source]
    restored = loader()
    restored.load_state_dict(state)
    assert [batch.tolist() for batch in restored] == remaining
    assert [batch.tolist() for batch in restored] == next_epoch


def test_mixed_sampler_restores_a_checkpoint_at_the_epoch_boundary():
    def sampler():
        return SimulatorBatchSampler([10, 12], [5, 6], shuffle=True, seed=13)

    source = sampler()
    iterator = iter(source)
    for _ in range(len(source)):
        next(iterator)
    restored = sampler()
    restored.load_state_dict(source.state_dict())
    assert list(restored) == list(source)


def test_candidate_cleanup_routes_owners_to_their_backends(tmp_path):
    from rlinf.agents.rpent.rpent_agent_loop import RPentAgentLoopWorker

    worker = object.__new__(RPentAgentLoopWorker)
    worker._candidate_cohort = {"id": "cohort"}
    libero = tmp_path / "libero-owner"
    dojo = tmp_path / "dojo-owner"
    for directory in (libero, dojo):
        directory.mkdir()
    worker._candidate_attempts = {
        ("cohort", 0): [("http://libero", libero)],
        ("cohort", 1): [("http://dojo", dojo)],
    }
    calls = {}

    async def release(endpoint, owners):
        calls[endpoint] = owners
        return {"owner_ids": owners, "cleanup_complete": True}

    worker._release_remote_candidates = release
    records = [
        {"candidate_id": 0, "status": "complete"},
        {"candidate_id": 1, "status": "cancelled"},
    ]
    asyncio.run(worker.finalize_rollout_candidates(records))
    assert calls == {"http://libero": [libero.name], "http://dojo": [dojo.name]}
    for directory, record in zip((libero, dojo), records, strict=True):
        receipt = json.loads((directory / "sampling-selection.json").read_text())
        assert receipt["cleanup_complete"]
        assert receipt["status"] == record["status"]
    assert not worker._candidate_attempts


def test_full_cohort_waits_for_native_workers_and_checks_release(monkeypatch):
    from rlinf.agents.rpent import cohort

    cfg = mixed_config(("libero", "robotwin"))
    cfg.tools = {"rpent": {"full_cohort_ready_timeout_s": 5}}
    for entry in cfg.env.rpent.values():
        entry.group_router = True
    simulators = mixed_simulators(cfg)
    ready = {"http://localhost:9000": 160, "http://localhost:9001": 0}

    def backend(endpoint, method, *args, **kwargs):
        if method == "backend.group_status":
            return {"group_size": 8, "capacity": 20, "admitted": 0, "waiting": 0}
        assert args == ("step-0-test",)
        return {"reserved": 160 if ready[endpoint] else 0, "ready": ready[endpoint]}

    monkeypatch.setattr(cohort, "rpc", backend)
    cohort.validate_routers(simulators, 8, strict=True)
    sleeps = []

    def advance(seconds):
        sleeps.append(seconds)
        ready["http://localhost:9001"] = 160

    monkeypatch.setattr(cohort.time, "sleep", advance)
    observed = cohort.wait_until_ready("step-0-test", simulators, 8, 5)
    assert sleeps
    assert all(value["ready"] == 160 for value in observed.values())
    with pytest.raises(RuntimeError, match="still owns simulator slots"):
        cohort.verify_released("step-0-test", simulators, {})
    ready.update(dict.fromkeys(ready, 0))
    assert all(
        value["ready"] == 0
        for value in cohort.verify_released("step-0-test", simulators, {}).values()
    )


def test_full_cohort_preflight_rejects_insufficient_router_capacity(monkeypatch):
    from rlinf.agents.rpent import cohort

    cfg = mixed_config(("libero", "robotwin"))
    for entry in cfg.env.rpent.values():
        entry.group_router = True
    monkeypatch.setattr(cohort, "rpc", lambda *args: {"group_size": 8, "capacity": 19})
    with pytest.raises(ValueError, match="needs 20 concurrent groups"):
        cohort.validate_routers(mixed_simulators(cfg), 8, strict=True)
