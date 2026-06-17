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

import json
import importlib.util
import sys
from pathlib import Path

SCHEDULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "rlinf"
    / "models"
    / "embodiment"
    / "rlt_stage2"
    / "schedule.py"
)
SCHEDULE_SPEC = importlib.util.spec_from_file_location(
    "rlinf.models.embodiment.rlt_stage2.schedule",
    SCHEDULE_PATH,
)
assert SCHEDULE_SPEC is not None and SCHEDULE_SPEC.loader is not None
schedule = importlib.util.module_from_spec(SCHEDULE_SPEC)
sys.modules[SCHEDULE_SPEC.name] = schedule
SCHEDULE_SPEC.loader.exec_module(schedule)

PHASE_ONLINE = schedule.PHASE_ONLINE
PHASE_WARMUP = schedule.PHASE_WARMUP
PHASE_WARMUP_WAIT_ONLINE = schedule.PHASE_WARMUP_WAIT_ONLINE
phase_id = schedule.phase_id
resolve_rollout_phase = schedule.resolve_rollout_phase
resolve_training_phase = schedule.resolve_training_phase
write_status_json = schedule.write_status_json
RLTStage2TrainingScheduler = schedule.RLTStage2TrainingScheduler
SKIP_REASON_BUFFER_NOT_READY = schedule.SKIP_REASON_BUFFER_NOT_READY
SKIP_REASON_NO_PENDING_UPDATES = schedule.SKIP_REASON_NO_PENDING_UPDATES


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _scheduler_cfg() -> AttrDict:
    return AttrDict(
        algorithm=AttrDict(
            warmup_post_collect_updates=2,
            replay_buffer=AttrDict(min_buffer_size=2),
            update_epoch=1,
        ),
        actor=AttrDict(
            model=AttrDict(
                rlt_stage2=AttrDict(),
            ),
        ),
    )


def test_resolve_training_phase_tracks_buffer_and_online_gate():
    assert (
        resolve_training_phase(buffer_ready=False, ready_for_online=False)
        == PHASE_WARMUP
    )
    assert (
        resolve_training_phase(buffer_ready=True, ready_for_online=False)
        == PHASE_WARMUP_WAIT_ONLINE
    )
    assert (
        resolve_training_phase(buffer_ready=True, ready_for_online=True)
        == PHASE_ONLINE
    )


def test_resolve_rollout_phase_tracks_student_control():
    assert (
        resolve_rollout_phase(ready_for_online=False, student_control_rate=0.0)
        == PHASE_WARMUP
    )
    assert (
        resolve_rollout_phase(ready_for_online=True, student_control_rate=0.0)
        == PHASE_WARMUP_WAIT_ONLINE
    )
    assert (
        resolve_rollout_phase(ready_for_online=True, student_control_rate=0.5)
        == PHASE_ONLINE
    )


def test_write_status_json_atomically(tmp_path):
    path = tmp_path / "status" / "rlt_status.json"
    write_status_json(
        str(path),
        {
            "phase": PHASE_ONLINE,
            "phase_id": phase_id(PHASE_ONLINE),
        },
    )

    payload = json.loads(path.read_text())
    assert payload["phase"] == PHASE_ONLINE
    assert payload["phase_id"] == 2
    assert not path.with_suffix(".json.tmp").exists()


def test_rlt_training_scheduler_owns_warmup_budget_and_status_metrics():
    scheduler = RLTStage2TrainingScheduler()
    cfg = _scheduler_cfg()

    not_ready = scheduler.plan(
        cfg,
        update_step=0,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 1.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 1.0,
            "total_episodes_added": 0.0,
        },
        global_min_replay_size=1,
        global_min_demo_size=0,
    )
    assert not_ready.schedule.skip_reason == SKIP_REASON_BUFFER_NOT_READY
    assert not_ready.readiness.buffer_ready is False

    ready = scheduler.plan(
        cfg,
        update_step=0,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 2.0,
            "episodes_since_train": 1.0,
            "total_transitions_added": 2.0,
            "total_episodes_added": 1.0,
        },
        global_min_replay_size=2,
        global_min_demo_size=0,
    )
    assert ready.schedule.updates_to_run == 2
    assert scheduler.state_dict()["warmup_ready_total_transitions"] == 2

    scheduler.finish_updates(2)
    assert scheduler.pending_update_budget == 0
    metrics = scheduler.metrics(
        plan=ready,
        update_step=2,
        global_counters={
            "transitions_since_train": 2.0,
            "episodes_since_train": 1.0,
            "total_transitions_added": 2.0,
            "total_episodes_added": 1.0,
        },
        global_min_replay_size=2,
        global_min_demo_size=0,
        should_train=True,
        skip_reason=0,
        critic_updates_run=2,
        actor_updates_run=1,
    )
    assert metrics["rlt_stage2/ready_for_online"] == 1.0
    assert metrics["rlt_stage2/pending_update_budget"] == 0.0

    status_payload = scheduler.status_payload(
        plan=ready,
        rank=0,
        update_step=2,
        global_min_replay_size=2,
        should_train=True,
        skip_reason=0,
        critic_updates_run=2,
        actor_updates_run=1,
        global_total_transitions_added=2,
        global_total_episodes_added=1,
    )
    assert status_payload["phase"] == PHASE_ONLINE
    assert status_payload["ready_for_online"] is True

    no_pending = scheduler.plan(
        cfg,
        update_step=2,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 0.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 2.0,
            "total_episodes_added": 1.0,
        },
        global_min_replay_size=2,
        global_min_demo_size=0,
    )
    assert no_pending.schedule.skip_reason == SKIP_REASON_NO_PENDING_UPDATES


def test_rlt_training_scheduler_readiness_respects_dataset_batch_gate():
    scheduler = RLTStage2TrainingScheduler()
    cfg = _scheduler_cfg()

    not_ready = scheduler.plan(
        cfg,
        update_step=0,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 3.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 3.0,
            "total_episodes_added": 0.0,
        },
        global_min_replay_size=3,
        global_min_demo_size=0,
        min_replay_buffer_size=4,
    )
    assert not_ready.readiness.warmup_min_size == 4
    assert not_ready.readiness.buffer_ready is False
    assert not_ready.schedule.skip_reason == SKIP_REASON_BUFFER_NOT_READY

    ready = scheduler.plan(
        cfg,
        update_step=0,
        has_demo_buffer=False,
        global_counters={
            "transitions_since_train": 4.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 4.0,
            "total_episodes_added": 0.0,
        },
        global_min_replay_size=4,
        global_min_demo_size=0,
        min_replay_buffer_size=4,
    )
    assert ready.readiness.buffer_ready is True
    assert ready.schedule.skip_reason == 0
