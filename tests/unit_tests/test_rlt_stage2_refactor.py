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

from __future__ import annotations

import pytest

try:
    import torch
except ModuleNotFoundError:
    torch = None

try:
    import numpy as _numpy  # noqa: F401
except ModuleNotFoundError:
    _HAS_NUMPY = False
else:
    _HAS_NUMPY = True

pytestmark = pytest.mark.skipif(
    torch is None or not _HAS_NUMPY,
    reason="RLT synthetic refactor tests require torch and numpy.",
)

if torch is not None and _HAS_NUMPY:
    from torch.utils.data import DataLoader

    from rlinf.data.embodied_buffer_dataset import (
        ReplayBufferDataset,
        replay_buffer_collate_fn,
    )
    from rlinf.data.embodied_io_struct import Trajectory
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer
    from rlinf.models.embodiment.rlt_stage2.rollout import (
        COLLECTION_PHASE_ONLINE,
        COLLECTION_PHASE_WARMUP,
        RLTStage2RolloutAdapter,
        TransitionSource,
    )
    from rlinf.models.embodiment.rlt_stage2.trajectory_adapter import (
        RLTStage2TrajectoryReplayAdapter,
    )
    from rlinf.workers.actor.fsdp_rlt_stage2_policy_worker import (
        RLTStage2FSDPPolicyWorker,
    )
    from rlinf.workers.env.policy_info_adapter import build_policy_info_adapter
    from toolkits.rlt import inspect_rlt_replay


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


def _cfg(
    *,
    env_type: str = "maniskill",
    intervention_enable: bool = True,
    intervention_mode: str = "local_correction",
    warmup_updates: int = 5,
    replay_subsample_stride: int = 0,
) -> AttrDict:
    return AttrDict(
        algorithm=AttrDict(
            loss_type="rlt_td3",
            warmup_post_collect_updates=warmup_updates,
            intervention=AttrDict(
                enable=intervention_enable,
                mode=intervention_mode,
            ),
        ),
        actor=AttrDict(
            model=AttrDict(
                model_type="rlt_stage2",
                num_action_chunks=2,
                action_dim=2,
                rlt_stage2=AttrDict(
                    replay_subsample_stride=replay_subsample_stride,
                    replay_allow_terminal_partial=True,
                    replay_feature_batch_size=32,
                ),
            ),
        ),
        env=AttrDict(
            train=AttrDict(
                env_type=env_type,
                auto_reset=True,
            ),
            eval=AttrDict(
                env_type=env_type,
                auto_reset=True,
            ),
        ),
        rollout=AttrDict(
            expert_model=AttrDict(act_as_vla_reference=False),
        ),
    )


def _synthetic_rollout_trajectory() -> Trajectory:
    actions = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0]],
            [[5.0, 6.0, 7.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros((3, 1, 2), dtype=torch.bool)
    dones[2, 0, 1] = True

    return Trajectory(
        max_episode_length=2,
        model_weights_id="synthetic",
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        forward_inputs={
            "x": torch.tensor(
                [
                    [[0.0, 0.1, 0.2]],
                    [[1.0, 1.1, 1.2]],
                    [[2.0, 2.1, 2.2]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.1, 0.1, 0.1, 0.1]],
                    [[0.2, 0.2, 0.2, 0.2]],
                ],
                dtype=torch.float32,
            ),
            "intervention_flags": torch.tensor(
                [
                    [[False, False]],
                    [[True, False]],
                ],
                dtype=torch.bool,
            ),
            "source_chunk": torch.tensor(
                [
                    [[int(TransitionSource.RL), int(TransitionSource.RL)]],
                    [[int(TransitionSource.HUMAN), int(TransitionSource.MIXED)]],
                ],
                dtype=torch.uint8,
            ),
            "collection_phase_id": torch.tensor(
                [
                    [[COLLECTION_PHASE_WARMUP]],
                    [[COLLECTION_PHASE_ONLINE]],
                ],
                dtype=torch.uint8,
            ),
            "record_transition": torch.ones((2, 1, 1), dtype=torch.bool),
        },
    )


def _synthetic_stride_rollout_trajectory() -> Trajectory:
    actions = torch.tensor(
        [
            [[1.0, 1.1, 2.0, 2.1]],
            [[3.0, 3.1, 4.0, 4.1]],
        ],
        dtype=torch.float32,
    )
    rewards = torch.tensor(
        [
            [[0.0, 0.0]],
            [[0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    dones = torch.zeros((3, 1, 2), dtype=torch.bool)
    dones[2, 0, 1] = True

    return Trajectory(
        max_episode_length=2,
        model_weights_id="synthetic-stride",
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        intervene_flags=torch.tensor(
            [
                [[[False], [False]]],
                [[[False], [True]]],
            ],
            dtype=torch.bool,
        ),
        forward_inputs={
            "x": torch.tensor(
                [
                    [[0.0, 0.1, 0.2]],
                    [[10.0, 10.1, 10.2]],
                    [[20.0, 20.1, 20.2]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[0.0, 0.0, 0.0, 0.0]],
                    [[0.1, 0.1, 0.1, 0.1]],
                    [[0.2, 0.2, 0.2, 0.2]],
                ],
                dtype=torch.float32,
            ),
            "source_chunk": torch.tensor(
                [
                    [[int(TransitionSource.RL), int(TransitionSource.RL)]],
                    [[int(TransitionSource.RL), int(TransitionSource.HUMAN)]],
                ],
                dtype=torch.uint8,
            ),
            "collection_phase_id": torch.tensor(
                [
                    [[COLLECTION_PHASE_WARMUP]],
                    [[COLLECTION_PHASE_ONLINE]],
                ],
                dtype=torch.uint8,
            ),
            "record_transition": torch.ones((2, 1, 1), dtype=torch.bool),
        },
        step_trace={
            "anchor_offsets": torch.tensor(
                [
                    [[1]],
                    [[1]],
                ],
                dtype=torch.long,
            ),
            "x": torch.tensor(
                [
                    [[[1.0, 1.1, 1.2]]],
                    [[[11.0, 11.1, 11.2]]],
                ],
                dtype=torch.float32,
            ),
            "a_tilde": torch.tensor(
                [
                    [[[1.0, 1.0, 1.0, 1.0]]],
                    [[[1.1, 1.1, 1.1, 1.1]]],
                ],
                dtype=torch.float32,
            ),
        },
    )


def _build_replay_trajectories() -> list[Trajectory]:
    adapter = RLTStage2TrajectoryReplayAdapter(_cfg())
    replay_trajectories, completed_episodes = adapter.build_replay_trajectories(
        _synthetic_rollout_trajectory()
    )
    assert completed_episodes == 1
    return replay_trajectories


def test_trajectory_adapter_emits_standard_replay_trajectories():
    replay_trajectories = _build_replay_trajectories()

    assert len(replay_trajectories) == 2
    first_inputs = replay_trajectories[0].forward_inputs
    second_inputs = replay_trajectories[1].forward_inputs

    required_keys = {
        "x",
        "a",
        "a_tilde",
        "action_chunk",
        "ref_chunk",
        "rewards",
        "next_x",
        "next_a_tilde",
        "next_ref_chunk",
        "dones",
        "intervention",
        "source",
        "source_chunk",
        "collection_phase_id",
        "success",
        "intervention_flag",
        "episode_id",
        "step_id",
    }
    assert required_keys.issubset(first_inputs)
    assert first_inputs["x"].shape == (1, 1, 3)
    assert first_inputs["action_chunk"].shape == (1, 1, 4)
    assert first_inputs["source_chunk"].shape == (1, 1, 2)
    assert first_inputs["dones"].item() == 0.0

    assert second_inputs["dones"].item() == 1.0
    assert bool(second_inputs["intervention_flag"].item()) is True
    assert second_inputs["collection_phase_id"].item() == COLLECTION_PHASE_ONLINE
    assert second_inputs["source"].item() == TransitionSource.MIXED
    torch.testing.assert_close(
        second_inputs["action_chunk"].reshape(-1),
        torch.tensor([5.0, 6.0, 7.0, 8.0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        second_inputs["a_tilde"].reshape(-1),
        torch.tensor([0.1, 0.1, 0.1, 0.1], dtype=torch.float32),
    )


def test_trajectory_adapter_builds_sparse_stride_transitions_from_step_trace():
    adapter = RLTStage2TrajectoryReplayAdapter(
        _cfg(replay_subsample_stride=1),
    )

    replay_trajectories, completed_episodes = adapter.build_replay_trajectories(
        _synthetic_stride_rollout_trajectory()
    )

    assert completed_episodes == 1
    assert len(replay_trajectories) == 4
    first_inputs = replay_trajectories[0].forward_inputs
    second_inputs = replay_trajectories[1].forward_inputs
    terminal_inputs = replay_trajectories[-1].forward_inputs

    torch.testing.assert_close(
        first_inputs["x"].reshape(-1),
        torch.tensor([0.0, 0.1, 0.2], dtype=torch.float32),
    )
    torch.testing.assert_close(
        first_inputs["next_x"].reshape(-1),
        torch.tensor([10.0, 10.1, 10.2], dtype=torch.float32),
    )
    assert first_inputs["source"].item() == TransitionSource.RL
    assert first_inputs["dones"].item() == 0.0
    assert first_inputs["step_id"].item() == 0

    torch.testing.assert_close(
        second_inputs["x"].reshape(-1),
        torch.tensor([1.0, 1.1, 1.2], dtype=torch.float32),
    )
    torch.testing.assert_close(
        second_inputs["next_x"].reshape(-1),
        torch.tensor([11.0, 11.1, 11.2], dtype=torch.float32),
    )
    assert second_inputs["step_id"].item() == 1

    assert terminal_inputs["source"].item() == TransitionSource.MIXED
    assert bool(terminal_inputs["intervention_flag"].item()) is True
    assert terminal_inputs["dones"].item() == 1.0
    assert terminal_inputs["step_id"].item() == 3
    torch.testing.assert_close(
        terminal_inputs["next_x"].reshape(-1),
        torch.tensor([20.0, 20.1, 20.2], dtype=torch.float32),
    )


def test_rlt_replay_buffer_roundtrip_uses_rlinf_trajectory_format(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=7,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    buffer.add_trajectories(replay_trajectories)
    buffer.close(wait=True)

    assert (tmp_path / "metadata.json").is_file()
    assert (tmp_path / "trajectory_index.json").is_file()
    assert len(list(tmp_path.glob("trajectory_*.pt"))) == 2

    loaded = TrajectoryReplayBuffer(
        seed=7,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    loaded.load_checkpoint(str(tmp_path))
    batch = loaded.sample(2)
    loaded.close(wait=True)

    assert loaded.total_samples == 2
    assert "forward_inputs" in batch
    forward_inputs = batch["forward_inputs"]
    assert forward_inputs["source_chunk"].shape == (2, 2)
    assert forward_inputs["rewards"].shape == (2, 2)
    assert forward_inputs["x"].shape == (2, 3)
    assert set(forward_inputs).issuperset(
        {
            "x",
            "a",
            "a_tilde",
            "action_chunk",
            "ref_chunk",
            "rewards",
            "next_x",
            "next_a_tilde",
            "next_ref_chunk",
            "dones",
            "intervention",
            "source",
            "source_chunk",
            "collection_phase_id",
            "success",
            "intervention_flag",
            "episode_id",
            "step_id",
        }
    )


def test_rlt_worker_training_batch_uses_replay_buffer_dataset():
    replay_trajectories = _build_replay_trajectories()
    replay_buffer = TrajectoryReplayBuffer(
        seed=13,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=False,
        auto_save_path="",
        trajectory_format="pt",
    )
    replay_buffer.add_trajectories(replay_trajectories)
    dataset = ReplayBufferDataset(
        replay_buffer=replay_buffer,
        demo_buffer=None,
        batch_size=2,
        min_replay_buffer_size=2,
        min_demo_buffer_size=0,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        drop_last=True,
        collate_fn=replay_buffer_collate_fn,
    )

    worker = object.__new__(RLTStage2FSDPPolicyWorker)
    worker.buffer_dataloader_iter = iter(dataloader)
    worker.device = torch.device("cpu")

    batch = RLTStage2FSDPPolicyWorker._next_rlt_replay_batch(worker, 2)
    replay_buffer.close(wait=True)

    assert "forward_inputs" not in batch
    assert batch["x"].shape == (2, 3)
    assert batch["a"].shape == (2, 4)
    assert batch["a_tilde"].shape == (2, 4)
    assert batch["action_chunk"].shape == (2, 4)
    assert batch["source_chunk"].shape == (2, 2)
    assert batch["rewards"].shape == (2, 2)


class _FakeRLTWorkerModel:
    def filter_rollout_state_dict(self, state_dict):
        return {
            key: value
            for key, value in state_dict.items()
            if key.startswith("actor.")
        }


def test_rlt_worker_rollout_sync_keeps_update_step_gate_version():
    worker = object.__new__(RLTStage2FSDPPolicyWorker)
    worker.model = _FakeRLTWorkerModel()
    worker.update_step = 17
    worker.version = 99
    worker._rollout_sync_key_count = 0
    worker.get_model_state_dict = lambda **kwargs: {
        "actor.weight": torch.ones(1),
        "critic.weight": torch.zeros(1),
    }

    state_dict = worker.get_rollout_state_dict()

    assert list(state_dict) == ["actor.weight"]
    assert worker.get_rollout_sync_param_names(state_dict) == ["actor.weight"]
    assert worker.get_rollout_sync_version() == 17
    assert worker._rollout_sync_key_count == 1


def test_inspect_rlt_replay_reads_standard_replay_directory(tmp_path):
    replay_trajectories = _build_replay_trajectories()
    buffer = TrajectoryReplayBuffer(
        seed=11,
        enable_cache=True,
        cache_size=4,
        sample_window_size=4,
        auto_save=True,
        auto_save_path=str(tmp_path),
        trajectory_format="pt",
    )
    buffer.add_trajectories(replay_trajectories)
    buffer.close(wait=True)

    replay_dir = inspect_rlt_replay._resolve_replay_dir(tmp_path)
    metadata, trajectories = inspect_rlt_replay._load_replay_directory(replay_dir)
    summary = inspect_rlt_replay._inspect(metadata, trajectories)

    assert summary["num_trajectories"] == 2
    assert summary["total_samples"] == 2
    assert summary["inspected_samples"] == 2
    assert summary["source"] == {"RL": 1, "MIXED": 1}
    assert summary["source_chunk"]["histogram"] == {
        "RL": 2,
        "HUMAN": 1,
        "MIXED": 1,
    }
    assert summary["collection_phase"] == {"WARMUP": 1, "ONLINE": 1}
    assert summary["reward"]["positive_transition_rate"] == 0.5
    assert summary["intervention_flag_rate"] == 0.5


def _maniskill_infos() -> dict[str, torch.Tensor]:
    return {
        "consecutive_grasp_current": torch.tensor([True]),
        "prealigned_current": torch.tensor([True]),
        "partial_insert_current": torch.tensor([False]),
        "success_current": torch.tensor([False]),
        "peg_head_goal_yz_dist": torch.tensor([0.01]),
        "peg_body_goal_yz_dist": torch.tensor([0.01]),
        "peg_head_hole_x": torch.tensor([0.02]),
        "peg_head_hole_abs_y": torch.tensor([0.01]),
        "peg_head_hole_abs_z": torch.tensor([0.01]),
    }


def test_rlt_stage2_policy_info_is_maniskill_local_correction_only():
    adapter = build_policy_info_adapter(
        _cfg(intervention_enable=True),
        train_batch_size=1,
        eval_batch_size=1,
    )

    initial = adapter.init_stage(stage_id=0, mode="train", env=None)
    assert initial is not None
    assert set(initial).issuperset(
        {
            "expert_takeover",
            "deviation",
            "deviation_count",
            "intervention_phase",
            "takeover_left",
            "takeover_used",
            "in_critical_phase",
            "record_transition",
        }
    )

    updated = adapter.update_stage(
        infos=_maniskill_infos(),
        chunk_dones=torch.tensor([[False, False]]),
        stage_id=0,
        mode="train",
        env=None,
    )
    assert updated["intervention_phase"].shape == (1, 1)
    assert updated["record_transition"].tolist() == [[True]]

    env_metrics = {}
    adapter.collect_rollout_metrics(
        env_metrics=env_metrics,
        rollout_result=type(
            "FakeRolloutResult",
            (),
            {
                "forward_inputs": {
                    "intervention_flags": torch.tensor([[True, False]]),
                    "intervention_phase": torch.tensor([[2.0]]),
                    "intervention_requested": torch.tensor([[True]]),
                    "ready_for_online": torch.tensor([[True]]),
                    "in_critical_phase": torch.tensor([[True]]),
                    "record_transition": torch.tensor([[True]]),
                    "student_control": torch.tensor([[True]]),
                }
            },
        )(),
    )
    assert "insert_intervention_rate" in env_metrics

    disabled = build_policy_info_adapter(
        _cfg(intervention_enable=False),
        train_batch_size=1,
        eval_batch_size=1,
    )
    assert disabled.__class__.__name__ == "NoopPolicyInfoAdapter"

    non_maniskill = build_policy_info_adapter(
        _cfg(env_type="libero"),
        train_batch_size=1,
        eval_batch_size=1,
    )
    assert non_maniskill.__class__.__name__ == "NoopPolicyInfoAdapter"


def test_policy_info_adapter_factory_uses_model_plugin_discovery():
    rlt_adapter = build_policy_info_adapter(
        _cfg(),
        train_batch_size=2,
        eval_batch_size=2,
    )
    assert rlt_adapter.__class__.__name__ == "RLTStage2PolicyInfoAdapter"

    cfg = _cfg(warmup_updates=0)
    cfg.actor.model.model_type = "openpi"
    noop_adapter = build_policy_info_adapter(
        cfg,
        train_batch_size=2,
        eval_batch_size=2,
    )
    assert noop_adapter.__class__.__name__ == "NoopPolicyInfoAdapter"


class _FakeStudentModel:
    def __init__(self):
        self.base = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)

    def predict_action_batch(self, env_obs, **kwargs):
        del env_obs, kwargs
        actions = torch.tensor(
            [[[1.0, 1.0], [2.0, 2.0]]],
            dtype=torch.float32,
        )
        return actions, {
            "prev_logprobs": torch.zeros((1, 1), dtype=torch.float32),
            "prev_values": torch.zeros((1, 1), dtype=torch.float32),
            "forward_inputs": {
                "x": torch.ones((1, 3), dtype=torch.float32),
                "a_tilde": self.base.clone(),
            },
        }

    def rollout_state_dict(self):
        return {"student": torch.ones(1)}

    def encode_obs(self, obs):
        states = obs["states"].to(torch.float32)
        return states, torch.zeros((states.shape[0], 4), dtype=torch.float32)


class _FakeExpertModel:
    def predict_action_batch(self, env_obs, **kwargs):
        del env_obs, kwargs
        return torch.full((1, 2, 2), 9.0), {}


def test_rollout_adapter_routes_warmup_base_then_online_local_correction():
    expert = _FakeExpertModel()
    adapter = RLTStage2RolloutAdapter(
        cfg=_cfg(
            intervention_enable=True,
            warmup_updates=5,
        ),
        student_model=_FakeStudentModel(),
        expert_model_getter=lambda: expert,
        has_expert_model_config=True,
    )
    policy_info = {
        "expert_takeover": torch.tensor([[True]]),
        "in_critical_phase": torch.tensor([[True]]),
        "record_transition": torch.tensor([[True]]),
        "intervention_phase": torch.tensor([[2.0]]),
    }

    warmup = adapter.predict(
        env_obs={"states": torch.zeros((1, 3))},
        policy_info=policy_info,
        model_kwargs={"mode": "train"},
        mode="train",
        allow_expert=True,
        update_version=4,
    )
    torch.testing.assert_close(
        warmup.actions,
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
    )
    assert warmup.expert_label_flag is False
    assert bool(warmup.result["forward_inputs"]["ready_for_online"].item()) is False
    assert (
        warmup.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.BASE)
    ).all()

    online = adapter.predict(
        env_obs={"states": torch.zeros((1, 3))},
        policy_info=policy_info,
        model_kwargs={"mode": "train"},
        mode="train",
        allow_expert=True,
        update_version=5,
    )
    torch.testing.assert_close(
        online.actions,
        torch.full((1, 2, 2), 9.0),
    )
    assert online.expert_label_flag is True
    assert bool(online.result["forward_inputs"]["ready_for_online"].item()) is True
    assert online.result["forward_inputs"]["intervention_flags"].all().item() is True
    assert (
        online.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.HUMAN)
    ).all()
    assert online.result["forward_inputs"]["student_control"].item() is True
    assert (
        online.result["forward_inputs"]["collection_phase_id"].item()
        == COLLECTION_PHASE_ONLINE
    )

    autonomous_online = adapter.predict(
        env_obs={"states": torch.zeros((1, 3))},
        policy_info={
            "expert_takeover": torch.tensor([[False]]),
            "in_critical_phase": torch.tensor([[True]]),
            "record_transition": torch.tensor([[True]]),
            "intervention_phase": torch.tensor([[0.0]]),
        },
        model_kwargs={"mode": "train"},
        mode="train",
        allow_expert=True,
        update_version=5,
    )
    torch.testing.assert_close(
        autonomous_online.actions,
        torch.tensor([[[1.0, 1.0], [2.0, 2.0]]], dtype=torch.float32),
    )
    assert autonomous_online.expert_label_flag is False
    assert autonomous_online.result["forward_inputs"]["student_control"].item() is True
    assert (
        autonomous_online.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.RL)
    ).all()

    online_before_critical_phase = adapter.predict(
        env_obs={"states": torch.zeros((1, 3))},
        policy_info={
            "expert_takeover": torch.tensor([[False]]),
            "in_critical_phase": torch.tensor([[False]]),
            "record_transition": torch.tensor([[False]]),
            "intervention_phase": torch.tensor([[0.0]]),
        },
        model_kwargs={"mode": "train"},
        mode="train",
        allow_expert=True,
        update_version=5,
    )
    torch.testing.assert_close(
        online_before_critical_phase.actions,
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float32),
    )
    assert (
        online_before_critical_phase.result["forward_inputs"][
            "student_control"
        ].item()
        is False
    )
    assert (
        online_before_critical_phase.result["forward_inputs"]["source_chunk"]
        == int(TransitionSource.BASE)
    ).all()
    assert (
        online_before_critical_phase.result["forward_inputs"][
            "record_transition"
        ].item()
        is False
    )


def test_rollout_adapter_encodes_sparse_step_trace_without_env_dependencies():
    adapter = RLTStage2RolloutAdapter(
        cfg=_cfg(
            intervention_enable=False,
            warmup_updates=0,
            replay_subsample_stride=1,
        ),
        student_model=_FakeStudentModel(),
        expert_model_getter=lambda: _FakeExpertModel(),
        has_expert_model_config=False,
    )

    trace = adapter.encode_step_trace(
        {
            "states": torch.tensor(
                [
                    [[2.0, 2.1, 2.2]],
                ],
                dtype=torch.float32,
            ),
            "_rlt_step_offsets": torch.tensor([[1]], dtype=torch.long),
        }
    )

    assert trace["anchor_offsets"].tolist() == [[1]]
    torch.testing.assert_close(
        trace["x"],
        torch.tensor([[[2.0, 2.1, 2.2]]], dtype=torch.float32),
    )
    assert trace["a_tilde"].shape == (1, 1, 4)
