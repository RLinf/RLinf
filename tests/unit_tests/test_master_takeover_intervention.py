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

import socket
import sys
import time
import types

import gymnasium as gym
import numpy as np
import pytest

sys.modules.setdefault("cv2", types.ModuleType("cv2"))

from rlinf.envs.realworld.common.takeover.x2robot_protocol import (  # noqa: E402
    MSG_JOINT,
    MSG_MODE,
    X2RobotTakeoverTCPConfig,
    X2RobotTakeoverTCPServer,
    make_ros_running_mode_getter,
    recv_frame,
)
from rlinf.envs.realworld.common.wrappers.master_takeover_intervention import (  # noqa: E402
    MasterTakeoverIntervention,
)


class FakeTakeoverAdapter:
    def __init__(self, states, initial_active=False, events=None):
        self.states = list(states)
        self.initial_active = initial_active
        self.index = -1
        self.started = False
        self.closed = False
        self.events = events
        self.sync_count = 0
        self.config = type("Config", (), {})()

    def start(self):
        self.started = True

    def close(self):
        self.closed = True

    def poll(self):
        if self.events is not None:
            self.events.append("poll")
        self.index += 1

    def sync_control_plane(self):
        self.sync_count += 1
        if self.events is not None:
            self.events.append("sync")

    def is_takeover_active(self):
        if self.index < 0:
            return self.initial_active
        return bool(self.states[min(self.index, len(self.states) - 1)][0])

    def is_connected(self):
        return True

    def get_takeover_action(self):
        if self.index < 0:
            return None
        action = self.states[min(self.index, len(self.states) - 1)][1]
        return None if action is None else np.asarray(action, dtype=np.float32)


class DummyTakeoverEnv(gym.Env):
    def __init__(
        self,
        joint_shape=(2, 7),
        events=None,
        executed_action=None,
        action_rejected=False,
    ):
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -10.0, 10.0, shape=(14,), dtype=np.float32
        )
        self.last_action = None
        self.joint_snapshot = np.zeros(joint_shape, dtype=np.float32)
        self.pose_snapshot = np.arange(14, dtype=np.float32).reshape(2, 7)
        self.events = events
        self.executed_action = executed_action
        self.action_rejected = action_rejected

    def step(self, action):
        if self.events is not None:
            self.events.append("env_step")
        self.last_action = np.asarray(action, dtype=np.float32)
        info = {}
        if self.executed_action is not None:
            info["executed_action"] = np.asarray(
                self.executed_action, dtype=np.float32
            )
        if self.action_rejected:
            info["action_rejected"] = True
            info["rejection_reason"] = "outside_absolute_pose_action_space"
        return np.zeros(14, dtype=np.float32), 0.0, False, False, info

    def reset(self, *, seed=None, options=None):
        return np.zeros(14, dtype=np.float32), {}

    def get_joint_snapshot(self):
        return self.joint_snapshot

    def get_arm_pose_snapshot(self):
        return self.pose_snapshot

    def hold_current_pose_for_takeover(self):
        if self.events is not None:
            self.events.append("hard_hold")
        return {
            "pose": self.pose_snapshot.copy(),
            "joint": self.joint_snapshot.copy(),
        }


def test_master_takeover_mode_1_passes_policy_action():
    env = DummyTakeoverEnv()
    adapter = FakeTakeoverAdapter(states=[(False, None)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    action = np.ones(14, dtype=np.float32)
    _, _, _, _, info = wrapped.step(action)

    np.testing.assert_array_equal(env.last_action, action)
    np.testing.assert_array_equal(info["executed_action"], action)
    assert info["intervene_flag"].shape == (1,)
    assert not info["intervene_flag"].item()
    assert "intervene_action" not in info


def test_master_takeover_mode_2_without_fresh_pose_holds_current_pose():
    events = []
    env = DummyTakeoverEnv(events=events)
    adapter = FakeTakeoverAdapter(states=[(True, None)], events=events)
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    action = np.ones(14, dtype=np.float32)
    _, _, _, _, info = wrapped.step(action)

    expected_hold = env.pose_snapshot.reshape(-1)
    np.testing.assert_array_equal(env.last_action, expected_hold)
    np.testing.assert_array_equal(info["executed_action"], expected_hold)
    assert info["takeover_active"]
    assert info["takeover_sync_hold"]
    assert not info["takeover_chunk_hold"]
    assert not info["intervene_flag"].item()
    assert "intervene_action" not in info
    assert events == ["poll", "hard_hold", "sync", "env_step"]


def test_master_takeover_first_sync_hold_hard_holds_before_sync():
    events = []
    env = DummyTakeoverEnv(events=events)
    adapter = FakeTakeoverAdapter(states=[(True, None)], events=events)
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    wrapped.step(np.ones(14, dtype=np.float32))

    assert events.index("hard_hold") < events.index("sync")
    assert events.index("sync") < events.index("env_step")


def test_master_takeover_mode_2_fresh_pose_overrides_policy_action():
    env = DummyTakeoverEnv()
    expert = np.arange(14, dtype=np.float32)
    adapter = FakeTakeoverAdapter(states=[(True, expert)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    policy = np.ones(14, dtype=np.float32)
    _, _, _, _, info = wrapped.step(policy)

    np.testing.assert_array_equal(env.last_action, expert)
    np.testing.assert_array_equal(info["executed_action"], expert)
    np.testing.assert_array_equal(info["intervene_action"], expert)
    assert info["intervene_flag"].item()


def test_master_takeover_records_post_clip_executed_action_as_expert():
    expert = np.arange(14, dtype=np.float32)
    executed = expert.copy()
    executed[0] = 0.25
    env = DummyTakeoverEnv(executed_action=executed)
    adapter = FakeTakeoverAdapter(states=[(True, expert)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    _, _, _, _, info = wrapped.step(np.ones(14, dtype=np.float32))

    np.testing.assert_array_equal(env.last_action, expert)
    np.testing.assert_array_equal(info["executed_action"], executed)
    np.testing.assert_array_equal(info["intervene_action"], executed)
    assert info["intervene_flag"].item()


def test_master_takeover_rejected_pose_does_not_write_expert_label():
    expert = np.arange(14, dtype=np.float32)
    executed = np.zeros(14, dtype=np.float32)
    env = DummyTakeoverEnv(executed_action=executed, action_rejected=True)
    adapter = FakeTakeoverAdapter(states=[(True, expert)])
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    _, _, _, _, info = wrapped.step(np.ones(14, dtype=np.float32))

    np.testing.assert_array_equal(env.last_action, expert)
    np.testing.assert_array_equal(info["executed_action"], executed)
    np.testing.assert_array_equal(info["raw_intervene_action"], expert)
    assert info["intervene_rejected"] is True
    assert not info["intervene_flag"].item()
    assert "intervene_action" not in info


def test_master_takeover_holds_after_takeover_until_chunk_boundary():
    env = DummyTakeoverEnv()
    expert = np.full(14, 3.0, dtype=np.float32)
    adapter = FakeTakeoverAdapter(
        states=[
            (True, expert),
            (False, None),
            (False, None),
        ]
    )
    wrapped = MasterTakeoverIntervention(env, adapter=adapter)

    wrapped.begin_action_chunk()
    wrapped.step(np.ones(14, dtype=np.float32))
    _, _, _, _, hold_info = wrapped.step(np.full(14, 2.0, dtype=np.float32))

    expected_hold = env.pose_snapshot.reshape(-1)
    np.testing.assert_array_equal(env.last_action, expected_hold)
    assert hold_info["takeover_chunk_hold"]
    assert not hold_info["takeover_sync_hold"]
    assert not hold_info["intervene_flag"].item()

    wrapped.end_action_chunk()
    policy_after_boundary = np.full(14, 4.0, dtype=np.float32)
    wrapped.step(policy_after_boundary)
    np.testing.assert_array_equal(env.last_action, policy_after_boundary)


def test_master_takeover_rejects_bad_joint_snapshot_shape():
    with pytest.raises(ValueError, match=r"shape \(2, 7\)"):
        MasterTakeoverIntervention(
            DummyTakeoverEnv(joint_shape=(1, 7)),
            adapter=FakeTakeoverAdapter(states=[]),
        )


def test_takeover_control_plane_sends_joint_before_mode_on_takeover():
    client_sock, server_sock = socket.socketpair()
    try:
        adapter = X2RobotTakeoverTCPServer(
            config=X2RobotTakeoverTCPConfig(takeover_delay_s=0.0),
            running_mode_getter=lambda: 2,
            joint_snapshot_getter=lambda: np.arange(14, dtype=np.float32).reshape(2, 7),
        )
        adapter._client_socket = server_sock
        adapter._current_mode = 2
        adapter._mode_dirty = True
        adapter._snapshot_dirty = True

        adapter.sync_control_plane()

        first = recv_frame(client_sock)
        second = recv_frame(client_sock)
        third = recv_frame(client_sock)
        assert first["header"]["msg_type"] == MSG_JOINT
        assert second["header"]["msg_type"] == MSG_MODE
        assert second["payload"]["running_mode"] == 2
        assert third["header"]["msg_type"] == MSG_JOINT
    finally:
        client_sock.close()
        server_sock.close()


def test_takeover_action_rejects_stale_pose_receive_time():
    adapter = X2RobotTakeoverTCPServer(
        config=X2RobotTakeoverTCPConfig(takeover_delay_s=0.0, max_pose_age_s=0.01),
        running_mode_getter=lambda: 2,
        joint_snapshot_getter=lambda: np.zeros((2, 7), dtype=np.float32),
    )
    now = time.time()
    with adapter._state_lock:
        adapter._current_mode = 2
        adapter._follow_start_time = now - 1.0
        adapter._min_pose_timestamp_us = 0
        adapter._master_pose_left = np.ones(7, dtype=np.float32)
        adapter._master_pose_right = np.ones(7, dtype=np.float32) * 2
        adapter._master_pose_timestamp_us = int(now * 1e6)
        adapter._master_pose_seq = 3
        adapter._master_pose_recv_time = now - 1.0

    assert adapter.get_takeover_action() is None


def test_takeover_action_rejects_pose_received_before_follow_gate():
    adapter = X2RobotTakeoverTCPServer(
        config=X2RobotTakeoverTCPConfig(takeover_delay_s=0.0, max_pose_age_s=1.0),
        running_mode_getter=lambda: 2,
        joint_snapshot_getter=lambda: np.zeros((2, 7), dtype=np.float32),
    )
    now = time.time()
    follow_start_time = now - 0.5
    with adapter._state_lock:
        adapter._current_mode = 2
        adapter._follow_start_time = follow_start_time
        adapter._min_pose_timestamp_us = int(follow_start_time * 1e6)
        adapter._master_pose_left = np.ones(7, dtype=np.float32)
        adapter._master_pose_right = np.ones(7, dtype=np.float32) * 2
        adapter._master_pose_timestamp_us = int(now * 1e6)
        adapter._master_pose_seq = 4
        adapter._master_pose_recv_time = follow_start_time - 0.1

    assert adapter.get_takeover_action() is None


def test_takeover_action_accepts_fresh_recv_time_with_skewed_master_timestamp():
    adapter = X2RobotTakeoverTCPServer(
        config=X2RobotTakeoverTCPConfig(takeover_delay_s=0.0, max_pose_age_s=1.0),
        running_mode_getter=lambda: 2,
        joint_snapshot_getter=lambda: np.zeros((2, 7), dtype=np.float32),
    )
    now = time.time()
    left = np.arange(7, dtype=np.float32)
    right = np.arange(7, 14, dtype=np.float32)
    with adapter._state_lock:
        adapter._current_mode = 2
        adapter._follow_start_time = now - 0.5
        adapter._min_pose_timestamp_us = int((now - 0.5) * 1e6)
        adapter._master_pose_left = left
        adapter._master_pose_right = right
        adapter._master_pose_timestamp_us = int((now - 100.0) * 1e6)
        adapter._master_pose_seq = 5
        adapter._master_pose_recv_time = now

    np.testing.assert_array_equal(
        adapter.get_takeover_action(),
        np.concatenate([left, right]).astype(np.float32),
    )


def test_takeover_action_accepts_fresh_pose_after_follow_gate():
    adapter = X2RobotTakeoverTCPServer(
        config=X2RobotTakeoverTCPConfig(takeover_delay_s=0.0, max_pose_age_s=1.0),
        running_mode_getter=lambda: 2,
        joint_snapshot_getter=lambda: np.zeros((2, 7), dtype=np.float32),
    )
    now = time.time()
    left = np.arange(7, dtype=np.float32)
    right = np.arange(7, 14, dtype=np.float32)
    with adapter._state_lock:
        adapter._current_mode = 2
        adapter._follow_start_time = now - 1.0
        adapter._min_pose_timestamp_us = int((now - 0.5) * 1e6)
        adapter._master_pose_left = left
        adapter._master_pose_right = right
        adapter._master_pose_timestamp_us = int(now * 1e6)
        adapter._master_pose_seq = 6
        adapter._master_pose_recv_time = now

    np.testing.assert_array_equal(
        adapter.get_takeover_action(),
        np.concatenate([left, right]).astype(np.float32),
    )


def test_ros_running_mode_getter_requires_rospy(monkeypatch):
    monkeypatch.setitem(sys.modules, "rospy", None)
    getter = make_ros_running_mode_getter(X2RobotTakeoverTCPConfig())

    with pytest.raises(RuntimeError, match="requires rospy"):
        getter()


def test_ros_running_mode_getter_rejects_non_int_param(monkeypatch):
    fake_rospy = types.SimpleNamespace(get_param=lambda *_args: "takeover")
    monkeypatch.setitem(sys.modules, "rospy", fake_rospy)
    getter = make_ros_running_mode_getter(X2RobotTakeoverTCPConfig())

    with pytest.raises(RuntimeError, match="must be an integer"):
        getter()


def test_ros_running_mode_getter_returns_int_param(monkeypatch):
    fake_rospy = types.SimpleNamespace(get_param=lambda *_args: "2")
    monkeypatch.setitem(sys.modules, "rospy", fake_rospy)
    getter = make_ros_running_mode_getter(X2RobotTakeoverTCPConfig())

    assert getter() == 2
