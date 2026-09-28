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

"""Hardware-free safety checks for the two YAM follower arms."""

from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.real.yam.config import DualYamJointEnvConfig
from rlinf.envs.real.yam.control_runtime import YamControlRuntime
from rlinf.envs.real.yam.types import YamArmState


class _Follower:
    def __init__(self, device, events):
        self.name = device.name
        self.action = np.asarray(device.action, dtype=np.float64).copy()
        self.fail_connect = device.fail_connect
        self.events = events
        self.commands = []
        self.hold_calls = 0
        self.close_calls = 0
        self.timestamp_s = 100.0

    def connect(self):
        self.events.append(("connect", self.name))
        if self.fail_connect:
            raise RuntimeError(f"cannot connect {self.name}")

    def read_state(self):
        return YamArmState(self.action[:6], self.action[6], self.timestamp_s)

    def command(self, target):
        self.action = np.asarray(target, dtype=np.float64).copy()
        self.commands.append(self.action.copy())

    def joint_limits(self):
        return np.column_stack((np.full(6, -np.pi), np.full(6, np.pi)))

    def assert_healthy(self, max_feedback_age_s):
        del max_feedback_age_s

    def hold(self):
        self.hold_calls += 1
        self.events.append(("hold", self.name))

    def close(self):
        self.close_calls += 1
        self.events.append(("close", self.name))


class _Factory:
    def __init__(self):
        self.events = []
        self.followers = []

    def create_follower(self, device):
        self.events.append(("create", device.name))
        follower = _Follower(device, self.events)
        self.followers.append(follower)
        return follower


def _runtime(*, config=None, left=None, right=None, fail_right=False):
    factory = _Factory()
    hardware = SimpleNamespace(
        left_follower=SimpleNamespace(
            name="left",
            action=np.zeros(7) if left is None else left,
            fail_connect=False,
        ),
        right_follower=SimpleNamespace(
            name="right",
            action=np.zeros(7) if right is None else right,
            fail_connect=fail_right,
        ),
    )
    runtime = YamControlRuntime(
        config or DualYamJointEnvConfig(),
        hardware,
        factory,
        clock=lambda: 100.0,
        sleeper=lambda _seconds: None,
    )
    return runtime, factory


def test_followers_connect_in_order_and_report_14d_measured_state():
    left = np.array([0.1] * 6 + [0.75])
    right = np.array([-0.1] * 6 + [0.25])
    runtime, factory = _runtime(left=left, right=right)
    assert factory.events == []

    runtime.connect_followers()

    np.testing.assert_allclose(runtime.read_state().as_vector(), np.r_[left, right])
    assert factory.events.index(("hold", "left")) < factory.events.index(
        ("create", "right")
    )
    runtime.close()


def test_failed_second_connection_releases_both_followers():
    runtime, factory = _runtime(fail_right=True)
    with pytest.raises(RuntimeError, match="cannot connect right"):
        runtime.connect_followers()
    assert [follower.close_calls for follower in factory.followers] == [1, 1]
    assert not runtime.followers_connected


def test_action_validation_happens_before_any_motor_write():
    runtime, factory = _runtime()
    runtime.connect_followers()
    with pytest.raises(ValueError, match="shape"):
        runtime.command(np.zeros(13))
    assert all(not follower.commands for follower in factory.followers)
    runtime.close()


def test_nonfinite_and_stale_inputs_hold_both_followers():
    runtime, factory = _runtime()
    runtime.connect_followers()
    bad_action = np.zeros(14)
    bad_action[0] = np.nan
    result = runtime.command(bad_action)
    assert result.rejection_reason == "non_finite_action"
    assert all(not follower.commands for follower in factory.followers)
    assert [follower.hold_calls for follower in factory.followers] == [2, 2]

    factory.followers[0].timestamp_s = 99.0
    with pytest.raises(RuntimeError, match="stale YAM follower feedback"):
        runtime.command(np.zeros(14))
    assert all(not follower.commands for follower in factory.followers)
    runtime.close()


def test_command_clamps_joint_step_and_gripper_then_closes_once():
    config = DualYamJointEnvConfig(max_joint_delta=0.2)
    runtime, factory = _runtime(config=config)
    runtime.connect_followers()
    requested = np.r_[np.full(6, 0.8), 2.0, np.full(6, -0.8), -1.0]

    result = runtime.command(requested)

    np.testing.assert_allclose(
        result.accepted, np.r_[np.full(6, 0.2), 1.0, np.full(6, -0.2), 0.0]
    )
    assert result.clipped
    runtime.close()
    runtime.close()
    assert [follower.close_calls for follower in factory.followers] == [1, 1]
    assert [name for event, name in factory.events if event == "close"] == [
        "right",
        "left",
    ]
