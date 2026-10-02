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

"""YAM VR controller, episode, and accepted-action contracts with mock hardware.

The station under test is the real one: the ``yam_pico`` device, the shared
intervention wrapper, and the YAM PICO episode wrapper, composed the way
``WrapperStack`` composes them. Only the hardware behind the device is
replaced -- the controllers report readings the test writes, and IK answers
with poses the test chooses.
"""

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
from rlinf.envs.real.wrappers.teleop.facts import EnvFacts
from rlinf.envs.real.wrappers.teleop.intervention import TeleopIntervention
from rlinf.envs.real.wrappers.teleop.layout import action_spec
from rlinf.envs.real.yam.config import YamPicoConfig
from rlinf.envs.real.yam.dual_yam_joint_env import DualYamJointEnv
from rlinf.envs.real.yam.kinematics import YamIKResult
from rlinf.envs.real.yam.pico_episode import YamPicoEpisode
from rlinf.robotics.parts.teleop.group import TeleopEntry, TeleopGroup
from rlinf.robotics.parts.teleop.yam_pico import SIDES, YamPico


class Arm:
    """A controller stub that answers with an action instead of mapping a pose.

    ``_YamPicoArm`` keeps the pose the grip was anchored to and turns the
    operator's motion into a scaled spatial error. That mapping has its own
    tests below; the station tests replace it so they can state the action
    they mean directly.
    """

    def __init__(self) -> None:
        self.ready = True
        self.held = False
        self.buttons: dict[str, bool] = {}
        self.action = np.array([0.5, 0, 0, 0, 0, 0, 0.0])
        self.reference = False
        self.resets = 0
        self.stopped = False

    def read(self) -> dict[str, object]:
        return {
            "held": self.held,
            "ready": self.ready,
            "calibrated": True,
            "control_value": float(self.held),
        }

    def read_buttons(self) -> dict[str, bool]:
        return self.buttons if self.ready else {}

    def reset_reference(self) -> None:
        self.reference = False
        self.resets += 1

    def command(
        self,
        reading,
        tcp_pose,
        action_scale,
        *,
        gripper_enabled=True,
        clip_motion=True,
    ):
        """Return the action this stub was told to produce for one reading."""
        del tcp_pose, action_scale, gripper_enabled, clip_motion
        held = bool(reading.get("held", False))
        action = self.action.copy()
        if held and not self.reference:
            # A freshly closed grip is anchored where the arm already stands.
            action[:6] = 0
        self.reference = held
        info = {
            "pico_active": held,
            "pico_ready": self.ready,
            "pico_calibrated": True,
            "pico_control_value": float(held),
        }
        return action.astype(np.float32), held and self.ready, info

    def stop(self) -> None:
        self.stopped = True


class Kinematics:
    """IK that answers with the target pose itself, unless told to fail."""

    def __init__(self) -> None:
        self.fail = False
        self.targets: list[np.ndarray] = []

    def fk(self, q, gripper):
        pose = np.eye(4)
        pose[:3, 3] = q[:3]
        pose[:3, :3] = Rotation.from_rotvec(q[3:6]).as_matrix()
        return pose

    def solve(self, target, seed, gripper):
        self.targets.append(target.copy())
        q = np.concatenate(
            [target[:3, 3], Rotation.from_matrix(target[:3, :3]).as_rotvec()]
        )
        return YamIKResult(
            not self.fail, q, 0.0, 0.0, 0.0, "injected_failure" if self.fail else None
        )


def build_station(*, arms=None, kinematics=None, **base_overrides):
    """Compose the PICO station the shared wrapper stack builds.

    Returns a namespace holding the wrapper a caller steps, the device under
    it, the base environment, and the injected controllers and IK.
    """
    base = DualYamJointEnv(
        {
            "is_dummy": True,
            "image_height": 8,
            "image_width": 8,
            "manual_episode_control_only": True,
            **base_overrides,
        }
    )
    base._pace = lambda: None
    config = YamPicoConfig(wait_for_record_button=False, button_debounce_s=0.0)
    spec = action_spec(base)
    facts = EnvFacts.about(base, spec.layout, spec.kinds)
    device = YamPico(
        config,
        joint_step_limits=facts.joint_step_limits,
        joint_lower=facts.joint_limit_min,
        joint_upper=facts.joint_limit_max,
        experts=arms if arms is not None else {side: Arm() for side in SIDES},
        kinematics=(
            kinematics
            if kinematics is not None
            else {side: Kinematics() for side in SIDES}
        ),
    )
    device.connect()
    group = TeleopGroup([TeleopEntry(device)], available=facts.kinds)
    composed = ComposedTeleop(group, facts.layout, timeout=group.hold_window)
    return SimpleNamespace(
        env=YamPicoEpisode(
            TeleopIntervention(base, composed, mark_flag=base.TELEOP_MARK_FLAG),
            config,
        ),
        device=device,
        base=base,
        arms=device._experts,
        kinematics=device._kinematics,
        layout=dict(facts.layout),
    )


@pytest.fixture
def station():
    composed = build_station()
    composed.env.reset()
    composed.env.step(np.zeros(14))  # Observe released grips, prime button edges.
    yield composed
    composed.env.close()


def press(station, button):
    """Release one frame, then hold, producing a button edge on the second."""
    station.arms["right"].buttons = {}
    station.env.step(np.zeros(14))
    station.arms["right"].buttons = {button: True}
    return station.env.step(np.zeros(14))


def test_single_arm_first_engagement_and_gripper_latch(station):
    arm = station.arms["left"]
    arm.held = True
    _, _, _, _, first = station.env.step(np.ones(14))
    np.testing.assert_allclose(first["intervene_action"][:6], 0.0)
    arm.action[6] = 1.0
    _, _, _, _, info = station.env.step(np.ones(14))
    assert info["intervene_action"][0] > 0
    assert info["intervene_action"][6] == 1.0
    np.testing.assert_array_equal(info["intervene_action"][7:], np.zeros(7))
    arm.action[6] = 0.0
    assert station.env.step(np.zeros(14))[-1]["intervene_action"][6] == 1.0
    arm.held = False
    held = station.base.get_hold_action()
    np.testing.assert_allclose(
        station.env.step(np.zeros(14))[-1]["intervene_action"], held
    )


@pytest.mark.parametrize("fault", ["stale", "nan"])
def test_fault_holds_both_and_requires_release_before_reengagement(station, fault):
    for arm in station.arms.values():
        arm.held = True
    station.env.step(np.zeros(14))
    held = station.base.get_hold_action()
    if fault == "stale":
        station.arms["right"].ready = False
    else:
        station.arms["right"].action[0] = np.nan
    info = station.env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"]
    np.testing.assert_allclose(info["intervene_action"], held)
    station.arms["right"].ready = True
    station.arms["right"].action[0] = 0.5
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"]
    for arm in station.arms.values():
        arm.held = False
    assert station.env.step(np.zeros(14))[-1]["yam_pico_fault"] is None
    station.arms["left"].held = True
    np.testing.assert_allclose(
        station.env.step(np.zeros(14))[-1]["intervene_action"], held
    )


def test_record_edges_abort_and_success_preserve_only_manual_boundary(station):
    station.arms["left"].held = True
    info = press(station, "right_menu_button")[-1]
    assert info["record_reset"] and not info["pre_record"]
    assert not station.env.step(np.zeros(14))[2]  # Held button cannot end again.
    result = press(station, "right_menu_button")
    assert result[1] == 1 and result[2] and result[-1]["success"]
    count = station.arms["left"].resets
    station.env.reset()
    assert station.arms["left"].resets == count
    station.env.reset()  # Early reset must clear the reference.
    assert station.arms["left"].resets > count
    station.arms["left"].held = False
    station.env.step(np.zeros(14))
    press(station, "right_menu_button")
    info = press(station, "left_menu_button")[-1]
    assert info["pre_record"] and info["record_reset"] and not info["success"]


class PackedObservations(gym.ObservationWrapper):
    def observation(self, obs):
        return {
            "states": obs["state"]["joint_position"],
            "main_images": obs["frames"]["top_rgb"],
            "extra_view_images": np.stack(
                [obs["frames"]["left_rgb"], obs["frames"]["right_rgb"]]
            ),
            "task_descriptions": "pick_block",
        }


def test_streaming_discards_operator_abort_but_keeps_fault_as_failure(
    station, tmp_path, monkeypatch
):
    """An explicit abort must not publish data; a hardware fault keeps evidence."""
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    collector = CollectEpisode(
        PackedObservations(station.env),
        str(tmp_path),
        export_format="lerobot",
        streaming=True,
    )
    events = []
    monkeypatch.setattr(collector, "_submit", lambda fn, *args: fn(*args))
    monkeypatch.setattr(
        collector, "_stream_add_frame", lambda frame: events.append("frame")
    )
    monkeypatch.setattr(
        collector, "_stream_discard_episode", lambda: events.append("discard")
    )
    monkeypatch.setattr(
        collector,
        "_stream_finish_episode",
        lambda is_success, recording_invalid=False: events.append(
            ("save", is_success, recording_invalid)
        ),
    )

    def edge(button):
        station.arms["right"].buttons = {}
        collector.step(np.zeros(14))
        station.arms["right"].buttons = {button: True}
        return collector.step(np.zeros(14))

    try:
        collector.reset()
        edge("right_menu_button")
        station.arms["left"].held = True
        collector.step(np.zeros(14))
        aborted = edge("left_menu_button")[-1]
        assert aborted["episode_discarded"]
        assert events.count("frame") > 0
        assert events[-1] == "discard"
        assert not any(isinstance(event, tuple) for event in events)

        station.arms["left"].held = False
        station.arms["right"].buttons = {}
        collector.step(np.zeros(14))
        edge("right_menu_button")
        station.arms["left"].held = True
        collector.step(np.zeros(14))
        station.arms["left"].ready = False
        fault = collector.step(np.zeros(14))[-1]
        assert fault["record_reset"] and not fault.get("episode_discarded", False)
        assert events[-1] == ("save", False, False)
    finally:
        collector.close()


def test_streaming_lerobot_writes_three_views_and_14d_vectors(station, tmp_path):
    """Exercise the collection example's actual disk writer without hardware."""
    pytest.importorskip("lerobot")
    pytest.importorskip("cv2")
    parquet = pytest.importorskip("pyarrow.parquet")
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    collector = CollectEpisode(
        PackedObservations(station.env),
        str(tmp_path),
        export_format="lerobot",
        streaming=True,
        robot_type="dual_yam",
        fps=30,
    )

    def edge(button):
        station.arms["right"].buttons = {}
        collector.step(np.zeros(14))
        station.arms["right"].buttons = {button: True}
        return collector.step(np.zeros(14))

    try:
        collector.reset()
        edge("right_menu_button")
        station.arms["left"].held = True
        accepted = collector.step(np.zeros(14))[-1]["accepted_action"]
        edge("right_menu_button")
    finally:
        collector.close()

    shard = tmp_path / "rank_0/id_0"
    data_files = list(shard.glob("data/**/*.parquet"))
    assert len(data_files) == 1
    table = parquet.read_table(data_files[0])
    assert table.num_rows > 0
    assert len(table["state"][0].as_py()) == 14
    assert len(table["actions"][0].as_py()) == 14
    assert any(np.allclose(action, accepted) for action in table["actions"].to_pylist())
    assert {"image", "extra_view_image-0", "extra_view_image-1"} <= set(
        table.column_names
    )
