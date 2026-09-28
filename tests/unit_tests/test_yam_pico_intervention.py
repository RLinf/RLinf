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

import shutil
import tempfile
import time
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


@pytest.mark.parametrize("side", ["left", "right"])
def test_ik_failure_holds_only_failed_arm_and_retries_without_release(station, side):
    for arm in station.arms.values():
        arm.held = True
    station.env.step(np.zeros(14))
    press(station, "right_menu_button")
    station.arms["right"].buttons = {}
    held = station.base.get_hold_action().copy()
    resets = {name: arm.resets for name, arm in station.arms.items()}
    for arm in station.arms.values():
        arm.action[6] = 1.0
    kin = station.kinematics[side]
    failed_index = 0 if side == "left" else 7
    healthy_index = 7 - failed_index
    healthy_side = "right" if side == "left" else "left"
    kin.fail = True
    solves = len(kin.targets)
    for attempt in range(3):
        _, reward, done, _, info = station.env.step(np.zeros(14))
        assert len(kin.targets) == solves + attempt + 1
        assert info["yam_pico_fault"] == f"{side}:ik:injected_failure"
        assert info["pico_active"] and info[healthy_side] and not info[side]
        assert info["pre_record"] and reward == 0 and not done
        assert info["record_reset"] == (attempt == 0)
        np.testing.assert_allclose(
            info["intervene_action"][failed_index : failed_index + 7],
            held[failed_index : failed_index + 7],
        )
        assert info["intervene_action"][healthy_index] > held[healthy_index]
        assert info["intervene_action"][healthy_index + 6] == 1.0
        for name, arm in station.arms.items():
            assert arm.held and arm.reference
            assert arm.resets == resets[name]

    kin.fail = False
    for arm in station.arms.values():
        arm.action[6] = 0.0
    info = station.env.step(np.zeros(14))[-1]
    assert info["yam_pico_fault"] is None and info["pico_active"]
    assert info["intervene_action"][0] > held[0]
    assert info["intervene_action"][7] > held[7]
    # An open command from a rejected frame must not leak into the retry.
    assert info["intervene_action"][failed_index + 6] == held[failed_index + 6]
    assert info["intervene_action"][healthy_index + 6] == 1.0


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


def test_fault_stops_recording_without_claiming_success(station):
    press(station, "right_menu_button")
    station.arms["left"].ready = False
    _, reward, done, _, info = station.env.step(np.zeros(14))
    assert info["record_reset"] and info["pre_record"]
    assert not done and reward == 0.0


def test_recorded_action_uses_runtime_acceptance(station, monkeypatch):
    original = station.env.env.step

    def clipped(action):
        obs, reward, done, truncated, info = original(action)
        info["accepted_action"] = np.full(14, 0.123, dtype=np.float32)
        return obs, reward, done, truncated, info

    monkeypatch.setattr(station.env.env, "step", clipped)
    np.testing.assert_allclose(
        station.env.step(np.zeros(14))[-1]["intervene_action"], 0.123
    )


@pytest.mark.parametrize("joint_step", [0.05, 0.08])
def test_large_valid_ik_target_is_interpolated_and_recorded(monkeypatch, joint_step):
    station = build_station(max_joint_delta=joint_step)
    try:
        station.env.reset()
        station.env.step(np.zeros(14))
        station.arms["left"].held = True
        station.env.step(np.zeros(14))
        before = station.base.get_hold_action().copy()

        def large_target(target, seed, gripper):
            del target, gripper
            return YamIKResult(
                True, seed + np.array([0.2, 0.1, 0.05, 0.1, 0, -0.1]), 0.0, 0.0, 0.0
            )

        monkeypatch.setattr(station.kinematics["left"], "solve", large_target)
        info = station.env.step(np.zeros(14))[-1]
        assert info["yam_pico_fault"] is None and info["pico_active"]
        np.testing.assert_allclose(
            info["intervene_action"][:6],
            before[:6] + joint_step * np.array([1, 0.5, 0.25, 0.5, 0, -0.5]),
        )
        np.testing.assert_allclose(info["intervene_action"][6:], before[6:])
        np.testing.assert_array_equal(info["intervene_action"], info["accepted_action"])
    finally:
        station.env.close()


def test_real_pico_arm_lifecycle_preserves_calibration(monkeypatch):
    from rlinf.robotics.parts.teleop.yam_pico import _YamPicoArm
    from rlinf.robotics.parts.transports.pico import PicoExpert

    monkeypatch.setattr(PicoExpert, "start", lambda self: None)
    arm = _YamPicoArm(hand="right", calibration={"enabled": False})
    arm._expert._latest_data = {
        "right_controller": {
            "position": [0.0, 0.0, 0.0],
            "orientation": [0.0, 0.0, 0.0, 1.0],
            "grip": 1.0,
        },
        "buttons": {"A": True},
    }
    arm._expert._last_update_time = time.time()
    tcp = np.array([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0])
    scale = np.array([0.01, 0.1, 1.0])
    action, engaged, _ = arm.command(arm.read(), tcp, scale)
    assert engaged
    np.testing.assert_allclose(action[:6], 0)
    assert arm.read_buttons()["A"]
    arm._expert._calibrated = True
    arm.reset_reference()
    assert arm._expert._calibrated and arm._ref_tcp_pos is None
    action, _, _ = arm.command(arm.read(), tcp, scale)
    np.testing.assert_allclose(action[:6], 0)
    arm.stop()


def test_two_real_subscribers_receive_and_close_independently():
    zmq = pytest.importorskip("zmq")
    from rlinf.robotics.parts.transports.pico import PicoExpert

    # ``ipc://`` addresses are capped at ``sizeof(sun_path)`` (103 bytes), and
    # pytest's own ``tmp_path`` on macOS already exceeds that.
    directory = tempfile.mkdtemp(prefix="yam-pico-")
    address = f"ipc://{directory}/pico.ipc"
    context = zmq.Context()
    publisher = None
    experts = []
    try:
        publisher = context.socket(zmq.PUB)
        publisher.bind(address)
        for side in ("left", "right"):
            experts.append(
                PicoExpert(
                    zmq_addr=address,
                    hand=side,
                    timeout_ms=20,
                    calibration={"enabled": False},
                )
            )
        assert experts[0]._socket is not experts[1]._socket
        assert experts[0]._thread is not experts[1]._thread
        message = {
            "headset_pose": [0, 1, 0, 0, 0, 0, 1],
            "buttons": {"X": True},
            "left_controller": {
                "grip": 1,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
            },
            "right_controller": {
                "grip": 0,
                "position": [0, 0, 0],
                "orientation": [0, 0, 0, 1],
            },
        }
        deadline = time.monotonic() + 2
        while not all(e.ready for e in experts) and time.monotonic() < deadline:
            publisher.send_json(message)
            time.sleep(0.01)
        assert all(e.ready for e in experts)
        # Only the gripping hand is driven; each reader sees its own grip.
        assert experts[0].get_reading()["held"]
        assert not experts[1].get_reading()["held"]
        assert experts[0].get_buttons()["X"]
        threads = [e._thread for e in experts]
    finally:
        for expert in experts:
            expert.stop()
        if publisher is not None:
            publisher.close(linger=0)
        context.term()
        shutil.rmtree(directory, ignore_errors=True)
    assert all(not thread.is_alive() for thread in threads)


def test_runtime_rejection_aborts_recording_and_reports_accepted_hold(
    station, monkeypatch
):
    press(station, "right_menu_button")
    original = station.env.env.step

    def reject(action):
        result = original(station.base.get_hold_action())
        result[-1]["action_rejected"] = "measured_joint_out_of_limits"
        return result

    monkeypatch.setattr(station.env.env, "step", reject)
    info = station.env.step(np.zeros(14))[-1]
    assert info["record_reset"] and info["pre_record"]
    assert info["yam_pico_fault"].startswith("runtime:")
    np.testing.assert_array_equal(info["accepted_action"], info["intervene_action"])


def test_unexpected_control_exception_closes_followers_and_arms(station, monkeypatch):
    def fail(*args):
        raise RuntimeError("injected FK error")

    monkeypatch.setattr(station.kinematics["left"], "fk", fail)
    with pytest.raises(RuntimeError, match="injected FK error"):
        station.env.step(np.zeros(14))
    assert station.base._closed
    assert all(arm.stopped for arm in station.arms.values())


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


@pytest.mark.parametrize("write_to_disk", [False, True])
def test_two_lerobot_episodes_exclude_preview_and_discarded_frames(
    station, tmp_path, monkeypatch, write_to_disk
):
    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    collector = CollectEpisode(
        PackedObservations(station.env),
        str(tmp_path),
        export_format="lerobot",
        only_success=True,
    )
    episodes = []
    if write_to_disk:
        pytest.importorskip("lerobot")
        from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

        original_create = LeRobotDatasetWriter.create

        def create(self, **kwargs):
            kwargs.update(image_writer_processes=0, image_writer_threads=1)
            return original_create(self, **kwargs)

        monkeypatch.setattr(LeRobotDatasetWriter, "create", create)
        original_write = collector._write_lerobot_episode

        def write(episode):
            original_write(episode)
            episodes.append(episode)

        monkeypatch.setattr(collector, "_write_lerobot_episode", write)
    else:
        monkeypatch.setattr(collector, "_write_lerobot_episode", episodes.append)
    try:
        collector.reset()
        station.arms["left"].held = False
        collector.step(np.zeros(14))

        def edge(button):
            station.arms["right"].buttons = {}
            collector.step(np.zeros(14))
            station.arms["right"].buttons = {button: True}
            return collector.step(np.zeros(14))

        edge("right_menu_button")
        collector.step(np.zeros(14))
        edge("left_menu_button")
        assert collector._buffers[0]["actions"] == []
        for _ in range(2):
            edge("right_menu_button")
            before = station.base.get_hold_action().copy()
            station.arms["left"].held = True
            _, _, _, _, info = collector.step(np.zeros(14))
            expected = info["accepted_action"].copy()
            edge("right_menu_button")
            collector._wait_futures()
            frame = episodes[-1][0]
            np.testing.assert_allclose(frame["state"], before)
            np.testing.assert_allclose(frame["actions"], expected)
            assert frame["actions"].shape == (14,)
            assert frame["image"].shape == (8, 8, 3)
            assert "extra_view_image-1" in frame
            assert episodes[-1][-1]["done"].all()
            collector.reset()
        assert len(episodes) == 2
    finally:
        collector.close()
    if write_to_disk:
        import json

        import pyarrow.parquet as pq

        shard = tmp_path / "rank_0/id_0"
        metadata = json.loads((shard / "meta/info.json").read_text())
        assert metadata["total_episodes"] == 2
        tables = [
            pq.read_table(path) for path in sorted(shard.glob("data/**/*.parquet"))
        ]
        assert len(tables) == 2
        for table, episode in zip(tables, episodes, strict=True):
            np.testing.assert_allclose(
                table["actions"].to_pylist(), [frame["actions"] for frame in episode]
            )


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
        collector.step(np.zeros(14))
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
    assert {"image", "extra_view_image-0", "extra_view_image-1"} <= set(
        table.column_names
    )
