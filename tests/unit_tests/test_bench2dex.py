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

import math

import pytest
import torch
import yaml

from rlinf.envs.sim.bench2dex.robots import JointTargets
from rlinf.envs.sim.bench2dex.success import TerminalSuccess


def _state(x=0.0, y=0.0, z=0.0, **kwargs):
    return {
        "pose_world": [x, y, z, 0.0, 0.0, 0.0, 1.0],
        "lin_vel_world": [0.0, 0.0, 0.0],
        "ang_vel_world": [0.0, 0.0, 0.0],
        **kwargs,
    }


def _tracker(condition, dwell=0.0, dt=1 / 60):
    return TerminalSuccess(
        {"terminal": {"raw_condition": condition, "dwell_time_s": dwell}}, dt=dt
    )


def _custom(task, **params):
    return {"type": "custom", "evaluator": f"success.custom.{task}", "params": params}


def test_terminal_requires_consecutive_dwell_and_latches_success():
    tracker = _tracker(
        {"type": "object_in_zone", "object": "cube", "aabb": [[0, 0, 0], [1, 1, 1]]},
        dwell=0.5,
    )
    inside = {"cube": _state(0.5, 0.5, 0.5)}
    outside = {"cube": _state(2.0, 0.5, 0.5)}
    for _ in range(29):
        tracker.update(inside)
    assert tracker.instant_success
    assert not tracker.success
    tracker.update(outside)
    assert not tracker.instant_success
    for _ in range(29):
        tracker.update(inside)
    assert not tracker.success
    tracker.update(inside)
    assert tracker.success
    tracker.update(outside)
    assert not tracker.instant_success
    assert tracker.success


@pytest.mark.parametrize(
    "terminal, message",
    [
        ({"condition": {"type": "all", "conditions": []}}, "raw_condition"),
        (
            {"raw_condition": {"type": "all", "conditions": []}, "dwell_time_s": -1},
            "non-negative",
        ),
    ],
)
def test_terminal_validates_configuration(terminal, message):
    with pytest.raises(ValueError, match=message):
        TerminalSuccess({"terminal": terminal})


def test_joint_targets_follow_simulation_order_and_expand_mimic_chains(tmp_path):
    dof_map = tmp_path / "active_dof_maps.yml"
    dof_map.write_text(
        yaml.safe_dump(
            {
                "robots": {
                    "test_robot": {
                        "full_joint_names": ["left", "right", "finger", "tip"],
                        "active_joint_names": ["left", "right"],
                        "mimic_dof": 2,
                        "urdf_path": "Bench2Dex/robot.urdf",
                    }
                }
            }
        )
    )
    (tmp_path / "robot.urdf").write_text(
        '<robot name="test_robot">'
        '<joint name="tip"><mimic joint="finger" multiplier="0.5" /></joint>'
        '<joint name="finger"><mimic joint="left" multiplier="-2" offset="0.1" />'
        "</joint></robot>"
    )
    joints = JointTargets(
        "test_robot", dof_map, tmp_path, ["right", "tip", "left", "finger"]
    )
    actions = torch.tensor([[0.2, 0.3]])
    torch.testing.assert_close(
        joints.expand(actions), torch.tensor([[0.2, -0.25, 0.3, -0.5]])
    )


def test_container_zone_uses_rotated_container_frame_and_offset():
    tracker = _tracker(
        {
            "type": "object_in_container_zone",
            "object": "cube",
            "container": "box",
            "container_frame": True,
            "container_center_offset": [0.1, 0.0, 0.0],
            "zone_lo": [0.05, -0.01, -0.01],
            "zone_hi": [0.15, 0.01, 0.01],
        }
    )
    states = {
        "cube": _state(0.2, 0.0),
        "box": _state(
            pose_world=[0.0, 0.0, 0.0, 0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]
        ),
    }
    tracker.update(states)
    assert not tracker.success
    states["cube"] = _state(0.0, 0.2)
    tracker.update(states)
    assert tracker.success


def test_piano_melodies_require_releasing_repeated_keys():
    tracker = _tracker(_custom("task_79_bimanual_piano_melody"))
    states = {"obj_325_piano_1": _state(qpos={})}
    qpos = states["obj_325_piano_1"]["qpos"]
    left = (18, 18, 23, 23, 24, 24, 23)
    right = (33, 33, 38, 38, 39, 39, 38)
    qpos.update({"joint_18_to_1": 0.005, "joint_33_to_1": 0.005})
    for _ in range(9):
        tracker.update(states)
    assert not tracker.success
    for i, (left_note, right_note) in enumerate(zip(left[1:], right[1:])):
        qpos.clear()
        tracker.update(states)
        qpos.update(
            {
                f"joint_{left_note}_to_1": 0.005,
                f"joint_{right_note}_to_1": 0.005,
            }
        )
        tracker.update(states)
        assert tracker.success == (i == len(left) - 2)


def test_task_03_matches_glasses_to_distinct_targets_and_requires_upright_static():
    tracker = _tracker(_custom("task_03_wine_glass_plate_balance"))
    states = {
        f"obj_194_wineglass_{i}": _state(
            pose_world=[x, 0.25, 0.8, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
        )
        for i, x in enumerate((-0.42, 0.58, 0.08), 1)
    }
    states["obj_194_wineglass_1"]["lin_vel_world"] = [0.1, 0.0, 0.0]
    tracker.update(states)
    assert not tracker.success
    states["obj_194_wineglass_1"]["lin_vel_world"] = [0.0, 0.0, 0.0]
    states["obj_194_wineglass_1"]["pose_world"][3:] = [0.0, 0.0, 0.0, 1.0]
    tracker.update(states)
    assert not tracker.success
    states["obj_194_wineglass_1"]["pose_world"][3:] = [
        math.sqrt(0.5),
        0.0,
        0.0,
        math.sqrt(0.5),
    ]
    states["obj_194_wineglass_1"]["pose_world"][0] = 0.58
    tracker.update(states)
    assert not tracker.success
    states["obj_194_wineglass_1"]["pose_world"][0] = -0.42
    tracker.update(states)
    assert tracker.success
