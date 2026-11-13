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

"""LIBERO snapshot transfer at the external simulator boundary."""

import pickle
import random
from types import SimpleNamespace

import numpy as np

from rlinf.envs.sim.libero.snapshot import capture_snapshot, restore_snapshot


class _Simulator:
    def __init__(self):
        self.state = np.arange(4, dtype=float)
        self.model = SimpleNamespace(
            body_pos=np.ones((2, 3)),
            body_quat=np.zeros((2, 4)),
            site_rgba=np.ones((1, 4)),
            geom_rgba=np.ones((2, 4)),
        )
        self.data = SimpleNamespace(
            ctrl=np.ones(2),
            qacc_warmstart=np.full(2, 3.0),
            qfrc_applied=np.zeros(2),
            xfrc_applied=np.zeros((2, 6)),
            mocap_pos=np.zeros((0, 3)),
            mocap_quat=np.zeros((0, 4)),
            userdata=np.zeros(0),
        )

    def __deepcopy__(self, memo):
        raise TypeError("Simulator handles cannot be copied between environments")

    def forward(self):
        self.data.qacc_warmstart[:] = 0


class _ControlEnv:
    def __init__(self):
        sim = _Simulator()
        self.env = SimpleNamespace(
            sim=sim,
            robots=[
                SimpleNamespace(
                    controller=SimpleNamespace(sim=sim, goal=np.array([0.5, 1.0])),
                    recent_actions=SimpleNamespace(buf=np.ones((2, 2))),
                    gripper=SimpleNamespace(current_action=np.array([1.0])),
                    torques=np.ones(2),
                )
            ],
            timestep=8,
            cur_time=0.4,
            done=False,
            _obs_cache={"robot0_eef_pos": np.array([0.1, 0.2, 0.3])},
            _observables={
                "camera": SimpleNamespace(
                    _time_since_last_sample=0.0,
                    _current_delay=0.0,
                    _current_observed_value=np.full((2, 2, 3), 17, dtype=np.uint8),
                    _sampled=True,
                )
            },
        )
        self.env._get_observations = lambda: {
            "image": self.env._observables["camera"]._current_observed_value.copy(),
            "position": self.env._obs_cache["robot0_eef_pos"].copy(),
        }

    def get_sim_state(self):
        return self.env.sim.state

    def set_state(self, state):
        self.env.sim.state = state.copy()

    def _post_process(self):
        pass


def test_libero_snapshot_clones_runtime_state_without_simulator_aliases():
    source = _ControlEnv()
    snapshot = pickle.loads(pickle.dumps(capture_snapshot(source)))
    expected_rng = (np.random.rand(), random.random())
    targets = [_ControlEnv(), _ControlEnv()]
    for target in targets:
        target.env.sim.state[:] = -1
        target.env.sim.model.body_pos[:] = -1
        target.env.robots[0].controller.goal[:] = -1
        target.env.timestep = 99
        target.env._obs_cache.clear()
        target.env._observables["camera"]._current_observed_value[:] = 0
        observation = restore_snapshot(target, snapshot)
        np.testing.assert_equal(target.get_sim_state(), source.get_sim_state())
        np.testing.assert_equal(target.env.sim.model.body_pos, 1)
        np.testing.assert_equal(target.env.sim.data.qacc_warmstart, [3.0, 3.0])
        np.testing.assert_equal(target.env.robots[0].controller.goal, [0.5, 1.0])
        np.testing.assert_equal(observation["image"], 17)
        np.testing.assert_equal(observation["position"], [0.1, 0.2, 0.3])
        assert target.env.robots[0].controller.sim is target.env.sim
        assert (target.env.timestep, target.env.cur_time, target.env.done) == (
            8,
            0.4,
            False,
        )
        assert (np.random.rand(), random.random()) == expected_rng
    targets[0].env.robots[0].controller.goal[:] = 9
    targets[0].env.robots[0].recent_actions.buf[:] = 9
    targets[0].env.robots[0].gripper.current_action[:] = 9
    np.testing.assert_equal(targets[1].env.robots[0].controller.goal, [0.5, 1.0])
    np.testing.assert_equal(targets[1].env.robots[0].recent_actions.buf, 1)
    np.testing.assert_equal(targets[1].env.robots[0].gripper.current_action, 1)
