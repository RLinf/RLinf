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

"""State transfer between LIBERO environments with the same task model."""

import copy
import random

import numpy as np

_SIM_INPUTS = (
    "ctrl",
    "qacc_warmstart",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "userdata",
)
_SIM_MODEL = ("body_pos", "body_quat", "site_rgba", "geom_rgba")
_OBSERVABLE_STATE = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)


def capture_snapshot(env) -> dict:
    """Capture physics, controller and observation state without simulator handles."""
    rob = env.env
    robots = []
    for robot in rob.robots:
        robots.append(
            {
                "controller": copy.deepcopy(
                    {k: v for k, v in vars(robot.controller).items() if k != "sim"}
                ),
                "history": copy.deepcopy(
                    {
                        k: v
                        for k, v in vars(robot).items()
                        if k.startswith("recent_") or k == "torques"
                    }
                ),
                "gripper_action": robot.gripper.current_action.copy(),
            }
        )
    return {
        "sim_state": env.get_sim_state().copy(),
        "sim_inputs": {k: getattr(rob.sim.data, k).copy() for k in _SIM_INPUTS},
        "sim_model": {k: getattr(rob.sim.model, k).copy() for k in _SIM_MODEL},
        "robots": robots,
        "clock": (rob.timestep, rob.cur_time, rob.done),
        "obs_cache": copy.deepcopy(rob._obs_cache),
        "observables": {
            name: {k: copy.deepcopy(getattr(obs, k)) for k in _OBSERVABLE_STATE}
            for name, obs in rob._observables.items()
        },
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def restore_snapshot(env, snapshot: dict) -> dict:
    """Restore a snapshot into the same LIBERO task and return its observation."""
    rob = env.env
    for key, value in snapshot["sim_model"].items():
        getattr(rob.sim.model, key)[:] = value
    env.set_state(snapshot["sim_state"])
    for key, value in snapshot["sim_inputs"].items():
        getattr(rob.sim.data, key)[:] = value
    rob.sim.forward()
    rob.sim.data.qacc_warmstart[:] = snapshot["sim_inputs"]["qacc_warmstart"]
    for robot, state in zip(rob.robots, snapshot["robots"]):
        vars(robot.controller).update(copy.deepcopy(state["controller"]))
        vars(robot).update(copy.deepcopy(state["history"]))
        robot.gripper.current_action = state["gripper_action"].copy()
    rob.timestep, rob.cur_time, rob.done = snapshot["clock"]
    rob._obs_cache = copy.deepcopy(snapshot["obs_cache"])
    for name, state in snapshot["observables"].items():
        vars(rob._observables[name]).update(copy.deepcopy(state))
    np.random.set_state(snapshot["numpy_rng"])
    random.setstate(snapshot["python_rng"])
    env._post_process()
    return rob._get_observations()
