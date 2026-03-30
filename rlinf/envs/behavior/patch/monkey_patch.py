# Copyright 2025 The RLinf Authors.
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
import os

import omnigibson as og
from omnigibson.learning.eval import Evaluator
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omnigibson.utils.usd_utils import ControllableObjectViewAPI


def apply() -> None:
    if getattr(ControllableObjectViewAPI, "__rlinf_patched__", False):
        return

    def _get_pattern_from_prim_path(cls, prim_path):
        scene_id, robot_name = prim_path.split("/")[2:4]
        if not scene_id.startswith("scene_"):
            raise ValueError(f"Unexpected prim path: {prim_path}")
        prefix, robot_type, _ = robot_name.split("__")
        return prim_path.replace(f"/{robot_name}", f"/{prefix}__{robot_type}__*")

    def _load_task_instance(self, instance_id: int) -> None:
        scene_model = self.env.task.scene_name
        tro_filename = self.env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=self.env.task.activity_name,
            activity_definition_id=self.env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{self.env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        for tro_key, state in tro_state.items():
            if tro_key == "robot_poses":
                robot_pose = state[self.robot.model_name][0]
                self.robot.set_position_orientation(
                    robot_pose["position"],
                    robot_pose["orientation"],
                    frame="scene",
                )
                self.env.scene.write_task_metadata(key=tro_key, data=state)
            else:
                self.env.task.object_scope[tro_key].load_state(state, serialized=False)

        for _ in range(25):
            og.sim.step_physics()
            for entity in self.env.task.object_scope.values():
                if entity.exists and not entity.is_system:
                    entity.keep_still()

    ControllableObjectViewAPI._get_pattern_from_prim_path = classmethod(
        _get_pattern_from_prim_path
    )
    ControllableObjectViewAPI.__rlinf_patched__ = True
    Evaluator.load_task_instance = _load_task_instance
