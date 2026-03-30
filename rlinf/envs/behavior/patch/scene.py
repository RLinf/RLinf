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

import omnigibson as og
import torch as th
from omnigibson.scenes.interactive_traversable_scene import InteractiveTraversableScene
from omnigibson.scenes.scene_base import create_object_from_init_info
from omnigibson.utils.python_utils import recursively_convert_to_torch


class RLinfInteractiveTraversableScene(InteractiveTraversableScene):
    def restore(self, scene_file, update_initial_file=False):
        assert not og.sim.is_stopped(), (
            "Simulator cannot be stopped when restoring scene!"
        )

        if isinstance(scene_file, str):
            if not scene_file.endswith(".json"):
                raise ValueError(f"Expected a json scene file, got {scene_file}")
            with open(scene_file, "r") as f:
                scene_info = json.load(f)
        else:
            scene_info = scene_file

        init_info = scene_info["init_info"]
        state = scene_info["state"]
        if not th.is_tensor(state["pos"]):
            state = recursively_convert_to_torch(state)

        for key, data in scene_info.get("metadata", {}).items():
            self.write_task_metadata(key=key, data=data)

        if init_info["class_name"] not in {
            self.__class__.__name__,
            InteractiveTraversableScene.__name__,
        }:
            raise ValueError(
                f"Got mismatch in scene type: current is {self.__class__.__name__}, "
                f"trying to load {init_info['class_name']}"
            )

        current_systems = set(self.active_systems.keys())
        load_systems = set(scene_info["state"]["registry"]["system_registry"].keys())
        for name in current_systems - load_systems:
            self.clear_system(name)
        for name in load_systems - current_systems:
            self.get_system(name, force_init=True)

        current_obj_names = set(self.object_registry.get_dict("name").keys())
        load_obj_names = set(scene_info["objects_info"]["init_info"].keys())

        objects_to_remove = [
            self.object_registry("name", name)
            for name in current_obj_names - load_obj_names
        ]
        og.sim.batch_remove_objects(objects_to_remove)

        objects_to_add = [
            create_object_from_init_info(scene_info["objects_info"]["init_info"][name])
            for name in load_obj_names - current_obj_names
        ]
        og.sim.batch_add_objects(objects_to_add, scenes=[self] * len(objects_to_add))

        self.load_state(state, serialized=False)

        if update_initial_file:
            self.update_initial_file(scene_file=scene_info)
