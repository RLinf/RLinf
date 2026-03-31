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

import random

import omnigibson as og
from omnigibson.objects.dataset_object import DatasetObject
from omnigibson.tasks.behavior_task import BehaviorTask
from omnigibson.tasks.task_base import BaseTask
from omnigibson.utils.ui_utils import create_module_logger

log = create_module_logger(module_name=__name__)


class RLinfBehaviorTask(BehaviorTask):
    def _ensure_callback_state(self):
        if not hasattr(self, "_callback_name"):
            self._callback_name = None
        if not hasattr(self, "_callback_scene"):
            self._callback_scene = None
        if not hasattr(self, "_ignored_cross_scene_callbacks"):
            self._ignored_cross_scene_callbacks = set()

    def _load(self, env):
        self._ensure_callback_state()
        self.update_activity(
            env=env,
            activity_name=self.activity_name,
            activity_definition_id=self.activity_definition_id,
            predefined_problem=self.predefined_problem,
        )

        _, self.feedback = self.initialize_activity(env=env)
        self.scene_name = getattr(env.scene, "scene_model", None)
        self._ignored_cross_scene_callbacks.clear()

        if self.highlight_task_relevant_objs:
            for entity in self.object_scope.values():
                if entity.synset == "agent":
                    continue
                if not entity.is_system and entity.exists:
                    entity.highlighted = True

        self._callback_scene = env.scene
        self._callback_name = f"{self.activity_name}_scene_{env.scene.idx}_refresh"
        og.sim.add_callback_on_add_obj(
            name=self._callback_name,
            callback=self._update_bddl_scope_from_added_obj,
        )
        og.sim.add_callback_on_remove_obj(
            name=self._callback_name,
            callback=self._update_bddl_scope_from_removed_obj,
        )
        og.sim.add_callback_on_system_init(
            name=self._callback_name,
            callback=self._update_bddl_scope_from_system_init,
        )
        og.sim.add_callback_on_system_clear(
            name=self._callback_name,
            callback=self._update_bddl_scope_from_system_clear,
        )

    def reset(self, env):
        BaseTask.reset(self, env)

        if self.use_presampled_robot_pose:
            robot = self.get_agent(env)
            presampled_poses = env.scene.get_task_metadata(key="robot_poses")
            assert robot.model_name in presampled_poses, (
                f"{robot.model_name} presampled pose is not found in task metadata; "
                "please set use_presampled_robot_pose to False in task config"
            )
            poses = presampled_poses[robot.model_name]
            robot_pose = (
                random.choice(poses) if self.randomize_presampled_pose else poses[0]
            )
            robot.set_position_orientation(
                robot_pose["position"],
                robot_pose["orientation"],
                frame="scene",
            )

        for obj in self.object_scope.values():
            if obj.exists and isinstance(obj, DatasetObject):
                obj.wake()

    def _update_bddl_scope_from_added_obj(self, obj):
        if not self._callback_matches_scene(obj, "object.add"):
            return
        for entity in self.object_scope.values():
            if (
                not entity.exists
                and not entity.is_system
                and obj.category in set(entity.og_categories)
            ):
                entity.set_entity(entity=obj)
                return

    def _update_bddl_scope_from_removed_obj(self, obj):
        if not self._callback_matches_scene(obj, "object.remove"):
            return
        for entity in self.object_scope.values():
            if entity.exists and not entity.is_system and obj.name == entity.name:
                entity.clear_entity()
                return

    def _update_bddl_scope_from_system_init(self, system):
        if not self._callback_matches_scene(system, "system.init"):
            return
        for entity in self.object_scope.values():
            if (
                not entity.exists
                and entity.is_system
                and entity.og_categories[0] == system.name
            ):
                entity.set_entity(entity=system)
                return

    def _update_bddl_scope_from_system_clear(self, system):
        if not self._callback_matches_scene(system, "system.clear"):
            return
        for entity in self.object_scope.values():
            if entity.exists and entity.is_system and system.name == entity.name:
                entity.clear_entity()
                return

    def _callback_matches_scene(self, source, source_type):
        self._ensure_callback_state()
        if self._callback_scene is None:
            return True

        source_scene = getattr(source, "scene", None)
        if source_scene is self._callback_scene:
            return True

        key = (source_type, getattr(source_scene, "idx", None))
        if key not in self._ignored_cross_scene_callbacks:
            self._ignored_cross_scene_callbacks.add(key)
            log.warning(
                "Ignoring cross-scene %s callback for task %s: expected scene %s, got scene %s",
                source_type,
                self.activity_name,
                self._callback_scene.idx,
                getattr(source_scene, "idx", None),
            )
        return False
