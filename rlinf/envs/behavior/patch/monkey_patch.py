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

import omnigibson.lazy as lazy
from omnigibson.learning.eval import Evaluator
from omnigibson.sensors.vision_sensor import VisionSensor, render
from omnigibson.utils.usd_utils import ControllableObjectViewAPI

from rlinf.envs.behavior.instance_loader import load_cached_activity_instance


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
        load_cached_activity_instance(
            self.env,
            instance_id=instance_id,
            reset_scene=True,
        )

    ControllableObjectViewAPI._get_pattern_from_prim_path = classmethod(
        _get_pattern_from_prim_path
    )
    ControllableObjectViewAPI.__rlinf_patched__ = True
    Evaluator.load_task_instance = _load_task_instance

    # OmniGibson / Replicator can fail when detaching annotators with an explicit
    # list path during dynamic camera resize. Use the same detach style as
    # VisionSensor._remove_modality_from_backend for compatibility.

    import omnigibson as og

    if (
        getattr(VisionSensor, "__rlinf_resize_setter_patched__", False)
        or og.__version__ != "3.7.2"
    ):
        return

    original_image_height = VisionSensor.image_height
    original_image_width = VisionSensor.image_width

    def _detach_annotator_safe(annotator, render_product):
        try:
            annotator.detach(render_product)
        except Exception:
            # Fallback for older backends that still expect explicit path lists.
            annotator.detach([render_product.path])

    def _reset_render_product(sensor, width: int, height: int):
        if sensor._viewport is not None:
            sensor._viewport.viewport_api.set_texture_resolution((width, height))
        sensor._image_width = width
        sensor._image_height = height

        for annotator in sensor._annotators.values():
            if annotator is None:
                continue
            _detach_annotator_safe(annotator, sensor._render_product)

        sensor._render_product.destroy()
        sensor._render_product = lazy.omni.replicator.core.create.render_product(
            sensor.prim_path, (width, height), force_new=True
        )

        for annotator in sensor._annotators.values():
            if annotator is None:
                continue
            annotator.attach([sensor._render_product])

        for _ in range(3):
            render()

    def _patched_image_height(self, height):
        if self._viewport is not None:
            width, _ = self._viewport.viewport_api.get_texture_resolution()
        else:
            width = self._image_width
        _reset_render_product(self, width=width, height=height)

    def _patched_image_width(self, width):
        if self._viewport is not None:
            _, height = self._viewport.viewport_api.get_texture_resolution()
        else:
            height = self._image_height
        _reset_render_product(self, width=width, height=height)

    VisionSensor.image_height = property(
        original_image_height.fget,
        _patched_image_height,
        original_image_height.fdel,
        original_image_height.__doc__,
    )
    VisionSensor.image_width = property(
        original_image_width.fget,
        _patched_image_width,
        original_image_width.fdel,
        original_image_width.__doc__,
    )
    VisionSensor.__rlinf_resize_setter_patched__ = True
