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

import numpy as np
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.registration import register_env


@register_env("PickCube3View-v1", max_episode_steps=50)
class PickCube3ViewEnv(PickCubeEnv):
    """PickCube with native sensor_data containing a dedicated 3rd view camera."""

    @property
    def _default_sensor_configs(self):
        base_pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        third_pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return [
            CameraConfig("base_camera", base_pose, 128, 128, np.pi / 2, 0.01, 100),
            # Align 3rd-view sensor with PickCube's legacy human render camera setup.
            CameraConfig("3rd_view_camera", third_pose, 128, 128, 1.0, 0.01, 100),
        ]
