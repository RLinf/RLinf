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

"""Isaac Lab configurations for the Go2 flat, rough, and stairs tasks."""

import copy
import math

import isaaclab.envs.mdp as core_mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)


def _apply_flat_reward_fixes(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Set the stepping, posture, and torque reward weights for flat terrain."""
    rw = cfg.rewards
    rw.feet_air_time.weight = 0.25
    rw.flat_orientation_l2.weight = -2.5
    rw.dof_torques_l2.weight = -1.0e-5


@configclass
class Go2A2OmniWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """Track omnidirectional velocity commands on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        # flat terrain (height scanner stays for obs compatibility)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None

        # no curriculum on flat terrain
        self.curriculum.terrain_levels = None

        # fix rough-env reward defaults
        _apply_flat_reward_fixes(self)

        # omni-dir cmds
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.05
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)

        # always-on height incentive
        self.rewards.base_height = RewTerm(
            func=core_mdp.base_height_l2,
            weight=-4.0,
            params={
                "target_height": 0.40,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )


def _keep_single_sub_terrain(cfg: UnitreeGo2RoughEnvCfg, terrain_key: str) -> None:
    """Use one terrain type across the terrain generator's grid."""
    generator = cfg.scene.terrain.terrain_generator
    if generator is None:
        return
    if terrain_key not in generator.sub_terrains:
        raise KeyError(f"Unknown sub-terrain key: {terrain_key}")

    terrain_generator = copy.deepcopy(generator)
    terrain_generator.sub_terrains = {
        terrain_key: terrain_generator.sub_terrains[terrain_key]
    }
    cfg.scene.terrain.terrain_generator = terrain_generator


def _fix_rough_reward_shaping(cfg: UnitreeGo2RoughEnvCfg) -> None:
    """Set the rough-terrain velocity tracking, gait, and posture rewards."""
    rw = cfg.rewards

    # stepping reward
    rw.feet_air_time.weight = 0.25

    # velocity tracking
    rw.track_lin_vel_xy_exp.weight = 2.0
    rw.track_ang_vel_z_exp.weight = 1.0

    # penalize thigh and calf ground contacts
    rw.undesired_contacts = RewTerm(
        func=core_mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[".*_thigh", ".*_calf"]
            ),
            "threshold": 1.0,
        },
    )

    # upright posture incentive
    rw.flat_orientation_l2 = RewTerm(func=core_mdp.flat_orientation_l2, weight=-5.0)

    # Measure the target height relative to the terrain beneath the base.
    rw.base_height = RewTerm(
        func=core_mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.38,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )

    rw.dof_torques_l2.weight = -1.0e-5


@configclass
class Go2B1RoughWalkEnvCfg(UnitreeGo2RoughEnvCfg):
    """Track omnidirectional velocity commands on random rough terrain."""

    def __post_init__(self):
        super().__post_init__()
        _keep_single_sub_terrain(self, "random_rough")

        # B1 uses the same omni command profile as A2
        cmd = self.commands.base_velocity
        cmd.heading_command = True
        cmd.rel_heading_envs = 1.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (-1.0, 1.0)
        cmd.ranges.lin_vel_y = (-1.0, 1.0)
        cmd.ranges.ang_vel_z = (-1.0, 1.0)
        cmd.ranges.heading = (-math.pi, math.pi)

        _fix_rough_reward_shaping(self)


@configclass
class Go2B2StairClimbEnvCfg(UnitreeGo2RoughEnvCfg):
    """Track forward velocity commands on uphill stairs."""

    def __post_init__(self):
        super().__post_init__()

        # use inverted stairs only (forward ~= uphill from spawn)
        gen = copy.deepcopy(self.scene.terrain.terrain_generator)
        gen.sub_terrains = {
            "pyramid_stairs_inv": gen.sub_terrains["pyramid_stairs_inv"],
        }
        gen.sub_terrains["pyramid_stairs_inv"].proportion = 1.0
        gen.sub_terrains["pyramid_stairs_inv"].step_height_range = (0.04, 0.16)
        self.scene.terrain.terrain_generator = gen

        self.scene.terrain.max_init_terrain_level = 1

        # straight-forward only
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.4, 0.9)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.ranges.heading = (0.0, 0.0)

        # keep reset heading close to forward
        self.events.reset_base.params["pose_range"]["yaw"] = (-0.2, 0.2)

        rw = self.rewards
        rw.flat_orientation_l2.weight = -2.0
        rw.track_lin_vel_xy_exp.weight = 1.5
        rw.track_lin_vel_xy_exp.params["std"] = 0.5
        rw.track_ang_vel_z_exp.weight = 0.75
        rw.track_ang_vel_z_exp.params["std"] = 0.5

        rw.base_height = RewTerm(
            func=core_mdp.base_height_l2,
            weight=-3.5,
            params={
                "target_height": 0.42,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # discourage extreme joint angles
        rw.dof_pos_limits.weight = -2.0

        rw.feet_air_time.weight = 0.04
        rw.feet_air_time.params["threshold"] = 0.35
        rw.dof_torques_l2.weight = -2.0e-5
        rw.dof_acc_l2.weight = -2.5e-7
        rw.action_rate_l2.weight = -0.003

        rw.undesired_contacts = RewTerm(
            func=core_mdp.undesired_contacts,
            weight=-0.7,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*_thigh", ".*_calf"]
                ),
                "threshold": 1.0,
            },
        )
