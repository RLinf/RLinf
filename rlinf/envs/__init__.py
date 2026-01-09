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


def get_env_cls(env_type, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.
    """
    if env_type == "maniskill":
        from rlinf.envs.maniskill.maniskill_env import ManiskillEnv

        return ManiskillEnv
    elif env_type == "libero":
        from rlinf.envs.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == "robotwin":
        from rlinf.envs.robotwin.RoboTwin_env import RoboTwin

        return RoboTwin
    elif env_type == "isaaclab":
        from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        if env_cfg is None:
            raise ValueError(
                "env_cfg is required for isaaclab environment type. "
                "Please provide env_cfg.init_params.id to select the task."
            )

        task_id = env_cfg.init_params.id
        assert task_id in REGISTER_ISAACLAB_ENVS, (
            f"Task type {task_id} has not been registered! "
            f"Available tasks: {list(REGISTER_ISAACLAB_ENVS.keys())}"
        )
        return REGISTER_ISAACLAB_ENVS[task_id]
    elif env_type == "metaworld":
        from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv
    elif env_type == "behavior":
        from rlinf.envs.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv
    elif env_type == "calvin":
        from rlinf.envs.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv
    elif env_type == "robocasa":
        from rlinf.envs.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    elif env_type == "realworld":
        from rlinf.envs.realworld.realworld_env import RealWorldEnv

        return RealWorldEnv
    elif env_type == "mujoco":
        from rlinf.envs.mujoco.serl_franka_lift_cube import SERLFrankaEnv
        return SERLFrankaEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
