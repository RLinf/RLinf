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

import warnings
from enum import Enum

from rlinf.utils.robosuite_compat import install_robosuite_egl_device_shim

# Must run before any simulator is imported, in worker processes and in the
# simulator subprocesses they spawn alike. See ``rlinf.utils.robosuite_compat``.
install_robosuite_egl_device_shim()


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    MANISKILL_RLT = "maniskill_rlt"
    LIBERO = "libero"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    ROBOCASA365 = "robocasa365"
    REAL = "real"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    OPENSORAWM = "opensora_wm"
    WANWM = "wan_wm"
    GENESIS = "genesis"
    EMBODICHAIN = "embodichain"
    ROBOVERSE = "roboverse"
    D4RL = "d4rl"
    DIFFUSION = "diffusion"
    POLARIS = "polaris"

    @classmethod
    def _missing_(cls, value: object) -> "SupportedEnvType | None":
        """Accept the spellings this enum used to have.

        The real-world package was ``envs/realworld`` and is now ``envs/real``;
        configs that named the old one keep working and say so.
        """
        if value == "realworld":
            warnings.warn(
                "env_type 'realworld' is retired. Use 'real'.",
                DeprecationWarning,
                stacklevel=2,
            )
            return cls.REAL
        return None


def get_env_cls(env_type: str, env_cfg=None):
    """
    Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g., "maniskill", "libero", "isaaclab", etc.)
        env_cfg: Optional environment configuration. Required for "isaaclab" environment type.

    Returns:
        Environment class corresponding to the environment type.
    """

    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        if env_cfg.get("enable_offload", False):
            from rlinf.envs.sim.maniskill.maniskill_offload_env import (
                ManiskillOffloadEnv,
            )

            return ManiskillOffloadEnv
        else:
            from rlinf.envs.sim.maniskill.maniskill_env import ManiskillEnv

            return ManiskillEnv
    elif env_type == SupportedEnvType.MANISKILL_RLT:
        from rlinf.envs.sim.maniskill.maniskill_rlt_env import ManiskillRLTEnv

        return ManiskillRLTEnv
    elif env_type == SupportedEnvType.LIBERO:
        from rlinf.envs.sim.libero.libero_env import LiberoEnv

        return LiberoEnv
    elif env_type == SupportedEnvType.ROBOTWIN:
        from rlinf.envs.sim.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv
    elif env_type == SupportedEnvType.ISAACLAB:
        from rlinf.envs.sim.isaaclab import REGISTER_ISAACLAB_ENVS

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
    elif env_type == SupportedEnvType.METAWORLD:
        from rlinf.envs.sim.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv
    elif env_type == SupportedEnvType.BEHAVIOR:
        from rlinf.envs.sim.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv
    elif env_type == SupportedEnvType.CALVIN:
        from rlinf.envs.sim.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv
    elif env_type == SupportedEnvType.ROBOCASA:
        from rlinf.envs.sim.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv
    elif env_type == SupportedEnvType.ROBOCASA365:
        from rlinf.envs.sim.robocasa365.robocasa365_env import Robocasa365Env

        return Robocasa365Env
    elif env_type == SupportedEnvType.REAL:
        from rlinf.envs.real import RealWorldEnv

        return RealWorldEnv
    elif env_type == SupportedEnvType.HABITAT:
        from rlinf.envs.sim.habitat.habitat_env import HabitatEnv

        return HabitatEnv
    elif env_type == SupportedEnvType.FRANKASIM:
        from rlinf.envs.sim.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv
    elif env_type == SupportedEnvType.GENESIS:
        from rlinf.envs.sim.genesis.genesis_env import GenesisEnv

        return GenesisEnv
    elif env_type == SupportedEnvType.OPENSORAWM:
        from rlinf.envs.sim.world_model.world_model_opensora_env import OpenSoraEnv

        return OpenSoraEnv
    elif env_type == SupportedEnvType.WANWM:
        from rlinf.envs.sim.world_model.world_model_wan_env import WanEnv

        return WanEnv
    elif env_type == SupportedEnvType.EMBODICHAIN:
        from rlinf.envs.sim.embodichain.embodichain_env import EmbodiChainEnv

        return EmbodiChainEnv
    elif env_type == SupportedEnvType.ROBOVERSE:
        from rlinf.envs.sim.roboverse.roboverse_env import RoboVerseEnv

        return RoboVerseEnv
    elif env_type == SupportedEnvType.D4RL:
        from rlinf.envs.sim.d4rl.d4rl_env import D4RLEnv

        return D4RLEnv
    elif env_type == SupportedEnvType.DIFFUSION:
        from rlinf.envs.sim.diffusion import GenerationEnv

        return GenerationEnv
    elif env_type == SupportedEnvType.POLARIS:
        from rlinf.envs.sim.polaris.polaris_env import PolarisEnv

        return PolarisEnv
    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
