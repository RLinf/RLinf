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

"""Galaxea A1 叠衣服任务(A1 原生空间):环境 → 主臂镜像介入 → 键盘控制面板。"""

from typing import Any, Mapping

import os
import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.a1.a1_env import A1DualEnv, A1DualRobotConfig


def create_a1_fold_env(
    override_cfg: Mapping[str, Any],
    worker_info=None,
    hardware_info=None,
    env_idx: int = 0,
    env_cfg: Mapping[str, Any] | None = None,
) -> gym.Env:
    import dataclasses as _dc
    _known = {f.name for f in _dc.fields(A1DualRobotConfig)}
    _cfg = {k: v for k, v in dict(override_cfg).items() if k in _known}   # 防御:忽略不属于本 dataclass 的键
    config = A1DualRobotConfig(**_cfg)
    env = A1DualEnv(config, worker_info, hardware_info, env_idx)
    env_cfg = env_cfg or {}
    if not config.is_dummy:
        if env_cfg.get("use_piper_intervention", True):
            from rlinf.envs.realworld.a1.piper_intervention import A1PiperMirrorIntervention
            env = A1PiperMirrorIntervention(env, swap_pedals=bool(env_cfg.get("swap_pedals", False)))
        from rlinf.envs.realworld.a1.keyboard_control import A1KeyboardControl
        env = A1KeyboardControl(env, milestones=int(env_cfg.get("milestones", 5)),
                                stage_prompts=_load_stage_prompts(env_cfg))
    elif int(os.environ.get("RLT_DUMMY_STAGE_EVERY", "0")) > 0:
        # 假数据测试阶段标注链路:同一个键盘 wrapper,按键由脚本按固定步数自动产生
        from rlinf.envs.realworld.a1.keyboard_control import A1KeyboardControl
        env = A1KeyboardControl(env, milestones=int(env_cfg.get("milestones", 5)),
                                script_every=int(os.environ["RLT_DUMMY_STAGE_EVERY"]),
                                stage_prompts=_load_stage_prompts(env_cfg))
    if os.environ.get("RLT_RECORD_DIR"):
        # 带图像的原始数据录制(后台线程 + ffmpeg 子进程),最外层才能看到介入/阶段/结局信息
        from rlinf.envs.realworld.a1.episode_recorder import A1EpisodeRecorder
        env = A1EpisodeRecorder(env, os.environ["RLT_RECORD_DIR"], fps=config.step_frequency,
                                task=getattr(config, "task_description", ""))
    return env


def _load_stage_prompts(env_cfg):
    """env.stage_prompts_file(stage_prompts_14.json,stages[i].en)→ 每阶段一条英文提示词;未配置返回 None。"""
    path = (env_cfg or {}).get("stage_prompts_file")
    if not path:
        return None
    import json
    with open(path) as f:
        return [s["en"] for s in json.load(f)["stages"]]


register(
    id="A1FoldShirtEnv-v1",
    entry_point="rlinf.envs.realworld.a1.tasks:create_a1_fold_env",
)

# 叠盒子复用同一个构造函数,区别全在配置:reset_qpos(起始姿态)、task_description、
# step_frequency(必须等于训练数据帧率,box 是 30fps)、milestones(8 个阶段)。
register(
    id="A1StackBoxEnv-v1",
    entry_point="rlinf.envs.realworld.a1.tasks:create_a1_fold_env",
)


def create_a1_playback_env(override_cfg, worker_info=None, hardware_info=None, env_idx: int = 0, env_cfg=None):
    from rlinf.envs.realworld.a1.playback_env import A1PlaybackConfig, A1PlaybackEnv
    return A1PlaybackEnv(A1PlaybackConfig(**dict(override_cfg)), worker_info, hardware_info, env_idx)


register(
    id="A1PlaybackEnv-v1",
    entry_point="rlinf.envs.realworld.a1.tasks:create_a1_playback_env",
)
