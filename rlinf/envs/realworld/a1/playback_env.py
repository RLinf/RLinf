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

"""离线回放环境:把 LeRobot v3 数据集(HIL 采集)按段回放给 RLinf rollout 管线,
用真实的 rollout→transition→replay 代码路径生成 RLT demo 预填数据。

- 观测 = 数据集帧(三相机 + 14 维状态),忽略策略动作;视频用 PyAV 按 meta/episodes 的
  时间戳区间顺序解码(不依赖 torchcodec)
- 每步 info["intervene_action"] = 记录动作,intervene_flag=1(存进回放的动作=记录动作,同时成为 BC 目标)
- 奖励:段末成功 +1;terminated 在段末
"""

from __future__ import annotations

import copy
import glob
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.utils.logging import get_logger
from rlinf.envs.realworld.a1.a1_env import env_to_rl

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


@dataclass
class A1PlaybackConfig:
    is_dummy: bool = False
    dataset_root: str = "/home/yh/projects/galaxea_a1/piper_data/a1_cloth_hil"
    repo_id: str = "local/a1_cloth_hil"
    episodes: Optional[list] = None
    success_reward: float = 1.0
    frame_hw: tuple = (480, 640)
    task_description: str = "fold the cloth"
    max_num_steps: int = 100000
    step_frequency: float = 0.0


def _frame_gen(path, from_ts, to_ts):
    """PyAV 顺序解码 [from_ts, to_ts) 的帧(RGB uint8)。"""
    import av
    c = av.open(path); s = c.streams.video[0]; s.thread_type = "AUTO"
    tb = float(s.time_base)
    if from_ts > 0:
        c.seek(int(from_ts / tb), stream=s, backward=True, any_frame=False)
    eps = 1e-3
    for fr in c.decode(s):
        t = fr.pts * tb
        if t < from_ts - eps:
            continue
        if t >= to_ts - eps:
            break
        yield fr.to_ndarray(format="rgb24")
    c.close()


class A1PlaybackEnv(gym.Env):
    def __init__(self, config: A1PlaybackConfig, worker_info=None, hardware_info=None, env_idx: int = 0):
        import pandas as pd
        self._log = get_logger()
        self.config = config
        root = config.dataset_root
        self.meta = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/meta/episodes/*/*.parquet"))]).set_index("episode_index").sort_index()
        self.data = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(root + "/data/*/*.parquet"))], ignore_index=True)
        self.root = root
        self.episodes = list(config.episodes) if config.episodes else [int(e) for e in self.meta.index]
        self._ep_ptr = -1
        h, w = config.frame_hw
        self.action_space = gym.spaces.Box(-np.ones(14, np.float32), np.ones(14, np.float32))
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({"qpos": gym.spaces.Box(-np.inf, np.inf, shape=(14,))}),
            "frames": gym.spaces.Dict({n: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8) for n in CAMS}),
        })
        self._log.info("[Playback] %d 段待回放(数据 %d 帧)", len(self.episodes), len(self.data))

    @property
    def task_description(self):
        return self.config.task_description

    def _load_episode(self, e):
        m = self.meta.loc[e]
        a, b = int(m["dataset_from_index"]), int(m["dataset_to_index"])
        rows = self.data[(self.data["episode_index"] == e)]
        self._st = np.stack(rows["observation.state"].values).astype(np.float32)
        self._act = env_to_rl(np.stack(rows["action"].values)).astype(np.float32)   # → RL 帧 [-1,1]
        self._iv = np.stack(rows["intervention"].values).reshape(-1).astype(np.float32)
        self._suc = float(np.asarray(rows["success"].iloc[0]).reshape(-1)[0])
        self._gens = {}
        for n in CAMS:
            k = "videos/observation.images.%s/" % n
            path = "%s/videos/observation.images.%s/chunk-%03d/file-%03d.mp4" % (self.root, n, int(m[k + "chunk_index"]), int(m[k + "file_index"]))
            self._gens[n] = _frame_gen(path, float(m[k + "from_timestamp"]), float(m[k + "to_timestamp"]))
        self._n = len(rows); self._i = 0
        self._last = {}

    def _obs(self):
        frames = {}
        for n in CAMS:
            try:
                img = next(self._gens[n]); self._last[n] = img
            except StopIteration:
                img = self._last.get(n)
                if img is None:
                    img = np.zeros((*self.config.frame_hw, 3), np.uint8)
            frames[n] = img
        return {"state": {"qpos": self._st[min(self._i, self._n - 1)].copy()}, "frames": frames}

    def reset(self, seed=None, options=None, **kwargs):
        self._ep_ptr += 1
        if self._ep_ptr >= len(self.episodes):
            self._log.warning("[Playback] 全部段已回放一遍,从头循环(请及时结束运行)")
            self._ep_ptr = 0
        e = self.episodes[self._ep_ptr]
        self._load_episode(e)
        self._log.info("[Playback] 段 %d(%d 帧,success=%d,介入帧 %d)", e, self._n, int(self._suc), int(self._iv.sum()))
        return copy.deepcopy(self._obs()), {}

    def step(self, action):
        if self._i >= self._n:                      # 段已结束但 chunk 内仍被 step:保持末态
            obs = {"state": {"qpos": self._st[-1].copy()}, "frames": {n: self._last.get(n, np.zeros((*self.config.frame_hw, 3), np.uint8)) for n in CAMS}}
            info = {"intervene_action": self._act[-1].copy(), "intervene_flag": np.ones(1), "recorded_intervention": 0.0, "episode_success": bool(self._suc > 0.5)}
            return copy.deepcopy(obs), 0.0, True, False, info
        rec_act = self._act[self._i].copy(); iv = float(self._iv[self._i])
        self._i += 1
        done = self._i >= self._n
        reward = (self.config.success_reward if self._suc > 0.5 else 0.0) if done else 0.0
        obs = self._obs() if not done else {"state": {"qpos": self._st[-1].copy()}, "frames": {n: self._last.get(n, np.zeros((*self.config.frame_hw, 3), np.uint8)) for n in CAMS}}
        info = {"intervene_action": rec_act, "intervene_flag": np.ones(1), "recorded_intervention": iv,
                "episode_success": bool(self._suc > 0.5)}
        return copy.deepcopy(obs), reward, bool(done), False, info

    def get_hold_action(self, fallback=None):
        return self._act[min(self._i, self._n - 1)].copy()
