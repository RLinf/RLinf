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

"""A1 RLT 控制面板 wrapper:一个 Tk 窗口,只在它有焦点时按键才生效(聊天打字不会误触)。

键位:
  1   = 开始第 1 阶段(reset 后阻塞等待;按下机械臂才开始动,并把控制权交给 RL actor)
  2-9 = 上一阶段完成(里程碑奖励 +1)并切到该阶段;不终止本段,机械臂不回零位
  空格 / n = 同上,直接进入下一阶段(阶段数超过 9 时用它;14 段叠盒子每完成一段按一次)
  a   = 手动把控制权交给 RL actor(数字键已自动置位,这里留作覆盖)
  g   = 回初始位(仅在等待开始时有效,不会开始本段)
  s   = 本段成功(+1,done)   f = 本段失败(0,done)
  r   = 作废本段(done,info["discard"]=True,奖励 0)
窗口同时显示当前状态(介入态、是否已交给 actor、里程碑数、步数)。
给了 stage_prompts 时,task_description 随阶段切换为该阶段的提示词(阶段 0 = 未开始时用第 1 段),
环境每步重新读取,VLA 下一块就用新提示词 —— 按键切阶段同时切提示词。
"""

from __future__ import annotations

import os
import queue as _queue
import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.utils.logging import get_logger


class _Panel:
    """Tk 面板线程:显示状态 + 捕获焦点内按键。"""

    def __init__(self):
        self.q: _queue.Queue = _queue.Queue()
        self.text = "starting ..."
        self.color = "#333333"
        self._ready = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait(5.0)

    def _run(self):
        import tkinter as tk
        root = tk.Tk(); root.title("A1 RLT Control")
        # The active instruction is part of the control state, so leave enough
        # room for the complete prompt instead of truncating it in a one-line
        # status label.  ``set`` remains the same API for the control wrapper.
        root.geometry("1100x340+50+50")
        self._label = tk.Label(root, text=self.text, font=("Helvetica", 20, "bold"),
                               fg="white", bg=self.color, justify="left", anchor="nw",
                               wraplength=1040, padx=20, pady=16)
        self._label.pack(fill="both", expand=True)
        hint = tk.Label(root, text="click me, then: g=back to init  1=start stage1  SPACE/n=next stage(+1)  s=success  f=fail  r=discard", font=("Helvetica", 14), fg="#dddddd", bg="#222222", anchor="w", padx=20)
        hint.pack(fill="x")
        root.bind("<Key>", lambda e: self.q.put(e.char.lower()) if e.char else None)
        self._root = root
        def refresh():
            self._label.config(text=self.text, bg=self.color); root.after(100, refresh)
        root.after(100, refresh); self._ready.set(); root.mainloop()

    def set(self, text, color):
        self.text, self.color = text, color

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except _queue.Empty:
                return out


class _ScriptPanel:
    """假数据测试用:代替 Tk 面板,按固定步数自动"按键"。
    阶段 0 时(reset 门控)立即按 1;之后每 every 次 poll(= 每步一次)按下一个阶段键,
    阶段满了再过 every 步按 s 结束本段。every 取质数,按键会落在块内各个位置。"""

    def __init__(self, every: int, n_stages: int, stage_getter):
        self.every, self.n, self._get = int(every), int(n_stages), stage_getter
        self._count = 0

    def set(self, text, color):
        pass

    def poll(self):
        st = self._get()
        if st == 0:
            self._count = 0
            return ["1"]
        self._count += 1
        if self._count % self.every:
            return []
        return ["n"] if st < self.n else ["s"]


class A1KeyboardControl(gym.Wrapper):
    """见模块文档。奖励:里程碑 +1、成功 +1;done:s/f/r。"""

    MILESTONE_REWARD = 1.0
    SUCCESS_REWARD = 1.0

    def __init__(self, env: gym.Env, milestones: int = 5, script_every: int | None = None,
                 stage_prompts: list[str] | None = None):
        super().__init__(env)
        self._log = get_logger()
        self.panel = (_ScriptPanel(script_every, milestones, lambda: self._stage)
                      if script_every else _Panel())
        self.n_milestones = milestones
        self._digits = "".join(str(i) for i in range(1, min(milestones, 9) + 1))
        if stage_prompts is not None and len(stage_prompts) != milestones:
            raise ValueError("stage_prompts 有 %d 条,与 milestones=%d 不一致" % (len(stage_prompts), milestones))
        self._prompts = list(stage_prompts) if stage_prompts else None
        self._switch = False
        self._done_milestones = 0
        self._steps = 0
        self._episode = 0
        self._stage = 0          # 0=尚未开始;1..n=当前阶段号,随每步写进 info["stage"]
        self.panel.set("IDLE - press 1 to start stage 1", "#1e6e28")

    @property
    def rlt_switch_flags(self) -> int:
        # 下游把这条线当"阶段号"用:0 = 基座 VLA 驾驶,k = 第 k 阶段的 actor 驾驶。
        # route 里仍按非零/零当布尔判断是否执行 actor 动作,语义不变;
        # rollout 则据此把 8 维阶段 one-hot 拼到 proprio 尾部。
        return int(self._stage) if self._switch else 0

    @property
    def stage(self) -> int:
        return self._stage

    @property
    def task_description(self) -> str:
        if self._prompts is None:
            return self.env.get_wrapper_attr("task_description")
        return self._prompts[max(self._stage, 1) - 1]

    def _status(self):
        st = getattr(self.env, "state", None)     # 介入 wrapper 的状态(若有)
        # This is the exact prompt sent in ``obs[\"task_descriptions\"]`` to
        # the frozen VLA/RLT rollout.  Showing it here makes a stage transition
        # directly auditable from the HIL panel.
        prompt = self.task_description
        return "ep %d | step %d | stage %d/%d | %s | %s\nPrompt: %s" % (
            self._episode, self._steps, self._stage, self.n_milestones,
            "RL ACTOR" if self._switch else "BASE VLA", st or "-", prompt)

    def reset(self, *, seed=None, options=None):
        self._switch = False; self._done_milestones = 0; self._steps = 0; self._stage = 0
        self.panel.poll()
        self.panel.set("RESETTING - robot moving to init", "#28508a")
        obs, info = self.env.reset(seed=seed, options=options)
        # 环境 worker 和 rollout worker 是并行初始化的:环境几秒就绪,而冻结 VLA 的
        # 16G 权重要三分钟才加载完。若此时就放行,人按了 1 也没有动作产出,机械臂纹丝不动,
        # 看起来像卡死。所以先显示"模型加载中"并吞掉按键,等模型就绪再开放门控。
        ready_flag = os.environ.get("RLT_READY_FLAG", "/tmp/rlt_rollout_ready")
        warned = False
        while not os.path.exists(ready_flag):
            if not warned:
                self.panel.set("LOADING MODEL ... please wait (~3 min)", "#8a5a1e")
                self._log.info("[Panel] 等待 rollout 模型就绪(%s)", ready_flag)
                warned = True
            self.panel.poll()                # 丢弃加载期间的误触
            time.sleep(0.2)
        self.panel.set("place the blank, then press 1 to start  (g = back to init)", "#1e6e28")
        while True:                          # 门控:按 1 才开始;按 g 只回初始位、不开始
            keys = self.panel.poll()
            if "1" in keys:
                self._stage = 1
                self._switch = True          # 阶段一开始就由 actor 驾驶
                break
            if "g" in keys:                  # 与采集脚本一致:g = 回初始位
                self._log.info("[Panel] g:回初始位")
                self.panel.set("RESETTING - robot moving to init", "#28508a")
                obs, info = self.env.reset(seed=seed, options=options)
                self.panel.set("place the blank, then press 1 to start  (g = back to init)", "#1e6e28")
            time.sleep(0.05)
        self._episode += 1
        self.panel.set(self._status(), "#1e8c3c")
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        reward = 0.0
        event = None
        for k in self.panel.poll():
            if k == "a" and not self._switch:
                self._switch = True; event = "enter_actor"; self._log.info("[Panel] 交给 RL actor")
            elif k in self._digits or k in (" ", "n"):
                nk = self._stage + 1 if k in (" ", "n") else int(k)
                if nk > self.n_milestones:
                    self._log.info("[Panel] 已在最后一个阶段 %d,完成后按 s", self._stage)
                    continue
                if nk == self._stage + 1:        # 只接受顺序推进,防误触跳阶段
                    # 按 k 表示第 k-1 阶段做完了:给上一阶段 +1,再把阶段号切到 k。
                    # 不设 terminated:auto_reset 会让机械臂回初始位,把盒子的半成品状态毁掉。
                    self._done_milestones += 1; reward += self.MILESTONE_REWARD
                    info["milestone"] = self._stage
                    self._stage = nk; self._switch = True
                    self._log.info("[Panel] 阶段 %d 完成 → 进入阶段 %d", nk - 1, nk)
                else:
                    self._log.info("[Panel] 阶段键 %s 无效(当前 %d,只能按 %d)", k, self._stage, self._stage + 1)
            elif k == "s":
                if self._stage < self.n_milestones:   # 防误按:只有进入最后一个阶段后 s 才算成功
                    self._log.info("[Panel] s 无效(当前阶段 %d,需在阶段 %d 完成后再按 s)", self._stage, self.n_milestones)
                    # 面板每步都会被 _status() 覆盖,这条提示原先只显示 1/15 秒、肉眼看不到;改成保留 3 秒
                    self._warn = ("s IGNORED: stage %d/%d not finished - press %d first" % (self._stage, self.n_milestones, self._stage + 1), time.time() + 3.0)
                    continue
                reward += self.SUCCESS_REWARD; terminated = True; info["episode_success"] = True
            elif k == "f":
                terminated = True; info["episode_success"] = False
            elif k == "r":
                terminated = True; info["discard"] = True; info["episode_success"] = False
        info["rlt_switch_flags"] = int(self._stage) if self._switch else 0
        info["rlt_policy_switch_event"] = event
        info["stage"] = self._stage          # 每步的阶段号,下游据此选 actor 头、分回放缓冲
        color = "#8a1e1e" if getattr(self.env, "intervening", False) else ("#8a5a1e" if self._switch else "#1e8c3c")
        warn = getattr(self, "_warn", None)
        if warn is not None and time.time() < warn[1]:
            self.panel.set(warn[0], "#8a1e1e")
        else:
            self.panel.set(self._status(), color)
        return obs, reward, terminated, truncated, info
