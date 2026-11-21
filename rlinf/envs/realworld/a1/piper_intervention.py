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

"""双松灵 PiPER 主臂"镜像 + 示教按钮"介入 wrapper(A1 原生动作空间)。

主臂(从臂模式,canL=左 canR=右,udev 按序列号固定命名):
  FOLLOW  态:关节位置镜像 A1 实际关节;示教器扳机电机使能、跟随 A1 夹爪指令
  FREE    态:不发关节指令(用户按示教按钮后固件重力补偿可拖);扳机失能、读扳机控夹爪

踏板(evdev,独占):左 = 暂停/分界,右 = 恢复策略
  FOLLOW ─左─▶ PAUSE_IN(A1 保持,扳机放开)→ 用户按示教按钮
  PAUSE_IN ─左─▶ INTERVENE(主臂→A1,扳机→夹爪;入口 0.5s 渐变)
  INTERVENE ─左─▶ PAUSE_OUT(A1 保持)→ 用户按按钮退出示教
  PAUSE_OUT/PAUSE_IN ─右─▶ RESUMING(固件恢复序列 → 回锁镜像)─▶ FOLLOW

非 FOLLOW 态的每一步都按 RLinf 约定写 info["intervene_action"](实际执行的 14 维动作)
和 info["intervene_flag"]=1,进入回放后成为 BC 目标(替换参考块)。
"""

from __future__ import annotations

import math
import queue as _queue
import select
import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.utils.logging import get_logger
from rlinf.envs.realworld.a1.a1_env import env_to_rl, rl_to_env

DEG = math.pi / 180.0
SIDES = ("left", "right")
IDX = {"left": (0, 6), "right": (7, 13)}       # 关节起点, 夹爪索引(A1 7+7)
CANS = {"left": "canL", "right": "canR"}
PEDAL_PHYS = {"left": "usb-0000:00:14.0-1.3/input0", "right": "usb-0000:00:14.0-1.4/input0"}
# 遥操作标定:A1_j = SIGNS·piper_j + OFFSETS(主臂读数→A1),镜像用逆映射
SIGNS = np.array([1, 1, 1, 1, -1, 1], np.float64)
OFFSETS = np.array([0, 0, 0, 1.5708, 0, 0], np.float64)
A1_LOW = np.array([-2.8798, 0.00, -3.3161, -2.8798, -1.6581, -2.8798])
A1_HIGH = np.array([2.8798, 3.1415, 0.000, 2.8798, 1.6581, 2.8798])
P_LOW = np.array([-2.618, 0.0, -2.967, -1.745, -1.22, -2.0944])
P_HIGH = np.array([2.618, 3.14, 0.0, 1.745, 1.22, 2.0944])
GRIP_K = 1.35             # 扳机 mm → A1 0-100
TRIG_MAX_MM = 74.0
MIRROR_STEP = 0.02        # rad/帧 @50Hz
RAMP_STEPS = 8            # 介入接合渐变基准(≈0.5s @15Hz);实际按 step_frequency 缩放,见 _ramp_steps


def fwd_map(pj):
    return np.clip(SIGNS * pj + OFFSETS, A1_LOW + 0.03, A1_HIGH - 0.03)


def inv_map(aj):
    return np.clip(SIGNS * (np.asarray(aj, np.float64) - OFFSETS), P_LOW, P_HIGH)


class Pedals:
    """两个 KM-key08 踏板(都发空格),按 USB 物理路径区分,独占读取。"""

    def __init__(self, swap=False):
        from evdev import InputDevice, ecodes, list_devices
        self.ecodes = ecodes
        self.q: _queue.Queue = _queue.Queue()
        phys = dict(PEDAL_PHYS) if not swap else {"left": PEDAL_PHYS["right"], "right": PEDAL_PHYS["left"]}
        self.devs = {}
        for p in list_devices():
            d = InputDevice(p)
            if d.name != "KM-key08" or ecodes.KEY_SPACE not in d.capabilities().get(ecodes.EV_KEY, []):
                continue
            for side, ph in phys.items():
                if d.phys == ph:
                    d.grab(); self.devs[d.fd] = (d, side)
        if len(self.devs) != 2:
            raise RuntimeError("未找到两个 KM-key08 踏板(找到 %d)" % len(self.devs))
        self._last = {"left": 0.0, "right": 0.0}
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            r, _, _ = select.select(list(self.devs), [], [], 0.5)
            for fd in r:
                d, side = self.devs[fd]
                try:
                    evs = list(d.read())
                except Exception:
                    continue
                for ev in evs:
                    if ev.type == self.ecodes.EV_KEY and ev.value == 1:
                        now = time.monotonic()
                        if now - self._last[side] > 0.3:
                            self._last[side] = now; self.q.put(side)

    def poll(self):
        out = []
        while True:
            try:
                out.append(self.q.get_nowait())
            except _queue.Empty:
                return out

    def close(self):
        for d, _ in self.devs.values():
            try:
                d.ungrab()
            except Exception:
                pass


class Masters:
    """50Hz 线程:MIRROR 时位置镜像 + 扳机跟随;FREE 时只读。命令经队列在本线程执行。"""

    def __init__(self, speed=50):
        from piper_sdk import C_PiperInterface_V2
        self._log = get_logger()
        self.p = {}
        for side in SIDES:
            p = C_PiperInterface_V2(CANS[side]); p.ConnectPort(); self.p[side] = p
        time.sleep(0.5)
        for side in SIDES:                      # 自检:适配器发送健康 + 未卡在示教模式
            fw = None
            for _ in range(4):
                try:
                    fw = self.p[side].GetPiperFirmwareVersion()
                except Exception as e:
                    fw = "异常 %r" % e
                if isinstance(fw, str) and fw.startswith("S-V"):
                    break
                time.sleep(0.6)
            if not (isinstance(fw, str) and fw.startswith("S-V")):
                raise RuntimeError("%s(%s)CAN 适配器发送已卡死(固件查询=%s):拔插适配器后重启" % (side, CANS[side], fw))
            if int(self.p[side].GetArmStatus().arm_status.ctrl_mode) == 0x02:
                raise RuntimeError("%s 主臂仍在示教模式:按一下示教按钮退出后重启" % side)
        self.speed = speed
        self.lock = threading.Lock()
        self.mode = "FREE"
        self.joints = {s: np.zeros(6) for s in SIDES}
        self.trig_mm = {s: 0.0 for s in SIDES}
        self.a1_joints = {s: None for s in SIDES}
        self.grip_cmd = {s: 0.0 for s in SIDES}
        self.mirror_cmd = {s: None for s in SIDES}
        self.cmd_q: _queue.Queue = _queue.Queue()
        self.busy = threading.Event()
        self._tick = 0
        self._stop = threading.Event()
        self._th = threading.Thread(target=self._loop, daemon=True); self._th.start()

    def _fb(self, side):
        jm = self.p[side].GetArmJointMsgs().joint_state
        return np.array([getattr(jm, "joint_%d" % i) for i in range(1, 7)], np.float64) * 1e-3 * DEG

    def _trig(self, side):
        a = self.p[side].GetArmGripperMsgs().gripper_state.grippers_angle * 1e-3
        return float(min(max(a, 0.0), TRIG_MAX_MM))

    def _joint_ctrl(self, side, q):
        self.p[side].MotionCtrl_2(0x01, 0x01, self.speed, 0)    # 断电重启后的固件只认每帧先发这一条
        self.p[side].JointCtrl(*[int(round(v / DEG * 1000)) for v in q])

    def _trig_ctrl(self, side, mm, enable=True):
        self.p[side].GripperCtrl(int(round(mm * 1000)), 500, 0x01 if enable else 0x00, 0)

    def _wait_live_fb(self, side, timeout=15.0):
        """等主臂真的在广播关节反馈(时间戳非零且在变)。

        主臂断电重启后处于待机(0x00),使能前不广播关节位置,GetArmJointMsgs 全是 0;
        原逻辑读到 0 就 JointCtrl(0,...),主臂会以 speed 猛甩到零位(实测"抽风"就是这个)。"""
        t0 = time.time(); last = None; last_enable = t0
        while time.time() - t0 < timeout:
            ts = getattr(self.p[side].GetArmJointMsgs(), "time_stamp", 0)
            if ts and last is not None and ts != last:
                return True
            if time.time() - last_enable > 1.0:          # 刚上电的主臂:先查固件唤醒(实测不查就不回反馈),再切 CAN 控制并使能
                try:
                    self.p[side].SearchPiperFirmwareVersion()
                except Exception:
                    pass
                time.sleep(0.3)
                self.p[side].MotionCtrl_1(0, 0, 2); time.sleep(0.05)
                self.p[side].MotionCtrl_2(0x01, 0x01, self.speed, 0); self.p[side].EnableArm(7)
                last_enable = time.time()
            last = ts; time.sleep(0.05)
        return False

    def _relock(self, side):
        p = self.p[side]
        p.MotionCtrl_1(0x00, 0x00, 0x02); time.sleep(0.1)
        p.MotionCtrl_2(0x00, 0x01, self.speed, 0); time.sleep(0.1)
        p.MotionCtrl_2(0x01, 0x01, self.speed, 0); time.sleep(0.1)
        for _ in range(2):
            p.EnableArm(7); time.sleep(0.2)
        if not self._wait_live_fb(side):
            raise RuntimeError("%s 主臂无关节反馈(断电后未就绪或 CAN 线松),拒绝回锁,不发位置指令" % side)
        q = self._fb(side)
        if not np.any(q):
            raise RuntimeError("%s 主臂关节读数全 0,拒绝回锁" % side)
        self._joint_ctrl(side, q); self.mirror_cmd[side] = q.copy()
        for _ in range(3):
            self._trig_ctrl(side, self.grip_cmd[side] / GRIP_K, True); time.sleep(0.03)

    def _free(self, side):
        for _ in range(3):
            self._trig_ctrl(side, 0.0, False); time.sleep(0.03)

    def request(self, cmd):
        self.busy.set(); self.cmd_q.put(cmd)

    def wait_idle(self, timeout=8.0):
        t0 = time.time()
        while self.busy.is_set() and time.time() - t0 < timeout:
            time.sleep(0.02)
        return not self.busy.is_set()

    def set_targets(self, st14, cmd14):
        with self.lock:
            for s in SIDES:
                j0, gi = IDX[s]
                self.a1_joints[s] = np.asarray(st14[j0:j0 + 6], np.float64)
                self.grip_cmd[s] = float(cmd14[gi])

    def read(self):
        with self.lock:
            return ({s: self.joints[s].copy() for s in SIDES}, dict(self.trig_mm))

    def stop(self):
        self._stop.set(); self._th.join(timeout=2.0)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                cmd = self.cmd_q.get_nowait()
            except _queue.Empty:
                cmd = None
            if cmd in ("relock", "free"):
                try:
                    for s in SIDES:
                        (self._relock if cmd == "relock" else self._free)(s)
                    self.mode = "MIRROR" if cmd == "relock" else "FREE"
                except Exception as e:
                    self._log.error("masters %s 失败: %r", cmd, e)
                finally:
                    self.busy.clear()
            try:
                fbs = {s: self._fb(s) for s in SIDES}; trs = {s: self._trig(s) for s in SIDES}
                with self.lock:
                    self.joints.update(fbs); self.trig_mm.update(trs)
                    a1 = dict(self.a1_joints); grips = dict(self.grip_cmd)
                if self.mode == "MIRROR":
                    self._tick += 1
                    for s in SIDES:
                        if a1[s] is None or self.mirror_cmd[s] is None:
                            continue
                        tgt = inv_map(a1[s]); c = self.mirror_cmd[s]
                        c = c + np.clip(tgt - c, -MIRROR_STEP, MIRROR_STEP); self.mirror_cmd[s] = c
                        self._joint_ctrl(s, c)
                        if self._tick % 5 == 0:
                            self._trig_ctrl(s, min(max(grips[s], 0.0), 100.0) / GRIP_K, True)
            except Exception as e:
                self._log.error("masters loop: %r", e)
            time.sleep(max(0.0, 0.02 - (time.time() - t0)))


class A1PiperMirrorIntervention(gym.Wrapper):
    """镜像主臂 + 示教按钮 + 踏板的介入 wrapper。见模块文档。"""

    def __init__(self, env: gym.Env, swap_pedals: bool = False, speed: int = 50):
        super().__init__(env)
        self._log = get_logger()
        self.pedals = Pedals(swap=swap_pedals)
        self.masters = Masters(speed=speed)
        self.state = "RESUMING"
        self.ramp_k = RAMP_STEPS * 4
        self.last_exec = None
        self.masters.request("relock"); self.masters.wait_idle(10.0)
        self.state = "FOLLOW"
        self._log.info("[PiperMirror] 主臂已回锁镜像;左踏板=暂停/分界 右踏板=恢复")

    @property
    def intervening(self) -> bool:
        return self.state != "FOLLOW"

    def _hold(self):
        h = self.env.unwrapped.get_hold_action(self.env.unwrapped.get_state())
        return np.asarray(h, np.float32)

    def _handle_pedals(self):
        for p in self.pedals.poll():
            if self.state == "FOLLOW" and p == "left":
                self.masters.request("free"); self.state = "PAUSE_IN"
                self._log.info("[PiperMirror] 暂停:按示教按钮,准备好再踩左踏板")
            elif self.state == "PAUSE_IN" and p == "left" and not self.masters.busy.is_set():
                self.state = "INTERVENE"; self.ramp_k = 0
                self._log.info("[PiperMirror] 介入开始")
            elif self.state in ("PAUSE_IN", "PAUSE_OUT") and p == "right" and not self.masters.busy.is_set():
                self.masters.request("relock"); self.state = "RESUMING"
            elif self.state == "INTERVENE" and p == "left":
                self.state = "PAUSE_OUT"
                self._log.info("[PiperMirror] 介入结束:退出示教按钮,再踩右踏板恢复策略")
            elif self.state == "PAUSE_OUT" and p == "left":
                self.state = "INTERVENE"; self.ramp_k = 0

    def action(self, policy_action):
        """返回 (实际执行动作, 是否介入)。"""
        self._handle_pedals()
        u = self.env.unwrapped
        st = u.get_state()
        if st is not None:
            cmd_env = u.last_cmd_env()
            if cmd_env is None:
                cmd_env = np.asarray(st, np.float64).copy(); cmd_env[[6, 13]] = 0.0
            self.masters.set_targets(st, cmd_env)                       # 镜像扳机用 env 帧夹爪 0-100
        if self.state == "RESUMING":
            if not self.masters.busy.is_set():
                self.state = "FOLLOW"; self._log.info("[PiperMirror] 主臂已回锁,策略恢复")
                self._last_kind = 0; return np.asarray(policy_action, np.float32), False
            self._last_kind = 1; return self._hold(), True
        if self.state == "FOLLOW":
            self._last_kind = 0; return np.asarray(policy_action, np.float32), False
        if self.state in ("PAUSE_IN", "PAUSE_OUT"):
            self._last_kind = 1; return self._hold(), True
        # INTERVENE
        mj, mt = self.masters.read()
        target_env = np.zeros(14, np.float64)
        for s in SIDES:
            j0, gi = IDX[s]
            target_env[j0:j0 + 6] = fwd_map(mj[s]); target_env[gi] = min(max(mt[s] * GRIP_K, 0.0), 100.0)
        target = env_to_rl(target_env).astype(np.float32)                 # → RL 帧 [-1,1]
        _rs = self._ramp_steps
        self._last_kind = 3                                                # 主动介入
        if self.ramp_k < _rs and u._last_cmd is not None:
            self.ramp_k += 1
            target = (u._last_cmd + (target - u._last_cmd) * (self.ramp_k / _rs)).astype(np.float32)
            self._last_kind = 2                                            # 接合渐变(合成动作,不是人的纠正)
        return target, True

    def step(self, action):
        exec_action, intervened = self.action(action)
        obs, rew, done, trunc, info = self.env.step(exec_action)
        info["intervene_state"] = self.state
        # 逐步分类:0 自主 / 1 保持(暂停、回锁) / 2 渐变 / 3 主动。回放据此决定哪些步做人类 BC(只有 3)。
        info["iv_kind"] = int(getattr(self, "_last_kind", 3 if intervened else 0))
        # 实际下发给 A1 的指令(含限位、右臂 J1 裁剪、hold、渐变),回放的 actions 栏一律记它
        try:
            ce = self.env.unwrapped.last_cmd_env()
            info["exec_action"] = env_to_rl(ce).astype(np.float32) if ce is not None else np.asarray(exec_action, np.float32)
        except Exception:
            info["exec_action"] = np.asarray(exec_action, np.float32)
        if intervened:
            info["intervene_action"] = np.asarray(exec_action, np.float32)
            info["intervene_flag"] = np.ones(1)
        return obs, rew, done, trunc, info

    def reset(self, **kwargs):
        # 若此刻人正握着主臂(介入/暂停态),绝不能直接回锁:回锁会重新使能主臂并把它拉向 A1,
        # 等于在操作员手上动。先在日志/面板提示,等踩右踏板交还(回到 FOLLOW)再继续复位。
        # 人正握着主臂时不能回锁(回锁会重新使能主臂并把它拉向 A1,等于在操作员手上动)。
        # 但也绝不能在这里阻塞等待:reset 阻塞会把整个环境循环占住,踏板交还后又立刻复位,
        # 表现就是卡在 RESETTING、且介入期间动作一顿一顿(实测栈采样 20/20 都在 env.step 里)。
        # 所以改成非阻塞:介入态就跳过回锁,由后续的踏板事件自然回到 FOLLOW。
        if self.state != "FOLLOW":
            self._log.warning(
                "[PiperMirror] 复位时仍在介入态(%s):跳过回锁,请退出示教并踩右踏板交还", self.state
            )
        elif self.masters.mode == "FREE":
            self.masters.request("relock"); self.masters.wait_idle(10.0); self.state = "FOLLOW"
        self.pedals.poll()
        out = self.env.reset(**kwargs)
        # 复位后必须立刻把镜像目标刷成 A1 的新姿态。
        # 原因:镜像线程 50Hz 跟踪的目标 a1_joints 只在 step() 里通过 set_targets 更新,
        # 而复位之后程序会停在控制面板的门控上等人按键,一次 step 都不会执行。
        # 于是主臂会一直镜像"复位前"的旧姿态 —— 叠衣服时两者接近所以看不出来,
        # 换成叠盒子的起始姿态后相差近 2 rad,表现就是主从臂明显不同步。
        try:
            u = self.env.unwrapped
            st = u.get_state()
            if st is not None:
                cmd_env = u.last_cmd_env()          # 夹爪要 env 帧 0-100,和 step 里一致
                if cmd_env is None:
                    cmd_env = np.asarray(st, np.float64).copy(); cmd_env[[6, 13]] = 0.0
                self.masters.set_targets(st, cmd_env)
                self._log.info("[PiperMirror] 复位后已刷新镜像目标")
        except Exception as e:
            self._log.warning("[PiperMirror] 刷新镜像目标失败: %r", e)
        return out

    @property
    def _ramp_steps(self) -> int:
        """按环境步频缩放交接斜坡,保证渐变时长恒为 ≈0.5s(30Hz 下即 16 步)。"""
        try:
            f = float(self.env.unwrapped.config.step_frequency)
        except Exception:
            f = 15.0
        return max(1, int(round(RAMP_STEPS * f / 15.0)))

    def get_hold_action(self, fallback=None):
        return self._hold()

    def close(self):
        try:
            self.masters.stop(); self.pedals.close()
        finally:
            return super().close()
