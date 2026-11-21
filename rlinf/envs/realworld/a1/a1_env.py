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

"""Galaxea A1 dual-arm env, **A1-native** joint space (UDP bridge to the a1_noetic stack).

Action / state layout = the training data (a1_cloth) layout, aloha 7+7:
``[L j1..j6, L gripper(0-100), R j1..j6, R gripper(0-100)]``, joints in rad.
No conversion anywhere: what the model sees is what the robot reports/executes.

Transport (host side of the docker a1_noetic stack):
  action → UDP 9997 (udp2joint: per-side JSON, limit + rate clipping inside)
  state  ← UDP 9990 (joint2udp: {"left": [...], "right": [...], "grip_*"})
"""

from __future__ import annotations

import copy
import json
import socket
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.utils.logging import get_logger

# A1 官方 URDF 限位(动作空间边界)
_A1_LOW = np.array([-2.8798, 0.00, -3.3161, -2.8798, -1.6581, -2.8798])
_A1_HIGH = np.array([2.8798, 3.1415, 0.000, 2.8798, 1.6581, 2.8798])
JL = list(range(0, 6)) + list(range(7, 13))   # 关节维
GI = [6, 13]                                   # 夹爪维
# RL 动作帧 = 每维归一到 [-1,1](tanh actor);env 帧 = 关节 rad + 夹爪 0-100
_ENV_LO = np.concatenate([_A1_LOW, [0.0], _A1_LOW, [0.0]])
_ENV_HI = np.concatenate([_A1_HIGH, [100.0], _A1_HIGH, [100.0]])
ENV_SCALE = (_ENV_HI - _ENV_LO) / 2.0
ENV_BIAS = (_ENV_HI + _ENV_LO) / 2.0


def env_to_rl(a_env):
    return (np.asarray(a_env, np.float64) - ENV_BIAS) / ENV_SCALE


def rl_to_env(a_rl):
    return np.asarray(a_rl, np.float64) * ENV_SCALE + ENV_BIAS

# a1_cloth ep0 首帧 action(采集时的真实起始姿态)
_DEFAULT_RESET_QPOS = [-0.4193, 1.0231, -1.1728, 1.9318, -0.4328, 0.3296, 0.0,
                       0.2085, 0.9624, -1.1927, 1.5944, -0.8103, -0.0929, 0.0]


@dataclass
class A1DualRobotConfig:
    """Configuration for :class:`A1DualEnv`."""

    is_dummy: bool = False
    step_frequency: float = 15.0
    """环境步频(Hz)= 训练数据帧率。"""

    obs_port: int = 9990
    act_addr: tuple = ("127.0.0.1", 9997)
    obs_timeout: float = 1.0
    # 额外关节限位(env 帧、弧度),{关节索引: [下限或 None, 上限或 None]},在下发给 A1 的最后一道关口截断,
    # 策略与人工介入都受约束。索引:左臂 0-5、左夹爪 6、右臂 7-12、右夹爪 13。
    joint_limit_overrides: dict = field(default_factory=dict)
    camera_stale_timeout: float = 0.3      # 某路相机超过该时长没有新帧 → 停止执行并重连
    camera_reconnect_timeout: float = 120.0  # 重连等待上限,超时报错退出

    camera_serials: dict = field(default_factory=lambda: {
        "cam_high": "337322070914",
        "cam_left_wrist": "409122272402",
        "cam_right_wrist": "409122272807",
    })
    frame_hw: tuple = (480, 640)

    reset_qpos: list = field(default_factory=lambda: list(_DEFAULT_RESET_QPOS))
    reset_step_limit: float = 0.06
    reset_lift_height: float = 0.12
    """reset 先把指尖竖直抬高多少米(0 = 不抬),再各关节同步插值回初始位,避免贴着台面横扫。"""
    reset_lift_z_cap: float = 0.45
    """抬起的目标高度上限(基座系,m);已高于此值就不抬。初始位指尖约 0.5。"""
    reset_lift_step: float = 0.03
    """抬起阶段关节单步限幅(rad/步)。"""
    """reset 时关节单步限幅(rad/步)。"""
    reset_grip_limit: float = 8.0
    manual_episode_control_only: bool = False
    """由 RealWorldEnv 读取(override_cfg 整包也会传进本 dataclass,故这里必须声明以免报未知字段)。"""

    max_num_steps: int = 900
    task_description: str = "fold the cloth"


class A1DualEnv(gym.Env):
    """Galaxea A1 dual-arm env, 14-D A1-native joint actions."""

    def __init__(self, config: A1DualRobotConfig, worker_info=None,
                 hardware_info=None, env_idx: int = 0):
        self._logger = get_logger()
        self.config = config
        self.env_idx = env_idx
        self._num_steps = 0
        self._state14: Optional[np.ndarray] = None
        self._state_time = 0.0
        self._last_cmd: Optional[np.ndarray] = None
        self._last_cmd_env: Optional[np.ndarray] = None

        self._init_action_obs_spaces()
        if self.config.is_dummy:
            return

        self._rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._rx.bind(("127.0.0.1", int(self.config.obs_port)))
        self._rx.setblocking(False)
        self._tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._open_cameras()
        t0 = time.time()
        while self._read_state() is None:
            time.sleep(0.05)
            if time.time() - t0 > 10:
                raise RuntimeError("A1 关节观测超时:确认 a1_piper_stack + joint2udp 在运行,"
                                   f"且端口 {self.config.obs_port} 未被占用")
        self._logger.info("[A1DualEnv] 已连接(A1 原生 14 维,%.0f Hz)", self.config.step_frequency)

    # ── spaces ──────────────────────────────────────────────────────────
    def _init_action_obs_spaces(self):
        self.action_space = gym.spaces.Box(-np.ones(14, np.float32), np.ones(14, np.float32))   # RL 帧 [-1,1]
        h, w = self.config.frame_hw
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({"qpos": gym.spaces.Box(-np.inf, np.inf, shape=(14,))}),
            # 关节速度(相邻两步位置差分):只给 RLT 的 actor/critic 当本体状态(论文:位置 + 速度),
            # 不放进 "state",免得混进冻结 VLA 的 14 维状态输入
            "rlt_qvel": gym.spaces.Box(-np.inf, np.inf, shape=(14,), dtype=np.float32),
            "frames": gym.spaces.Dict({
                name: gym.spaces.Box(0, 255, shape=(h, w, 3), dtype=np.uint8)
                for name in self.config.camera_serials}),
        })
        self._base_observation_space = copy.deepcopy(self.observation_space)

    @property
    def task_description(self):
        return self.config.task_description

    # ── transport ──────────────────────────────────────────────────────
    def _read_state(self) -> Optional[np.ndarray]:
        latest = None
        while True:
            try:
                d, _ = self._rx.recvfrom(4096)
            except (BlockingIOError, OSError):
                break
            try:
                m = json.loads(d.decode())
                if m.get("left") and m.get("right"):
                    latest = m
            except Exception:
                pass
        if latest is not None:
            st = np.empty(14, np.float32)
            st[0:6] = latest["left"][:6]
            st[6] = float(latest.get("grip_left", 0.0))
            st[7:13] = latest["right"][:6]
            st[13] = float(latest.get("grip_right", 0.0))
            self._state14 = st
            self._state_time = time.time()
        if time.time() - self._state_time > self.config.obs_timeout:
            return None
        return self._state14

    def get_state(self) -> Optional[np.ndarray]:
        """当前 A1 状态(介入 wrapper 用)。"""
        return None if self._state14 is None else self._state14.copy()

    def send_action(self, v14_rl: np.ndarray):
        """RL 帧 [-1,1] 动作 → env 帧 → A1。"""
        v_rl = np.clip(np.asarray(v14_rl, np.float64).reshape(-1), -1.0, 1.0)
        self._send_env(rl_to_env(v_rl))
        self._last_cmd = v_rl.astype(np.float32)

    def _send_env(self, v_env: np.ndarray):
        v = np.clip(np.asarray(v_env, np.float64).reshape(-1), _ENV_LO, _ENV_HI)
        for j, (lo, hi) in (self.config.joint_limit_overrides or {}).items():
            j = int(j)
            if lo is not None and v[j] < lo:
                v[j] = lo
            if hi is not None and v[j] > hi:
                v[j] = hi
        host, port = self.config.act_addr
        for side, j0, gi in (("left", 0, 6), ("right", 7, 13)):
            self._tx.sendto(json.dumps({
                "side": side, "joints": [float(x) for x in v[j0:j0 + 6]], "gripper": float(v[gi]),
            }).encode(), (host, int(port)))
        self._last_cmd_env = v.astype(np.float32)

    def get_hold_action(self, fallback=None) -> np.ndarray:
        """RLinf 介入平滑用(RL 帧):保持当前指令;无指令则用当前关节 + 夹爪 0。"""
        if self._last_cmd is not None:
            return self._last_cmd.copy()
        st = self.get_state()
        if st is None:
            return np.asarray(fallback, np.float32)
        e = st.astype(np.float64); e[GI] = 0.0
        return env_to_rl(e).astype(np.float32)

    def last_cmd_env(self) -> Optional[np.ndarray]:
        return None if getattr(self, "_last_cmd_env", None) is None else self._last_cmd_env.copy()

    # ── gym API ────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        _pend = getattr(self, "_sleep_until", None)
        if _pend is not None:                      # 上一步推迟的补足睡眠(见下)
            self._sleep_until = None
            time.sleep(max(0.0, _pend - time.time()))
        t0 = time.time()
        if not self.config.is_dummy:
            self.send_action(action)
        self._num_steps += 1
        # 取观测必须在补足睡眠之前:原顺序是"先睡满 1/f 再读三路相机",
        # 读相机的耗时没被计入周期,实际周期 = 1/f + t_cam。实测 30Hz 配置只跑到 13.2 步/秒
        # (33ms 睡眠 + 约 50ms 阻塞读相机 ≈ 76ms),机械臂慢、遥操作顿挫、按键响应慢都源于此。
        obs = self._get_observation()
        # 按绝对截止时间补足周期:只按本函数耗时补足,会漏掉 wrapper/chunk_step 的处理时间
        # (实测每步多约 5ms,30Hz 只到 26 步/秒)。截止时间网格把这些开销吸收进周期;
        # 落后超过一个周期(如块间等待)就重新对齐,不补发、不突进。
        period = 1.0 / self.config.step_frequency
        now = time.time()
        nxt = getattr(self, "_next_deadline", None)
        nxt = (now + period - (now - t0)) if nxt is None else nxt + period
        if now > nxt + period:
            nxt = now
        if getattr(self, "_defer_sleep_once", False):
            # 调用方要尽快拿到本步观测(块内预取):睡眠推到下一步开头,节拍不变
            self._defer_sleep_once = False
            self._sleep_until = nxt
        else:
            time.sleep(max(0.0, nxt - now))
        self._next_deadline = nxt
        truncated = self._num_steps >= self.config.max_num_steps
        return obs, 0.0, False, truncated, {}

    def reset(self, seed=None, options=None, **kwargs):
        self._num_steps = 0
        self._qv_prev = None                    # 复位后速度从 0 重新计
        self._next_deadline = None
        self._sleep_until = None
        if self.config.is_dummy:
            return self._get_observation(), {}
        target = np.asarray(self.config.reset_qpos, np.float64)
        cur = self._read_state()
        if cur is None:
            raise RuntimeError("A1 关节观测断流,无法 reset")
        cmd = cur.astype(np.float64).copy()
        cmd[GI] = target[GI]                      # 夹爪观测是驱动原生单位,起步直接用目标
        for q in self._reset_plan(cmd, target):   # 先竖直抬起指尖,再各关节同步插值到初始位
            cmd[JL] = q[JL]
            for i in GI:
                cmd[i] += np.clip(target[i] - cmd[i], -self.config.reset_grip_limit, self.config.reset_grip_limit)
            self._send_env(cmd); self._last_cmd = env_to_rl(cmd).astype(np.float32)
            time.sleep(1.0 / self.config.step_frequency)
        for _ in range(int(5 * self.config.step_frequency)):      # 指令已到位,最多再等 5s 让实际关节跟上
            for i in GI:
                cmd[i] += np.clip(target[i] - cmd[i], -self.config.reset_grip_limit, self.config.reset_grip_limit)
            self._send_env(cmd)
            time.sleep(1.0 / self.config.step_frequency)
            st = self._read_state()
            if st is not None and np.abs(target[JL] - st[JL]).max() < 0.05:
                break
        time.sleep(0.3)
        self._read_state()
        return self._get_observation(), {}

    def _joint_bounds(self):
        lo, hi = _ENV_LO.astype(np.float64).copy(), _ENV_HI.astype(np.float64).copy()
        for j, (l, h) in (self.config.joint_limit_overrides or {}).items():
            if l is not None:
                lo[int(j)] = max(lo[int(j)], l)
            if h is not None:
                hi[int(j)] = min(hi[int(j)], h)
        return lo + 0.02, hi - 0.02

    def _reset_plan(self, start: np.ndarray, target: np.ndarray) -> list:
        """复位关节路径(env 帧 14 维,夹爪维不用)。
        原先各关节各自按单步限幅走:走得少的先到、走得多的还在转,姿势别扭;且不先抬高,
        阶段 5 结束时指尖常贴近台面,一动就贴着台面横扫(按 URDF 正运动学回放最近 10 段:低位横扫平均 0.1m、最大 0.43m)。
        现在:① 两臂各自把指尖竖直抬高 reset_lift_height(水平不动、腕少动、不越限位);
             ② 所有关节同步线性插值到初始位(同时出发同时到)。"""
        from .a1_fk import lift_path
        lo, hi = self._joint_bounds()
        cur = np.clip(start.astype(np.float64), lo, hi)
        lifted = cur.copy()
        lifts = []
        if self.config.reset_lift_height > 0:
            for j0 in (0, 7):
                sl = slice(j0, j0 + 6)
                try:
                    p = lift_path(cur[sl], lo[sl], hi[sl], dz=self.config.reset_lift_height,
                                  z_cap=self.config.reset_lift_z_cap, max_step=self.config.reset_lift_step)
                except Exception as e:                          # 逆解异常不影响复位,退化为直接插值
                    self._logger.warning("reset 抬起逆解失败(%s),直接插值" % e)
                    p = [cur[sl].copy()]
                lifts.append((sl, p))
        n1 = max([len(p) for _, p in lifts], default=1)
        plan = []
        for k in range(1, n1):
            q = cur.copy()
            for sl, p in lifts:
                q[sl] = p[min(k, len(p) - 1)]
            plan.append(q)
        if plan:
            lifted = plan[-1].copy()
        tgt = np.clip(target.astype(np.float64), lo, hi)
        n2 = max(1, int(np.ceil(np.abs(tgt[JL] - lifted[JL]).max() / self.config.reset_step_limit)))
        for k in range(1, n2 + 1):
            plan.append(lifted + (tgt - lifted) * (k / n2))
        return plan

    # ── observation ────────────────────────────────────────────────────
    def _get_observation(self) -> dict:
        obs = self._get_observation_base()
        q = np.asarray(obs["state"]["qpos"], np.float32); now = time.time()
        prev = getattr(self, "_qv_prev", None)
        if prev is None or now - prev[1] <= 1e-3:
            qvel = np.zeros_like(q)
        else:
            qvel = (q - prev[0]) / (now - prev[1])
        self._qv_prev = (q.copy(), now)
        obs["rlt_qvel"] = qvel.astype(np.float32)
        return obs

    def _get_observation_base(self) -> dict:
        if self.config.is_dummy:
            # 每步重新 sample 三整幅随机图很慢(实测拖到 9 步/秒),与真机开销不符;
            # 采样一次后每步深拷贝,开销接近真机"取帧 + copy"
            if getattr(self, "_dummy_obs", None) is None:
                self._dummy_obs = self._base_observation_space.sample()
            return copy.deepcopy(self._dummy_obs)
        st = self._read_state()
        if st is None:
            raise RuntimeError("A1 关节观测断流(>%.1fs)" % self.config.obs_timeout)
        return copy.deepcopy({"state": {"qpos": st.astype(np.float32)}, "frames": self._get_camera_frames()})

    # ── cameras ────────────────────────────────────────────────────────
    def _start_camera(self, serial):
        import pyrealsense2 as rs
        h, w = self.config.frame_hw
        pipe = rs.pipeline(); cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)
        pipe.start(cfg)
        return pipe

    def _open_cameras(self):
        self._pipes = {}
        opened = []
        current_name = None
        current_serial = None
        try:
            # 逐路记录名称和序列号，避免底层 SDK 的超时信息无法定位具体相机。
            for name, serial in self.config.camera_serials.items():
                current_name, current_serial = name, serial
                self._logger.info(
                    "[A1DualEnv] 打开相机 %s(%s)", name, serial
                )
                pipe = self._start_camera(serial)
                self._pipes[name] = pipe
                opened.append((name, serial, pipe))

            # 保持原有的预取 10 帧行为，但按相机分别记录等待过程。
            for name, serial, pipe in opened:
                current_name, current_serial = name, serial
                self._logger.info(
                    "[A1DualEnv] 等待相机首帧 %s(%s)", name, serial
                )
                for _ in range(10):
                    pipe.wait_for_frames()
                self._logger.info(
                    "[A1DualEnv] 相机首帧正常 %s(%s)", name, serial
                )
        except Exception as exc:
            # 某一路失败时，释放此前已经成功启动的 pipeline，避免下一次
            # 启动因旧 pipeline 尚未停止而再次出现设备占用/取帧超时。
            self._logger.error(
                "[A1DualEnv] 相机初始化失败 %s(%s): %s",
                current_name,
                current_serial,
                exc,
                exc_info=True,
            )
            for name, serial, pipe in opened:
                try:
                    pipe.stop()
                    self._logger.info(
                        "[A1DualEnv] 已停止相机 pipeline %s(%s)", name, serial
                    )
                except Exception as stop_exc:
                    self._logger.warning(
                        "[A1DualEnv] 停止相机 pipeline 失败 %s(%s): %s",
                        name,
                        serial,
                        stop_exc,
                    )
            self._pipes = {}
            raise RuntimeError(
                "相机初始化失败: %s(%s): %s"
                % (current_name, current_serial, exc)
            ) from exc
        self._last_frames = {}
        self._last_fresh_t = {name: time.time() for name in self._pipes}

    def _recover_camera(self, name):
        """相机断流:阻塞在这里(不发新指令,机械臂保持最后目标),重建管道直到拿到新帧。"""
        import pyrealsense2 as rs
        serial = self.config.camera_serials[name]
        t0 = time.time()
        logging.getLogger(__name__).error(
            "[A1Env] 相机 %s(%s) %.1fs 无新帧,停止执行并重连 ...", name, serial,
            t0 - self._last_fresh_t.get(name, t0))
        try:
            self._pipes[name].stop()
        except Exception:
            pass
        attempt = 0
        while True:
            attempt += 1
            if time.time() - t0 > self.config.camera_reconnect_timeout:
                raise RuntimeError("相机 %s(%s) 重连 %.0fs 未成功" % (name, serial, self.config.camera_reconnect_timeout))
            try:
                if serial not in [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]:
                    time.sleep(0.5); continue
                pipe = self._start_camera(serial)
                fs = pipe.wait_for_frames(timeout_ms=2000)
                img = np.asanyarray(fs.get_color_frame().get_data()).copy()
            except Exception as e:
                if attempt % 10 == 1:
                    logging.getLogger(__name__).warning("[A1Env] 相机 %s 重连第 %d 次失败: %r", name, attempt, e)
                try:
                    pipe.stop()
                except Exception:
                    pass
                time.sleep(0.5); continue
            self._pipes[name] = pipe
            self._last_frames[name] = img
            self._last_fresh_t[name] = time.time()
            self._next_deadline = None             # 停顿后重新对齐节拍
            logging.getLogger(__name__).error("[A1Env] 相机 %s 已重连(停顿 %.1fs),继续执行", name, time.time() - t0)
            return img

    def _get_camera_frames(self) -> dict:
        out = {}
        for name in list(self._pipes):
            pipe = self._pipes[name]
            try:
                # 非阻塞取最新帧。相机自身以 30fps 自由运行,wait_for_frames 会阻塞到下一帧到达,
                # 三路串行平均各等半个周期 ≈ 50ms;poll 取不到就沿用上一帧,代价只是偶尔重复一帧。
                fs = pipe.poll_for_frames()
                if fs and fs.get_color_frame():
                    img = np.asanyarray(fs.get_color_frame().get_data()).copy()
                    self._last_frames[name] = img
                    self._last_fresh_t[name] = time.time()
                else:
                    img = self._last_frames.get(name)
                    if img is None:                        # 仅开机首帧尚未到达时阻塞一次
                        fs = pipe.wait_for_frames(timeout_ms=300)
                        img = np.asanyarray(fs.get_color_frame().get_data()).copy()
                        self._last_frames[name] = img
                        self._last_fresh_t[name] = time.time()
            except Exception:
                img = self._last_frames.get(name)
                if img is None:
                    raise
            # 沿用旧帧必须有上限:相机掉线后 poll 不报错、只是永远取不到帧,
            # 原逻辑会无限沿用断线前最后一帧,策略看着冻结画面继续推理和动作。
            if time.time() - self._last_fresh_t.get(name, time.time()) > self.config.camera_stale_timeout:
                try:                                   # 先阻塞等一帧,排除"刚好没轮到"的误判(如长时间未 step 后)
                    fs = pipe.wait_for_frames(timeout_ms=200)
                    img = np.asanyarray(fs.get_color_frame().get_data()).copy()
                    self._last_frames[name] = img
                    self._last_fresh_t[name] = time.time()
                except Exception:
                    img = self._recover_camera(name)
            out[name] = img
        return out

    def close(self):
        if not self.config.is_dummy:
            for pipe in getattr(self, "_pipes", {}).values():
                try:
                    pipe.stop()
                except Exception:
                    pass
