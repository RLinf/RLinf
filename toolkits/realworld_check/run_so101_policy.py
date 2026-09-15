# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""使用本地模型或 LeRobot gRPC 服务运行 SO-101 推理。

两个后端共用姿态准备、动作单位转换、控制循环和 DAgger 接管流程。
``grpc`` 后端使用 LeRobot 的 ``RobotClient`` 协议，``local`` 后端在
控制端进程中加载 RLinf 模型。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import select
import sys
import termios
import time
import tty
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Lock, Thread

# Keep direct ``python toolkits/.../run_so101_policy.py`` execution
# working as well as the example wrapper entry point.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig

from rlinf.models.embodiment.openpi.so101_inference import SO101LocalPolicyBackend
from toolkits.realworld_check.so101_dagger import (
    DaggerEpisodeRecorder,
    DaggerSessionController,
)
from toolkits.realworld_check.so101_leader import SO101LeaderExpert

JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--server-address",
        default="127.0.0.1:15051",
        help="本地 SSH 隧道地址；云端服务端口为 50051",
    )
    parser.add_argument("--task", default="抓取青色目标物体并放到盒子里面")
    parser.add_argument("--serial-port", default=os.environ.get("SO101_SERIAL_PORT"))
    parser.add_argument("--robot-id", default=os.environ.get("SO101_ROBOT_ID"))
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--camera", default=os.environ.get("SO101_CAMERA"))
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=15)
    parser.add_argument(
        "--pose-file", type=Path, default=Path("~/.config/so101/poses.json")
    )
    parser.add_argument("--pose-duration", type=float, default=1.5)
    parser.add_argument(
        "--fold-duration-s",
        type=float,
        default=5.0,
        help="主臂收回折叠位的时长；默认 5 秒以降低断联风险。",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--actions-per-chunk", type=int, default=20)
    parser.add_argument("--policy-type", default="pi05")
    parser.add_argument("--pretrained-name-or-path", default="rlinf-so101-backend")
    # gRPC 模式下这是服务端设备；local 模式下是当前进程的模型设备。
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument(
        "--inference-backend",
        choices=("grpc", "local"),
        default=os.environ.get("SO101_INFERENCE_BACKEND", "grpc"),
        help="策略后端：grpc 连接服务端，local 在本进程 GPU 推理。",
    )
    parser.add_argument(
        "--local-checkpoint",
        type=Path,
        default=os.environ.get("SO101_LOCAL_CHECKPOINT"),
        help="local 后端使用的 RLinf actor checkpoint。",
    )
    parser.add_argument(
        "--local-norm-stats",
        type=Path,
        default=os.environ.get("SO101_LOCAL_NORM_STATS"),
        help="local 后端使用的 OpenPI norm_stats.json。",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只校验配置，不连接设备和服务"
    )
    parser.add_argument("--allow-hardware-motion", action="store_true")
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="启用 SO-101 主臂接管测试；空格触发三秒倒计时后切换主臂。",
    )
    parser.add_argument("--leader-port", default=os.environ.get("SO101_LEADER_PORT"))
    parser.add_argument("--leader-id", default=os.environ.get("SO101_LEADER_ID"))
    parser.add_argument("--leader-calibration-dir", type=Path, default=None)
    parser.add_argument(
        "--intervention-hold-s",
        type=float,
        default=3.0,
        help="主臂完成对齐后留给操作者握稳主臂的时间。",
    )
    parser.add_argument(
        "--leader-align-duration-s",
        type=float,
        default=3.0,
        help="触发接管后主臂追到从臂当前姿态的时间。",
    )
    parser.add_argument(
        "--dagger-data-root",
        type=Path,
        default=Path("~/datasets/so101-dagger"),
        help="DAgger LeRobot v3 数据集目录。",
    )
    return parser.parse_args()


def _load_poses(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """读取采集与推理共用的 LeRobot 原生单位姿态。"""
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if (
        data.get("joint_order") != list(JOINT_NAMES)
        or data.get("units") != "lerobot_native"
    ):
        raise ValueError(
            f"姿态文件必须使用 SO-101 关节顺序和 lerobot_native 单位：{path}"
        )
    result = []
    for name in ("standard_position", "fold_position"):
        pose = data.get(name)
        if not isinstance(pose, dict) or set(pose) != set(JOINT_NAMES):
            raise ValueError(f"姿态文件缺少完整的 {name}：{path}")
        result.append({joint: float(pose[joint]) for joint in JOINT_NAMES})
    return result[0], result[1]


def build_robot_config(args: argparse.Namespace) -> SO101FollowerConfig:
    """生成官方 SO-101 配置，字段名称与数据集保持一致。"""
    camera = OpenCVCameraConfig(
        index_or_path=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    )
    return SO101FollowerConfig(
        port=args.serial_port,
        id=args.robot_id,
        calibration_dir=args.calibration_dir,
        cameras={"wrist": camera},
        # 由官方客户端队列处理延迟；姿态准备阶段使用绝对目标。
        max_relative_target=None,
    )


def _send_pose(
    client: RobotClient, pose: dict[str, float], duration: float, fps: int
) -> None:
    """以连续插值发送绝对关节目标，避免姿态切换时突跳。"""
    read_observation = getattr(
        client, "read_robot_observation", client.robot.get_observation
    )
    send_action = getattr(client, "send_robot_action", client.robot.send_action)
    observation = read_observation()
    start = np.asarray(
        [observation[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float32
    )
    target = np.asarray([pose[joint] for joint in JOINT_NAMES], dtype=np.float32)
    steps = max(1, int(round(duration * fps)))
    period = 1.0 / fps
    deadline = time.monotonic()
    for index in range(1, steps + 1):
        ratio = index / steps
        alpha = ratio * ratio * (3.0 - 2.0 * ratio)
        values = (1.0 - alpha) * start + alpha * target
        send_action(
            {
                f"{joint}.pos": float(value)
                for joint, value in zip(JOINT_NAMES, values, strict=True)
            }
        )
        deadline += period
        time.sleep(max(0.0, deadline - time.monotonic()))


class _ControllableRobotClient(RobotClient):
    """给官方控制循环增加暂停门控，队列与 timestep 算法保持原实现。"""

    def __init__(self, config):
        # SO-101 已完成 LeRobot 校准；构造阶段强制复用已有文件，避免进入
        # LeRobot 的交互校准输入流程。
        from lerobot.robots.so_follower import SOFollower

        original_connect = SOFollower.connect

        def connect_with_existing_calibration(robot, calibrate=True):
            del calibrate
            # USB 总线刚打开时偶发一次错误状态包（常见于首次启动或
            # 上一次进程异常退出后）。重试前尽量释放已建立的半连接，
            # 但不改变官方 connect 的校准和配置流程。
            for attempt in range(3):
                try:
                    return original_connect(robot, calibrate=False)
                except ConnectionError:
                    if attempt == 2:
                        raise
                    try:
                        # ``is_connected`` also requires every camera to be
                        # connected, so it can be false while the motor bus is
                        # still open after a partial connect.
                        if robot.bus.is_connected:
                            robot.bus.disconnect(
                                robot.config.disable_torque_on_disconnect
                            )
                        for camera in robot.cameras.values():
                            if camera.is_connected:
                                camera.disconnect()
                    except Exception:  # noqa: BLE001 - 保留原始连接错误
                        pass
                    time.sleep(0.2 * (attempt + 1))

        SOFollower.connect = connect_with_existing_calibration
        try:
            super().__init__(config)
        finally:
            SOFollower.connect = original_connect
        self.inference_enabled = Event()
        # 从臂串口由观测读取、策略动作发送、姿态复位共同使用，必须串行访问。
        self._robot_io_lock = Lock()
        self.intervention_enabled = Event()
        self.intervention_requested = Event()
        self._intervention_lock = Lock()
        self._intervention_hold_s = 3.0
        self._leader_align_duration_s = 3.0
        self._leader_expert: SO101LeaderExpert | None = None
        self._intervention_target: dict[str, float] | None = None
        self._alignment_started = False
        self._closing = Event()
        self._shutdown_log_filter = _ExpectedShutdownRpcFilter(self._closing)
        self.logger.addFilter(self._shutdown_log_filter)
        self._record_callback = None
        self.recording_enabled = Event()
        self._last_policy_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self._last_expert_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self._last_executed_action = np.zeros(len(JOINT_NAMES), dtype=np.float32)
        self._local_policy: SO101LocalPolicyBackend | None = None
        self._local_observations: Queue[tuple[dict, int]] = Queue(maxsize=1)
        self._local_inference_pending = Event()

    def use_local_policy(self, policy: SO101LocalPolicyBackend) -> None:
        """Select in-process inference instead of the gRPC transport."""
        self._local_policy = policy
        self.action_chunk_size = 20

    def run_local_policy(self) -> None:
        """Generate local action chunks while the control loop executes actions."""
        self.start_barrier.wait()
        while self.running:
            try:
                payload, first_timestep = self._local_observations.get(timeout=0.1)
            except Empty:
                continue
            try:
                assert self._local_policy is not None
                actions = self._local_policy.predict(payload)
                timed_actions = [
                    TimedAction(
                        timestamp=time.time(),
                        timestep=first_timestep + index,
                        action=torch.from_numpy(action.copy()),
                    )
                    for index, action in enumerate(actions)
                ]
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                self.must_go.set()
            except Exception:  # noqa: BLE001 - terminate control after policy failure
                self.logger.exception("Local SO-101 policy inference failed")
                self.shutdown_event.set()
            finally:
                self._local_inference_pending.clear()

    def _ready_to_send_observation(self) -> bool:
        if self._local_policy is not None and self._local_inference_pending.is_set():
            return False
        return super()._ready_to_send_observation()

    def stop(self):
        """标记客户端退出，再关闭通道和设备。"""
        self._closing.set()
        super().stop()

    def attach_leader(
        self,
        leader: SO101LeaderExpert,
        hold_s: float,
        align_duration_s: float,
    ) -> None:
        """接入主臂读取器；主臂串口只在 DAgger 模式打开。"""
        if hold_s < 0 or align_duration_s < 0:
            raise ValueError(
                "intervention hold and alignment durations must be non-negative"
            )
        self._leader_expert = leader
        self._intervention_hold_s = float(hold_s)
        self._leader_align_duration_s = float(align_duration_s)

    def request_intervention(self) -> bool:
        """先冻结从臂，再完成主臂对齐和握稳倒计时。"""
        with self._intervention_lock:
            if (
                self.intervention_enabled.is_set()
                or self.intervention_requested.is_set()
            ):
                return False
            self._alignment_started = False
            self.intervention_requested.set()
        # 先立刻设置请求标志，再读取当前位置。控制线程看到标志后只消费
        # 队列，不再向从臂写入动作，避免接管按键与策略写入并发抢串口。
        self.clear_actions()
        try:
            raw = self.read_robot_observation()
            self._intervention_target = {
                f"{joint}.pos": float(raw[f"{joint}.pos"]) for joint in JOINT_NAMES
            }
        except Exception:
            with self._intervention_lock:
                self.intervention_requested.clear()
            raise
        # 保持观测发送，供云端策略生成影子动作；动作队列在接管期间不再下发。
        print(
            f"[接管] 从臂已保持当前位置；主臂将在 {self._leader_align_duration_s:.1f} 秒内对齐，"
            "请勿触碰主臂",
            flush=True,
        )
        if self._leader_expert is None:
            self.cancel_intervention()
            raise RuntimeError("DAgger 接管未连接主臂读取器。")
        with self._intervention_lock:
            self._alignment_started = True
        try:
            # 对齐阶段主臂保持扭矩，操作者此时不能握持，避免人机对抗。
            self._leader_expert.align_to_native_action(
                self._intervention_target,
                duration_s=self._leader_align_duration_s,
                fps=30.0,
            )
        except Exception:
            self.cancel_intervention()
            raise
        for remaining in range(int(self._intervention_hold_s), 0, -1):
            print(f"[接管] 请握稳主臂，{remaining} 秒后开始人工控制", flush=True)
            time.sleep(1.0)
        # 倒计时结束才释放扭矩，随后读取主臂动作控制从臂。
        self._leader_expert.release_for_manual()
        with self._intervention_lock:
            self.intervention_enabled.set()
            self.intervention_requested.clear()
        print("[接管] 主臂已释放；现在可以操作主臂", flush=True)
        return True

    def cancel_intervention(self) -> None:
        """清除接管状态并恢复策略控制。"""
        with self._intervention_lock:
            self.intervention_requested.clear()
            self.intervention_enabled.clear()
            self._intervention_target = None
            self._alignment_started = False
        self.clear_actions()

    def _send_leader_action(self) -> np.ndarray:
        if self._leader_expert is None:
            raise RuntimeError("DAgger 接管未连接主臂读取器。")
        normalized = self._leader_expert.get_action()
        action = self._to_lerobot_action(normalized)
        self.send_robot_action(self._action_tensor_to_action_dict(action))
        self._last_expert_action = action.detach().cpu().numpy()
        self._last_executed_action = self._last_expert_action.copy()
        return action

    def set_record_callback(self, callback) -> None:
        """注册一帧 DAgger 记录回调。"""
        self._record_callback = callback

    def finish_shutdown(self) -> None:
        """接收线程结束后移除本次会话的退出日志过滤器。"""
        self.logger.removeFilter(self._shutdown_log_filter)

    @staticmethod
    def _to_lerobot_action(action):
        """把云端 RLinf 归一化 action 转成 LeRobot 原生电机单位。"""
        import torch

        values = torch.as_tensor(action, dtype=torch.float32).clone()
        if values.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"SO-101 action 必须是 ({len(JOINT_NAMES)},)，得到 {tuple(values.shape)}"
            )
        values[:-1] *= 100.0
        values[-1] = (values[-1] + 1.0) * 50.0
        return values

    def control_loop_action(self, verbose: bool = False):
        """执行官方队列中的下一步动作，并集中完成 SO-101 单位转换。"""
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            timed_action = self.action_queue.get_nowait()
        action = self._to_lerobot_action(timed_action.get_action())
        self.send_robot_action(self._action_tensor_to_action_dict(action))
        self._last_policy_action = action.detach().cpu().numpy()
        self._last_executed_action = self._last_policy_action.copy()
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        return action

    def consume_policy_action_without_send(self) -> np.ndarray | None:
        """消费影子策略动作但跳过机械臂写入，避免接管阶段队列堆积。"""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            timed_action = self.action_queue.get_nowait()
        action = self._to_lerobot_action(timed_action.get_action())
        self._last_policy_action = action.detach().cpu().numpy()
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        return self._last_policy_action

    def control_loop_observation(self, task: str, verbose: bool = False):
        """把 LeRobot 原始 SO-101 观测转换为 RLinf 输入字段。"""
        raw = self.read_robot_observation()
        native = np.asarray(
            [raw[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float32
        )
        state = native.copy()
        state[:-1] /= 100.0
        state[-1] = state[-1] / 50.0 - 1.0
        state = np.clip(state, -1.0, 1.0)
        if "wrist" not in raw:
            raise KeyError("SO-101 官方相机观测缺少 wrist 字段。")
        payload = {
            "observation.state": state.astype(np.float32),
            "observation.images.wrist": np.asarray(raw["wrist"], dtype=np.uint8),
            "task": task,
        }
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self.action_queue_lock:
            current_queue_size = self.action_queue.qsize()
        if self._local_policy is not None:
            first_timestep = max(latest_action + 1, 0)
            self._local_inference_pending.set()
            self._local_observations.put_nowait((payload, first_timestep))
            self.must_go.clear()
            sent_timestep = first_timestep
        else:
            observation = TimedObservation(
                timestamp=time.time(),
                observation=payload,
                timestep=max(latest_action + 1, 0),
            )
            observation.must_go = self.must_go.is_set() and self.action_queue.empty()
            self.send_observation(observation)
            if observation.must_go:
                self.must_go.clear()
            sent_timestep = observation.get_timestep()
        if verbose:
            print(
                f"[policy] observation timestep={sent_timestep} queue={current_queue_size}"
            )
        return payload

    def read_robot_observation(self):
        """串行读取从臂观测，并重试瞬时串口占用。"""
        return self._robot_io_call(self.robot.get_observation)

    def send_robot_action(self, action: dict[str, float]) -> None:
        """串行发送从臂目标，并重试瞬时串口占用。"""
        self._robot_io_call(self.robot.send_action, action)

    def _robot_io_call(self, operation, *args):
        """保护 LeRobot 串口调用，避免观测和写入互相抢占设备。"""
        with self._robot_io_lock:
            for attempt in range(3):
                try:
                    return operation(*args)
                except ConnectionError as exc:
                    message = str(exc)
                    transient = (
                        "Port is in use" in message
                        or "Incorrect status packet" in message
                    )
                    if not transient or attempt == 2:
                        raise
                    time.sleep(0.05 * (attempt + 1))
        raise RuntimeError("从臂串口操作重试异常结束")

    def clear_actions(self) -> None:
        """清除暂停前尚未执行的动作，并让下一帧强制触发推理。"""
        with self.action_queue_lock:
            self.action_queue.queue.clear()
        self.must_go.set()

    def pause(self) -> None:
        """暂停动作执行，保持机械臂当前位置。"""
        self.inference_enabled.clear()
        self.clear_actions()

    def resume(self) -> None:
        """从最新观测恢复推理。"""
        self.clear_actions()
        self.inference_enabled.set()

    def set_recording(self, enabled: bool) -> None:
        """控制 DAgger 记录门；对齐、握稳和复位阶段保持关闭。"""
        if enabled:
            self.recording_enabled.set()
        else:
            self.recording_enabled.clear()

    def control_loop(self, task: str, verbose: bool = False):
        """复用官方收发方法，仅在启用状态推进控制周期。"""
        self.start_barrier.wait()
        while self.running:
            started_at = time.perf_counter()
            observation_payload = None
            intervened = False
            try:
                if self.intervention_enabled.is_set():
                    # 主臂接管阶段停止发送云端动作，只执行最新主臂位置。
                    self.consume_policy_action_without_send()
                    self._send_leader_action()
                    intervened = True
                    if self._ready_to_send_observation():
                        observation_payload = self.control_loop_observation(
                            task, verbose
                        )
                elif self.inference_enabled.is_set():
                    if self.actions_available():
                        if self.intervention_requested.is_set():
                            self.consume_policy_action_without_send()
                        else:
                            self.control_loop_action(verbose)
                    if self._ready_to_send_observation():
                        observation_payload = self.control_loop_observation(
                            task, verbose
                        )
            except ConnectionError as exc:
                if "Port is in use" not in str(exc):
                    raise
                # 主臂/从臂切换瞬间仍可能被底层驱动拒绝一次；保持线程存活，
                # 下一周期重新读取并发送，避免人工接管后从臂控制永久停止。
                time.sleep(0.05)
                continue
            if (
                observation_payload is not None
                and self._record_callback is not None
                and self.recording_enabled.is_set()
            ):
                self._record_callback(
                    observation_payload,
                    self._last_policy_action.copy(),
                    self._last_expert_action.copy(),
                    self._last_executed_action.copy(),
                    intervened,
                )
            time.sleep(
                max(
                    0.0, self.config.environment_dt - (time.perf_counter() - started_at)
                )
            )


def _start_client_threads(
    client: _ControllableRobotClient, task: str, *, receive_remote_actions: bool
) -> tuple[Thread, Thread]:
    """Start the control loop and the selected policy worker."""
    control = Thread(target=client.control_loop, args=(task,), daemon=True)
    receiver_target = (
        client.receive_actions if receive_remote_actions else client.run_local_policy
    )
    receiver = Thread(target=receiver_target, daemon=True)
    receiver.start()
    control.start()
    return receiver, control


# The runtime retains LeRobot's queue and timestep behavior for both backends.
SO101PolicyRuntime = _ControllableRobotClient


class _ExpectedShutdownRpcFilter(logging.Filter):
    """只过滤主动退出导致的 CANCELLED，保留其他 RPC 错误。"""

    def __init__(self, closing: Event):
        super().__init__()
        self.closing = closing

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.closing.is_set():
            return True
        message = record.getMessage()
        return not (
            record.levelno >= logging.ERROR
            and "Error receiving actions" in message
            and ("StatusCode.CANCELLED" in message or "Channel closed!" in message)
        )


def _start_pose_motion(
    client: RobotClient,
    pose: dict[str, float],
    duration: float,
    fps: int,
) -> tuple[Thread, list[Exception]]:
    """后台执行起始姿态移动，让云端加载策略与本地复位并行。"""
    errors: list[Exception] = []

    def move() -> None:
        try:
            _send_pose(client, pose, duration, fps)
        except Exception as exc:  # 交给主线程统一处理，避免静默退出。
            errors.append(exc)

    thread = Thread(target=move, name="so101-pose-motion", daemon=True)
    thread.start()
    return thread, errors


class _InteractiveKeys:
    """单字符控制，避免用户每次命令都要按回车。"""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("交互运行需要从可读终端启动。")
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read(self) -> str:
        while True:
            readable, _, _ = select.select([self._fd], [], [])
            if readable:
                return os.read(self._fd, 1).decode(errors="ignore").lower()


def _ask_yes_no(keys: _InteractiveKeys, prompt: str, default: bool) -> bool:
    """读取一次 y/n 结果，用于记录本次推理是否计入统计。"""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        print(f"{prompt} {suffix} ", end="", flush=True)
        answer = keys.read()
        if answer in {"y", "是"}:
            print("是", flush=True)
            return True
        if answer in {"n", "否"}:
            print("否", flush=True)
            return False
        print("请输入 y 或 n。", flush=True)


def _record_run_result(
    keys: _InteractiveKeys, stats: dict[str, int], run_number: int
) -> None:
    """询问并累计一次已暂停的推理结果。"""
    if _ask_yes_no(keys, f"第 {run_number} 次执行是否计入总次数？", True):
        success = _ask_yes_no(keys, f"第 {run_number} 次执行是否成功？", False)
        stats["total"] += 1
        stats["success"] += int(success)
    total = stats["total"]
    rate = 100.0 * stats["success"] / total if total else 0.0
    print(
        f"[统计] 计入 {total} 次，成功 {stats['success']} 次，成功率 {rate:.1f}%",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    required = {
        "--serial-port": args.serial_port,
        "--robot-id": args.robot_id,
        "--camera": args.camera,
    }
    if args.dagger:
        required.update(
            {"--leader-port": args.leader_port, "--leader-id": args.leader_id}
        )
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "SO-101 hardware values are user-specific; provide "
            + ", ".join(missing)
            + " or set the corresponding SO101_* environment variables."
        )
    if not args.dry_run and not args.allow_hardware_motion:
        raise RuntimeError("真实运行需要显式添加 --allow-hardware-motion。")
    if args.policy_type != "pi05":
        raise ValueError("SO-101 RLinf inference requires --policy-type pi05.")
    if args.actions_per_chunk != 20:
        raise ValueError("SO-101 Pi05 inference requires --actions-per-chunk 20.")
    if args.inference_backend == "local" and not args.dry_run:
        missing_local = [
            name
            for name, value in {
                "--local-checkpoint": args.local_checkpoint,
                "--local-norm-stats": args.local_norm_stats,
            }.items()
            if value is None
        ]
        if missing_local:
            raise RuntimeError(
                "local 推理需要提供 "
                + ", ".join(missing_local)
                + " 或对应的 SO101_LOCAL_* 环境变量。"
            )
    robot_config = build_robot_config(args)
    if args.dry_run:
        print(
            f"[配置] backend={args.inference_backend} server={args.server_address} "
            f"robot={robot_config.port} "
            f"camera={args.camera} policy={args.policy_type} chunk={args.actions_per_chunk} "
            f"dagger={args.dagger}"
        )
        return

    standard, folded = _load_poses(args.pose_file)
    client: _ControllableRobotClient | None = None
    leader_expert: SO101LeaderExpert | None = None
    dagger_session: DaggerSessionController | None = None
    dagger_recorder: DaggerEpisodeRecorder | None = None
    threads: tuple[Thread, ...] | None = None
    connected = False
    keys: _InteractiveKeys | None = None
    stats = {"total": 0, "success": 0}
    run_number = 0
    pose_thread: Thread | None = None
    pose_errors: list[Exception] = []
    policy_thread: Thread | None = None
    policy_result: list[bool] = []
    policy_errors: list[Exception] = []
    try:
        # 从这里开始就由统一 finally 持有清理责任：即使策略启动失败、
        # 主臂初始化或数据目录创建失败，也不会绕过从臂安全收臂。
        client = _ControllableRobotClient(
            RobotClientConfig(
                robot=robot_config,
                server_address=args.server_address,
                policy_type=args.policy_type,
                pretrained_name_or_path=args.pretrained_name_or_path,
                policy_device=args.policy_device,
                client_device="cpu",
                actions_per_chunk=args.actions_per_chunk,
                fps=args.fps,
                chunk_size_threshold=0.5,
            )
        )
        if args.dagger:
            leader_expert = SO101LeaderExpert(
                port=args.leader_port,
                robot_id=args.leader_id,
                calibration_dir=(
                    str(args.leader_calibration_dir)
                    if args.leader_calibration_dir is not None
                    else None
                ),
            )
            client.attach_leader(
                leader_expert,
                hold_s=args.intervention_hold_s,
                align_duration_s=args.leader_align_duration_s,
            )
            dagger_session = DaggerSessionController()
            dagger_root = args.dagger_data_root.expanduser()
            if dagger_root.exists():
                dagger_root = dagger_root.with_name(
                    f"{dagger_root.name}-{time.strftime('%Y%m%d-%H%M%S')}"
                )
            dagger_recorder = DaggerEpisodeRecorder(
                root=dagger_root,
                task=args.task,
                image_shape=(args.camera_height, args.camera_width, 3),
                fps=args.fps,
            )
            client.set_record_callback(dagger_recorder.append)
            print(f"[DAgger] 数据目录：{dagger_root}", flush=True)
        print(
            f"[启动] {args.inference_backend} 后端加载策略；同时进入标准位置",
            flush=True,
        )
        pose_thread, pose_errors = _start_pose_motion(
            client, standard, args.pose_duration, args.fps
        )

        def start_policy() -> None:
            try:
                if args.inference_backend == "local":
                    assert args.local_checkpoint is not None
                    assert args.local_norm_stats is not None
                    client.use_local_policy(
                        SO101LocalPolicyBackend(
                            args.local_checkpoint,
                            args.local_norm_stats,
                            args.policy_device,
                        )
                    )
                    client.shutdown_event.clear()
                    policy_result.append(True)
                else:
                    policy_result.append(client.start())
            except Exception as exc:
                policy_errors.append(exc)

        policy_thread = Thread(
            target=start_policy, name="so101-policy-start", daemon=True
        )
        policy_thread.start()
        pose_thread.join()
        policy_thread.join()
        if pose_errors:
            raise RuntimeError(f"进入标准位置失败：{pose_errors[0]}") from pose_errors[
                0
            ]
        if policy_errors:
            raise RuntimeError(
                f"启动 {args.inference_backend} 策略后端失败：{policy_errors[0]}"
            ) from policy_errors[0]
        if not policy_result or not policy_result[0]:
            raise RuntimeError(f"{args.inference_backend} 策略后端启动失败。")
        connected = True
        threads = _start_client_threads(
            client,
            args.task,
            receive_remote_actions=args.inference_backend == "grpc",
        )
        if args.dagger:
            print(
                "[就绪] 已到标准位置；按 s 开始，策略阶段 c 保存/a 丢弃；完成后按 r 复位，布置场景后按 s；q 退出",
                flush=True,
            )
        else:
            print(
                "[就绪] 已到标准位置；按 s 开始，空格结束当前回合，r 回标准位，q 收回从臂并退出",
                flush=True,
            )
        with _InteractiveKeys() as keys:
            while True:
                command = keys.read()
                if args.dagger:
                    assert dagger_session is not None and dagger_recorder is not None
                    operation = dagger_session.handle(command)
                    if operation == "start_episode":
                        dagger_recorder.start()
                        client.set_recording(True)
                        client.resume()
                        run_number += 1
                        print(
                            f"[DAgger] 第 {run_number} 条开始；空格接管，策略成功按 c 保存，按 a 丢弃",
                            flush=True,
                        )
                    elif operation == "start_handover":
                        client.set_recording(False)
                        client.request_intervention()
                        dagger_session.handover_finished()
                        client.set_recording(True)
                        print("[接管] 人工控制中；按 c 保存，按 a 丢弃", flush=True)
                    elif operation == "save_episode":
                        client.set_recording(False)
                        client.pause()
                        was_intervening = client.intervention_enabled.is_set()
                        # 先停止主臂读取线程，再承接当前位置；随后才写盘。
                        # 这样操作员松手时主臂已经由当前位置目标承重。
                        client.cancel_intervention()
                        if leader_expert is not None and was_intervening:
                            leader_expert.hold_current()
                        count = dagger_recorder.save(success=True)
                        print(
                            f"[保存] 第 {run_number} 条已保存（{count} 帧）；按 r 复位，布置场景后按 s，按 q 退出",
                            flush=True,
                        )
                    elif operation in {"discard_episode", "discard_episode_after_hold"}:
                        client.set_recording(False)
                        client.pause()
                        was_intervening = client.intervention_enabled.is_set()
                        client.cancel_intervention()
                        if leader_expert is not None and was_intervening:
                            leader_expert.hold_current()
                        if operation == "discard_episode_after_hold":
                            for remaining in range(3, 0, -1):
                                print(
                                    f"[放弃] 两臂保持当前位置，请撤离双手，{remaining} 秒后完成",
                                    flush=True,
                                )
                                time.sleep(1.0)
                        dagger_recorder.discard()
                        client.intervention_enabled.clear()
                        print(
                            "[放弃] 当前 episode 已丢弃；按 r 复位，布置场景后按 s，按 q 退出",
                            flush=True,
                        )
                    elif operation == "reset_only":
                        client.cancel_intervention()
                        client.pause()
                        if leader_expert is not None:
                            leader_expert.move_to_native_action(
                                folded,
                                duration_s=args.fold_duration_s,
                                fps=args.fps,
                                release_torque=True,
                            )
                        _send_pose(client, standard, args.pose_duration, args.fps)
                        dagger_session.reset_finished()
                        print(
                            "[准备] 场景布置完成；按 s 开始下一条，按 q 收臂退出",
                            flush=True,
                        )
                    elif operation == "fold_and_exit":
                        client.set_recording(False)
                        was_intervening = client.intervention_enabled.is_set()
                        if dagger_recorder.active:
                            dagger_recorder.discard()
                        client.cancel_intervention()
                        client.pause()
                        if was_intervening and leader_expert is not None:
                            leader_expert.hold_current()
                            for remaining in range(3, 0, -1):
                                print(
                                    f"[退出] 主臂保持当前位置，请撤离双手，{remaining} 秒后收臂",
                                    flush=True,
                                )
                                time.sleep(1.0)
                        break
                    else:
                        if command in {"c", "a", "r", "q", "s", " "}:
                            print(
                                f"[提示] 当前状态 {dagger_session.state.value}，按键无效",
                                flush=True,
                            )
                    continue
                if command == "s" and not client.inference_enabled.is_set():
                    run_number += 1
                    print(
                        f"[推理] 第 {run_number} 次执行；"
                        "按空格结束当前回合，按 r 归位，按 q 退出",
                        flush=True,
                    )
                    client.cancel_intervention()
                    client.resume()
                elif command == " ":
                    if client.inference_enabled.is_set():
                        client.pause()
                        print("[暂停] 保持当前位置", flush=True)
                        _record_run_result(keys, stats, run_number)
                    else:
                        print("[提示] 当前未在推理，按 s 开始", flush=True)
                elif command == "r":
                    was_running = client.inference_enabled.is_set()
                    client.cancel_intervention()
                    client.pause()
                    if was_running and not args.dagger:
                        _record_run_result(keys, stats, run_number)
                    print("[复位] 正在进入标准位置", flush=True)
                    _send_pose(client, standard, args.pose_duration, args.fps)
                elif command == "q":
                    was_running = client.inference_enabled.is_set()
                    client.cancel_intervention()
                    client.pause()
                    if was_running and not args.dagger:
                        _record_run_result(keys, stats, run_number)
                    break
    finally:
        # ``RobotClient.__init__`` connects and enables the follower before
        # ``client.start()`` contacts the policy server.  Therefore a failed
        # server connection must still fold the follower before releasing
        # torque; checking only ``connected`` would let the arm drop.
        robot = getattr(client, "robot", None)
        robot_connected = bool(getattr(robot, "is_connected", False))
        if robot_connected:
            try:
                client.pause()
                if args.dagger:
                    client.cancel_intervention()
                if leader_expert is not None:
                    print("[收臂] 主臂正在进入折叠位置", flush=True)
                    leader_expert.move_to_native_action(
                        folded,
                        duration_s=args.fold_duration_s,
                        fps=args.fps,
                        release_torque=True,
                    )
                print("[收臂] 从臂正在进入折叠位置", flush=True)
                _send_pose(client, folded, args.pose_duration, args.fps)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve original failure
                print(f"[安全收臂失败] {cleanup_error}", flush=True)
            finally:
                try:
                    client.stop()
                except Exception as stop_error:  # noqa: BLE001 - best-effort cleanup
                    print(f"[断开设备失败] {stop_error}", flush=True)
                if threads is not None:
                    for thread in threads:
                        thread.join(timeout=3.0)
                try:
                    client.finish_shutdown()
                except Exception as filter_error:  # noqa: BLE001 - best-effort cleanup
                    print(f"[日志清理失败] {filter_error}", flush=True)
        if leader_expert is not None:
            leader_expert.close()
        if dagger_recorder is not None:
            dagger_recorder.close()
        if connected and not args.dagger:
            total = stats["total"]
            rate = 100.0 * stats["success"] / total if total else 0.0
            print(
                f"[统计] 本次会话：计入 {total} 次，成功 {stats['success']} 次，成功率 {rate:.1f}%",
                flush=True,
            )


if __name__ == "__main__":
    main()
