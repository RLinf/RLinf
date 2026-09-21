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

"""Run SO-101 inference with a local model or LeRobot gRPC service.

Both backends share pose preparation, action conversion, the control loop, and
DAgger intervention handling. The ``grpc`` backend uses LeRobot's
``RobotClient`` protocol; ``local`` loads the RLinf model in this process.
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

import numpy as np
import torch
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction, TimedObservation
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so_follower import SO101FollowerConfig

from examples.embodiment.so101.dagger import (
    DaggerEpisodeRecorder,
    DaggerSessionController,
)
from examples.embodiment.so101.leader import SO101LeaderExpert
from examples.embodiment.so101.policy_backend import SO101LocalPolicyBackend

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
        default=os.environ.get("SO101_SERVER_ADDRESS", "127.0.0.1:50051"),
        help="LeRobot policy server endpoint used by the gRPC backend.",
    )
    parser.add_argument("--task", default=os.environ.get("SO101_TASK"))
    parser.add_argument("--serial-port", default=os.environ.get("SO101_SERIAL_PORT"))
    parser.add_argument("--robot-id", default=os.environ.get("SO101_ROBOT_ID"))
    parser.add_argument("--calibration-dir", type=Path, default=None)
    parser.add_argument("--camera", default=os.environ.get("SO101_CAMERA"))
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=15)
    pose_file = os.environ.get("SO101_POSE_FILE")
    parser.add_argument(
        "--pose-file",
        type=Path,
        default=Path(pose_file) if pose_file else None,
        help="JSON file containing standard_position and fold_position.",
    )
    parser.add_argument("--pose-duration", type=float, default=1.5)
    parser.add_argument(
        "--fold-duration-s",
        type=float,
        default=5.0,
        help="Seconds used to move the leader to its folded pose.",
    )
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--actions-per-chunk", type=int, default=20)
    parser.add_argument("--policy-type", default="pi05")
    parser.add_argument("--pretrained-name-or-path", default="rlinf-so101-backend")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument(
        "--inference-backend",
        choices=("grpc", "local"),
        default=os.environ.get("SO101_INFERENCE_BACKEND", "grpc"),
        help="Use a gRPC policy server or load the model in this process.",
    )
    parser.add_argument(
        "--local-checkpoint",
        type=Path,
        default=os.environ.get("SO101_LOCAL_CHECKPOINT"),
        help="RLinf actor checkpoint used by the local backend.",
    )
    parser.add_argument(
        "--local-norm-stats",
        type=Path,
        default=os.environ.get("SO101_LOCAL_NORM_STATS"),
        help="OpenPI norm_stats.json used by the local backend.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate arguments without connecting to hardware or a server.",
    )
    parser.add_argument("--allow-hardware-motion", action="store_true")
    parser.add_argument(
        "--dagger",
        action="store_true",
        help="Enable leader-arm intervention and DAgger recording.",
    )
    parser.add_argument("--leader-port", default=os.environ.get("SO101_LEADER_PORT"))
    parser.add_argument("--leader-id", default=os.environ.get("SO101_LEADER_ID"))
    parser.add_argument("--leader-calibration-dir", type=Path, default=None)
    parser.add_argument(
        "--intervention-hold-s",
        type=float,
        default=3.0,
        help="Seconds to hold the aligned leader before manual control.",
    )
    parser.add_argument(
        "--leader-align-duration-s",
        type=float,
        default=3.0,
        help="Seconds used to align the leader with the follower.",
    )
    parser.add_argument(
        "--dagger-data-root",
        type=Path,
        default=Path(os.environ.get("SO101_DAGGER_DATA_ROOT", "outputs/so101-dagger")),
        help="Destination for the DAgger LeRobot v3 dataset.",
    )
    return parser.parse_args()


def _load_poses(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    """Load standard and folded poses expressed in LeRobot native units."""
    data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    if (
        data.get("joint_order") != list(JOINT_NAMES)
        or data.get("units") != "lerobot_native"
    ):
        raise ValueError(
            "Pose file must use the SO-101 joint order and lerobot_native "
            f"units: {path}"
        )
    result = []
    for name in ("standard_position", "fold_position"):
        pose = data.get(name)
        if not isinstance(pose, dict) or set(pose) != set(JOINT_NAMES):
            raise ValueError(f"Pose file has an incomplete {name}: {path}")
        result.append({joint: float(pose[joint]) for joint in JOINT_NAMES})
    return result[0], result[1]


def build_robot_config(args: argparse.Namespace) -> SO101FollowerConfig:
    """Build the LeRobot SO-101 configuration used by the control loop."""
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
        # The LeRobot action queue handles latency; pose setup uses absolute targets.
        max_relative_target=None,
    )


def _send_pose(
    client: RobotClient, pose: dict[str, float], duration: float, fps: int
) -> None:
    """Send an absolute pose along a smooth interpolation."""
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
    """Add pause and intervention controls to LeRobot's client loop."""

    def __init__(self, config):
        super().__init__(config)
        self.inference_enabled = Event()
        # Observation, action, and reset calls share the follower serial bus.
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
        """Mark expected shutdown before closing the channel and devices."""
        self._closing.set()
        super().stop()

    def attach_leader(
        self,
        leader: SO101LeaderExpert,
        hold_s: float,
        align_duration_s: float,
    ) -> None:
        """Attach the leader reader used during DAgger intervention."""
        if hold_s < 0 or align_duration_s < 0:
            raise ValueError(
                "intervention hold and alignment durations must be non-negative"
            )
        self._leader_expert = leader
        self._intervention_hold_s = float(hold_s)
        self._leader_align_duration_s = float(align_duration_s)

    def request_intervention(self) -> bool:
        """Hold the follower, align the leader, and begin intervention."""
        with self._intervention_lock:
            if (
                self.intervention_enabled.is_set()
                or self.intervention_requested.is_set()
            ):
                return False
            self._alignment_started = False
            self.intervention_requested.set()
        # Stop follower writes before reading its target for leader alignment.
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
        # Continue observations and consume shadow policy actions during intervention.
        print(
            f"[Intervention] Follower held; leader alignment takes "
            f"{self._leader_align_duration_s:.1f}s. Keep hands clear.",
            flush=True,
        )
        if self._leader_expert is None:
            self.cancel_intervention()
            raise RuntimeError("DAgger intervention requires an attached leader.")
        with self._intervention_lock:
            self._alignment_started = True
        try:
            # Keep leader torque enabled until alignment completes.
            self._leader_expert.align_to_native_action(
                self._intervention_target,
                duration_s=self._leader_align_duration_s,
                fps=30.0,
            )
        except Exception:
            self.cancel_intervention()
            raise
        for remaining in range(int(self._intervention_hold_s), 0, -1):
            print(
                f"[Intervention] Hold the leader; manual control starts in "
                f"{remaining}s.",
                flush=True,
            )
            time.sleep(1.0)
        # Release torque only when the operator can take control.
        self._leader_expert.release_for_manual()
        with self._intervention_lock:
            self.intervention_enabled.set()
            self.intervention_requested.clear()
        print("[Intervention] Leader released for manual control.", flush=True)
        return True

    def cancel_intervention(self) -> None:
        """Clear intervention state before policy control resumes."""
        with self._intervention_lock:
            self.intervention_requested.clear()
            self.intervention_enabled.clear()
            self._intervention_target = None
            self._alignment_started = False
        self.clear_actions()

    def _send_leader_action(self) -> np.ndarray:
        if self._leader_expert is None:
            raise RuntimeError("DAgger intervention requires an attached leader.")
        normalized = self._leader_expert.get_action()
        action = self._to_lerobot_action(normalized)
        self.send_robot_action(self._action_tensor_to_action_dict(action))
        self._last_expert_action = action.detach().cpu().numpy()
        self._last_executed_action = self._last_expert_action.copy()
        return action

    def set_record_callback(self, callback) -> None:
        """Register the callback that records one DAgger frame."""
        self._record_callback = callback

    def finish_shutdown(self) -> None:
        """Remove the expected-shutdown log filter after threads exit."""
        self.logger.removeFilter(self._shutdown_log_filter)

    @staticmethod
    def _to_lerobot_action(action):
        """Convert a normalized RLinf action to native LeRobot units."""
        import torch

        values = torch.as_tensor(action, dtype=torch.float32).clone()
        if values.shape != (len(JOINT_NAMES),):
            raise ValueError(
                f"SO-101 action must be ({len(JOINT_NAMES)},), got {tuple(values.shape)}"
            )
        values[:-1] *= 100.0
        values[-1] = (values[-1] + 1.0) * 50.0
        return values

    def control_loop_action(self, verbose: bool = False):
        """Execute the next queued action after converting SO-101 units."""
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
        """Consume a shadow policy action without writing to the follower."""
        with self.action_queue_lock:
            if self.action_queue.empty():
                return None
            timed_action = self.action_queue.get_nowait()
        action = self._to_lerobot_action(timed_action.get_action())
        self._last_policy_action = action.detach().cpu().numpy()
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        return self._last_policy_action

    def control_loop_observation(
        self, task: str, verbose: bool = False, *, send_to_policy: bool = True
    ):
        """Capture one observation and optionally submit it for inference."""
        raw = self.read_robot_observation()
        native = np.asarray(
            [raw[f"{joint}.pos"] for joint in JOINT_NAMES], dtype=np.float32
        )
        state = native.copy()
        state[:-1] /= 100.0
        state[-1] = state[-1] / 50.0 - 1.0
        state = np.clip(state, -1.0, 1.0)
        if "wrist" not in raw:
            raise KeyError("SO-101 camera observation is missing the wrist field.")
        payload = {
            "observation.state": state.astype(np.float32),
            "observation.images.wrist": np.asarray(raw["wrist"], dtype=np.uint8),
            "task": task,
        }
        with self.latest_action_lock:
            latest_action = self.latest_action
        with self.action_queue_lock:
            current_queue_size = self.action_queue.qsize()
        if not send_to_policy:
            return payload
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
        """Read the follower while retrying transient serial contention."""
        return self._robot_io_call(self.robot.get_observation)

    def send_robot_action(self, action: dict[str, float]) -> None:
        """Send a follower target while retrying transient serial contention."""
        self._robot_io_call(self.robot.send_action, action)

    def _robot_io_call(self, operation, *args):
        """Serialize a LeRobot call on the follower bus."""
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
        raise RuntimeError("SO-101 follower I/O retries ended unexpectedly")

    def clear_actions(self) -> None:
        """Drop queued actions and force inference on the next observation."""
        with self.action_queue_lock:
            self.action_queue.queue.clear()
        self.must_go.set()

    def pause(self) -> None:
        """Pause action execution and hold the follower."""
        self.inference_enabled.clear()
        self.clear_actions()

    def resume(self) -> None:
        """Resume inference from the latest observation."""
        self.clear_actions()
        self.inference_enabled.set()

    def set_recording(self, enabled: bool) -> None:
        """Enable or disable DAgger frame recording."""
        if enabled:
            self.recording_enabled.set()
        else:
            self.recording_enabled.clear()

    def control_loop(self, task: str, verbose: bool = False):
        """Advance LeRobot's control loop while inference is enabled."""
        self.start_barrier.wait()
        while self.running:
            started_at = time.perf_counter()
            observation_payload = None
            intervened = False
            try:
                if self.intervention_enabled.is_set():
                    # Execute leader actions and consume policy actions as shadows.
                    self.consume_policy_action_without_send()
                    self._send_leader_action()
                    intervened = True
                    send_to_policy = self._ready_to_send_observation()
                    if send_to_policy or self.recording_enabled.is_set():
                        observation_payload = self.control_loop_observation(
                            task, verbose, send_to_policy=send_to_policy
                        )
                elif self.inference_enabled.is_set():
                    if self.actions_available():
                        if self.intervention_requested.is_set():
                            self.consume_policy_action_without_send()
                        else:
                            self.control_loop_action(verbose)
                    send_to_policy = self._ready_to_send_observation()
                    if send_to_policy or self.recording_enabled.is_set():
                        observation_payload = self.control_loop_observation(
                            task, verbose, send_to_policy=send_to_policy
                        )
            except ConnectionError as exc:
                if "Port is in use" not in str(exc):
                    raise
                # A handover may briefly contend on the serial bus; retry next cycle.
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
    """Filter CANCELLED messages produced by an expected shutdown."""

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
    """Move to the initial pose while the policy backend loads."""
    errors: list[Exception] = []

    def move() -> None:
        try:
            _send_pose(client, pose, duration, fps)
        except Exception as exc:  # Report the worker failure on the main thread.
            errors.append(exc)

    thread = Thread(target=move, name="so101-pose-motion", daemon=True)
    thread.start()
    return thread, errors


class _InteractiveKeys:
    """Read single-character controls from an interactive terminal."""

    def __enter__(self):
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive control requires a readable terminal.")
        self._fd = sys.stdin.fileno()
        self._old_settings = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, *_args) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_settings)

    def read(self, timeout: float | None = None) -> str | None:
        """Read one key, returning ``None`` when a timeout expires."""
        while True:
            readable, _, _ = select.select([self._fd], [], [], timeout)
            if readable:
                return os.read(self._fd, 1).decode(errors="ignore").lower()
            return None


def _ask_yes_no(keys: _InteractiveKeys, prompt: str, default: bool) -> bool:
    """Read a yes/no answer for inference-run accounting."""
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        print(f"{prompt} {suffix} ", end="", flush=True)
        answer = keys.read()
        if answer == "y":
            print("yes", flush=True)
            return True
        if answer == "n":
            print("no", flush=True)
            return False
        print("Enter y or n.", flush=True)


def _record_run_result(
    keys: _InteractiveKeys, stats: dict[str, int], run_number: int
) -> None:
    """Prompt for and accumulate the result of one paused inference run."""
    if _ask_yes_no(keys, f"Count run {run_number}?", True):
        success = _ask_yes_no(keys, f"Was run {run_number} successful?", False)
        stats["total"] += 1
        stats["success"] += int(success)
    total = stats["total"]
    rate = 100.0 * stats["success"] / total if total else 0.0
    print(
        f"[Stats] Counted {total} runs, {stats['success']} successful ({rate:.1f}%).",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    required = {
        "--task": args.task,
        "--serial-port": args.serial_port,
        "--robot-id": args.robot_id,
        "--camera": args.camera,
    }
    if not args.dry_run:
        required["--pose-file"] = args.pose_file
    if args.dagger:
        required.update(
            {"--leader-port": args.leader_port, "--leader-id": args.leader_id}
        )
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Provide " + ", ".join(missing) + " or its corresponding SO101_* variable."
        )
    if not args.dry_run and not args.allow_hardware_motion:
        raise RuntimeError("Add --allow-hardware-motion to start the robot.")
    if args.policy_type != "pi05":
        raise ValueError("SO-101 RLinf inference requires --policy-type pi05.")
    if args.actions_per_chunk != 20:
        raise ValueError("SO-101 Pi05 inference requires --actions-per-chunk 20.")
    positive = {
        "--fps": args.fps,
        "--camera-fps": args.camera_fps,
        "--camera-width": args.camera_width,
        "--camera-height": args.camera_height,
    }
    invalid_positive = [name for name, value in positive.items() if value <= 0]
    if invalid_positive:
        raise ValueError(
            "These values must be positive: " + ", ".join(invalid_positive)
        )
    non_negative = {
        "--pose-duration": args.pose_duration,
        "--fold-duration-s": args.fold_duration_s,
        "--intervention-hold-s": args.intervention_hold_s,
        "--leader-align-duration-s": args.leader_align_duration_s,
    }
    invalid_non_negative = [name for name, value in non_negative.items() if value < 0]
    if invalid_non_negative:
        raise ValueError(
            "These values must be non-negative: " + ", ".join(invalid_non_negative)
        )
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
                "Local inference requires "
                + ", ".join(missing_local)
                + " or the corresponding SO101_LOCAL_* variables."
            )
    robot_config = build_robot_config(args)
    if args.dry_run:
        print(
            f"[Config] backend={args.inference_backend} server={args.server_address} "
            f"robot={robot_config.port} "
            f"camera={args.camera} policy={args.policy_type} chunk={args.actions_per_chunk} "
            f"dagger={args.dagger}"
        )
        return

    assert args.pose_file is not None
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
        # This block owns shutdown after the follower is connected.
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
            print(f"[DAgger] Dataset: {dagger_root}", flush=True)
        print(
            f"[Startup] Loading the {args.inference_backend} backend while moving "
            "to the standard pose.",
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
            raise RuntimeError(
                f"Failed to reach the standard pose: {pose_errors[0]}"
            ) from pose_errors[0]
        if policy_errors:
            raise RuntimeError(
                f"Failed to start the {args.inference_backend} backend: "
                f"{policy_errors[0]}"
            ) from policy_errors[0]
        if not policy_result or not policy_result[0]:
            raise RuntimeError(f"Failed to start the {args.inference_backend} backend.")
        connected = True
        threads = _start_client_threads(
            client,
            args.task,
            receive_remote_actions=args.inference_backend == "grpc",
        )
        if args.dagger:
            print(
                "[Ready] Press s to start; c saves and a discards an episode; "
                "r resets and q exits.",
                flush=True,
            )
        else:
            print(
                "[Ready] Press s to start, space to end a run, r to reset, "
                "and q to fold and exit.",
                flush=True,
            )
        with _InteractiveKeys() as keys:
            while True:
                command = keys.read(timeout=0.25)
                if command is None:
                    if not client.running:
                        raise RuntimeError("The policy runtime stopped unexpectedly.")
                    continue
                if args.dagger:
                    assert dagger_session is not None and dagger_recorder is not None
                    operation = dagger_session.handle(command)
                    if operation == "start_episode":
                        dagger_recorder.start()
                        client.set_recording(True)
                        client.resume()
                        run_number += 1
                        print(
                            f"[DAgger] Episode {run_number} started; press space to "
                            "intervene, c to save, or a to discard.",
                            flush=True,
                        )
                    elif operation == "start_handover":
                        client.set_recording(False)
                        client.request_intervention()
                        dagger_session.handover_finished()
                        client.set_recording(True)
                        print(
                            "[Intervention] Manual control active; press c to save "
                            "or a to discard.",
                            flush=True,
                        )
                    elif operation == "save_episode":
                        client.set_recording(False)
                        client.pause()
                        was_intervening = client.intervention_enabled.is_set()
                        # Hold the leader before the operator releases it and data is saved.
                        client.cancel_intervention()
                        if leader_expert is not None and was_intervening:
                            leader_expert.hold_current()
                        count = dagger_recorder.save(success=True)
                        print(
                            f"[Saved] Episode {run_number} contains {count} frames; "
                            "press r to reset or q to exit.",
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
                                    f"[Discard] Both arms held; keep hands clear for "
                                    f"{remaining}s.",
                                    flush=True,
                                )
                                time.sleep(1.0)
                        dagger_recorder.discard()
                        client.intervention_enabled.clear()
                        print(
                            "[Discarded] Press r to reset or q to exit.",
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
                            "[Ready] Arrange the scene, then press s to start or q to exit.",
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
                                    f"[Exit] Leader held; keep hands clear for {remaining}s.",
                                    flush=True,
                                )
                                time.sleep(1.0)
                        break
                    else:
                        if command in {"c", "a", "r", "q", "s", " "}:
                            print(
                                f"[Input] Key ignored in state {dagger_session.state.value}.",
                                flush=True,
                            )
                    continue
                if command == "s" and not client.inference_enabled.is_set():
                    run_number += 1
                    print(
                        f"[Inference] Run {run_number}; press space to finish, "
                        "r to reset, or q to exit.",
                        flush=True,
                    )
                    client.cancel_intervention()
                    client.resume()
                elif command == " ":
                    if client.inference_enabled.is_set():
                        client.pause()
                        print("[Paused] Holding the current pose.", flush=True)
                        _record_run_result(keys, stats, run_number)
                    else:
                        print("[Input] Press s to start inference.", flush=True)
                elif command == "r":
                    was_running = client.inference_enabled.is_set()
                    client.cancel_intervention()
                    client.pause()
                    if was_running and not args.dagger:
                        _record_run_result(keys, stats, run_number)
                    print("[Reset] Moving to the standard pose.", flush=True)
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
                    print("[Shutdown] Folding the leader.", flush=True)
                    leader_expert.move_to_native_action(
                        folded,
                        duration_s=args.fold_duration_s,
                        fps=args.fps,
                        release_torque=True,
                    )
                print("[Shutdown] Folding the follower.", flush=True)
                _send_pose(client, folded, args.pose_duration, args.fps)
            except Exception as cleanup_error:  # noqa: BLE001 - preserve original failure
                print(f"[Shutdown] Failed to fold safely: {cleanup_error}", flush=True)
            finally:
                try:
                    client.stop()
                except Exception as stop_error:  # noqa: BLE001 - best-effort cleanup
                    print(
                        f"[Shutdown] Failed to disconnect devices: {stop_error}",
                        flush=True,
                    )
                if threads is not None:
                    for thread in threads:
                        thread.join(timeout=3.0)
                try:
                    client.finish_shutdown()
                except Exception as filter_error:  # noqa: BLE001 - best-effort cleanup
                    print(
                        f"[Shutdown] Failed to remove log filter: {filter_error}",
                        flush=True,
                    )
        if leader_expert is not None:
            leader_expert.close()
        if dagger_recorder is not None:
            dagger_recorder.close()
        if connected and not args.dagger:
            total = stats["total"]
            rate = 100.0 * stats["success"] / total if total else 0.0
            print(
                f"[Stats] Session: {total} runs, {stats['success']} successful "
                f"({rate:.1f}%).",
                flush=True,
            )


if __name__ == "__main__":
    main()
