# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect and preview Panda Task Program demonstrations; never starts training."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
from embodichain.lab.gym.envs.demo import execute_demo_episode
from embodichain.lab.gym.envs.types import ControllerAction
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.lab.gym.utils.registration import (
    discover_task_packages,
    execute_init_hooks,
    get_env_spec,
)
from embodichain.utils.utility import load_config

from rlinf.envs.sim.embodichain.stack_cube import StackCubeExpertEnv
from rlinf.utils.logging import get_logger
from toolkits.lerobot.prepare_stack_cube_data import BASE_POSITIONS, prepare_panda

LOG = get_logger()
PROMPT = (
    "Stack the red block on the blue block, then stack the green block on the red block"
)
TASK = (
    Path(__file__).resolve().parents[2]
    / "rlinf/envs/sim/embodichain/configs/tasks/manipulation/stack_cube/task.panda.yaml"
)


class EpisodeRecorder(gym.Wrapper):
    """Keep pre-action images/state paired with the command actually executed."""

    def __init__(self, env: StackCubeExpertEnv, observation: dict) -> None:
        super().__init__(env)
        self.observation = observation
        self.rows = []
        self.images = {"front": [], "wrist": []}
        self.measurements = []

    def step(self, action):
        return self._record_step(action, relative=False)

    def step_relative(self, action):
        """Record a saved relative command executed by the same controller."""
        return self._record_step(action, relative=True)

    def _record_step(self, action, *, relative):
        state = self.env.dataset_state()
        for camera in self.images:
            image = (
                self.observation["sensor"][camera]["color"][0, ..., :3].cpu().numpy()
            )
            if image.dtype != np.uint8:
                image = np.clip(
                    image * (255 if image.max() <= 1 else 1), 0, 255
                ).astype(np.uint8)
            self.images[camera].append(image.copy())
        result = self.env.step_relative(action) if relative else self.env.step(action)
        self.rows.append(
            {
                "observation.state": state,
                "action": self.env.last_relative_action[0].cpu().numpy().copy(),
                "frame_index": len(self.rows),
                "timestamp": len(self.rows) / 20,
                "task_index": 0,
            }
        )
        self.observation = result[0]
        self.measurements.append(self.env.stack_metrics())
        if len(self.rows) % 80 == 0:
            LOG.info(
                "Control step %s: metrics=%s",
                len(self.rows),
                self.measurements[-1],
            )
        return result


def write_video(path: Path, frames: list[np.ndarray]) -> None:
    """Encode original-rate sensor frames using broadly supported H.264."""
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "20",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    try:
        for frame in frames:
            process.stdin.write(frame.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"Video encoding failed: {path}")


def stable_completion_step(measurements: list[dict], dwell: int = 20) -> int | None:
    """Return the first control step with a consecutive measured stable dwell."""
    consecutive = 0
    for step, metrics in enumerate(measurements, 1):
        consecutive = consecutive + 1 if metrics["stable_frame"] else 0
        if consecutive >= dwell:
            return step
    return None


def save_episode(root: Path, recorder: EpisodeRecorder, report: dict) -> None:
    """Persist one inspectable LeRobot episode and independent physical evidence."""
    root.mkdir(parents=True, exist_ok=False)
    frame = pd.DataFrame(recorder.rows)
    frame["episode_index"] = 0
    frame["index"] = np.arange(len(frame), dtype=np.int64)
    data = root / "data/chunk-000"
    data.mkdir(parents=True)
    frame.to_parquet(data / "episode_000000.parquet", index=False)
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": [8],
            "names": [
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw",
                "finger_left",
                "finger_right_signed",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": [7],
            "names": ["dx", "dy", "dz", "rx", "ry", "rz", "gripper"],
        },
    }
    for camera, images in recorder.images.items():
        key = f"observation.images.{camera}"
        write_video(root / f"videos/chunk-000/{key}/episode_000000.mp4", images)
        features[key] = {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "info": {
                "video.height": 256,
                "video.width": 256,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.fps": 20,
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    preview = [
        np.concatenate((front, wrist), axis=1)
        for front, wrist in zip(
            recorder.images["front"], recorder.images["wrist"], strict=True
        )
    ]
    write_video(root / "preview.mp4", preview)
    metadata = root / "meta"
    metadata.mkdir()
    info = {
        "codebase_version": "v2.1",
        "robot_type": "panda",
        "fps": 20,
        "features": features,
        "total_episodes": 1,
        "total_frames": len(frame),
        "total_chunks": 1,
        "total_videos": 2,
        "chunks_size": 1000,
        "total_tasks": 1,
        "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }
    (metadata / "info.json").write_text(json.dumps(info, indent=2))
    (metadata / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": PROMPT}) + "\n"
    )
    (metadata / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": len(frame), "tasks": [PROMPT]}) + "\n"
    )
    stats = {}
    for key in ("observation.state", "action"):
        values = np.stack(frame[key]).astype(np.float64)
        stats[key] = {
            "min": values.min(0).tolist(),
            "max": values.max(0).tolist(),
            "mean": values.mean(0).tolist(),
            "std": values.std(0).tolist(),
            "count": [len(values)],
        }
    for camera, images in recorder.images.items():
        means, squares, minima, maxima = [], [], [], []
        for image in images:
            pixels = image.reshape(-1, 3).astype(np.float64) / 255
            means.append(pixels.mean(0))
            squares.append((pixels**2).mean(0))
            minima.append(pixels.min(0))
            maxima.append(pixels.max(0))
        mean = np.mean(means, axis=0)
        values = {
            "min": np.min(minima, axis=0),
            "max": np.max(maxima, axis=0),
            "mean": mean,
            "std": np.sqrt(np.maximum(np.mean(squares, axis=0) - mean**2, 0)),
        }
        stats[f"observation.images.{camera}"] = {
            key: value.reshape(3, 1, 1).tolist() for key, value in values.items()
        } | {"count": [len(images)]}
    (metadata / "stats.json").write_text(json.dumps(stats, indent=2))
    (metadata / "episodes_stats.jsonl").write_text(
        json.dumps({"episode_index": 0, "stats": stats}) + "\n"
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    (root / "measurements.json").write_text(json.dumps(recorder.measurements))


def main() -> None:
    """Generate review artifacts only; physical acceptance is explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panda-urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--task-config",
        type=Path,
        default=TASK,
        help="Task Program deployment; the default preserves the smoke expert.",
    )
    parser.add_argument(
        "--requests", type=Path, help="Frozen request manifest; all attempts are saved."
    )
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--position-jitter", type=float, default=0.005)
    parser.add_argument(
        "--replay", type=Path, help="Replay one exported episode's actual 7-D actions."
    )
    args = parser.parse_args()
    task = args.task_config.resolve(strict=True)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    requests = None
    if args.requests is not None:
        requests = json.loads(args.requests.read_text())["requests"]
        if not requests:
            parser.error("request manifest must not be empty")
        args.episodes = len(requests)
        args.seed = requests[0]["seed"]
    replay_report = None
    if args.replay is not None:
        if args.requests is not None:
            parser.error("--replay and --requests are mutually exclusive")
        replay_report = json.loads((args.replay / "report.json").read_text())
        args.episodes = 1
        args.seed = replay_report["seed"]
    if args.episodes < 1 or not 0 <= args.position_jitter <= 0.02:
        parser.error(
            "episodes must be positive and position-jitter must be in [0, 0.02]"
        )
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("output must be empty; preserve previous collection evidence")
    args.output.mkdir(parents=True, exist_ok=True)
    asset = prepare_panda(args.panda_urdf, args.output / "assets")
    config_root = task.parents[3]
    sources = list(task.parent.rglob("*.yaml")) + [
        config_root / "components/embodiments/isaaclab_panda.yaml",
        config_root / "components/execution_policies/stack_cube.yaml",
    ]
    hashes = {}
    for source in sources:
        relative = source.relative_to(config_root)
        target = args.output / "config_snapshot" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    from rlinf.envs.sim.embodichain import stack_cube

    for source in [
        Path(__file__),
        Path(stack_cube.__file__),
        Path(__file__).with_name("prepare_stack_cube_data.py"),
    ]:
        target = args.output / "source_snapshot" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    native_libraries = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, "__file__", None)
        if filename and "dexsim" in f"{name}:{filename}" and filename.endswith(".so"):
            path = Path(filename).resolve()
            native_libraries[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    (args.output / "run_manifest.json").write_text(
        json.dumps(
            {
                "source_hashes": hashes,
                "native_library_sha256": native_libraries,
                "torch_version": torch.__version__,
                "training_started": False,
                "panda_fingers": "independently_actuated",
                "task_config": str(task),
                "task_config_sha256": hashlib.sha256(task.read_bytes()).hexdigest(),
                "position_jitter_m": args.position_jitter
                if args.requests is None
                else None,
                "request_manifest": str(args.requests) if args.requests else None,
                "request_manifest_sha256": hashlib.sha256(
                    args.requests.read_bytes()
                ).hexdigest()
                if args.requests
                else None,
            },
            indent=2,
        )
    )
    if args.requests is not None:
        shutil.copy2(args.requests, args.output / "request_manifest.json")
    torch.set_float32_matmul_precision("highest")
    discover_task_packages()
    execute_init_hooks()
    raw = load_config(task)
    cfg = config_to_cfg(
        raw, manager_modules=get_manager_modules(), source_path=str(task)
    )
    cfg.robot.fpath = str(asset.resolve())
    cfg.sim_cfg.headless = True
    cfg.sim_cfg.device = "cuda:0"
    cfg.sim_cfg.gpu_id = 0
    cfg.filter_dataset_saving = True
    cfg.seed = args.seed
    factory = get_env_spec(raw["id"]).task_program_adapter_factory
    env = StackCubeExpertEnv(cfg, task_program_adapter_factory=factory)
    reports = []
    recorder = None
    request = None
    seed = args.seed
    try:
        for episode in range(args.episodes):
            if stop_requested:
                break
            seed = requests[episode]["seed"] if requests else args.seed + episode
            recorder = None
            request = requests[episode] if requests else None
            if replay_report is not None:
                request = replay_report.get("request")
            if request is None:
                # Preserve the original CLI and reviewed preview initial states.
                rng = np.random.default_rng(seed)
                positions = BASE_POSITIONS.copy()
                positions[:, :2] += rng.uniform(
                    -args.position_jitter, args.position_jitter, (3, 2)
                )
                if 42 <= seed <= 49:
                    positions[:, 0] = np.clip(positions[:, 0], 0.4, 0.6)
                    positions[:, 1] = np.clip(positions[:, 1], -0.1, 0.1)
                if replay_report is not None:
                    positions = np.asarray(replay_report["initial"]["positions"])
                request = {
                    "request_id": f"legacy-{seed}",
                    "seed": seed,
                    "split": "preview",
                    "profile": "legacy",
                    "objects": {
                        uid: {"position": positions[index].tolist(), "yaw_rad": 0.0}
                        for index, uid in enumerate(("blue", "red", "green"))
                    },
                    "arm_joint_offsets": [0.0] * 7,
                    "visual": None,
                }
            observation, _ = env.reset_request(
                request,
                initial_state=replay_report.get("initial_state")
                if replay_report
                else None,
            )
            initial_state = env.initial_state()
            initial = env.stack_metrics()
            recorder = EpisodeRecorder(env, observation)
            # The collection transaction owns reset, including the terminal
            # settling frames. The SDK executor preserves this outer guard.
            env._demo_no_auto_reset = True
            if replay_report is None:
                result = execute_demo_episode(
                    recorder, episode_index=episode, should_stop=lambda: stop_requested
                )
                # Count settling in the episode budget. A late wobble must not
                # be hidden by an earlier transient success window.
                settle_start = len(recorder.rows)
                while len(recorder.rows) < 600 and not stop_requested:
                    recorder.step(ControllerAction(env.robot.get_qpos().clone()))
                    if len(recorder.rows) - settle_start >= 20 and all(
                        metric["stable_frame"] for metric in recorder.measurements[-20:]
                    ):
                        break
            else:
                saved = pd.read_parquet(
                    args.replay / "data/chunk-000/episode_000000.parquet"
                )
                for command in saved["action"]:
                    if stop_requested:
                        break
                    recorder.step_relative(
                        torch.tensor(command, device=env.device).unsqueeze(0)
                    )
                result = SimpleNamespace(
                    completed=True,
                    all_success=None,
                    terminal_reason="replay_finished",
                    segments=[],
                )
            if not recorder.rows:
                (args.output / "interrupted_request.json").write_text(
                    json.dumps(
                        {"request": request, "seed": seed, "frames": 0}, indent=2
                    )
                )
                break
            completion = stable_completion_step(recorder.measurements)
            final_stable = len(recorder.measurements) >= 20 and all(
                metric["stable_frame"] for metric in recorder.measurements[-20:]
            )
            qualified = result.completed and final_stable and not stop_requested
            report = {
                "seed": seed,
                "request": request,
                "initial_state": initial_state,
                "program_completed": result.completed,
                "interrupted": stop_requested,
                "program_success": result.all_success,
                "terminal_reason": result.terminal_reason,
                "stable_completion_step": completion,
                "qualified_completion_step": len(recorder.rows) if qualified else None,
                "passed_450": qualified and len(recorder.rows) <= 450,
                "passed_600": qualified and len(recorder.rows) <= 600,
                "initial": initial,
                "final": recorder.measurements[-1],
                "frames": len(recorder.rows),
                "position_jitter": args.position_jitter
                if request["profile"] in ("smoke", "legacy")
                else None,
                "training_started": False,
                "replay_source": str(args.replay) if args.replay is not None else None,
                "controller": "relative_eef_dls_scale0.5_lambda0.01_100hz",
                "panda_source": str(args.panda_urdf.resolve()),
                "panda_sha256": hashlib.sha256(asset.read_bytes()).hexdigest(),
                "segments": [
                    {
                        "name": segment.name,
                        "start_step": segment.start_step,
                        "end_step": segment.end_step,
                        "success": segment.success,
                        "failure_reason": segment.failure_reason,
                    }
                    for segment in result.segments
                ],
            }
            save_episode(args.output / f"episode_{episode:03d}", recorder, report)
            reports.append(report)
            (args.output / "summary.json").write_text(json.dumps(reports, indent=2))
            LOG.info(
                "Episode %s: stable completion at %s, passed450=%s",
                episode,
                completion,
                report["passed_450"],
            )
    except Exception as exc:
        (args.output / "collection_failure.json").write_text(
            json.dumps(
                {
                    "seed": seed,
                    "request": request,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "recorded_frames": len(recorder.rows) if recorder else 0,
                },
                indent=2,
            )
        )
        LOG.exception("Stack Cube collection failed")
        raise
    finally:
        env.sim.destroy(exit_process=False)


if __name__ == "__main__":
    main()
