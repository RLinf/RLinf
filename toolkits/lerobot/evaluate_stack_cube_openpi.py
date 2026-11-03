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

"""Evaluate PI 0.5 on frozen Stack Cube resets without expert assistance.

Inference runs in a spawned process so OpenPI's matmul precision does not
change the differential IK solver's precision. Complete and partial reports
are written separately; infrastructure failures are never counted as successes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from multiprocessing.connection import Connection
from pathlib import Path

import numpy as np


def preprocessing_protocol(*, eval_sft_image_crop: bool = True) -> dict:
    """Describe the shared Stack Cube SFT/inference tensor contract."""
    return {
        "config_name": "pi05_isaaclab_stack_cube",
        "action_horizon": 10,
        "action_chunk": 5,
        "denoising_steps": 5,
        "eval_sft_image_crop": eval_sft_image_crop,
        "front_crop": {"origin": [0, 0], "size": [212, 212], "resize": [224, 224]}
        if eval_sft_image_crop
        else None,
        "wrist_crop": None,
        "sensor_image_layout": "HWC RGB uint8",
        "model_image_resolution": [224, 224],
        "state_encoding": "eef_xyz_euler_xyz_signed_fingers",
        "discrete_state_input": False,
        "phase_input": False,
        "image_mask": {
            "base_0_rgb": True,
            "left_wrist_0_rgb": True,
            "right_wrist_0_rgb": False,
        },
    }


def summarize_episode(measurements: list[dict], *, dwell: int = 20) -> dict:
    """Count consecutive physical steps, preserving once and final outcomes."""
    streak = 0
    first_stable = None
    for step, metric in enumerate(measurements, 1):
        streak = streak + 1 if metric["stable_frame"] else 0
        if streak >= dwell and first_stable is None:
            first_stable = step
    return {
        "control_steps": len(measurements),
        "first_layer_once": any(m["layers"][0] for m in measurements),
        "full_stack_once": any(m["baseline_success"] for m in measurements),
        "full_stack_at_end": bool(
            measurements and measurements[-1]["baseline_success"]
        ),
        "stable_once": first_stable is not None,
        "stable_at_end": streak >= dwell,
        "first_stable_step": first_stable,
        "final_stable_steps": streak,
    }


def aggregate_reports(reports: list[dict], expected: int) -> dict:
    """Report counts against the frozen denominator, including pending episodes."""
    keys = (
        "first_layer_once",
        "full_stack_once",
        "full_stack_at_end",
        "stable_once",
        "stable_at_end",
    )
    return {
        "expected_episodes": expected,
        "completed_episodes": len(reports),
        "counts": {key: sum(report[key] for report in reports) for key in keys},
        "rates": {
            key: sum(report[key] for report in reports) / expected for key in keys
        },
        "complete": len(reports) == expected,
    }


def load_replay_episodes(collection: Path, requests: list[dict]) -> dict[int, dict]:
    """Match saved expert episodes to frozen requests, rejecting ambiguities."""
    import pandas as pd

    episodes = {}
    manifest_path = collection / "request_manifest.json"
    if manifest_path.is_file():
        source_requests = json.loads(manifest_path.read_text())["requests"]
        requested_seeds = {int(request["seed"]) for request in requests}
        paths = [
            collection / f"episode_{index:03d}" / "report.json"
            for index, request in enumerate(source_requests)
            if int(request["seed"]) in requested_seeds
        ]
    else:
        paths = list(collection.glob("episode_*/report.json"))
    for path in paths:
        report = json.loads(path.read_text())
        request = report["request"]
        seed = int(request["seed"])
        if seed in episodes:
            raise ValueError(f"Duplicate replay seed: {seed}")
        data = pd.read_parquet(path.parent / "data/chunk-000/episode_000000.parquet")
        actions = np.stack(data["action"].to_numpy()).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 7 or not np.isfinite(actions).all():
            raise ValueError(f"Invalid expert actions in {path.parent}")
        episodes[seed] = {
            "request": request,
            "report": report,
            "actions": actions,
            "source": str(path.parent),
        }
    selected = {}
    for request in requests:
        episode = episodes[int(request["seed"])]
        if episode["request"] != request:
            raise ValueError(
                f"Replay request differs from frozen manifest: {request['seed']}"
            )
        if "initial_state" not in episode["report"]:
            raise ValueError("Expert replay requires exact recorded initial state")
        selected[int(request["seed"])] = episode
    return selected


def apply_visual_request(env, visual: dict) -> dict:
    """Apply evaluation-only light and table color changes before recording."""
    import torch

    color = np.asarray(visual["table_rgb"], dtype=np.float64)
    intensity = float(visual["light_intensity"])
    if (
        color.shape != (3,)
        or not np.isfinite(color).all()
        or np.any((color < 0) | (color > 1))
    ):
        raise ValueError("Table RGB must contain three finite values in [0, 1]")
    if not np.isfinite(intensity) or intensity <= 0:
        raise ValueError("Light intensity must be finite and positive")
    light = env.sim.get_light("main_light")
    table = env.sim.get_rigid_object("table")
    if light is None or table is None:
        raise RuntimeError("Stack Cube visual entities are missing")
    light.set_intensity(torch.tensor([intensity], device=env.device))
    for material in table.get_visual_material_inst():
        material.set_base_color([*color.tolist(), 1.0])
    with env.sim.render_frame(capture_camera_images=True):
        return env.get_obs()


def _policy_process(
    connection,
    checkpoint: str,
    stats: str,
    gpu: str,
    crop: bool,
):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    try:
        import torch
        from omegaconf import OmegaConf

        from rlinf.models import get_model

        torch.set_float32_matmul_precision("high")
        cfg = OmegaConf.create(
            {
                "model_type": "openpi",
                "model_path": checkpoint,
                "precision": None,
                "is_lora": False,
                "action_dim": 7,
                "num_action_chunks": 5,
                "num_steps": 5,
                "openpi_data": {"norm_stats_path": stats},
                "openpi": {
                    "task": "eval",
                    "config_name": "pi05_isaaclab_stack_cube",
                    "action_horizon": 10,
                    "action_chunk": 5,
                    "action_env_dim": 7,
                    "model_action_dim": 32,
                    "paligemma_variant": "gemma_2b",
                    "action_expert_variant": "gemma_300m",
                    "max_token_len": 200,
                    "num_images_in_input": 2,
                    "num_steps": 5,
                    "discrete_state_input": False,
                    "eval_sft_image_crop": crop,
                    "torch_compile": False,
                },
            }
        )
        model = get_model(cfg).to("cuda:0").eval()
        connection.send(
            {
                "ready": True,
                "model_config": OmegaConf.to_container(cfg),
                "runtime": {
                    "python": sys.executable,
                    "prefix": sys.prefix,
                    "numpy": np.__version__,
                    "torch": torch.__version__,
                },
            }
        )
        generator = None
        while True:
            message = connection.recv()
            if message["command"] == "close":
                break
            if message["command"] == "reset":
                seed = int(message["seed"])
                torch.manual_seed(seed)
                np.random.seed(seed)
                generator = torch.Generator(device="cuda:0").manual_seed(seed)
                connection.send({"reset": True})
                continue
            observation = {
                key: torch.from_numpy(value) if isinstance(value, np.ndarray) else value
                for key, value in message["observation"].items()
            }
            actions, _ = model.predict_action_batch(
                observation, mode="eval", rng=generator
            )
            actions = actions.detach().float().cpu().numpy()
            if actions.shape != (1, 5, 7) or not np.isfinite(actions).all():
                raise ValueError(f"Invalid policy action batch: {actions.shape}")
            connection.send({"actions": actions})
    except Exception:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def receive_policy(
    connection: Connection, process: subprocess.Popen, timeout: float = 1800
) -> dict:
    """Read an inference result, detecting child exit, errors, and timeout."""
    deadline = time.monotonic() + timeout
    while not connection.poll(1):
        if process.poll() is not None:
            raise RuntimeError(
                f"Inference process exited with code {process.returncode}"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError("Inference process did not respond before deadline")
    try:
        message = connection.recv()
    except EOFError as exc:
        raise RuntimeError("Inference process closed its connection") from exc
    if "error" in message:
        raise RuntimeError(message["error"])
    return message


def start_policy_process(
    checkpoint: Path,
    stats: Path,
    gpu: str,
    crop: bool,
    model_python: Path | None = None,
) -> tuple[Connection, subprocess.Popen]:
    """Start inference in the supplied interpreter or the current interpreter.

    The subprocess starts independently of the simulator's sys.path. The caller
    owns its connection and process and releases both with ``stop_policy_process``.
    """
    connection, child = mp.Pipe()
    options = json.dumps(
        {
            "checkpoint": str(checkpoint),
            "stats": str(stats),
            "gpu": gpu,
            "crop": crop,
        }
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[2]))
    environment.setdefault("JAX_PLATFORMS", "cpu")
    try:
        process = subprocess.Popen(
            [
                str(model_python or sys.executable),
                str(Path(__file__).resolve()),
                "--policy-worker",
                str(child.fileno()),
                options,
            ],
            pass_fds=(child.fileno(),),
            env=environment,
        )
    except BaseException:
        connection.close()
        raise
    finally:
        child.close()
    return connection, process


def stop_policy_process(connection: Connection, process: subprocess.Popen) -> None:
    """Close inference, terminating a child that fails to acknowledge shutdown."""
    try:
        if process.poll() is None:
            try:
                connection.send({"command": "close"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    finally:
        connection.close()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Run all frozen evaluation requests and persist physical evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument(
        "--panda-urdf",
        type=Path,
        required=True,
        help="Prepared expert Panda URDF, including panda_tcp.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-gpu", required=True)
    parser.add_argument(
        "--model-python",
        type=Path,
        help="Independent OpenPI interpreter; the caller runs the simulator SDK.",
    )
    parser.add_argument("--sim-gpu", required=True)
    parser.add_argument(
        "--replay-collection",
        type=Path,
        help="Run inference but execute saved expert actions; diagnostic only.",
    )
    parser.add_argument("--max-steps", type=int, default=450)
    parser.add_argument(
        "--eval-sft-image-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match the existing SFT front-view 95%% top-left crop (enabled for this experiment).",
    )
    args = parser.parse_args()
    if args.model_python is not None:
        args.model_python = args.model_python.expanduser().absolute()
        if os.name != "posix" or not os.access(args.model_python, os.X_OK):
            parser.error("--model-python requires an executable interpreter on POSIX")
    requests = json.loads(args.requests.read_text())["requests"]
    if not requests or args.max_steps < 1:
        parser.error("requests must not be empty and max-steps must be positive")
    replay_episodes = (
        load_replay_episodes(args.replay_collection, requests)
        if args.replay_collection
        else {}
    )
    if any(
        len(episode["actions"]) > args.max_steps for episode in replay_episodes.values()
    ):
        parser.error(
            "replay episode exceeds max-steps; never silently truncate recorded actions"
        )
    if replay_episodes:
        asset_hash = _hash(args.panda_urdf)
        for episode in replay_episodes.values():
            if episode["report"].get("panda_sha256") != asset_hash:
                parser.error(
                    "Replay must use the exact prepared Panda asset recorded by collection"
                )
    args.checkpoint = args.checkpoint.resolve(strict=True)
    if not args.checkpoint.is_dir():
        parser.error("checkpoint must be a directory containing exported weights")
    export_marker = args.checkpoint / "actor/checkpoint_metadata.json"
    if (args.checkpoint / "actor/model_state_dict/full_weights.pt").is_file():
        if not export_marker.is_file() or not json.loads(export_marker.read_text()).get(
            "complete"
        ):
            parser.error(
                "SFT checkpoint requires complete actor/checkpoint_metadata.json"
            )
    weight_paths = [
        args.checkpoint / name
        for name in (
            "actor/model_state_dict/full_weights.pt",
            "model_state_dict/full_weights.pt",
            "full_weights.pt",
            "model.safetensors",
        )
    ]
    weights = next((path for path in weight_paths if path.is_file()), None)
    if weights is None:
        parser.error("checkpoint does not contain supported weights")
    weights_stat = weights.stat()
    weights_sha256 = _hash(weights)
    args.norm_stats = args.norm_stats.resolve(strict=True)
    if args.norm_stats.is_dir():
        args.norm_stats /= "norm_stats.json"
    if args.output.exists():
        parser.error("output already exists; use a fresh directory to preserve results")
    args.output.mkdir(parents=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.sim_gpu
    connection, process = start_policy_process(
        args.checkpoint,
        args.norm_stats,
        args.model_gpu,
        args.eval_sft_image_crop,
        args.model_python,
    )
    env = None
    reports = []
    try:
        ready = receive_policy(connection, process)
        import torch
        from omegaconf import OmegaConf

        from rlinf.envs.sim.embodichain.stack_cube_env import StackCubeEnv
        from toolkits.lerobot.collect_embodichain_stack_cube import write_video

        torch.set_float32_matmul_precision("highest")
        config = OmegaConf.create(
            {
                "gym_config_path": "rlinf/envs/sim/embodichain/configs/tasks/manipulation/stack_cube/task.panda.yaml",
                "request_manifest": str(args.requests.resolve()),
                "panda_urdf": str(args.panda_urdf.resolve()),
                "seed": requests[0]["seed"],
                "sim_device": "cuda:0",
                "max_episode_steps": args.max_steps,
                "auto_reset": False,
                "is_eval": True,
            }
        )
        runtime = StackCubeEnv(
            config,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
        env = runtime.env

        config_root = (
            Path(__file__).resolve().parents[2] / "rlinf/envs/sim/embodichain/configs"
        )
        config_files = list(
            (config_root / "tasks/manipulation/stack_cube").rglob("*.yaml")
        ) + [
            config_root / "components/embodiments/isaaclab_panda.yaml",
            config_root / "components/execution_policies/stack_cube.yaml",
        ]
        for source in config_files:
            destination = (
                args.output / "config_snapshot" / source.relative_to(config_root)
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        protocol = {
            "model_config_name": "pi05_isaaclab_stack_cube",
            "action_horizon": 10,
            "action_chunk": 5,
            "denoising_steps": 5,
            "state": "8D EEF XYZ + Euler XYZ + signed finger positions",
            "discrete_state_input": False,
            "episode_phase_input": False,
            "action": "7D relative EEF rotation-vector increments + binary gripper",
            "views": ["front", "wrist"],
            "sensor_resolution": [256, 256],
            "model_resolution": [224, 224],
            "eval_sft_image_crop": args.eval_sft_image_crop,
            "front_crop": "top-left 212x212 then resize to224"
            if args.eval_sft_image_crop
            else "none",
            "wrist_crop": "none",
            "control_hz": 20,
            "max_steps": args.max_steps,
            "stable_dwell_steps": 20,
            "linear_speed_threshold": 0.02,
            "angular_speed_threshold": 0.5,
            "norm_stats_sha256": _hash(args.norm_stats),
            "config_sha256": {
                str(path.relative_to(config_root)): _hash(path) for path in config_files
            },
        }
        protocol_hash = hashlib.sha256(
            json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        preprocessing = preprocessing_protocol(
            eval_sft_image_crop=args.eval_sft_image_crop
        )
        preprocessing_hash = hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        source_files = [
            Path(__file__).resolve(),
            Path(__file__).resolve().parents[2]
            / "rlinf/envs/sim/embodichain/stack_cube.py",
            Path(__file__).resolve().parents[2]
            / "rlinf/envs/sim/embodichain/embodichain_env.py",
        ]
        for source in source_files:
            destination = (
                args.output
                / "source_snapshot"
                / source.relative_to(Path(__file__).resolve().parents[2])
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        metadata = {
            "preprocessing_protocol": preprocessing,
            "preprocessing_protocol_sha256": preprocessing_hash,
            "protocol": protocol,
            "protocol_sha256": protocol_hash,
            "config_sha256": {
                str(path.relative_to(config_root)): _hash(path) for path in config_files
            },
            "checkpoint": str(args.checkpoint),
            "weights": str(weights),
            "weights_sha256": weights_sha256,
            "weights_bytes": weights_stat.st_size,
            "checkpoint_export_metadata": json.loads(export_marker.read_text())
            if export_marker.is_file()
            else None,
            "source_sha256": {str(path): _hash(path) for path in source_files},
            "norm_stats_sha256": _hash(args.norm_stats),
            "requests_sha256": _hash(args.requests),
            "panda_urdf_sha256": _hash(args.panda_urdf),
            "model_config": ready["model_config"],
            "model_runtime": ready.get("runtime"),
            "model_python_requested": str(args.model_python)
            if args.model_python
            else None,
            "simulation_runtime": {
                "python": sys.executable,
                "prefix": sys.prefix,
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "max_steps": args.max_steps,
            "policy_assistance": bool(replay_episodes),
            "action_source": "expert_replay" if replay_episodes else "learned_policy",
            "learned_policy_evaluation": not bool(replay_episodes),
            "env_api": "rlinf",
            "sim_matmul_precision": "highest",
            "model_matmul_precision": "high",
        }
        (args.output / "run_manifest.json").write_text(json.dumps(metadata, indent=2))
        (args.output / "request_manifest.json").write_text(args.requests.read_text())
        for index, request in enumerate(requests):
            replay = replay_episodes.get(int(request["seed"]))
            initial_state = replay["report"]["initial_state"] if replay else None
            episode_steps = len(replay["actions"]) if replay else args.max_steps
            physical_request = request | {"visual": None}
            raw_obs, _ = runtime.reset(
                options={
                    "request": physical_request,
                    "initial_state": initial_state,
                }
            )
            if request.get("visual") is not None:
                rendered = apply_visual_request(env, request["visual"])
                raw_obs = env.policy_observation(rendered)
            initial = env.initial_state()
            connection.send({"command": "reset", "seed": request["seed"]})
            receive_policy(connection, process)
            measurements, states, actions, predictions = [], [], [], []
            images = {"front": [], "wrist": []}
            while len(measurements) < episode_steps:
                observation = raw_obs
                wire = {
                    key: value.cpu().numpy()
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in observation.items()
                }
                message = {"command": "predict", "observation": wire}
                connection.send(message)
                batch = receive_policy(connection, process)["actions"]
                predictions.append(batch[0])
                executed_chunk = (
                    replay["actions"][len(measurements) : len(measurements) + 5]
                    if replay
                    else batch[0]
                )
                for command in executed_chunk:
                    if len(measurements) >= episode_steps:
                        break
                    observation = raw_obs
                    states.append(observation["states"][0].cpu().numpy())
                    for camera, key in (
                        ("front", "main_images"),
                        ("wrist", "wrist_images"),
                    ):
                        images[camera].append(observation[key][0].cpu().numpy())
                    action = torch.as_tensor(command[None], device=env.device)
                    raw_obs, _, terminated, truncated, _ = runtime.step(action)
                    if bool(terminated.any()) or (
                        bool(truncated.any())
                        != (len(measurements) + 1 == args.max_steps)
                    ):
                        raise RuntimeError(
                            "RLinf wrapper episode boundary disagrees with evaluation budget."
                        )
                    actions.append(command)
                    measurements.append(env.stack_metrics())
            # Include the terminal image; prior images are pre-action frames.
            terminal = raw_obs
            for camera, key in (("front", "main_images"), ("wrist", "wrist_images")):
                images[camera].append(terminal[key][0].cpu().numpy())
            report = summarize_episode(measurements) | {
                "request": request,
                "initial_state": initial,
                "episode_index": index,
                "action_source": "expert_replay" if replay else "learned_policy",
                "learned_policy_evaluation": replay is None,
                "inference_calls": len(predictions),
                "protocol_sha256": protocol_hash,
                "replay_source": replay["source"] if replay else None,
                "saved_expert_initial_state": initial_state,
            }
            if replay:
                report["replay_comparison"] = {
                    "source_frames": int(replay["report"]["frames"]),
                    "frame_count_matches": len(actions)
                    == int(replay["report"]["frames"]),
                    "initial_qpos_max_abs_error": float(
                        np.max(
                            np.abs(
                                np.asarray(initial["qpos"])
                                - np.asarray(initial_state["qpos"])
                            )
                        )
                    ),
                    "final_position_max_abs_error_m": float(
                        np.max(
                            np.abs(
                                np.asarray(measurements[-1]["positions"])
                                - np.asarray(replay["report"]["final"]["positions"])
                            )
                        )
                    ),
                    "source_stable_at_end": bool(
                        replay["report"]["final"]["stable_frame"]
                    ),
                }
            episode_dir = args.output / f"episode_{index:03d}"
            episode_dir.mkdir()
            for camera, frames in images.items():
                write_video(episode_dir / f"{camera}.mp4", frames)
            np.savez_compressed(
                episode_dir / "trajectory.npz",
                states=np.array(states),
                actions=np.array(actions),
                predictions=np.array(predictions),
            )
            (episode_dir / "measurements.json").write_text(json.dumps(measurements))
            (episode_dir / "report.json").write_text(json.dumps(report, indent=2))
            reports.append(report)
            summary = aggregate_reports(reports, len(requests)) | {
                "episodes": reports,
                "action_source": metadata["action_source"],
                "learned_policy_evaluation": metadata["learned_policy_evaluation"],
                "protocol_sha256": protocol_hash,
            }
            (args.output / "partial_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            print(
                f"Episode {index + 1}/{len(requests)} stable_at_end={report['stable_at_end']}",
                flush=True,
            )
        final_weights_stat = weights.stat()
        if (final_weights_stat.st_size, final_weights_stat.st_mtime_ns) != (
            weights_stat.st_size,
            weights_stat.st_mtime_ns,
        ):
            raise RuntimeError(
                "Checkpoint changed during evaluation; results are not final."
            )
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    except BaseException:
        (args.output / "failure.json").write_text(
            json.dumps(
                {"completed_episodes": len(reports), "error": traceback.format_exc()},
                indent=2,
            )
        )
        raise
    finally:
        try:
            stop_policy_process(connection, process)
        finally:
            if env is not None:
                env.sim.destroy(exit_process=False)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--policy-worker":
        options = json.loads(sys.argv[3])
        _policy_process(Connection(int(sys.argv[2])), **options)
    else:
        main()
