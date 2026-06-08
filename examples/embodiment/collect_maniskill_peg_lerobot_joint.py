#!/usr/bin/env python3
"""Collect replayable PegInsertionSide-v1 demos for ``rlt_maniskill_joint``.

This collector keeps the joint-space pipeline end to end:

1. Use ManiSkill's official Panda motion-planning solver under ``pd_joint_pos``
   to obtain successful reference trajectories.
2. Convert the recorded solver actions into ``pd_joint_delta_pos`` actions.
3. Replay those converted actions in a fresh ``pd_joint_delta_pos`` env.
4. Save only the replayed trajectory when it succeeds.

LeRobot schema:
    image, wrist_image, state, actions, task

State/action semantics:
    state[0:9]   = Panda qpos (7 arm joints + 2 finger joints)
    actions[0:8] = pd_joint_delta_pos action
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    class tqdm:  # type: ignore[no-redef]
        def __init__(self, total: int, desc: str = ""):
            self.total = total
            self.desc = desc
            self.count = 0

        def update(self, n: int = 1) -> None:
            self.count += n
            print(f"{self.desc}: {self.count}/{self.total}", flush=True)

        def close(self) -> None:
            pass


def _bootstrap_repo_paths() -> Path:
    script_path = Path(__file__).resolve()
    rlinf_root = script_path.parents[2]
    repo_root = rlinf_root.parent

    for candidate in (rlinf_root, repo_root / "openpi-RLT" / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    os.environ.setdefault("EMBODIED_PATH", str(script_path.parent))
    return rlinf_root


RLINF_ROOT = _bootstrap_repo_paths()

from rlinf.envs.maniskill.peg_insertion_side_variants import (  # noqa: E402
    PANDA_WIDE_WRISTCAM_UID as SHARED_PANDA_WIDE_WRISTCAM_UID,
    PEG_INSERTION_SIDE_WIDE_ENV_ID,
    PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID,
    register_rlinf_peg_insertion_side_variants,
)

LOG = logging.getLogger("collect_maniskill_peg_lerobot_joint")

ENV_ID = PEG_INSERTION_SIDE_WIDE_ENV_ID
CAMERA_ENV_ID = PEG_INSERTION_SIDE_WIDE_OBSERVER_WIDE_WRIST_ENV_ID
CAMERA_ROBOT_UID = SHARED_PANDA_WIDE_WRISTCAM_UID
STATE_DIM = 9
ACTION_DIM = 8
DEFAULT_TASK = "insert the peg in the hole"
DEFAULT_ARM_DELTA_LOWER = -0.1
DEFAULT_ARM_DELTA_UPPER = 0.1

MAIN_CAMERA_CANDIDATES = ("3rd_view_camera", "base_camera")
WRIST_CAMERA_CANDIDATES = ("wide_hand_camera", "hand_camera")

SOLVER_MODULE_CANDIDATES = (
    "mani_skill.examples.motionplanning.panda.solutions.peg_insertion_side",
    "mani_skill.examples.motionplanning.panda.peg_insertion_side",
)

@dataclasses.dataclass(frozen=True)
class FrameRecord:
    obs: dict[str, Any]
    state: np.ndarray
    qpos: np.ndarray


def _to_numpy(value: Any, *, squeeze_env_dim: bool = True) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if squeeze_env_dim and arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _bool_scalar(value: Any) -> bool:
    arr = _to_numpy(value)
    if arr.size == 0:
        return False
    return bool(np.asarray(arr).reshape(-1)[0])


def _solver_success(result: Any) -> bool:
    if isinstance(result, dict):
        if "success" in result:
            return _bool_scalar(result["success"])
        return False
    if isinstance(result, tuple):
        for item in reversed(result):
            if isinstance(item, dict) and "success" in item:
                return _bool_scalar(item["success"])
            if isinstance(item, (bool, np.bool_)) or hasattr(item, "detach"):
                return _bool_scalar(item)
    return _bool_scalar(result)


def _import_solver():
    errors: list[str] = []
    for module_name in SOLVER_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(f"{module_name}: {exc}")
            continue
        solve = getattr(module, "solve", None)
        if solve is not None:
            return solve, module_name
        errors.append(f"{module_name}: missing solve()")

    raise ImportError(
        "Could not import ManiSkill PegInsertionSide motion-planning solver. "
        "Tried:\n  " + "\n  ".join(errors)
    )


def _missing_dep_error(package: str, install_hint: str) -> RuntimeError:
    return RuntimeError(
        f"Missing runtime dependency '{package}'. Install it in the RLinf/ManiSkill "
        f"environment before collecting data. Suggested package: {install_hint}"
    )


def _register_camera_collection_env() -> None:
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise _missing_dep_error("mani_skill", "mani_skill") from exc
    register_rlinf_peg_insertion_side_variants()


def _extract_record(obs: dict[str, Any]) -> FrameRecord:
    qpos = _to_numpy(obs["agent"]["qpos"]).astype(np.float32)
    if qpos.shape[0] < 9:
        raise ValueError(f"Expected Panda qpos with at least 9 values, got {qpos.shape}")
    state = qpos[:STATE_DIM].astype(np.float32)
    return FrameRecord(obs=obs, state=state, qpos=qpos.astype(np.float32))


def _broadcast_vector(value: Any, *, dim: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full((dim,), float(arr[0]), dtype=np.float32)
    if arr.size != dim:
        raise ValueError(f"Expected {dim} values, got {arr.shape}")
    return arr.astype(np.float32)


def _joint_delta_arm_bounds(env: Any) -> tuple[np.ndarray, np.ndarray]:
    base_env = getattr(env, "unwrapped", env)
    agent = getattr(base_env, "agent", None)
    controller = getattr(agent, "controller", None)

    candidates: list[Any] = []
    if controller is not None:
        candidates.append(controller)
        subcontrollers = getattr(controller, "controllers", None)
        if isinstance(subcontrollers, dict):
            if "arm" in subcontrollers:
                candidates.append(subcontrollers["arm"])
            candidates.extend(subcontrollers.values())
        elif isinstance(subcontrollers, (list, tuple)):
            candidates.extend(subcontrollers)

    for candidate in candidates:
        config = getattr(candidate, "config", None)
        if config is None:
            continue

        joint_names = getattr(config, "joint_names", None)
        if joint_names is not None and len(joint_names) != 7:
            continue

        lower = getattr(config, "lower", None)
        upper = getattr(config, "upper", None)
        if lower is None or upper is None:
            continue

        try:
            lower_vec = _broadcast_vector(lower, dim=7)
            upper_vec = _broadcast_vector(upper, dim=7)
        except ValueError:
            continue

        return lower_vec, upper_vec

    lower_vec = np.full((7,), DEFAULT_ARM_DELTA_LOWER, dtype=np.float32)
    upper_vec = np.full((7,), DEFAULT_ARM_DELTA_UPPER, dtype=np.float32)
    LOG.warning(
        "Could not read pd_joint_delta_pos arm bounds from env; "
        "falling back to Panda defaults [%s, %s].",
        DEFAULT_ARM_DELTA_LOWER,
        DEFAULT_ARM_DELTA_UPPER,
    )
    return lower_vec, upper_vec


def _normalize_to_minus1_plus1(
    value: np.ndarray,
    *,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    lower = np.asarray(lower, dtype=np.float32)
    upper = np.asarray(upper, dtype=np.float32)
    clipped = np.clip(value, lower, upper)
    scale = upper - lower
    safe_scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)
    normalized = 2.0 * (clipped - lower) / safe_scale - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def _camera_image(obs: dict[str, Any], camera_name: str) -> np.ndarray | None:
    sensors = obs.get("sensor_data", {})
    sensor = sensors.get(camera_name)
    if not isinstance(sensor, dict) or "rgb" not in sensor:
        return None
    image = _to_numpy(sensor["rgb"]).astype(np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Camera {camera_name} produced invalid RGB shape {image.shape}")
    return image


def _available_rgb_cameras(obs: dict[str, Any]) -> list[str]:
    sensors = obs.get("sensor_data", {})
    names: list[str] = []
    for name, sensor in sensors.items():
        if isinstance(sensor, dict) and sensor.get("rgb") is not None:
            names.append(name)
    return names


def _select_camera(
    obs: dict[str, Any],
    requested: str,
    candidates: tuple[str, ...],
    role: str,
) -> str:
    if requested:
        if _camera_image(obs, requested) is None:
            raise ValueError(
                f"Requested {role} camera '{requested}' is unavailable. "
                f"Available RGB cameras: {_available_rgb_cameras(obs)}"
            )
        return requested

    for camera_name in candidates:
        if _camera_image(obs, camera_name) is not None:
            return camera_name

    raise ValueError(
        f"No {role} camera found. Tried {candidates}; "
        f"available RGB cameras: {_available_rgb_cameras(obs)}"
    )


def _video_output_dir(repo_id: str, requested_video_dir: str) -> Path:
    if requested_video_dir:
        return Path(requested_video_dir).expanduser()
    dataset_path = _resolve_output_path(repo_id)
    return dataset_path.with_name(f"{dataset_path.name}_videos")


def _pad_to_height(image: np.ndarray, height: int) -> np.ndarray:
    if image.shape[0] == height:
        return image
    pad = np.zeros((height - image.shape[0], image.shape[1], 3), dtype=np.uint8)
    return np.concatenate([image, pad], axis=0)


def _video_view(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=-1)
    if image.ndim != 3:
        raise ValueError(f"Expected video image rank 3, got {image.shape}")
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Expected video image with 3 channels, got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _make_video_frame(frame: dict[str, Any]) -> np.ndarray:
    main = _video_view(frame["image"])
    wrist = _video_view(frame["wrist_image"])
    height = max(main.shape[0], wrist.shape[0])
    main = _pad_to_height(main, height)
    wrist = _pad_to_height(wrist, height)
    gap = np.full((height, 4, 3), 32, dtype=np.uint8)
    return np.concatenate([main, gap, wrist], axis=1)


def _write_episode_video(
    frames: list[dict[str, Any]],
    *,
    video_dir: Path,
    episode_index: int,
    seed: int,
    fps: int,
) -> None:
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / f"episode_{episode_index:06d}_seed_{seed:06d}.mp4"
    video_frames = np.stack([_make_video_frame(frame) for frame in frames], axis=0)

    try:
        import imageio.v3 as iio

        iio.imwrite(video_path, video_frames, fps=fps)
        return
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001
        LOG.warning("imageio.v3 failed to write %s: %s", video_path, exc)

    try:
        import imageio

        imageio.mimsave(video_path, list(video_frames), fps=fps)
    except ImportError as exc:
        raise _missing_dep_error("imageio", "imageio imageio-ffmpeg") from exc


def _resolve_output_path(repo_id: str) -> Path:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
    except ImportError as exc:
        raise _missing_dep_error("lerobot", "lerobot") from exc

    repo_path = Path(repo_id).expanduser()
    if repo_path.is_absolute():
        return repo_path
    return HF_LEROBOT_HOME / repo_id


def _create_dataset(
    *,
    repo_id: str,
    image_shape: tuple[int, int, int],
    wrist_image_shape: tuple[int, int, int],
    fps: int,
    image_writer_threads: int,
    image_writer_processes: int,
):
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise _missing_dep_error("lerobot", "lerobot") from exc

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=image_writer_threads,
        image_writer_processes=image_writer_processes,
    )


def _build_env(args: argparse.Namespace, *, control_mode: str):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise _missing_dep_error("gymnasium", "gymnasium") from exc
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise _missing_dep_error("mani_skill", "mani_skill") from exc

    _register_camera_collection_env()

    sim_freq = max(args.sim_freq, args.control_freq * 8)
    sim_freq -= sim_freq % args.control_freq

    env_kwargs: dict[str, Any] = {
        "id": CAMERA_ENV_ID,
        "obs_mode": "rgb",
        "control_mode": control_mode,
        "reward_mode": args.reward_mode,
        "render_mode": "rgb_array",
        "sim_backend": args.sim_backend,
        "sim_config": {"sim_freq": sim_freq, "control_freq": args.control_freq},
        "sensor_configs": {
            "shader_pack": args.shader_pack,
            "width": args.image_width,
            "height": args.image_height,
        },
        "max_episode_steps": args.max_episode_steps,
    }
    env_kwargs["robot_uids"] = args.robot_uids or CAMERA_ROBOT_UID

    return gym.make(**env_kwargs)


def _run_solver_reference(
    *,
    env: Any,
    solve: Any,
    seed: int,
) -> tuple[list[FrameRecord], list[np.ndarray]] | None:
    records: list[FrameRecord] = []
    actions: list[np.ndarray] = []

    orig_reset = env.reset
    orig_step = env.step

    def reset_hook(*hook_args, **hook_kwargs):
        out = orig_reset(*hook_args, **hook_kwargs)
        obs = out[0] if isinstance(out, tuple) else out
        records.clear()
        actions.clear()
        records.append(_extract_record(obs))
        return out

    def step_hook(action, *hook_args, **hook_kwargs):
        actions.append(_to_numpy(action).astype(np.float32).reshape(-1))
        out = orig_step(action, *hook_args, **hook_kwargs)
        obs = out[0]
        records.append(_extract_record(obs))
        return out

    env.reset = reset_hook  # type: ignore[method-assign]
    env.step = step_hook  # type: ignore[method-assign]
    try:
        result = solve(env, seed=seed, debug=False, vis=False)
        if not _solver_success(result) or len(records) < 2 or len(actions) < 1:
            return None
        return list(records), list(actions)
    finally:
        env.reset = orig_reset  # type: ignore[method-assign]
        env.step = orig_step  # type: ignore[method-assign]


def _convert_solver_action_to_joint_delta(
    curr_qpos: np.ndarray,
    solver_action: np.ndarray,
    arm_delta_lower: np.ndarray,
    arm_delta_upper: np.ndarray,
) -> np.ndarray:
    curr_qpos = np.asarray(curr_qpos, dtype=np.float32).reshape(-1)
    solver_action = np.asarray(solver_action, dtype=np.float32).reshape(-1)
    if curr_qpos.shape[0] < 9:
        raise ValueError(f"Expected qpos dim >= 9, got {curr_qpos.shape}")
    if solver_action.shape[0] < 8:
        raise ValueError(f"Expected solver action dim >= 8, got {solver_action.shape}")

    arm_delta = solver_action[:7] - curr_qpos[:7]
    gripper = solver_action[7:8]
    arm_action = _normalize_to_minus1_plus1(
        arm_delta,
        lower=arm_delta_lower,
        upper=arm_delta_upper,
    )
    return np.concatenate([arm_action, gripper], axis=0).astype(np.float32)


def _build_frames(
    *,
    records: list[FrameRecord],
    actions: list[np.ndarray],
    task: str,
    main_camera: str,
    wrist_camera: str,
) -> list[dict[str, Any]]:
    if len(records) < 2:
        raise ValueError("Need at least two observations to build one action episode")
    if len(actions) != len(records) - 1:
        raise ValueError(
            f"Expected {len(records) - 1} actions for {len(records)} records, got {len(actions)}"
        )

    frames: list[dict[str, Any]] = []
    for idx, curr in enumerate(records[:-1]):
        image = _camera_image(curr.obs, main_camera)
        if image is None:
            raise ValueError(f"Main camera '{main_camera}' missing from observation")

        wrist_image = _camera_image(curr.obs, wrist_camera)
        if wrist_image is None:
            raise ValueError(f"Wrist camera '{wrist_camera}' missing from observation")

        frames.append(
            {
                "image": image,
                "wrist_image": wrist_image,
                "state": curr.state.astype(np.float32),
                "actions": np.asarray(actions[idx], dtype=np.float32),
                "task": task,
            }
        )
    return frames


def _replay_joint_delta_episode(
    *,
    env: Any,
    seed: int,
    reference_records: list[FrameRecord],
    solver_actions: list[np.ndarray],
    arm_delta_lower: np.ndarray,
    arm_delta_upper: np.ndarray,
) -> tuple[list[FrameRecord], list[np.ndarray]] | None:
    obs, _ = env.reset(seed=seed)
    replay_records = [_extract_record(obs)]
    replay_actions: list[np.ndarray] = []
    last_info: dict[str, Any] = {}

    for step_idx, solver_action in enumerate(solver_actions):
        curr_qpos = replay_records[-1].qpos
        action = _convert_solver_action_to_joint_delta(
            curr_qpos,
            solver_action,
            arm_delta_lower,
            arm_delta_upper,
        )
        obs, _reward, terminated, truncated, info = env.step(action)
        last_info = info
        replay_actions.append(action.astype(np.float32))
        replay_records.append(_extract_record(obs))

        if _bool_scalar(info.get("success", False)):
            break
        if _bool_scalar(terminated) or _bool_scalar(truncated):
            break

        if step_idx + 1 >= len(reference_records) - 1:
            break

    if len(replay_records) < 2 or len(replay_actions) != len(replay_records) - 1:
        return None
    if not _bool_scalar(last_info.get("success", False)):
        LOG.info("Rejecting seed %d: pd_joint_delta_pos replay did not succeed", seed)
        return None
    return replay_records, replay_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect successful widened PegInsertionSide-v1 demos as a replayable "
            "rlt_maniskill_joint LeRobot dataset."
        )
    )
    parser.add_argument("--repo-id", default="local/rlt_maniskill_joint")
    parser.add_argument("--num-episodes", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=2000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--main-camera", default="")
    parser.add_argument("--wrist-camera", default="")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--shader-pack", default="default")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--robot-uids", default="")
    parser.add_argument(
        "--solver-control-mode",
        default="pd_joint_pos",
        help="Keep this as pd_joint_pos for ManiSkill's Panda motion-planning solver.",
    )
    parser.add_argument(
        "--target-control-mode",
        default="pd_joint_delta_pos",
        help="Controller used for the saved dataset actions.",
    )
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument("--image-writer-processes", type=int, default=4)
    parser.add_argument(
        "--save-videos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save one side-by-side mp4 per successful episode for visual inspection.",
    )
    parser.add_argument(
        "--video-dir",
        default="",
        help="Video output directory. Defaults to <dataset_path>_videos.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    output_path = _resolve_output_path(args.repo_id)
    video_dir = _video_output_dir(args.repo_id, args.video_dir)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Dataset already exists at {output_path}. Pass --overwrite to replace it."
            )
        LOG.info("Removing existing dataset at %s", output_path)
        shutil.rmtree(output_path)
    if args.save_videos and video_dir.exists() and args.overwrite:
        LOG.info("Removing existing video directory at %s", video_dir)
        shutil.rmtree(video_dir)

    solve, solver_module = _import_solver()
    LOG.info("Using ManiSkill solver: %s", solver_module)

    solver_env = _build_env(args, control_mode=args.solver_control_mode)
    replay_env = _build_env(args, control_mode=args.target_control_mode)
    arm_delta_lower, arm_delta_upper = _joint_delta_arm_bounds(replay_env)
    LOG.info(
        "Replay arm delta bounds: lower=%s upper=%s",
        np.round(arm_delta_lower, 6).tolist(),
        np.round(arm_delta_upper, 6).tolist(),
    )

    dataset = None
    main_camera = ""
    wrist_camera = ""
    saved = 0
    attempts = 0
    solver_failures = 0
    replay_failures = 0
    pbar = tqdm(total=args.num_episodes, desc="Successful episodes")

    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            episode_seed = args.seed + attempts
            attempts += 1

            try:
                reference = _run_solver_reference(
                    env=solver_env,
                    solve=solve,
                    seed=episode_seed,
                )
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Solver failed on seed %d: %s", episode_seed, exc)
                solver_failures += 1
                continue

            if reference is None:
                solver_failures += 1
                continue

            reference_records, solver_actions = reference

            if not main_camera:
                main_camera = _select_camera(
                    reference_records[0].obs,
                    args.main_camera,
                    MAIN_CAMERA_CANDIDATES,
                    "main",
                )
                wrist_camera = _select_camera(
                    reference_records[0].obs,
                    args.wrist_camera,
                    WRIST_CAMERA_CANDIDATES,
                    "wrist",
                )
                LOG.info(
                    "Selected cameras: image=%s, wrist_image=%s",
                    main_camera,
                    wrist_camera,
                )

            replay = _replay_joint_delta_episode(
                env=replay_env,
                seed=episode_seed,
                reference_records=reference_records,
                solver_actions=solver_actions,
                arm_delta_lower=arm_delta_lower,
                arm_delta_upper=arm_delta_upper,
            )
            if replay is None:
                replay_failures += 1
                continue

            replay_records, replay_actions = replay
            frames = _build_frames(
                records=replay_records,
                actions=replay_actions,
                task=args.task,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
            )

            if dataset is None:
                dataset = _create_dataset(
                    repo_id=args.repo_id,
                    image_shape=tuple(frames[0]["image"].shape),
                    wrist_image_shape=tuple(frames[0]["wrist_image"].shape),
                    fps=args.fps,
                    image_writer_threads=args.image_writer_threads,
                    image_writer_processes=args.image_writer_processes,
                )

            for frame in frames:
                dataset.add_frame(frame)
            dataset.save_episode()
            if args.save_videos:
                _write_episode_video(
                    frames,
                    video_dir=video_dir,
                    episode_index=saved,
                    seed=episode_seed,
                    fps=args.fps,
                )

            saved += 1
            pbar.update(1)

    finally:
        pbar.close()
        if dataset is not None and getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        solver_env.close()
        replay_env.close()

    if saved < args.num_episodes:
        raise RuntimeError(
            f"Only saved {saved}/{args.num_episodes} successful episodes after "
            f"{attempts} attempts. solver_failures={solver_failures}, "
            f"replay_failures={replay_failures}."
        )

    LOG.info("Saved %d successful episodes after %d attempts to %s", saved, attempts, output_path)


if __name__ == "__main__":
    main()
