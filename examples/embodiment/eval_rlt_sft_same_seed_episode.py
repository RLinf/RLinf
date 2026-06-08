#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import Any


def _bootstrap_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    embodied_path = script_path.parent
    rlinf_root = embodied_path.parents[1]
    repo_root = rlinf_root.parent

    for candidate in (rlinf_root, repo_root / "openpi-RLT" / "src", embodied_path):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    os.environ.setdefault("EMBODIED_PATH", str(embodied_path))
    return repo_root, rlinf_root, embodied_path


REPO_ROOT, RLINF_ROOT, EMBODIED_PATH = _bootstrap_paths()

import numpy as np
import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.models.embodiment.openpi import get_model

from collect_maniskill_peg_lerobot_joint import (  # noqa: E402
    CAMERA_ENV_ID,
    CAMERA_ROBOT_UID,
    DEFAULT_TASK,
    MAIN_CAMERA_CANDIDATES,
    WRIST_CAMERA_CANDIDATES,
    _available_rgb_cameras,
    _camera_image,
    _make_video_frame,
    _register_camera_collection_env,
    _select_camera,
)


ENV_ID = CAMERA_ENV_ID
ACTION_DIM = 8
STATE_DIM = 9
_INDEX_OVERRIDE_RE = re.compile(
    r"^(?P<prefix>[\w.]+)\[(?P<index>\d+)\]\.(?P<suffix>[^=]+)=(?P<value>.*)$"
)


def _missing_dep_error(package: str) -> RuntimeError:
    return RuntimeError(
        f"Missing dependency '{package}'. Run this script in the same environment used for SFT/eval."
    )


def _to_numpy(value: Any, *, squeeze_env_dim: bool = True) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if squeeze_env_dim and arr.ndim > 0 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _to_scalar_int(value: Any) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _truthy_first(value: Any) -> bool:
    if value is None:
        return False
    arr = _to_numpy(value).reshape(-1)
    return bool(arr[0]) if arr.size else False


def _episode_seed_from_videos(dataset_path: Path, episode_index: int) -> int | None:
    video_dir = dataset_path.with_name(f"{dataset_path.name}_videos")
    if not video_dir.exists():
        return None
    pattern = f"episode_{episode_index:06d}_seed_*.mp4"
    matches = sorted(video_dir.glob(pattern))
    if not matches:
        return None
    match = re.search(r"_seed_(\d+)\.mp4$", matches[0].name)
    return int(match.group(1)) if match else None


def _episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int]:
    data_index = getattr(dataset, "episode_data_index", None)
    if data_index is not None and episode_index < len(data_index["from"]):
        return (
            _to_scalar_int(data_index["from"][episode_index]),
            _to_scalar_int(data_index["to"][episode_index]),
        )

    matches = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        if _to_scalar_int(sample["episode_index"]) == episode_index:
            matches.append(idx)
    if not matches:
        raise RuntimeError(f"Episode {episode_index} not found in dataset.")
    return min(matches), max(matches) + 1


def _load_episode(dataset_path: Path, episode_index: int) -> list[dict[str, Any]]:
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise _missing_dep_error("lerobot") from exc

    dataset = LeRobotDataset(str(dataset_path), download_videos=False)
    start, end = _episode_bounds(dataset, episode_index)
    rows = [dataset[idx] for idx in range(start, end)]
    if not rows:
        raise RuntimeError(f"Episode {episode_index} is empty.")
    return rows


def _extract_state(raw_obs: dict[str, Any]) -> np.ndarray:
    qpos = _to_numpy(raw_obs["agent"]["qpos"]).astype(np.float32)
    if qpos.shape[0] < STATE_DIM:
        raise RuntimeError(f"Expected Panda qpos with at least {STATE_DIM} values, got {qpos.shape}")
    return qpos[:STATE_DIM].astype(np.float32)


def _state_error(replay_state: np.ndarray, dataset_state: Any) -> tuple[float, float]:
    diff = replay_state - _to_numpy(dataset_state).astype(np.float32).reshape(-1)[:STATE_DIM]
    return float(np.max(np.abs(diff))), float(np.mean(np.abs(diff)))


def _raw_obs_to_policy_obs(
    raw_obs: dict[str, Any],
    *,
    main_camera: str,
    wrist_camera: str,
    task: str,
) -> dict[str, Any]:
    image = _camera_image(raw_obs, main_camera)
    wrist_image = _camera_image(raw_obs, wrist_camera)
    if image is None:
        raise RuntimeError(
            f"Main camera '{main_camera}' missing. Available: {_available_rgb_cameras(raw_obs)}"
        )
    if wrist_image is None:
        raise RuntimeError(
            f"Wrist camera '{wrist_camera}' missing. Available: {_available_rgb_cameras(raw_obs)}"
        )
    return {
        "main_images": torch.as_tensor(image).unsqueeze(0),
        "wrist_images": torch.as_tensor(wrist_image).unsqueeze(0),
        "extra_view_images": None,
        "states": torch.as_tensor(_extract_state(raw_obs), dtype=torch.float32).unsqueeze(0),
        "task_descriptions": [task],
    }


def _make_frame_from_raw_obs(
    raw_obs: dict[str, Any],
    *,
    main_camera: str,
    wrist_camera: str,
) -> dict[str, Any]:
    image = _camera_image(raw_obs, main_camera)
    wrist_image = _camera_image(raw_obs, wrist_camera)
    if image is None or wrist_image is None:
        raise RuntimeError(
            f"Could not extract video cameras {main_camera}/{wrist_camera}; "
            f"available={_available_rgb_cameras(raw_obs)}"
        )
    return {"image": image, "wrist_image": wrist_image}


def _dataset_image_to_hwc_uint8(value: Any) -> np.ndarray:
    image = _to_numpy(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image, 0.0, 1.0) * 255.0
    image = image.astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
        image = np.transpose(image, (1, 2, 0))
    return image


def _make_frame_from_dataset_row(row: dict[str, Any]) -> dict[str, Any]:
    wrist = row.get("wrist_image", row.get("extra_view_image"))
    image = _dataset_image_to_hwc_uint8(row["image"])
    wrist_image = _dataset_image_to_hwc_uint8(wrist)
    return {
        "image": image,
        "wrist_image": wrist_image,
    }


def _dataset_row_to_policy_obs(row: dict[str, Any], *, task: str) -> dict[str, Any]:
    frame = _make_frame_from_dataset_row(row)
    return {
        "main_images": torch.as_tensor(frame["image"]).unsqueeze(0),
        "wrist_images": torch.as_tensor(frame["wrist_image"]).unsqueeze(0),
        "extra_view_images": None,
        "states": torch.as_tensor(_to_numpy(row["state"]), dtype=torch.float32).unsqueeze(0),
        "task_descriptions": [task],
    }


def _image_diff(label: str, lhs: np.ndarray, rhs: np.ndarray) -> str:
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    if lhs.shape != rhs.shape:
        return f"{label}: shape_mismatch live={lhs.shape} dataset={rhs.shape}"
    diff = np.abs(lhs - rhs)
    return (
        f"{label}: max_abs={float(diff.max()):.6g} "
        f"mean_abs={float(diff.mean()):.6g}"
    )


def _action_diff(label: str, lhs: np.ndarray, rhs: np.ndarray) -> str:
    lhs = np.asarray(lhs, dtype=np.float32)
    rhs = np.asarray(rhs, dtype=np.float32)
    if lhs.shape != rhs.shape:
        return f"{label}: shape_mismatch lhs={lhs.shape} rhs={rhs.shape}"
    diff = np.abs(lhs - rhs)
    return (
        f"{label}: max_abs={float(diff.max()):.6g} "
        f"mean_abs={float(diff.mean()):.6g}"
    )


def _print_reset_alignment(
    *,
    model,
    raw_obs: dict[str, Any],
    dataset_row: dict[str, Any],
    main_camera: str,
    wrist_camera: str,
    task: str,
    policy_seed: int,
) -> None:
    live_main = _camera_image(raw_obs, main_camera)
    live_wrist = _camera_image(raw_obs, wrist_camera)
    dataset_frame = _make_frame_from_dataset_row(dataset_row)
    if live_main is not None:
        print(_image_diff("reset_image_vs_dataset_image", live_main, dataset_frame["image"]))
    if live_wrist is not None:
        print(
            _image_diff(
                "reset_wrist_vs_dataset_wrist",
                live_wrist,
                dataset_frame["wrist_image"],
            )
        )

    live_state = _extract_state(raw_obs)
    dataset_state = _to_numpy(dataset_row["state"]).astype(np.float32).reshape(-1)[:STATE_DIM]
    print(_action_diff("reset_state_vs_dataset_state", live_state, dataset_state))

    def _predict_once(obs: dict[str, Any]) -> np.ndarray:
        random.seed(policy_seed)
        np.random.seed(policy_seed)
        torch.manual_seed(policy_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(policy_seed)
        with torch.no_grad():
            chunk, _ = model.predict_action_batch(
                obs,
                mode="eval",
                compute_values=False,
            )
        return chunk.detach().cpu().to(torch.float32).numpy()[0, :, :ACTION_DIM]

    live_policy_obs = _raw_obs_to_policy_obs(
        raw_obs,
        main_camera=main_camera,
        wrist_camera=wrist_camera,
        task=task,
    )
    dataset_policy_obs = _dataset_row_to_policy_obs(dataset_row, task=task)
    live_chunk = _predict_once(live_policy_obs)
    dataset_chunk = _predict_once(dataset_policy_obs)
    gt_chunk0 = _to_numpy(dataset_row["actions"]).astype(np.float32).reshape(-1)[:ACTION_DIM]

    print(_action_diff("reset_live_pred_chunk_vs_dataset_pred_chunk", live_chunk, dataset_chunk))
    print(
        "reset_live_pred_first="
        + np.array2string(live_chunk[0], precision=6, separator=", ")
    )
    print(
        "reset_dataset_pred_first="
        + np.array2string(dataset_chunk[0], precision=6, separator=", ")
    )
    print("reset_dataset_gt_first=" + np.array2string(gt_chunk0, precision=6, separator=", "))


def _write_video(frames: list[dict[str, Any]], path: Path, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    video_frames = np.stack([_make_video_frame(frame) for frame in frames], axis=0)
    try:
        import imageio.v3 as iio

        iio.imwrite(path, video_frames, fps=fps)
        return
    except ImportError:
        pass
    except Exception:
        pass

    try:
        import imageio

        imageio.mimsave(path, list(video_frames), fps=fps)
    except ImportError as exc:
        raise _missing_dep_error("imageio") from exc


def _build_env(args: argparse.Namespace):
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise _missing_dep_error("gymnasium") from exc
    try:
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise _missing_dep_error("mani_skill") from exc

    _register_camera_collection_env()
    sim_freq = max(args.sim_freq, args.control_freq * 8)
    sim_freq -= sim_freq % args.control_freq
    return gym.make(
        ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        reward_mode=args.reward_mode,
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        robot_uids=args.robot_uids or CAMERA_ROBOT_UID,
        sim_config={"sim_freq": sim_freq, "control_freq": args.control_freq},
        sensor_configs={
            "shader_pack": args.shader_pack,
            "width": args.image_width,
            "height": args.image_height,
        },
        max_episode_steps=args.max_episode_steps,
    )


def _split_index_overrides(overrides: list[str]) -> tuple[list[str], list[tuple[str, int, str, str]]]:
    hydra_overrides = []
    index_overrides = []
    for override in overrides:
        match = _INDEX_OVERRIDE_RE.match(override)
        if match is None:
            hydra_overrides.append(override)
            continue
        index_overrides.append(
            (
                match.group("prefix"),
                int(match.group("index")),
                match.group("suffix"),
                match.group("value"),
            )
        )
    return hydra_overrides, index_overrides


def _apply_index_overrides(cfg, index_overrides: list[tuple[str, int, str, str]]) -> None:
    for prefix, index, suffix, value in index_overrides:
        container = OmegaConf.select(cfg, prefix)
        if container is None:
            raise KeyError(f"Override target does not exist: {prefix}[{index}].{suffix}")
        if index >= len(container):
            raise IndexError(
                f"Override target index out of range: {prefix}[{index}].{suffix}"
            )
        OmegaConf.update(container[index], suffix, value, merge=True)


def _apply_direct_path_args(cfg, args: argparse.Namespace) -> None:
    if args.model_path:
        OmegaConf.update(cfg, "actor.model.model_path", args.model_path, merge=True)

    if args.dataset_path:
        train_data_paths = cfg.data.get("train_data_paths")
        if train_data_paths is None or len(train_data_paths) == 0:
            OmegaConf.update(
                cfg,
                "data.train_data_paths",
                [{"dataset_path": args.dataset_path, "weight": 1.0}],
                merge=True,
            )
        else:
            OmegaConf.update(
                cfg.data.train_data_paths[0],
                "dataset_path",
                args.dataset_path,
                merge=True,
            )
        if not args.repo_id:
            OmegaConf.update(cfg, "actor.openpi_data.repo_id", args.dataset_path, merge=True)

    if args.repo_id:
        OmegaConf.update(cfg, "actor.openpi_data.repo_id", args.repo_id, merge=True)

    if args.norm_stats_path:
        OmegaConf.update(
            cfg,
            "actor.openpi_data.norm_stats_path",
            args.norm_stats_path,
            merge=True,
        )


def _compose_cfg(args: argparse.Namespace):
    config_dir = RLINF_ROOT / "examples" / "sft" / "config"
    hydra_overrides, index_overrides = _split_index_overrides(list(args.overrides))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=hydra_overrides)
    _apply_index_overrides(cfg, index_overrides)
    _apply_direct_path_args(cfg, args)
    cfg = validate_cfg(cfg)
    if not args.keep_compile:
        OmegaConf.update(
            cfg,
            "actor.model.openpi.pytorch_compile_mode",
            None,
            force_add=True,
        )
    return cfg


def _plot_rollout(
    *,
    steps: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    state_max_abs: np.ndarray,
    state_mean_abs: np.ndarray,
    output_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise _missing_dep_error("matplotlib") from exc

    action_dim = pred.shape[1]
    ncols = 2
    nrows = int(np.ceil((action_dim + 1) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.7 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    names = [f"joint_delta_{idx}" for idx in range(min(action_dim, 7))]
    if action_dim >= 8:
        names.append("gripper")
    while len(names) < action_dim:
        names.append(f"action_{len(names)}")

    for dim in range(action_dim):
        ax = axes[dim]
        ax.plot(steps, gt[:, dim], label="dataset_gt", linewidth=1.8)
        ax.plot(steps, pred[:, dim], label="policy_exec", linewidth=1.4, alpha=0.85)
        ax.plot(steps, pred[:, dim] - gt[:, dim], label="policy_minus_gt", linewidth=1.0, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(names[dim])
        ax.set_ylim(-1.25, 1.25)
        ax.grid(True, alpha=0.25)

    drift_ax = axes[action_dim]
    drift_ax.plot(steps, state_max_abs, label="state_max_abs")
    drift_ax.plot(steps, state_mean_abs, label="state_mean_abs")
    drift_ax.set_title("closed_loop_state_drift_vs_dataset")
    drift_ax.grid(True, alpha=0.25)
    drift_ax.legend(loc="upper left", fontsize=8)

    for idx in range(action_dim + 1, len(axes)):
        axes[idx].axis("off")

    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes[-ncols:]:
        ax.set_xlabel("executed env step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _run_policy_rollout(
    *,
    args: argparse.Namespace,
    model,
    env,
    rows: list[dict[str, Any]],
    seed: int,
    main_camera: str,
    wrist_camera: str,
    task: str,
) -> dict[str, Any]:
    raw_obs, _ = env.reset(seed=seed)
    frames = [_make_frame_from_raw_obs(raw_obs, main_camera=main_camera, wrist_camera=wrist_camera)]
    pred_actions: list[np.ndarray] = []
    gt_actions: list[np.ndarray] = []
    steps: list[int] = []
    state_max_abs: list[float] = []
    state_mean_abs: list[float] = []

    first_success_step = None
    final_success = False
    stopped_reason = "max_steps"
    env_step = 0

    random.seed(args.policy_seed)
    np.random.seed(args.policy_seed)
    torch.manual_seed(args.policy_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.policy_seed)

    model.eval()
    with torch.no_grad():
        while env_step < args.max_steps:
            env_obs = _raw_obs_to_policy_obs(
                raw_obs,
                main_camera=main_camera,
                wrist_camera=wrist_camera,
                task=task,
            )
            chunk, _ = model.predict_action_batch(
                env_obs,
                mode="eval",
                compute_values=False,
            )
            chunk_np = chunk.detach().cpu().to(torch.float32).numpy()[0, :, :ACTION_DIM]
            exec_count = min(args.action_exec_chunks, chunk_np.shape[0], args.max_steps - env_step)

            for chunk_idx in range(exec_count):
                action = chunk_np[chunk_idx].astype(np.float32)
                if args.clip_actions:
                    action = np.clip(action, -1.0, 1.0).astype(np.float32)
                raw_obs, reward, terminated, truncated, info = env.step(action)
                env_step += 1

                target_idx = min(env_step - 1, len(rows) - 1)
                gt = _to_numpy(rows[target_idx]["actions"]).astype(np.float32).reshape(-1)[:ACTION_DIM]
                max_abs, mean_abs = _state_error(_extract_state(raw_obs), rows[target_idx]["state"])

                pred_actions.append(action)
                gt_actions.append(gt)
                steps.append(env_step)
                state_max_abs.append(max_abs)
                state_mean_abs.append(mean_abs)
                frames.append(
                    _make_frame_from_raw_obs(raw_obs, main_camera=main_camera, wrist_camera=wrist_camera)
                )

                success = _truthy_first(info.get("success", False))
                final_success = success
                if success and first_success_step is None:
                    first_success_step = env_step

                if (
                    env_step == 1
                    or env_step % args.log_every == 0
                    or success
                    or _truthy_first(terminated)
                    or _truthy_first(truncated)
                ):
                    print(
                        f"step={env_step} reward={_to_numpy(reward).reshape(-1).tolist()} "
                        f"terminated={_to_numpy(terminated).reshape(-1).tolist()} "
                        f"truncated={_to_numpy(truncated).reshape(-1).tolist()} "
                        f"success={_to_numpy(info.get('success', False)).reshape(-1).tolist()} "
                        f"state_vs_dataset_frame_{target_idx}: "
                        f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
                    )

                if success:
                    stopped_reason = "success"
                    break
                if _truthy_first(terminated) or _truthy_first(truncated):
                    stopped_reason = "terminated_or_truncated"
                    break
                if env_step >= args.max_steps:
                    stopped_reason = "max_steps"
                    break

            if stopped_reason != "max_steps" or env_step >= args.max_steps:
                break

    return {
        "frames": frames,
        "steps": np.asarray(steps, dtype=np.int64),
        "pred_actions": np.asarray(pred_actions, dtype=np.float32),
        "gt_actions": np.asarray(gt_actions, dtype=np.float32),
        "state_max_abs": np.asarray(state_max_abs, dtype=np.float32),
        "state_mean_abs": np.asarray(state_mean_abs, dtype=np.float32),
        "first_success_step": first_success_step,
        "final_success": final_success,
        "stopped_reason": stopped_reason,
        "steps_run": env_step,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Closed-loop eval an SFT checkpoint from the same initial seed as one "
            "collected rlt_maniskill_joint LeRobot episode, then save videos and action/state plots."
        )
    )
    parser.add_argument("--config-name", default="rlt_maniskill_joint_pi05_sft")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None, help="Env reset seed. Defaults to seed parsed from <dataset>_videos.")
    parser.add_argument("--policy-seed", type=int, default=0, help="Torch/numpy/random seed for stochastic OpenPI action sampling.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--main-camera", default="")
    parser.add_argument("--wrist-camera", default="")
    parser.add_argument("--image-width", type=int, default=384)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--control-freq", type=int, default=10)
    parser.add_argument("--sim-freq", type=int, default=100)
    parser.add_argument("--sim-backend", default="physx_cpu")
    parser.add_argument("--shader-pack", default="default")
    parser.add_argument("--reward-mode", default="sparse")
    parser.add_argument("--robot-uids", default="")
    parser.add_argument("--max-episode-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--action-exec-chunks", type=int, default=10)
    parser.add_argument(
        "--clip-actions",
        action="store_true",
        help="Clip policy actions to [-1, 1] before env.step. Disabled by default to match RLinf panda-qpos prepare_actions.",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--keep-compile", action="store_true")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    if args.repo_id is None:
        args.repo_id = str(dataset_path)
    if not os.environ.get("HF_LEROBOT_HOME"):
        os.environ["HF_LEROBOT_HOME"] = str(dataset_path.parent)

    seed = args.seed
    if seed is None:
        seed = _episode_seed_from_videos(dataset_path, args.episode_index)
    if seed is None:
        raise RuntimeError("Could not infer collection seed. Pass --seed explicitly.")

    rows = _load_episode(dataset_path, args.episode_index)
    cfg = _compose_cfg(args)
    model = get_model(cfg.actor.model, torch_dtype=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    env = _build_env(args)
    output_dir = (
        Path(args.output_dir).expanduser()
        if args.output_dir
        else dataset_path / f"same_seed_eval_episode_{args.episode_index:06d}_seed_{seed:06d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        raw_obs, _ = env.reset(seed=seed)
        main_camera = _select_camera(
            raw_obs, args.main_camera, MAIN_CAMERA_CANDIDATES, "main"
        )
        wrist_camera = _select_camera(
            raw_obs, args.wrist_camera, WRIST_CAMERA_CANDIDATES, "wrist"
        )
        initial_state = _extract_state(raw_obs)
        max_abs, mean_abs = _state_error(initial_state, rows[0]["state"])
        print(
            f"episode={args.episode_index} seed={seed} frames={len(rows)} "
            f"camera=image:{main_camera} wrist:{wrist_camera} "
            f"action_exec_chunks={args.action_exec_chunks} "
            f"policy_seed={args.policy_seed}"
        )
        print(f"step0_state_vs_dataset: max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}")
        _print_reset_alignment(
            model=model,
            raw_obs=raw_obs,
            dataset_row=rows[0],
            main_camera=main_camera,
            wrist_camera=wrist_camera,
            task=args.task,
            policy_seed=args.policy_seed,
        )

        rollout = _run_policy_rollout(
            args=args,
            model=model,
            env=env,
            rows=rows,
            seed=seed,
            main_camera=main_camera,
            wrist_camera=wrist_camera,
            task=args.task,
        )
    finally:
        env.close()

    dataset_video_frames = [_make_frame_from_dataset_row(row) for row in rows]
    policy_video = output_dir / "policy_closed_loop.mp4"
    dataset_video = output_dir / "dataset_expert.mp4"
    plot_path = output_dir / "policy_vs_dataset_actions.png"
    npz_path = output_dir / "policy_vs_dataset_actions.npz"

    _write_video(rollout["frames"], policy_video, args.fps)
    _write_video(dataset_video_frames, dataset_video, args.fps)
    _plot_rollout(
        steps=rollout["steps"],
        pred=rollout["pred_actions"],
        gt=rollout["gt_actions"],
        state_max_abs=rollout["state_max_abs"],
        state_mean_abs=rollout["state_mean_abs"],
        output_path=plot_path,
    )
    np.savez_compressed(
        npz_path,
        steps=rollout["steps"],
        policy_actions=rollout["pred_actions"],
        dataset_actions=rollout["gt_actions"],
        state_max_abs=rollout["state_max_abs"],
        state_mean_abs=rollout["state_mean_abs"],
    )

    if rollout["pred_actions"].size:
        abs_diff = np.abs(rollout["pred_actions"] - rollout["gt_actions"])
        per_dim_mae = abs_diff.mean(axis=0)
        worst_dim = int(np.argmax(per_dim_mae))
        print(f"mae_all={abs_diff.mean():.6f} max_abs_all={abs_diff.max():.6f}")
        print(
            f"worst_dim_by_mae={worst_dim} mae={per_dim_mae[worst_dim]:.6f}"
        )
        print(
            "per_dim_mae="
            + ", ".join(f"{idx}:{value:.6f}" for idx, value in enumerate(per_dim_mae))
        )

    print(f"steps_run={rollout['steps_run']}")
    print(f"stopped_reason={rollout['stopped_reason']}")
    print(f"first_success_step={rollout['first_success_step']}")
    print(f"final_success={rollout['final_success']}")
    print(f"saved_policy_video={policy_video}")
    print(f"saved_dataset_video={dataset_video}")
    print(f"saved_plot={plot_path}")
    print(f"saved_npz={npz_path}")


if __name__ == "__main__":
    main()
