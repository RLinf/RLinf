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

    for candidate in (rlinf_root, repo_root / "openpi-RLT" / "src"):
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


_INDEX_OVERRIDE_RE = re.compile(
    r"^(?P<prefix>[\w.]+)\[(?P<index>\d+)\]\.(?P<suffix>[^=]+)=(?P<value>.*)$"
)


def _missing_dep_error(package: str) -> RuntimeError:
    return RuntimeError(
        f"Missing dependency '{package}'. Run this script in the same environment used for SFT/eval."
    )


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _to_scalar_int(value: Any) -> int:
    return int(_to_numpy(value).reshape(-1)[0])


def _episode_bounds(dataset: Any, episode_index: int) -> tuple[int, int]:
    data_index = getattr(dataset, "episode_data_index", None)
    if data_index is not None:
        if episode_index < len(data_index["from"]):
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


def _image_to_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(np.asarray(value))


def _prompt_to_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    if hasattr(value, "item"):
        try:
            return str(value.item())
        except Exception:
            pass
    return str(value)


def _source_row_to_rollout_obs(row: dict[str, Any]) -> dict[str, Any]:
    if "image" not in row:
        raise KeyError("Dataset row does not contain 'image'.")
    if "state" not in row:
        raise KeyError("Dataset row does not contain 'state'.")

    image = row["image"]
    wrist_image = row.get("wrist_image")
    if wrist_image is None:
        wrist_image = row.get("extra_view_image")

    prompt = row.get("prompt", row.get("task", "insert the peg in the hole"))
    return {
        "main_images": _image_to_tensor(image).unsqueeze(0),
        "wrist_images": (
            _image_to_tensor(wrist_image).unsqueeze(0) if wrist_image is not None else None
        ),
        "extra_view_images": None,
        "states": torch.as_tensor(row["state"], dtype=torch.float32).unsqueeze(0),
        "task_descriptions": [_prompt_to_string(prompt)],
    }


def _extract_gt_action(row: dict[str, Any], action_dim: int) -> np.ndarray:
    if "actions" not in row:
        raise KeyError("Dataset row does not contain 'actions'.")
    action = _to_numpy(row["actions"]).astype(np.float32).reshape(-1)
    if action.shape[0] < action_dim:
        raise ValueError(f"Expected action dim >= {action_dim}, got {action.shape}.")
    return action[:action_dim]


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


def _predict_first_actions(
    *,
    model,
    rows: list[dict[str, Any]],
    action_dim: int,
    seed: int,
    max_steps: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    limit = len(rows) if max_steps is None else min(len(rows), max_steps)
    pred_actions = []
    gt_actions = []
    frame_indices = []

    with torch.no_grad():
        for local_idx, row in enumerate(rows[:limit]):
            env_obs = _source_row_to_rollout_obs(row)
            pred_chunk, _ = model.predict_action_batch(
                env_obs,
                mode="eval",
                compute_values=False,
            )
            pred = pred_chunk.detach().cpu().to(torch.float32).numpy()[0, 0, :action_dim]
            gt = _extract_gt_action(row, action_dim)
            frame_index = _to_scalar_int(row.get("frame_index", local_idx))

            pred_actions.append(pred)
            gt_actions.append(gt)
            frame_indices.append(frame_index)

    return (
        np.asarray(frame_indices, dtype=np.int64),
        np.asarray(pred_actions, dtype=np.float32),
        np.asarray(gt_actions, dtype=np.float32),
    )


def _predict_chunk_actions(
    *,
    model,
    rows: list[dict[str, Any]],
    action_dim: int,
    seed: int,
    max_steps: int | None,
    chunk_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.eval()
    usable = max(0, len(rows) - chunk_len + 1)
    limit = usable if max_steps is None else min(usable, max_steps)
    pred_chunks = []
    gt_chunks = []
    frame_indices = []

    with torch.no_grad():
        for local_idx, row in enumerate(rows[:limit]):
            env_obs = _source_row_to_rollout_obs(row)
            pred_chunk, _ = model.predict_action_batch(
                env_obs,
                mode="eval",
                compute_values=False,
            )
            pred = (
                pred_chunk.detach()
                .cpu()
                .to(torch.float32)
                .numpy()[0, :chunk_len, :action_dim]
            )
            gt = np.asarray(
                [
                    _extract_gt_action(rows[local_idx + offset], action_dim)
                    for offset in range(chunk_len)
                ],
                dtype=np.float32,
            )
            frame_index = _to_scalar_int(row.get("frame_index", local_idx))

            pred_chunks.append(pred)
            gt_chunks.append(gt)
            frame_indices.append(frame_index)

    return (
        np.asarray(frame_indices, dtype=np.int64),
        np.asarray(pred_chunks, dtype=np.float32),
        np.asarray(gt_chunks, dtype=np.float32),
    )


def _plot_actions(
    *,
    steps: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise _missing_dep_error("matplotlib") from exc

    action_dim = pred.shape[1]
    names = [f"joint_delta_{idx}" for idx in range(min(action_dim, 7))]
    if action_dim >= 8:
        names.append("gripper")
    while len(names) < action_dim:
        names.append(f"action_{len(names)}")

    ncols = 2
    nrows = int(np.ceil(action_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.7 * nrows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for dim in range(action_dim):
        ax = axes[dim]
        ax.plot(steps, gt[:, dim], label="dataset_gt", linewidth=1.8)
        ax.plot(steps, pred[:, dim], label="model_pred_first", linewidth=1.4, alpha=0.85)
        ax.plot(steps, pred[:, dim] - gt[:, dim], label="pred_minus_gt", linewidth=1.0, alpha=0.55)
        ax.axhline(0.0, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(names[dim])
        ax.set_ylim(-1.25, 1.25)
        ax.grid(True, alpha=0.25)

    for dim in range(action_dim, len(axes)):
        axes[dim].axis("off")

    axes[0].legend(loc="upper right", fontsize=8)
    for ax in axes[:action_dim]:
        ax.set_ylabel("action")
    for ax in axes[-ncols:]:
        ax.set_xlabel("episode step / frame_index")

    fig.suptitle(title)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _print_summary(pred: np.ndarray, gt: np.ndarray) -> None:
    abs_diff = np.abs(pred - gt)
    per_dim_mae = abs_diff.mean(axis=0)
    per_dim_max = abs_diff.max(axis=0)
    worst_dim = int(np.argmax(per_dim_mae))

    print(f"compare_shape pred={pred.shape} gt={gt.shape}")
    print(f"mae_all={abs_diff.mean():.6f} max_abs_all={abs_diff.max():.6f}")
    print(f"worst_dim_by_mae={worst_dim} mae={per_dim_mae[worst_dim]:.6f}")
    print("per_dim_mae=" + ", ".join(f"{idx}:{value:.6f}" for idx, value in enumerate(per_dim_mae)))
    print("per_dim_max=" + ", ".join(f"{idx}:{value:.6f}" for idx, value in enumerate(per_dim_max)))
    if pred.shape[0] > 0:
        print("first_pred=" + np.array2string(pred[0], precision=6, separator=", "))
        print("first_gt=" + np.array2string(gt[0], precision=6, separator=", "))


def _print_chunk_summary(pred_chunks: np.ndarray, gt_chunks: np.ndarray) -> None:
    if pred_chunks.size == 0:
        print("chunk_compare: no usable windows")
        return
    abs_diff = np.abs(pred_chunks - gt_chunks)
    print(
        f"chunk_compare_shape pred={pred_chunks.shape} gt={gt_chunks.shape} "
        f"mae_all={abs_diff.mean():.6f} max_abs_all={abs_diff.max():.6f}"
    )
    per_offset_mae = abs_diff.mean(axis=(0, 2))
    per_dim_mae = abs_diff.mean(axis=(0, 1))
    print(
        "chunk_offset_mae="
        + ", ".join(f"{idx}:{value:.6f}" for idx, value in enumerate(per_offset_mae))
    )
    print(
        "chunk_per_dim_mae="
        + ", ".join(f"{idx}:{value:.6f}" for idx, value in enumerate(per_dim_mae))
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot one LeRobot episode's dataset actions against actions predicted "
            "from the same dataset images/states by an RLT ManiSkill joint SFT checkpoint."
        )
    )
    parser.add_argument("--config-name", default="rlt_maniskill_joint_pi05_sft")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--norm-stats-path", default=None)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--action-dim", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument(
        "--compare-chunk",
        action="store_true",
        help="Also compare predicted chunk[k] against dataset action[t+k]. This matches RLinf chunk_step eval semantics.",
    )
    parser.add_argument(
        "--chunk-len",
        type=int,
        default=10,
        help="Chunk length used by --compare-chunk.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default="",
        help="Output png path. Defaults to <dataset_path>/debug_episode_<idx>_actions.png.",
    )
    parser.add_argument(
        "--save-npz",
        default="",
        help="Optional npz path. Defaults to the png path with .npz suffix.",
    )
    parser.add_argument(
        "--keep-compile",
        action="store_true",
        help="Keep torch.compile enabled if configured. Disabled by default for faster startup.",
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser()
    if args.repo_id is None:
        args.repo_id = str(dataset_path)

    if not os.environ.get("HF_LEROBOT_HOME"):
        os.environ["HF_LEROBOT_HOME"] = str(dataset_path.parent)

    cfg = _compose_cfg(args)
    rows = _load_episode(dataset_path, args.episode_index)
    model = get_model(cfg.actor.model, torch_dtype=None)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_steps = None if args.max_steps <= 0 else args.max_steps
    steps, pred, gt = _predict_first_actions(
        model=model,
        rows=rows,
        action_dim=args.action_dim,
        seed=args.seed,
        max_steps=max_steps,
    )

    output_path = (
        Path(args.output).expanduser()
        if args.output
        else dataset_path / f"debug_episode_{args.episode_index:06d}_actions.png"
    )
    npz_path = (
        Path(args.save_npz).expanduser()
        if args.save_npz
        else output_path.with_suffix(".npz")
    )

    _plot_actions(
        steps=steps,
        pred=pred,
        gt=gt,
        output_path=output_path,
        title=(
            f"episode {args.episode_index}: model first action vs dataset action "
            f"({cfg.actor.model.openpi.config_name})"
        ),
    )
    np.savez_compressed(npz_path, steps=steps, pred_actions=pred, gt_actions=gt)

    _print_summary(pred, gt)

    if args.compare_chunk:
        chunk_steps, pred_chunks, gt_chunks = _predict_chunk_actions(
            model=model,
            rows=rows,
            action_dim=args.action_dim,
            seed=args.seed,
            max_steps=max_steps,
            chunk_len=args.chunk_len,
        )
        _print_chunk_summary(pred_chunks, gt_chunks)
        chunk_npz_path = output_path.with_name(output_path.stem + "_chunk.npz")
        np.savez_compressed(
            chunk_npz_path,
            steps=chunk_steps,
            pred_chunks=pred_chunks,
            gt_chunks=gt_chunks,
        )
        print(f"saved_chunk_npz={chunk_npz_path}")

    print(f"saved_plot={output_path}")
    print(f"saved_npz={npz_path}")


if __name__ == "__main__":
    main()
