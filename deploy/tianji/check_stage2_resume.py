"""Validate the resolved RTJ Stage2 model and tactile replay before startup.

--config [Hydra overrides] checks a fresh run or runner.resume_dir.
CHECKPOINT_DIR [Hydra overrides] checks an explicit full training checkpoint.
No robot, sensor, Ray worker or training process is started here.
"""

import inspect
import json
import os
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

RLINF_ROOT = Path(os.environ.get("RLINF_ROOT", Path(__file__).resolve().parents[2])).resolve()
TACTILE_KEYS = ("tactile_images", "tactile_valid", "tactile_age_ms")


def resolve_config(overrides: list[str]):
    """Use the same primary config and environment as the training/DAgger entry."""
    config_name = os.environ.get("TIANJI_STAGE2_CONFIG", "tianji2_rlt_stage2_async")
    with initialize_config_dir(
        config_dir=str(RLINF_ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        return compose(config_name=config_name, overrides=overrides)


def check_input_modes(cfg) -> bool:
    """Reject inconsistent model/environment switches before services start."""
    switches = [
        bool(OmegaConf.select(cfg, path, default=False))
        for path in (
            "actor.model.tactile_enabled",
            "env.train.override_cfg.tactile_enabled",
            "env.eval.override_cfg.tactile_enabled",
        )
    ]
    if len(set(switches)) != 1:
        raise ValueError("actor and train/eval env tactile_enabled must match")
    return switches[0]


def check_tactile_replay(directory: Path, model_cfg=None) -> int:
    """Check every indexed trajectory, retaining pixels on disk via mmap."""
    if str(RLINF_ROOT) not in sys.path:
        sys.path.insert(0, str(RLINF_ROOT))
    from rlinf.utils.xense_tactile import TACTILE_IMAGE_SHAPE

    directory = directory.expanduser().resolve()
    metadata = json.loads((directory / "metadata.json").read_text())
    index = json.loads((directory / "trajectory_index.json").read_text())
    if metadata.get("trajectory_format", "pt") != "pt":
        raise ValueError("tactile replay preflight requires pt trajectory format")
    ids = index.get("trajectory_id_list", [])
    entries = index.get("trajectory_index", {})
    if not isinstance(ids, list) or not isinstance(entries, dict):
        raise ValueError("invalid replay index format")
    if int(metadata.get("size", 0)) != len(ids):
        raise ValueError("replay metadata size does not match trajectory_id_list")
    for trajectory_id in ids:
        entry = entries[str(trajectory_id)]
        filename = f"trajectory_{int(trajectory_id)}_{entry['model_weights_id']}.pt"
        path = directory / filename
        if path.parent.resolve() != directory:
            raise ValueError("invalid replay trajectory path")
        trajectory = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        if model_cfg is not None and model_cfg.get("executor_action_grid", False):
            from rlinf.algorithms.rlt.action_grid import ActionGrid

            grid = ActionGrid.from_config(model_cfg)
            fi = trajectory.get("forward_inputs", {})
            grid_meta = fi.get("target_action_grid")
            if grid_meta is None:
                raise ValueError(
                    f"{path.name}: old replay lacks target_action_grid; convert explicitly"
                )
            grid.validate_metadata(grid_meta.reshape(-1, 7))
            target = fi.get("target_policy_action")
            if target is None or target.shape[-1] != grid.target_steps * int(
                model_cfg.action_dim
            ):
                raise ValueError(f"{path.name}: missing executor-grid policy targets")
        for field in ("curr_obs", "next_obs"):
            obs = trajectory.get(field)
            if not isinstance(obs, dict) or any(key not in obs for key in TACTILE_KEYS):
                raise ValueError(
                    f"{path.name}/{field} missing tactile fields; cannot train tactile model on non-tactile replay"
                )
            images, valid, age = (obs[key] for key in TACTILE_KEYS)
            if not all(torch.is_tensor(value) for value in (images, valid, age)):
                raise ValueError(f"{path.name}/{field} tactile fields must be Tensors")
            if (
                images.ndim < 5
                or tuple(images.shape[-4:]) != TACTILE_IMAGE_SHAPE
                or images.dtype != torch.uint8
            ):
                raise ValueError(
                    f"{path.name}/{field} invalid tactile shape/dtype (expected {TACTILE_IMAGE_SHAPE})"
                )
            if (
                valid.shape != images.shape[:-3]
                or age.shape != valid.shape
                or valid.dtype != torch.bool
            ):
                raise ValueError(f"{path.name}/{field} invalid tactile mask/age shape/dtype")
            if (
                age.dtype != torch.float32
                or not bool(torch.isfinite(age).all())
                or bool(((age < 0) | (age > 100)).any())
            ):
                raise ValueError(f"{path.name}/{field} invalid tactile age")
    return len(ids)


def _plain(value):
    return (
        OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
    )


def check_checkpoint(checkpoint: Path, cfg) -> None:
    """Strict-load the actor, compare residual semantics, and inspect replay."""
    if str(RLINF_ROOT) not in sys.path:
        sys.path.insert(0, str(RLINF_ROOT))
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

    checkpoint = checkpoint.expanduser().resolve()
    model_cfg = OmegaConf.to_container(cfg.actor.model, resolve=True)
    kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key in inspect.signature(RLTMLPPolicy).parameters
    }
    torch.set_num_threads(1)
    model = RLTMLPPolicy(**kwargs)
    state_path = checkpoint / "actor/model_state_dict/full_weights.pt"
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(
            "Checkpoint network is incompatible with the current model configuration"
        ) from exc
    run_cfg = checkpoint.parents[2] / "tensorboard/all/config.yaml"
    if run_cfg.is_file():
        old = OmegaConf.load(run_cfg).actor.model
        for key in (
            "residual_limits",
            "action_scales",
            "residual_in_cap_units",
            "critic_action_space",
            "tactile_enabled",
            "tactile_embed_dim",
            "executor_action_grid",
            "action_grid_offset",
            "action_grid_scale",
            "action_grid_dense",
        ):
            if key.startswith("tactile_") and not bool(
                model_cfg.get("tactile_enabled", False)
            ):
                continue
            before = OmegaConf.select(old, key, default=None)
            after = OmegaConf.select(cfg.actor.model, key, default=None)
            if _plain(before) != _plain(after):
                raise ValueError(
                    f"{key} mismatch: checkpoint {_plain(before)}, current {_plain(after)}"
                )
    else:
        print(
            f"STAGE2_RESUME_NOTE missing historical config {run_cfg}; skipped semantic check",
            file=sys.stderr,
        )
    if check_input_modes(cfg):
        replay_root = checkpoint / "actor/sac_components/replay_buffer"
        ranks = sorted(replay_root.glob("rank_*"))
        if not ranks:
            raise ValueError("tactile resume checkpoint missing replay rank directories")
        count = sum(check_tactile_replay(rank, cfg.actor.model) for rank in ranks)
        print(f"STAGE2_TACTILE_REPLAY_OK trajectories={count}", flush=True)


def main() -> int:
    try:
        if len(sys.argv) < 2:
            raise ValueError(
                "Usage: check_stage2_resume.py --config|CHECKPOINT_DIR [Hydra overrides]"
            )
        explicit = None if sys.argv[1] == "--config" else Path(sys.argv[1])
        cfg = resolve_config(sys.argv[2:])
        enabled = check_input_modes(cfg)
        checkpoint = explicit or OmegaConf.select(
            cfg, "runner.resume_dir", default=None
        )
        preload = OmegaConf.select(
            cfg, "algorithm.replay_buffer.load_path", default=None
        )
        if checkpoint:
            if preload:
                raise ValueError(
                    "Resume loads checkpoint replay only; replay_buffer.load_path must be null"
                )
            check_checkpoint(Path(checkpoint), cfg)
        elif enabled and preload:
            count = check_tactile_replay(Path(preload), cfg.actor.model)
            print(f"STAGE2_TACTILE_REPLAY_OK trajectories={count}", flush=True)
        print(
            f"STAGE2_INPUT_CHECK_OK tactile={str(enabled).lower()} "
            f"resume={bool(checkpoint)} preload={preload or 'none'}",
            flush=True,
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print(f"STAGE2_RESUME_INCOMPATIBLE {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
