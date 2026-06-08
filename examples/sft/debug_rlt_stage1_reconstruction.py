#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def _bootstrap_paths() -> tuple[Path, Path, Path]:
    script_path = Path(__file__).resolve()
    sft_path = script_path.parent
    rlinf_root = sft_path.parents[1]
    repo_root = rlinf_root.parent

    for candidate in (rlinf_root, repo_root / "openpi-RLT" / "src"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

    os.environ.setdefault("EMBODIED_PATH", str(sft_path))
    return repo_root, rlinf_root, sft_path


REPO_ROOT, RLINF_ROOT, SFT_PATH = _bootstrap_paths()

import torch
from hydra import compose
from hydra.core.global_hydra import GlobalHydra
from hydra.initialize import initialize_config_dir
from omegaconf import open_dict
from torch.utils._pytree import tree_map

from rlinf.config import validate_cfg
from rlinf.models.embodiment.rlt_stage1.rl_token import RLTokenModel
from rlinf.models.embodiment.rlt_stage1.vla_wrapper import Stage1VLAWrapper
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether a Stage1 RL token model can reconstruct joint-prefix embeddings. "
            "This script uses the joint SFT VLA checkpoint to extract prefix embeddings, "
            "and the Stage1 RL token checkpoint to reconstruct them."
        )
    )
    parser.add_argument(
        "--config-name",
        default="rlt_stage1_maniskill_joint",
        help="Hydra config under examples/sft/config.",
    )
    parser.add_argument(
        "--stage1-actor-dir",
        required=True,
        help="Stage1 actor checkpoint dir, typically .../checkpoints/global_step_xxx/actor",
    )
    parser.add_argument(
        "--vla-checkpoint-dir",
        required=True,
        help="Joint SFT actor checkpoint dir used to extract prefix embeddings.",
    )
    parser.add_argument(
        "--rl-token-path",
        default=None,
        help="Path to rl_token_model.pt. Defaults to <stage1-actor-dir>/rl_token/rl_token_model.pt",
    )
    parser.add_argument("--max-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--save-debug-tensors", default=None)
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Extra Hydra overrides, e.g. actor.openpi_data.repo_id=/path/to/data",
    )
    return parser.parse_args()


def _load_cfg(config_name: str, overrides: list[str]):
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(SFT_PATH / "config")):
        cfg = compose(config_name=config_name, overrides=overrides)
    return validate_cfg(cfg)


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_observation_to_device(observation: Any, device: torch.device) -> Any:
    register_pytree_dataclasses(observation)
    return tree_map(
        lambda x: (
            torch.as_tensor(x, device=device).contiguous().clone() if x is not None else x
        ),
        observation,
    )


def _build_first_batches(cfg, max_batches: int):
    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = cfg
    worker._world_size = 1
    worker._rank = 0
    if not os.environ.get("HF_LEROBOT_HOME"):
        train_data_paths = cfg.data.get("train_data_paths", [])
        if len(train_data_paths) > 0:
            first_dataset = train_data_paths[0].get("dataset_path")
            if first_dataset:
                os.environ["HF_LEROBOT_HOME"] = os.path.dirname(first_dataset)
    data_paths = [item.dataset_path for item in cfg.data.train_data_paths]
    data_loader, _ = FSDPVlaSftWorker.build_dataloader(worker, data_paths)

    batches = []
    iterator = iter(data_loader)
    for _ in range(max_batches):
        try:
            batches.append(next(iterator))
        except StopIteration:
            break
    return batches


def _masked_cosine(z: torch.Tensor, z_hat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    cos = torch.nn.functional.cosine_similarity(z_hat, z, dim=-1)
    weights = pad_mask.to(dtype=cos.dtype)
    return (cos * weights).sum() / weights.sum().clamp(min=1.0)


def _masked_mse(z: torch.Tensor, z_hat: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    mse = (z_hat - z).pow(2).mean(dim=-1)
    weights = pad_mask.to(dtype=mse.dtype)
    return (mse * weights).sum() / weights.sum().clamp(min=1.0)


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    flat = tensor.detach().to(torch.float32).reshape(-1)
    finite = flat[torch.isfinite(flat)]
    if finite.numel() == 0:
        return {"shape": list(tensor.shape), "finite_count": 0}
    return {
        "shape": list(tensor.shape),
        "finite_count": int(finite.numel()),
        "mean": float(finite.mean().item()),
        "std": float(finite.std(unbiased=False).item() if finite.numel() > 1 else 0.0),
        "min": float(finite.min().item()),
        "max": float(finite.max().item()),
        "abs_mean": float(finite.abs().mean().item()),
    }


def _load_rl_token_model(cfg, rl_token_path: Path, device: torch.device) -> RLTokenModel:
    stage1_cfg = cfg.actor.model.rlt_stage1
    model = RLTokenModel(
        embedding_dim=int(stage1_cfg.get("embedding_dim", 2048)),
        encoder_layers=int(stage1_cfg.get("encoder_layers", 2)),
        encoder_heads=int(stage1_cfg.get("encoder_heads", 8)),
        decoder_layers=int(stage1_cfg.get("decoder_layers", 2)),
        decoder_heads=int(stage1_cfg.get("decoder_heads", 8)),
    ).to(device)

    checkpoint = torch.load(rl_token_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    cfg = _load_cfg(args.config_name, list(args.overrides))
    with open_dict(cfg):
        cfg.actor.model.model_path = args.vla_checkpoint_dir

    rl_token_path = Path(args.rl_token_path).expanduser() if args.rl_token_path else (
        Path(args.stage1_actor_dir).expanduser() / "rl_token" / "rl_token_model.pt"
    )
    if not rl_token_path.exists():
        raise FileNotFoundError(f"RL token checkpoint not found: {rl_token_path}")

    device = torch.device(args.device)
    stage1_cfg = cfg.actor.model.rlt_stage1
    vla = Stage1VLAWrapper(
        model_path=cfg.actor.model.model_path,
        config_name=stage1_cfg.config_name,
        norm_stats_path=stage1_cfg.get("norm_stats_path", None),
        num_images_in_input=int(stage1_cfg.get("num_images_in_input", 2)),
        num_action_chunks=int(cfg.actor.model.num_action_chunks),
        action_dim=int(cfg.actor.model.action_dim),
        num_steps=int(stage1_cfg.get("num_steps", 5)),
        device=device,
    )
    rl_token_model = _load_rl_token_model(cfg, rl_token_path, device)
    batches = _build_first_batches(cfg, args.max_batches)
    if not batches:
        raise RuntimeError("No batch could be loaded from the stage1 dataset.")

    batch_reports: list[dict[str, Any]] = []
    debug_tensors: list[dict[str, torch.Tensor]] = []

    for batch_index, batch in enumerate(batches):
        observation, actions = batch
        del actions
        observation = _move_observation_to_device(observation, device)

        with torch.no_grad():
            z, pad_mask = vla.extract_embeddings(observation)
            l_ro, z_rl, z_hat = rl_token_model(
                z.to(device, dtype=torch.float32),
                pad_mask.to(device),
            )

        cosine = _masked_cosine(z, z_hat, pad_mask)
        mse = _masked_mse(z, z_hat, pad_mask)
        valid_tokens = int(pad_mask.to(torch.int32).sum().item())

        batch_reports.append(
            {
                "batch_index": batch_index,
                "l_ro": float(l_ro.item()),
                "masked_mse": float(mse.item()),
                "masked_cosine": float(cosine.item()),
                "valid_tokens": valid_tokens,
                "z_stats": _tensor_stats(z),
                "z_hat_stats": _tensor_stats(z_hat),
                "z_rl_stats": _tensor_stats(z_rl),
            }
        )

        if args.save_debug_tensors:
            debug_tensors.append(
                {
                    "z": z.cpu(),
                    "z_hat": z_hat.cpu(),
                    "z_rl": z_rl.cpu(),
                    "pad_mask": pad_mask.cpu(),
                }
            )

    summary = {
        "config_name": args.config_name,
        "stage1_actor_dir": str(Path(args.stage1_actor_dir).expanduser()),
        "vla_checkpoint_dir": str(Path(args.vla_checkpoint_dir).expanduser()),
        "rl_token_path": str(rl_token_path),
        "num_batches": len(batch_reports),
        "avg_l_ro": float(sum(item["l_ro"] for item in batch_reports) / len(batch_reports)),
        "avg_masked_mse": float(
            sum(item["masked_mse"] for item in batch_reports) / len(batch_reports)
        ),
        "avg_masked_cosine": float(
            sum(item["masked_cosine"] for item in batch_reports) / len(batch_reports)
        ),
        "batches": batch_reports,
    }

    print(json.dumps(summary, indent=2))

    if args.save_json:
        save_json_path = Path(args.save_json).expanduser()
        save_json_path.parent.mkdir(parents=True, exist_ok=True)
        save_json_path.write_text(json.dumps(summary, indent=2))

    if args.save_debug_tensors:
        save_tensor_path = Path(args.save_debug_tensors).expanduser()
        save_tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(debug_tensors, save_tensor_path)


if __name__ == "__main__":
    main()
