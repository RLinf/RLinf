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

"""Merge single / ensemble STEAM critic checkpoints into one ensemble checkpoint.

Fuses independently-trained :class:`SteamCriticModel` checkpoints (or members
extracted from an existing ensemble) into a single inference checkpoint. The
output ``actor`` directory contains ``config.json`` (with ``ensemble_size`` set
to the member count), tokenizer / processor assets, a merged ``members.*`` state
dict in ``model_state_dict/full_weights.pt``, and a ``merge_manifest.json``
recording each member's provenance. Optimizer / FSDP DCP state is intentionally
not copied — the result is for inference / visualization only.

Each member source is a ``PATH`` (single-model checkpoint) or ``PATH:member_idx``
(extract ``members.member_idx`` from an ensemble checkpoint). A path may point at
an ``actor`` directory, a ``model_state_dict`` directory, or a direct
``full_weights.pt`` file.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

ASSET_NAMES = (
    "added_tokens.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
)

CONFIG_COMPAT_KEYS = (
    "model_type",
    "num_bins",
    "stride_k",
    "fusion_hidden_dim",
    "num_frames_per_pair",
    "vision_repo_id",
    "language_repo_id",
    "precision",
)


@dataclass(frozen=True)
class MemberSpec:
    """One requested output member."""

    raw: str
    checkpoint: Path
    source_member_idx: int | None


@dataclass(frozen=True)
class ResolvedCheckpoint:
    """Resolved checkpoint paths."""

    actor_dir: Path
    weights_path: Path
    config_path: Path


def _parse_member_spec(raw: str) -> MemberSpec:
    """Parse ``PATH`` or ``PATH:member_idx``.

    A plain path is treated as a single-model checkpoint. ``PATH:2`` extracts
    ``members.2`` from an ensemble checkpoint. Paths with ordinary colons are
    uncommon on Linux; use absolute POSIX paths for this tool.
    """
    if ":" not in raw:
        return MemberSpec(raw=raw, checkpoint=Path(raw), source_member_idx=None)

    maybe_path, maybe_idx = raw.rsplit(":", 1)
    if maybe_idx.isdigit():
        return MemberSpec(
            raw=raw,
            checkpoint=Path(maybe_path),
            source_member_idx=int(maybe_idx),
        )
    return MemberSpec(raw=raw, checkpoint=Path(raw), source_member_idx=None)


def _resolve_checkpoint(path: Path) -> ResolvedCheckpoint:
    """Resolve actor/model_state_dict/full_weights paths to canonical files."""
    path = path.expanduser().resolve()

    if path.is_file():
        weights_path = path
        model_state_dir = weights_path.parent
        actor_dir = (
            model_state_dir.parent
            if model_state_dir.name == "model_state_dict"
            else path.parent
        )
    elif (path / "full_weights.pt").is_file():
        weights_path = path / "full_weights.pt"
        actor_dir = path.parent if path.name == "model_state_dict" else path
    elif (path / "model_state_dict" / "full_weights.pt").is_file():
        weights_path = path / "model_state_dict" / "full_weights.pt"
        actor_dir = path
    else:
        raise FileNotFoundError(
            f"Could not find full_weights.pt under {path}. Expected an actor dir, "
            "model_state_dict dir, or direct full_weights.pt file."
        )

    config_candidates = [
        actor_dir / "config.json",
        actor_dir.parent / "config.json",
        weights_path.parent / "config.json",
    ]
    for config_path in config_candidates:
        if config_path.is_file():
            return ResolvedCheckpoint(
                actor_dir=actor_dir,
                weights_path=weights_path,
                config_path=config_path,
            )

    raise FileNotFoundError(
        f"Could not find config.json for checkpoint {path}. Tried: "
        f"{[str(p) for p in config_candidates]}"
    )


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _assert_config_compatible(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    candidate_name: str,
) -> None:
    """Fail when a member architecture differs from the reference."""
    mismatches = []
    missing_candidate_keys = []
    for key in CONFIG_COMPAT_KEYS:
        reference_value = reference.get(key)
        candidate_value = candidate.get(key)
        if reference_value == candidate_value:
            continue
        if reference_value is not None and candidate_value is None:
            missing_candidate_keys.append((key, reference_value))
            continue
        mismatches.append((key, reference_value, candidate_value))
    if missing_candidate_keys:
        detail = "; ".join(
            f"{key}: using reference={value!r}" for key, value in missing_candidate_keys
        )
        logger.warning(
            "%s is missing compatible config metadata (%s). Tensor keys and "
            "shapes will still be checked.",
            candidate_name,
            detail,
        )
    if mismatches:
        detail = "; ".join(
            f"{key}: reference={ref!r}, {candidate_name}={val!r}"
            for key, ref, val in mismatches
        )
        raise ValueError(
            f"Incompatible checkpoint config for {candidate_name}: {detail}"
        )


def _member_state_dict(
    state_dict: dict[str, torch.Tensor],
    spec: MemberSpec,
) -> dict[str, torch.Tensor]:
    """Return a single-member state dict with keys normalized to ``model.*``."""
    if spec.source_member_idx is None:
        if not all(key.startswith("model.") for key in state_dict):
            member_like = any(key.startswith("members.") for key in state_dict)
            hint = (
                " Input looks like an ensemble checkpoint; append ':member_idx' "
                "to the member, e.g. /path/to/actor:2."
                if member_like
                else ""
            )
            raise ValueError(
                f"Member {spec.raw} was provided as a single model, but its keys "
                f"do not all start with 'model.'.{hint}"
            )
        return state_dict

    prefix = f"members.{spec.source_member_idx}."
    extracted = {
        key.removeprefix(prefix): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not extracted:
        raise ValueError(
            f"Member {spec.raw} requested {prefix!r}, but no matching keys exist."
        )
    if not all(key.startswith("model.") for key in extracted):
        bad = [key for key in sorted(extracted) if not key.startswith("model.")][:10]
        raise ValueError(
            f"Extracted keys from {spec.raw} are not normalized model keys; bad={bad}"
        )
    return extracted


def _copy_assets(reference_actor_dir: Path, output_actor_dir: Path) -> None:
    for name in ASSET_NAMES:
        src = reference_actor_dir / name
        if src.exists():
            shutil.copy2(src, output_actor_dir / name)


def _write_config(
    reference_config: dict[str, Any],
    output_actor_dir: Path,
    *,
    ensemble_size: int,
) -> None:
    config = dict(reference_config)
    config["ensemble_size"] = int(ensemble_size)
    # Drop the retired inference-mode knobs if an older checkpoint carries
    # them; the ensemble always aggregates with the worst-of-N rule now.
    config.pop("inference_mode", None)
    config.pop("uwo_lambda", None)

    with (output_actor_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def merge_ensemble_checkpoints(
    members: list[str],
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Merge ``members`` into one ensemble inference checkpoint at ``output``.

    Args:
        members: Member sources in output order. Each is ``PATH`` (single-model
            checkpoint) or ``PATH:member_idx`` (extract ``members.member_idx``
            from an ensemble checkpoint).
        output: Output ``actor`` directory; the merged weights are written to
            ``output/model_state_dict/full_weights.pt``.
        overwrite: Replace an existing ``output`` directory when ``True``.

    Returns:
        The resolved output ``actor`` directory.
    """
    specs = [_parse_member_spec(raw) for raw in members]
    if not specs:
        raise ValueError("At least one member is required.")

    output_actor_dir = Path(output).expanduser().resolve()
    if output_actor_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_actor_dir}. "
                "Pass overwrite=True to replace it."
            )
        shutil.rmtree(output_actor_dir)
    output_model_dir = output_actor_dir / "model_state_dict"
    output_model_dir.mkdir(parents=True)

    resolved = [_resolve_checkpoint(spec.checkpoint) for spec in specs]
    configs = [_load_json(item.config_path) for item in resolved]
    reference_config = configs[0]
    for idx, config in enumerate(configs[1:], start=1):
        _assert_config_compatible(
            reference_config,
            config,
            candidate_name=f"member #{idx} ({specs[idx].raw})",
        )

    _copy_assets(resolved[0].actor_dir, output_actor_dir)
    _write_config(reference_config, output_actor_dir, ensemble_size=len(specs))

    merged: dict[str, torch.Tensor] = {}
    reference_keys: set[str] | None = None
    manifest: dict[str, Any] = {
        "members": {},
        "output": str(output_actor_dir),
        "note": (
            "Merged for inference/visualization. No optimizer or FSDP "
            "dcp_checkpoint state is included."
        ),
    }

    for output_idx, (spec, item) in enumerate(zip(specs, resolved)):
        logger.info("Loading member %d: %s", output_idx, spec.raw)
        state_dict = torch.load(
            item.weights_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        member_sd = _member_state_dict(state_dict, spec)
        member_keys = set(member_sd)
        if reference_keys is None:
            reference_keys = member_keys
        elif member_keys != reference_keys:
            only_ref = sorted(reference_keys - member_keys)[:10]
            only_cur = sorted(member_keys - reference_keys)[:10]
            raise ValueError(
                f"Member {output_idx} keys differ from member 0: "
                f"only_member0={only_ref}, only_member{output_idx}={only_cur}"
            )

        for key, value in member_sd.items():
            merged[f"members.{output_idx}.{key}"] = value

        manifest["members"][str(output_idx)] = {
            "raw": spec.raw,
            "actor_dir": str(item.actor_dir),
            "weights_path": str(item.weights_path),
            "source_member_idx": spec.source_member_idx,
            "num_tensors": len(member_sd),
        }

        probe = "model.value_head.3.weight"
        if probe in member_sd:
            tensor = member_sd[probe]
            logger.info(
                "  %s: shape=%s dtype=%s norm=%.6f",
                probe,
                tuple(tensor.shape),
                tensor.dtype,
                float(tensor.float().norm()),
            )

    out_weights = output_model_dir / "full_weights.pt"
    logger.info("Saving %d tensors to %s", len(merged), out_weights)
    torch.save(merged, out_weights)

    with (output_actor_dir / "merge_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    logger.info("Done. Actor checkpoint: %s", output_actor_dir)
    return output_actor_dir


__all__ = [
    "ASSET_NAMES",
    "CONFIG_COMPAT_KEYS",
    "MemberSpec",
    "ResolvedCheckpoint",
    "merge_ensemble_checkpoints",
]
