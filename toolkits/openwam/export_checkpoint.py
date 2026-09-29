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

"""Export an RLinf OpenWAM FSDP (SFT) checkpoint as a native OpenWAM deploy dir.

RLinf's FSDP worker saves ``<global_step_N>/actor/model_state_dict/full_weights.pt``
holding the whole ``OpenWAMPolicy`` state dict: ``architecture.*`` (the OpenWAM
architecture, identical key set to the deploy ``checkpoint_step_*.safetensors``).
OpenWAM's own eval/deploy tooling only reads a self-contained checkpoint
directory, so this script rebuilds one::

    <output>/
      checkpoint_step_<N>.safetensors   architecture weights (VLM excluded, as OpenWAM does)
      config.yaml, normalization_stats.npy, tokenizer/, vlm_backbone/, ...
                                        copied (or symlinked) from the source checkpoint

Example::

    python toolkits/openwam/export_checkpoint.py \
        --rlinf-checkpoint results/libero_sft_openwam/checkpoints/global_step_1000 \
        --source-checkpoint /path/to/openwam-libero-sft-30000 \
        --output /path/to/openwam-libero-sft-rlinf-step1000 --verify cuda
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import tempfile
from pathlib import Path
from typing import Any

ARCHITECTURE_PREFIX = "architecture."
VLM_PREFIX = "vlm_backbone."


def _safetensors_keys(path: Path) -> set[str]:
    """Read a safetensors header without loading tensors."""
    with open(path, "rb") as handle:
        (header_size,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_size))
    return {key for key in header if key != "__metadata__"}


def find_source_safetensors(source_dir: Path) -> Path:
    candidates = sorted(source_dir.glob("checkpoint_step_*.safetensors"))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint_step_*.safetensors in source checkpoint {source_dir}"
        )

    def step(path: Path) -> int:
        digits = path.stem.rsplit("_", 1)[-1]
        return int(digits) if digits.isdigit() else -1

    return max(candidates, key=step)


def infer_step(rlinf_checkpoint: Path) -> int | None:
    for part in reversed(rlinf_checkpoint.parts):
        if part.startswith("global_step_") and part[len("global_step_") :].isdigit():
            return int(part[len("global_step_") :])
    return None


def find_run_normalization_stats(rlinf_checkpoint: Path) -> Path | None:
    """Find stats persisted by an OpenWAM SFT actor checkpoint."""
    candidates = (
        rlinf_checkpoint / "actor" / "normalization_stats.npy",
        rlinf_checkpoint / "normalization_stats.npy",
    )
    return next((path for path in candidates if path.is_file()), None)


def load_policy_state_dict(rlinf_checkpoint: Path) -> dict[str, Any]:
    """Return the full ``OpenWAMPolicy`` state dict saved by the FSDP actor."""
    import torch

    if rlinf_checkpoint.is_file():
        return torch.load(
            rlinf_checkpoint, map_location="cpu", mmap=True, weights_only=False
        )
    full_weights = rlinf_checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    if not full_weights.is_file():
        full_weights = rlinf_checkpoint / "model_state_dict" / "full_weights.pt"
    if full_weights.is_file():
        return torch.load(
            full_weights, map_location="cpu", mmap=True, weights_only=False
        )

    dcp_dir = rlinf_checkpoint / "actor" / "dcp_checkpoint"
    if not dcp_dir.is_dir():
        dcp_dir = rlinf_checkpoint / "dcp_checkpoint"
    if not dcp_dir.is_dir():
        raise FileNotFoundError(
            f"{rlinf_checkpoint} has neither model_state_dict/full_weights.pt nor a "
            "dcp_checkpoint directory (set actor.fsdp_config.save_full_model_weights)"
        )
    # Sharded-only checkpoints: consolidate through torch's DCP format converter.
    from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

    with tempfile.TemporaryDirectory() as tmp:
        consolidated = os.path.join(tmp, "consolidated.pt")
        dcp_to_torch_save(str(dcp_dir), consolidated)
        state = torch.load(consolidated, map_location="cpu", weights_only=False)
    policy_state = _find_policy_state(state)
    if policy_state is None:
        raise ValueError(
            f"{dcp_dir} holds no state dict with '{ARCHITECTURE_PREFIX}*' keys; "
            "is this an RLinf OpenWAM checkpoint?"
        )
    return policy_state


def _find_policy_state(state: Any) -> dict[str, Any] | None:
    """Locate the dict whose keys carry the ``architecture.`` prefix."""
    if isinstance(state, dict):
        if any(str(key).startswith(ARCHITECTURE_PREFIX) for key in state):
            return state
        for value in state.values():
            found = _find_policy_state(value)
            if found is not None:
                return found
    return None


def split_policy_state_dict(
    policy_state: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Split into (architecture weights, unexpected keys)."""
    import torch

    architecture: dict[str, Any] = {}
    unexpected: list[str] = []
    for key, value in policy_state.items():
        if not isinstance(value, torch.Tensor):
            unexpected.append(key)
            continue
        if key.startswith(ARCHITECTURE_PREFIX):
            architecture[key[len(ARCHITECTURE_PREFIX) :]] = value
        else:
            unexpected.append(key)
    return architecture, unexpected


def export_checkpoint(
    rlinf_checkpoint: str | os.PathLike,
    source_checkpoint: str | os.PathLike,
    output: str | os.PathLike,
    *,
    step: int | None = None,
    link_assets: bool = False,
    allow_key_mismatch: bool = False,
) -> Path:
    """Write a self-contained OpenWAM checkpoint dir and return the safetensors path."""
    from safetensors.torch import save_file

    rlinf_checkpoint = Path(rlinf_checkpoint).expanduser().resolve()
    source_dir = Path(source_checkpoint).expanduser().resolve()
    output_dir = Path(output).expanduser().resolve()
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source OpenWAM checkpoint not found: {source_dir}")
    if not (source_dir / "config.yaml").is_file():
        raise FileNotFoundError(f"{source_dir} has no config.yaml")
    if step is None:
        step = infer_step(rlinf_checkpoint)
    if step is None:
        raise ValueError("Pass --step: the RLinf checkpoint path has no global_step_N")

    policy_state = load_policy_state_dict(rlinf_checkpoint)
    architecture, unexpected = split_policy_state_dict(policy_state)
    if unexpected:
        raise ValueError(
            f"RLinf state dict has keys outside architecture.*: {unexpected[:10]}"
        )
    if not architecture:
        raise ValueError("RLinf state dict contains no architecture.* tensors")
    # OpenWAM keeps the frozen VLM in <ckpt>/vlm_backbone/, not in safetensors.
    architecture = {
        key: value
        for key, value in architecture.items()
        if not key.startswith(VLM_PREFIX)
    }

    source_file = find_source_safetensors(source_dir)
    source_keys = {
        key for key in _safetensors_keys(source_file) if not key.startswith(VLM_PREFIX)
    }
    missing = sorted(source_keys - architecture.keys())
    extra = sorted(architecture.keys() - source_keys)
    if (missing or extra) and not allow_key_mismatch:
        raise ValueError(
            "Exported architecture keys differ from the source checkpoint "
            f"{source_file.name}: missing={missing[:5]} ({len(missing)}), "
            f"extra={extra[:5]} ({len(extra)}). Is --source-checkpoint the "
            "checkpoint this SFT run started from?"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    run_stats = find_run_normalization_stats(rlinf_checkpoint)
    for entry in source_dir.iterdir():
        if entry.suffix == ".safetensors":
            continue
        target = output_dir / entry.name
        if target.exists() or target.is_symlink():
            continue
        if link_assets:
            os.symlink(entry, target, target_is_directory=entry.is_dir())
        elif entry.is_dir():
            shutil.copytree(entry, target)
        else:
            shutil.copy2(entry, target)

    if run_stats is not None:
        # The source checkpoint's stats describe the initialization data.
        # Prefer the artifact captured by the SFT worker when the run used a
        # different dataset root or an explicitly shared stats file.
        target = output_dir / "normalization_stats.npy"
        if target.exists() or target.is_symlink():
            target.unlink()
        shutil.copy2(run_stats, target)

    safetensors_path = output_dir / f"checkpoint_step_{step}.safetensors"
    save_file(
        {key: value.contiguous() for key, value in architecture.items()},
        str(safetensors_path),
        metadata={
            "format": "pt",
            "rlinf_checkpoint": str(rlinf_checkpoint),
            "source_checkpoint": str(source_file),
        },
    )
    return safetensors_path


def verify_export(output_dir: Path, safetensors_path: Path, device: str) -> None:
    """Load the export with OpenWAM's own loader and compare a few tensors."""
    import torch
    from openwam.deploy import load_from_checkpoint_dir
    from safetensors import safe_open

    _, architecture = load_from_checkpoint_dir(
        str(output_dir), device=device, ckpt_name=safetensors_path.name
    )
    state = architecture.state_dict()
    checked = 0
    with safe_open(str(safetensors_path), framework="pt") as handle:
        for key in handle.keys():
            if "action_backbone" not in key and checked >= 8:
                continue
            expected = handle.get_tensor(key)
            actual = state[key].detach().to("cpu")
            expected_for_compare = expected.to(dtype=actual.dtype)
            if not torch.equal(actual, expected_for_compare):
                raise RuntimeError(f"Loaded tensor differs from export: {key}")
            checked += 1
            if checked >= 64:
                break
    print(f"verify: OpenWAM loaded {output_dir.name}; {checked} tensors match")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rlinf-checkpoint",
        required=True,
        help="RLinf global_step_N directory (or a full_weights.pt file)",
    )
    parser.add_argument(
        "--source-checkpoint",
        required=True,
        help="OpenWAM checkpoint dir the SFT run started from (config/tokenizer/stats)",
    )
    parser.add_argument("--output", required=True, help="Output checkpoint directory")
    parser.add_argument("--step", type=int, default=None, help="Step in the file name")
    parser.add_argument(
        "--link-assets",
        action="store_true",
        help="Symlink config/tokenizer/vlm_backbone from the source instead of copying",
    )
    parser.add_argument(
        "--allow-key-mismatch",
        action="store_true",
        help="Export even if the key set differs from the source safetensors",
    )
    parser.add_argument(
        "--verify",
        default=None,
        metavar="DEVICE",
        help="Reload the export with openwam.deploy.load_from_checkpoint_dir on DEVICE",
    )
    args = parser.parse_args()

    safetensors_path = export_checkpoint(
        args.rlinf_checkpoint,
        args.source_checkpoint,
        args.output,
        step=args.step,
        link_assets=args.link_assets,
        allow_key_mismatch=args.allow_key_mismatch,
    )
    size_gb = safetensors_path.stat().st_size / 2**30
    print(f"wrote {safetensors_path} ({size_gb:.1f} GiB)")
    if args.verify:
        verify_export(safetensors_path.parent, safetensors_path, args.verify)


if __name__ == "__main__":
    main()
