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

"""SuperMimic checkpoint layout, integrity metadata, and RNG capture.

A checkpoint is ``global_step_<N>/`` holding ``supermimic.pt`` (shared
weights, optimizers, schedulers, counters, reset-bin statistics), one
``rank_<r>.pt`` per rank (RNG streams), and ``checkpoint_meta.json`` with
sizes and SHA-256 sums. It is written under ``global_step_<N>.tmp`` and
renamed once complete, so a directory without the ``.tmp`` suffix and with
matching sums is a finished checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import time
from pathlib import Path

import numpy as np
import torch

MODEL_FILE = "supermimic.pt"
META_FILE = "checkpoint_meta.json"
# 3: torso-anchored body reward, central-difference reference velocity, and
# the goal window in the metadata. Earlier formats trained another reward.
FORMAT_VERSION = 3
_STEP_DIR = re.compile(r"^global_step_(\d+)$")
# Metadata that fixes the policy's inputs. A config must match all of them.
POLICY_KEYS = (
    "goal_dim",
    "obs_dim",
    "goal_encoding",
    "goal_window",
    "velocity_tracking",
    "critic_features",
    "wm_state_loss",
    "reward_anchor",
    "input_norm",
    "residual_frame",
)
# Checkpoints written before these fields used the paper recipe. A missing
# field therefore means that older recipe, not zita-ch's. A config that
# omits the field still selects zita-ch; only the saved metadata does this.
_MISSING_POLICY = {
    "critic_features": "root4",
    "wm_state_loss": "paper",
    "reward_anchor": "torso",
    "input_norm": "none",
    "residual_frame": "heading",
}


def policy_value(meta: dict, key: str):
    """One policy-interface field.

    A checkpoint that predates a field loads as the recipe that was current
    then: ``root4`` critic features, the unweighted state loss, a torso
    reward anchor, no running norm, and the heading-frame residual.
    """
    value = meta.get(key)
    if value is None and key in _MISSING_POLICY:
        return _MISSING_POLICY[key]
    return value


def rank_file(rank: int) -> str:
    """File name of one rank's RNG state."""
    return f"rank_{int(rank)}.pt"


def temp_dir(directory: str | os.PathLike) -> Path:
    """Staging directory a checkpoint is written to before it is published."""
    path = Path(directory)
    return path.with_name(path.name + ".tmp")


def sha256_file(path: str | os.PathLike) -> str:
    """Hex SHA-256 of a file, read in 1 MiB chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_meta(directory: str | os.PathLike, meta: dict) -> dict:
    """Record every file's size and sum. Raises if a rank file is missing."""
    directory = Path(directory)
    world = int(meta["world_size"])
    names = [MODEL_FILE] + [rank_file(r) for r in range(world)]
    missing = [name for name in names if not (directory / name).is_file()]
    if missing:
        raise RuntimeError(
            f"checkpoint {directory} is missing {missing}; every rank must write "
            "to the same directory, so log_path must be on storage shared by all nodes"
        )
    meta = dict(meta)
    meta["format"] = FORMAT_VERSION
    meta["files"] = {
        name: {
            "bytes": (directory / name).stat().st_size,
            "sha256": sha256_file(directory / name),
        }
        for name in names
    }
    meta["written_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = directory / (META_FILE + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, sort_keys=True))
    os.replace(tmp, directory / META_FILE)
    return meta


def read_meta(directory: str | os.PathLike) -> dict | None:
    """Parsed ``checkpoint_meta.json``, or ``None`` when it is missing."""
    path = Path(directory) / META_FILE
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def verify_checkpoint(directory: str | os.PathLike, check_sums: bool = True) -> dict:
    """Return the metadata of a complete checkpoint or raise ``ValueError``."""
    directory = Path(directory)
    if directory.name.endswith(".tmp"):
        raise ValueError(f"{directory} is an unfinished checkpoint")
    try:
        meta = read_meta(directory)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{directory}: unreadable {META_FILE}: {error}") from error
    if meta is None:
        raise ValueError(f"{directory} has no {META_FILE}")
    if meta.get("format") != FORMAT_VERSION:
        raise ValueError(
            f"{directory} is checkpoint format {meta.get('format')!r}, this code "
            f"reads format {FORMAT_VERSION}; earlier formats used another reward "
            "and reference velocity, so retrain instead of loading it"
        )
    for name, info in meta.get("files", {}).items():
        path = directory / name
        if not path.is_file():
            raise ValueError(f"{directory} is missing {name}")
        if path.stat().st_size != int(info["bytes"]):
            raise ValueError(f"{directory}/{name} has the wrong size")
        if check_sums and sha256_file(path) != info["sha256"]:
            raise ValueError(f"{directory}/{name} does not match its SHA-256")
    if MODEL_FILE not in meta.get("files", {}):
        raise ValueError(f"{directory} metadata does not list {MODEL_FILE}")
    return meta


def is_complete_checkpoint(directory: str | os.PathLike) -> bool:
    """True when ``verify_checkpoint`` accepts the directory."""
    try:
        verify_checkpoint(directory)
    except ValueError:
        return False
    return True


def find_latest_checkpoint(root: str | os.PathLike) -> str | None:
    """Newest complete ``global_step_<N>`` under ``root``; skips broken ones."""
    root = Path(root)
    if not root.is_dir():
        return None
    steps = []
    for child in root.iterdir():
        match = _STEP_DIR.match(child.name)
        if match and child.is_dir():
            steps.append((int(match.group(1)), child))
    for _, child in sorted(steps, reverse=True):
        if is_complete_checkpoint(child):
            return str(child)
    return None


def publish(tmp: str | os.PathLike, final: str | os.PathLike) -> None:
    """Rename a finished ``.tmp`` directory into place.

    An existing directory of the same step is moved aside first and removed
    only after the new one is in place.
    """
    tmp, final = Path(tmp), Path(final)
    old = final.with_name(final.name + ".old")
    if old.exists():
        shutil.rmtree(old)
    if final.exists():
        os.replace(final, old)
    os.replace(tmp, final)
    if old.exists():
        shutil.rmtree(old)


def capture_rng(device: torch.device) -> dict:
    """RNG streams as tensors and plain types, loadable with weights_only."""
    version, internal, gauss = random.getstate()
    name, keys, pos, has_gauss, cached = np.random.get_state()
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python": {
            "version": int(version),
            "internal": torch.tensor(internal, dtype=torch.int64),
            "gauss": gauss,
        },
        "numpy": {
            "name": str(name),
            "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
    }
    if torch.device(device).type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(torch.device(device))
    return state


def restore_rng(state: dict, device: torch.device) -> None:
    """Restore the torch, CUDA, Python and NumPy RNG streams of one rank."""
    torch.set_rng_state(state["torch_cpu"].cpu())
    python = state["python"]
    random.setstate(
        (
            python["version"],
            tuple(int(v) for v in python["internal"].tolist()),
            python["gauss"],
        )
    )
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].numpy().astype(np.uint32),
            numpy_state["pos"],
            numpy_state["has_gauss"],
            numpy_state["cached"],
        )
    )
    if torch.device(device).type == "cuda" and "torch_cuda" in state:
        torch.cuda.set_rng_state(state["torch_cuda"].cpu(), torch.device(device))
