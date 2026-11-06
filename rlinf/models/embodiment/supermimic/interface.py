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

"""Policy-interface record stored next to every SuperMimic checkpoint.

Several settings change what the networks read or predict without changing
any tensor shape, so a mismatched checkpoint would load silently and act on
the wrong inputs. The actor writes these settings beside its checkpoint;
resuming and evaluation compare them with the config and refuse a mismatch.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

INTERFACE_FILE = "supermimic_interface.json"

# Settings of ``actor.model`` / ``rollout.model`` every consumer must share.
MODEL_KEYS = (
    "goal_encoding",
    "goal_anchor",
    "obs_dim",
    "velocity_tracking",
    "critic_features",
    "input_norm",
    "wm_state_loss",
)
# Settings of ``algorithm`` that shape what the trained networks predict.
TRAINING_KEYS = ("reward_anchor", "residual_frame", "termination_body")


def interface_record(
    model_cfg: DictConfig, algorithm_cfg: DictConfig | None = None
) -> dict[str, Any]:
    """Interface settings of a model config and, for training, its algorithm."""
    record = {key: model_cfg.get(key) for key in MODEL_KEYS}
    if algorithm_cfg is not None:
        record.update({key: algorithm_cfg.get(key) for key in TRAINING_KEYS})
    return record


def write_interface(directory: str | os.PathLike, record: dict[str, Any]) -> None:
    """Write ``record`` into ``directory``."""
    path = Path(directory) / INTERFACE_FILE
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def find_interface(path: str | os.PathLike, levels: int = 3) -> dict | None:
    """Record beside ``path`` or in up to ``levels`` parent directories.

    ``path`` is an actor checkpoint directory or a file inside it, such as
    ``actor/model_state_dict/full_weights.pt``.
    """
    current = Path(path)
    if current.is_file():
        current = current.parent
    for _ in range(levels + 1):
        candidate = current / INTERFACE_FILE
        if candidate.is_file():
            return json.loads(candidate.read_text())
        current = current.parent
    return None


def check_interface(
    found: dict[str, Any] | None, expected: dict[str, Any], source: str
) -> None:
    """Raise when a checkpoint's record differs from the config.

    A checkpoint without a record predates this format and is rejected.
    """
    if found is None:
        raise ValueError(
            f"{source} has no {INTERFACE_FILE}; it was written before the current "
            "SuperMimic checkpoint format and cannot be loaded"
        )
    for key, value in expected.items():
        if found.get(key) != value:
            raise ValueError(
                f"{source} was trained with {key}={found.get(key)!r}, but this "
                f"config has {key}={value!r}"
            )
