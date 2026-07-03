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

"""Shared ``meta/mixture_config.yaml`` read/write for offline advantage pipelines.

Model-agnostic YAML I/O shared by RECAP and STEAM: both record per-tag advantage
metadata under ``tags[<tag>]`` of each dataset's ``meta/mixture_config.yaml``,
and both write the file under the dataset's ``meta/`` directory. PyYAML is the
only dependency so CPU-only tools import this without torch / lerobot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import yaml

PathLike = Union[str, Path]


def mixture_config_path(dataset_path: PathLike) -> Path:
    """Return ``<dataset_path>/meta/mixture_config.yaml``."""
    return Path(dataset_path) / "meta" / "mixture_config.yaml"


def read_mixture_config(dataset_path: PathLike) -> dict[str, Any]:
    """Load ``meta/mixture_config.yaml`` as a dict (``{}`` when absent/empty)."""
    p = mixture_config_path(dataset_path)
    if not p.exists():
        return {}
    with open(p, "r") as f:
        loaded = yaml.safe_load(f)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise RuntimeError(
            f"mixture_config.yaml at {p} is not a mapping; refusing to read"
        )
    return loaded


def write_mixture_config(dataset_path: PathLike, config: dict[str, Any]) -> Path:
    """Write the whole ``config`` dict to ``meta/mixture_config.yaml``."""
    p = mixture_config_path(dataset_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return p


def write_mixture_config_tag(
    dataset_path: PathLike, tag: str, entry: dict[str, Any]
) -> Path:
    """Merge ``tags[tag] = entry`` into ``meta/mixture_config.yaml``.

    Only the ``tags`` sub-mapping is touched; every other top-level field is
    preserved verbatim. Returns the config path that was written.
    """
    existing = read_mixture_config(dataset_path)
    tags = existing.get("tags") or {}
    if not isinstance(tags, dict):
        raise RuntimeError(
            f"mixture_config.yaml 'tags' field at "
            f"{mixture_config_path(dataset_path)} is not a mapping"
        )
    tags[str(tag)] = entry
    existing["tags"] = tags
    return write_mixture_config(dataset_path, existing)


__all__ = [
    "mixture_config_path",
    "read_mixture_config",
    "write_mixture_config",
    "write_mixture_config_tag",
]
