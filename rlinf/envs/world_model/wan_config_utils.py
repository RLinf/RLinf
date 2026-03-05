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

"""Helpers for Wan world model configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _read_cfg_str(cfg: Any, key: str) -> str | None:
    value = cfg.get(key, None) if hasattr(cfg, "get") else getattr(cfg, key, None)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def get_wan_tokenizer_search_paths(cfg: Any) -> list[str]:
    """Return auto-detected local tokenizer paths based on WAN checkpoint root."""
    ckpt_root = _read_cfg_str(cfg, "wan_wm_hf_ckpt_path")
    if ckpt_root is None:
        return []

    ckpt_dir = Path(ckpt_root).expanduser()
    return [
        str(ckpt_dir / "google"),
        str(ckpt_dir / "tokenizer"),
        str(ckpt_dir),
    ]


def resolve_wan_tokenizer_path(cfg: Any) -> str | None:
    """Resolve tokenizer path for WAN pipeline.

    Resolution order:
    1. ``wan_tokenizer_path`` (explicit user override; local path or HF repo id)
    2. ``tokenizer_path`` (backward-compatible alias)
    3. Existing local paths under ``wan_wm_hf_ckpt_path``:
       ``google/``, ``tokenizer/``, root
    """
    for key in ("wan_tokenizer_path", "tokenizer_path"):
        explicit = _read_cfg_str(cfg, key)
        if explicit is not None:
            return explicit

    for candidate in get_wan_tokenizer_search_paths(cfg):
        if Path(candidate).exists():
            return candidate

    return None
