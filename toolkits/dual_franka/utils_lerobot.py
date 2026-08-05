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
"""Shared helpers for dual-Franka LeRobot toolkit scripts."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

EP_RE = re.compile(r"^episode_(\d+)\.parquet$")


def get_toolkit_logger(name: str) -> logging.Logger:
    """Return an RLinf logger, or a standalone StreamHandler fallback.

    Args:
        name: Logger name used when ``rlinf.utils.logging`` is unavailable.
    """
    try:
        from rlinf.utils.logging import get_logger  # type: ignore[import-not-found]

        return get_logger()
    except ImportError:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger


def add_log_file(logger: logging.Logger, log_path: str) -> logging.FileHandler:
    """Attach a file handler to ``logger``.

    Args:
        logger: Target logger.
        log_path: Destination path. Existing content is overwritten.

    Returns:
        The handler that was attached, so the caller can detach it later.
    """
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    return handler


def header(logger: logging.Logger, title: str) -> None:
    """Log a visually distinct section banner.

    Args:
        logger: Target logger.
        title: The header title to log.
    """
    bar = "=" * 78
    logger.info("\n%s\n%s\n%s", bar, title, bar)


def find_id_dirs(root: Path) -> list[Path]:
    """Return ``id_*`` subdirectories of ``root`` sorted by name.

    Args:
        root: Directory that may contain ``id_*`` children.
    """
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("id_")],
        key=lambda x: x.name,
    )


def load_info(id_dir: Path) -> dict[str, Any]:
    """Load ``meta/info.json`` from a per-id directory."""
    return json.loads((id_dir / "meta" / "info.json").read_text(encoding="utf-8"))


def read_jsonl(p: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dictionaries, ignoring blank lines."""
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(p: Path, rows: list[dict[str, Any]]) -> None:
    """Atomically write ``rows`` as JSONL via a ``.tmp`` file."""
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
    )
    tmp.replace(p)


def fmt_stats(d: dict[str, Any]) -> str:
    """Pretty-print a stats dict (``min``/``max``/``mean``/``std``/``count``)."""
    parts: list[str] = []
    for k in ("min", "max", "mean", "std", "count"):
        if k in d:
            v = d[k]
            if isinstance(v, list) and len(v) == 1:
                vv = v[0]
                parts.append(f"{k}={vv:.4f}" if isinstance(vv, float) else f"{k}={vv}")
            else:
                parts.append(f"{k}={v}")
    return "{" + ", ".join(parts) + "}"
