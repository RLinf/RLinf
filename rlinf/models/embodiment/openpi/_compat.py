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
"""Runtime compatibility shims that adapt upstream OpenPI to the LeRobot
release line we have to live with.

Two specific gaps are bridged:

1. **``lerobot.common.datasets`` import path.**  Upstream OpenPI's
   ``training/data_loader.py`` imports
   ``lerobot.common.datasets.lerobot_dataset``, the layout used by
   ``lerobot < 0.4``.  Modern releases (>= 0.4) moved the package to the flat
   ``lerobot.datasets.lerobot_dataset`` path and dropped the ``common``
   intermediate.  We expose the modern modules under the legacy import path
   by registering aliases in :mod:`sys.modules` so OpenPI can keep working
   with no source changes.

2. **``meta.tasks`` shape change.**  ``LeRobotDatasetMetadata.tasks`` used to
   be a ``dict[int, str]`` mapping ``task_index → task name``.  In recent
   LeRobot releases it became a ``pandas.DataFrame`` indexed by task name with
   a ``task_index`` column.  OpenPI's ``PromptFromLeRobotTask`` transform
   still expects the dict shape, so a per-frame ``task_index=0`` lookup
   raises ``ValueError: task_index=0 not found in task mapping`` against the
   DataFrame.  We patch the transform in place so it normalises a DataFrame
   to the dict shape during construction.

Both shims are **idempotent** — calling ``install_compat_shims`` multiple
times is safe — and **no-op when not needed** (e.g. when OpenPI's upstream
catches up, or when the legacy LeRobot layout is already importable).

Until upstream OpenPI lands the equivalent fixes, importing the
:mod:`rlinf.models.embodiment.openpi` package automatically installs both
shims so downstream SFT / RL recipes work out of the box.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from typing import Any

_logger = logging.getLogger(__name__)

_SHIM_INSTALLED_ATTR = "_rlinf_openpi_compat_installed"


def _install_lerobot_common_shim() -> None:
    """Alias ``lerobot.common.datasets[.lerobot_dataset]`` to the modern path.

    No-op when the legacy modules are already importable (so this stays
    invisible if a user is running an older LeRobot release) and when
    LeRobot itself is not installed (we let OpenPI surface that error).
    """
    try:
        importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        return
    except ImportError:
        pass

    try:
        modern_dataset = importlib.import_module(
            "lerobot.datasets.lerobot_dataset"
        )
        modern_datasets_pkg = importlib.import_module("lerobot.datasets")
    except ImportError:
        # LeRobot is not installed; nothing to alias.  OpenPI will fail with
        # its own clearer error if/when it actually needs lerobot.
        return

    # Build a minimal ``lerobot.common`` package object.  We only need the
    # one subpackage path to exist for ``import lerobot.common.datasets...``
    # to succeed; we don't expose anything else under ``lerobot.common``.
    common_pkg = sys.modules.get("lerobot.common")
    if common_pkg is None:
        common_pkg = types.ModuleType("lerobot.common")
        # ``__path__`` must exist for Python to treat this as a package.
        common_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["lerobot.common"] = common_pkg

    sys.modules.setdefault("lerobot.common.datasets", modern_datasets_pkg)
    sys.modules.setdefault(
        "lerobot.common.datasets.lerobot_dataset", modern_dataset
    )

    # Expose the subpackages as attributes too, so ``from lerobot.common
    # import datasets`` and ``from lerobot.common.datasets import
    # lerobot_dataset`` both resolve the same modules.
    common_pkg.datasets = modern_datasets_pkg  # type: ignore[attr-defined]
    modern_datasets_pkg.lerobot_dataset = modern_dataset  # type: ignore[attr-defined]

    _logger.debug(
        "Installed lerobot.common.datasets compat alias → %s",
        modern_dataset.__name__,
    )


def _install_openpi_pandas_tasks_patch() -> None:
    """Make OpenPI's ``PromptFromLeRobotTask`` accept a pandas DataFrame.

    OpenPI assumes ``LeRobotDatasetMetadata.tasks`` is a ``dict[int, str]``
    mapping.  Modern LeRobot releases return a DataFrame indexed by the task
    string with a ``task_index`` column instead.  We replace the upstream
    dataclass's ``__init__`` with a thin wrapper that converts the DataFrame
    to the expected dict before delegating, leaving everything else (class
    identity, ``isinstance`` checks, picklability of instances across
    DataLoader workers) untouched.

    Subclassing the dataclass is tempting but fails under multiprocessing:
    the worker pickler can't import a locally-scoped subclass and raises
    ``AttributeError: Can't pickle local object``.  Patching ``__init__`` on
    the original class avoids that entirely.
    """
    try:
        openpi_transforms = importlib.import_module("openpi.transforms")
    except ImportError:
        # OpenPI not installed — nothing to patch.
        return

    original = getattr(openpi_transforms, "PromptFromLeRobotTask", None)
    if original is None or getattr(original, _SHIM_INSTALLED_ATTR, False):
        return

    try:
        import pandas as pd
    except ImportError:
        return

    original_init = original.__init__

    def _patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        tasks = self.tasks
        if isinstance(tasks, pd.DataFrame):
            if "task_index" in tasks.columns:
                keys = tasks["task_index"].astype(int).tolist()
                values = tasks.index.tolist()
            else:
                # Fallback shape: index is the integer key column.
                keys = [int(k) for k in tasks.index.tolist()]
                values = (
                    tasks.iloc[:, 0].tolist() if tasks.shape[1] else keys
                )
            # ``frozen=True`` blocks regular setattr; bypass it.
            object.__setattr__(self, "tasks", dict(zip(keys, values)))

    _patched_init.__wrapped__ = original_init  # type: ignore[attr-defined]
    original.__init__ = _patched_init
    setattr(original, _SHIM_INSTALLED_ATTR, True)
    _logger.debug(
        "Patched openpi.transforms.PromptFromLeRobotTask.__init__ to accept "
        "pandas.DataFrame tasks metadata."
    )


def install_compat_shims() -> None:
    """Install every OpenPI compat shim if not already installed.

    Safe to call repeatedly.  Should be invoked before any OpenPI module
    that imports ``lerobot.common.datasets`` or constructs a
    ``PromptFromLeRobotTask`` is loaded.
    """
    _install_lerobot_common_shim()
    _install_openpi_pandas_tasks_patch()


# ---------------------------------------------------------------------------
# Persistent (on-disk) variant
# ---------------------------------------------------------------------------

_PERSISTENT_PTH_NAME = "rlinf_openpi_compat.pth"
_PERSISTENT_PTH_BODY = (
    "import rlinf.models.embodiment.openpi._compat as _c; "
    "_c.install_compat_shims()\n"
)


def install_persistent_shims(site_packages_dir: str | None = None) -> str | None:
    """Drop a ``.pth`` file in *site_packages_dir* that runs both shims at site
    init, so they fire even in freshly-spawned subprocesses (e.g. torch
    DataLoader workers under ``multiprocessing.get_start_method() == "spawn"``).

    The in-process :func:`install_compat_shims` is sufficient for the parent
    process, but ``multiprocessing.spawn`` workers start fresh Python
    interpreters and re-import OpenPI before our setup code gets a chance to
    run.  Python processes ``.pth`` files inside ``site-packages/`` during
    :mod:`site` initialisation; any line starting with ``import`` is exec'd
    on the spot, so this is the earliest hook into a worker's lifecycle.

    Args:
        site_packages_dir: Directory to write the ``.pth`` file into.  When
            ``None``, the first writable ``site-packages`` directory on
            :data:`sys.path` is picked.

    Returns:
        Path to the written ``.pth`` file, or ``None`` if no suitable site
        directory could be found.
    """
    import os
    import site

    if site_packages_dir is None:
        candidates = []
        try:
            candidates.extend(site.getsitepackages())
        except Exception:
            pass
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
        site_packages_dir = next(
            (c for c in candidates if c and os.path.isdir(c) and os.access(c, os.W_OK)),
            None,
        )
    if site_packages_dir is None:
        _logger.warning(
            "Cannot install persistent openpi compat shim — no writable "
            "site-packages directory found on sys.path."
        )
        return None

    pth_path = os.path.join(site_packages_dir, _PERSISTENT_PTH_NAME)
    # Idempotent — leave existing content alone if it already matches.
    try:
        with open(pth_path) as f:
            existing = f.read()
    except OSError:
        existing = None
    if existing != _PERSISTENT_PTH_BODY:
        with open(pth_path, "w") as f:
            f.write(_PERSISTENT_PTH_BODY)
        _logger.info("Installed persistent openpi compat shim at %s", pth_path)
    return pth_path


def _main_cli() -> None:
    """Entry point: ``python -m rlinf.models.embodiment.openpi._compat install``.

    Used by ``requirements/install.sh`` after OpenPI is installed.
    """
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "action",
        choices=["install"],
        help="`install` writes the .pth file that triggers the shims in "
        "every Python process started in this venv.",
    )
    p.add_argument(
        "--site-packages",
        default=None,
        help="Override site-packages directory (defaults to the first writable "
        "entry on sys.path).",
    )
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.action == "install":
        pth = install_persistent_shims(site_packages_dir=args.site_packages)
        if pth is None:
            raise SystemExit(2)
        print(pth)


if __name__ == "__main__":
    _main_cli()
