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

"""Small compatibility shims for OpenPI and modern LeRobot releases."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

_PATCHED = "_rlinf_openpi_compat_patched"


def _alias_legacy_lerobot_import() -> None:
    """Expose modern LeRobot modules at OpenPI's legacy import path."""
    try:
        importlib.import_module("lerobot.common.datasets.lerobot_dataset")
        return
    except ImportError:
        pass

    try:
        datasets = importlib.import_module("lerobot.datasets")
        dataset = importlib.import_module("lerobot.datasets.lerobot_dataset")
    except ImportError:
        return

    common = sys.modules.get("lerobot.common")
    if common is None:
        common = types.ModuleType("lerobot.common")
        common.__path__ = []  # type: ignore[attr-defined]
        sys.modules["lerobot.common"] = common
    sys.modules.setdefault("lerobot.common.datasets", datasets)
    sys.modules.setdefault("lerobot.common.datasets.lerobot_dataset", dataset)
    common.datasets = datasets  # type: ignore[attr-defined]
    datasets.lerobot_dataset = dataset  # type: ignore[attr-defined]


def _patch_task_metadata() -> None:
    """Convert LeRobot's task DataFrame to OpenPI's expected mapping."""
    try:
        transforms = importlib.import_module("openpi.transforms")
        pandas = importlib.import_module("pandas")
    except ImportError:
        return
    task_transform = getattr(transforms, "PromptFromLeRobotTask", None)
    if task_transform is None or getattr(task_transform, _PATCHED, False):
        return
    original_init = task_transform.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        tasks = self.tasks
        if isinstance(tasks, pandas.DataFrame):
            if "task_index" in tasks.columns:
                values = tasks.index.tolist()
                keys = tasks["task_index"].astype(int).tolist()
            else:
                keys = [int(key) for key in tasks.index.tolist()]
                values = tasks.iloc[:, 0].tolist() if tasks.shape[1] else keys
            object.__setattr__(self, "tasks", dict(zip(keys, values, strict=True)))

    patched_init.__wrapped__ = original_init  # type: ignore[attr-defined]
    task_transform.__init__ = patched_init
    setattr(task_transform, _PATCHED, True)


def install_compat_shims() -> None:
    """Install the shims; safely do nothing when optional packages are absent."""
    _alias_legacy_lerobot_import()
    _patch_task_metadata()
