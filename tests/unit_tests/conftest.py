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

"""Make repository and shared test modules importable by unit tests."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import pytest

for _path in (Path(__file__).resolve().parents[2], Path(__file__).resolve().parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_session_exitstatus: Optional[int] = None


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Record the final session exit status before unconfigure hooks run."""
    global _session_exitstatus
    _session_exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config: pytest.Config) -> None:
    """Exit with pytest's status before torch_npu and Ray C++ destructors race."""
    if _session_exitstatus is None:
        return
    if "torch_npu" not in sys.modules or "ray._raylet" not in sys.modules:
        return
    ray_mod = sys.modules.get("ray")
    if (
        ray_mod is not None
        and hasattr(ray_mod, "is_initialized")
        and ray_mod.is_initialized()
    ):
        ray_mod.shutdown(_exiting_interpreter=True)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_session_exitstatus)
