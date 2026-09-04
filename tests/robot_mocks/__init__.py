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

"""Fake vendor SDKs for testing production robot drivers without hardware.

Drivers import their SDKs when they connect. Installing these modules first
allows the real driver lifecycle, data conversion, and command code to run in
tests.

Use it as a context manager::

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        robot = build_robot("DualFranka", left_robot_ip="1.2.3.4", ...)
        robot.connect()

The bench check exposes the same setup through ``--mock``.
"""

from __future__ import annotations

import contextlib
import os
import pathlib
import sys
import types
from typing import Any, Iterator

from . import arms, cameras, grippers, sdks, teleop
from ._fakes import Recorder, module

__all__ = ["Recorder", "mocked_sdks", "module", "sdk_modules"]

#: Module attributes patched when a dependency was imported at module scope.
#: Keys are dotted module names.
_PATCHES: dict[str, dict[str, Any]] = {}


def _no_processes() -> Any:
    """Return a ``psutil`` proxy whose ``Popen`` starts no processes."""
    import psutil as real

    class Popen:
        def __init__(self, args, **_kwargs):
            self.args = args
            self.returncode = None

        def terminate(self):
            self.returncode = 0

        def kill(self):
            self.returncode = -9

        def wait(self, timeout=None):
            self.returncode = 0
            return 0

        def poll(self):
            return self.returncode

        def is_running(self):
            return self.returncode is None

        def status(self):
            return "running" if self.returncode is None else "terminated"

        def name(self):
            return self.args[0] if self.args else "process"

        def children(self, recursive=False):
            return []

    class _Psutil(types.ModuleType):
        """Delegate to real ``psutil`` except for explicitly replaced members."""

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    fake = _Psutil("psutil")
    # Keep the real spec: libraries ask find_spec whether psutil is installed.
    fake.__spec__ = real.__spec__
    fake.Popen = Popen
    # Force the ROS transport to use the stubbed Popen path.
    fake.process_iter = lambda *_a, **_k: iter(())
    return fake


def sdk_modules() -> dict[str, types.ModuleType]:
    """Return fake SDK modules keyed by their import names."""
    made: dict[str, types.ModuleType] = {}
    # ROS transport and the Franka ROS driver launch helper processes.
    made["psutil"] = _no_processes()
    made.update(cameras.modules())
    made.update(arms.modules())
    made.update(grippers.modules())
    made.update(sdks.modules())
    made.update(teleop.modules())
    return made


def _reach_worker_processes() -> dict[str, str]:
    """Return environment variables that install fakes in worker processes."""
    here = str(pathlib.Path(__file__).resolve().parent)
    tests = str(pathlib.Path(__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [here, tests, *(p for p in existing.split(os.pathsep) if p)]
    return {"PYTHONPATH": os.pathsep.join(parts), "RLINF_ROBOT_MOCKS": "1"}


@contextlib.contextmanager
def mocked_sdks(*, extra: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Install fake SDKs for the duration of the context.

    Only the vendor SDKs are faked. Everything else behaves as it does in a
    real run: a part with a ``node_rank`` is hosted in a scheduler worker,
    which installs the same fakes for itself. A test that wants a part in this
    process declares it without a ``node_rank``, as production does.

    Args:
        extra: More modules to install, by dotted name.

    Yields:
        Installed modules available for test assertions.
    """
    made = sdk_modules()
    if extra:
        made.update(extra)

    saved = {name: sys.modules.get(name) for name in made}
    sys.modules.update(made)

    patches: list[tuple[Any, str, Any]] = []
    saved_environ = {}

    # Workers install the fakes for themselves, so placement runs for real.
    for name, value in _reach_worker_processes().items():
        saved_environ[name] = os.environ.get(name)
        os.environ[name] = value
    # Modules that start processes rather than only talking to an SDK. A fake
    # module in sys.modules arrives too late for these, because they import
    # psutil at module scope.
    processes = _no_processes()

    # DOSW1 binds its SDK at module scope inside a try/except, so a module that
    # was already imported holds None however early the fake is installed.
    airbot = made.get("airbot_sdk.Airbot")
    airbot_config = made.get("airbot_sdk.configs.config")
    dosw1_patch = (
        {
            "_AirbotRobot": airbot.AirbotRobot,
            "_AirbotSDKConfig": airbot_config.DosW1Config,
        }
        if airbot is not None and airbot_config is not None
        else {}
    )
    for dotted, attributes in {
        "rlinf.robotics.parts.arms.franka_ros": {"psutil": processes},
        "rlinf.robotics.parts.transports.ros.ros_controller": {"psutil": processes},
        "rlinf.robotics.parts.arms.dosw1": dosw1_patch,
        **_PATCHES,
    }.items():
        target = sys.modules.get(dotted)
        if target is None:
            continue
        for name, value in attributes.items():
            patches.append((target, name, getattr(target, name, None)))
            setattr(target, name, value)

    try:
        yield made
    finally:
        for target, name, original in reversed(patches):
            if original is None:
                delattr(target, name)
            else:
                setattr(target, name, original)
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original
        for name, original in saved_environ.items():
            if original is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = original
