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

"""Vendor SDKs, faked, so a real part can be run without its hardware.

Every part imports its SDK inside ``_open`` or ``connect`` rather than at module
scope. Putting a fake in :data:`sys.modules` first is therefore enough for the
part's own code -- its lifecycle, its state conversion, its action dispatch --
to run unchanged against something that answers. Nothing about the part is
stubbed; only what is on the other end of the cable.

That is the difference from a test double: a double proves the code around the
part is right, and this proves the part is.

Use it as a context manager::

    from robot_mocks import mocked_sdks

    with mocked_sdks():
        robot = build_robot("DualFranka", left_robot_ip="1.2.3.4", ...)
        robot.connect()

The bench check takes ``--mock`` and does exactly this, so the same command
that verifies a real robot also runs in CI.
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

#: Attributes to swap on modules that import their dependency at module scope,
#: where a :data:`sys.modules` entry arrives too late. Keyed by dotted module.
_PATCHES: dict[str, dict[str, Any]] = {}


def _no_processes() -> Any:
    """A ``psutil`` that starts no processes but is otherwise the real one.

    The Franka ROS arm launches its impedance controller and the ROS transport
    launches roscore, both through ``psutil.Popen``. Only that has to be
    stubbed: the scheduler uses the rest of psutil, so everything else is
    delegated rather than faked.
    """
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
        """The real psutil, minus the ability to start anything."""

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

    fake = _Psutil("psutil")
    # Keep the real spec: libraries ask find_spec whether psutil is installed.
    fake.__spec__ = real.__spec__
    fake.Popen = Popen
    # Nothing of ours is running, so the roscore search finds nothing and the
    # transport starts its own -- which is the stub above.
    fake.process_iter = lambda *_a, **_k: iter(())
    return fake


def sdk_modules() -> dict[str, types.ModuleType]:
    """Every faked SDK, by the name a part imports it as."""
    made: dict[str, types.ModuleType] = {}
    # Parts that launch a process rather than only talking to an SDK: the ROS
    # transport starts roscore, and the Franka arm its impedance controller.
    made["psutil"] = _no_processes()
    made.update(cameras.modules())
    made.update(arms.modules())
    made.update(grippers.modules())
    made.update(sdks.modules())
    made.update(teleop.modules())
    return made


def _reach_worker_processes() -> dict[str, str]:
    """Environment that makes a scheduler worker install the fakes too.

    A worker is a separate process, and a fake in this one's ``sys.modules``
    never reaches it. Putting this package on ``PYTHONPATH`` lets the worker
    import ``sitecustomize`` from it at interpreter start, and the flag is what
    tells that hook to act.
    """
    here = str(pathlib.Path(__file__).resolve().parent)
    tests = str(pathlib.Path(__file__).resolve().parent.parent)
    existing = os.environ.get("PYTHONPATH", "")
    parts = [here, tests, *(p for p in existing.split(os.pathsep) if p)]
    return {"PYTHONPATH": os.pathsep.join(parts), "RLINF_ROBOT_MOCKS": "1"}


@contextlib.contextmanager
def mocked_sdks(
    *, extra: dict[str, Any] | None = None, remote: bool = False
) -> Iterator[dict[str, Any]]:
    """Install the fake SDKs for the duration of the block.

    Args:
        extra: More modules to install, by dotted name.
        remote: Place parts on nodes, as a real run does, with each worker
            installing the fakes for itself. Left False, every part is built
            in this process, which is faster and needs no cluster.

    Yields:
        The installed modules, so a test can assert on what a part did to them.
    """
    made = sdk_modules()
    if extra:
        made.update(extra)

    saved = {name: sys.modules.get(name) for name in made}
    sys.modules.update(made)

    patches: list[tuple[Any, str, Any]] = []
    saved_environ = {}

    if remote:
        # Workers install the fakes for themselves, so placement runs for real.
        for name, value in _reach_worker_processes().items():
            saved_environ[name] = os.environ.get(name)
            os.environ[name] = value
    else:
        # A faked SDK lives in this process, and a part placed on a node is
        # built in another one that never saw it. Build every part here: what
        # this checks is the code, not the cluster.
        from rlinf.robotics.placement import specs as _specs

        def _place_here(self):
            return self.part_cls.spawn(
                *self.args, node_rank=None, name=self.name, **self.kwargs
            )

        patches.append((_specs.PartSpec, "place", _specs.PartSpec.place))
        _specs.PartSpec.place = _place_here
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
