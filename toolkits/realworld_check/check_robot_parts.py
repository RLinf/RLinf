#!/usr/bin/env python3
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

"""Check that a robot composes, connects, observes, and lets go.

This is the bench counterpart to the unit tests: they cover composition with
fake hardware, and this runs the same path against the real thing. It reports
what a robot is made of, which connection backs each part, where each was
placed, and what every part observes -- then disconnects and checks that
nothing still claims to be connected.

Run it after changing a part, a connection, or how a robot is composed::

    python -m toolkits.realworld_check.check_robot_parts Franka \\
        --arg robot_ip=10.0.0.1 --arg node_rank=1

    python -m toolkits.realworld_check.check_robot_parts DualFranka \\
        --arg left_robot_ip=10.0.0.1 --arg right_robot_ip=10.0.0.2

Values are parsed as Python literals when they look like one, so
``--arg node_rank=1`` passes an int and ``--arg robot_ip=10.0.0.1`` a string.

Exit code is 0 when every part connects, observes and releases; 1 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import pathlib
import sys
import traceback
from typing import Any

from rlinf.robotics.parts.base import Connection, PartGroup, RobotPart

#: The fakes and the contracts live with the tests rather than in the package:
#: they check RLinf rather than being part of it. This script is the one caller
#: that is neither, so it reaches for them here.
_TESTS = pathlib.Path(__file__).resolve().parents[2] / "tests"
if str(_TESTS) not in sys.path:
    sys.path.insert(0, str(_TESTS))

from robot_contracts import ObservationContract  # noqa: E402


def _mocked_sdks():
    """The fake vendor SDKs, which live with the tests rather than the package.

    Only the SDKs are faked, so parts are placed on nodes exactly as a real
    run places them, and each worker installs the fakes for itself.
    """
    from robot_mocks import mocked_sdks

    return mocked_sdks()


def literal(text: str) -> Any:
    """Parse a value the way a config would spell it."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def walk(part: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Every readable part of a robot, by dotted path.

    A part that carries others is checked and then walked into, because both
    are parts: an arm reads its own joints and the gripper on its bus reads
    itself, and a policy sees them both.
    """
    found: list[tuple[str, Any]] = []
    if not isinstance(part, PartGroup) and prefix:
        found.append((prefix.rstrip("."), part))
    for name, child in part.children.items():
        found += walk(child, f"{prefix}{name}.")
    return found


def describe(robot: Any) -> list[str]:
    """What the robot is made of, and what backs each part."""
    lines = []
    for path, part in walk(robot):
        backing = type(part).__name__
        host = getattr(part, "_host", None)
        if host is not None:
            backing = f"{type(part).__name__} on {type(host).__name__}"
        lines.append(f"    {path:24} {backing}")
    return lines


def _as_build_arguments(robot_type, kwargs, registry_cls):
    """Pack the arguments the way this robot's ``build`` wants them.

    Most robots take their settings as keywords. Some take a config object
    instead, and that object is the one the registry already knows about, so
    the same ``--arg name=value`` works for both.
    """
    import dataclasses
    import inspect

    registration = registry_cls.registry.get(robot_type)
    if registration is None or registration.build is None:
        return kwargs
    parameters = inspect.signature(registration.build).parameters
    config_parameter = parameters.get("config")
    # Franka spells its settings ``**config``, which is the keywords it already
    # takes. Only a robot asking for one config *object* needs packing.
    wants_object = (
        config_parameter is not None
        and config_parameter.kind is not inspect.Parameter.VAR_KEYWORD
    )
    if not wants_object or "config" in kwargs:
        return kwargs

    config_cls = registration.config_cls
    fields = {f.name for f in dataclasses.fields(config_cls)}
    build_parameters = set(parameters)
    settings = {name: value for name, value in kwargs.items() if name in fields}
    rest = {
        name: value
        for name, value in kwargs.items()
        if name not in fields and name in build_parameters
    }
    # A robot may want more of its config than the registered class declares:
    # DOSW1's hardware reads the env's config, which carries the ports and the
    # human-in-the-loop switch. Anything left over is set on the config so the
    # same --arg reaches it.
    extra = {
        name: value
        for name, value in kwargs.items()
        if name not in fields and name not in build_parameters
    }
    config = config_cls(**settings)
    for name, value in extra.items():
        setattr(config, name, value)
    packed = sorted(settings) + [f"{name}*" for name in sorted(extra)]
    print(f"      packing {packed} into {config_cls.__name__}")
    return {"config": config, **rest}


def parity_failures(robot_type: str, kwargs: dict[str, Any], placed: Any) -> list[str]:
    """A placed robot must describe itself the way a local one does.

    Only worth asking when something really was placed: a view is derived from
    the driver class, so the two agree unless a driver's contract depends on
    the machine it is running on -- reading an env var, sizing a buffer from
    the hardware it finds. That is exactly the case a single-machine test
    cannot see, and it is why this runs here, where a cluster is already up.
    """
    from dataclasses import replace

    from rlinf.robotics.discovery import build_robot

    here = build_robot(robot_type, **kwargs)
    for owner in here.owners():
        if owner._remote_info is not None:
            owner._remote_info = replace(owner._remote_info, node_rank=None)
    here.connect()
    try:
        local = {path: part.observation_features for path, part in walk(here)}
    finally:
        here.disconnect()

    hosted = {path: part.observation_features for path, part in walk(placed)}
    found = []
    for path in sorted(set(local) | set(hosted)):
        if local.get(path) != hosted.get(path):
            found.append(
                f"{path} describes {sorted(hosted.get(path) or {})} hosted but "
                f"{sorted(local.get(path) or {})} here"
            )
    return found


def check(robot_type: str, kwargs: dict[str, Any], remote: bool = False) -> int:
    print(f"[1/5] composing {robot_type} with {kwargs}")
    # Importing the robots is what registers them.
    import rlinf.robotics.robots  # noqa: F401
    from rlinf.robotics.discovery import RobotDiscovery, build_robot

    kwargs = _as_build_arguments(robot_type, kwargs, RobotDiscovery)
    robot = build_robot(robot_type, **kwargs)

    print("[2/5] parts, before connecting")
    for line in describe(robot):
        print(line)

    print("[3/5] connecting")
    robot.connect()
    if not robot.is_connected:
        print("    FAIL: the robot does not report itself connected")
        return 1
    for owner in robot.owners():
        where = "here" if owner.node_rank is None else f"node {owner.node_rank}"
        print(f"    {type(owner).__name__:24} open {where}")

    print("[4/5] observing every part")
    failures = []
    for path, part in walk(robot):
        if isinstance(part, Connection) and not isinstance(part, RobotPart):
            failures.append(f"{path} is a Connection and should not be in the tree")
            continue
        try:
            observation = part.get_observation()
        except Exception as error:  # noqa: BLE001 - a bench check reports anything
            failures.append(f"{path}: {type(error).__name__}: {error}")
            continue
        shapes = {
            key: getattr(value, "shape", type(value).__name__)
            for key, value in observation.items()
        }
        # The same comparison the conformance suite makes, so a bench run and
        # a contributor's test agree about what a part promised.
        mismatches = ObservationContract(part, path).failures()
        failures += mismatches
        note = "  " + "; ".join(mismatches) if mismatches else ""
        print(f"    {path:24} {shapes}{note}")

    if remote:
        print("[4b/5] comparing a placed robot with a local one")
        mismatches = parity_failures(robot_type, kwargs, robot)
        failures += mismatches
        for line in mismatches:
            print(f"    {line}")
        if not mismatches:
            print("    every part describes the same either side of the boundary")

    print("[5/5] disconnecting")
    robot.disconnect()
    still = [path for path, part in walk(robot) if getattr(part, "is_connected", False)]
    if still:
        failures.append(f"still connected after disconnect: {still}")
    if robot.is_connected:
        failures.append("the robot still reports itself connected")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("\nOK: every part connected, observed what it declares, and let go.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("robot_type", help="a registered robot type, e.g. Franka")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="run against faked vendor SDKs instead of hardware, so the same "
        "command works on a laptop and on the bench",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="also compare a placed robot with a local one, part by part",
    )
    parser.add_argument(
        "--arg",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="an argument for the robot's build(); repeatable",
    )
    args = parser.parse_args()

    kwargs = {}
    for item in args.arg:
        if "=" not in item:
            parser.error(f"--arg wants NAME=VALUE, got {item!r}")
        name, _, value = item.partition("=")
        kwargs[name] = literal(value)

    try:
        if args.mock:
            print("[mock] vendor SDKs are faked; this checks the code, not a robot")
            with _mocked_sdks():
                return check(args.robot_type, kwargs, remote=args.remote)
        return check(args.robot_type, kwargs, remote=args.remote)
    except Exception:  # noqa: BLE001 - a bench check reports anything
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
