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

"""Real-world environments: task envs, teleop devices, config, and layout."""

from __future__ import annotations

import importlib
import re
import sys
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.real.dosw1.base import DOSW1Config, DOSW1Env
from rlinf.envs.real.franka.base import FrankaEnv
from rlinf.envs.real.franka.dual_franka_joint import (
    DualFrankaJointEnv,
)
from rlinf.envs.real.gim_arm.base import GimArmEnv, GimArmRobotConfig
from rlinf.envs.real.task_env import RobotTask, RobotTaskEnv
from rlinf.envs.real.wrappers.teleop.config import (  # noqa: E402
    NO_DEVICE,
    resolve_teleop_device,
    resolve_teleop_devices,
)
from rlinf.envs.real.wrappers.teleop.intervention import (  # noqa: E402
    TeleopDevice,
    TeleopIntervention,
    TeleopSample,
)
from rlinf.envs.real.xsquare.base import Turtle2Env, Turtle2RobotConfig
from rlinf.robotics import ControllablePart, Group, Robot

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# --- from test_robot_task_env.py --------------------------------------


class DummyDriver(ControllablePart):
    def __init__(self) -> None:
        self.connected = False
        self.last_action: dict[str, Any] | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def observation_features(self) -> dict[str, Any]:
        return {"position": {"shape": (1,)}}

    @property
    def action_features(self) -> dict[str, Any]:
        return {"target": {"shape": (1,)}}

    def connect(self) -> None:
        self.connected = True

    def reset(self) -> None:
        self.last_action = None

    def get_observation(self) -> dict[str, Any]:
        return {"position": np.zeros(1)}

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        self.last_action = action
        return action

    def disconnect(self) -> None:
        self.connected = False


class DummyTask(RobotTask):
    @property
    def description(self) -> str:
        return "Move the test arm."

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {"position": gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)}
        )

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {
                "arms": gym.spaces.Dict(
                    {
                        "arm": gym.spaces.Dict(
                            {
                                "arm": gym.spaces.Dict(
                                    {
                                        "target": gym.spaces.Box(
                                            -1.0,
                                            1.0,
                                            shape=(1,),
                                            dtype=np.float32,
                                        )
                                    }
                                )
                            }
                        )
                    }
                )
            }
        )

    def reset(
        self,
        robot: Robot,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del seed, options
        robot.reset()
        return {"position": np.zeros(1, dtype=np.float32)}, {}

    def step(
        self,
        robot: Robot,
        action: dict[str, Any],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        robot.send_action(action)
        return {"position": np.ones(1, dtype=np.float32)}, 1.0, True, False, {}


def test_robot_task_env_composes_task_and_robot_lifecycles():
    driver = DummyDriver()
    robot = Robot(arm=Group(arm=driver))
    env = RobotTaskEnv(robot, DummyTask())
    action = {"arm": {"arm": {"target": np.array([0.5])}}}

    observation, _ = env.reset(seed=3)
    transition = env.step(action)

    assert env.task_description == "Move the test arm."
    assert observation["position"].tolist() == [0.0]
    assert transition[0]["position"].tolist() == [1.0]
    assert driver.last_action is not None
    assert driver.last_action["target"].tolist() == [0.5]
    env.close()
    assert not driver.is_connected


# --- from test_realworld_robotics_compatibility.py --------------------


def _assert_legacy_transition(env) -> None:
    observation, _ = env.reset()
    transition = env.step(env.action_space.sample())

    assert set(observation) == {"state", "frames"}
    assert set(transition[0]) == {"state", "frames"}
    assert len(transition) == 5
    env.close()


def test_franka_dummy_preserves_legacy_policy_schema():
    env = FrankaEnv(
        override_cfg={
            "is_dummy": True,
            "camera_serials": ["dummy"],
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dual_franka_dummy_preserves_legacy_policy_schema():
    env = DualFrankaJointEnv(
        override_cfg={
            "is_dummy": True,
            "base_camera_serials": ["dummy"],
            "enable_camera_player": False,
            "step_frequency": 10000.0,
        },
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (16,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_gim_arm_dummy_preserves_legacy_policy_schema():
    env = GimArmEnv(
        config=GimArmRobotConfig(
            is_dummy=True,
            camera_serials=[],
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_dosw1_dummy_preserves_legacy_policy_schema():
    env = DOSW1Env(
        config=DOSW1Config(
            is_dummy=True,
            camera_serials=[],
            camera_names=[],
            enable_camera_player=False,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (14,)
    assert env.robot is None
    _assert_legacy_transition(env)


def test_turtle2_dummy_preserves_legacy_policy_schema():
    env = Turtle2Env(
        config=Turtle2RobotConfig(
            is_dummy=True,
            step_frequency=10000.0,
        ),
        worker_info=None,
        robot_info=None,
        env_idx=0,
    )

    assert env.action_space.shape == (7,)
    assert env.robot is None
    _assert_legacy_transition(env)


# --- from test_teleop_intervention.py ---------------------------------


class FakeEnv:
    """Records the actions it is stepped with."""

    def __init__(self) -> None:
        self.stepped: list[np.ndarray] = []
        self.reset_calls = 0
        self.closed = False

    def step(self, action):
        self.stepped.append(np.asarray(action))
        return {"obs": 1}, 0.0, False, False, {}

    def reset(self, **kwargs):
        self.reset_calls += 1
        return {"obs": 1}, {}

    def close(self):
        self.closed = True


class ScriptedDevice(TeleopDevice):
    """Replays a list of samples, one per read."""

    def __init__(self, samples: list[TeleopSample]) -> None:
        self.samples = list(samples)
        self.reads = 0
        self.resets = 0
        self.closed = False
        self.before_steps = 0
        self.fallback_action: Optional[np.ndarray] = None

    def read(self, env: Any, policy_action: np.ndarray) -> TeleopSample:
        sample = self.samples[min(self.reads, len(self.samples) - 1)]
        self.reads += 1
        return sample

    def reset(self, env: Any) -> None:
        self.resets += 1

    def before_step(self, env: Any) -> None:
        self.before_steps += 1

    def fallback(self, env: Any, policy_action: np.ndarray) -> np.ndarray:
        if self.fallback_action is not None:
            return self.fallback_action
        return policy_action

    def close(self) -> None:
        self.closed = True


POLICY = np.array([0.0, 0.0, 0.0])


EXPERT = np.array([1.0, 1.0, 1.0])


def test_active_sample_replaces_the_policy_action():
    """While the operator drives, their action is the one applied."""
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=EXPERT, active=True)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], EXPERT)
    assert np.array_equal(info["intervene_action"], EXPERT)


def test_inactive_device_leaves_the_policy_action_alone():
    """A device with nothing to say never touches the action or the info."""
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env, ScriptedDevice([TeleopSample(action=None, active=False)])
    )

    _, _, _, _, info = wrapper.step(POLICY)

    assert np.array_equal(env.stepped[0], POLICY)
    assert "intervene_action" not in info


def test_control_is_held_between_samples_then_released():
    """The hold window keeps the operator in control across quiet samples.

    A person moves far slower than the device samples, so without the window
    the applied action would flicker between operator and policy mid-motion.
    """
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)  # operator moves
    wrapper.step(POLICY)  # quiet, but still inside the window
    assert np.array_equal(env.stepped[1], EXPERT)

    device.timeout = 0.0  # window elapses
    wrapper.step(POLICY)
    assert np.array_equal(env.stepped[2], POLICY)


def test_an_unfilled_part_keeps_the_policy_action():
    """Composition replaced the per-device fallback hook.

    A group starts from the action the policy asked for and overwrites only the
    parts its devices fill, so anything unfilled is the policy's, and a device
    holding state it commands keeps it by filling that part every step.
    """
    import numpy as np

    from rlinf.envs.real.wrappers.teleop.composed import ComposedTeleop
    from rlinf.robotics.teleop import (
        ActionKind,
        TeleopBinding,
        TeleopEntry,
        TeleopGroup,
    )

    class Fixed(TeleopBinding):
        PRODUCES = {"hand": ActionKind.HAND}

        def action(self, reading, context):
            from rlinf.robotics.teleop import TeleopAction

            return TeleopAction(parts={"hand": np.full(6, 0.5)}, driving=True)

        def reset(self):
            pass

    class Device:
        is_connected = True

        def get_observation(self):
            return {}

        def connect(self):
            pass

        def disconnect(self):
            pass

    layout = {"arm": slice(0, 6), "hand": slice(6, 12)}
    group = TeleopGroup(
        [TeleopEntry(Device(), Fixed())],
        available={"arm": ActionKind.CARTESIAN_DELTA, "hand": ActionKind.HAND},
    )
    device = ComposedTeleop(group, layout)

    policy = np.arange(12, dtype=np.float64)
    sample = device.read(_FakeLayoutEnv(), policy)

    assert np.allclose(sample.action[:6], policy[:6])  # untouched by any device
    assert np.allclose(sample.action[6:], 0.5)  # filled by the glove


class _FakeLayoutEnv:
    """Just enough env for context gathering, which finds nothing here."""

    unwrapped = None

    def get_wrapper_attr(self, name):
        raise AttributeError(name)


def test_mark_flag_is_opt_in():
    """Only formats that key on the flag pay for it."""
    sample = TeleopSample(action=EXPERT, active=True)

    plain = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]))
    flagged = TeleopIntervention(FakeEnv(), ScriptedDevice([sample]), mark_flag=True)

    assert "intervene_flag" not in plain.step(POLICY)[4]
    assert flagged.step(POLICY)[4]["intervene_flag"] == np.ones(1)


def test_device_info_reaches_the_step_info():
    """Device state a collector records rides along with the transition."""
    env = FakeEnv()
    wrapper = TeleopIntervention(
        env,
        ScriptedDevice([TeleopSample(action=EXPERT, active=True, info={"left": True})]),
    )

    assert wrapper.step(POLICY)[4]["left"] is True


def test_reset_resyncs_the_device_and_drops_the_hold():
    """A new episode must not inherit the previous one's intervention."""
    env = FakeEnv()
    device = ScriptedDevice(
        [
            TeleopSample(action=EXPERT, active=True),
            TeleopSample(action=EXPERT, active=False),
        ]
    )
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)
    assert wrapper.intervening

    wrapper.reset()

    assert device.resets == 1
    assert env.reset_calls == 1
    assert not wrapper.intervening


def test_close_releases_the_device_before_the_env():
    """Devices hold serial ports; they are released with the wrapper."""
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.close()

    assert device.closed
    assert env.closed


def test_before_step_runs_ahead_of_the_env():
    """Devices that stream need a hook before the env advances."""
    env = FakeEnv()
    device = ScriptedDevice([TeleopSample(action=None, active=False)])
    wrapper = TeleopIntervention(env, device)

    wrapper.step(POLICY)

    assert device.before_steps == 1


def test_read_is_abstract():
    """A device is only required to say what the operator wants."""
    with pytest.raises(TypeError):
        TeleopDevice()


# --- from test_teleop_config.py ---------------------------------------

SINGLE_ARM = ("spacemouse", "gello", "pico")


def test_retired_single_key_still_selects_its_device_and_warns():
    with pytest.warns(DeprecationWarning, match="'teleop_device' is retired"):
        device = resolve_teleop_device({"teleop_device": "gello"}, supported=SINGLE_ARM)

    assert device == "gello"


def test_missing_config_falls_back_to_the_default():
    assert resolve_teleop_device({}, supported=SINGLE_ARM) == NO_DEVICE
    assert (
        resolve_teleop_device({}, supported=SINGLE_ARM, default="spacemouse")
        == "spacemouse"
    )


def test_retired_boolean_still_selects_its_device_and_warns():
    """Configs written before the rename keep working for a release."""
    with pytest.warns(DeprecationWarning, match="use_pico"):
        device = resolve_teleop_device({"use_pico": True}, supported=SINGLE_ARM)

    assert device == "pico"


def test_all_retired_booleans_off_means_no_device():
    """Explicitly disabling every device is not the same as saying nothing."""
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device(
            {"use_spacemouse": False, "use_gello": False},
            supported=SINGLE_ARM,
            default="spacemouse",
        )

    assert device == NO_DEVICE


def test_two_retired_booleans_on_is_an_error():
    with pytest.raises(ValueError, match="Only one teleop device"):
        resolve_teleop_device(
            {"use_spacemouse": True, "use_pico": True}, supported=SINGLE_ARM
        )


def test_disagreeing_old_and_new_keys_are_refused():
    """Env configs layer, so a half-migrated stack can hold both keys.

    Silently preferring either one hands somebody the wrong device with a robot
    already moving, so this is an error rather than a precedence rule.
    """
    with pytest.raises(ValueError, match="cannot be reconciled"):
        resolve_teleop_device(
            {"teleop_device": "pico", "use_spacemouse": True},
            supported=SINGLE_ARM,
        )


def test_agreeing_old_and_new_keys_only_warn():
    with pytest.warns(DeprecationWarning, match="redundant"):
        device = resolve_teleop_device(
            {"teleop_device": "pico", "use_pico": True}, supported=SINGLE_ARM
        )

    assert device == "pico"


def test_device_the_env_cannot_drive_is_refused():
    """A dual-arm Franka has no single-arm Cartesian teleop path."""
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_device(
            {"teleop_device": "spacemouse"}, supported=("gello_joint", "pico")
        )


def test_none_is_always_allowed():
    with pytest.warns(DeprecationWarning):
        device = resolve_teleop_device({"teleop_device": "none"}, supported=("pico",))

    assert device == NO_DEVICE


def test_shipped_configs_use_the_new_key():
    """Every retired spelling is gone from every config in the repo."""
    roots = [_ROOT / "examples", _ROOT / "evaluations", _ROOT / "tests"]
    offenders = []
    for root in roots:
        for path in root.rglob("*.yaml"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if any(
                    stripped.startswith(f"{flag}:")
                    for flag in (
                        "teleop_device",
                        "use_spacemouse",
                        "use_gello",
                        "use_gello_joint",
                        "use_pico",
                    )
                ):
                    offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


# --- from test_real_env_layout.py -------------------------------------

_REAL = _ROOT / "rlinf" / "envs" / "real"


_ROBOTS = ("franka", "dosw1", "gim_arm", "xsquare")


EXPECTED_IDS = {
    "FrankaEnv-v1",
    "PegInsertionEnv-v1",
    "FrankaBinRelocationEnv-v1",
    "BottleEnv-v1",
    "DexpnpEnv-v1",
    "DualFrankaJointEnv-v1",
    "DualFrankaTCPEnv-v1",
    "DOSW1PickEnv-v1",
    "ButtonEnv-v1",
    "GimArmPegInsertionEnv-v1",
}


def test_no_robot_keeps_a_tasks_subpackage():
    """Tasks sit at the top of their robot folder, not one level down."""
    leftovers = [name for name in _ROBOTS if (_REAL / name / "tasks").exists()]

    assert leftovers == []


def test_every_robot_folder_has_a_base():
    """Shared machinery is named so nobody mistakes it for a task."""
    missing = [name for name in _ROBOTS if not (_REAL / name / "base.py").exists()]

    assert missing == []


def test_all_task_ids_are_registered():
    """Reaching for any name registers every task.

    ``rlinf.envs.real`` loads lazily so that importing one teleop device does
    not drag in the env stack, which means registration happens on first use
    rather than on import. ``get_env_cls`` reaches it exactly this way.
    """
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    assert EXPECTED_IDS <= set(registry)


def test_every_entry_point_resolves():
    """A registered id whose entry point cannot be imported fails at rollout."""
    from gymnasium.envs.registration import registry

    from rlinf.envs.real import RealWorldEnv

    assert RealWorldEnv is not None
    unresolved = []
    for env_id in sorted(EXPECTED_IDS):
        entry_point = registry[env_id].entry_point
        module_name, _, attribute = str(entry_point).partition(":")
        module = importlib.import_module(module_name)
        if getattr(module, attribute, None) is None:
            unresolved.append(f"{env_id} -> {entry_point}")

    assert unresolved == []


def test_task_tables_cover_the_wrapped_robots():
    """The TASKS table is the one place a robot's tasks are declared."""
    from rlinf.envs.real import dosw1, franka, xsquare

    declared = set(franka.TASKS) | set(dosw1.TASKS) | set(xsquare.TASKS)

    # GimArm registers its env class directly, with no wrapper stack.
    assert declared == EXPECTED_IDS - {"GimArmPegInsertionEnv-v1"}


def test_pose_math_is_not_filed_under_a_robot():
    """construct_adjoint_matrix is SE(3) math the wrappers share.

    Leaving it in the Franka package made every wrapper importing it pull that
    package in, which is what turned the task registry into an import cycle.
    """
    from rlinf.envs.real.utils import pose

    assert hasattr(pose, "construct_adjoint_matrix")
    assert not (_REAL / "franka" / "utils.py").exists()


def test_task_configs_state_only_their_compliance_deltas():
    """A task's config should show its tuning, not eighteen repeated gains.

    Every Franka task used to carry a full copy of the impedance gains, so the
    handful of numbers that actually differed were invisible.
    """
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.bin_relocation import BinEnvConfig
    from rlinf.envs.real.franka.bottle import BottleConfig
    from rlinf.envs.real.franka.dex_pnp import DexpnpConfig
    from rlinf.envs.real.franka.peg_insertion import PegInsertionConfig

    deltas = {
        cls.__name__: {
            key
            for key, value in cls().compliance_param.items()
            if COMPLIANCE_DEFAULTS[key] != value
        }
        for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig)
    }

    # Every task keeps the full set of gains; only some are task-specific.
    for cls in (PegInsertionConfig, BottleConfig, BinEnvConfig, DexpnpConfig):
        assert set(cls().compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert {name: len(keys) for name, keys in deltas.items()} == {
        "PegInsertionConfig": 1,
        "BottleConfig": 8,
        "BinEnvConfig": 11,
        "DexpnpConfig": 6,
    }


def test_unknown_compliance_gain_is_refused():
    """A misspelled gain reaches the impedance controller and is ignored there."""
    import pytest

    from rlinf.envs.real.franka.base import compliance

    with pytest.raises(KeyError, match="Unknown compliance gains"):
        compliance(translational_stifness=1000)


# --- wrapper families ------------------------------------------------------


def test_wrappers_are_split_by_what_they_change():
    """Three families, each with an obvious home for a new wrapper.

    teleop replaces the action, transforms rewrite how it is expressed, and
    episode decides when a rollout starts, ends, and what it scored. The old
    flat wrappers/ package mixed all three.
    """
    real = _ROOT / "rlinf" / "envs" / "real"
    wrappers = real / "wrappers"

    assert wrappers.is_dir(), "the three families live under one parent"
    for family in ("teleop", "transforms", "episode"):
        assert (wrappers / family / "__init__.py").exists(), family

    # The top level is robot folders and the env machinery, nothing else.
    loose = sorted(
        path.stem for path in real.glob("*.py") if path.name != "__init__.py"
    )
    assert loose == ["env", "registry", "task_env", "venv"], loose


def test_no_teleop_wrapper_is_left_outside_teleop():
    """Every intervention lives with the devices it reads."""
    real = _ROOT / "rlinf" / "envs" / "real"
    strays = sorted(
        path.name
        for family in ("transforms", "episode")
        for path in (real / "wrappers" / family).glob("*.py")
        if "intervention" in path.name and "leader_follower" not in path.name
    )

    assert strays == []


def test_a_held_button_device_does_not_keep_control_after_release():
    """PICO says exactly when it is driving, so it sets no hold window.

    The window exists for devices that sample faster than a person moves;
    applying it here would keep commanding the robot after the grip is released.
    """
    from rlinf.robotics.teleop import PicoBinding, PicoTcpBinding

    assert PicoBinding.HOLD_WINDOW == 0.0
    assert PicoTcpBinding.HOLD_WINDOW == 0.0


def test_streaming_device_lifecycle_without_hardware():
    """The command thread starts once aligned and is joined on close."""
    from rlinf.envs.real.wrappers.teleop.intervention import TeleopSample
    from rlinf.envs.real.wrappers.teleop.streaming import TeleopStreamer

    ticks = []

    class Fake(TeleopStreamer):
        def read(self, env, policy_action):
            return TeleopSample(action=None, active=False)

        def stream_once(self, env):
            ticks.append(1)

    device = Fake(period=0.001, enabled=True)
    device.before_reset(None, {})
    device.reset(None)
    device.after_reset(None)
    deadline = time.monotonic() + 1.0
    while not ticks and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ticks, "stream thread never ran"
    assert device.streaming

    device.close()

    assert not device.streaming


# --- keyboard sessions -----------------------------------------------------


def _keyboard_session(monkeypatch, queued):
    """A KeyboardSession over a fake listener replaying `queued` key batches."""
    from rlinf.envs.real.wrappers.episode import session as session_module

    class FakeListener:
        def __init__(self):
            self.batches = list(queued)

        def pop_pressed_keys(self):
            return self.batches.pop(0) if self.batches else []

        def get_key(self):
            batch = self.pop_pressed_keys()
            return batch[0] if batch else None

    monkeypatch.setattr(session_module, "KeyboardListener", FakeListener)

    class Env:
        def __init__(self):
            self.resets = 0

        def reset(self, seed=None, options=None):
            self.resets += 1
            return {}, {}

        def step(self, action):
            return {}, 0.0, False, False, {}

    return session_module.KeyboardSession(Env())


def test_repeat_presses_within_the_debounce_window_are_dropped(monkeypatch):
    """Foot pedals bounce, and a USB key-down burst arrives as several presses."""
    session = _keyboard_session(monkeypatch, [["a"], ["a"], ["b"]])

    assert list(session.presses()) == ["a"]
    assert list(session.presses()) == []  # same key, still inside the window
    assert list(session.presses()) == ["b"]  # different key passes


def test_presses_queued_between_episodes_do_not_leak(monkeypatch):
    """A pedal tapped while the arm homes must not start the next episode."""
    session = _keyboard_session(monkeypatch, [["c"], ["a"]])

    session.reset()

    assert session.env.resets == 1
    assert list(session.presses()) == ["a"]  # the queued "c" was dropped


def test_every_keyboard_wrapper_shares_the_session(monkeypatch):
    """One place owns the listener, the debounce, and the drain."""
    from rlinf.envs.real.wrappers.episode import (
        KeyboardEvalControlWrapper,
        KeyboardRewardDoneMultiStageWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
    )
    from rlinf.envs.real.wrappers.episode.session import KeyboardSession

    for wrapper in (
        KeyboardEvalControlWrapper,
        KeyboardRLTPolicySwitchWrapper,
        KeyboardStartEndWrapper,
        KeyboardRewardDoneWrapper,
        KeyboardRewardDoneMultiStageWrapper,
    ):
        assert issubclass(wrapper, KeyboardSession), wrapper.__name__


def test_episode_wrappers_report_through_the_logger():
    """These run beside a robot; a stray print is not where an operator looks."""
    episode_dir = _ROOT / "rlinf" / "envs" / "real" / "episode"
    offenders = sorted(
        path.name
        for path in episode_dir.glob("*.py")
        if re.search(r"^\s*print\(", path.read_text(), re.M)
    )

    assert offenders == []


def test_euler_conversion_is_one_wrapper_for_any_arm_count():
    """One arm and two differ only in how many poses tcp_pose carries."""
    import numpy as np
    from gymnasium import spaces

    from rlinf.envs.real.wrappers.transforms import (
        DualQuat2EulerWrapper,
        Quat2EulerWrapper,
    )

    class Env:
        def __init__(self, dim):
            self.observation_space = spaces.Dict(
                {
                    "state": spaces.Dict(
                        {"tcp_pose": spaces.Box(-np.inf, np.inf, (dim,))}
                    )
                }
            )

    identity_quat = np.array([0.0, 0.0, 0.0, 1.0])
    one = np.concatenate([np.array([1.0, 2.0, 3.0]), identity_quat])

    single = Quat2EulerWrapper(Env(7))
    dual = DualQuat2EulerWrapper(Env(14))

    assert single.observation_space["state"]["tcp_pose"].shape == (6,)
    assert dual.observation_space["state"]["tcp_pose"].shape == (12,)

    got = single.observation({"state": {"tcp_pose": one.copy()}})["state"]["tcp_pose"]
    assert np.allclose(got, [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])

    both = dual.observation({"state": {"tcp_pose": np.concatenate([one, one])}})
    assert np.allclose(both["state"]["tcp_pose"], [1.0, 2.0, 3.0, 0, 0, 0] * 2)


# --- the stack an env actually gets, built by the real builders --------------
#
# The tests above drive scripted devices and fake envs. These run the dummy
# envs through build_stack, so the wrapper stack, the transforms,
# and the task configs are the ones a rollout would see.


def _dummy_franka(env_cls=None, **overrides):
    from rlinf.envs.real.franka.base import FrankaEnv

    cfg = {
        "is_dummy": True,
        "camera_serials": ["dummy"],
        "enable_camera_player": False,
        "step_frequency": 10000.0,
    }
    cfg.update(overrides)
    return (env_cls or FrankaEnv)(
        override_cfg=cfg, worker_info=None, robot_info=None, env_idx=0
    )


def _chain(env):
    """Wrapper class names from the outside in."""
    names = []
    while hasattr(env, "env"):
        names.append(type(env).__name__)
        env = env.env
    return names


def test_wrapper_stack_converts_the_pose_it_hands_the_policy():
    """The quaternion the arm reports is not what the policy receives.

    A policy trained on Euler angles needs the conversion at rollout, so the
    stack is what makes the observation match the training data.
    """
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    raw, _ = env.reset()
    assert raw["state"]["tcp_pose"].shape == (7,)

    wrapped = build_stack(
        env, {"teleop": "none", "no_gripper": False, "use_relative_frame": True}
    )
    observation, _ = wrapped.reset()

    assert _chain(wrapped) == ["Quat2EulerWrapper", "RelativeFrame"]
    assert observation["state"]["tcp_pose"].shape == (6,)
    wrapped.close()


def test_no_teleop_device_leaves_no_intervention_in_the_stack():
    """An autonomous rollout must not carry a wrapper waiting on hardware."""
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )

    assert not any("Intervention" in name for name in _chain(wrapped))
    wrapped.close()


def test_no_gripper_narrows_the_action_the_policy_must_produce():
    """``no_gripper`` drops the gripper channel rather than ignoring it."""
    from rlinf.envs.real.wrappers import build_stack

    env = _dummy_franka()
    assert env.action_space.shape == (7,)

    wrapped = build_stack(
        env,
        {"teleop": "none", "no_gripper": True, "use_relative_frame": False},
    )

    assert "GripperCloseEnv" in _chain(wrapped)
    assert wrapped.action_space.shape == (6,)
    wrapped.close()


def test_a_task_env_runs_with_its_own_config():
    """A task is its config plus a hook or two, and it steps like any env."""
    from rlinf.envs.real.franka.base import COMPLIANCE_DEFAULTS
    from rlinf.envs.real.franka.peg_insertion import PegInsertionEnv

    env = _dummy_franka(
        PegInsertionEnv, target_ee_pose=[0.5, 0.0, 0.1, -3.14, 0.0, 0.0]
    )

    assert env.config.task_description == "peg and insertion"
    # The task states one gain; the rest come from the shared defaults.
    assert set(env.config.compliance_param) == set(COMPLIANCE_DEFAULTS)
    assert env.config.compliance_param["translational_stiffness"] == 2000

    env.reset()
    observation, reward, terminated, truncated, info = env.step(
        env.action_space.sample()
    )

    assert set(observation) == {"state", "frames"}
    assert isinstance(bool(terminated), bool)
    env.close()


def test_every_registered_task_builds_through_its_entry_point():
    """A registered id whose factory raises fails only at rollout time."""
    import gymnasium as gym

    from rlinf.envs.real import RealWorldEnv  # noqa: F401  (registers the tasks)

    cfg = {
        "teleop": "none",
        "no_gripper": False,
        "use_relative_frame": False,
    }
    built = []
    for env_id in ("FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"):
        env = gym.make(
            env_id,
            override_cfg={
                "is_dummy": True,
                "camera_serials": ["dummy"],
                "enable_camera_player": False,
                "step_frequency": 10000.0,
            },
            worker_info=None,
            robot_info=None,
            env_idx=0,
            env_cfg=cfg,
        )
        env.reset()
        env.close()
        built.append(env_id)

    assert built == ["FrankaEnv-v1", "PegInsertionEnv-v1", "BottleEnv-v1"]


def test_converted_pose_stays_inside_the_observation_space():
    """The wrapper's output dtype has to match the space it declares.

    SciPy returns float64 from ``as_euler`` while the envs declare float32
    poses, so the converted observation used to fall outside its own space.
    Anything allocating buffers from the space, or checking ``contains``, sees
    that.
    """
    from rlinf.envs.real.wrappers import build_stack

    wrapped = build_stack(
        _dummy_franka(),
        {"teleop": "none", "no_gripper": False, "use_relative_frame": False},
    )
    observation, _ = wrapped.reset()

    assert (
        observation["state"]["tcp_pose"].dtype
        == wrapped.observation_space["state"]["tcp_pose"].dtype
    )
    assert wrapped.observation_space.contains(observation)
    wrapped.close()


# --- composing several teleop devices ---------------------------------


class _FakeInner:
    """Just the env attributes the teleop builders read."""

    def __init__(self, **config: Any) -> None:
        self.config = SimpleNamespace(**config)


def test_one_named_device_resolves_to_one_entry():
    assert resolve_teleop_devices({"teleop": "gello"}, supported=SINGLE_ARM) == [
        "gello"
    ]


def test_saying_nothing_resolves_to_no_entries():
    assert resolve_teleop_devices({}, supported=SINGLE_ARM) == []
    assert resolve_teleop_devices({"teleop": "none"}, supported=SINGLE_ARM) == []


def test_a_list_keeps_every_device_it_names():
    assert resolve_teleop_devices(
        {"teleop": ["spacemouse", "glove"]}, supported=("spacemouse", "glove")
    ) == ["spacemouse", "glove"]


def test_an_entry_carries_its_own_options():
    entries = resolve_teleop_devices(
        {"teleop": [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]},
        supported=("gello_joint",),
    )

    assert entries == [{"gello_joint": {"port": "/dev/left", "drives": "left"}}]


def test_one_device_may_appear_twice_on_different_branches():
    """Two leader arms are the same device driving different halves."""
    entries = resolve_teleop_devices(
        {
            "teleop": [
                {"gello_joint": {"drives": "left"}},
                {"gello_joint": {"drives": "right"}},
            ]
        },
        supported=("gello_joint",),
    )

    assert [entry["gello_joint"]["drives"] for entry in entries] == ["left", "right"]


def test_a_listed_device_the_env_cannot_drive_is_refused():
    with pytest.raises(ValueError, match="Unsupported teleop device"):
        resolve_teleop_devices(
            {"teleop": ["spacemouse", "glove"]}, supported=("gello_joint", "pico")
        )


def test_a_list_supersedes_a_retired_key_underneath_it():
    """Env configs layer, so a run config often sits on an older base."""
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "teleop_device": "none"},
            supported=SINGLE_ARM,
        )

    assert entries == ["spacemouse"]


def test_a_list_supersedes_a_retired_boolean():
    with pytest.warns(DeprecationWarning, match="supersedes"):
        entries = resolve_teleop_devices(
            {"teleop": ["spacemouse"], "use_pico": True}, supported=SINGLE_ARM
        )

    assert entries == ["spacemouse"]


def test_an_empty_list_is_refused():
    with pytest.raises(ValueError, match="'teleop' is empty"):
        resolve_teleop_devices({"teleop": []}, supported=SINGLE_ARM)


def test_none_cannot_share_the_list():
    with pytest.raises(ValueError, match="cannot share the list"):
        resolve_teleop_devices({"teleop": ["none", "spacemouse"]}, supported=SINGLE_ARM)


def test_none_alone_in_a_list_means_nobody_takes_over():
    assert resolve_teleop_devices({"teleop": ["none"]}, supported=SINGLE_ARM) == []


def test_a_two_key_entry_is_refused():
    with pytest.raises(ValueError, match="mapping of one name"):
        resolve_teleop_devices(
            {"teleop": [{"spacemouse": {}, "glove": {}}]},
            supported=("spacemouse", "glove"),
        )


def test_no_device_is_named_in_the_wrapper_stack():
    """Composition is a config question, so the builder answers it alone."""
    import inspect

    from rlinf.envs.real import wrappers

    source = inspect.getsource(wrappers)
    for device in ("spacemouse", "glove", "gello_joint", "pico"):
        assert device not in source, f"the wrapper stack still names {device!r}"


def test_a_leader_arm_reads_the_joint_convention_from_the_env():
    """How joint targets are read belongs to the env, not to the operator."""
    from rlinf.envs.real.wrappers.teleop.builder import DEVICES, EnvFacts
    from rlinf.robotics.teleop import ActionKind

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = DEVICES["gello_joint"](
        {"left_gello_port": "/dev/left"}, {"drives": "left"}, facts
    )

    assert entry.binding.use_delta is True
    assert entry.binding.action_scale == 0.25


def test_an_entry_option_wins_over_the_env_default():
    from rlinf.envs.real.wrappers.teleop.builder import DEVICES, EnvFacts
    from rlinf.robotics.teleop import ActionKind

    facts = EnvFacts(
        layout={"left.arm": slice(0, 7), "left.end_effector": slice(7, 8)},
        kinds={
            "left.arm": ActionKind.JOINT_DELTA,
            "left.end_effector": ActionKind.GRIPPER,
        },
        joint_action_scale=0.25,
    )
    entry = DEVICES["gello_joint"](
        {"left_gello_port": "/dev/left"},
        {"drives": "left", "action_scale": 0.5},
        facts,
    )

    assert entry.binding.action_scale == 0.5
    assert entry.binding.use_delta is True


def test_the_streamer_comes_from_the_registry_not_the_stack():
    """A device that also commands the robot says so where it is built."""
    from rlinf.envs.real.wrappers.teleop.builder import STREAMERS, EnvFacts

    quiet = EnvFacts(layout={}, kinds={}, direct_stream=False)
    assert STREAMERS["gello_joint"]({}, quiet, []) is None


# --- an env says what its action means --------------------------------


def _declared(cls, **attrs):
    """The parts a class declares, given only what its declaration reads."""
    return cls.action_parts(SimpleNamespace(**attrs))


def test_every_env_declares_parts_that_tile_its_action():
    """A declaration that disagrees with step() is worse than none at all."""
    from rlinf.envs.real.dosw1.base import DOSW1Env
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.envs.real.xsquare.base import Turtle2Env

    cases = [
        (7, _declared(FrankaEnv, _is_hand=False)),
        (12, _declared(FrankaEnv, _is_hand=True)),
        (7, _declared(GimArmEnv)),
        (7, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[1]))),
        (14, _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))),
        (14, _declared(DOSW1Env)),
    ]
    for width, parts in cases:
        assert sum(part.width for part in parts) == width


def test_a_two_armed_robot_names_both_arms():
    """Inferring this from the action width gave one arm and a huge gripper."""
    from rlinf.envs.real.xsquare.base import Turtle2Env

    parts = _declared(Turtle2Env, config=SimpleNamespace(use_arm_ids=[0, 1]))

    assert [part.name for part in parts] == [
        "left.arm",
        "left.end_effector",
        "right.arm",
        "right.end_effector",
    ]


def test_two_arms_of_the_same_width_can_mean_different_things():
    """Six numbers are a twist to one robot and joint angles to another."""
    from rlinf.envs.real.franka.base import FrankaEnv
    from rlinf.envs.real.gim_arm.base import GimArmEnv
    from rlinf.robotics.teleop import ActionKind

    franka_arm = _declared(FrankaEnv, _is_hand=False)[0]
    gim_arm = _declared(GimArmEnv)[0]

    assert franka_arm.width == gim_arm.width == 6
    assert franka_arm.kind is ActionKind.CARTESIAN_DELTA
    assert gim_arm.kind is ActionKind.JOINT_POSITION


def test_an_env_that_declares_nothing_cannot_be_teleoperated():
    from rlinf.envs.real.wrappers.teleop.layout import action_spec

    class Bare:
        unwrapped = None

        def get_wrapper_attr(self, name):
            raise AttributeError(name)

    Bare.unwrapped = Bare()
    with pytest.raises(AttributeError, match="does not declare action_parts"):
        action_spec(Bare())


def test_a_declaration_that_does_not_tile_the_action_is_refused():
    import gymnasium as gym

    from rlinf.envs.real.wrappers.teleop.layout import action_spec
    from rlinf.robotics.teleop import ActionKind, ActionPart

    class Wrong:
        action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

        def action_parts(self):
            return (ActionPart("arm", 6, ActionKind.CARTESIAN_DELTA),)

        def get_wrapper_attr(self, name):
            return getattr(self, name)

    env = Wrong()
    env.unwrapped = env
    with pytest.raises(ValueError, match="declares parts covering 6"):
        action_spec(env)


def test_a_device_that_means_something_else_is_refused():
    """The failure this whole declaration exists to prevent."""
    from rlinf.robotics.teleop import (
        ActionKind,
        SpaceMouseBinding,
        TeleopEntry,
        TeleopGroup,
    )

    class Device:
        is_connected = True

        def get_observation(self):
            return {}

        def connect(self):
            pass

        def disconnect(self):
            pass

    joint_arm = {
        "arm": ActionKind.JOINT_POSITION,
        "end_effector": ActionKind.GRIPPER,
    }
    with pytest.raises(ValueError, match="mean different things"):
        TeleopGroup([TeleopEntry(Device(), SpaceMouseBinding())], available=joint_arm)


# --- every shipped config names devices its robot can drive -----------


def _task_classes():
    """Gym id -> env class, across every robot that declares tasks."""
    from rlinf.envs.real import dosw1, franka, gim_arm, xsquare

    classes = {}
    for module in (franka, dosw1, xsquare, gim_arm):
        classes.update(getattr(module, "TASKS", {}))
    return classes


def _env_configs():
    """Every shipped env config: its path, its section, and its gym id.

    A run config layers an ``env/<name>`` file into ``env.train`` or
    ``env.eval`` and may override the device there, so both halves are read.
    """
    import yaml

    roots = [_ROOT / "examples", _ROOT / "evaluations"]
    env_files = {}
    for root in roots:
        for path in root.rglob("env/*.yaml"):
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            env_id = (doc.get("init_params") or {}).get("id")
            # Simulated envs register elsewhere and have no teleop to check.
            if env_id and str(doc.get("env_type", "")) in ("real", "realworld"):
                env_files[path.stem] = (path, doc, str(env_id))

    for path, doc, env_id in env_files.values():
        yield path, doc, env_id

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for root in roots:
        for path in root.rglob("*.yaml"):
            if path.parent.name == "env":
                continue
            try:
                doc = yaml.safe_load(path.read_text()) or {}
            except yaml.YAMLError:
                continue
            for entry in doc.get("defaults") or []:
                match = pattern.search(str(entry))
                if not match:
                    continue
                base = env_files.get(match.group(1))
                if base is None:
                    continue
                section = ((doc.get("env") or {}).get(match.group(2))) or {}
                if any(key in section for key in ("teleop", "teleop_device")):
                    yield path, section, base[2]


def test_shipped_configs_name_devices_their_env_can_drive():
    """A device an env cannot drive is only discovered with a robot moving.

    Every config in the repo is resolved against the env class its gym id
    names, so a typo or an unsupported device fails here instead of there.
    """
    classes = _task_classes()
    offenders = []
    for path, section, env_id in _env_configs():
        env_cls = classes.get(env_id)
        if env_cls is None:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                resolve_teleop_devices(
                    section,
                    supported=getattr(env_cls, "TELEOP", ()),
                    default=getattr(env_cls, "TELEOP_DEFAULT", NO_DEVICE),
                )
        except ValueError as error:
            offenders.append(f"{path.relative_to(_ROOT)} ({env_id}): {error}")

    assert offenders == []


def test_every_shipped_config_names_a_registered_task():
    """A gym id nobody registers is a config that cannot start."""
    classes = _task_classes()
    unknown = sorted(
        {
            env_id
            for _, _, env_id in _env_configs()
            if env_id not in classes and env_id.endswith("-v1")
        }
    )

    assert unknown == []


def test_the_retired_env_type_still_resolves_and_warns():
    """The package is `real`; configs that still say `realworld` keep working."""
    from rlinf.envs import SupportedEnvType

    assert SupportedEnvType("real") is SupportedEnvType.REAL
    with pytest.warns(DeprecationWarning, match="'realworld' is retired"):
        assert SupportedEnvType("realworld") is SupportedEnvType.REAL


def test_no_worker_compares_the_env_type_to_a_bare_string():
    """A literal would have gone quietly wrong when the value was renamed."""
    offenders = []
    for path in (_ROOT / "rlinf" / "workers").rglob("*.py"):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if re.search(r'env_type\s*[!=]=\s*["\']', line):
                offenders.append(f"{path.relative_to(_ROOT)}:{number}")

    assert offenders == []


def _resolved(doc, value):
    """Follow a ``${a.b.c}`` interpolation within the same document."""
    if not isinstance(value, str) or not value.startswith("${"):
        return value
    node = doc
    for key in value[2:-1].split("."):
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return _resolved(doc, node)


def _merged_sections(path, doc):
    """Each env section of a run config, as Hydra composes it.

    A section layers its own keys over the ``env/<name>`` file its ``defaults``
    names, and ``override_cfg`` merges key by key rather than wholesale. What
    the section leaves out it inherits, which is how a task ends up on a
    different robot than the one it trained on.
    """
    import yaml

    pattern = re.compile(r"env/([\w-]+)@env\.(train|eval)")
    for entry in doc.get("defaults") or []:
        match = pattern.search(str(entry))
        if not match:
            continue
        base_path = path.parent / "env" / f"{match.group(1)}.yaml"
        if not base_path.exists():
            continue
        try:
            base = yaml.safe_load(base_path.read_text()) or {}
        except yaml.YAMLError:
            continue
        section = ((doc.get("env") or {}).get(match.group(2))) or {}
        merged = {**base, **section}
        merged["override_cfg"] = {
            **(base.get("override_cfg") or {}),
            **(section.get("override_cfg") or {}),
        }
        yield match.group(2), merged, str((base.get("init_params") or {}).get("id"))


def test_shipped_configs_give_the_policy_the_action_width_it_expects():
    """An env whose action is a different width than the model's is unusable.

    ``realworld_dexpnp_rlpd_cnn_async`` configured a dexterous hand for
    training and left evaluation to inherit the default gripper, so the same
    policy faced a 12-wide action in one section and a 7-wide one in the other.
    """
    import yaml

    from rlinf.envs.real.franka.base import FrankaEnv

    classes = _task_classes()
    offenders = []
    for path in (_ROOT / "examples").rglob("*.yaml"):
        if path.parent.name == "env":
            continue
        try:
            doc = yaml.safe_load(path.read_text()) or {}
        except yaml.YAMLError:
            continue
        if not isinstance(doc, dict):
            continue
        action_dim = _resolved(
            doc, ((doc.get("rollout") or {}).get("model") or {}).get("action_dim")
        )
        if not isinstance(action_dim, int):
            continue
        for name, section, env_id in _merged_sections(path, doc):
            # Only a single-arm Franka lays its action out this way.
            env_cls = classes.get(env_id)
            if env_cls is None or not issubclass(env_cls, FrankaEnv):
                continue
            end_effector = str(
                section["override_cfg"].get("end_effector_type", "franka_gripper")
            )
            parts = FrankaEnv.action_parts(
                SimpleNamespace(_is_hand=end_effector.endswith("hand"))
            )
            width = sum(part.width for part in parts)
            if width != action_dim:
                offenders.append(
                    f"{path.relative_to(_ROOT)} env.{name}: {end_effector} gives "
                    f"{width}, model wants {action_dim}"
                )

    assert offenders == []


def test_direct_stream_gello_opens_one_reader_per_port():
    """The streamer drives the arms the group composed, not its own copies.

    It used to build a second pair from the config's *global* ports. That is
    two pollers competing for one serial port, and a per-entry ``port``
    override that only the action path had heard of.
    """
    import sys
    import types

    import numpy as np

    opened = []

    fake = types.ModuleType("rlinf.robotics.parts.teleop.readers.gello_joint")

    class GelloJointExpert:
        def __init__(self, port):
            opened.append(port)
            self.ready = True

        def get_action(self):
            return np.zeros(7), np.zeros(1)

        def close(self):
            pass

    fake.GelloJointExpert = GelloJointExpert
    saved = sys.modules.get("rlinf.robotics.parts.teleop.readers.gello_joint")
    sys.modules["rlinf.robotics.parts.teleop.readers.gello_joint"] = fake
    try:
        from rlinf.envs.real.wrappers.teleop.builder import STREAMERS, EnvFacts
        from rlinf.robotics.parts.teleop.devices import TeleopLeaderArm
        from rlinf.robotics.teleop import TeleopEntry

        arms = {
            side: TeleopLeaderArm(port=f"/dev/{side}", joint_space=True)
            for side in ("left", "right")
        }
        entries = [TeleopEntry(arm, None, drives=side) for side, arm in arms.items()]
        for arm in arms.values():
            arm.connect()
        assert opened == ["/dev/left", "/dev/right"]

        facts = EnvFacts(layout={}, kinds={}, direct_stream=True)
        streamer = STREAMERS["gello_joint"]({}, facts, entries)

        assert opened == ["/dev/left", "/dev/right"], (
            f"the streamer opened more readers: {opened}"
        )
        assert streamer.left_arm is arms["left"]
        assert streamer.right_arm is arms["right"]
    finally:
        if saved is None:
            sys.modules.pop("rlinf.robotics.parts.teleop.readers.gello_joint", None)
        else:
            sys.modules["rlinf.robotics.parts.teleop.readers.gello_joint"] = saved
