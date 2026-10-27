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

"""Hardware-free checks for the LeKiwi bring-up commands."""

import contextlib
import io
import logging
import unittest
from unittest import mock

from robot_mocks.lekiwi import FakeKeyboard, FakeKeySource, FakeWheelBus

from rlinf.robotics.parts.cameras.webcam import WebcamCamera
from rlinf.robotics.parts.mobility.lekiwi import body_to_wheel_raw, limit_wheel_raw
from toolkits.lekiwi.camera import pick_device
from toolkits.lekiwi.remote import run_receiver
from toolkits.lekiwi.teleop import KeyboardControl, parse_args, run_teleoperation
from toolkits.lekiwi.wheels import report, spin_each_wheel


def wheel_command(
    x: float,
    y: float,
    theta: float,
    *,
    wheel_radius: float = 0.05,
    base_radius: float = 0.125,
    max_raw: int = 3000,
) -> dict[str, int]:
    """Return the servo command the base sends for one body velocity."""
    raw = body_to_wheel_raw(
        x, y, theta, wheel_radius=wheel_radius, base_radius=base_radius
    )
    return limit_wheel_raw(raw, max_raw)[0]


class TestLeKiwi(unittest.TestCase):
    def setUp(self) -> None:
        self.args = parse_args(["--port", "fake", "--fps", "1000"])
        self.logger = logging.getLogger("test_lekiwi")
        self.logger.addHandler(logging.NullHandler())
        self.logger.propagate = False
        self.params = {
            "wheel_radius": self.args.wheel_radius,
            "base_radius": self.args.base_radius,
            "max_raw": self.args.max_raw,
            "fps": self.args.fps,
            "logger": self.logger,
        }

    def test_keys_and_opposing_directions(self) -> None:
        control = KeyboardControl()
        for key, expected in {
            "w": (0.1, 0.0, 0.0),
            "s": (-0.1, 0.0, 0.0),
            "a": (0.0, 0.1, 0.0),
            "d": (0.0, -0.1, 0.0),
            "z": (0.0, 0.0, 30.0),
            "x": (0.0, 0.0, -30.0),
        }.items():
            with self.subTest(key=key):
                self.assertEqual(control.action({key}), expected)
        self.assertEqual(control.action(set("wasdzx")), (0.0, 0.0, 0.0))
        self.assertEqual(control.action(set()), (0.0, 0.0, 0.0))

    def test_speed_changes_once_per_press_and_stays_in_range(self) -> None:
        control = KeyboardControl()
        for _ in range(5):
            self.assertEqual(control.action({"r", "w"})[0], 0.2)
        control.action(set())
        self.assertEqual(control.action({"r", "w"})[0], 0.3)
        for _ in range(5):
            control.action(set())
            self.assertEqual(control.action({"r", "w"})[0], 0.3)
        for _ in range(5):
            control.action({"f"})
            control.action(set())
        self.assertEqual(control.action({"w"})[0], 0.1)

    def test_release_and_quit_stop_wheels(self) -> None:
        keyboard = FakeKeyboard([{"W"}, set(), {"a"}, {"q", "w"}])
        bus = FakeWheelBus()
        run_teleoperation(keyboard, bus, **self.params)
        self.assertEqual(bus.commands[0], wheel_command(0.1, 0, 0))
        self.assertFalse(any(bus.commands[1].values()))
        self.assertEqual(len(bus.commands), 4)
        self.assertFalse(any(bus.commands[-1].values()))
        self.assert_stopped(bus, keyboard)

    def test_listener_loss_stops_without_replaying_held_key(self) -> None:
        keyboard = FakeKeyboard([{"w"}, None])
        bus = FakeWheelBus()
        run_teleoperation(keyboard, bus, **self.params)
        self.assertEqual(len(bus.commands), 2)
        self.assertFalse(any(bus.commands[-1].values()))
        self.assert_stopped(bus, keyboard)

    def test_no_keyboard_never_opens_usb(self) -> None:
        keyboard = FakeKeyboard([], available=False)
        bus = FakeWheelBus()
        with self.assertRaisesRegex(RuntimeError, "Keyboard capture"):
            run_teleoperation(keyboard, bus, **self.params)
        self.assertFalse(bus.connect_attempted)

    def test_interrupt_and_input_error_stop_wheels(self) -> None:
        for error in (KeyboardInterrupt(), RuntimeError("input failure")):
            with self.subTest(error=type(error).__name__):
                keyboard = FakeKeyboard([{"z"}, error])
                bus = FakeWheelBus()
                with self.assertRaises(type(error)):
                    run_teleoperation(keyboard, bus, **self.params)
                self.assert_stopped(bus, keyboard)

    def test_movement_error_stops_and_closes_usb(self) -> None:
        keyboard = FakeKeyboard([{"w"}])
        bus = FakeWheelBus(fail_motion=True)
        with self.assertRaisesRegex(OSError, "USB error"):
            run_teleoperation(keyboard, bus, **self.params)
        self.assert_stopped(bus, keyboard)

    def test_unreachable_wheel_does_not_prevent_other_stops_or_port_close(self) -> None:
        keyboard = FakeKeyboard([{"z"}])
        bus = FakeWheelBus(fail_motion=True, lost_wheel="left")
        with self.assertRaises(OSError):
            run_teleoperation(keyboard, bus, **self.params)
        self.assertEqual(bus.stop_attempts[-3:], ["left", "back", "right"])
        self.assertEqual(bus.disable_attempts[-3:], ["left", "back", "right"])
        self.assertEqual(bus.velocity["back"], 0)
        self.assertEqual(bus.velocity["right"], 0)
        self.assertFalse(bus.is_connected)
        self.assertFalse(keyboard.is_connected)

    def test_partial_connection_failure_closes_port(self) -> None:
        keyboard = FakeKeyboard([])
        bus = FakeWheelBus(fail_connect=True)
        with self.assertRaisesRegex(OSError, "handshake"):
            run_teleoperation(keyboard, bus, **self.params)
        self.assert_stopped(bus, keyboard)

    def test_preview_runs_without_motor_bus(self) -> None:
        keyboard = FakeKeyboard([{"w"}, set(), {"q"}])
        run_teleoperation(keyboard, None, **self.params)
        self.assertFalse(keyboard.is_connected)

    def test_remote_receiver_drives_from_streamed_keys(self) -> None:
        source = FakeKeySource([(True, {"w"}), (True, {"q", "w"})])
        bus = FakeWheelBus()
        run_receiver(source, bus, **self.params)
        self.assertEqual(bus.commands[0], wheel_command(0.1, 0, 0))
        self.assertFalse(any(bus.velocity.values()))
        self.assertFalse(any(bus.torque.values()))
        self.assertFalse(bus.is_connected)

    def test_remote_receiver_stops_when_the_sender_disappears(self) -> None:
        source = FakeKeySource([(True, {"w"}), (False, set()), (True, {"q"})])
        bus = FakeWheelBus()
        run_receiver(source, bus, **self.params)
        self.assertFalse(any(bus.commands[1].values()))
        self.assertFalse(bus.is_connected)

    def test_invalid_options_are_rejected(self) -> None:
        for flags in (
            [],
            ["--dry-run", "--fps", "0"],
            ["--dry-run", "--fps", "nan"],
            ["--dry-run", "--wheel-radius", "-1"],
            ["--dry-run", "--base-radius", "inf"],
            ["--dry-run", "--max-raw", "32768"],
            ["--dry-run", "--max-raw", "0"],
            ["--listen"],
            ["--listen", "--port", "fake", "--heartbeat-timeout", "0"],
            ["--listen", "--port", "fake", "--remote-port", "0"],
            ["--listen", "--remote", "robot"],
            ["--remote", "robot", "--port", "fake"],
            ["--remote", "robot", "--dry-run"],
            ["--remote", "robot", "--connect-timeout", "nan"],
        ):
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(flags)
        self.assertTrue(parse_args(["--dry-run"]).dry_run)
        self.assertTrue(parse_args(["--listen", "--port", "fake"]).listen)
        self.assertEqual(parse_args(["--remote", "robot"]).remote, "robot")

    def test_wheel_check_reads_each_wheel_back(self) -> None:
        bus = FakeWheelBus()
        measured = spin_each_wheel(bus, raw=100, seconds=0.02, logger=self.logger)
        self.assertEqual(measured, dict.fromkeys(("left", "back", "right"), 100))
        self.assertTrue(report(100, measured, self.logger))
        self.assertFalse(any(bus.velocity.values()))
        self.assertFalse(any(bus.torque.values()))
        self.assertFalse(bus.is_connected)

    def test_wheel_check_flags_a_wheel_that_did_not_turn(self) -> None:
        self.assertFalse(
            report(100, {"left": 100, "back": 0, "right": 100}, self.logger)
        )

    def test_camera_check_requires_a_choice_when_several_cameras_exist(self) -> None:
        with mock.patch.object(WebcamCamera, "discover", return_value={"2", "0"}):
            with self.assertRaisesRegex(RuntimeError, "Several webcams"):
                pick_device(None, self.logger)
            self.assertEqual(pick_device("2", self.logger), "2")

    def assert_stopped(self, bus: FakeWheelBus, keyboard: FakeKeyboard) -> None:
        self.assertFalse(any(bus.velocity.values()))
        self.assertFalse(any(bus.torque.values()))
        self.assertFalse(bus.is_connected)
        self.assertFalse(keyboard.is_connected)


if __name__ == "__main__":
    unittest.main()
