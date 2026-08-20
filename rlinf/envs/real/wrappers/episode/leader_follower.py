# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

import gymnasium as gym


class LeaderFollowerKeyboardIntervention(gym.Wrapper):
    """Keyboard intervention wrapper for leader-follower teleoperation envs."""

    def __init__(self, env):
        super().__init__(env)
        base_env = self._base_env()
        register_callback = getattr(base_env, "set_keyboard_event_callback", None)
        if callable(register_callback):
            register_callback(self._handle_key_event)

    def step(self, action):
        step_result = self._handle_key_event(reset_phase=False)
        if step_result is not None:
            return step_result
        return self.env.step(action)

    def _handle_key_event(self, reset_phase: bool = False):
        base_env = self._base_env()
        keyboard = getattr(base_env, "_keyboard", None)
        if keyboard is None:
            return None
        manual_episode_control_only = bool(
            getattr(
                getattr(base_env, "config", None), "manual_episode_control_only", False
            )
        )
        in_free_teleop = bool(getattr(base_env, "in_free_teleop", False))
        key = keyboard.get_key()

        control_mode = getattr(base_env, "control_mode", None)

        if key is None:
            return None

        if key == "s":
            if reset_phase or in_free_teleop:
                setattr(base_env, "start_episode_requested", True)
            return None

        if reset_phase:
            return None

        if control_mode is None:
            return None
        control_mode_type = type(control_mode)
        model_mode = getattr(control_mode_type, "MODEL", None)
        pause_mode = getattr(control_mode_type, "PAUSE", None)
        teleop_mode = getattr(control_mode_type, "TELEOP", None)
        if model_mode is None or pause_mode is None or teleop_mode is None:
            return None

        if key == "r":
            self._log_info("[LeaderFollowerEnv] -> FreeTeleop (episode aborted)")
            setattr(base_env, "in_free_teleop", True)
            return self._build_truncated_result()

        if key == "d":
            self._log_info("[LeaderFollowerEnv] Manual done (episode saved)")
            setattr(base_env, "manual_done", True)
            return self._build_truncated_result()

        if manual_episode_control_only and key in {"p", "t", "m"}:
            return None

        set_control_mode = getattr(base_env, "set_control_mode", None)

        def _update_mode(target_mode, source: str) -> None:
            if callable(set_control_mode):
                set_control_mode(target_mode, source=source)
            else:
                setattr(base_env, "control_mode", target_mode)

        if key == "p" and control_mode in (model_mode, teleop_mode):
            self._log_info(
                "[LeaderFollowerEnv] %s -> PAUSE",
                getattr(control_mode, "name", ""),
            )
            _update_mode(pause_mode, "keyboard_p")
        elif key == "t" and control_mode == pause_mode:
            self._log_info("[LeaderFollowerEnv] PAUSE -> TELEOP")
            _update_mode(teleop_mode, "keyboard_t")
            snapshot_fn = getattr(base_env, "snapshot_teleop_init", None)
            if callable(snapshot_fn):
                snapshot_fn()
        elif key == "m" and control_mode == pause_mode:
            self._log_info("[LeaderFollowerEnv] PAUSE -> MODEL")
            _update_mode(model_mode, "keyboard_m")

        return None

    def _build_truncated_result(self):
        base_env = self._base_env()
        obs_fn = getattr(base_env, "_get_observation", None)
        if not callable(obs_fn):
            return None
        observation = obs_fn()
        control_mode = getattr(base_env, "control_mode", None)
        control_mode_value = (
            getattr(control_mode, "value", control_mode)
            if control_mode is not None
            else 0
        )
        return (
            observation,
            0.0,
            False,
            True,
            {
                "control_mode": control_mode_value,
                "manual_done": bool(getattr(base_env, "manual_done", False)),
            },
        )

    def _log_info(self, message: str, *args) -> None:
        logger = getattr(self._base_env(), "_logger", None)
        if logger is not None:
            logger.info(message, *args)

    def _base_env(self):
        return getattr(self.env, "unwrapped", self.env)
