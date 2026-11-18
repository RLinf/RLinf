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

"""Gate standalone real-world evaluation episodes with Pico controller buttons."""

from __future__ import annotations

from typing import Any, Mapping, Optional, SupportsFloat

import gymnasium as gym

from .pico_control import PicoEpisodeControlWrapper


class PicoEvalControlWrapper(PicoEpisodeControlWrapper):
    """Let an operator pace and score a policy rollout from the Pico buttons.

    The shared evaluation loop owns the episode lifecycle. It resets the
    environment before the first chunk and, with ``auto_reset``, after every
    chunk in which an episode ended. This wrapper only decides when an episode
    may start and how it ends: ``reset`` blocks until the start button is
    pressed, and the success or failure button ends the episode with reward 1.0
    or 0.0. Later steps of the action chunk that contained the press are not
    executed.

    The loop runs ``rollout_epoch * max_steps_per_rollout_epoch`` environment
    steps in total, so one evaluation can contain any number of episodes. The
    reset the loop issues after the last chunk has no episode to serve and
    returns the last observation without moving the robot or waiting for the
    start button, and restarts the step count.

    When ``teleop`` provides a Pico reader, the wrapper reads button events from
    it. Otherwise it opens its own button-only reader on ``zmq_addr``, which
    :meth:`close` releases.

    Args:
        env: Environment whose ``unwrapped`` object provides ``observe()``.
        budget_steps: Environment steps in the whole evaluation.
        zmq_addr: Pico ZMQ endpoint for the wrapper's own reader. ``None``
            selects the reader's default endpoint.
        max_stale_s: Age in seconds after which Pico data counts as lost.
            ``None`` selects the reader's default.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        budget_steps: int,
        zmq_addr: Optional[str] = None,
        max_stale_s: Optional[float] = None,
    ) -> None:
        super().__init__(env)
        if (
            not isinstance(budget_steps, int)
            or isinstance(budget_steps, bool)
            or budget_steps <= 0
        ):
            raise ValueError(
                f"budget_steps must be a positive integer, got {budget_steps!r}."
            )
        self.budget_steps = budget_steps
        self._reader_kwargs: dict[str, Any] = {
            "calibration": {"enabled": False},
        }
        if zmq_addr is not None:
            self._reader_kwargs["zmq_addr"] = zmq_addr
        if max_stale_s is not None:
            self._reader_kwargs["max_stale_s"] = max_stale_s
        self._steps = 0
        self._successes = 0
        self._scored = 0
        self._owned_reader: Any = None
        self._closed = False

    @classmethod
    def from_cfg(cls, env: gym.Env, cfg: Mapping[str, Any]) -> PicoEvalControlWrapper:
        """Build the wrapper from the ``env.eval`` configuration.

        Raises:
            ValueError: If a setting contradicts the lifecycle this wrapper
                relies on. The message names the offending key.
        """
        if not cfg.get("auto_reset", True):
            raise ValueError(
                "pico_reward_wrapper 'eval' requires auto_reset: true so the "
                "evaluation loop resets after every scored episode."
            )
        if cfg.get("ignore_terminations", False):
            raise ValueError(
                "pico_reward_wrapper 'eval' requires ignore_terminations: false "
                "so the success and failure buttons end an episode."
            )
        if cfg.get("total_num_envs", 1) != 1:
            raise ValueError(
                "pico_reward_wrapper 'eval' requires total_num_envs: 1; one "
                "Pico controller scores one robot."
            )
        if cfg.get("keyboard_reward_wrapper", None):
            raise ValueError(
                "pico_reward_wrapper 'eval' cannot be combined with "
                "keyboard_reward_wrapper; both would gate the same episodes."
            )
        data_collection = cfg.get("data_collection", None) or {}
        if data_collection.get("enabled", False):
            raise ValueError(
                "pico_reward_wrapper 'eval' requires data_collection.enabled: false."
            )
        if not callable(getattr(env.unwrapped, "observe", None)):
            raise ValueError(
                f"pico_reward_wrapper 'eval' needs {type(env.unwrapped).__name__} "
                "to provide observe() for a fresh observation after the start button."
            )
        rollout_epoch = cfg.get("rollout_epoch", 1)
        steps_per_epoch = cfg.get("max_steps_per_rollout_epoch", None)
        if steps_per_epoch is None:
            raise ValueError(
                "pico_reward_wrapper 'eval' requires max_steps_per_rollout_epoch."
            )
        pico = cfg.get("pico", None) or {}
        return cls(
            env,
            budget_steps=rollout_epoch * steps_per_epoch,
            zmq_addr=pico.get("zmq_addr", None),
            max_stale_s=pico.get("max_stale_s", None),
        )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        if self._steps >= self.budget_steps:
            self._steps = 0
            self._successes = 0
            self._scored = 0
            self._recording = False
            self._running = False
            return self._last_obs, {}
        self._recording = False
        self._running = False
        self._drain_presses()
        _, info = self.env.reset(seed=seed, options=options)
        # Presses made while the arms were moving must not start the episode.
        self._drain_presses()
        self._wait_for_start()
        # The scene is rearranged while the arms wait at home, so the image
        # captured by the reset is stale by the time the operator presses start.
        self._last_obs = self.env.unwrapped.observe()
        return self._last_obs, info

    def step(
        self, action: Any
    ) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        self._steps += 1
        if not self._running:
            # The episode was scored earlier in this chunk. The start button is
            # only read in reset(), so there is nothing to wait for or detect.
            return self._idle_response(event=None)
        result = super().step(action)
        event = result[4].get("pico_episode_event")
        if event in ("end_success", "end_failure"):
            self._log(
                "Pico eval: episode ended at %d of %d budget steps",
                self._steps,
                self.budget_steps,
            )
            self._scored += 1
            self._successes += event == "end_success"
            self._log(
                "success rate: %d/%d, %d%%",
                self._successes,
                self._scored,
                100 * self._successes // self._scored,
            )
        return result

    def close(self) -> None:
        """Park the robot, then release the button reader. Safe to repeat."""
        self._closed = True
        try:
            super().close()
        finally:
            reader, self._owned_reader = self._owned_reader, None
            if reader is not None:
                reader.stop()

    def _pico_readers(self) -> list[Any]:
        readers = super()._pico_readers()
        if readers or self._closed:
            return readers
        if self._owned_reader is None:
            from rlinf.robotics.parts.transports.pico import PicoExpert

            # PicoExpert connects and starts its thread on construction, so
            # it is created on first use and never for a teleop-owned reader.
            self._owned_reader = PicoExpert(**self._reader_kwargs)
        return [self._owned_reader]
