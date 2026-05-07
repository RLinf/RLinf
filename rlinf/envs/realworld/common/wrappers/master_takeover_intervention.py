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

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.takeover import (
    X2RobotTakeoverTCPConfig,
    X2RobotTakeoverTCPServer,
)
from rlinf.envs.realworld.common.takeover.x2robot_protocol import (
    make_ros_running_mode_getter,
)


class MasterTakeoverIntervention(gym.ActionWrapper):
    """Select policy or master takeover actions without recording sync waits."""

    def __init__(
        self,
        env: gym.Env,
        config: Mapping[str, Any] | None = None,
        adapter: X2RobotTakeoverTCPServer | None = None,
    ):
        super().__init__(env)

        joint_snapshot = np.asarray(
            self.get_wrapper_attr("get_joint_snapshot")(), dtype=np.float32
        )
        if joint_snapshot.shape != (2, 7):
            raise ValueError(
                "Master takeover requires dual-arm joint snapshots with shape "
                f"(2, 7), got {joint_snapshot.shape}."
            )

        if adapter is None:
            tcp_cfg = X2RobotTakeoverTCPConfig.from_dict(config)
            adapter = X2RobotTakeoverTCPServer(
                config=tcp_cfg,
                running_mode_getter=make_ros_running_mode_getter(tcp_cfg),
                joint_snapshot_getter=self.get_wrapper_attr("get_joint_snapshot"),
                logger=logging.getLogger(__name__),
            )
        self.adapter = adapter
        adapter_config = getattr(adapter, "config", None)
        self._slave_hold_settle_s = float(
            getattr(adapter_config, "slave_hold_settle_s", 0.0)
            or (config.get("slave_hold_settle_s", 0.0) if config is not None else 0.0)
        )
        self.adapter.start()

        self._chunk_active = False
        self._hold_until_chunk_end = False
        self._was_takeover_active = self.adapter.is_takeover_active()
        self._takeover_sync_wait_steps = 0
        self._pending_decision: dict[str, Any] | None = None
        if self._was_takeover_active:
            self._clear_takeover_target()
            self._set_takeover_gate(True)
            self._hard_hold_action()

    def begin_action_chunk(self) -> None:
        self._chunk_active = True
        self._hold_until_chunk_end = False

    def end_action_chunk(self) -> None:
        self._chunk_active = False
        self._hold_until_chunk_end = False

    def _call_env_hook(self, name: str, *args):
        try:
            hook = self.get_wrapper_attr(name)
        except AttributeError:
            return None
        return hook(*args)

    def _set_takeover_gate(self, enabled: bool) -> None:
        self._call_env_hook("set_takeover_gate", enabled)

    def _sync_smooth_target_to_current_pose(self) -> None:
        self._call_env_hook("sync_smooth_target_to_current_pose")

    def _clear_takeover_target(self) -> None:
        self._call_env_hook("clear_takeover_target")

    def _set_pose_backend_override(self, backend: str, source: str) -> None:
        self._call_env_hook("set_pose_control_backend_override", backend, source)

    def _make_decision(
        self,
        *,
        action: np.ndarray | None,
        should_step: bool,
        source: str,
        replaced: bool = False,
    ) -> dict[str, Any]:
        return {
            "action": action,
            "should_step": should_step,
            "source": source,
            "replaced": replaced,
        }

    def action(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, bool, bool, bool, str, str]:
        decision = self.select_takeover_action(action)
        if decision["should_step"]:
            return (
                decision["action"],
                bool(decision["replaced"]),
                False,
                False,
                str(decision["source"]),
                "pose",
            )
        return (
            action,
            False,
            decision["source"] == "chunk_tail_skip",
            decision["source"] == "sync_wait",
            str(decision["source"]),
            "pose",
        )

    def set_pending_takeover_decision(self, decision: Mapping[str, Any]) -> None:
        self._pending_decision = dict(decision)

    def select_takeover_action(self, action: np.ndarray) -> dict[str, Any]:
        self.adapter.poll()
        takeover_active = self.adapter.is_takeover_active()

        if self._was_takeover_active and not takeover_active:
            self._sync_smooth_target_to_current_pose()
            self._clear_takeover_target()
            self._set_takeover_gate(False)
            if self._chunk_active:
                self._hold_until_chunk_end = True

        if takeover_active and not self._was_takeover_active:
            self._clear_takeover_target()
            self._set_takeover_gate(True)
            self._hard_hold_action()
            self._hold_until_chunk_end = False
            self._takeover_sync_wait_steps = 0

        self._was_takeover_active = takeover_active

        expert_action = self.adapter.get_takeover_action()
        if expert_action is not None:
            return self._make_decision(
                action=expert_action.astype(np.float32, copy=False),
                should_step=True,
                source="expert",
                replaced=True,
            )

        if takeover_active:
            self._takeover_sync_wait_steps += 1
            if self._takeover_sync_wait_steps == 1 and self._slave_hold_settle_s > 0:
                time.sleep(self._slave_hold_settle_s)
            self.adapter.sync_control_plane()
            time.sleep(0.01)
            return self._make_decision(
                action=None,
                should_step=False,
                source="sync_wait",
            )

        if self._hold_until_chunk_end:
            self.adapter.sync_control_plane()
            return self._make_decision(
                action=None,
                should_step=False,
                source="chunk_tail_skip",
            )

        return self._make_decision(
            action=np.asarray(action, dtype=np.float32),
            should_step=True,
            source="policy",
        )

    def step(self, action):
        if self._pending_decision is None:
            decision = self.select_takeover_action(action)
        else:
            decision = self._pending_decision
            self._pending_decision = None
        if not decision["should_step"]:
            raise RuntimeError(
                "Master takeover sync_wait/chunk_tail_skip decisions only support "
                "the chunk_step() collection path. Use run_realworld_eval.sh with "
                "the Turtle2 takeover collect config instead of direct env.step()."
            )
        return self.step_takeover_action(decision)

    def step_takeover_action(self, decision: Mapping[str, Any]):
        action = np.asarray(decision["action"], dtype=np.float32)
        source = str(decision["source"])
        backend = "takeover" if source == "expert" else "smooth"
        self._set_pose_backend_override(backend, source)

        obs, rew, done, truncated, info = self.env.step(action)
        self.adapter.sync_control_plane()

        if source == "expert":
            info["intervene_flag"] = np.asarray([True], dtype=bool)
            info["intervene_action"] = action.copy()
        return obs, rew, done, truncated, info

    def close(self):
        try:
            if self._was_takeover_active or self.adapter.is_takeover_active():
                self._sync_smooth_target_to_current_pose()
            self._clear_takeover_target()
            self._set_takeover_gate(False)
        finally:
            self.adapter.close()
        return self.env.close()

    def _hold_action(self) -> np.ndarray:
        pose_snapshot = np.asarray(
            self.get_wrapper_attr("get_arm_pose_snapshot")(), dtype=np.float32
        )
        return self._flatten_pose_snapshot(pose_snapshot)

    def _hard_hold_action(self) -> np.ndarray:
        try:
            hold_snapshot = self.get_wrapper_attr("hold_current_pose_for_takeover")()
        except AttributeError:
            return self._hold_action()

        if isinstance(hold_snapshot, Mapping):
            pose_snapshot = hold_snapshot.get("pose")
        else:
            pose_snapshot = hold_snapshot
        return self._flatten_pose_snapshot(np.asarray(pose_snapshot, dtype=np.float32))

    @staticmethod
    def _flatten_pose_snapshot(pose_snapshot: np.ndarray) -> np.ndarray:
        if pose_snapshot.shape != (2, 7):
            raise ValueError(
                "Master takeover chunk-boundary recovery requires dual-arm pose "
                f"snapshots with shape (2, 7), got {pose_snapshot.shape}."
            )
        return pose_snapshot.reshape(-1).astype(np.float32, copy=False)
