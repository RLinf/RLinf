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
    """Override absolute dual-arm policy actions with X2Robot master poses."""

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
        self._logger = logging.getLogger(__name__)
        adapter_config = getattr(adapter, "config", None)
        self._debug_log = bool(
            getattr(adapter_config, "debug_log", False)
            or (config is not None and bool(config.get("debug_log", False)))
        )
        self._slave_hold_settle_s = float(
            getattr(adapter_config, "slave_hold_settle_s", 0.0)
            or (config.get("slave_hold_settle_s", 0.0) if config is not None else 0.0)
        )
        self.adapter.start()

        self._chunk_active = False
        self._hold_until_chunk_end = False
        self._was_takeover_active = self.adapter.is_takeover_active()
        self._chunk_step_index = 0
        self._takeover_sync_hold_steps = 0

    def begin_action_chunk(self) -> None:
        self._chunk_active = True
        self._hold_until_chunk_end = False
        self._chunk_step_index = 0

    def end_action_chunk(self) -> None:
        self._chunk_active = False
        self._hold_until_chunk_end = False

    def action(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, bool, bool, bool, str, str]:
        self.adapter.poll()
        takeover_active = self.adapter.is_takeover_active()
        if self._chunk_active and self._was_takeover_active and not takeover_active:
            self._hold_until_chunk_end = True
            if self._debug_log:
                self._logger.info(
                    "Master takeover wrapper mode exit inside chunk: chunk_step=%s; holding until boundary",
                    self._chunk_step_index,
                )
        if takeover_active and not self._was_takeover_active:
            self._takeover_sync_hold_steps = 0
            if self._debug_log:
                self._logger.info(
                    "Master takeover wrapper mode enter: chunk_active=%s chunk_step=%s; policy actions will be blocked until fresh master pose",
                    self._chunk_active,
                    self._chunk_step_index,
                )
        self._was_takeover_active = takeover_active

        expert_action = self.adapter.get_takeover_action()
        if expert_action is not None:
            return (
                expert_action.astype(np.float32, copy=False),
                True,
                False,
                False,
                "expert",
                "pose",
            )

        if takeover_active:
            self._takeover_sync_hold_steps += 1
            return self._hold_action(), False, False, True, "sync_hold", "pose"

        if self._hold_until_chunk_end:
            return self._hold_action(), False, True, False, "chunk_hold", "pose"

        return action, False, False, False, "policy", "pose"

    def step(self, action):
        step_start = time.time()
        (
            new_action,
            replaced,
            chunk_holding,
            sync_holding,
            decision,
            selected_control_mode,
        ) = self.action(action)
        action_selected_time = time.time()
        pre_step_sync_done = False
        if sync_holding and self._takeover_sync_hold_steps == 1:
            new_action = self._hard_hold_action()
            if self._slave_hold_settle_s > 0:
                if self._debug_log:
                    self._logger.info(
                        "Master takeover wrapper settling slave hold before master sync: settle_s=%.3f",
                        self._slave_hold_settle_s,
                    )
                time.sleep(self._slave_hold_settle_s)
            self.adapter.sync_control_plane()
            pre_step_sync_done = True
        obs, rew, done, truncated, info = self.env.step(new_action)
        env_step_done_time = time.time()
        if not pre_step_sync_done:
            self.adapter.sync_control_plane()
        control_plane_done_time = time.time()
        executed_for_log = np.asarray(
            info.get("executed_action", new_action), dtype=np.float32
        )
        action_rejected = bool(info.get("action_rejected", False))
        if self._debug_log and (
            self.adapter.is_takeover_active() or decision != "policy"
        ):
            self._logger.info(
                "Master takeover wrapper step: decision=%s chunk_step=%s sync_hold_steps=%s selected_dt=%.4f env_dt=%.4f sync_dt=%.4f action=%s executed=%s",
                decision,
                self._chunk_step_index,
                self._takeover_sync_hold_steps,
                action_selected_time - step_start,
                env_step_done_time - action_selected_time,
                control_plane_done_time - env_step_done_time,
                np.array2string(np.asarray(new_action), precision=4, threshold=20),
                np.array2string(executed_for_log, precision=4, threshold=20),
            )
        self._chunk_step_index += 1
        info["takeover_control_mode"] = "pose"
        info["executed_action"] = executed_for_log
        accepted_intervention = replaced and not action_rejected
        info["intervene_flag"] = np.asarray([accepted_intervention], dtype=bool)
        if replaced and action_rejected:
            info["raw_intervene_action"] = np.asarray(new_action, dtype=np.float32)
            info["intervene_rejected"] = True
        elif accepted_intervention:
            info["intervene_action"] = executed_for_log
        info["takeover_active"] = self.adapter.is_takeover_active()
        info["takeover_chunk_hold"] = chunk_holding
        info["takeover_sync_hold"] = sync_holding
        info["master_takeover_connected"] = self.adapter.is_connected()
        return obs, rew, done, truncated, info

    def close(self):
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
