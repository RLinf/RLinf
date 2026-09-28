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
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.utils.nested_dict_process import clone_nested_to_cpu


class HistoryManager:
    def __init__(self, reward_cfg: DictConfig, num_envs: int):
        self.num_envs = num_envs
        self.history_buffers = self.setup_history_buffers(reward_cfg)

        self.history_keys = sorted(
            {
                history_key
                for history_buffer in self.history_buffers
                for history_key in history_buffer["history_keys"]
            }
        )

        self.history_entries: list[list[dict[str, Any]]] = [[] for _ in range(num_envs)]

        self.history_counts = [0 for _ in range(num_envs)]

    def setup_history_buffers(self, reward_cfg: DictConfig) -> list[dict[str, Any]]:
        history_buffers = reward_cfg.get("model", {}).get("history_buffers", None)
        if history_buffers is None:
            raise ValueError(
                "HistoryManager requires 'history_buffers' in YAML under reward.model.history_buffers."
            )

        history_buffers = [
            self.setup_history_buffer(history_buffer_name, history_buffer_cfg)
            for history_buffer_name, history_buffer_cfg in history_buffers.items()
        ]
        self.validate_history_buffers(history_buffers)

        self.max_history_size = max(
            history_buffer["history_size"] for history_buffer in history_buffers
        )
        return history_buffers

    def setup_history_buffer(
        self, history_buffer_name: str, history_buffer_cfg: dict[str, Any]
    ) -> dict[str, Any]:
        history_size = history_buffer_cfg.get("history_size")
        if not history_size:
            logging.warning(
                f"Using empty history buffer {history_buffer_name} with a 0 history_size as it's not defined."
            )
            history_size = 0

        min_history_size = history_buffer_cfg.get("min_history_size", 0)

        input_interval = history_buffer_cfg.get("input_interval")
        if not input_interval:
            logging.warning(
                f"Using empty history buffer {history_buffer_name} with a history_size={history_size} as it's not defined."
            )
            input_interval = max(history_size, 1)

        history_keys = history_buffer_cfg.get("history_keys")
        if not history_keys:
            raise ValueError(
                f"History buffer '{history_buffer_cfg}' doesn't define 'history_keys'."
            )

        input_on_done = history_buffer_cfg.get("input_on_done", False)
        input_on_done_full_window = history_buffer_cfg.get(
            "input_on_done_full_window", False
        )

        return {
            "name": history_buffer_name,
            "history_size": history_size,
            "min_history_size": min_history_size,
            "input_interval": input_interval,
            "history_keys": history_keys,
            "input_on_done": input_on_done,
            "input_on_done_full_window": input_on_done_full_window,
        }

    def validate_history_buffers(self, history_buffers: list[dict[str, Any]]) -> None:
        history_names = [history_buffer["name"] for history_buffer in history_buffers]
        history_name_set = set(history_names)
        if len(history_names) != len(history_name_set):
            raise ValueError(
                "History buffer names must be unique for proper extraction."
            )

    def append_to_history_entries(self, observations: dict[str, Any] | None) -> None:
        if observations is None:
            return
        for env_id in range(self.num_envs):
            history_entry = {}
            for history_key in self.history_keys:
                history_values = observations.get(history_key, None)
                if history_values is None:
                    continue
                history_entry[history_key] = clone_nested_to_cpu(history_values[env_id])
            self.history_entries[env_id].append(history_entry)
            self.history_counts[env_id] += 1

    def build_history_input(
        self, dones: torch.Tensor
    ) -> tuple[dict[str, Any], dict[str, list[int]]]:
        history_input: dict[str, dict[str, list[list]]] = {}
        history_length: dict[str, list[int]] = {}

        def append_to_history_input(
            history_buffer, history_range, env_idx: int
        ) -> None:
            history_buffer_name = history_buffer["name"]

            if history_buffer_name not in history_length:
                history_length[history_buffer_name] = [0 for _ in range(self.num_envs)]
            input_history_entries = self.history_entries[env_idx][history_range]
            history_length[history_buffer_name][env_idx] += len(input_history_entries)

            if history_buffer_name not in history_input:
                history_input[history_buffer_name] = {}
            for history_key in history_buffer["history_keys"]:
                if history_key not in history_input[history_buffer_name]:
                    history_input[history_buffer_name][history_key] = [
                        [] for _ in range(self.num_envs)
                    ]
                history_input[history_buffer_name][history_key][env_idx].extend(
                    [
                        entry[history_key]
                        for entry in input_history_entries
                        if history_key in entry
                    ]
                )

        if (dones.shape[0] != self.num_envs) or (dones.ndim != 1):
            raise ValueError(
                f"Expect the dones to have a shape of (self.num_envs,) = ({self.num_envs},), got {dones.shape}"
            )

        for env_idx, done in enumerate(dones):
            for history_buffer in self.history_buffers:
                if (
                    len(self.history_entries[env_idx])
                    < history_buffer["min_history_size"]
                ):
                    history_range = slice(0, 0)
                elif (
                    self.history_counts[env_idx] % history_buffer["input_interval"] == 0
                ):
                    history_range = slice(
                        max(
                            0,
                            len(self.history_entries[env_idx])
                            - history_buffer["history_size"],
                        ),
                        len(self.history_entries[env_idx]),
                    )
                elif done and history_buffer["input_on_done"]:
                    history_range = slice(
                        max(
                            0,
                            len(self.history_entries[env_idx])
                            - (
                                history_buffer["history_size"]
                                if history_buffer["input_on_done_full_window"]
                                else self.history_counts[env_idx]
                                % history_buffer["input_interval"]
                            ),
                        ),
                        len(self.history_entries[env_idx]),
                    )
                else:
                    continue
                append_to_history_input(history_buffer, history_range, env_idx)

            if done:
                self.clear_history(env_idx)
            else:
                self.trim_history(env_idx)

        return history_input, history_length

    def clear_history(self, env_id: int) -> None:
        self.history_entries[env_id].clear()
        self.history_counts[env_id] = 0

    def trim_history(self, env_idx: int) -> None:
        self.history_entries[env_idx] = self.history_entries[env_idx][
            -self.max_history_size :
        ]
