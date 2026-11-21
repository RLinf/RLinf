# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Model-independent scheduling shared by RTC evaluation and Tianji Stage1."""

from collections import deque


class RTCChunkSchedule:
    """Request by executed horizon; latency estimates never shorten that horizon."""

    def __init__(self, min_exec_horizon, initial_delay_steps=1, delay_buffer_size=8):
        if min_exec_horizon < 1 or initial_delay_steps < 0 or delay_buffer_size < 1:
            raise ValueError("Invalid RTC execution horizon or delay buffer")
        self.min_exec_horizon = int(min_exec_horizon)
        self._delays = deque([int(initial_delay_steps)], maxlen=delay_buffer_size)

    def request_due(self, executed_horizon, *, pending=False):
        return not pending and executed_horizon >= self.min_exec_horizon

    def record_delay(self, observed_delay_steps):
        self._delays.append(max(int(observed_delay_steps), 0))

    @property
    def predicted_delay_steps(self):
        return max(self._delays)
