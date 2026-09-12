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

"""Environment attribute contracts seen by the env worker."""

import asyncio

import pytest

pytest.importorskip("mani_skill")

import torch  # noqa: E402

from rlinf.envs.sim.maniskill.maniskill_offload_env import (  # noqa: E402
    ManiskillOffloadEnv,
)
from rlinf.envs.utils import get_env_attr  # noqa: E402


def _bare_offload_env() -> ManiskillOffloadEnv:
    """Build the proxy without starting its simulator worker process."""
    return object.__new__(ManiskillOffloadEnv)


def test_optional_protocol_methods_resolve_locally():
    """get_env_attr probes must see the local protocol methods, not RPC proxies."""
    env = _bare_offload_env()

    wait_delay = get_env_attr(env, "wait_delay")
    assert callable(wait_delay)
    assert asyncio.iscoroutinefunction(wait_delay)

    insert_delay_metrics = get_env_attr(env, "insert_delay_metrics")
    metrics = insert_delay_metrics()
    assert isinstance(metrics, torch.Tensor)
    assert metrics.dtype == torch.float32
    assert metrics.numel() == 0
