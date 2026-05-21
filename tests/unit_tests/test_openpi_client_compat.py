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

import numpy as np
import pytest

from openpi_client import action_chunk_broker, msgpack_numpy


def test_msgpack_numpy_round_trip_array():
    payload = {"image": np.arange(6, dtype=np.uint8).reshape(2, 3)}

    restored = msgpack_numpy.unpackb(msgpack_numpy.packb(payload))

    np.testing.assert_array_equal(restored["image"], payload["image"])


def test_action_chunk_broker_reuses_chunk_until_horizon():
    class Policy:
        def __init__(self):
            self.calls = 0

        def infer(self, obs):
            self.calls += 1
            return {
                "actions": np.array(
                    [[self.calls, 0], [self.calls, 1], [self.calls, 2]]
                ),
                "metadata": obs["metadata"],
            }

    policy = Policy()
    broker = action_chunk_broker.ActionChunkBroker(policy, action_horizon=2)

    first = broker.infer({"metadata": "chunk"})
    second = broker.infer({"metadata": "ignored"})
    third = broker.infer({"metadata": "next"})

    np.testing.assert_array_equal(first["actions"], np.array([1, 0]))
    np.testing.assert_array_equal(second["actions"], np.array([1, 1]))
    np.testing.assert_array_equal(third["actions"], np.array([2, 0]))
    assert first["metadata"] == "chunk"
    assert second["metadata"] == "chunk"
    assert third["metadata"] == "next"
    assert policy.calls == 2


def test_action_chunk_broker_rejects_non_positive_horizon():
    with pytest.raises(ValueError, match="action_horizon"):
        action_chunk_broker.ActionChunkBroker(object(), action_horizon=0)
