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


import torch

from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.utils import _build_channel_message
from rlinf.workers.env.env_worker import EnvWorker


class _DummyChannel:
    def __init__(self):
        self.items = []

    def put(self, item, key, async_op=False):
        del async_op
        self.items.append((item, key))


def test_send_env_batch_shard_to_channel_sends_presplit_batch():
    worker = EnvWorker.__new__(EnvWorker)
    worker._rank = 2
    channel = _DummyChannel()
    env_batch = {
        "obs": {
            "main_images": torch.zeros((1, 3, 4), dtype=torch.float32),
            "task_descriptions": ["task-0"],
        },
        "final_obs": None,
    }

    worker.send_env_batch_shard_to_channel(
        channel,
        env_batch,
        batch_idx=5,
        mode="train",
        last_run=True,
    )

    assert len(channel.items) == 1
    item, key = channel.items[0]
    assert key == CommMapper.build_channel_key(None, None, extra="train_obs")
    assert item["batch"] is env_batch
    assert item["batch_index"] == _build_channel_message(2, 5, "train", True, "obs")
