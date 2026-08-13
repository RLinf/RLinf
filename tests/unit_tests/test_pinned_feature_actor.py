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

import pytest
import torch

from rlinf.utils.backbone_cache import ROLLOUT_BACKBONE_SAMPLE_IDS_KEY
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


def test_actor_training_exception_clears_cache_and_sample_ids():
    class Cache:
        cleared = False

        def clear(self):
            self.cleared = True

    class Actor:
        _clear_pinned_rollout_backbone_cache = (
            EmbodiedFSDPActor._clear_pinned_rollout_backbone_cache
        )

        def __init__(self):
            self._pinned_rollout_backbone_cache = Cache()
            self._pinned_backbone_metadata = {"lease_id": "lease-1"}
            self.rollout_batch = {
                "forward_inputs": {
                    ROLLOUT_BACKBONE_SAMPLE_IDS_KEY: torch.tensor([0, 1])
                }
            }

        @staticmethod
        def _run_training_impl():
            raise RuntimeError("injected training failure")

    actor = Actor()
    cache = actor._pinned_rollout_backbone_cache

    with pytest.raises(RuntimeError, match="injected training failure"):
        EmbodiedFSDPActor.run_training(actor)

    assert cache.cleared
    assert actor._pinned_rollout_backbone_cache is None
    assert actor._pinned_backbone_metadata is None
    assert ROLLOUT_BACKBONE_SAMPLE_IDS_KEY not in actor.rollout_batch["forward_inputs"]
