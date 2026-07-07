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

from omegaconf import OmegaConf

from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _Placement:
    def __init__(self, sizes):
        self._sizes = sizes

    def get_world_size(self, component):
        return self._sizes[component]


def _make_worker(delay_sampler=None):
    worker = MultiStepRolloutWorker.__new__(MultiStepRolloutWorker)
    worker.cfg = OmegaConf.create({"env": {"delay_sampler": delay_sampler}})
    worker.placement = _Placement({"rollout": 2, "env": 2})
    worker.rollout_queue_size = 0
    return worker


def test_delay_sampler_mode_rollout_receives_one_env_per_step():
    worker = _make_worker(
        delay_sampler={"type": "uniform", "min_delay": 0.1, "max_delay": 0.2}
    )

    assert worker._decoupled_env_mode_setup_batch_size(batch_size=16) == [1]


def test_default_decoupled_mapping_is_unchanged_without_delay_sampler():
    worker = _make_worker(delay_sampler=None)

    assert worker._decoupled_env_mode_setup_batch_size(batch_size=16) == [8]
