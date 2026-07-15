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

from rlinf.workers.env.env_worker import EnvWorker


def _make_worker(delay_sampler=None, enable_decoupled=True):
    worker = EnvWorker.__new__(EnvWorker)
    cfg = {"env": {}}
    if delay_sampler is not None:
        cfg["env"]["delay_sampler"] = delay_sampler
    worker.cfg = OmegaConf.create(cfg)
    # _use_delayed_per_env_send checks delay_sampler + env_decoupled_mode
    worker.env_decoupled_mode = enable_decoupled
    worker.delay_sampler = None
    return worker


def test_delay_disabled_when_delay_sampler_is_none():
    worker = _make_worker(delay_sampler=None, enable_decoupled=True)

    assert not worker._use_delayed_per_env_send(mode="train")


def test_delay_disabled_when_not_decoupled():
    worker = _make_worker(
        delay_sampler={"type": "uniform", "min_delay": 0.1, "max_delay": 0.2},
        enable_decoupled=False,
    )

    assert not worker._use_delayed_per_env_send(mode="train")


def test_delay_enabled_with_sampler_and_decoupled():
    worker = _make_worker(
        delay_sampler={"type": "uniform", "min_delay": 0.1, "max_delay": 0.2},
        enable_decoupled=True,
    )
    from rlinf.utils.delay_sampler import DelaySampler

    worker.delay_sampler = DelaySampler.create(worker.cfg.env.get("delay_sampler"))

    assert worker._use_delayed_per_env_send(mode="train")
    assert not worker._use_delayed_per_env_send(mode="eval")
