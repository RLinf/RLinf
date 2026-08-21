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

from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


def test_sft_worker_keeps_legacy_floor_epoch_length() -> None:
    class DataLoader:
        drop_last = False

        def __len__(self) -> int:
            return 5

    worker = object.__new__(FSDPSftWorker)
    worker.data_loader = DataLoader()
    worker.gradient_accumulation = 2

    # Incomplete accumulation windows remain dropped. FastWAM's released
    # recipes use gradient_accumulation=1, where floor and ceil are identical.
    assert worker.get_max_steps_per_epoch() == 2


def _fastwam_worker(total_training_steps=None, max_steps=-1, model_type="fastwam"):
    class DataLoader:
        def __len__(self) -> int:
            return 2170

    worker = object.__new__(FSDPVlaSftWorker)
    worker.cfg = OmegaConf.create(
        {
            "runner": {"max_epochs": 10, "max_steps": max_steps},
            "actor": {
                "model": {"model_type": model_type},
                "optim": {"total_training_steps": total_training_steps},
            },
        }
    )
    OmegaConf.set_struct(worker.cfg, True)
    worker.data_loader = DataLoader()
    worker.gradient_accumulation = 1
    return worker


def test_fastwam_derives_missing_total_training_steps() -> None:
    worker = _fastwam_worker()

    worker._set_fastwam_total_training_steps_if_missing()

    assert worker.cfg.actor.optim.total_training_steps == 21700


def test_fastwam_total_training_steps_honors_runner_cap() -> None:
    worker = _fastwam_worker(max_steps=1)

    worker._set_fastwam_total_training_steps_if_missing()

    assert worker.cfg.actor.optim.total_training_steps == 1


def test_fastwam_keeps_explicit_total_training_steps() -> None:
    worker = _fastwam_worker(total_training_steps=123)

    worker._set_fastwam_total_training_steps_if_missing()

    assert worker.cfg.actor.optim.total_training_steps == 123


def test_fastwam_total_training_steps_does_not_change_other_models() -> None:
    worker = _fastwam_worker(model_type="openpi")

    worker._set_fastwam_total_training_steps_if_missing()

    assert worker.cfg.actor.optim.total_training_steps is None
