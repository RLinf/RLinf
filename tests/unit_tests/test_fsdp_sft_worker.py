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

from unittest.mock import Mock

from omegaconf import OmegaConf

from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


def _worker(total_training_steps=None, *, max_epochs=-1, max_steps=-1, steps=1):
    worker = object.__new__(FSDPSftWorker)
    worker._cfg = OmegaConf.create(
        {"optim": {"total_training_steps": total_training_steps}}
    )
    worker.cfg = OmegaConf.create(
        {"runner": {"max_epochs": max_epochs, "max_steps": max_steps}}
    )
    worker._logger = Mock()
    worker.get_max_steps_per_epoch = Mock(return_value=steps)
    return worker


def test_sft_worker_infers_horizon_from_the_smallest_runner_limit() -> None:
    worker = _worker(max_epochs=3, max_steps=10, steps=4)

    worker._set_total_training_steps_if_unset()

    assert worker._cfg.optim.total_training_steps == 10
    worker._logger.info.assert_called_once()


def test_sft_worker_preserves_an_explicit_scheduler_horizon() -> None:
    worker = _worker(total_training_steps=99, max_epochs=3, max_steps=10, steps=4)

    worker._set_total_training_steps_if_unset()

    assert worker._cfg.optim.total_training_steps == 99
    worker._logger.info.assert_not_called()



def test_sft_worker_keeps_legacy_floor_epoch_length() -> None:
    class DataLoader:
        drop_last = False

        def __len__(self) -> int:
            return 5

    worker = _worker()
    worker.data_loader = DataLoader()
    worker.gradient_accumulation = 2

    assert worker.get_max_steps_per_epoch() == 2
