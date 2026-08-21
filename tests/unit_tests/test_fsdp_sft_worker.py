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

from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


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
