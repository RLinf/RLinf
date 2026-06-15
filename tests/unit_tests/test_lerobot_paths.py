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

from rlinf.data.lerobot_paths import resolve_lerobot_repo_id


def test_resolve_lerobot_repo_id_accepts_omegaconf_listconfig():
    data_paths = OmegaConf.create(
        [
            {
                "dataset_path": "/tmp/local_lerobot/id_4",
                "weight": 1.0,
            }
        ]
    )

    assert resolve_lerobot_repo_id(data_paths) == "/tmp/local_lerobot/id_4"
