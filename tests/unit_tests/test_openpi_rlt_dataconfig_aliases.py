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

import dataclasses

import pytest

pytest.importorskip("torch")
pytest.importorskip("openpi")

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def _normalize_config(config):
    cfg_dict = dataclasses.asdict(config)
    cfg_dict.pop("name", None)
    return cfg_dict


def test_rlt_joint_legacy_alias_matches_canonical_config():
    canonical = get_openpi_config("pi05_rlt_joint")
    legacy = get_openpi_config("pi05_rlt_maniskill_joint")

    assert _normalize_config(canonical) == _normalize_config(legacy)
