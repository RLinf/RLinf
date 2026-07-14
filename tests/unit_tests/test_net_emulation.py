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

import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.collective.net_emulation import CrossDCPair, NetEmulationConfig


def test_net_emulation_config_parses_legacy_crossdc_pairs():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "proxy": {"node_rank": 0, "num_cpus": 1, "name": "proxy"},
            "crossdc_pairs": [
                {"src": "Env:0", "dst": "Actor:0", "delay_ms": 10},
            ],
            "bandwidth_groups": [],
        }
    )

    net_cfg = NetEmulationConfig.from_cfg(cfg)

    assert net_cfg is not None
    assert net_cfg.crossdc_pairs == (
        CrossDCPair(src="Env:0", dst="Actor:0", delay_ms=10.0),
    )


def test_net_emulation_config_expands_crossdc_pair_endpoint_lists():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "proxy": {"node_rank": 0, "num_cpus": 1, "name": "proxy"},
            "crossdc_pairs": [
                {
                    "src": ["Env:0", "Env:1"],
                    "dst": ["Actor:0", "Actor:1"],
                    "delay_ms": 10,
                },
            ],
            "bandwidth_groups": [],
        }
    )

    net_cfg = NetEmulationConfig.from_cfg(cfg)

    assert net_cfg is not None
    assert [(pair.src, pair.dst, pair.delay_ms) for pair in net_cfg.crossdc_pairs] == [
        ("Env:0", "Actor:0", 10.0),
        ("Env:0", "Actor:1", 10.0),
        ("Env:1", "Actor:0", 10.0),
        ("Env:1", "Actor:1", 10.0),
    ]


@pytest.mark.parametrize("field_name", ["src", "dst"])
def test_net_emulation_config_rejects_empty_crossdc_pair_endpoint_lists(field_name):
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
            "proxy": {"node_rank": 0, "num_cpus": 1, "name": "proxy"},
            "crossdc_pairs": [
                {
                    "src": ["Env:0"],
                    "dst": ["Actor:0"],
                    "delay_ms": 10,
                },
            ],
            "bandwidth_groups": [],
        }
    )
    cfg.crossdc_pairs[0][field_name] = []

    with pytest.raises(ValueError, match=field_name):
        NetEmulationConfig.from_cfg(cfg)
