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

import pickle

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.scheduler.manager.net_emulation import (
    CrossDCPair,
    NetEmulationConfig,
    NetEmulationManager,
)


def test_net_emulation_config_parses_legacy_crossdc_pairs():
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
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


def test_net_emulation_config_endpoint_ranges_equal_explicit_lists():
    def build_cfg(src, dst, members):
        return OmegaConf.create(
            {
                "enabled": True,
                "symmetric": True,
                "crossdc_pairs": [
                    {"src": src, "dst": dst, "delay_ms": 10},
                ],
                "bandwidth_groups": [
                    {"members": members, "bandwidth_mbps": 1000},
                ],
            }
        )

    explicit_cfg = build_cfg(
        ["Env:0", "Env:1"],
        ["Actor:0", "Actor:1"],
        ["Env:0", "Env:1", "Actor:0", "Actor:1"],
    )
    range_cfg = build_cfg(["Env:0-1"], ["Actor:0-1"], ["Env:0-1", "Actor:0-1"])

    assert NetEmulationConfig.from_cfg(range_cfg) == NetEmulationConfig.from_cfg(
        explicit_cfg
    )


@pytest.mark.parametrize("field_name", ["src", "dst"])
def test_net_emulation_config_rejects_empty_crossdc_pair_endpoint_lists(field_name):
    cfg = OmegaConf.create(
        {
            "enabled": True,
            "symmetric": True,
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


def test_net_emulation_config_disabled_returns_none():
    cfg = OmegaConf.create({"enabled": False, "crossdc_pairs": []})

    assert NetEmulationConfig.from_cfg(cfg) is None
    assert NetEmulationConfig.from_cfg(None) is None


def _manager(**overrides):
    """Build a NetEmulationManager directly, bypassing the Ray actor launch."""
    cfg = {
        "enabled": True,
        "symmetric": True,
        "crossdc_pairs": [
            {"src": "EnvGroup:0", "dst": "ActorGroup:0", "delay_ms": 100}
        ],
        "bandwidth_groups": [],
    }
    cfg.update(overrides)
    return NetEmulationManager(cfg)


def test_reserve_returns_zero_for_unemulated_links():
    manager = _manager()

    assert manager.reserve("EnvGroup:0", "RolloutGroup:0", 1024) == 0.0


def test_reserve_applies_link_delay_in_both_directions():
    manager = _manager()

    assert manager.reserve("EnvGroup:0", "ActorGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )
    # symmetric: true mirrors every configured pair.
    assert manager.reserve("ActorGroup:0", "EnvGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )


def test_reserve_ignores_group_suffix_in_endpoint_names():
    """Endpoints may be written with or without the trailing ``Group``."""
    manager = _manager(
        crossdc_pairs=[{"src": "Env:0", "dst": "Actor:0", "delay_ms": 100}]
    )

    assert manager.reserve("EnvGroup:0", "ActorGroup:0", 0) == pytest.approx(
        0.1, abs=0.02
    )


def test_reserve_charges_transfer_time_against_the_bandwidth_budget():
    # 8 Mbps == 1 MB/s, so a 1 MB payload occupies the link for one second.
    manager = _manager(
        bandwidth_groups=[
            {"members": ["EnvGroup:0"], "bandwidth_mbps": 8},
            {"members": ["ActorGroup:0"], "bandwidth_mbps": 8},
        ]
    )

    one_mb = 1_000_000
    assert manager.reserve("EnvGroup:0", "ActorGroup:0", one_mb) == pytest.approx(
        1.1, abs=0.02
    )
    # The uplink is still busy, so a second send queues behind the first.
    assert manager.reserve("EnvGroup:0", "ActorGroup:0", one_mb) == pytest.approx(
        2.1, abs=0.02
    )


def test_estimate_payload_size_counts_tensor_storage():
    tensor = torch.zeros(256, dtype=torch.float32)  # 1024 bytes of data

    assert NetEmulationManager.estimate_payload_size_bytes(tensor) == 1024
    assert NetEmulationManager.estimate_payload_size_bytes(None) == 0


def test_estimate_payload_size_walks_nested_containers():
    payload = {"a": torch.zeros(256, dtype=torch.float32), "b": [torch.zeros(256)]}

    size = NetEmulationManager.estimate_payload_size_bytes(payload)

    # Two tensors of 1024 bytes each, plus per-tensor overhead and pickled keys.
    assert size > 2048
    assert size < 2048 + 4 * 256


def test_estimate_payload_size_falls_back_to_pickle_for_plain_objects():
    payload = {"task": "pick up the cube", "step": 7}

    size = NetEmulationManager.estimate_payload_size_bytes(payload)

    assert size == len(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))


def test_reserve_broadcast_charges_the_uplink_once():
    # 8 Mbps == 1 MB/s on each side.
    manager = _manager(
        crossdc_pairs=[
            {"src": "Env:0", "dst": ["Actor:0", "Actor:1"], "delay_ms": 100}
        ],
        bandwidth_groups=[
            {"members": ["Env:0"], "bandwidth_mbps": 8},
            {"members": ["Actor:0", "Actor:1"], "bandwidth_mbps": 8},
        ],
    )

    one_mb = 1_000_000
    # Both receivers share one bandwidth group, so the payload crosses the link
    # once: 1s uplink + 0.1s delay, not 2s.
    assert manager.reserve_broadcast(
        "EnvGroup:0", ["ActorGroup:0", "ActorGroup:1"], one_mb
    ) == pytest.approx(1.1, abs=0.02)


def test_reserve_broadcast_waits_for_the_slowest_destination():
    manager = _manager(
        crossdc_pairs=[
            {"src": "Env:0", "dst": "Actor:0", "delay_ms": 50},
            {"src": "Env:0", "dst": "Actor:1", "delay_ms": 200},
        ],
    )

    assert manager.reserve_broadcast(
        "EnvGroup:0", ["ActorGroup:0", "ActorGroup:1"], 0
    ) == pytest.approx(0.2, abs=0.02)


def test_reserve_broadcast_ignores_unemulated_destinations():
    manager = _manager()

    assert (
        manager.reserve_broadcast("EnvGroup:0", ["RolloutGroup:0", "EnvGroup:1"], 4096)
        == 0.0
    )
