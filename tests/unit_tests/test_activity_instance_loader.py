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
"""Unit tests for ActivityInstanceLoader ID filtering + deterministic sampling.

Pure Python (no OmniGibson). Covers the behavior ported from the old repo:
- offline + multi-id ``activity_instance_id`` samples only from the requested
  subset (not from every discovered instance).
- ``from_omni_cfg(seed_offset=...)`` seeds a dedicated ``_rng`` so that the same
  seed produces the same sampling sequence.
- disabled + single id works; disabled + multi-id raises ``ValueError``.
"""

import random
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from rlinf.envs.behavior.instance_loader import (
    ActivityInstanceFile,
    ActivityInstanceLoader,
)


def _make_instance_files(ids):
    return [
        ActivityInstanceFile(
            instance_id=i,
            path=f"/tmp/instances/turning_on_radio_0_{i}_template-tro_state.json",
            file_format="tro_state",
        )
        for i in ids
    ]


def _make_omni_cfg(**task_overrides):
    """Build an OmegaConf dict with the fields ``from_omni_cfg`` reads."""
    task = {
        "activity_name": "turning_on_radio",
        "activity_definition_id": 0,
        "activity_instance_id": 1,
        "activity_instance_dir": "/tmp/instances",
        "instance_resample_mode": "offline",
        "instance_file_format": "tro_state",
        "online_object_sampling": False,
        "use_presampled_robot_pose": True,
    }
    task.update(task_overrides)
    return OmegaConf.create({"seed": 0, "task": task})


ALL_IDS = list(range(1, 11))  # instances 1..10


def _discover_stub(return_value):
    return patch(
        "rlinf.envs.behavior.instance_loader.discover_activity_instance_files",
        return_value=return_value,
    )


# ---------------------------------------------------------------------------
# offline + multi-id: requested-subset filtering
# ---------------------------------------------------------------------------


class TestOfflineRequestedSubset:
    def test_filters_to_requested_ids(self):
        with _discover_stub(_make_instance_files(ALL_IDS)):
            loader = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=[3, 5, 7],
                    instance_resample_mode="offline",
                )
            )
        loaded_ids = sorted(e.instance_id for e in loader.activity_instances)
        assert loaded_ids == [3, 5, 7]

    def test_sampling_with_replacement_stays_in_subset(self):
        with _discover_stub(_make_instance_files(ALL_IDS)):
            loader = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=[3, 5, 7],
                    instance_resample_mode="offline",
                )
            )
        # count > len(subset) forces with-replacement sampling.
        sampled = loader._sample_activity_instances(20)
        assert {e.instance_id for e in sampled} <= {3, 5, 7}

    def test_single_id_list_not_treated_as_subset_filter(self):
        # A single-element list normalizes to a scalar and, in offline mode with
        # no requested subset, keeps all discovered instances.
        with _discover_stub(_make_instance_files(ALL_IDS)):
            loader = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=[3],
                    instance_resample_mode="offline",
                )
            )
        assert loader.activity_instance_id == 3
        assert sorted(e.instance_id for e in loader.activity_instances) == ALL_IDS


# ---------------------------------------------------------------------------
# seed / seed_offset determinism
# ---------------------------------------------------------------------------


class TestSeedDeterminism:
    def test_same_seed_offset_same_rng_and_sequence(self):
        with _discover_stub(_make_instance_files(range(1, 21))):
            loader_a = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=list(range(1, 21)),
                    instance_resample_mode="offline",
                ),
                seed_offset=5,
            )
            loader_b = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=list(range(1, 21)),
                    instance_resample_mode="offline",
                ),
                seed_offset=5,
            )
        assert loader_a._rng.getstate() == loader_b._rng.getstate()
        seq_a = [e.instance_id for e in loader_a._sample_activity_instances(8)]
        seq_b = [e.instance_id for e in loader_b._sample_activity_instances(8)]
        assert seq_a == seq_b

    def test_different_seed_offset_different_rng(self):
        with _discover_stub(_make_instance_files(range(1, 21))):
            loader_a = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(instance_resample_mode="offline"),
                seed_offset=0,
            )
            loader_b = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(instance_resample_mode="offline"),
                seed_offset=1,
            )
        assert loader_a._rng.getstate() != loader_b._rng.getstate()

    def test_seed_derivation_matches_config_seed_plus_offset(self):
        cfg = _make_omni_cfg(
            activity_instance_id=list(range(1, 21)),
            instance_resample_mode="offline",
        )
        cfg.seed = 10
        with _discover_stub(_make_instance_files(range(1, 21))):
            loader = ActivityInstanceLoader.from_omni_cfg(cfg, seed_offset=5)
        assert loader._rng.getstate() == random.Random(15).getstate()


# ---------------------------------------------------------------------------
# disabled mode: single id works, multi-id raises
# ---------------------------------------------------------------------------


class TestDisabledMode:
    def test_single_id_ok(self):
        with _discover_stub(_make_instance_files(ALL_IDS)):
            loader = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=5,
                    instance_resample_mode="disabled",
                )
            )
        assert loader.activity_instance_id == 5
        # disabled mode keeps all discovered instances.
        assert sorted(e.instance_id for e in loader.activity_instances) == ALL_IDS

    def test_single_element_list_normalizes_to_int(self):
        with _discover_stub(_make_instance_files(ALL_IDS)):
            loader = ActivityInstanceLoader.from_omni_cfg(
                _make_omni_cfg(
                    activity_instance_id=[5],
                    instance_resample_mode="disabled",
                )
            )
        assert loader.activity_instance_id == 5

    def test_multi_id_raises(self):
        with _discover_stub(_make_instance_files(ALL_IDS)):
            with pytest.raises(ValueError, match="requires exactly one"):
                ActivityInstanceLoader.from_omni_cfg(
                    _make_omni_cfg(
                        activity_instance_id=[1, 2, 3],
                        instance_resample_mode="disabled",
                    )
                )

    def test_unknown_single_id_raises(self):
        with _discover_stub(_make_instance_files(ALL_IDS)):
            with pytest.raises(ValueError, match="not present"):
                ActivityInstanceLoader.from_omni_cfg(
                    _make_omni_cfg(
                        activity_instance_id=999,
                        instance_resample_mode="disabled",
                    )
                )


# ---------------------------------------------------------------------------
# tro_state scene bootstrap
# ---------------------------------------------------------------------------


class TestTroStateBootstrap:
    def test_bootstraps_scene_to_instance_zero(self):
        cfg = _make_omni_cfg(
            activity_instance_id=[3, 5, 7],
            instance_resample_mode="offline",
        )
        with _discover_stub(_make_instance_files(ALL_IDS)):
            loader = ActivityInstanceLoader.from_omni_cfg(cfg)
        # The OmniGibson scene must bootstrap from the seed template (instance 0)
        # while the loader retains the original requested instance id.
        assert cfg.task.activity_instance_id == 0
        assert loader.activity_instance_id == 3
