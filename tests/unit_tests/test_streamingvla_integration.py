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

"""Opt-in integration tests for external StreamingVLA assets."""

import dataclasses
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.integration


def _external_path(variable: str) -> Path | None:
    value = os.environ.get(variable)
    return Path(value).expanduser() if value else None


@pytest.mark.skipif(
    _external_path("STREAMINGVLA_LIBERO_DATASET") is None,
    reason="Set STREAMINGVLA_LIBERO_DATASET to run the real-record test.",
)
def test_real_lerobot_record_supplies_action_states():
    """Read action states through the production loader, not dataset metadata."""
    from rlinf.models.embodiment.streamingvla.data import (
        create_streamingvla_data_loader,
    )
    from rlinf.models.embodiment.streamingvla.dataconfig import (
        get_streamingvla_config,
    )

    dataset_path = _external_path("STREAMINGVLA_LIBERO_DATASET")
    assert dataset_path is not None
    config = get_streamingvla_config(
        "pi05_libero_sfp",
        batch_size=2,
        data_kwargs={
            "repo_id": str(dataset_path),
            "assets": {
                "assets_dir": str(dataset_path.parent),
                "asset_id": dataset_path.name,
            },
            "extra_delta_transform": False,
            "action_env_dim": 7,
            "use_action_states": True,
        },
        seed=42,
    )
    config = dataclasses.replace(config, num_workers=0)
    observation, actions = next(
        iter(create_streamingvla_data_loader(config, shuffle=False))
    )

    assert observation.action_states is not None
    assert observation.action_states.shape == (2, 32)
    assert actions.shape == (2, 10, 32)


@pytest.mark.skipif(
    _external_path("STREAMINGVLA_BASE_CHECKPOINT") is None,
    reason="Set STREAMINGVLA_BASE_CHECKPOINT to run the strict-load test.",
)
def test_real_pi05_base_checkpoint_loads_strictly():
    """Construct the full model and verify one tensor after strict loading."""
    from omegaconf import OmegaConf
    from safetensors import safe_open

    from rlinf.models.embodiment.streamingvla import get_model

    checkpoint_path = _external_path("STREAMINGVLA_BASE_CHECKPOINT")
    assert checkpoint_path is not None
    model_config = OmegaConf.create(
        {
            "model_type": "streamingvla",
            "model_path": str(checkpoint_path),
            "precision": "bf16",
            "streamingvla": {
                "config_name": "pi05_libero_sfp",
                "use_sfp": True,
                "use_action_states": True,
                "action_horizon": 10,
                "model_action_dim": 32,
                "sigma": 0.16,
                "noise_decay": 4.0,
            },
        }
    )
    model = get_model(model_config, torch.bfloat16)

    weights_path = checkpoint_path / "model.safetensors"
    with safe_open(str(weights_path), framework="pt", device="cpu") as weights:
        expected = weights.get_tensor("action_in_proj.weight")
    actual = model.svla_model.action_in_proj.weight.detach().cpu()
    torch.testing.assert_close(
        actual, expected.to(dtype=actual.dtype), rtol=0.0, atol=0.0
    )
