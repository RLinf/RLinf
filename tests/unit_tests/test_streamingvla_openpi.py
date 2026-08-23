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

"""OpenPI-dependent tests for isolated StreamingVLA transforms and loading."""

import importlib.util

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("openpi") is None,
    reason="StreamingVLA OpenPI dependencies are not installed.",
)


def test_local_quantile_normalization_is_reversible_and_isolated():
    """StreamingVLA's formula round-trips without patching OpenPI globally."""
    from openpi import transforms as openpi_transforms
    from openpi.shared.normalize import NormStats

    original_method = openpi_transforms.Normalize._normalize_quantile
    from rlinf.models.embodiment.streamingvla.transforms import (
        StreamingVLANormalize,
        StreamingVLAUnnormalize,
    )

    stats = NormStats(
        mean=np.zeros(2),
        std=np.ones(2),
        q01=np.array([-2.0, -4.0]),
        q99=np.array([3.0, 1.0]),
    )
    values = np.array([[1.5, -2.0, 7.0]], dtype=np.float32)
    normalized = StreamingVLANormalize({"actions": stats}, use_quantiles=True)(
        {"actions": values.copy()}
    )
    restored = StreamingVLAUnnormalize({"actions": stats}, use_quantiles=True)(
        normalized
    )

    expected = np.array([[0.5, -0.5, 7.0]], dtype=np.float32)
    np.testing.assert_allclose(normalized["actions"], expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(restored["actions"], values, rtol=1e-5, atol=1e-6)
    assert openpi_transforms.Normalize._normalize_quantile is original_method


def test_padding_and_libero_mapping_preserve_real_action_states():
    """The actual episode field is mapped and padded independently of metadata."""
    from openpi.models import model as openpi_model

    from rlinf.models.embodiment.streamingvla.transforms import (
        PadStreamingVLAStatesActions,
        StreamingVLALiberoInputs,
    )

    sample = {
        "observation/image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
        "observation/state": np.arange(7, dtype=np.float32),
        "observation/action_states": np.arange(7, dtype=np.float32) + 10,
        "actions": np.ones((10, 7), dtype=np.float32),
        "prompt": "pick up the object",
    }
    mapped = StreamingVLALiberoInputs(openpi_model.ModelType.PI0)(sample)
    padded = PadStreamingVLAStatesActions(model_action_dim=32)(mapped)

    np.testing.assert_array_equal(padded["action_states"][:7], np.arange(7) + 10)
    assert padded["action_states"].shape == (32,)
    assert padded["actions"].shape == (10, 32)
    assert padded["state"].shape == (32,)
    assert padded["image_mask"]["right_wrist_0_rgb"] == np.False_


def test_strict_safetensors_loader_accepts_base_and_wrapper_layouts(tmp_path):
    """Both supported layouts load strictly and mixed prefixes fail."""
    import safetensors.torch

    from rlinf.models.embodiment.streamingvla import load_streamingvla_checkpoint

    class _Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.svla_model = torch.nn.Linear(2, 2)

    source = _Wrapper()
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    safetensors.torch.save_file(
        source.svla_model.state_dict(), str(base_dir / "model.safetensors")
    )
    target = _Wrapper()
    load_streamingvla_checkpoint(target, base_dir)
    for expected, actual in zip(
        source.svla_model.parameters(), target.svla_model.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected)

    wrapper_dir = tmp_path / "wrapper"
    wrapper_dir.mkdir()
    safetensors.torch.save_file(
        source.state_dict(), str(wrapper_dir / "model.safetensors")
    )
    load_streamingvla_checkpoint(target, wrapper_dir)

    mixed_dir = tmp_path / "mixed"
    mixed_dir.mkdir()
    state = source.state_dict()
    mixed = {
        "svla_model.weight": state["svla_model.weight"],
        "bias": state["svla_model.bias"],
    }
    safetensors.torch.save_file(mixed, str(mixed_dir / "model.safetensors"))
    with pytest.raises(ValueError, match="mixes wrapper-prefixed"):
        load_streamingvla_checkpoint(target, mixed_dir)


def test_training_wrapper_explicitly_rejects_inference():
    """The PR cannot silently enter rollout or action-generation paths."""
    from rlinf.models.embodiment.streamingvla.streamingvla_action_model import (
        StreamingVLAForSFTActionPrediction,
    )

    wrapper = object.__new__(StreamingVLAForSFTActionPrediction)
    torch.nn.Module.__init__(wrapper)
    with pytest.raises(NotImplementedError, match="rollout"):
        wrapper.default_forward()
    with pytest.raises(NotImplementedError, match="inference"):
        wrapper.predict_action_batch()
