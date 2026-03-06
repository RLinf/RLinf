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

"""Tests for value_policy_config module.

Covers:
- _load_state_dict_from_checkpoint  (file/dir, safetensors/pt, missing)
- _has_tokenizer_files              (presence/absence of tokenizer files)
- load_norm_stats                   (loading, missing files, nested keys)
- _build_input_transforms           (valid/invalid env_type)
- ValuePolicy construction
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Heavy dependencies -- skip the entire module when unavailable.
torch = pytest.importorskip("torch")
safetensors_torch = pytest.importorskip("safetensors.torch")

# The target module depends on vla_lib which may not be installed.
# Use try/except to skip gracefully.
try:
    from rlinf.models.embodiment.vla_lib_value_model.value_policy_config import (
        _has_tokenizer_files,
        _load_state_dict_from_checkpoint,
        load_norm_stats,
    )
except ImportError:
    pytest.skip(
        "vla_lib dependencies not available",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# _has_tokenizer_files
# ---------------------------------------------------------------------------


class TestHasTokenizerFiles:
    """Test detection of tokenizer files in a checkpoint directory."""

    def test_has_tokenizer_json(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        assert _has_tokenizer_files(tmp_path) is True

    def test_has_tokenizer_config_json(self, tmp_path):
        (tmp_path / "tokenizer_config.json").write_text("{}")
        assert _has_tokenizer_files(tmp_path) is True

    def test_has_special_tokens_map(self, tmp_path):
        (tmp_path / "special_tokens_map.json").write_text("{}")
        assert _has_tokenizer_files(tmp_path) is True

    def test_no_tokenizer_files(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        assert _has_tokenizer_files(tmp_path) is False

    def test_empty_directory(self, tmp_path):
        assert _has_tokenizer_files(tmp_path) is False

    def test_multiple_tokenizer_files(self, tmp_path):
        (tmp_path / "tokenizer.json").write_text("{}")
        (tmp_path / "tokenizer_config.json").write_text("{}")
        (tmp_path / "special_tokens_map.json").write_text("{}")
        assert _has_tokenizer_files(tmp_path) is True


# ---------------------------------------------------------------------------
# _load_state_dict_from_checkpoint -- safetensors
# ---------------------------------------------------------------------------


class TestLoadStateDictSafetensors:
    """Test _load_state_dict_from_checkpoint with .safetensors files."""

    def test_load_single_safetensors_file(self, tmp_path):
        weights = {"layer.weight": torch.randn(4, 4)}
        file_path = tmp_path / "model.safetensors"
        safetensors_torch.save_file(weights, str(file_path))

        result = _load_state_dict_from_checkpoint(file_path)
        assert "layer.weight" in result
        assert torch.allclose(result["layer.weight"], weights["layer.weight"])

    def test_load_safetensors_directory(self, tmp_path):
        w1 = {"a.weight": torch.randn(2, 2)}
        w2 = {"b.weight": torch.randn(3, 3)}
        safetensors_torch.save_file(w1, str(tmp_path / "shard-00001.safetensors"))
        safetensors_torch.save_file(w2, str(tmp_path / "shard-00002.safetensors"))

        result = _load_state_dict_from_checkpoint(tmp_path)
        assert "a.weight" in result
        assert "b.weight" in result
        assert torch.allclose(result["a.weight"], w1["a.weight"])
        assert torch.allclose(result["b.weight"], w2["b.weight"])

    def test_load_safetensors_directory_merges_all_shards(self, tmp_path):
        tensors_per_shard = 3
        all_keys = set()
        for i in range(tensors_per_shard):
            d = {f"layer_{i}.weight": torch.randn(2, 2)}
            safetensors_torch.save_file(
                d, str(tmp_path / f"shard-{i:05d}.safetensors")
            )
            all_keys.update(d.keys())

        result = _load_state_dict_from_checkpoint(tmp_path)
        assert set(result.keys()) == all_keys


# ---------------------------------------------------------------------------
# _load_state_dict_from_checkpoint -- pt/pth
# ---------------------------------------------------------------------------


class TestLoadStateDictPt:
    """Test _load_state_dict_from_checkpoint with .pt/.pth files."""

    def test_load_single_pt_file(self, tmp_path):
        weights = {"fc.weight": torch.randn(5, 5)}
        file_path = tmp_path / "model.pt"
        torch.save(weights, str(file_path))

        result = _load_state_dict_from_checkpoint(file_path)
        assert "fc.weight" in result
        assert torch.allclose(result["fc.weight"], weights["fc.weight"])

    def test_load_single_pth_file(self, tmp_path):
        weights = {"fc.bias": torch.randn(5)}
        file_path = tmp_path / "model.pth"
        torch.save(weights, str(file_path))

        result = _load_state_dict_from_checkpoint(file_path)
        assert "fc.bias" in result

    def test_load_pt_directory(self, tmp_path):
        w1 = {"x.w": torch.randn(2, 2)}
        w2 = {"y.w": torch.randn(3, 3)}
        torch.save(w1, str(tmp_path / "part1.pt"))
        torch.save(w2, str(tmp_path / "part2.pt"))

        result = _load_state_dict_from_checkpoint(tmp_path)
        assert "x.w" in result
        assert "y.w" in result

    def test_load_pth_directory(self, tmp_path):
        w = {"z.w": torch.randn(4)}
        torch.save(w, str(tmp_path / "weights.pth"))

        result = _load_state_dict_from_checkpoint(tmp_path)
        assert "z.w" in result


# ---------------------------------------------------------------------------
# _load_state_dict_from_checkpoint -- priority and errors
# ---------------------------------------------------------------------------


class TestLoadStateDictPriority:
    """Test priority of safetensors over pt and error conditions."""

    def test_safetensors_preferred_over_pt_in_directory(self, tmp_path):
        """When both safetensors and pt exist, safetensors should be loaded."""
        st_weights = {"from_safetensors": torch.randn(2, 2)}
        pt_weights = {"from_pt": torch.randn(2, 2)}
        safetensors_torch.save_file(st_weights, str(tmp_path / "model.safetensors"))
        torch.save(pt_weights, str(tmp_path / "model.pt"))

        result = _load_state_dict_from_checkpoint(tmp_path)
        # Safetensors takes precedence
        assert "from_safetensors" in result
        assert "from_pt" not in result

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
            _load_state_dict_from_checkpoint(tmp_path)

    def test_directory_with_unrelated_files_raises(self, tmp_path):
        (tmp_path / "readme.txt").write_text("hello")
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="No checkpoint files found"):
            _load_state_dict_from_checkpoint(tmp_path)

    def test_nonexistent_path_raises(self, tmp_path):
        fake_path = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError):
            _load_state_dict_from_checkpoint(fake_path)


# ---------------------------------------------------------------------------
# load_norm_stats -- basic loading
# ---------------------------------------------------------------------------


def _make_norm_stats_json(
    keys=("state", "action"),
    with_quantiles=True,
    with_minmax=False,
    wrap_in_norm_stats_key=False,
):
    """Helper to create a norm_stats.json payload."""
    data = {}
    for k in keys:
        entry = {
            "mean": [0.0, 1.0, 2.0],
            "std": [1.0, 1.0, 1.0],
            "q01": [0.01, 0.01, 0.01] if with_quantiles else None,
            "q99": [0.99, 0.99, 0.99] if with_quantiles else None,
            "min": [-1.0, -1.0, -1.0] if with_minmax else None,
            "max": [1.0, 1.0, 1.0] if with_minmax else None,
        }
        data[k] = entry
    if wrap_in_norm_stats_key:
        data = {"norm_stats": data}
    return json.dumps(data)


# We need to mock the NormStats import since vla_lib may not be installed.
# The function creates NormStats objects, so we provide a lightweight stand-in
# when the real one is available.
_norm_stats_available = True
try:
    from rlinf.datasets.vla_lib.lerobot_datasets.normalize import NormStats
except ImportError:
    _norm_stats_available = False


@pytest.mark.skipif(not _norm_stats_available, reason="vla_lib not installed")
class TestLoadNormStats:
    """Test load_norm_stats from various directory layouts."""

    def test_load_from_norm_stats_subdir(self, tmp_path):
        """Standard path: checkpoint_dir/norm_stats/{asset_id}/norm_stats.json."""
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(keys=("state", "action"))
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert "state" in result
        assert "action" in result
        np.testing.assert_array_equal(result["state"].mean, [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(result["state"].std, [1.0, 1.0, 1.0])

    def test_load_from_stats_subdir(self, tmp_path):
        """Alternative path: checkpoint_dir/stats/{asset_id}/norm_stats.json."""
        stats_dir = tmp_path / "stats" / "droid"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(keys=("state",))
        )

        result = load_norm_stats(tmp_path, asset_id="droid")
        assert "state" in result

    def test_load_from_root_norm_stats(self, tmp_path):
        """Fallback path: checkpoint_dir/norm_stats.json."""
        (tmp_path / "norm_stats.json").write_text(
            _make_norm_stats_json(keys=("state",))
        )

        result = load_norm_stats(tmp_path, asset_id="anything")
        assert "state" in result

    def test_load_wrapped_in_norm_stats_key(self, tmp_path):
        """JSON with outer 'norm_stats' key is unwrapped correctly."""
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state",), wrap_in_norm_stats_key=True
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert "state" in result

    def test_quantile_fields_loaded(self, tmp_path):
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state",), with_quantiles=True
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert result["state"].q01 is not None
        assert result["state"].q99 is not None
        np.testing.assert_allclose(result["state"].q01, [0.01, 0.01, 0.01])
        np.testing.assert_allclose(result["state"].q99, [0.99, 0.99, 0.99])

    def test_missing_quantile_fields_are_none(self, tmp_path):
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state",), with_quantiles=False
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert result["state"].q01 is None
        assert result["state"].q99 is None

    def test_minmax_fields_loaded(self, tmp_path):
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state",), with_minmax=True
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert result["state"].min is not None
        assert result["state"].max is not None
        np.testing.assert_array_equal(result["state"].min, [-1.0, -1.0, -1.0])
        np.testing.assert_array_equal(result["state"].max, [1.0, 1.0, 1.0])

    def test_missing_minmax_fields_are_none(self, tmp_path):
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state",), with_minmax=False
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert result["state"].min is None
        assert result["state"].max is None

    def test_result_values_are_float32(self, tmp_path):
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state",), with_quantiles=True, with_minmax=True
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert result["state"].mean.dtype == np.float32
        assert result["state"].std.dtype == np.float32
        assert result["state"].q01.dtype == np.float32
        assert result["state"].q99.dtype == np.float32
        assert result["state"].min.dtype == np.float32
        assert result["state"].max.dtype == np.float32

    def test_auto_discovery_from_norm_stats_subdir(self, tmp_path):
        """When asset_id does not match but a subdirectory exists in norm_stats/."""
        # Create norm_stats/custom_env/norm_stats.json
        custom_dir = tmp_path / "norm_stats" / "custom_env"
        custom_dir.mkdir(parents=True)
        (custom_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(keys=("state",))
        )

        # asset_id is "libero" but only "custom_env" subdir exists --
        # the auto-discovery logic adds it as a fallback path.
        result = load_norm_stats(tmp_path, asset_id="libero")
        assert "state" in result

    def test_multiple_keys_in_norm_stats(self, tmp_path):
        stats_dir = tmp_path / "norm_stats" / "libero"
        stats_dir.mkdir(parents=True)
        (stats_dir / "norm_stats.json").write_text(
            _make_norm_stats_json(
                keys=("state", "action", "gripper"),
            )
        )

        result = load_norm_stats(tmp_path, asset_id="libero")
        assert len(result) == 3
        assert set(result.keys()) == {"state", "action", "gripper"}


# ---------------------------------------------------------------------------
# load_norm_stats -- missing files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _norm_stats_available, reason="vla_lib not installed")
class TestLoadNormStatsMissing:
    """Test error handling when norm_stats.json is not found."""

    def test_empty_dir_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Could not find norm_stats.json"):
            load_norm_stats(tmp_path, asset_id="libero")

    def test_wrong_asset_id_no_fallback_raises(self, tmp_path):
        """If the specific asset subdir does not exist and no fallbacks, raise."""
        # Create a norm_stats dir but with no subdirectories and no root file
        (tmp_path / "norm_stats").mkdir()
        with pytest.raises(FileNotFoundError, match="Could not find norm_stats.json"):
            load_norm_stats(tmp_path, asset_id="nonexistent")

    def test_dir_with_unrelated_json_raises(self, tmp_path):
        (tmp_path / "config.json").write_text("{}")
        with pytest.raises(FileNotFoundError, match="Could not find norm_stats.json"):
            load_norm_stats(tmp_path, asset_id="libero")


# ---------------------------------------------------------------------------
# _build_input_transforms
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _norm_stats_available, reason="vla_lib not installed")
class TestBuildInputTransforms:
    """Test _build_input_transforms for env_type validation and transform pipeline."""

    @pytest.fixture(autouse=True)
    def _import_build_input_transforms(self):
        from rlinf.models.embodiment.vla_lib_value_model.value_policy_config import (
            _build_input_transforms,
        )

        self._build_input_transforms = _build_input_transforms

    def test_unknown_env_type_raises(self):
        with pytest.raises(ValueError, match="Unknown environment type"):
            self._build_input_transforms(
                env_type="unknown_env",
                model_type="pi05",
                action_dim=32,
                default_prompt=None,
                norm_stats=None,
                use_quantile_norm=True,
            )

    def test_libero_env_type_returns_list(self):
        """Libero env type should return a non-empty list of transforms."""
        transforms = self._build_input_transforms(
            env_type="libero",
            model_type="pi05",
            action_dim=32,
            default_prompt="do something",
            norm_stats=None,
            use_quantile_norm=True,
        )
        assert isinstance(transforms, list)
        assert len(transforms) > 0

    def test_libero_without_norm_stats_has_no_normalize(self):
        transforms = self._build_input_transforms(
            env_type="libero",
            model_type="pi05",
            action_dim=32,
            default_prompt=None,
            norm_stats=None,
            use_quantile_norm=True,
        )
        from rlinf.datasets.vla_lib.lerobot_datasets.transforms import (
            Normalize,
        )

        assert not any(isinstance(t, Normalize) for t in transforms)

    def test_libero_with_norm_stats_has_normalize(self):
        # Provide minimal norm stats
        ns = {
            "state": NormStats(
                mean=np.zeros(7, dtype=np.float32),
                std=np.ones(7, dtype=np.float32),
                q01=np.zeros(7, dtype=np.float32),
                q99=np.ones(7, dtype=np.float32),
            )
        }
        transforms = self._build_input_transforms(
            env_type="libero",
            model_type="pi05",
            action_dim=32,
            default_prompt=None,
            norm_stats=ns,
            use_quantile_norm=True,
        )
        from rlinf.datasets.vla_lib.lerobot_datasets.transforms import (
            Normalize,
        )

        assert any(isinstance(t, Normalize) for t in transforms)


# ---------------------------------------------------------------------------
# ValuePolicy -- construction
# ---------------------------------------------------------------------------


class TestValuePolicyConstruction:
    """Test ValuePolicy construction and defaults."""

    @pytest.fixture
    def _make_policy(self):
        from rlinf.models.embodiment.vla_lib_value_model.value_policy import (
            ValuePolicy,
        )

        def _factory(**kwargs):
            mock_model = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.eval = MagicMock()

            with patch(
                "rlinf.models.embodiment.vla_lib_value_model.value_policy.compose",
                return_value=lambda x: x,
            ):
                return ValuePolicy(
                    model=mock_model,
                    transforms=(),
                    device="cpu",
                    **kwargs,
                )

        return _factory

    def test_metadata_defaults_to_empty_dict(self, _make_policy):
        policy = _make_policy()
        assert policy.metadata == {}

    def test_return_range_stored(self, _make_policy):
        policy = _make_policy(return_min=-2.0, return_max=1.0)
        assert policy.return_min == -2.0
        assert policy.return_max == 1.0

    def test_num_return_bins_stored(self, _make_policy):
        policy = _make_policy(num_return_bins=101)
        assert policy.num_return_bins == 101


# ---------------------------------------------------------------------------
# ValuePolicy -- default parameters
# ---------------------------------------------------------------------------


class TestValuePolicyDefaults:
    """Test default parameter values for ValuePolicy."""

    @pytest.fixture
    def _policy(self):
        from rlinf.models.embodiment.vla_lib_value_model.value_policy import (
            ValuePolicy,
        )

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        with patch(
            "rlinf.models.embodiment.vla_lib_value_model.value_policy.compose",
            return_value=lambda x: x,
        ):
            return ValuePolicy(
                model=mock_model,
                transforms=(),
                device="cpu",
            )

    def test_default_num_return_bins(self, _policy):
        assert _policy.num_return_bins == 201

    def test_default_return_min(self, _policy):
        assert _policy.return_min == -1.0

    def test_default_return_max(self, _policy):
        assert _policy.return_max == 0.0


# ---------------------------------------------------------------------------
# ValuePolicy -- infer_batch empty list
# ---------------------------------------------------------------------------


class TestValuePolicyInferBatchEmpty:
    """Test that infer_batch handles empty input gracefully."""

    def test_infer_batch_empty_list(self):
        from rlinf.models.embodiment.vla_lib_value_model.value_policy import (
            ValuePolicy,
        )

        mock_model = MagicMock()
        mock_model.to = MagicMock(return_value=mock_model)
        mock_model.eval = MagicMock()

        with patch(
            "rlinf.models.embodiment.vla_lib_value_model.value_policy.compose",
            return_value=lambda x: x,
        ):
            policy = ValuePolicy(
                model=mock_model,
                transforms=(),
                device="cpu",
            )

        result = policy.infer_batch([])
        assert result == []


# ---------------------------------------------------------------------------
# PI05CriticConfig
# ---------------------------------------------------------------------------


class TestPI05CriticConfig:
    """Test PI05CriticConfig construction and defaults."""

    @pytest.fixture(autouse=True)
    def _import_config(self):
        from rlinf.models.embodiment.vla_lib_value_model.configuration import (
            PI05CriticConfig,
        )

        self.PI05CriticConfig = PI05CriticConfig

    def test_default_construction(self):
        config = self.PI05CriticConfig()
        assert config is not None

    def test_default_critic_expert_variant(self):
        config = self.PI05CriticConfig()
        assert config.critic_expert_variant == "gemma_100m"

    def test_default_num_bins(self):
        config = self.PI05CriticConfig()
        assert config.num_bins == 201

    def test_default_v_min(self):
        config = self.PI05CriticConfig()
        assert config.v_min == -1.0

    def test_default_v_max(self):
        config = self.PI05CriticConfig()
        assert config.v_max == 0.0

    def test_custom_values(self):
        config = self.PI05CriticConfig(
            critic_expert_variant="gemma_300m",
            num_bins=101,
            v_min=-2.0,
            v_max=2.0,
        )
        assert config.critic_expert_variant == "gemma_300m"
        assert config.num_bins == 101
        assert config.v_min == -2.0
        assert config.v_max == 2.0

    def test_legacy_params_silently_ignored(self):
        """Legacy params (critic_forward_mode, expert_loss_type, etc.) should not raise."""
        config = self.PI05CriticConfig(
            critic_forward_mode="expert",
            expert_loss_type="categorical",
            vlm_loss_weight=0.5,
            expert_loss_weight=2.0,
        )
        assert config.critic_expert_variant == "gemma_100m"

    def test_inherits_from_pi05config(self):
        from rlinf.models.embodiment.vla_lib_value_model.configuration import PI05Config

        assert issubclass(self.PI05CriticConfig, PI05Config)

    def test_parent_fields_accessible(self):
        config = self.PI05CriticConfig()
        # PI05Config parent should provide these fields
        assert hasattr(config, "action_dim")
        assert hasattr(config, "action_horizon")
        assert hasattr(config, "max_token_len")


# ---------------------------------------------------------------------------
# CriticOutput dataclass
# ---------------------------------------------------------------------------


class TestCriticOutput:
    """Test CriticOutput dataclass defaults and field accessibility."""

    @pytest.fixture(autouse=True)
    def _import_output(self):
        from rlinf.models.embodiment.vla_lib_value_model.modeling_pi05_critic import (
            CriticOutput,
        )

        self.CriticOutput = CriticOutput

    def test_default_construction_all_none(self):
        output = self.CriticOutput()
        assert output.loss is None
        assert output.predicted_values is None
        assert output.logits is None
        assert output.probs is None
        assert output.atoms is None
        assert output.expert_loss is None
        assert output.hidden_states is None
        assert output.cat_acc_best is None
        assert output.cat_acc_neighbor is None
        assert output.cat_mae is None

    def test_set_loss(self):
        loss = torch.tensor(0.5)
        output = self.CriticOutput(loss=loss)
        assert torch.equal(output.loss, loss)

    def test_set_predicted_values(self):
        vals = torch.randn(4)
        output = self.CriticOutput(predicted_values=vals)
        assert torch.equal(output.predicted_values, vals)

    def test_is_model_output(self):
        from transformers.modeling_outputs import ModelOutput

        assert issubclass(self.CriticOutput, ModelOutput)


# ---------------------------------------------------------------------------
# ValueHead
# ---------------------------------------------------------------------------


class TestValueHead:
    """Test ValueHead module construction and forward pass."""

    @pytest.fixture(autouse=True)
    def _import_value_head(self):
        from rlinf.models.embodiment.vla_lib_value_model.modeling_pi05_critic import (
            ValueHead,
        )

        self.ValueHead = ValueHead

    def test_categorical_output_dim_num_bins(self):
        head = self.ValueHead(
            hidden_size=64,
            num_bins=201,
            v_min=-1.0,
            v_max=0.0,
        )
        assert head.value_proj.out_features == 201

    def test_cls_embedding_shape(self):
        head = self.ValueHead(
            hidden_size=64, num_bins=201, v_min=-1.0, v_max=0.0
        )
        cls_emb = head.get_cls_embedding(batch_size=4)
        assert cls_emb.shape == (4, 1, 64)

    def test_forward_categorical(self):
        head = self.ValueHead(
            hidden_size=64,
            num_bins=201,
            v_min=-1.0,
            v_max=0.0,
        )
        x = torch.randn(4, 64)
        out = head(x)
        assert out.shape == (4, 201)

    def test_atoms_shape(self):
        head = self.ValueHead(
            hidden_size=64,
            num_bins=201,
            v_min=-1.0,
            v_max=0.0,
        )
        assert head.atoms is not None
        assert head.atoms.shape == (201,)

    def test_atoms_range(self):
        head = self.ValueHead(
            hidden_size=64,
            num_bins=201,
            v_min=-1.0,
            v_max=0.0,
        )
        np.testing.assert_allclose(
            head.atoms[0].item(), -1.0, atol=1e-5
        )
        np.testing.assert_allclose(
            head.atoms[-1].item(), 0.0, atol=1e-5
        )

    def test_cls_embedding_batch_size_1(self):
        head = self.ValueHead(
            hidden_size=32, num_bins=10, v_min=0.0, v_max=1.0
        )
        cls_emb = head.get_cls_embedding(batch_size=1)
        assert cls_emb.shape == (1, 1, 32)

    def test_delta_z_computed(self):
        head = self.ValueHead(
            hidden_size=64,
            num_bins=201,
            v_min=-1.0,
            v_max=0.0,
        )
        expected_delta = (0.0 - (-1.0)) / (201 - 1)
        np.testing.assert_allclose(head.delta_z, expected_delta, rtol=1e-5)
