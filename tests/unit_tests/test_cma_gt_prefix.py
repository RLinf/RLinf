# Copyright 2025 The RLinf Authors.
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
import torch

from rlinf.models.embodiment.cma.cma_action_model import CMAConfig, CMAPolicy
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


def make_minimal_cma_config(use_gt_prefix=False, gt_prefix_length=0):
    return CMAConfig(
        use_gt_prefix=use_gt_prefix,
        gt_prefix_length=gt_prefix_length,
        hidden_size=128,
        num_action_classes=4,
        instruction_encoder_config={
            "sensor_uuid": "instruction",
            "vocab_size": 2504,
            "use_pretrained_embeddings": False,
            "embedding_file": "",
            "fine_tune_embeddings": True,
            "embedding_size": 50,
            "hidden_size": 128,
            "rnn_type": "LSTM",
            "final_state_only": False,
            "bidirectional": True,
        },
        depth_encoder_config={
            "cnn_type": "VlnResnetDepthEncoder",
            "output_size": 128,
            "backbone": "resnet50",
            "ddppo_checkpoint": "",
            "trainable": False,
        },
        rgb_encoder_config={
            "cnn_type": "TorchVisionResNet50",
            "output_size": 256,
            "trainable": False,
        },
        state_encoder_config={
            "hidden_size": 512,
            "rnn_type": "GRU",
        },
    )


def make_mock_env_obs_with_gt(
    batch_size=2, gt_action_valid=None, gt_current_action=None, current_step=None
):
    if gt_action_valid is None:
        gt_action_valid = [True] * batch_size
    if gt_current_action is None:
        gt_current_action = [1] * batch_size
    if current_step is None:
        current_step = [1] * batch_size
    return {
        "habitat_gt_action_valid": gt_action_valid,
        "habitat_gt_current_action": gt_current_action,
        "habitat_current_step": current_step,
    }


class TestCMAConfigDefaults:
    def test_cma_config_default_gt_prefix_disabled(self):
        cfg = CMAConfig()
        assert cfg.use_gt_prefix is False
        assert cfg.gt_prefix_length == 0

    def test_cma_config_gt_prefix_length_zero_when_disabled(self):
        cfg = CMAConfig(use_gt_prefix=False)
        assert cfg.use_gt_prefix is False
        assert cfg.gt_prefix_length == 0


class TestCMAFeatureOffNoOp:
    @pytest.fixture
    def cma_policy_feature_off(self):
        cfg = make_minimal_cma_config(use_gt_prefix=False, gt_prefix_length=0)
        return CMAPolicy(cfg=cfg)

    def test_feature_off_returns_model_actions_unchanged(self, cma_policy_feature_off):
        policy = cma_policy_feature_off
        batch_size = 2
        model_actions = torch.tensor([[2], [3]], dtype=torch.long)
        env_obs_with_gt = make_mock_env_obs_with_gt(batch_size=batch_size)

        executed, gt_executed = policy._resolve_current_action_execution(
            env_obs_with_gt, model_actions
        )

        assert torch.equal(executed, model_actions)
        assert not gt_executed.any()

    def test_feature_off_ignores_gt_metadata_when_present(self, cma_policy_feature_off):
        policy = cma_policy_feature_off
        batch_size = 2
        model_actions = torch.tensor([[2], [3]], dtype=torch.long)
        env_obs_with_invalid_gt = make_mock_env_obs_with_gt(
            batch_size=batch_size,
            gt_action_valid=[True, True],
            gt_current_action=[0, 0],
            current_step=[1, 1],
        )

        executed, gt_executed = policy._resolve_current_action_execution(
            env_obs_with_invalid_gt, model_actions
        )

        assert torch.equal(executed, model_actions)
        assert not gt_executed.any()

    def test_feature_off_with_empty_gt_metadata(self, cma_policy_feature_off):
        policy = cma_policy_feature_off
        model_actions = torch.tensor([[2], [3]], dtype=torch.long)
        env_obs_no_gt = {}

        executed, gt_executed = policy._resolve_current_action_execution(
            env_obs_no_gt, model_actions
        )

        assert torch.equal(executed, model_actions)
        assert not gt_executed.any()


class TestCMAFeatureOffWithGtPrefixLengthZero:
    def test_gt_prefix_length_zero_returns_model_actions(self):
        cfg = make_minimal_cma_config(use_gt_prefix=True, gt_prefix_length=0)
        policy = CMAPolicy(cfg=cfg)
        batch_size = 2
        model_actions = torch.tensor([[2], [3]], dtype=torch.long)
        env_obs_with_gt = make_mock_env_obs_with_gt(batch_size=batch_size)

        executed, gt_executed = policy._resolve_current_action_execution(
            env_obs_with_gt, model_actions
        )

        assert torch.equal(executed, model_actions)
        assert not gt_executed.any()


class TestCMAGTPrefixOnlyInCMAConfig:
    def test_only_cma_yaml_has_gt_prefix_flags(self):
        import yaml

        cma_yaml_path = "examples/embodiment/config/model/cma.yaml"
        other_yaml_paths = [
            "examples/embodiment/config/model/openvla.yaml",
            "examples/embodiment/config/model/gr00t.yaml",
            "examples/embodiment/config/model/pi0.yaml",
            "examples/embodiment/config/model/mlp_policy.yaml",
        ]

        with open(cma_yaml_path) as f:
            cma_config = yaml.safe_load(f)

        assert "use_gt_prefix" in cma_config
        assert "gt_prefix_length" in cma_config

        for yaml_path in other_yaml_paths:
            with open(yaml_path) as f:
                other_config = yaml.safe_load(f)
            assert "use_gt_prefix" not in other_config, (
                f"{yaml_path} should not have use_gt_prefix"
            )
            assert "gt_prefix_length" not in other_config, (
                f"{yaml_path} should not have gt_prefix_length"
            )


class TestNonCMAModelsIgnoreGTPrefixMetadata:
    @pytest.fixture
    def mlp_policy(self):
        policy = MLPPolicy(
            obs_dim=32,
            action_dim=4,
            num_action_chunks=1,
            add_value_head=True,
            add_q_head=False,
        )
        policy.eval()
        return policy

    def make_mlp_env_obs(self, batch_size=2, states=None, with_gt_metadata=False):
        obs = {
            "states": states if states is not None else torch.randn(batch_size, 32),
        }
        if with_gt_metadata:
            obs["habitat_gt_action_valid"] = [True] * batch_size
            obs["habitat_gt_current_action"] = [1] * batch_size
            obs["habitat_current_step"] = [1] * batch_size
        return obs

    def test_mlp_policy_ignores_gt_prefix_metadata(self, mlp_policy):
        policy = mlp_policy
        shared_states = torch.randn(2, 32)
        obs_without_gt = self.make_mlp_env_obs(
            batch_size=2, states=shared_states, with_gt_metadata=False
        )
        obs_with_gt = self.make_mlp_env_obs(
            batch_size=2, states=shared_states.clone(), with_gt_metadata=True
        )

        actions_without_gt, result_without_gt = policy.predict_action_batch(
            obs_without_gt, mode="eval"
        )
        actions_with_gt, result_with_gt = policy.predict_action_batch(
            obs_with_gt, mode="eval"
        )

        assert torch.equal(actions_without_gt, actions_with_gt)
        assert torch.equal(
            result_without_gt["prev_logprobs"], result_with_gt["prev_logprobs"]
        )
        assert torch.equal(
            result_without_gt["prev_values"], result_with_gt["prev_values"]
        )

    def test_mlp_policy_has_no_gt_prefix_method(self, mlp_policy):
        policy = mlp_policy
        assert not hasattr(policy, "_resolve_current_action_execution")
        assert not hasattr(policy, "use_gt_prefix")
        assert not hasattr(policy, "gt_prefix_length")
