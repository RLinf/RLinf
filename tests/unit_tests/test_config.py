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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg


@pytest.fixture
def phyai_config(monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        return compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")


def test_phyai_eval_and_actor_resolve_same_openpi_config(monkeypatch, phyai_config):
    monkeypatch.setenv("PHYAI_KERNEL_CONFIG", "/unused/kernel.yaml")
    cfg = phyai_config
    validate_embodied_cfg(cfg)

    assert cfg.rollout.model.model_type == cfg.actor.model.model_type
    assert cfg.rollout.model.openpi.config_name == cfg.actor.model.openpi.config_name
    assert cfg.rollout.model.openpi.num_images_in_input == 2
    assert cfg.rollout.model.num_action_chunks == cfg.actor.model.num_action_chunks
    assert cfg.rollout.model.model_path == cfg.actor.model.model_path
    kernel_config = OmegaConf.to_container(
        cfg.rollout.phyai.kernel_config, resolve=True
    )
    assert kernel_config["schema"] == "phyai.kernel/v1"
    assert kernel_config["profile"] == "static"
    assert kernel_config["defaults"]["fallback"] == "error"
    assert [rule["use"] for rule in kernel_config["overrides"]] == [
        "flashinfer.attention_paged.fa2",
        "flashinfer.activation.gelu_and_mul",
        "torch.activation.gelu",
    ]


def test_phyai_training_requires_bucket_weight_sync(phyai_config):
    cfg = phyai_config
    cfg.weight_syncer.type = "patch"

    with pytest.raises(AssertionError, match="weight_syncer.type='bucket'"):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            "algorithm.loss_type",
            "embodied_dagger",
            "does not support algorithm.loss_type='embodied_dagger'",
        ),
        (
            "algorithm.loss_type",
            "rlt_ac",
            "does not support algorithm.loss_type='rlt_ac'",
        ),
        (
            "algorithm.loss_type",
            "rlt_td3",
            "does not support algorithm.loss_type='rlt_td3'",
        ),
        (
            "algorithm.adv_type",
            "opd",
            "does not support algorithm.adv_type='opd'",
        ),
    ],
)
def test_phyai_training_rejects_unsupported_algorithms(
    phyai_config, path, value, message
):
    cfg = phyai_config
    if path == "algorithm.adv_type":
        cfg.algorithm.recompute_logprobs = False
    OmegaConf.update(cfg, path, value)

    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("actor.model.add_value_head", False, "add_value_head=True"),
        ("actor.model.openpi.value_after_vlm", False, "value_after_vlm=True"),
        ("actor.model.openpi.value_vlm_mode", "last_token", "mean_token"),
        ("actor.model.openpi.joint_logprob", True, "joint_logprob=False"),
        ("actor.model.openpi.noise_method", "flow_noise", "flow_ode or flow_sde"),
        ("actor.model.openpi.num_images_in_input", 1, "num_images_in_input=2"),
    ],
)
def test_phyai_training_validates_actor_contract(phyai_config, path, value, message):
    cfg = phyai_config
    OmegaConf.update(cfg, path, value)

    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("pi05", False, "model.pi05=True"),
        ("openpi.config_name", "", "non-empty model.openpi.config_name"),
        ("openpi.paligemma_variant", "gemma_3b", "paligemma_variant='gemma_2b'"),
        (
            "openpi.action_expert_variant",
            "gemma_1b",
            "action_expert_variant='gemma_300m'",
        ),
        ("openpi.num_images_in_input", 4, "num_images_in_input in \\[1, 3\\]"),
    ],
)
def test_phyai_eval_validates_plugin_contract(phyai_config, path, value, message):
    cfg = phyai_config
    cfg.runner.only_eval = True
    OmegaConf.update(cfg.rollout.model, path, value)

    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(cfg)
