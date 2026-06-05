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

import rlinf.config as rlinf_config


def _minimal_fsdp_sac_cfg(**algorithm_overrides):
    algorithm = {
        "loss_type": "embodied_sac",
        "edac_eta": 0.0,
    }
    algorithm.update(algorithm_overrides)
    return OmegaConf.create(
        {
            "runner": {
                "task_type": "embodied",
            },
            "cluster": {},
            "algorithm": algorithm,
            "actor": {
                "training_backend": "fsdp",
                "model": {},
            },
        }
    )


def _raise_if_cluster_is_constructed(*args, **kwargs):
    raise AssertionError("Cluster should not be initialized before SAC validation")


def test_validate_cfg_rejects_fsdpsac_edac_before_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg(edac_eta=0.1)
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(NotImplementedError, match="FSDP.*edac_eta|edac_eta.*FSDP"):
        rlinf_config.validate_cfg(cfg)


def test_validate_cfg_rejects_algorithm_only_crossq_before_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg(q_head_type="crossq")
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(ValueError, match="actor.model.q_head_type.*crossq"):
        rlinf_config.validate_cfg(cfg)


def test_validate_cfg_rejects_model_only_crossq_before_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg()
    cfg.actor.model.model_type = "mlp_policy"
    cfg.actor.model.q_head_type = "crossq"
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(ValueError, match="algorithm.q_head_type.*crossq"):
        rlinf_config.validate_cfg(cfg)


@pytest.mark.parametrize(
    ("algorithm_q_head_type", "model_q_head_type"),
    [
        ("bogus", None),
        (None, "bogus"),
        ("bogus", "bogus"),
    ],
)
def test_validate_cfg_rejects_invalid_q_head_type_before_cluster(
    monkeypatch, algorithm_q_head_type, model_q_head_type
):
    cfg = _minimal_fsdp_sac_cfg()
    if algorithm_q_head_type is not None:
        cfg.algorithm.q_head_type = algorithm_q_head_type
    if model_q_head_type is not None:
        cfg.actor.model.q_head_type = model_q_head_type
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(ValueError, match="q_head_type.*bogus"):
        rlinf_config.validate_cfg(cfg)


def test_validate_cfg_allows_model_only_default_q_head_until_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg()
    cfg.actor.model.q_head_type = "default"
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(AssertionError, match="Cluster should not be initialized"):
        rlinf_config.validate_cfg(cfg)


def test_validate_cfg_rejects_flow_crossq_before_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg(q_head_type="crossq")
    cfg.actor.model.model_type = "flow_policy"
    cfg.actor.model.q_head_type = "crossq"
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(ValueError, match="CrossQ.*flow_policy|flow_policy.*CrossQ"):
        rlinf_config.validate_cfg(cfg)


def test_validate_cfg_rejects_crossq_without_model_type_before_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg(q_head_type="crossq")
    cfg.actor.model.q_head_type = "crossq"
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(ValueError, match="CrossQ.*model_type"):
        rlinf_config.validate_cfg(cfg)


@pytest.mark.parametrize("model_type", ["mlp_policy", "cnn_policy"])
def test_validate_cfg_allows_supported_crossq_models_until_cluster(
    monkeypatch, model_type
):
    cfg = _minimal_fsdp_sac_cfg(q_head_type="crossq")
    cfg.actor.model.model_type = model_type
    cfg.actor.model.q_head_type = "crossq"
    monkeypatch.setattr(rlinf_config, "Cluster", _raise_if_cluster_is_constructed)

    with pytest.raises(AssertionError, match="Cluster should not be initialized"):
        rlinf_config.validate_cfg(cfg)
