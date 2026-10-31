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

import rlinf.config as config_module


class ClusterStartedError(RuntimeError):
    """Raised when validation reaches cluster initialization."""


def _reject_cluster_start(*args, **kwargs):
    raise ClusterStartedError("Cluster started before CrossQ validation")


def _minimal_fsdp_sac_cfg(
    algorithm_q_head_type="default",
    model_q_head_type="default",
    model_type="mlp_policy",
):
    return OmegaConf.create(
        {
            "runner": {"task_type": "embodied"},
            "cluster": {},
            "algorithm": {
                "loss_type": "embodied_sac",
                "q_head_type": algorithm_q_head_type,
            },
            "actor": {
                "training_backend": "fsdp",
                "model": {
                    "model_type": model_type,
                    "q_head_type": model_q_head_type,
                },
            },
        }
    )


@pytest.mark.parametrize(
    ("algorithm_q_head_type", "model_q_head_type"),
    [("crossq", "default"), ("default", "crossq")],
)
def test_validate_cfg_rejects_crossq_mismatch_before_cluster(
    monkeypatch, algorithm_q_head_type, model_q_head_type
):
    cfg = _minimal_fsdp_sac_cfg(
        algorithm_q_head_type=algorithm_q_head_type,
        model_q_head_type=model_q_head_type,
    )
    monkeypatch.setattr(config_module, "Cluster", _reject_cluster_start)

    with pytest.raises(ValueError, match="q_head_type"):
        config_module.validate_cfg(cfg)


def test_validate_cfg_rejects_unsupported_crossq_model_before_cluster(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg(
        algorithm_q_head_type="crossq",
        model_q_head_type="crossq",
        model_type="flow_policy",
    )
    monkeypatch.setattr(config_module, "Cluster", _reject_cluster_start)

    with pytest.raises(ValueError, match="CrossQ.*flow_policy|flow_policy.*CrossQ"):
        config_module.validate_cfg(cfg)


def test_validate_cfg_does_not_apply_crossq_guard_to_other_losses(monkeypatch):
    cfg = _minimal_fsdp_sac_cfg(
        algorithm_q_head_type="crossq",
        model_q_head_type="default",
    )
    cfg.algorithm.loss_type = "actor_critic"
    monkeypatch.setattr(config_module, "Cluster", _reject_cluster_start)

    with pytest.raises(ClusterStartedError, match="before CrossQ validation"):
        config_module.validate_cfg(cfg)


def test_composed_crossq_example_passes_precluster_validation(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    embodiment_path = repo_root / "examples" / "embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(embodiment_path))
    monkeypatch.setattr(config_module, "Cluster", _reject_cluster_start)

    with initialize_config_dir(
        config_dir=str(embodiment_path / "config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_crossq_mlp")

    with pytest.raises(ClusterStartedError, match="before CrossQ validation"):
        config_module.validate_cfg(cfg)
