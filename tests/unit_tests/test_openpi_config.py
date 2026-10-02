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

import inspect
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.rollout import utils as rollout_utils
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def test_phyai_eval_and_actor_resolve_same_openpi_config(monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    monkeypatch.delenv("PHYAI_KERNEL_CONFIG", raising=False)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")

    assert cfg.rollout.model.model_type == cfg.actor.model.model_type
    assert cfg.rollout.model.openpi.config_name == cfg.actor.model.openpi.config_name
    assert cfg.rollout.model.openpi.num_images_in_input == 2
    assert cfg.rollout.model.num_action_chunks == cfg.actor.model.num_action_chunks
    assert cfg.rollout.phyai.kernel_config.endswith(
        "examples/configs/kernel_policies/pi05/rlinf_bf16.yaml"
    )


def test_phyai_training_requires_bucket_weight_sync(monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")
    cfg.weight_syncer.type = "patch"

    with pytest.raises(AssertionError, match="weight_syncer.type='bucket'"):
        validate_embodied_cfg(cfg)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("actor.model.add_value_head", False, "add_value_head=True"),
        ("actor.model.openpi.value_after_vlm", False, "value_after_vlm=True"),
        ("actor.model.openpi.value_vlm_mode", "last_token", "mean_token"),
        ("actor.model.openpi.joint_logprob", True, "joint_logprob=False"),
        ("actor.model.openpi.noise_method", "flow_noise", "flow_ode or flow_sde"),
    ],
)
def test_phyai_training_validates_actor_contract(monkeypatch, path, value, message):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")
    OmegaConf.update(cfg, path, value)

    with pytest.raises(AssertionError, match=message):
        validate_embodied_cfg(cfg)


@pytest.mark.asyncio
async def test_actor_weight_sync_uses_selective_state():
    class _SenderSyncer:
        def __init__(self) -> None:
            self.initialized = False
            self.names = []
            self.init_calls = []
            self.sync_calls = []

        def sender_initialized(self):
            return self.initialized

        async def init_sender(self, *, param_names_need_sync, **_kwargs):
            self.initialized = True
            self.names = list(param_names_need_sync)
            self.init_calls.append(self.names)

        async def sync(self, _state_dict, _send, version):
            self.sync_calls.append((list(self.names), version))

    actor = object.__new__(EmbodiedFSDPActor)
    actor.enable_offload = False
    actor.weight_syncer = _SenderSyncer()
    actor.param_names_need_sync = ["model.trainable", "value_head.weight"]
    actor._is_weight_sender = False
    actor._sync_weight_comm_options = None
    actor._group_name = "actor"
    actor._rollout_group_name = "rollout"
    actor._rollout_all_ranks = [0]
    actor.get_rollout_state_dict = lambda: {
        "model.trainable": torch.ones(1),
        "model.frozen": torch.ones(1),
        "value_head.weight": torch.ones(1),
    }
    actor.version = 3
    actor.get_rollout_sync_version = lambda: 7
    actor.log_info = lambda _message: None

    sync = inspect.unwrap(EmbodiedFSDPActor.sync_model_to_rollout)
    await sync(actor)
    await sync(actor)

    selected = ["model.trainable", "value_head.weight"]
    assert actor.weight_syncer.init_calls == [selected]
    assert actor.weight_syncer.sync_calls == [
        (selected, 7),
        (selected, 7),
    ]


def test_hf_embodied_backend_selection_without_phyai():
    cfg = OmegaConf.create({"rollout": {"rollout_backend": "hf"}})

    assert rollout_utils.get_embodied_rollout_worker(cfg) is MultiStepRolloutWorker
