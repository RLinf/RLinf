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

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
    build_rlt_config,
)
from rlinf.models.embodiment.prefix_ft.config import (
    apply_prefix_head_z_dim,
    extra_z_dim_from_cfg,
    is_prefix_ac_loss,
    is_rlt_env_loss,
    resolve_prefix_feature_model_config,
    resolve_prefix_pool,
)
from rlinf.models.embodiment.prefix_ft.history import StateHistoryBuffer
from rlinf.models.embodiment.prefix_ft.pool import pool_prefix
from rlinf.models.embodiment.prefix_ft.protocol import (
    PrefixFeatureModel,
    extract_prefix_obs,
)
from rlinf.models.embodiment.prefix_ft.types import PREFIX_OBS_KEYS


class _PrefixPoolStub(OpenPiPytorchActionModel):
    def __init__(self, rlt_cfg: OpenPiPytorchRLTConfig):
        torch.nn.Module.__init__(self)
        self.rlt_cfg = rlt_cfg


class FakeVLA:
    z_dim = 8

    def extract_prefix_obs(self, env_obs):
        batch = env_obs["states"].shape[0]
        return {
            "z_rl": torch.ones(batch, self.z_dim),
            "proprio": env_obs["states"].to(dtype=torch.float32),
            "ref_chunk": torch.zeros(batch, 4, 7),
        }


def test_pool_prefix_masked_mean():
    hidden = torch.tensor(
        [
            [[1.0, 0.0], [3.0, 0.0], [9.0, 0.0]],
            [[2.0, 0.0], [4.0, 0.0], [8.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])
    pooled = pool_prefix(hidden, mask, mode="masked_mean")
    torch.testing.assert_close(pooled[0], torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(pooled[1], torch.tensor([2.0, 0.0]))


def test_pool_prefix_mean_and_last():
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    mean = pool_prefix(hidden, None, mode="mean")
    torch.testing.assert_close(mean, hidden.mean(dim=1))
    last = pool_prefix(hidden, None, mode="last")
    torch.testing.assert_close(last, hidden[:, -1])
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    last_valid = pool_prefix(hidden, mask, mode="last")
    torch.testing.assert_close(last_valid[0], hidden[0, 1])
    torch.testing.assert_close(last_valid[1], hidden[1, 2])


def test_resolve_prefix_pool_backward_compat():
    assert resolve_prefix_pool(use_rlt=True) == "rlt_token"
    assert resolve_prefix_pool(use_rlt=False) == "masked_mean"
    assert (
        resolve_prefix_pool(use_rlt=True, stage2_z_source="vlm_prefix", rlt_use_mask=True)
        == "masked_mean"
    )
    assert resolve_prefix_pool(use_rlt=True, prefix_pool="last") == "last"


def test_build_rlt_config_reads_parent_prefix_pool():
    openpi_cfg = OmegaConf.create({"use_rlt": False, "rlt_use_mask": True})
    parent = OmegaConf.create(
        {"prefix": {"pool": "masked_mean", "image_only": False}, "openpi": openpi_cfg}
    )
    cfg = build_rlt_config(openpi_cfg, parent_cfg=parent)
    assert cfg.prefix_pool == "masked_mean"
    assert cfg.rlt_image_only is False
    legacy = build_rlt_config(OmegaConf.create({"use_rlt": True}))
    assert legacy.prefix_pool == "rlt_token"


def test_feature_model_config_alias():
    legacy = OmegaConf.create(
        {"rollout": {"rlt_feature_model": {"model_type": "openpi_rlinf"}}}
    )
    assert (
        resolve_prefix_feature_model_config(legacy).model_type == "openpi_rlinf"
    )
    modern = OmegaConf.create(
        {"rollout": {"prefix_feature_model": {"model_type": "openpi"}}}
    )
    assert resolve_prefix_feature_model_config(modern).model_type == "openpi"


def test_prefix_ac_loss_alias():
    assert is_prefix_ac_loss("prefix_ac")
    assert is_prefix_ac_loss("rlt_ac")
    assert is_rlt_env_loss("prefix_ac")
    assert is_rlt_env_loss("rlt_td3")
    assert not is_prefix_ac_loss("embodied_sac")


def test_openpi_rlinf_pool_does_not_require_rlt():
    pooled = _PrefixPoolStub(
        OpenPiPytorchRLTConfig(use_rlt=False, prefix_pool="masked_mean")
    )
    assert not pooled._stage2_requires_rlt()
    token = _PrefixPoolStub(
        OpenPiPytorchRLTConfig(use_rlt=True, prefix_pool="rlt_token")
    )
    assert token._stage2_requires_rlt()


def test_official_openpi_config_resolves_pool_without_use_rlt():
    from rlinf.models.embodiment.openpi.openpi_action_model import OpenPi0Config

    cfg = OpenPi0Config(use_rlt=False, prefix_pool="masked_mean")
    assert (
        resolve_prefix_pool(use_rlt=cfg.use_rlt, prefix_pool=cfg.prefix_pool)
        == "masked_mean"
    )


def test_fake_vla_protocol_and_mlp():
    feature = FakeVLA()
    assert isinstance(feature, PrefixFeatureModel)
    obs = extract_prefix_obs(feature, {"states": torch.zeros(2, 3)})
    assert tuple(obs.keys()) == PREFIX_OBS_KEYS
    policy = RLTMLPPolicy(
        z_dim=feature.z_dim,
        proprio_dim=3,
        action_dim=7,
        num_action_chunks=4,
    )
    action, _, _ = policy.sac_forward(obs)
    assert action.shape == (2, 28)


def test_history_disabled_is_identity():
    obs = FakeVLA().extract_prefix_obs({"states": torch.ones(1, 3)})
    hist = StateHistoryBuffer(enable=False, steps=4, proprio_dim=3)
    fused = hist.fuse(obs)
    assert fused["z_rl"].shape[-1] == 8
    assert hist.extra_z_dim == 0
    cfg = OmegaConf.create({"z_dim": 2048, "state_history": {"enable": False}})
    assert extra_z_dim_from_cfg(cfg) == 0


def test_history_enabled_concat_reset_and_pad():
    hist = StateHistoryBuffer(enable=True, steps=4, proprio_dim=2, pad="zero")
    prefix = torch.zeros(1, 8)
    obs = {
        "z_rl": prefix,
        "proprio": torch.ones(1, 2),
        "ref_chunk": torch.zeros(1, 1, 1),
    }
    fused = hist.fuse(obs)
    assert fused["z_rl"].shape[-1] == 8 + 8
    torch.testing.assert_close(fused["z_rl"][0, 8:14], torch.zeros(6))
    torch.testing.assert_close(fused["z_rl"][0, 14:], torch.ones(2))

    hist.reset(mask=torch.tensor([True]))
    obs2 = {
        "z_rl": prefix,
        "proprio": torch.full((1, 2), 5.0),
        "ref_chunk": torch.zeros(1, 1, 1),
    }
    fused2 = hist.fuse(obs2)
    torch.testing.assert_close(fused2["z_rl"][0, 8:14], torch.zeros(6))
    torch.testing.assert_close(fused2["z_rl"][0, 14:], torch.full((2,), 5.0))

    repeat = StateHistoryBuffer(enable=True, steps=4, proprio_dim=2, pad="repeat")
    fused_repeat = repeat.fuse(obs)
    torch.testing.assert_close(
        fused_repeat["z_rl"][0, 8:], torch.ones(8)
    )


def test_history_peek_does_not_commit():
    hist = StateHistoryBuffer(enable=True, steps=2, proprio_dim=1, pad="zero")
    obs_t = {
        "z_rl": torch.zeros(1, 2),
        "proprio": torch.tensor([[1.0]]),
        "ref_chunk": torch.zeros(1, 1, 1),
    }
    hist.fuse(obs_t)
    peek = hist.fuse(
        {
            "z_rl": torch.zeros(1, 2),
            "proprio": torch.tensor([[9.0]]),
            "ref_chunk": torch.zeros(1, 1, 1),
        },
        commit=False,
    )
    torch.testing.assert_close(peek["z_rl"][0, 2:], torch.tensor([1.0, 9.0]))
    committed = hist.fuse(
        {
            "z_rl": torch.zeros(1, 2),
            "proprio": torch.tensor([[9.0]]),
            "ref_chunk": torch.zeros(1, 1, 1),
        }
    )
    torch.testing.assert_close(committed["z_rl"][0, 2:], torch.tensor([1.0, 9.0]))


def test_effective_z_dim_matches_mlp():
    full_cfg = OmegaConf.create(
        {
            "algorithm": {"state_history": {"enable": True, "steps": 4}},
            "actor": {
                "model": {
                    "model_type": "rlt_mlp_policy",
                    "z_dim": 2048,
                    "proprio_dim": 19,
                    "action_dim": 7,
                    "num_action_chunks": 10,
                    "ref_num_action_chunks": 10,
                }
            },
        }
    )
    apply_prefix_head_z_dim(full_cfg.actor.model, full_cfg)
    assert int(full_cfg.actor.model.z_dim) == 2048 + 4 * 19
    from rlinf.models.embodiment.mlp_policy import get_model

    model = get_model(full_cfg.actor.model)
    assert model.z_dim == 2048 + 4 * 19

    off_cfg = OmegaConf.create(
        {
            "algorithm": {"state_history": {"enable": False, "steps": 4}},
            "actor": {"model": {"z_dim": 2048, "proprio_dim": 19}},
        }
    )
    apply_prefix_head_z_dim(off_cfg.actor.model, off_cfg)
    assert int(off_cfg.actor.model.z_dim) == 2048


def test_fused_history_eaten_by_unmodified_mlp():
    feature = FakeVLA()
    hist = StateHistoryBuffer(enable=True, steps=4, proprio_dim=3)
    obs = hist.fuse(feature.extract_prefix_obs({"states": torch.zeros(2, 3)}))
    policy = RLTMLPPolicy(
        z_dim=feature.z_dim + hist.extra_z_dim,
        proprio_dim=3,
        action_dim=7,
        num_action_chunks=4,
    )
    action, _, _ = policy.sac_forward(obs)
    assert action.shape == (2, 28)
