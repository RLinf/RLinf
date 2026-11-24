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

"""Unit tests for the SAPG tensor utilities (CPU only, no simulator needed)."""

import pytest
import torch

from rlinf.algorithms.sapg import (
    build_sapg_exploration,
    create_sinusoidal_encoding,
    relabel_sapg_batch,
    sample_repeat_idxs,
    sapg_block_ids,
    sapg_entropy_coef_for_states,
    sapg_follower_indices,
)


def test_sinusoidal_encoding_shape_and_range():
    values = torch.linspace(50.0, 0.0, 6)
    emb = create_sinusoidal_encoding(values, dim=16, n=100.0)
    assert emb.shape == (6, 16)
    assert emb.min() >= -1.0 and emb.max() <= 1.0
    # distinct inputs must give distinct encodings
    assert not torch.allclose(emb[0], emb[-1])


def test_sinusoidal_encoding_requires_even_dim():
    try:
        create_sinusoidal_encoding(torch.zeros(2), dim=15)
    except AssertionError:
        return
    raise AssertionError("odd dim should raise")


def test_build_exploration_blocks():
    num_envs, num_blocks = 24, 6
    spec = build_sapg_exploration(
        num_envs=num_envs, num_blocks=num_blocks, embd_size=16
    )
    assert spec.embedding.shape == (num_envs, 16)
    assert spec.intrinsic_coef.shape == (num_envs,)
    assert spec.block_size == num_envs // num_blocks
    # The leader is the last block and uses intrinsic_coef_end.
    assert torch.isclose(spec.intrinsic_coef[0], torch.tensor(0.5))
    assert torch.isclose(spec.intrinsic_coef[-1], torch.tensor(0.0))
    # within a block every env shares the coefficient
    assert torch.allclose(
        spec.intrinsic_coef[: spec.block_size],
        spec.intrinsic_coef[0].expand(spec.block_size),
    )
    # different blocks have different embeddings
    assert not torch.allclose(spec.embedding[0], spec.embedding[spec.block_size])


def test_build_exploration_none_intrinsic():
    spec = build_sapg_exploration(
        num_envs=8, num_blocks=2, embd_size=8, intrinsic_reward_type="none"
    )
    assert torch.allclose(spec.intrinsic_coef, torch.zeros(8))


def test_build_exploration_rejects_indivisible():
    try:
        build_sapg_exploration(num_envs=10, num_blocks=3)
    except AssertionError:
        return
    raise AssertionError("indivisible num_envs should raise")


def test_sample_repeat_idxs():
    g = torch.Generator().manual_seed(0)
    idxs = sample_repeat_idxs(num_blocks=6, off_policy_ratio=3, generator=g)
    assert idxs[0] == 0
    assert len(idxs) == 4
    assert len(set(idxs)) == len(idxs)
    # ratio larger than available followers is clipped
    idxs_all = sample_repeat_idxs(num_blocks=3, off_policy_ratio=10)
    assert idxs_all[0] == 0 and len(idxs_all) == 3


def test_entropy_coef_recovers_behaviour_blocks():
    """Coefficients follow the conditioning code used by the loss."""
    num_envs, num_blocks, embd_size = 12, 3, 8
    spec = build_sapg_exploration(
        num_envs=num_envs, num_blocks=num_blocks, embd_size=embd_size
    )
    states = torch.cat([torch.randn(num_envs, 5), spec.embedding], dim=-1)
    coef = sapg_entropy_coef_for_states(
        states, num_blocks=num_blocks, embd_size=embd_size
    )
    assert torch.allclose(coef, spec.intrinsic_coef)

    # Leader copies use the coefficient of the final block.
    leader_states = states.clone()
    leader_states[..., -embd_size:] = spec.embedding[-1]
    leader_coef = sapg_entropy_coef_for_states(
        leader_states, num_blocks=num_blocks, embd_size=embd_size
    )
    assert torch.allclose(leader_coef, spec.intrinsic_coef[-1].expand(num_envs))


def test_sapg_loss_subtracts_block_weighted_entropy():
    """SAPG loss = PPO loss - per-block entropy bonus, with gradients to entropy."""
    from rlinf.algorithms.losses import (
        compute_ppo_actor_critic_loss,
        compute_sapg_actor_critic_loss,
    )

    torch.manual_seed(0)
    bsz, action_dim = 8, 3
    logprobs = torch.randn(bsz, action_dim, requires_grad=True)
    old_logprobs = logprobs.detach() + 0.05 * torch.randn(bsz, action_dim)
    advantages = torch.randn(bsz, action_dim)
    values = torch.randn(bsz, 1)
    returns = torch.randn(bsz, 1)
    prev_values = values.detach() + 0.01
    entropy = torch.rand(bsz, action_dim, requires_grad=True)
    coef = torch.linspace(0.5, 0.0, bsz)

    common = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "values": values,
        "returns": returns,
        "prev_values": prev_values,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "value_clip": 1.0,
        "huber_delta": 10.0,
    }
    ppo_loss, _ = compute_ppo_actor_critic_loss(**common)
    sapg_loss, metrics = compute_sapg_actor_critic_loss(
        **common, entropy=entropy, sapg_entropy_coef=coef, critic_coef=1.0
    )
    expected_term = (entropy.sum(dim=-1) * coef).mean()
    assert torch.allclose(ppo_loss - expected_term, sapg_loss, atol=1e-6)
    assert torch.allclose(metrics["actor/sapg_entropy_term"], expected_term.detach())

    sapg_loss.backward()
    assert entropy.grad is not None

    # Without entropy coefficients SAPG falls back to plain PPO.
    plain_loss, _ = compute_sapg_actor_critic_loss(**common, critic_coef=1.0)
    assert torch.allclose(plain_loss, ppo_loss)


def test_sapg_aggregation_keeps_behaviour_data_and_recomputes_targets():
    codes = build_sapg_exploration(3, 3, embd_size=4).embedding
    states = torch.cat(
        (torch.arange(6.0).reshape(2, 3, 1), codes.expand(2, -1, -1)), dim=-1
    )
    rewards = torch.tensor([[[1.0], [2.0], [3.0]], [[4.0], [5.0], [6.0]]])
    dones = torch.zeros(3, 3, 1, dtype=torch.bool)
    dones[1, 1] = True
    logprobs = -torch.arange(6.0).reshape(2, 3, 1)
    batch = {
        "forward_inputs": {"states": states, "action": rewards + 10},
        "next_obs": {"states": states[..., :1] + 10},
        "rewards": rewards,
        "dones": dones,
        "prev_logprobs": logprobs,
        "prev_values": torch.zeros(3, 3, 1),
        "returns": rewards + 5,
        "advantages": rewards + 5,
    }

    def score_values(leader_states):
        assert (sapg_block_ids(leader_states, codes) == 2).all()
        return leader_states[..., :1]

    followers = sapg_follower_indices(sapg_block_ids(states[0], codes), 3, [0, 2])
    relabel_sapg_batch(batch, followers, codes[-1], 0.5, score_values)
    torch.testing.assert_close(batch["forward_inputs"]["states"][:, :3], states)
    torch.testing.assert_close(batch["prev_logprobs"][:, 3], logprobs[:, 1])
    torch.testing.assert_close(
        batch["forward_inputs"]["action"][:, 3], rewards[:, 1] + 10
    )
    torch.testing.assert_close(batch["returns"][:, :3], rewards + 5)
    torch.testing.assert_close(
        batch["prev_values"][:, 3, 0], torch.tensor([1.0, 4.0, 14.0])
    )
    torch.testing.assert_close(batch["returns"][:, 3, 0], torch.tensor([2.0, 12.0]))
    torch.testing.assert_close(batch["advantages"][:, 3, 0], torch.tensor([1.0, 8.0]))


def test_sapg_policy_conditions_rollout_and_evaluation():
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.mlp_policy.sapg_mlp_policy import SAPGMLPPolicy

    policy = SAPGMLPPolicy(
        2, 2, 1, True, OmegaConf.create({"num_blocks": 3, "embd_size": 4})
    )
    with torch.no_grad():
        policy.actor_logstd[0].fill_(-1)
        policy.actor_logstd[-1].fill_(1)
    _, train = policy.predict_action_batch({"states": torch.zeros(6, 2)})
    _, evaluation = policy.predict_action_batch(
        {"states": torch.zeros(5, 2)}, mode="eval"
    )
    train_ids = sapg_block_ids(train["forward_inputs"]["states"], policy.block_codes)
    eval_ids = sapg_block_ids(
        evaluation["forward_inputs"]["states"], policy.block_codes
    )
    assert train_ids.tolist() == [0, 1, 2, 0, 1, 2]
    assert eval_ids.tolist() == [2] * 5
    # Convert inference tensors before taking the training gradients.
    forward_inputs = {
        key: value.clone() for key, value in train["forward_inputs"].items()
    }
    scored = policy(forward_inputs=forward_inputs)
    torch.testing.assert_close(scored["logprobs"], train["prev_logprobs"])
    assert (scored["entropy"][train_ids == 2] > scored["entropy"][train_ids == 0]).all()
    (scored["values"].sum() + scored["logprobs"].sum()).backward()
    assert policy.block_embeddings.grad.abs().sum(dim=1).gt(0).all()
    assert policy.actor_logstd.grad.abs().sum(dim=1).gt(0).all()


def test_plain_mlp_rollout_contract_is_preserved():
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    policy = MLPPolicy(
        obs_dim=3,
        action_dim=2,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
    )
    states = torch.randn(4, 3)
    for mode in ("train", "eval"):
        kwargs = policy.rollout_action_kwargs(mode, {})
        assert kwargs == {"mode": mode, "return_obs": True}
        actions, result = policy.predict_action_batch({"states": states}, **kwargs)
        assert actions.shape == (4, 1, 2)
        torch.testing.assert_close(result["forward_inputs"]["states"], states)


def test_plain_mlp_forward_supports_value_only_and_action_means():
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy

    policy = MLPPolicy(
        obs_dim=3,
        action_dim=2,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
    )
    states = torch.randn(4, 3)
    value_only = policy.default_forward(
        {"states": states},
        compute_logprobs=False,
        compute_entropy=False,
        compute_values=True,
    )
    assert value_only.keys() == {"values"}
    assert value_only["values"].shape == (4, 1)

    action_means = policy.default_forward(
        {"states": states},
        compute_logprobs=False,
        compute_entropy=False,
        compute_values=False,
        compute_action_means=True,
    )
    assert action_means.keys() == {"action_means"}
    assert action_means["action_means"].shape == (4, 2)


def test_registered_sapg_policy_keeps_all_blocks_in_actor_shards():
    from omegaconf import OmegaConf

    from rlinf.data.schema.embodied_types import Trajectory
    from rlinf.models import get_model

    cfg = OmegaConf.create(
        {
            "model_type": "sapg_mlp_policy",
            "precision": "32",
            "load_to_device": False,
            "is_lora": False,
            "obs_dim": 2,
            "action_dim": 2,
            "num_action_chunks": 1,
            "add_value_head": True,
            "sapg": {"num_blocks": 3, "embd_size": 4},
        }
    )
    policy = get_model(cfg)
    env_obs = {"states": torch.zeros(12, 2)}
    sampling = policy.rollout_action_kwargs("train", {})
    _, rollout = policy.predict_action_batch(env_obs, **sampling)
    states = rollout["forward_inputs"]["states"].unsqueeze(0)
    trajectory = Trajectory(
        prev_logprobs=rollout["prev_logprobs"].unsqueeze(0),
        forward_inputs={"states": states},
    )
    for shard in trajectory.split(2):
        block_ids = sapg_block_ids(
            shard.forward_inputs["states"][0], policy.block_codes
        )
        indices = sapg_follower_indices(block_ids, 3, [0, 2])
        assert block_ids[indices].tolist() == [1, 1]
    eval_kwargs = policy.rollout_action_kwargs("eval", {})
    _, evaluation = policy.predict_action_batch(env_obs, **eval_kwargs)
    eval_ids = sapg_block_ids(
        evaluation["forward_inputs"]["states"], policy.block_codes
    )
    assert eval_ids.tolist() == [2] * 12


def test_sapg_joint_probability_loss_preserves_large_off_policy_ratios():
    from rlinf.algorithms.registry import policy_loss

    logprobs = torch.tensor([[2.0, 2.0], [-2.0, -2.0]], requires_grad=True)
    advantages = torch.tensor([[-1.0], [1.0]])
    loss, _ = policy_loss(
        task_type="embodied",
        loss_type="sapg",
        logprob_type="action_level",
        reward_type="action_level",
        single_action_dim=2,
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),
        advantages=advantages,
        values=torch.zeros(2, 1),
        prev_values=torch.zeros(2, 1),
        returns=torch.zeros(2, 1),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=None,
    )
    joint_ratio = logprobs.sum(-1, keepdim=True).exp()
    expected = torch.maximum(
        -advantages * joint_ratio,
        -advantages * joint_ratio.clamp(0.8, 1.2),
    ).mean()
    torch.testing.assert_close(loss, expected)
    actual_grad = torch.autograd.grad(loss, logprobs, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logprobs)[0]
    torch.testing.assert_close(actual_grad, expected_grad)
    assert actual_grad[0].min() > 3.0


def test_sapg_entropy_and_bounds_use_rollout_loss_weights():
    from rlinf.algorithms.registry import policy_loss

    weights = torch.tensor([[2.0], [4.0]])
    zeros = torch.zeros_like(weights)
    loss, _ = policy_loss(
        task_type="embodied",
        loss_type="sapg",
        logprob_type="action_level",
        reward_type="action_level",
        single_action_dim=2,
        logprobs=torch.zeros(2, 2),
        old_logprobs=torch.zeros(2, 2),
        advantages=zeros,
        values=zeros,
        prev_values=zeros,
        returns=zeros,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        value_clip=0.2,
        huber_delta=None,
        loss_mask=torch.ones_like(weights, dtype=torch.bool),
        loss_mask_ratio=weights.reciprocal(),
        entropy=torch.tensor([[1.0, 2.0], [2.0, 4.0]]),
        sapg_entropy_coef=torch.tensor([0.5, 0.0]),
        action_means=torch.tensor([[2.1, -1.1], [-3.1, 0.0]]),
        bounds_loss_coef=0.1,
    )
    expected = -0.5 * 3.0 * 2.0 / 2.0 + 0.1 * (1.0 * 2.0 + 4.0 * 4.0) / 2.0
    torch.testing.assert_close(loss, torch.tensor(expected))


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("algorithm.recompute_logprobs", True, "behaviour-policy"),
        ("algorithm.logprob_type", "token_level", "joint"),
        ("env.train.envs", {"allegro": {}}, "single env.train"),
        ("actor.model.sapg.intrinsic_reward_type", "rnd", "intrinsic_reward_type"),
        ("actor.model.sapg.embedding_end", 50.0, "distinct exploration codes"),
    ],
)
def test_sapg_rejects_configs_that_change_the_algorithm(key, value, message):
    from pathlib import Path

    from omegaconf import OmegaConf

    from rlinf.config import _validate_sapg_train_cfg

    root = Path(__file__).resolve().parents[2]
    cfg = OmegaConf.load(
        root / "examples/embodiment/config/isaaclab_allegro_sapg_mlp.yaml"
    )
    cfg.actor.model = OmegaConf.load(
        root / "examples/embodiment/config/model/sapg_mlp_policy.yaml"
    )
    OmegaConf.update(cfg, key, value)
    # Invalid algorithm settings must fail before hardware placement is needed.
    with pytest.raises(ValueError, match=message):
        _validate_sapg_train_cfg(cfg, cfg.actor.model, None)


@pytest.mark.parametrize("success_count", [0.0, 2.0])
def test_dex_hand_retains_terminal_states_and_counts_actual_success(
    monkeypatch, success_count
):
    from types import SimpleNamespace

    import gymnasium as gym
    from omegaconf import OmegaConf

    from rlinf.envs.sim.isaaclab import isaaclab_env
    from rlinf.envs.sim.isaaclab.tasks.dex_hand import (
        IsaaclabDexHandEnv,
        _DexHandSubprocessAdapter,
        _ExternalResetMixin,
    )

    class VendorHand(gym.Env):
        def __init__(self):
            self.device = torch.device("cpu")
            self.states = torch.zeros(2, 1)
            self.successes = torch.zeros(2)
            self.reset_calls = 0
            self.extras = {}
            self.scene = SimpleNamespace(write_data_to_sim=lambda: None)
            self.sim = SimpleNamespace(forward=lambda: None)

        def _get_observations(self):
            return {"policy": self.states.clone()}

        def _reset_idx(self, env_ids):
            self.reset_calls += 1
            self.states[env_ids] = 0
            self.successes[env_ids] = 0

        def reset(self, *, seed=None, options=None):
            self._reset_idx(torch.arange(2))
            return self._get_observations(), {}

        def step(self, action):
            self.states.fill_(7)
            self.successes.fill_(success_count)
            # DirectRLEnv normally resets before returning the observation.
            self._reset_idx(torch.arange(2))
            return (
                self._get_observations(),
                torch.full((2,), 2.0),
                torch.ones(2, dtype=torch.bool),
                torch.zeros(2, dtype=torch.bool),
                {},
            )

    class ExternallyResetHand(_ExternalResetMixin, VendorHand):
        pass

    simulator = ExternallyResetHand()
    adapter = _DexHandSubprocessAdapter(simulator)

    class LocalSimulatorProcess:
        def __init__(self, env_fn):
            self.adapter = adapter

        def device(self):
            return self.adapter.device

        def reset(self, **kwargs):
            return self.adapter.reset(**kwargs)

        def step(self, actions):
            return self.adapter.step(actions)

    # Replace only the simulator process boundary, retaining the RLinf env logic.
    monkeypatch.setattr(isaaclab_env, "SubProcIsaacLabEnv", LocalSimulatorProcess)
    cfg = OmegaConf.create(
        {
            "init_params": {
                "id": "Isaac-Allegro-Hand-v0",
                "num_envs": 2,
                "task_description": "Reorient",
            },
            "seed": 0,
            "video_cfg": {},
            "auto_reset": True,
            "use_rel_reward": False,
            "ignore_terminations": False,
            "max_episode_steps": 300,
            "reward_coef": 0.25,
        }
    )
    env = IsaaclabDexHandEnv(cfg, 2, 0, 1, None)
    before = simulator.reset_calls
    observations, rewards, _, _, infos = env.chunk_step(torch.zeros(2, 1, 1))
    torch.testing.assert_close(rewards, torch.full((2, 1), 0.5))
    assert simulator.reset_calls - before == 1
    torch.testing.assert_close(
        infos[-1]["final_observation"]["states"], torch.full((2, 1), 7.0)
    )
    torch.testing.assert_close(observations[-1]["states"], torch.zeros(2, 1))
    episode = infos[-1]["final_info"]["episode"]
    assert episode["success_once"].tolist() == [success_count > 0] * 2
    torch.testing.assert_close(
        episode["success_count"], torch.full((2,), success_count)
    )
