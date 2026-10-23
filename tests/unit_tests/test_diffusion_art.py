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

"""DiffusionART tensor-kernel tests."""

import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_openpi_art_cfg
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.utils.nested_dict_process import process_nested_dict_for_train
from rlinf.workers.actor.fsdp_art_policy_worker import (
    EmbodiedARTFSDPPolicy,
    _art_loss,
    _art_scalar_covariance,
    _art_trajectory_guidance,
    _prompt_disjoint_windows,
    _validate_art_rollout,
)


def test_art_velocity_target_validation_and_prompt_windows(monkeypatch):
    """The supported ART path keeps its sign and validates outside the hot path."""
    assert tuple(inspect.signature(art_loss).parameters) == (
        "current_velocity",
        "old_velocity",
        "sigma",
        "sigma_next",
        "guidance_direction",
        "guidance_magnitude",
        "loss_space",
        "mask_metric",
        "mask_threshold",
        "reference_noise_level",
        "corrective_mask_enabled",
        "coordinate_mask",
    )
    states = torch.zeros(4, 2, 2)
    sigmas = torch.tensor([[1.0, 0.5]]).expand(4, -1)
    sigmas_next = torch.tensor([[0.5, 0.0]]).expand(4, -1)
    magnitudes = torch.full((4,), 0.25)
    validate_art_rollout(states, sigmas, sigmas_next, magnitudes)

    old = torch.zeros(4, 2)
    current = old.clone().requires_grad_(True)
    direction = torch.tensor([[1.0, -1.0]]).expand_as(old)

    def fail_hot_path_scan(*args, **kwargs):
        raise AssertionError("art_loss must not scan tensor values in the hot path")

    with monkeypatch.context() as context:
        context.setattr(torch, "isfinite", fail_hot_path_scan)
        terms = art_loss(
            current,
            old,
            sigmas[:, 1],
            sigmas_next[:, 1],
            direction,
            magnitudes,
            mask_metric="fixed_mse",
            mask_threshold=1.0,
        )

    expected_target = old - magnitudes[:, None] * direction
    torch.testing.assert_close(terms["target"], expected_target)
    torch.testing.assert_close(
        terms["per_item_loss"],
        sigmas[:, 1] * (expected_target - current).square().mean(1),
    )
    terms["per_item_loss"].sum().backward()
    assert torch.all(current.grad * expected_target < 0)

    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    order = [3, 4, 0, 7, 2, 5, 1, 6]
    windows = prompt_disjoint_windows(groups, updates=2, order=order)
    assert windows == ((3, 4, 0, 7), (2, 5, 1, 6))
    assert set(windows[0]).isdisjoint(windows[1])


def test_art_rollout_rejects_non_decreasing_sigma():
    states = torch.zeros(1, 2, 2)
    with pytest.raises(ValueError, match="sigma_next < sigma"):
        validate_art_rollout(
            states,
            torch.tensor([[1.0, 0.5]]),
            torch.tensor([[0.5, 0.5]]),
            torch.ones(1),
        )


def test_art_complete_group_covariance_and_constant_rewards():
    """Guidance uses complete prompt groups and freezes constant-reward groups."""
    endpoints = torch.tensor(
        [[0.0, 0.0], [2.0, 4.0], [10.0, 5.0], [14.0, 9.0], [7.0, -3.0], [9.0, 5.0]],
        requires_grad=True,
    )
    rewards = torch.tensor([0.0, 2.0, 1.0, 3.0, 0.1, 0.1], requires_grad=True)
    groups = torch.tensor([0, 0, 1, 1, 2, 2])
    stats = art_scalar_covariance(
        endpoints,
        rewards,
        groups,
        group_count=3,
        zero_signal_mode="exact",
    )

    covariance = torch.tensor([[2.0, 4.0], [4.0, 4.0], [0.0, 0.0]])
    direction = covariance / covariance.square().mean(1, keepdim=True).sqrt().clamp_min(
        1e-5
    )
    torch.testing.assert_close(stats["group_covariance"], covariance)
    torch.testing.assert_close(stats["group_directions"], direction)
    torch.testing.assert_close(
        stats["group_guidance_magnitudes"], torch.tensor([1.0, 1.0, 0.0])
    )
    torch.testing.assert_close(stats["group_active"], torch.tensor([True, True, False]))
    torch.testing.assert_close(stats["group_counts"], torch.full((3,), 2.0))
    assert all(not value.requires_grad for value in stats.values())


def test_art_covariance_uses_only_executed_coordinates():
    endpoints = torch.tensor(
        [[[0.0], [100.0]], [[2.0], [4.0]], [[999.0], [8.0]], [[123.0], [888.0]]]
    )
    mask = torch.tensor([[True, False], [True, True], [False, True], [False, False]])
    stats = _art_scalar_covariance(
        endpoints,
        torch.tensor([0.0, 1.0, 2.0, 999.0]),
        torch.zeros(4, dtype=torch.long),
        1,
        coordinate_mask=mask[..., None],
        zero_signal_mode="exact",
        direction_mode="normalized_aggregate",
    )
    torch.testing.assert_close(
        stats["group_covariance"], torch.tensor([[[1.0], [2.0]]])
    )
    torch.testing.assert_close(
        stats["group_coordinate_counts"], torch.tensor([[[2.0], [2.0]]])
    )
    torch.testing.assert_close(stats["group_counts"], torch.tensor([3.0]))
    torch.testing.assert_close(
        stats["group_directions"].square().mean(), torch.tensor(1.0)
    )


@pytest.mark.parametrize("valid_count", [0, 1, 4])
def test_art_constant_and_singleton_action_groups_have_zero_guidance(valid_count):
    mask = torch.arange(4) < valid_count
    stats = _art_scalar_covariance(
        torch.arange(8.0).reshape(4, 2),
        torch.full((4,), 0.1),
        torch.zeros(4, dtype=torch.long),
        1,
        coordinate_mask=mask[:, None],
        zero_signal_mode="exact",
    )
    assert not stats["group_active"].any()
    assert stats["group_guidance_magnitudes"].eq(0).all()
    assert stats["group_directions"].eq(0).all()
    assert all(torch.isfinite(value).all() for value in stats.values())


def test_art_trajectory_groups_keep_decisions_and_reset_groups_separate():
    endpoints = torch.tensor([0.0, 2.0, 100.0, 104.0, 2.0, 0.0, 104.0, 100.0]).reshape(
        2, 4, 1, 1
    )
    guidance = _art_trajectory_guidance(
        endpoints,
        torch.tensor([0.0, 1.0, 1.0, 1.0]),
        torch.ones(2, 4, 1, dtype=torch.bool),
        2,
        zero_signal_mode="exact",
    )
    torch.testing.assert_close(
        guidance["group_ids"], torch.tensor([[0, 0, 1, 1], [2, 2, 3, 3]])
    )
    torch.testing.assert_close(guidance["directions"][0, :2], torch.ones(2, 1, 1))
    torch.testing.assert_close(guidance["directions"][1, :2], -torch.ones(2, 1, 1))
    assert guidance["magnitudes"][:, 2:].eq(0).all()
    assert guidance["episode_ids"][0].equal(guidance["episode_ids"][1])


def test_art_loss_masks_padding_and_keeps_inward_restoration():
    old = torch.zeros(2, 2, 3)
    current = torch.tensor(
        [
            [[-0.5, -0.5, 999.0], [999.0, 999.0, 999.0]],
            [[-2.0, -2.0, 999.0], [999.0, 999.0, 999.0]],
        ],
        requires_grad=True,
    )
    mask = torch.tensor([[[True, True, False], [False, False, False]]]).expand_as(old)
    terms = _art_loss(
        current,
        old,
        torch.ones(2),
        torch.full((2,), 0.5),
        torch.ones_like(old),
        torch.ones(2),
        mask_metric="fixed_mse",
        mask_threshold=0.1,
        coordinate_mask=mask,
    )
    assert terms["masked"].tolist() == [True, False]
    torch.testing.assert_close(terms["movement_energy"], torch.tensor([0.25, 4.0]))
    torch.testing.assert_close(terms["per_item_loss"], torch.tensor([0.0, 1.0]))
    terms["per_item_loss"].sum().backward()
    assert current.grad[~mask].eq(0).all()
    assert current.grad[1, 0, :2].lt(0).all()
    assert not old.requires_grad


def _art_trajectory_worker(first_chunk_only=False):
    worker = object.__new__(EmbodiedARTFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": 2,
                "art_updates_per_rollout": 2,
                "art_first_chunk_only": first_chunk_only,
                "art_timestep_fraction": 0.5,
            },
        }
    )
    decisions, episodes, actions, dims, timesteps = 3, 4, 2, 2, 4
    endpoints = torch.arange(float(decisions * episodes * actions * dims)).reshape(
        decisions, episodes, actions, dims
    )
    dones = torch.zeros(decisions + 1, episodes, actions, dtype=torch.bool)
    dones[1, 0, 0] = True
    rewards = torch.zeros(decisions, episodes, actions)
    rewards[0, 0, 0] = 1.0
    rewards[1:, 0] = 999.0
    rewards[-1, -1, -1] = 1.0
    states = (
        torch.arange(float(timesteps))[None, None, :, None, None]
        .expand(decisions, episodes, timesteps, 4, 3)
        .clone()
    )
    worker.rollout_batch = {
        "rewards": rewards,
        "dones": dones,
        "loss_mask": compute_loss_mask(dones)[0].any(-1, keepdim=True),
        "forward_inputs": {
            "art_endpoint": endpoints,
            "art_states": states,
            "art_old_velocity": (states[..., :actions, :dims] + 10).clone(),
            "art_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25])
            .expand(decisions, episodes, -1)
            .clone(),
            "art_sigmas_next": torch.tensor([0.75, 0.5, 0.25, 0.0])
            .expand(decisions, episodes, -1)
            .clone(),
            "observation/state": torch.arange(episodes)[None, :, None].expand(
                decisions, -1, -1
            ),
        },
    }
    return worker


@pytest.mark.parametrize("first_chunk_only", [False, True])
def test_art_trajectory_preparation_preserves_episode_windows_and_cached_velocities(
    first_chunk_only,
):
    worker = _art_trajectory_worker(first_chunk_only)
    generator = torch.Generator().manual_seed(7)
    metrics, active, windows = worker._prepare_art_trajectory_training_inputs(generator)
    assert active
    assert metrics["actor/art_reward_mean"] == 0.5
    assert metrics["actor/art_reward_std"] == 0.5
    assert set(windows[0].tolist()).isdisjoint(windows[1].tolist())
    assert set((windows[0] % 4).tolist()).isdisjoint((windows[1] % 4).tolist())
    assert all(len(set((window % 4 // 2).tolist())) == 2 for window in windows)
    rows = sum(len(window) for window in windows)
    assert rows == (4 if first_chunk_only else 12)
    worker.rollout_batch = process_nested_dict_for_train(
        worker.rollout_batch, torch.arange(rows)
    )
    worker._select_art_training_timesteps(generator)
    inputs = worker.rollout_batch["forward_inputs"]
    selected = inputs["art_states"][..., 0, 0]
    torch.testing.assert_close(inputs["art_old_velocity"][..., 0, 0], selected + 10)
    torch.testing.assert_close(inputs["art_sigmas"], 1 - selected / 4)
    torch.testing.assert_close(inputs["art_sigmas_next"], 0.75 - selected / 4)
    assert selected.shape == (rows, 2)
    for window in windows:
        batch = worker._index_nested(worker.rollout_batch, window)
        replay = worker._model_forward_inputs(batch["forward_inputs"])
        torch.testing.assert_close(
            replay["observation/state"][:, 0],
            batch["forward_inputs"]["art_episode_ids"],
        )
        assert not any(key.startswith("art_") for key in replay)
    if not first_chunk_only:
        assert not inputs["art_coordinate_mask"][4, ...].any()


def test_openpi_art_recipe_composes_with_matching_batch_budget(monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="libero_object_0_art_openpi")
    validate_openpi_art_cfg(cfg)
    assert cfg.actor.global_batch_size == (
        cfg.env.train.total_num_envs
        * cfg.env.train.rollout_epoch
        * (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        // cfg.algorithm.art_updates_per_rollout
    )
    assert cfg.actor.model.openpi.is_nft is False
    assert cfg.actor.model.num_steps == 4


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("env.train.auto_reset", True, "auto_reset=False"),
        (
            "env.train.use_fixed_reset_state_ids",
            False,
            "use_fixed_reset_state_ids=True",
        ),
        ("actor.model.openpi.is_nft", True, "is_nft=False"),
        ("actor.model.openpi.noise_method", "flow_sde", "flow_ode"),
        ("actor.model.openpi.action_chunk", 10, "match num_action_chunks"),
        ("actor.model.openpi.action_env_dim", 6, "action_env_dim=action_dim"),
        ("actor.model.openpi.model_action_dim", 6, "action_env_dim=action_dim"),
        ("algorithm.art_reward_reduction", "mean", "episode returns"),
        ("env.train.max_steps_per_rollout_epoch", 100, "complete episodes"),
    ],
)
def test_openpi_art_rejects_incompatible_rollouts(key, value, message, monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name="libero_object_0_art_openpi")
    OmegaConf.update(cfg, key, value)
    with pytest.raises(AssertionError, match=message):
        validate_openpi_art_cfg(cfg)


def test_openpi_art_rollout_replay_and_two_frozen_target_updates():
    from rlinf.models.embodiment.openpi.modules.siglip import SigLIPViT
    from rlinf.models.embodiment.openpi.modules.utils import (
        set_torch_compile,
        torch_compile_enabled,
    )
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig

    compiled = torch_compile_enabled()
    threads = torch.get_num_threads()
    set_torch_compile(False)
    torch.set_num_threads(1)
    try:
        config = Pi0Config(
            dtype="float32",
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_dim=4,
            action_horizon=4,
            max_token_len=4,
        )
        # Meta construction avoids allocating the production vision encoder.
        with torch.device("meta"):
            model = Pi0RL(
                config,
                num_steps=4,
                action_chunk=2,
                action_env_dim=2,
                rl_cfg=Pi0RLConfig(rl_mode="art", train_expert_only=True),
            )
        model.img = SigLIPViT(variant="mu/14", num_classes=64)
        model.to_empty(device="cpu")
        with torch.no_grad():
            generator = torch.Generator().manual_seed(31)
            for parameter in model.parameters():
                parameter.normal_(std=0.02, generator=generator)
        model._input_transform_fn = lambda data: data
        model._output_transform_fn = lambda data: {
            "actions": data["actions"][..., :2] * 10
        }
        observation = config.fake_obs(batch_size=2)
        noise = torch.randn(2, 4, 4, generator=torch.Generator().manual_seed(32))
        model.eval()
        with torch.no_grad():
            actions, result = model._predict_train(
                observation, noise=noise, rng=None, compute_values=False
            )
        inputs = result["forward_inputs"]
        assert inputs["art_states"].shape == (2, 4, 4, 4)
        assert inputs["art_old_velocity"].shape == (2, 4, 2, 2)
        torch.testing.assert_close(inputs["art_endpoint"], actions / 10)
        torch.testing.assert_close(inputs["art_states"][:, 0], noise)
        for index in range(3):
            expected = (
                inputs["art_states"][:, index, :2, :2]
                - 0.25 * inputs["art_old_velocity"][:, index]
            )
            torch.testing.assert_close(
                inputs["art_states"][:, index + 1, :2, :2], expected
            )
        model.rl_cfg = replace(model.rl_cfg, rl_mode="default", is_nft=True)
        with torch.no_grad():
            nft_actions, nft_result = model._predict_train(
                observation, noise=noise, rng=None, compute_values=False
            )
        torch.testing.assert_close(nft_actions, actions)
        assert "nft_x0" in nft_result["forward_inputs"]
        assert "art_endpoint" not in nft_result["forward_inputs"]
        model.rl_cfg = replace(model.rl_cfg, rl_mode="art", is_nft=False)
        for name, value in observation.images.items():
            inputs[f"obs_image__{name}"] = value
            inputs[f"obs_image_mask__{name}"] = observation.image_masks[name]
        inputs["obs_state"] = observation.state
        replay = EmbodiedARTFSDPPolicy._model_forward_inputs(inputs)
        with torch.no_grad():
            output = model.nft_forward(
                replay,
                nft_inputs={
                    "x_t": inputs["art_states"][:, 1],
                    "timesteps": inputs["art_sigmas"][:, 1],
                },
            )
        torch.testing.assert_close(
            output["v_theta"][..., :2], inputs["art_old_velocity"][:, 1]
        )
        guidance = _art_trajectory_guidance(
            inputs["art_endpoint"][None],
            torch.tensor([0.0, 1.0]),
            torch.ones(1, 2, 2, dtype=torch.bool),
            2,
            zero_signal_mode="exact",
        )
        old = inputs["art_old_velocity"].clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        model.train()
        for row in (0, 1):
            replay = EmbodiedARTFSDPPolicy._model_forward_inputs(
                inputs, torch.tensor([row])
            )
            output = model.nft_forward(
                replay,
                nft_inputs={
                    "x_t": inputs["art_states"][row : row + 1, 1],
                    "timesteps": inputs["art_sigmas"][row : row + 1, 1],
                },
            )
            terms = _art_loss(
                output["v_theta"][..., :2],
                old[row : row + 1, 1],
                inputs["art_sigmas"][row : row + 1, 1],
                inputs["art_sigmas_next"][row : row + 1, 1],
                guidance["directions"][0, row : row + 1],
                guidance["magnitudes"][0, row : row + 1],
                corrective_mask_enabled=False,
            )
            optimizer.zero_grad()
            terms["per_item_loss"].mean().backward()
            assert model.action_out_proj.weight.grad is not None
            assert torch.isfinite(model.action_out_proj.weight.grad).all()
            assert model.action_out_proj.weight.grad.abs().sum() > 0
            previous_weight = model.action_out_proj.weight.detach().clone()
            optimizer.step()
            assert not torch.equal(model.action_out_proj.weight, previous_weight)
            torch.testing.assert_close(inputs["art_old_velocity"], old)
    finally:
        set_torch_compile(compiled)
        torch.set_num_threads(threads)
