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
import os
import subprocess
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.diffusion.art import (
    art_loss,
    art_scalar_covariance,
    art_trajectory_guidance,
    compute_art_loss,
    prepare_art_training_inputs,
    prompt_disjoint_windows,
    validate_art_rollout,
)
from rlinf.config import validate_cfg, validate_openpi_art_cfg
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.metric_utils import compute_loss_mask
from rlinf.workers.actor.fsdp_art_policy_worker import (
    EmbodiedARTFSDPPolicy,
    _art_mask_count_metrics,
)
from rlinf.workers.rollout.hf.art_huggingface_worker import ARTMultiStepRolloutWorker


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
    stats = art_scalar_covariance(
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
    stats = art_scalar_covariance(
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
    guidance = art_trajectory_guidance(
        endpoints,
        torch.tensor([0.0, 1.0, 1.0, 1.0]),
        torch.ones(2, 4, 1, dtype=torch.bool),
        2,
        zero_signal_mode="exact",
    )
    assert guidance["stats"]["group_active"].tolist() == [True, False, True, False]
    torch.testing.assert_close(guidance["directions"][0, :2], torch.ones(2, 1, 1))
    torch.testing.assert_close(guidance["directions"][1, :2], -torch.ones(2, 1, 1))
    assert guidance["magnitudes"][:, 2:].eq(0).all()


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
    terms = art_loss(
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


@pytest.mark.parametrize("loss_space", ["velocity_linear_sigma", "velocity_uniform"])
def test_art_loss_gradient_uses_only_executed_coordinates(loss_space):
    old = torch.zeros(5, 2)
    current = torch.tensor(
        [[-0.5, 999.0], [-2.0, -2.0], [1.0, 999.0], [2.0, 999.0], [0.5, 0.25]],
        requires_grad=True,
    )
    mask = torch.tensor(
        [[True, False], [True, True], [True, False], [True, False], [True, True]]
    )
    sigma = torch.tensor([1.0, 0.5, 0.25, 0.0, 0.75])
    magnitude = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    direction = torch.ones_like(old)
    terms = art_loss(
        current,
        old,
        sigma,
        sigma / 2,
        direction,
        magnitude,
        loss_space=loss_space,
        mask_metric="fixed_mse",
        mask_threshold=0.1,
        coordinate_mask=mask,
    )
    assert terms["keep_mask"].tolist() == [False, True, False, False, True]
    weight = sigma if loss_space == "velocity_linear_sigma" else torch.ones_like(sigma)
    scale = 2 * weight * terms["keep_mask"] / (len(current) * mask.sum(1))
    guidance_gradient = scale[:, None] * torch.where(
        mask, magnitude[:, None] * direction, 0.0
    )
    restoration_gradient = scale[:, None] * torch.where(
        mask, current.detach() - old, 0.0
    )
    total_gradient = torch.autograd.grad(terms["per_item_loss"].mean(), current)[0]
    torch.testing.assert_close(total_gradient, guidance_gradient + restoration_gradient)
    assert torch.count_nonzero(total_gradient[~mask]) == 0


def _art_trajectory_worker():
    worker = object.__new__(EmbodiedARTFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": 2,
                "art_updates_per_rollout": 2,
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
            "art_anchor": torch.zeros_like(endpoints),
            "art_states": states,
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


def test_art_trajectory_preparation_preserves_episode_windows_and_timesteps():
    worker = _art_trajectory_worker()
    generator = torch.Generator().manual_seed(7)
    metrics, active, windows = worker._prepare_art_trajectory_training_inputs(generator)
    assert active
    assert metrics["actor/art_reward_mean"] == 0.5
    assert metrics["actor/art_reward_std"] == 0.5
    assert set(windows[0].tolist()).isdisjoint(windows[1].tolist())
    assert set((windows[0] % 4).tolist()).isdisjoint((windows[1] % 4).tolist())
    assert all(len(set((window % 4 // 2).tolist())) == 2 for window in windows)
    rows = sum(len(window) for window in windows)
    assert rows == 12
    worker._select_art_training_timesteps(generator)
    inputs = worker.rollout_batch["forward_inputs"]
    selected = inputs["art_states"][..., 0, 0]
    assert "art_old_velocity" not in inputs
    torch.testing.assert_close(inputs["art_sigmas"], 1 - selected / 4)
    torch.testing.assert_close(inputs["art_sigmas_next"], 0.75 - selected / 4)
    assert selected.shape == (rows, 2)
    for window in windows:
        batch = worker._index_nested(worker.rollout_batch, window)
        replay = worker._model_forward_inputs(batch["forward_inputs"])
        # Flattened rows are decision-major, so the episode is the row modulo 4.
        assert replay["observation/state"][:, 0].tolist() == (window % 4).tolist()
        assert not any(key.startswith("art_") for key in replay)
    assert not inputs["art_coordinate_mask"][4, ...].any()


@pytest.fixture(params=[2, 4])
def art_config_cluster(monkeypatch, request):
    node = NodeInfo(
        node_labels=[],
        node_rank=0,
        ray_id="test-node",
        node_ip="127.0.0.1",
        num_cpus=32,
        python_interpreter_path="/usr/bin/python3",
        default_env_vars={},
        env_vars={},
        hardware_resources=[
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[
                    HardwareInfo(type=Accelerator.HW_TYPE, model="NV_GPU:Mock")
                    for _ in range(request.param)
                ],
            )
        ],
    )
    group = NodeGroupInfo(
        label=NodeGroupInfo.DEFAULT_GROUP_LABEL,
        nodes=[node],
        hardware_type=Accelerator.HW_TYPE,
    )
    cluster = SimpleNamespace(get_node_group=lambda: group)
    monkeypatch.setattr("rlinf.config.Cluster", lambda **kwargs: cluster)


@pytest.mark.parametrize(
    ("config_name", "suite"),
    [
        (f"libero_{suite}_0_art{mode}_openpi", f"libero_{suite}")
        for suite in ("object", "spatial")
        for mode in ("", "_cluster")
    ],
)
def test_openpi_art_recipe_composes_with_matching_batch_budget(
    monkeypatch, art_config_cluster, config_name, suite
):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=config_name)
    cfg = validate_cfg(cfg)
    assert cfg.env.train.task_suite_name == cfg.env.eval.task_suite_name == suite
    assert cfg.env.train.task_id_filter == cfg.env.eval.task_id_filter == [0]
    assert cfg.env.train.total_num_envs == 128
    assert (
        cfg.actor.global_batch_size
        == (
            cfg.env.train.total_num_envs
            * cfg.env.train.rollout_epoch
            * 48
            // cfg.algorithm.art_updates_per_rollout
        )
        == 3072
    )
    assert cfg.actor.model.openpi.is_nft is False
    assert cfg.actor.model.num_steps == 4
    assert cfg.algorithm.art_timestep_fraction == 1.0
    assert cfg.actor.optim.use_fp32_master_params
    assert (
        cfg.env.train.init_params.ignore_done and cfg.env.eval.init_params.ignore_done
    )
    assert cfg.rollout.seed == cfg.actor.seed
    assert cfg.env.train.art_grouped_reset
    assert cfg.algorithm.art_grouping == (
        "cluster" if "_cluster" in config_name else "index"
    )


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


def _tiny_art_pi0():
    """Build a CPU Pi0 ART policy whose actions are its endpoints times ten."""
    from rlinf.models.embodiment.openpi.modules.siglip import SigLIPViT
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi.tasks.rl import Pi0RL, Pi0RLConfig

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
    model._output_transform_fn = lambda data: {"actions": data["actions"][..., :2] * 10}
    return model, config


@pytest.fixture
def single_thread_eager_torch():
    from rlinf.models.embodiment.openpi.modules.utils import (
        set_torch_compile,
        torch_compile_enabled,
    )

    compiled = torch_compile_enabled()
    threads = torch.get_num_threads()
    set_torch_compile(False)
    torch.set_num_threads(1)
    yield
    set_torch_compile(compiled)
    torch.set_num_threads(threads)


class _VelocityPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(4, 4, 3))

    def forward(self, forward_type, forward_inputs, art_inputs, compute_values):
        ids = forward_inputs["observation/state"][:, 0].long()
        assert forward_type == ForwardType.ART
        return {"v_theta": art_inputs["x_t"] + self.offset[ids]}


@pytest.fixture
def cpu_art_backend(monkeypatch):
    # Replace GPU/distributed boundaries only; execute the real worker kernels,
    # loss, optimizer, scheduler, and reference construction below.
    platform = SimpleNamespace(
        current_device=lambda: torch.device("cpu"),
        synchronize=lambda: None,
        ipc_collect=lambda: None,
        empty_cache=lambda: None,
    )
    monkeypatch.setattr(Worker, "torch_platform", platform)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 1)
    return platform


def _cpu_art_actor(platform):
    actor = object.__new__(EmbodiedARTFSDPPolicy)
    actor._rank, actor._world_size, actor.version = 0, 1, 0
    actor.device, actor.torch_platform = torch.device("cpu"), platform
    actor.cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": 2,
                "art_updates_per_rollout": 2,
                "art_timestep_fraction": 1.0,
            },
            "actor": {
                "seed": 42,
                "micro_batch_size": 1,
                "global_batch_size": 2,
                "model": {
                    "model_type": "openpi",
                    "num_action_chunks": 2,
                    "action_dim": 2,
                },
            },
            "env": {"train": {"max_episode_steps": 100}},
        }
    )
    states = torch.arange(4.0).view(1, 1, 4, 1, 1).expand(1, 4, 4, 4, 3).clone()
    inputs = {
        "art_endpoint": torch.arange(16.0).reshape(1, 4, 2, 2),
        "art_anchor": torch.zeros(1, 4, 2, 2),
        "art_states": states,
        "art_old_velocity": states + 10,
        "art_sigmas": torch.tensor([1.0, 0.75, 0.5, 0.25]).expand(1, 4, 4),
        "art_sigmas_next": torch.tensor([0.75, 0.5, 0.25, 0.0]).expand(1, 4, 4),
        "observation/state": torch.arange(4).view(1, 4, 1),
    }
    actor.rollout_batch = {
        "forward_inputs": inputs,
        "rewards": torch.tensor([[[1.0, 0], [0, 0], [0, 0], [0, 1.0]]]),
        "dones": torch.zeros(2, 4, 2, dtype=torch.bool),
    }
    actor.model = _VelocityPolicy()
    actor.optimizer = torch.optim.AdamW(
        actor.model.parameters(), lr=0.01, weight_decay=0
    )
    actor.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        actor.optimizer, lambda step: 1
    )
    actor.amp_context = nullcontext()
    actor.grad_scaler = torch.amp.GradScaler("cpu", enabled=False)
    actor.is_weight_offloaded, actor.is_optimizer_offloaded = False, False
    actor.before_micro_batch = lambda *args, **kwargs: nullcontext()

    def optimizer_step():
        norm = torch.nn.utils.clip_grad_norm_(actor.model.parameters(), 1)
        actor.optimizer.step()
        return norm.item(), [0.01]

    actor.optimizer_step = optimizer_step
    return actor


def test_art_reference_is_frozen_and_scheduler_steps_once(cpu_art_backend):
    actor = _cpu_art_actor(cpu_art_backend)
    metrics = inspect.unwrap(EmbodiedARTFSDPPolicy.run_training)(actor)
    inputs = actor.rollout_batch["forward_inputs"]
    torch.testing.assert_close(inputs["art_old_velocity"], inputs["art_states"])
    assert actor.lr_scheduler.last_epoch == 1
    assert actor.optimizer.state[actor.model.offset]["step"].item() == 2
    assert metrics["actor/active_count"] > 0
    assert metrics["actor/u1/mask_frac_global"] == 0
    assert torch.count_nonzero(actor.model.offset[:, :2, :2]) > 0
    assert torch.count_nonzero(actor.model.offset[:, 2:]) == 0
    assert torch.count_nonzero(actor.model.offset[:, :, 2:]) == 0


def test_art_mask_ratios_pool_only_active_items():
    empty = torch.zeros(4)
    active = torch.tensor([2.0, 1.0, 1.0, 0.04])
    result = _art_mask_count_metrics(empty + active, "actor")
    assert result["actor/mask_frac_global"] == 0.5
    assert result["actor/movement_score_active"] == pytest.approx(0.02)
    assert _art_mask_count_metrics(empty, "actor")["actor/mask_frac_global"] == 0


@pytest.mark.parametrize("exclude_final", [False, True])
@pytest.mark.parametrize("reward_reduction", ["mean", "sum"])
def test_art_preparation_builds_guidance_and_preserves_timestep_rng_order(
    exclude_final, reward_reduction
):
    states = torch.arange(8 * 4 * 2).reshape(8, 4, 2).float()
    sigmas = torch.tensor([[1.0, 0.75, 0.5, 0.25]]).expand(8, -1)
    sigmas_next = torch.tensor([[0.75, 0.5, 0.25, 0.0]]).expand(8, -1)
    endpoints = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]] * 2)
    rewards = torch.tensor([[0.0], [1.0], [2.0], [3.0], [0.1], [0.1], [0.1], [0.1]])
    rewards = torch.cat((rewards, rewards * 2), dim=1)
    prompts = torch.ones(8, 2, 3)
    inputs = {
        "art_endpoint": endpoints,
        "art_states": states,
        "art_sigmas": sigmas,
        "art_sigmas_next": sigmas_next,
        "prompt_embeds": prompts,
    }
    config = OmegaConf.create(
        {
            "group_size": 4,
            "art_timestep_fraction": 0.5,
            "art_exclude_final_step": exclude_final,
            "art_reward_reduction": reward_reduction,
        }
    )
    pool = 3 if exclude_final else 4
    count = max(1, int(pool * 0.5))
    expected_generator = torch.Generator().manual_seed(37)
    expected_steps = torch.stack(
        [torch.randperm(pool, generator=expected_generator)[:count] for _ in range(8)]
    )
    generator = torch.Generator().manual_seed(37)
    metrics, has_guidance = prepare_art_training_inputs(
        inputs, rewards, config=config, generator=generator
    )
    assert has_guidance
    assert "art_endpoint" not in inputs
    assert inputs["prompt_embeds"] is prompts
    rows = torch.arange(8)[:, None]
    torch.testing.assert_close(inputs["art_states"], states[rows, expected_steps])
    torch.testing.assert_close(inputs["art_sigmas"], sigmas[rows, expected_steps])
    torch.testing.assert_close(
        inputs["art_sigmas_next"], sigmas_next[rows, expected_steps]
    )
    torch.testing.assert_close(generator.get_state(), expected_generator.get_state())
    direction = torch.tensor([1.0, 2.0]) / torch.tensor(2.5).sqrt()
    torch.testing.assert_close(
        inputs["art_guidance_direction"][:4], direction.expand(4, -1)
    )
    torch.testing.assert_close(inputs["art_guidance_direction"][4:], torch.zeros(4, 2))
    torch.testing.assert_close(
        inputs["art_guidance_magnitude"], torch.tensor([1.0] * 4 + [0.0] * 4)
    )
    assert metrics["actor/active_group_frac"] == 0.5
    reduced_rewards = rewards.sum(1) if reward_reduction == "sum" else rewards.mean(1)
    assert metrics["actor/art_reward_mean"] == pytest.approx(
        reduced_rewards.mean().item()
    )
    assert metrics["actor/art_reward_std"] == pytest.approx(
        reduced_rewards.std(correction=0).item()
    )


def test_art_preparation_reports_zero_guidance_for_constant_rewards():
    inputs = {
        "art_endpoint": torch.arange(8).reshape(4, 2).float(),
        "art_states": torch.ones(4, 1, 2),
        "art_sigmas": torch.ones(4, 1),
        "art_sigmas_next": torch.zeros(4, 1),
    }
    metrics, active = prepare_art_training_inputs(
        inputs,
        torch.full((4, 1), 0.1),
        config=OmegaConf.create({"group_size": 4, "art_exclude_final_step": True}),
        generator=torch.Generator().manual_seed(0),
    )
    assert not active
    assert metrics["actor/active_group_frac"] == 0
    assert metrics["actor/covariance_rms"] == 0
    assert metrics["actor/guidance_rms"] == 0
    assert inputs["art_states"].shape == (4, 1, 2)


@pytest.mark.parametrize("mask_metric", ["sde_kl", "fixed_mse"])
@pytest.mark.parametrize("loss_space", ["velocity_linear_sigma", "velocity_uniform"])
@pytest.mark.parametrize("mask_enabled", [False, True])
def test_art_objective_freezes_targets_and_masks_only_active_outward_movement(
    mask_metric, loss_space, mask_enabled
):
    current = torch.tensor([[-0.2, -0.2], [0.2, 0.2], [-0.2, -0.2]], requires_grad=True)
    old = torch.zeros_like(current, requires_grad=True)
    direction = torch.ones_like(current, requires_grad=True)
    magnitude = torch.tensor([1.0, 1.0, 0.0], requires_grad=True)
    sigma = torch.full((3,), 0.5)
    config = OmegaConf.create(
        {
            "art_mask_metric": mask_metric,
            "art_loss_space": loss_space,
            "art_corrective_mask_enabled": mask_enabled,
        }
    )
    loss, metrics = compute_art_loss(
        current,
        old,
        sigma,
        torch.zeros_like(sigma),
        direction,
        magnitude,
        config=config,
    )
    weight = 0.5 if loss_space == "velocity_linear_sigma" else 1.0
    unmasked = (current + magnitude.detach()[:, None]).square().mean(1) * weight
    expected = (unmasked[1] if mask_enabled else unmasked[:2].sum()) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert metrics["actor/art_loss"] == pytest.approx(expected.item())
    assert metrics["actor/active_count"] == 2
    assert metrics["actor/unmasked_loss"] == pytest.approx(unmasked[:2].mean().item())
    assert metrics["actor/mask_frac"] == (0.5 if mask_enabled else 0)
    assert torch.count_nonzero(current.grad[2]) == 0
    assert (torch.count_nonzero(current.grad[0]) == 0) == mask_enabled
    assert torch.count_nonzero(current.grad[1]) == 2
    assert old.grad is None
    assert direction.grad is None
    assert magnitude.grad is None


@pytest.mark.parametrize("updates", [1, 2, 4])
def test_art_prompt_windows_cover_each_row_once_and_balance_each_group(updates):
    groups = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    order = [3, 4, 0, 7, 2, 5, 1, 6]
    windows = prompt_disjoint_windows(groups, updates=updates, order=order)
    assert sorted(row for window in windows for row in window) == list(range(8))
    for window in windows:
        torch.testing.assert_close(
            torch.bincount(groups[list(window)]), torch.full((2,), 4 // updates)
        )


@pytest.mark.parametrize(
    ("model_type", "latent_shape"), [("sd3", (4, 3)), ("wan22_ti2v_5b", (2, 4, 3))]
)
def test_image_video_art_keep_full_latents_and_recompute_the_reference(
    cpu_art_backend, model_type, latent_shape
):
    class LatentVelocityPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = torch.nn.Parameter(torch.zeros(4, *latent_shape))

        def forward(self, forward_type, forward_inputs, art_inputs, compute_values):
            ids = forward_inputs["observation/state"][:, 0].long()
            assert forward_type == ForwardType.ART
            return {"v_theta": art_inputs["x_t"] + self.offset[ids]}

    actor = _cpu_art_actor(cpu_art_backend)
    actor.cfg.actor.model.model_type = model_type
    actor.cfg.algorithm.art_timestep_fraction = 0.5
    inputs = actor.rollout_batch["forward_inputs"]
    states = torch.arange(4.0).reshape(1, 4, 1, *([1] * len(latent_shape)))
    inputs["art_states"] = states.expand(1, 4, 4, *latent_shape).clone()
    inputs["art_old_velocity"] = inputs["art_states"] + 10
    inputs["art_endpoint"] = inputs["art_states"][:, :, 0].clone()
    actor.model = LatentVelocityPolicy()
    actor.optimizer = torch.optim.AdamW(actor.model.parameters(), lr=0.01)
    actor.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(actor.optimizer, lambda _: 1)
    before = actor.model.offset.detach().clone()
    metrics = inspect.unwrap(EmbodiedARTFSDPPolicy.run_training)(actor)
    selected = actor.rollout_batch["forward_inputs"]
    assert selected["art_states"].shape == (4, 2, *latent_shape)
    torch.testing.assert_close(selected["art_old_velocity"], selected["art_states"])
    assert "art_coordinate_mask" not in selected and "art_endpoint" not in selected
    assert metrics["actor/active_group_frac"] == 1
    assert actor.lr_scheduler.last_epoch == 1
    assert not torch.equal(actor.model.offset, before)
    assert actor.optimizer.state[actor.model.offset]["step"].item() == 2


@pytest.mark.parametrize("config_name", ["sd3_art_ocr", "wan22_ti2v_5b_art_video_ocr"])
def test_image_video_art_configs_validate_after_libero_integration(
    monkeypatch, art_config_cluster, config_name
):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    monkeypatch.setenv("DIFFUSION_PATH", str(root / "examples/diffusion"))
    monkeypatch.setenv("REPO_PATH", str(root))
    with initialize_config_dir(
        version_base="1.1", config_dir=str(root / "examples/diffusion/config")
    ):
        cfg = compose(config_name=config_name)
    cfg = validate_cfg(cfg)
    assert cfg.algorithm.loss_type == "embodied_art"
    assert not cfg.algorithm.filter_rewards
    cfg.algorithm.filter_rewards = True
    with pytest.raises(AssertionError, match="filter_rewards=False"):
        validate_cfg(cfg)


def test_openpi_art_anchor_depends_on_state_and_leaves_sampling_unchanged(
    single_thread_eager_torch,
):
    model, config = _tiny_art_pi0()
    model.eval()
    observation = config.fake_obs(batch_size=2)
    reference = torch.randn(4, 4, generator=torch.Generator().manual_seed(5))
    with torch.no_grad():
        torch.manual_seed(0)
        actions, plain = model._predict_train(
            observation, noise=None, rng=None, compute_values=False
        )
        plain_rng = torch.get_rng_state()
        torch.manual_seed(0)
        anchored_actions, anchored = model._predict_train(
            observation,
            noise=None,
            rng=None,
            compute_values=False,
            art_anchor_noise=reference,
        )
        assert torch.equal(plain_rng, torch.get_rng_state())
        endpoints = [
            model._predict_train(
                observation,
                noise=sign * reference.expand(2, -1, -1),
                rng=None,
                compute_values=False,
            )[1]["forward_inputs"]["art_endpoint"]
            for sign in (1, -1)
        ]
    assert torch.equal(actions, anchored_actions)
    assert "art_anchor" not in plain["forward_inputs"]
    anchor = anchored["forward_inputs"]["art_anchor"]
    assert anchor.shape == anchored["forward_inputs"]["art_endpoint"].shape
    torch.testing.assert_close(anchor, 0.5 * (endpoints[0] + endpoints[1]))
    # Identical observations share an anchor while their sampled chunks differ.
    torch.testing.assert_close(anchor[0], anchor[1])
    assert not torch.equal(actions[0], actions[1])
    with pytest.raises(ValueError, match="must broadcast"):
        model._predict_train(
            observation,
            noise=None,
            rng=None,
            compute_values=False,
            art_anchor_noise=torch.zeros(3),
        )
    model.rl_cfg = replace(model.rl_cfg, rl_mode="default", is_nft=True)
    with pytest.raises(ValueError, match="requires openpi.rl_mode='art'"):
        model._predict_train(
            observation,
            noise=None,
            rng=None,
            compute_values=False,
            art_anchor_noise=reference,
        )


@pytest.mark.parametrize("model_type", ["sd3", "wan_t2v", "wan_ti2v"])
@pytest.mark.parametrize("cfg_enabled", [False, True])
def test_diffusion_art_forward_preserves_nft_velocity(model_type, cfg_enabled):
    from rlinf.models.diffusion.sd3.stable_diffusion3 import (
        StableDiffusion3,
        StableDiffusion3Config,
    )
    from rlinf.models.diffusion.wan.base import Wan22Config
    from rlinf.models.diffusion.wan.t2v import Wan22T2VModel
    from rlinf.models.diffusion.wan.ti2v import Wan22TI2VModel

    # The external transformer/pipeline boundary needs no downloaded weights.
    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, hidden_states, encoder_hidden_states, **kwargs):
            bias = encoder_hidden_states.flatten(1).mean(1)
            bias = bias.view(-1, *([1] * (hidden_states.ndim - 1)))
            return (hidden_states * self.scale + bias,)

    pipeline = SimpleNamespace(
        transformer=Transformer(),
        config=SimpleNamespace(expand_timesteps=model_type == "wan_ti2v"),
    )
    if model_type == "sd3":
        model = StableDiffusion3(StableDiffusion3Config(cfg=cfg_enabled), pipeline)
        shape = (2, 1, 2, 2, 2)
    else:
        cls = Wan22TI2VModel if model_type == "wan_ti2v" else Wan22T2VModel
        model = cls(Wan22Config(cfg=cfg_enabled), pipeline)
        shape = (2, 3, 2, 2, 2)
    inputs = {
        "prompt_embeds": torch.ones(2, 2, 2),
        "pooled_prompt_embeds": torch.ones(2, 2),
        "negative_prompt_embeds": torch.zeros(2, 2, 2),
        "negative_pooled_prompt_embeds": torch.zeros(2, 2),
    }
    state_inputs = {"x_t": torch.ones(shape), "timesteps": torch.tensor([1.0, 0.5])}
    if model_type == "wan_ti2v":
        condition = torch.full((2, 2, 3, 2, 2), 0.25)
        mask = torch.ones_like(condition)
        mask[:, :, 0] = 0
        inputs.update(latent_condition=condition, first_frame_mask=mask)
    art = model(
        forward_type=ForwardType.ART, forward_inputs=inputs, art_inputs=state_inputs
    )["v_theta"]
    nft = model(
        forward_type=ForwardType.NFT, forward_inputs=inputs, nft_inputs=state_inputs
    )["v_theta"]
    assert art.shape == nft.shape == shape
    torch.testing.assert_close(art, nft)
    art.sum().backward()
    assert model.transformer.scale.grad.abs() > 0


@pytest.mark.parametrize("train_expert_only", [False, True])
def test_openpi_art_and_nft_share_velocity_but_keep_distinct_horizons(
    single_thread_eager_torch, train_expert_only
):
    model, config = _tiny_art_pi0()
    model.rl_cfg = replace(model.rl_cfg, train_expert_only=train_expert_only)
    if train_expert_only:
        model.freeze_vlm()
    model.eval()
    observation = config.fake_obs(batch_size=2)
    forward_inputs = {
        "obs_state": observation.state,
        "tokenized_prompt": observation.tokenized_prompt,
        "tokenized_prompt_mask": observation.tokenized_prompt_mask,
        **{f"obs_image__{key}": value for key, value in observation.images.items()},
        **{
            f"obs_image_mask__{key}": value
            for key, value in observation.image_masks.items()
        },
    }
    state_inputs = {"x_t": torch.randn(2, 4, 4), "timesteps": torch.full((2,), 0.5)}
    with torch.no_grad():
        art = model(
            forward_type=ForwardType.ART,
            forward_inputs=forward_inputs,
            art_inputs=state_inputs,
        )["v_theta"]
        model.rl_cfg = replace(model.rl_cfg, rl_mode="default", is_nft=True)
        nft = model(
            forward_type=ForwardType.NFT,
            forward_inputs=forward_inputs,
            nft_inputs=state_inputs,
        )["v_theta"]
    assert art.shape == (2, 4, 4)
    assert nft.shape == (2, 2, 4)
    torch.testing.assert_close(art[:, :2], nft)

    model.rl_cfg = replace(model.rl_cfg, rl_mode="art", is_nft=False)
    assert model.train() is model
    assert model.training and model.llm.training
    assert model.img.training is not train_expert_only
    training_velocity = model(
        forward_type=ForwardType.ART,
        forward_inputs=forward_inputs,
        art_inputs=state_inputs,
    )["v_theta"]
    if train_expert_only:
        torch.testing.assert_close(training_velocity, art, rtol=0, atol=0)
    training_velocity.sum().backward()
    assert model.action_out_proj.weight.grad.abs().sum() > 0


def test_art_rollout_shares_reference_and_keeps_eval_unchanged():
    from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType

    worker = object.__new__(ARTMultiStepRolloutWorker)
    worker.cfg = OmegaConf.create({"actor": {"seed": 3}})
    worker.version = 2
    calls = []
    worker.hf_model = SimpleNamespace(
        action_horizon=4,
        model_action_dim=3,
        predict_action_batch=lambda **kwargs: (
            calls.append(kwargs) or (torch.zeros(1, 1), {"forward_inputs": {}})
        ),
    )
    reference = worker._art_anchor_noise()
    assert reference.shape == (4, 3)
    assert torch.equal(reference, worker._art_anchor_noise())
    worker.version = 3
    assert not torch.equal(reference, worker._art_anchor_noise())

    worker.model_cfg = OmegaConf.create({"model_type": "openpi"})
    worker.algorithm_cfg = OmegaConf.create({})
    worker.enable_dagger = False
    worker.expert_model = None
    worker._train_sampling_params = {}
    worker._eval_sampling_params = {}
    worker._timer_metrics = {}
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    policy_inputs = {"art_cluster_features": torch.ones(1, 26)}
    _, result = worker.predict({"policy_inputs": policy_inputs}, mode="train")
    torch.testing.assert_close(result["forward_inputs"], policy_inputs)
    assert torch.equal(calls[0]["art_anchor_noise"], worker._art_anchor_noise())
    _, result = worker.predict({}, mode="eval")
    assert "art_anchor_noise" not in calls[1]
    assert not result["forward_inputs"]


def test_cluster_training_gathers_states_across_ranks(tmp_path):
    import sys

    script = tmp_path / "global_cluster.py"
    script.write_text("""
import sys
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_art_policy_worker import EmbodiedARTFSDPPolicy


def worker(rank, directory):
    dist.init_process_group("gloo", init_method="file://" + str(Path(directory, "store")), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    actor = object.__new__(EmbodiedARTFSDPPolicy)
    actor._rank, actor._world_size = rank, 2
    actor.device = torch.device("cpu")
    actor.cfg = OmegaConf.create({"algorithm": {"group_size": 4, "art_cluster_radius": 1., "art_cluster_min_size": 4}})
    endpoints = torch.tensor([0., 1., 0., 1.]).reshape(1,4,1,1).expand(2,-1,-1,-1).clone()
    returns = torch.tensor([[0., 1., 0., 1.], [0., 1., 0., 1.]])
    mask = torch.ones(2,4,1,dtype=torch.bool)
    mask[1,2:] = False
    guidance, metrics = actor._art_cluster_guidance(endpoints, torch.zeros_like(endpoints), returns, mask, {"art_cluster_features": torch.zeros(2,4,1), "art_cluster_bucket": torch.zeros(2,4,dtype=torch.long)})
    assert metrics["actor/cluster_count"] == 1
    assert guidance["directions"].shape == endpoints.shape
    assert (guidance["directions"][0] == 1).all()
    assert (guidance["directions"][1,:2] == 1).all() and not guidance["directions"][1,2:].any()
    torch.testing.assert_close(guidance["coordinate_mask"], mask[...,None])
    dist.destroy_process_group()


if __name__ == "__main__":
    mp.spawn(worker, args=(sys.argv[1],), nprocs=2)
""")
    subprocess.run(
        [sys.executable, str(script), str(tmp_path)],
        env=dict(
            os.environ,
            OMP_NUM_THREADS="1",
            GLOO_SOCKET_IFNAME="lo0" if sys.platform == "darwin" else "lo",
        ),
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_art_anchor_preserves_same_state_covariance():
    endpoints = torch.tensor([[-9.0, -11.0, 11.0, 9.0]]).reshape(1, 4, 1, 1)
    anchors = torch.tensor([[-10.0, -10.0, 10.0, 10.0]]).reshape_as(endpoints)
    returns = torch.tensor([1.0, 0.0, 1.0, 0.0])
    mask = torch.ones(1, 4, 1, dtype=torch.bool)
    raw = art_trajectory_guidance(endpoints, returns, mask, 2)
    centered = art_trajectory_guidance(endpoints - anchors, returns, mask, 2)
    torch.testing.assert_close(raw["directions"], centered["directions"])


@pytest.mark.parametrize("form_clusters", [False, True])
def test_cluster_training_keeps_exact_reset_groups_and_masks_unused_rows(form_clusters):
    natural = _art_trajectory_worker()
    clustered = _art_trajectory_worker()
    clustered._rank, clustered._world_size = 0, 1
    clustered.cfg.algorithm.update(
        art_grouping="cluster", art_cluster_radius=0.1, art_cluster_min_size=2
    )
    inputs = clustered.rollout_batch["forward_inputs"]
    features = torch.arange(3)[:, None] * 10 + torch.arange(4)[None, :] // 2
    if not form_clusters:
        features = torch.arange(12).reshape(3, 4)
    inputs["art_cluster_features"] = features[..., None].float()
    inputs["art_cluster_bucket"] = torch.zeros(3, 4, dtype=torch.long)
    _, _, expected_windows = natural._prepare_art_trajectory_training_inputs(
        torch.Generator().manual_seed(42)
    )
    metrics, _, actual_windows = clustered._prepare_art_trajectory_training_inputs(
        torch.Generator().manual_seed(42)
    )
    assert all(torch.equal(a, b) for a, b in zip(expected_windows, actual_windows))
    expected = natural.rollout_batch["forward_inputs"]
    actual = clustered.rollout_batch["forward_inputs"]
    for key in (
        "art_guidance_direction",
        "art_guidance_magnitude",
        "art_coordinate_mask",
    ):
        if form_clusters:
            torch.testing.assert_close(actual[key], expected[key])
        else:
            torch.testing.assert_close(actual[key][:4], expected[key][:4])
            assert not actual[key][4:].any()
    assert metrics["actor/cluster_count"] == (2 if form_clusters else 0)


def test_art_state_cluster_constraints():
    from rlinf.algorithms.diffusion.art import art_state_clusters

    features = torch.tensor([[0.0], [0.01], [0.02], [0.03], [0.04], [10.0], [20.0]])
    buckets = torch.tensor([0, 0, 0, 0, 0, 0, 1])
    trajectories = torch.tensor([0, 0, 1, 2, 3, 4, 5])
    eligible = torch.ones(7, dtype=torch.bool)
    options = {"radius": 0.1, "min_size": 4, "max_size": 4}
    labels, seeds = art_state_clusters(
        features, buckets, trajectories, eligible, **options
    )
    repeated = art_state_clusters(features, buckets, trajectories, eligible, **options)
    assert torch.equal(labels, repeated[0]) and torch.equal(seeds, repeated[1])
    assert len(seeds) == 1 and (labels < 0).sum() == 3
    for cluster, seed in enumerate(seeds):
        members = labels == cluster
        assert len(trajectories[members].unique()) == members.sum() == 4
        assert buckets[members].unique().numel() == 1
        assert (features[members] - features[seed]).norm(dim=1).max() <= 0.1
