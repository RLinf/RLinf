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

"""Numerical checks of the SuperMimic objectives, reset sampler, evaluation
guards, training config limits, input norms, forward kinematics and simulator.
None needs Ray. The two-rank check needs Gloo, the kinematics and world-model
checks need mjlab, the fused-Adam check needs a CUDA device, and the MuJoCo
Warp checks need mjlab and a CUDA device.
"""

import importlib.util
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir

from rlinf.algorithms.supermimic.objectives import (
    actor_objective,
    gae_targets,
    termination_bce_scale,
)
from rlinf.algorithms.supermimic.reset_sampler import RETURN_EMA, BinResetSampler
from rlinf.algorithms.supermimic.validate import validate_supermimic_cfg
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    DT,
    EPISODE_CLIP_END,
    EPISODE_CONT,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    JOINT_POS,
    ROOT_LIN_VEL,
    XI_DIM,
)
from rlinf.envs.sim.g1_mimic.eval_metrics import check_job_coverage
from rlinf.envs.sim.g1_mimic.features import default_state
from rlinf.envs.sim.g1_mimic.g1_mimic_eval_env import (
    G1MimicEvalEnv,
    checked_eval_states,
    checked_policy_action,
)
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import differentiable_fk, failure_signals
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.supermimic.normalization import (
    RunningMeanStd,
    merged_batch_stats,
)

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None, reason="mjlab is not installed"
)
needs_gloo = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with Gloo is not available",
)
needs_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused Adam needs a CUDA device"
)
needs_warp = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mjlab") is None,
    reason="MuJoCo Warp needs mjlab and a CUDA device",
)
VELOCITY_MODEL = [
    "actor.model.goal_encoding=paper",
    "actor.model.obs_dim=93",
    "actor.model.critic_features=root4",
    "actor.model.input_norm=none",
    "actor.model.velocity_tracking=true",
    "actor.model.torch_compile=false",
]


def _compose(overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name="supermimic_lafan1_goal730", overrides=overrides)


def test_actor_objective_value_and_gradient():
    rewards = torch.tensor([[2.0, 3.0]], requires_grad=True)
    failure = torch.tensor([[0.25, 0.5]], requires_grad=True)
    bootstrap = torch.tensor([4.0], requires_grad=True)
    result = actor_objective(rewards, failure, bootstrap, gamma=0.5)
    # 2 + .5*.75*3 + .5**2*.75*.5*4.
    torch.testing.assert_close(result, torch.tensor(3.5))
    gradients = torch.autograd.grad(result, (rewards, failure, bootstrap))
    for actual, expected in zip(
        gradients,
        (
            torch.tensor([[1.0, 0.375]]),
            torch.tensor([[-2.0, -0.75]]),
            torch.tensor([0.09375]),
        ),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)


def test_gae_matches_discounted_returns_at_lambda_one():
    rewards = torch.tensor([[1.0, 2.0, 3.0]])
    failure = torch.tensor([[0.0, 1.0, 0.0]])
    values = torch.tensor([[99.0, -4.0, 17.0, 4.0]])
    actual = gae_targets(rewards, failure, values, gamma=0.5, lam=1.0)
    torch.testing.assert_close(actual, torch.tensor([[2.0, 2.0, 5.0]]))


@pytest.mark.parametrize("gamma", [0.01, 0.8, 0.99])
def test_reset_returns_match_backward_recurrence(gamma):
    horizon = 500
    lengths = [horizon, 7, 1]
    sampler = BinResetSampler(torch.tensor([horizon]), 3, 10.0, gamma, horizon, CPU)
    rewards = torch.rand(3, horizon, generator=torch.Generator().manual_seed(17))
    for step in range(horizon):
        active = torch.tensor([step < length for length in lengths])
        sampler.log_step(
            rewards[:, step],
            torch.zeros(3, dtype=torch.long),
            torch.full((3,), step),
            active,
        )
    kinds = torch.tensor([EPISODE_TIMEOUT, EPISODE_FAILURE, EPISODE_CLIP_END])
    # Failure and clip-end do not bootstrap, including non-finite predictions.
    bootstrap = torch.tensor([3.0, float("nan"), float("inf")])
    sums = torch.zeros(sampler.num_bins, dtype=torch.float64)
    counts = torch.zeros_like(sums)
    for row, length in enumerate(lengths):
        value = 3.0 if row == 0 else 0.0
        for step in range(length - 1, -1, -1):
            value = float(rewards[row, step]) + gamma * value
            bin_id = step // sampler.bin_frames
            sums[bin_id] += value
            counts[bin_id] += 1
    sampler.finish(torch.arange(3), kinds, bootstrap)
    expected = torch.where(
        counts > 0,
        (1 - RETURN_EMA) * 10 + RETURN_EMA * sums / counts.clamp_min(1),
        10.0,
    )
    torch.testing.assert_close(
        sampler.bin_return.double(), expected, atol=2e-5, rtol=2e-6
    )
    assert torch.isfinite(sampler.mixture(update_memory=False)).all()
    assert sampler.sample(4)[0].numel() == 4


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_invalid_timeout_bootstrap_is_rejected(value):
    sampler = BinResetSampler(torch.tensor([50]), 1, 10.0, 0.99, 2, CPU)
    sampler.log_step(torch.tensor([1.0]), torch.tensor([0]), torch.tensor([0]))
    before = sampler.bin_return.clone()
    with pytest.raises(RuntimeError, match="non-finite timeout bootstrap"):
        sampler.finish(
            torch.tensor([0]), torch.tensor([EPISODE_TIMEOUT]), torch.tensor([value])
        )
    torch.testing.assert_close(sampler.bin_return, before)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_evaluation_rejects_active_nonfinite_rows_and_masks_finished_rows(value):
    active = torch.tensor([True, False])
    action = torch.ones(2, ACTION_DIM)
    action[1] = value
    torch.testing.assert_close(
        checked_policy_action(action, active),
        torch.cat((torch.ones(1, ACTION_DIM), torch.zeros(1, ACTION_DIM))),
    )
    action[0, 0] = value
    with pytest.raises(RuntimeError, match=r"active evaluation rows \[0\]"):
        checked_policy_action(action, active)
    for field in ("next_state", "ref_state"):
        state = default_state(2, "cpu", torch.float32)
        reference = state.clone()
        state[1] = reference[1] = value
        assert all(
            torch.isfinite(part).all()
            for part in checked_eval_states(state, reference, active)
        )
        (state if field == "next_state" else reference)[0, 0] = value
        with pytest.raises(RuntimeError, match=r"active rows \[0\]"):
            checked_eval_states(state, reference, active)


@pytest.mark.parametrize(
    "jobs,valid,complete",
    [
        ([2, 0, 1, -1], [1, 1, 1, 0], True),
        ([0, 2, -1, -1], [1, 1, 0, 0], False),
        ([0, 1, 1, 2], [1, 1, 1, 1], False),
        ([-1, -1, -1, -1], [0, 0, 0, 0], False),
    ],
)
def test_evaluation_summary_requires_every_job_once(jobs, valid, complete):
    rows = {"job": torch.tensor(jobs), "valid": torch.tensor(valid)}
    if complete:
        check_job_coverage(rows, 3)
        return
    with pytest.raises(RuntimeError, match="each of the 3 evaluation jobs"):
        check_job_coverage(rows, 3)


@pytest.mark.parametrize(
    "overrides,error",
    [
        (["env.train.max_episode_seconds=0.1"], "allows 5 control steps"),
        (["algorithm.replay_length=9"], r"replay_length must be in \[10, inf\]"),
        (["actor.fsdp_config.strategy=fsdp"], "strategy=fsdp2"),
        (["actor.model.precision=bf16"], "actor.model.precision must be 32"),
        (
            ["actor.fsdp_config.mixed_precision.param_dtype=bf16"],
            "param_dtype must be null or fp32",
        ),
        (
            ["actor.fsdp_config.grad_scaler.enabled=True"],
            "grad_scaler.enabled must be False",
        ),
        (["env.train.max_episode_seconds=0.16", "algorithm.replay_length=10"], None),
    ],
)
def test_training_config_rejects_unsupported_values(overrides, error):
    cfg = _compose(["env.train.motion_path=/data/clips", *overrides])
    if error is None:
        validate_supermimic_cfg(cfg)
        return
    with pytest.raises(ValueError, match=error):
        validate_supermimic_cfg(cfg)


def test_norm_written_in_place_folds_like_its_source():
    source, copy = RunningMeanStd(3), RunningMeanStd(3)
    source.fold(torch.full((100, 3), 10.0))
    with torch.no_grad():
        for name, buffer in source.named_buffers():
            copy.get_buffer(name).copy_(buffer)
    copy.refresh_host_count()
    for norm in (source, copy):
        norm.fold(torch.full((1, 3), 20.0))
    torch.testing.assert_close(copy.mean, torch.full((3,), 1020.0 / 101))
    for name, buffer in source.named_buffers():
        torch.testing.assert_close(copy.get_buffer(name), buffer)


_RANK_ROWS = (
    torch.tensor([[0.0, 1.0], [2.0, -1.0]]),
    torch.tensor([[1.25, 4.0], [2.25, 2.0], [1.75, 3.0]]),
)


def _fold_merged_stats(rank: int, port: int, out: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        norm = RunningMeanStd(2)
        norm.update_from_packed(*merged_batch_stats(_RANK_ROWS[rank], dist.group.WORLD))
        torch.save(dict(norm.named_buffers()), f"{out}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@needs_gloo
def test_policy_input_stats_merge_over_ranks_like_one_batch(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(
        _fold_merged_stats, args=(port, str(tmp_path)), nprocs=2, join=True
    )
    one_batch = RunningMeanStd(2)
    one_batch.fold(torch.cat(_RANK_ROWS))
    for rank in range(2):
        buffers = torch.load(tmp_path / f"rank{rank}.pt")
        for name, buffer in one_batch.named_buffers():
            torch.testing.assert_close(buffers[name], buffer)


@needs_cuda
def test_fused_adam_step_matches_host_step_and_skips_nonfinite():
    from rlinf.workers.actor.fsdp_supermimic_policy_worker import (
        clip_step,
        fused_clip_step,
    )

    device = torch.device("cuda:0")
    gen = torch.Generator().manual_seed(5)
    init = [torch.randn(4, 3, generator=gen), torch.randn(3, generator=gen)]
    host_params = [torch.nn.Parameter(p.to(device)) for p in init]
    fused_params = [torch.nn.Parameter(p.to(device)) for p in init]
    host = torch.optim.Adam(host_params, lr=1e-2)
    fused = torch.optim.Adam(fused_params, lr=1e-2, fused=True)

    def set_grads(grads):
        for params in (host_params, fused_params):
            for param, grad in zip(params, grads, strict=True):
                param.grad = grad.to(device)

    def fused_step():
        grads = [param.grad for param in fused_params]
        norm = torch.nn.utils.get_total_norm(grads)
        return fused_clip_step(fused, grads, norm, 1.0, 0.1)

    # The first step stays under the norm limit; the others are clipped.
    for scale in (0.1, 5.0, 30.0):
        set_grads([scale * torch.randn(p.shape, generator=gen) for p in init])
        norm = torch.nn.utils.get_total_norm([param.grad for param in host_params])
        clip_step(host, host_params, float(norm), 1.0, 0.1)
        assert float(fused_step()) == 0.0
    for host_param, fused_param in zip(host_params, fused_params, strict=True):
        torch.testing.assert_close(fused_param, host_param)
        for key in ("exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                fused.state[fused_param][key], host.state[host_param][key]
            )
        assert float(fused.state[fused_param]["step"]) == 3.0

    before = [
        (
            param.clone(),
            {key: value.clone() for key, value in fused.state[param].items()},
        )
        for param in fused_params
    ]
    for bad in (float("nan"), float("inf")):
        grads = [torch.ones(p.shape) for p in init]
        grads[0][1, 2] = bad
        set_grads(grads)
        assert float(fused_step()) == 1.0
        for param, (value, state) in zip(fused_params, before, strict=True):
            assert param.grad is None
            assert torch.equal(param, value)
            for key, saved in state.items():
                assert torch.equal(fused.state[param][key], saved), key


@needs_mjlab
def test_world_model_bce_over_two_ranks_matches_one_batch():
    cfg = validate_supermimic_cfg(
        _compose(
            [
                "env.train.motion_path=/data/clips",
                "algorithm.termination_head=learned",
                *VELOCITY_MODEL,
            ]
        )
    )
    torch.manual_seed(0)
    model = get_model(cfg.actor.model).to(CPU)
    model.configure_training(cfg.algorithm)
    rows, k = 8, model.k
    gen = torch.Generator().manual_seed(3)
    pose = default_state(rows, "cpu", torch.float32)
    end_kind = torch.full((rows, k), EPISODE_CONT)
    end_kind[0] = EPISODE_FAILURE
    batch = {
        "state0": pose + 0.01 * torch.randn(pose.shape, generator=gen),
        "actions": 0.1 * torch.randn(rows, k, ACTION_DIM, generator=gen),
        "xi": 0.1 * torch.randn(rows, XI_DIM, generator=gen),
        "targets": pose.unsqueeze(1)
        + 0.02 * torch.randn(rows, k, pose.shape[-1], generator=gen),
        "refs": pose.unsqueeze(1).expand(-1, k, -1),
        "end_kind": end_kind,
        # Rank 0: one failing motion row; rank 1: three motion rows that hold.
        "task": torch.tensor([0, 1, 1, 1, 0, 0, 0, 1]),
    }
    motion_total = int((batch["task"] == 0).sum())

    def gradients(part: slice, share: float) -> dict[str, torch.Tensor]:
        part_rows = int(batch["task"][part].numel())
        motion = int((batch["task"][part] == 0).sum())
        model.zero_grad(set_to_none=True)
        loss = model(
            forward_type=ForwardType.SUPERMIMIC_WORLD_MODEL,
            **{name: value[part] for name, value in batch.items()},
            bce_scale=termination_bce_scale(part_rows, motion, rows, motion_total),
        )["loss"]
        (loss * share).backward()
        return {name: p.grad.clone() for name, p in model.named_parameters()}

    merged = gradients(slice(None), 1.0)
    # FSDP averages the rank gradients and the actor weights each rank's loss
    # by its share of the rows, so the update sums the row-weighted gradients.
    first, second = gradients(slice(0, 4), 0.5), gradients(slice(4, 8), 0.5)
    head = [g for name, g in merged.items() if name.startswith("termination.")]
    assert head and max(g.abs().max().item() for g in head) > 0.1, head
    for name, gradient in merged.items():
        torch.testing.assert_close(first[name] + second[name], gradient)


@needs_mjlab
@pytest.mark.parametrize("joint", ["waist_roll_joint", "waist_pitch_joint"])
def test_torso_tilt_and_gradient_match_joint_rotation(joint):
    state = default_state(2, "cpu", torch.float32)
    reference = state.clone()
    index = JOINT_POS.start + G1_JOINT_NAMES.index(joint)
    angles, ref_angles = torch.tensor([1.4, 0.4]), torch.tensor([0.0, 0.2])
    state[:, index], reference[:, index] = angles, ref_angles
    state.requires_grad_()
    _, torso_tilt, torso_failed = failure_signals(state, reference, "torso")
    _, pelvis_tilt, pelvis_failed = failure_signals(state, reference, "pelvis")
    torch.testing.assert_close(torso_tilt, ref_angles.cos() - angles.cos())
    torch.testing.assert_close(pelvis_tilt, torch.zeros(2))
    assert bool(torso_failed[0]) and not bool(pelvis_failed.any())
    gradient = torch.autograd.grad(torso_tilt.sum(), state)[0]
    torch.testing.assert_close(gradient[:, index], angles.sin())


@needs_mjlab
@pytest.mark.parametrize(
    "storage,query",
    [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:0", "cpu")],
)
def test_reference_fk_matches_online_fk(storage, query):
    if "cuda" in storage + query and not torch.cuda.is_available():
        pytest.skip("a CUDA/HIP device is required for cross-device queries")
    clips = [default_state(n, storage, torch.float32) for n in (3, 4)]
    for index, clip in enumerate(clips):
        clip[:, 0] = index + 0.1 * torch.arange(len(clip), device=storage)
        clip[:, JOINT_POS.start] = 0.05 * torch.arange(len(clip), device=storage)
    motion = MotionLibrary(clips)
    ids, times = (
        torch.tensor([1, 0, 1], device=query),
        torch.tensor([100, -1, 2], device=query),
    )
    result = motion.kinematics_at(ids, times)
    expected = differentiable_fk(motion.state_at(ids, times))
    assert set(motion._kin_cache) == {motion.frames.device}
    assert all(
        part.device == motion.frames.device
        for part in motion._kin_cache[motion.frames.device]
    )
    for actual, wanted in zip(result, expected, strict=True):
        assert actual.device == ids.device
        torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-5)


@needs_warp
@pytest.mark.parametrize("mode", ["add", "set"])
def test_push_writes_only_the_horizontal_root_velocity(mode):
    from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env
    from rlinf.envs.sim.g1_mimic.push_recovery import default_standing_state

    device = torch.device("cuda:0")
    pose = default_standing_state()
    pose[ROOT_LIN_VEL] = torch.tensor([0.3, -0.1, 0.4])
    motion = MotionLibrary([pose.repeat(30, 1)], q0=pose[JOINT_POS])
    env = MjlabG1Env(2, motion, device, eval_mode=True)
    ids = torch.arange(2, device=device)
    env.reset(ids, torch.zeros_like(ids), torch.zeros_like(ids))
    before = env.state.clone()
    xy = slice(ROOT_LIN_VEL.start, ROOT_LIN_VEL.start + 2)
    assert before[1, xy].abs().min() > 0.05, before[1, xy]
    push = torch.tensor([[1.25, -0.5]], device=device)
    after = env.apply_horizontal_velocity_push(ids[1:], push, mode=mode).clone()
    expected = before.clone()
    expected[1, xy] = push[0] + (before[1, xy] if mode == "add" else 0.0)
    torch.testing.assert_close(after, expected, atol=1e-5, rtol=1e-5)
    next_state = env.step(torch.zeros(2, ACTION_DIM, device=device))["next_state"]
    moved = next_state[1, :2] - after[1, :2]
    torch.testing.assert_close(moved, expected[1, xy] * DT, atol=5e-3, rtol=0)


@needs_warp
def test_velocity_step_test_repeats_for_its_seed(tmp_path):
    from rlinf.envs.sim.g1_mimic.push_recovery import default_standing_state

    pose = default_standing_state()
    clips = tmp_path / "clips"
    clips.mkdir()
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": pose[JOINT_POS],
            "state": pose.repeat(64, 1),
        },
        clips / "clip_000.pt",
    )
    cfg = validate_supermimic_cfg(
        _compose(
            [
                f"env.train.motion_path={clips}",
                f"env.eval.motion_path={clips}",
                "env.eval.num_clips=1",
                "env.eval.total_num_envs=2",
                "env.eval.eval_tasks=[velocity_g2]",
                "env.eval.velocity_g2.trials=1",
                "env.eval.velocity_g2.seconds=0.2",
                "env.eval.velocity_g2.steady_seconds=0.1",
                "env.eval.max_steps_per_rollout_epoch=10",
                # Two envs cover the 672 step commands in 336 chunks.
                "env.eval.rollout_epoch=336",
                "runner.val_check_interval=1",
                *VELOCITY_MODEL,
            ]
        )
    ).env.eval

    def evaluate(env):
        obs, _ = env.reset()
        terminated = torch.zeros(1, dtype=torch.bool)
        while not terminated.all():
            _, _, terminated, _, infos = env.chunk_step(torch.zeros(2, 1, ACTION_DIM))
        for _ in range(int(cfg.rollout_epoch)):
            env.update_reset_state_ids()
        return obs["states"], infos[0]["episode"]

    torch.manual_seed(1)
    env = G1MimicEvalEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1)
    first = evaluate(env)
    torch.manual_seed(2)
    torch.rand(100, device="cuda")
    process_rng = torch.get_rng_state()
    repeated = evaluate(env)
    assert torch.equal(torch.get_rng_state(), process_rng)
    rebuilt = evaluate(
        G1MimicEvalEnv(cfg, num_envs=2, seed_offset=0, total_num_processes=1)
    )
    for states, rows in (repeated, rebuilt):
        torch.testing.assert_close(states, first[0])
        for name, value in first[1].items():
            torch.testing.assert_close(rows[name], value)
