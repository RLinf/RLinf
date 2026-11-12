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

"""Numerical checks of the SuperMimic objectives, reset sampler, update
cadence, evaluation guards, training config limits, input norms, state
representations, networks and simulator. None needs Ray. The two-rank check
needs Gloo, the kinematics, representation and network checks need mjlab, and
the MuJoCo Warp check needs mjlab and a CUDA device.
"""

import importlib.util
import math
import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir

from rlinf.algorithms.supermimic.objectives import (
    NOMINAL_ERRORS,
    imagined_return,
    lambda_returns,
    world_model_loss,
)
from rlinf.algorithms.supermimic.reset_sampler import (
    RECORD_STEPS,
    RETURN_GAMMA,
    ReturnResetSampler,
)
from rlinf.algorithms.supermimic.validate import validate_supermimic_cfg
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    ACTOR_GOAL_DIM,
    DT,
    JOINT_POS,
    JOINT_VEL,
    MAX_TRACKING_REWARD,
    ROOT_LIN_VEL,
    ROOT_POS,
    ROOT_QUAT,
    SIM_CONTEXT_DIM,
    STATE_DIM,
    WORLD_MODEL_GOAL_DIM,
)
from rlinf.envs.sim.g1_mimic.eval_metrics import check_job_coverage
from rlinf.envs.sim.g1_mimic.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.g1_mimic_eval_env import (
    check_eval_states,
    checked_policy_action,
)
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import failure_signals
from rlinf.hybrid_engines.fsdp.utils import get_grad_norm
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.supermimic.normalization import (
    EmpiricalNorm,
    merged_moments,
    pack_moments,
)
from rlinf.runners.supermimic_runner import boundary_update_spans, due_updates
from rlinf.workers.actor.fsdp_supermimic_policy_worker import clip_step

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None, reason="mjlab is not installed"
)
needs_gloo = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with Gloo is not available",
)
needs_warp = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("mjlab") is None,
    reason="MuJoCo Warp needs mjlab and a CUDA device",
)


def _compose(config_name: str, overrides: list[str]):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name=config_name, overrides=overrides)


def _standing_state(rows: int) -> torch.Tensor:
    state = torch.zeros(rows, STATE_DIM)
    state[:, 2] = 0.78
    state[:, 3] = 1.0
    return state


def _pitched(angle: torch.Tensor) -> torch.Tensor:
    zero = torch.zeros_like(angle)
    return torch.stack((torch.cos(angle / 2), zero, torch.sin(angle / 2), zero), -1)


def _rotmat_to_quat(rot: torch.Tensor) -> torch.Tensor:
    """wxyz quaternion of rotations well away from a half turn."""
    w = 0.5 * torch.sqrt(1.0 + rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2])
    return torch.stack(
        (
            w,
            (rot[..., 2, 1] - rot[..., 1, 2]) / (4 * w),
            (rot[..., 0, 2] - rot[..., 2, 0]) / (4 * w),
            (rot[..., 1, 0] - rot[..., 0, 1]) / (4 * w),
        ),
        dim=-1,
    )


def test_imagined_return_value_and_gradient():
    rewards = torch.tensor([[2.0, 3.0]], requires_grad=True)
    term_prob = torch.tensor([[0.25, 0.5]], requires_grad=True)
    last_value = torch.tensor([4.0], requires_grad=True)
    result = imagined_return(rewards, term_prob, last_value, gamma=0.5)
    # 2 + .5*.75*3 + .5**2*.75*.5*4.
    torch.testing.assert_close(result, torch.tensor([3.5]))
    gradients = torch.autograd.grad(result.sum(), (rewards, term_prob, last_value))
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


def test_lambda_returns_span_one_step_to_full_returns():
    values = torch.tensor([[99.0], [-4.0], [17.0]])
    last_value = torch.tensor([4.0])
    rewards = torch.tensor([[1.0], [2.0], [3.0]])
    term_prob = torch.tensor([[0.0], [1.0], [0.0]])
    full = lambda_returns(values, last_value, rewards, term_prob, gamma=0.5, lam=1.0)
    torch.testing.assert_close(full, torch.tensor([[2.0], [2.0], [5.0]]))
    one_step = lambda_returns(values, last_value, rewards, term_prob, 0.5, lam=0.0)
    torch.testing.assert_close(one_step, torch.tensor([[-1.0], [2.0], [5.0]]))


def test_world_model_loss_weights_each_group_by_its_width():
    pred = {"root_pos": torch.zeros(4, 3), "joint_vel": torch.zeros(4, 29)}
    true = {"root_pos": torch.ones(4, 3), "joint_vel": 2 * torch.ones(4, 29)}
    loss, mse = world_model_loss(pred, true)
    torch.testing.assert_close(mse, torch.tensor([1.0, 4.0]))
    expected = (
        3 / NOMINAL_ERRORS["root_pos"] ** 2 + 29 * 4 / NOMINAL_ERRORS["joint_vel"] ** 2
    ) / 32
    torch.testing.assert_close(loss, torch.tensor(expected))


def test_reset_frames_skip_the_last_frame_of_every_clip():
    torch.manual_seed(0)
    sampler = ReturnResetSampler(torch.tensor([5, 12, 30]), 1, CPU)
    frames = sampler.sample(4000)
    assert int(frames.min()) >= 0 and int(frames.max()) <= 28
    assert not torch.isin(frames, torch.tensor([4, 11, 29])).any()


def test_reset_values_fold_the_mean_return_of_each_episode_visit():
    frame_count = 200
    sampler = ReturnResetSampler(torch.tensor([frame_count]), 2, CPU)
    rewards = torch.rand(2, RECORD_STEPS, generator=torch.Generator().manual_seed(7))
    frames = torch.stack((torch.arange(RECORD_STEPS), 50 + torch.arange(RECORD_STEPS)))
    failed = torch.zeros(2, RECORD_STEPS, dtype=torch.bool)
    timed_out = torch.zeros_like(failed)
    clip_end = torch.zeros_like(failed)
    failed[0, 9] = True
    timed_out[1, 4] = True
    for t in range(RECORD_STEPS):
        ends = (failed[:, t], clip_end[:, t], timed_out[:, t])
        sampler.record(frames[:, t], rewards[:, t], *ends)

    start = sampler.value_max
    samples: dict[int, list[float]] = {}
    for env in range(2):
        # A fall ends the return; a time limit and the buffer end bootstrap
        # from the next frame's bin, which still holds its initial value.
        returns, tail = [0.0] * RECORD_STEPS, start
        for t in reversed(range(RECORD_STEPS)):
            if failed[env, t]:
                tail = 0.0
            elif timed_out[env, t]:
                tail = start
            returns[t] = float(rewards[env, t]) + RETURN_GAMMA * tail
            tail = returns[t]
        visits, episode = {}, 0
        for t in range(RECORD_STEPS):
            bin_id = int(frames[env, t]) * sampler.bin_count // frame_count
            visits.setdefault((episode, bin_id), []).append(returns[t])
            episode += int(failed[env, t] or timed_out[env, t])
        for (_, bin_id), values in visits.items():
            samples.setdefault(bin_id, []).append(sum(values) / len(values))
    expected = torch.full((sampler.bin_count,), start)
    for bin_id, values in samples.items():
        expected[bin_id] = sum(values) / len(values)
    torch.testing.assert_close(sampler.value, expected)
    assert float(sampler.count.sum()) == sum(map(len, samples.values()))
    torch.testing.assert_close(sampler.probs.sum(), torch.tensor(1.0))
    assert int(sampler.probs.argmax()) == int(sampler.value.argmin())

    other = ReturnResetSampler(torch.tensor([frame_count + 10]), 2, CPU)
    with pytest.raises(ValueError, match="reset bins"):
        other.load_state_dict(sampler.state_dict())


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
    state = _standing_state(2)
    state[1] = value
    check_eval_states(state, active)
    state[0, 0] = value
    with pytest.raises(RuntimeError, match=r"active rows \[0\]"):
        check_eval_states(state, active)


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
        (["algorithm.replay_length=7"], r"replay_length must be in \[8, inf\]"),
        (["algorithm.gamma=1.0"], r"gamma must be in \(0.0, 1.0\)"),
        (["actor.model.representation=paper"], "representation must be one of"),
        (["+env.train.representation=maxcoord"], "differs from the policy value"),
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
        (["env.train.max_episode_seconds=0.16", "algorithm.replay_length=8"], None),
    ],
)
def test_training_config_rejects_unsupported_values(overrides, error):
    cfg = _compose(
        "supermimic_lafan1_gencoord", ["env.train.motion_path=/data/clips", *overrides]
    )
    if error is None:
        validate_supermimic_cfg(cfg)
        return
    with pytest.raises(ValueError, match=error):
        validate_supermimic_cfg(cfg)


def test_maxcoord_config_shares_its_representation_with_the_envs():
    cfg = validate_supermimic_cfg(
        _compose(
            "supermimic_lafan1_maxcoord",
            [
                "env.train.motion_path=/data/clips",
                "env.eval.motion_path=/data/clips",
                "runner.val_check_interval=1",
            ],
        )
    )
    assert cfg.env.train.representation == cfg.env.eval.representation == "maxcoord"


def test_norm_folds_batches_like_their_union():
    gen = torch.Generator().manual_seed(1)
    first, second = torch.randn(5, 3, generator=gen), torch.randn(9, 3, generator=gen)
    norm = EmpiricalNorm(3)
    torch.testing.assert_close(norm(first), first / 1.01)
    norm.fold(pack_moments(first))
    norm.fold(pack_moments(second))
    union = torch.cat((first, second))
    var, mean = torch.var_mean(union, dim=0, correction=0)
    torch.testing.assert_close(norm.mean, mean)
    torch.testing.assert_close(norm.var, var)
    assert float(norm.count) == 14.0


_RANK_ROWS = (
    torch.tensor([[0.0, 1.0], [2.0, -1.0]]),
    torch.tensor([[1.25, 4.0], [2.25, 2.0], [1.75, 3.0]]),
)


def _fold_merged_moments(rank: int, port: int, out: str) -> None:
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        norm = EmpiricalNorm(2)
        norm.fold(merged_moments(_RANK_ROWS[rank], dist.group.WORLD))
        torch.save(dict(norm.named_buffers()), f"{out}/rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@needs_gloo
def test_merged_moments_over_ranks_fold_like_one_batch(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(
        _fold_merged_moments, args=(port, str(tmp_path)), nprocs=2, join=True
    )
    one_batch = EmpiricalNorm(2)
    one_batch.fold(pack_moments(torch.cat(_RANK_ROWS)))
    for rank in range(2):
        buffers = torch.load(tmp_path / f"rank{rank}.pt")
        for name, buffer in one_batch.named_buffers():
            torch.testing.assert_close(buffers[name], buffer)


def test_due_updates_follow_the_replay_ratio():
    counts, anchor = [], None
    for step in range(11):
        count, anchor = due_updates(step, anchor, 0.4)
        counts.append(count)
    assert counts == [1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
    for ratio in (0.4, 2.5):
        total, anchor = 0, None
        for step in range(1000):
            count, anchor = due_updates(step, anchor, ratio)
            total += count
        assert total == 1 + math.floor(999 * ratio)


def test_batched_updates_stop_on_save_and_eval_boundaries():
    # global_step 1998, three owed updates, save every 2000: two, then one.
    assert boundary_update_spans(1998, 3, 16000, 2000, -1) == [2, 1]
    assert boundary_update_spans(1998, 1, 16000, 2000, -1) == [1]
    assert boundary_update_spans(0, 3, 16000, 2000, -1) == [3]
    assert boundary_update_spans(2000, 3, 16000, 2000, -1) == [3]
    assert boundary_update_spans(1998, 5, 2001, 2000, -1) == [2, 1]
    assert boundary_update_spans(15999, 3, 16000, 2000, -1) == [1]
    assert boundary_update_spans(1998, 3, 16000, -1, -1) == [3]
    assert boundary_update_spans(1998, 3, 16000, 0, 0) == [3]
    assert boundary_update_spans(1990, 30, 16000, 2000, 1000) == [10, 20]
    assert boundary_update_spans(1990, 1020, 16000, 2000, 1000) == [10, 1000, 10]


def test_clip_step_clips_the_norm_then_each_element():
    first = torch.nn.Parameter(torch.zeros(2))
    second = torch.nn.Parameter(torch.zeros(1))
    optimizers = [torch.optim.SGD([first], lr=1.0), torch.optim.SGD([second], lr=1.0)]
    first.grad, second.grad = torch.tensor([3.0, 0.0]), torch.tensor([4.0])
    # Norm 5 scales to [0.6, 0, 0.8]; the element clip then caps 0.8 at 0.7.
    clip_step(optimizers, [first, second], 5.0, max_norm=1.0, value_clip=0.7)
    torch.testing.assert_close(first.detach(), torch.tensor([-0.6, 0.0]))
    torch.testing.assert_close(second.detach(), torch.tensor([-0.7]))
    assert first.grad is None and second.grad is None
    first.grad, second.grad = torch.tensor([0.1, 0.2]), torch.tensor([0.3])
    clip_step(optimizers, [first, second], 0.37, max_norm=1.0, value_clip=None)
    torch.testing.assert_close(first.detach(), torch.tensor([-0.7, -0.2]))
    torch.testing.assert_close(second.detach(), torch.tensor([-1.0]))


@pytest.mark.parametrize("grads", [([3.0, 0.0], [4.0]), ([0.1, 0.2], [0.3])])
def test_clip_step_with_a_device_norm_matches_the_host_norm(grads):
    params = {}
    for as_tensor in (False, True):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(1))
        first.grad, second.grad = (torch.tensor(g) for g in grads)
        norm = get_grad_norm([first, second], None, as_tensor=as_tensor)
        assert torch.is_tensor(norm) == as_tensor
        clip_step(
            [torch.optim.Adam([first, second], lr=0.1, fused=True)],
            [first, second],
            norm,
            max_norm=1.0,
            value_clip=0.7,
        )
        params[as_tensor] = torch.cat([first.detach(), second.detach()])
    torch.testing.assert_close(params[True], params[False])


def test_pelvis_falls_on_height_or_tilt():
    reference = _standing_state(3)
    state = reference.clone()
    angle = torch.tensor([0.0, 1.4, 0.4])
    state[:, ROOT_QUAT] = _pitched(angle)
    state[0, 2] -= 0.3
    height, tilt, failed = failure_signals(state, reference)
    torch.testing.assert_close(height, torch.tensor([-0.3, 0.0, 0.0]))
    torch.testing.assert_close(tilt, 1.0 - torch.cos(angle))
    assert failed.tolist() == [True, True, False]


@needs_mjlab
def test_perfect_tracking_earns_the_reward_ceiling():
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state, soft_joint_limits
    from rlinf.envs.sim.g1_mimic.reward import differentiable_fk, tracking_reward

    state = asset_default_state().repeat(2, 1)
    bodies = differentiable_fk(state)
    action = torch.zeros(2, ACTION_DIM)
    action[1] = 1.0
    reward = tracking_reward(
        state[:, ROOT_POS],
        quat_to_rotmat(state[:, ROOT_QUAT]),
        bodies,
        bodies,
        state[:, JOINT_POS],
        action,
        torch.zeros(2, ACTION_DIM),
        soft_joint_limits(),
    )
    rate_penalty = DT * 0.1 * ACTION_DIM
    torch.testing.assert_close(
        reward, torch.tensor([MAX_TRACKING_REWARD, MAX_TRACKING_REWARD - rate_penalty])
    )


def _random_states(rows: int, seed: int) -> torch.Tensor:
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_joint_limits

    gen = torch.Generator().manual_seed(seed)
    low, high = asset_joint_limits()
    state = _standing_state(rows)
    state[:, :3] += 0.2 * torch.randn(rows, 3, generator=gen)
    state[:, ROOT_QUAT] += 0.2 * torch.randn(rows, 4, generator=gen)
    state[:, ROOT_QUAT] /= state[:, ROOT_QUAT].norm(dim=-1, keepdim=True)
    middle = 0.3 + 0.4 * torch.rand(rows, ACTION_DIM, generator=gen)
    state[:, JOINT_POS] = low + (high - low) * middle
    state[:, ROOT_LIN_VEL.start :] = torch.randn(rows, 35, generator=gen)
    return state


def _stored(representation, states: torch.Tensor) -> torch.Tensor:
    from rlinf.envs.sim.g1_mimic.reward import differentiable_fk

    flat = states.reshape(-1, STATE_DIM)
    pos, rot, lin, ang = differentiable_fk(flat)
    full = representation.stored_state(flat, (pos, _rotmat_to_quat(rot), lin, ang))
    return full.reshape(*states.shape[:-1], -1)


@needs_mjlab
@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
def test_representation_reproduces_the_tracked_bodies(name):
    from rlinf.algorithms.supermimic.representations import make_representation
    from rlinf.envs.sim.g1_mimic.reward import differentiable_fk

    representation = make_representation(name)
    state = _random_states(32, seed=0)
    full = _stored(representation, state)
    enc = representation.encode(full)
    assert full.shape[-1] == representation.state_dim
    assert enc.shape[-1] == representation.encoding_dim
    root = representation.root(full)
    for actual, expected in zip(
        representation.bodies(root, enc), differentiable_fk(state), strict=True
    ):
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    q, qd = representation.joints(enc)
    torch.testing.assert_close(q, state[:, JOINT_POS], atol=1e-3, rtol=0)
    torch.testing.assert_close(qd, state[:, JOINT_VEL], atol=1e-3, rtol=0)
    blocks = representation.loss_blocks(root, enc)
    assert {k: v.shape[-1] for k, v in blocks.items()} == representation.block_widths


@needs_mjlab
@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
def test_each_training_phase_reaches_only_its_network(name):
    from omegaconf import OmegaConf

    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
    from rlinf.models.embodiment.supermimic.networks import hidden_matrix_names
    from rlinf.models.embodiment.supermimic.supermimic_policy import (
        SuperMimicPolicy,
    )

    torch.manual_seed(0)
    pose = asset_default_state()
    clips = []
    for length in (40, 30):
        clip = pose.repeat(length, 1)
        clip[:, 0] = torch.linspace(0.0, 1.0, length)
        clip[:, JOINT_POS] += 0.05 * torch.sin(torch.linspace(0, 3, length))[:, None]
        clips.append(clip)
    motion = MotionLibrary(clips, q0=pose[JOINT_POS])
    policy = SuperMimicPolicy(name, torch_compile=False)
    policy.configure_training(OmegaConf.create({"H": 4, "gamma": 0.99}))
    assert all(".weight" in n for n in hidden_matrix_names(policy))
    rows, k, horizon = 6, 8, 4
    frames = torch.randint(0, motion.num_frames - k, (rows,))
    states = torch.stack([motion.rows(frames + i) for i in range(k)], dim=1)
    full = _stored(policy.representation, states)
    actions = 0.1 * torch.randn(rows, k, ACTION_DIM)
    context = 0.3 * torch.randn(rows, SIM_CONTEXT_DIM)
    networks = {
        "dynamics": policy.dynamics,
        "actor": policy.actor,
        "critic": policy.critic,
    }

    def reached(forward_type, **inputs):
        policy.zero_grad(set_to_none=True)
        policy.set_trainable(forward_type)
        out = policy(forward_type=forward_type, **inputs)
        assert torch.isfinite(out["loss"])
        out["loss"].backward()
        names = {
            name
            for name, net in networks.items()
            if any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in net.parameters()
            )
        }
        return out, names

    _, names = reached(
        ForwardType.SUPERMIMIC_WORLD_MODEL,
        states=full,
        actions=actions,
        context=context,
    )
    assert names == {"dynamics"}
    last = motion.clip_end(frames) - 1
    times = torch.minimum(frames[:, None] + torch.arange(horizon + 1), last[:, None])
    out, names = reached(
        ForwardType.SUPERMIMIC_ACTOR,
        start=full[:, 0],
        prev_action=actions[:, 0],
        context=context,
        windows=motion.goal_window(times),
        refs=motion.body_state(times[:, :horizon]),
    )
    assert names == {"actor"}
    enc_dim = policy.representation.encoding_dim
    width = enc_dim + WORLD_MODEL_GOAL_DIM + SIM_CONTEXT_DIM
    assert out["steps"].shape == (rows, horizon, width)
    _, names = reached(
        ForwardType.SUPERMIMIC_CRITIC,
        features=out["steps"].reshape(-1, width),
        targets=torch.zeros(rows * horizon),
    )
    assert names == {"critic"}
    obs = torch.cat(
        (policy.representation.encode(full[:, 0]), torch.zeros(rows, ACTOR_GOAL_DIM)),
        dim=-1,
    )
    action, _ = policy.predict_action_batch({"states": obs})
    assert action.shape == (rows, 1, ACTION_DIM)


@needs_mjlab
@pytest.mark.parametrize(
    "storage,query",
    [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:0", "cpu")],
)
def test_reference_fk_matches_online_fk(storage, query):
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state
    from rlinf.envs.sim.g1_mimic.reward import differentiable_fk

    if "cuda" in storage + query and not torch.cuda.is_available():
        pytest.skip("a CUDA/HIP device is required for cross-device queries")
    clips = [asset_default_state().repeat(n, 1).to(storage) for n in (3, 4)]
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
    assert all(part.device == motion.frames.device for part in motion._kin_cache)
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
    env = MjlabG1Env(2, motion, device, randomize=False)
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
    env.step(torch.zeros(2, ACTION_DIM, device=device))
    moved = env.state[1, :2] - after[1, :2]
    torch.testing.assert_close(moved, expected[1, xy] * DT, atol=5e-3, rtol=0)
