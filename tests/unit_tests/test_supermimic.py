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
guards, training config limits, input norms and forward kinematics. They need
neither Ray nor a GPU; the two kinematics checks need mjlab and add
cross-device cases when a GPU is present.
"""

import importlib.util
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from rlinf.algorithms.supermimic.objectives import actor_objective, gae_targets
from rlinf.algorithms.supermimic.reset_sampler import RETURN_EMA, BinResetSampler
from rlinf.algorithms.supermimic.validate import validate_supermimic_cfg
from rlinf.envs.sim.g1_mimic.constants import (
    ACTION_DIM,
    EPISODE_CLIP_END,
    EPISODE_FAILURE,
    EPISODE_TIMEOUT,
    G1_JOINT_NAMES,
    JOINT_POS,
)
from rlinf.envs.sim.g1_mimic.eval_metrics import check_job_coverage
from rlinf.envs.sim.g1_mimic.features import default_state
from rlinf.envs.sim.g1_mimic.g1_mimic_eval_env import (
    checked_eval_states,
    checked_policy_action,
)
from rlinf.envs.sim.g1_mimic.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.reward import differentiable_fk, failure_signals
from rlinf.models.embodiment.supermimic.normalization import RunningMeanStd

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None, reason="mjlab is not installed"
)


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
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        cfg = compose(
            config_name="supermimic_lafan1_goal730",
            overrides=["env.train.motion_path=/data/clips", *overrides],
        )
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
