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
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.algorithms.supermimic.objectives import (
    NOMINAL_ERRORS,
    imagined_return,
    lambda_returns,
    world_model_loss,
)
from rlinf.algorithms.supermimic.replay import TRANSITION_FIELDS, TransitionRing
from rlinf.algorithms.supermimic.reset_sampler import (
    RECORD_STEPS,
    RETURN_GAMMA,
    ReturnResetSampler,
)
from rlinf.algorithms.supermimic.validate import (
    check_supermimic_placement,
    validate_supermimic_cfg,
)
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
from rlinf.envs.sim.g1_mimic.eval.eval_metrics import check_job_coverage
from rlinf.envs.sim.g1_mimic.g1_mimic_env import G1MimicEnv
from rlinf.envs.sim.g1_mimic.g1_mimic_eval_env import (
    check_eval_states,
    checked_policy_action,
)
from rlinf.envs.sim.g1_mimic.kinematics.features import quat_to_rotmat
from rlinf.envs.sim.g1_mimic.tasks.motion import MotionLibrary
from rlinf.envs.sim.g1_mimic.tasks.reward import failure_signals
from rlinf.hybrid_engines.fsdp.utils import get_grad_norm
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.supermimic.normalization import (
    EmpiricalNorm,
    merged_moments,
    pack_moments,
    split_moments,
)
from rlinf.runners.supermimic_runner import (
    SuperMimicRunner,
    boundary_update_spans,
    due_updates,
)
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_supermimic_policy_worker import (
    clip_step,
    critic_minibatch_rows,
    keep_step_placement,
)

CPU = torch.device("cpu")
CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
EVAL_CONFIG_DIR = Path(__file__).resolve().parents[2] / "evaluations" / "supermimic"
_ACCEL_AVAILABLE = (
    Worker.torch_platform is not None and Worker.torch_platform.is_available()
)
_CUDA_AVAILABLE = _ACCEL_AVAILABLE and Worker.torch_device_type == "cuda"
needs_mjlab = pytest.mark.skipif(
    importlib.util.find_spec("mjlab") is None, reason="mjlab is not installed"
)
needs_gloo = pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="torch.distributed with Gloo is not available",
)
needs_warp = pytest.mark.skipif(
    not _CUDA_AVAILABLE or importlib.util.find_spec("mjlab") is None,
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

    # Single-frame clips must never underflow to -1 or jump into a prior clip.
    single = ReturnResetSampler(torch.tensor([1]), 1, CPU)
    assert torch.equal(single.sample(16), torch.zeros(16, dtype=torch.long))
    mixed = ReturnResetSampler(torch.tensor([1, 5, 6, 10]), 1, CPU)
    mixed_frames = mixed.sample(2000)
    assert int(mixed_frames.min()) >= 0 and int(mixed_frames.max()) <= 8
    assert not torch.isin(mixed_frames, torch.tensor([4, 9])).any()


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


def test_push_summary_groups_trials_by_configured_speed():
    from rlinf.envs.sim.g1_mimic.eval.push_recovery import summarize_trials, trial_plan

    speeds = [0.7, 1.0]
    rows = trial_plan(speeds, trials=2, direction=[1.0, 0.0], seed=0)
    for row in rows:
        row["speed_m_s"] = torch.tensor(row["speed_m_s"]).item()
        row["outcome"], row["recovery_time_s"] = "recovered", 1.0
    assert rows[0]["speed_m_s"] != 0.7
    by_speed = summarize_trials(rows, speeds)["by_speed"]
    assert [(group["speed_m_s"], group["trials"]) for group in by_speed] == [
        (0.7, 2),
        (1.0, 2),
    ]


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
            ["actor.fsdp_config.mixed_precision.cast_forward_inputs=True"],
            "cast_forward_inputs must be False",
        ),
        (
            ["actor.fsdp_config.mixed_precision.reduce_dtype=bf16"],
            "reduce_dtype must be fp32",
        ),
        (
            ["actor.fsdp_config.mixed_precision.reduce_dtype=null"],
            "reduce_dtype must be fp32",
        ),
        (
            ["actor.fsdp_config.amp_autocast.enabled=True"],
            "amp_autocast.enabled must be False",
        ),
        (
            [
                "actor.fsdp_config.mixed_precision.param_dtype=null",
                "actor.fsdp_config.amp_autocast.enabled=True",
            ],
            "amp_autocast.enabled cannot be combined",
        ),
        (
            [
                "actor.fsdp_config.mixed_precision.param_dtype=null",
                "actor.fsdp_config.mixed_precision.reduce_dtype=null",
                "actor.fsdp_config.amp_autocast.enabled=True",
            ],
            None,
        ),
        (
            ["actor.fsdp_config.grad_scaler.enabled=True"],
            "grad_scaler.enabled must be False",
        ),
        (["env.train.max_episode_seconds=0.16", "algorithm.replay_length=8"], None),
        (
            ["actor.cuda_graph=True", "actor.optim.lr_scheduler=cosine"],
            "actor.optim needs lr_scheduler=constant",
        ),
        (
            ["actor.cuda_graph=True", "+actor.muon_optim.lr_warmup_steps=10"],
            "with 10 warmup steps",
        ),
        (
            ["actor.cuda_graph=True", "+actor.muon_optim.lr_warmup_steps=null"],
            None,
        ),
        (
            ["actor.cuda_graph=True", "actor.world_model_optim.capturable=False"],
            "actor.world_model_optim needs capturable=True",
        ),
        (["actor.cuda_graph=True"], None),
        (["+runner.enable_decoupled_mode=True"], "enable_decoupled_mode=False"),
        (["rollout.rollout_backend=phyai"], "rollout_backend='huggingface'"),
        (["env.train.video_cfg.save_video=True"], "save_video must be false"),
        (
            ["+env.train.data_collection.enabled=True"],
            "data_collection.enabled must be false",
        ),
        (["+env.delay_sampler={type: fixed}"], "delay_sampler is not supported"),
        (["+env.train.eval_tasks=[tracking]"], "env.train.eval_tasks must be unset"),
        (
            [
                "runner.val_check_interval=1",
                "env.eval.eval_tasks=[tracking,push_recovery]",
            ],
            "env.eval.eval_tasks must be",
        ),
        (["+runner.weight_sync_interval=2"], "weight_sync_interval must be 1"),
    ],
)
def test_training_config_rejects_unsupported_values(overrides, error):
    from rlinf.config import validate_fsdp_cfg

    cfg = _compose(
        "supermimic_lafan1_gencoord", ["env.train.motion_path=/data/clips", *overrides]
    )
    if error is None:
        validate_fsdp_cfg(cfg.actor)
        validate_supermimic_cfg(cfg)
        return
    with pytest.raises(ValueError, match=error):
        validate_supermimic_cfg(cfg)


@pytest.mark.parametrize("world, error", [(2, None), (4, "would be empty")])
def test_placement_leaves_every_critic_minibatch_rows(world, error):
    cfg = _compose(
        "supermimic_lafan1_gencoord",
        [
            "env.train.motion_path=/data/clips",
            "algorithm.imagine_batch=8",
            "algorithm.H=2",
            "algorithm.critic_minibatches=8",
        ],
    )
    placement = SimpleNamespace(get_world_size=lambda name: world)
    if error is None:
        check_supermimic_placement(cfg, placement)
        return
    with pytest.raises(ValueError, match=error):
        check_supermimic_placement(cfg, placement)


@pytest.mark.parametrize("ckpt_path", ["null", "/data/full_weights.pt"])
def test_evaluation_config_needs_the_policy_weights(monkeypatch, ckpt_path):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base="1.1"):
        cfg = compose(
            config_name="supermimic_lafan1_eval",
            overrides=[
                "env.eval.motion_path=/data/clips",
                f"runner.ckpt_path={ckpt_path}",
            ],
        )
    assert "actor" not in cfg and "algorithm" not in cfg
    if ckpt_path != "null":
        validate_supermimic_cfg(cfg, only_eval=True)
        return
    with pytest.raises(ValueError, match="needs runner.ckpt_path"):
        validate_supermimic_cfg(cfg, only_eval=True)


def test_push_recovery_eval_config_is_standalone_and_accepts_maxcoord(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    with initialize_config_dir(config_dir=str(EVAL_CONFIG_DIR), version_base="1.1"):
        cfg = compose(
            config_name="supermimic_push_recovery",
            overrides=[
                "runner.ckpt_path=/data/full_weights.pt",
                "rollout.model.representation=maxcoord",
            ],
        )
    assert "actor" not in cfg and "algorithm" not in cfg
    validated = validate_supermimic_cfg(cfg, only_eval=True)
    assert validated.env.eval.representation == "maxcoord"


def test_exported_config_records_the_motion_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPERMIMIC_MOTION_PATH", "/data/lafan1_g1")
    from rlinf.envs.sim.g1_mimic.eval.eval_metrics import export_eval_results

    cfg = OmegaConf.create(
        {
            "runner": {
                "ckpt_path": "/data/full_weights.pt",
                "logger": {"experiment_name": "supermimic_lafan1_eval"},
            },
            "env": {"eval": {"motion_path": "${oc.env:SUPERMIMIC_MOTION_PATH}"}},
        }
    )
    export_eval_results(
        tmp_path,
        cfg,
        {"tasks": ["tracking"], "num_clips": 1, "num_jobs": 1},
        {"normalized_return": 1.0},
        {},
        {},
    )
    saved = OmegaConf.load(tmp_path / "config.yaml")
    assert saved.env.eval.motion_path == "/data/lafan1_g1"


def test_lafan1_q0_export_needs_no_license_or_output(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[2] / "toolkits/supermimic/prepare_lafan1.py"
    spec = importlib.util.spec_from_file_location("prepare_lafan1", path)
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    pose = torch.linspace(-0.5, 0.5, 29)
    monkeypatch.setattr(prepare, "default_q0", lambda: pose.clone())
    out = tmp_path / "g1_q0.pt"
    monkeypatch.setattr("sys.argv", ["prepare_lafan1.py", "--export-q0", str(out)])

    assert prepare.main() == 0
    assert torch.equal(torch.load(out, weights_only=True), pose)

    seen_headers = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, size=-1):
            return b""

    def fake_urlopen(req, timeout):
        seen_headers.update(dict(req.header_items()))
        return _FakeResponse()

    monkeypatch.setattr(prepare.urllib.request, "urlopen", fake_urlopen)
    dest = tmp_path / "sample.csv"
    prepare._fetch(
        "https://example.invalid/sample.csv", dest, use_curl=False, timeout=5
    )
    assert seen_headers.get("User-agent") == "RLinf/1.0"
    assert dest.is_file()


def test_embodied_runners_delegate_eval_finalization_to_env_hook():
    from types import SimpleNamespace

    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.runners.embodied_runner import EmbodiedRunner

    class _Handle:
        def __init__(self, value):
            self._value = value

        def wait(self):
            return self._value

    class _FakeEnv:
        def __init__(self, finalized):
            self._finalized = finalized
            self.calls = []

        def evaluate(self, **_kwargs):
            return _Handle([{"success_once": torch.tensor([1.0, 0.0])}])

        def execute_on(self, *ranks):
            self._ranks = ranks
            return self

        def finalize_eval_results(self, env_results, step=None):
            self.calls.append((self._ranks, len(env_results), step))
            return _Handle([self._finalized])

    class _FakeRollout:
        def evaluate(self, **_kwargs):
            return _Handle([None])

    cfg = OmegaConf.create({"runner": {"enable_decoupled_mode": False}})
    logged = []
    fake_logger = SimpleNamespace(info=logged.append)

    eval_runner = object.__new__(EmbodiedEvalRunner)
    eval_runner.cfg = cfg
    eval_runner.env = _FakeEnv(({"normalized_return": 0.85}, "/tmp/eval_out"))
    eval_runner.rollout = _FakeRollout()
    eval_runner.env_channel = eval_runner.rollout_channel = None
    eval_runner.logger = fake_logger
    assert eval_runner.evaluate() == {"normalized_return": 0.85}
    assert eval_runner.env.calls == [((0,), 1, None)]
    assert any("/tmp/eval_out" in msg for msg in logged)

    train_runner = object.__new__(EmbodiedRunner)
    train_runner.cfg = cfg
    train_runner.global_step = 2000
    train_runner.env = _FakeEnv(({"normalized_return": 0.80}, None))
    train_runner.rollout = _FakeRollout()
    train_runner.env_channel = train_runner.rollout_channel = None
    train_runner.logger = fake_logger
    assert train_runner.evaluate() == {"normalized_return": 0.80}
    assert train_runner.env.calls == [((0,), 1, 2000)]

    fallback_runner = object.__new__(EmbodiedEvalRunner)
    fallback_runner.cfg = cfg
    fallback_runner.env = _FakeEnv(None)
    fallback_runner.rollout = _FakeRollout()
    fallback_runner.env_channel = fallback_runner.rollout_channel = None
    fallback_runner.logger = fake_logger
    assert fallback_runner.evaluate()["success_once"] == pytest.approx(0.5)


def test_maxcoord_config_shares_its_representation_with_the_envs():
    from rlinf.envs.layout import EnvLayout

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
    lengths = EnvLayout(cfg.env.train, 1, "env.train").reference_lengths("trajectory")
    assert lengths == {None: float(cfg.env.train.max_episode_steps)}


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

    left, right = torch.randn(6, 4, generator=gen), torch.randn(6, 5, generator=gen)
    got_left, got_right = split_moments(
        pack_moments(torch.cat((left, right), dim=-1)), 4
    )
    torch.testing.assert_close(got_left, pack_moments(left), atol=0, rtol=0)
    torch.testing.assert_close(got_right, pack_moments(right), atol=0, rtol=0)


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


def _ring_ending_every_third_step(ends: list[bool], steps: int = 12):
    """A ring of ``len(ends)`` envs; env ``i`` ends every third step if ``ends[i]``."""
    rows = len(ends)
    ring = TransitionRing(rows, 20, 1, CPU)
    for t in range(steps):
        step = {
            name: torch.zeros((rows, width) if width else rows, dtype=dtype)
            for name, (width, dtype) in TRANSITION_FIELDS.items()
        }
        step["state"] = torch.full((rows, 1), float(t))
        step["terminated"] = torch.tensor(ends) & (t % 3 == 2)
        ring.add(step)
    return ring


def test_replay_windows_come_only_from_envs_with_long_episodes():
    ring = _ring_ending_every_third_step([True, False])
    assert ring.has_window(8)
    windows = ring.sample(8, 64)
    assert not windows["terminated"][:, :-1].any()
    time = windows["state"][..., 0]
    assert torch.equal(time - time[:, :1], torch.arange(8.0).expand(64, -1))

    short = _ring_ending_every_third_step([True, True])
    assert not short.has_window(8)
    assert short.has_window(3)
    with pytest.raises(RuntimeError, match="no legal 8-step window"):
        short.sample(8, 4)


def test_replay_draws_reuse_one_set_of_window_starts():
    ring = _ring_ending_every_third_step([True, False])
    torch.manual_seed(0)
    sampled = [ring.sample(8, 16) for _ in range(3)]
    torch.manual_seed(0)
    starts = ring.windows(8)
    drawn = [ring.draw(starts, 16) for _ in range(3)]
    for one, other in zip(sampled, drawn, strict=True):
        for name in one:
            assert torch.equal(one[name], other[name])


class _Done:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return self.value


def _runner_with_replay_answers(answers: list[list[bool]], replay_length: int = 1000):
    runner = object.__new__(SuperMimicRunner)
    runner.ratio, runner.k, runner.replay_length = 0.4, 8, replay_length
    runner.anchor, runner.env_steps, runner.replay_steps = None, 0, 0
    replies = iter(answers)
    runner.collected = []

    def collect(steps):
        runner.collected.append(steps)
        runner.env_steps += steps
        runner.replay_steps += steps
        return _Done(), _Done(), next(replies)

    runner._collect = collect
    return runner


def test_env_statistics_average_rank_means_and_add_episode_counts():
    runner = object.__new__(SuperMimicRunner)
    per_rank = [
        {"sim/reward_mean": 0.02, "sim/episodes_finished": 30.0},
        {"sim/reward_mean": 0.04, "sim/episodes_finished": 10.0},
    ]
    merged = runner._merge_env_metrics(per_rank)
    assert merged["sim/reward_mean"] == pytest.approx(0.03)
    assert merged["sim/episodes_finished"] == 40.0


def test_due_updates_wait_until_every_rank_can_draw_a_window():
    runner = _runner_with_replay_answers([[True, False], [True, False], [True, True]])
    _, _, due = runner._collect_until_update()
    assert runner.collected == [9, 1, 1]
    assert due == 1
    assert runner.anchor == 7.5

    stuck = _runner_with_replay_answers([[True, False]] * 3, replay_length=2)
    with pytest.raises(RuntimeError, match=r"actor ranks \[1\]"):
        stuck._collect_until_update()
    assert stuck.collected == [9, 1, 1]


def test_flat_tensor_packer_round_trips_plain_module():
    from torch import nn

    from rlinf.models.embodiment.supermimic.ipc import FlatTensorPacker

    class _Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 3)
            self.register_buffer("count", torch.tensor(0.0, dtype=torch.float64))

    torch.manual_seed(0)
    src, dst = _Toy(), _Toy()
    src.count.fill_(42.0)
    names = ["count", "linear.bias", "linear.weight"]

    src_packer = FlatTensorPacker(src, names)
    dst_packer = FlatTensorPacker(dst, names)
    dst_packer.buffer.copy_(src_packer.pack())
    dst_packer.unpack()

    for name in names:
        assert torch.equal(src.state_dict()[name], dst.state_dict()[name]), name


@needs_mjlab
def test_flat_tensor_packer_round_trips_actor_weights_and_norm_buffers():
    from rlinf.models.embodiment.supermimic.ipc import FlatTensorPacker
    from rlinf.models.embodiment.supermimic.supermimic_policy import (
        SuperMimicPolicy,
        sync_param_names,
    )

    torch.manual_seed(0)
    src = SuperMimicPolicy(representation="gencoord", torch_compile=False)
    dst = SuperMimicPolicy(representation="gencoord", torch_compile=False)
    src.fold_policy_stats(
        pack_moments(torch.randn(8, src.representation.encoding_dim)),
        pack_moments(torch.randn(8, ACTOR_GOAL_DIM)),
    )
    names = sync_param_names(src)
    assert names and all(not n.startswith(("dynamics.", "critic.")) for n in names)

    src_packer = FlatTensorPacker(src, names)
    dst_packer = FlatTensorPacker(dst, names)
    dst_packer.buffer.copy_(src_packer.pack())
    dst_packer.unpack()

    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    for name in names:
        assert torch.equal(src_sd[name], dst_sd[name]), name


def test_flat_dict_buffer_round_trips_obs_and_transition_packets():
    from rlinf.models.embodiment.supermimic.ipc import (
        FlatDictBuffer,
        obs_packet_schema,
        transition_packet_schema,
    )

    torch.manual_seed(0)
    num_envs, state_dim, enc_dim = 4, STATE_DIM, 68
    obs_schema = obs_packet_schema(num_envs, enc_dim)
    obs_src = FlatDictBuffer(obs_schema, CPU)
    obs_dst = FlatDictBuffer(obs_schema, CPU)
    obs_data = {
        "enc": torch.randn(num_envs, enc_dim),
        "goal": torch.randn(num_envs, ACTOR_GOAL_DIM),
        "enc_stats": pack_moments(torch.randn(num_envs, enc_dim)),
        "goal_stats": pack_moments(torch.randn(num_envs, ACTOR_GOAL_DIM)),
    }
    obs_dst.buffer.copy_(obs_src.pack(obs_data))
    for name, value in obs_data.items():
        assert torch.equal(obs_dst.views[name], value), name

    tr_schema = transition_packet_schema(num_envs, state_dim, enc_dim)
    tr_src = FlatDictBuffer(tr_schema, CPU)
    tr_dst = FlatDictBuffer(tr_schema, CPU)
    tr_data = {
        "state": torch.randn(num_envs, state_dim),
        "action": torch.randn(num_envs, ACTION_DIM),
        "prev_action": torch.randn(num_envs, ACTION_DIM),
        "motion_time": torch.randint(0, 100, (num_envs,), dtype=torch.long),
        "terminated": torch.tensor([True, False, False, True]),
        "truncated": torch.tensor([False, True, False, False]),
        "sim_context": torch.randn(num_envs, SIM_CONTEXT_DIM),
        "enc_stats": obs_data["enc_stats"],
        "goal_stats": obs_data["goal_stats"],
        "target_stats": pack_moments(torch.randn(num_envs, ACTION_DIM)),
    }
    tr_dst.buffer.copy_(tr_src.pack(tr_data))
    for name, value in tr_data.items():
        assert torch.equal(tr_dst.views[name], value), name


def test_replay_has_window_fast_path_and_older_fallback():
    ring = TransitionRing(num_envs=2, capacity=32, state_dim=STATE_DIM, device=CPU)

    def push(done: bool):
        ring.add(
            {
                "state": torch.zeros(2, STATE_DIM),
                "action": torch.zeros(2, ACTION_DIM),
                "prev_action": torch.zeros(2, ACTION_DIM),
                "motion_time": torch.zeros(2, dtype=torch.long),
                "terminated": torch.full((2,), done, dtype=torch.bool),
                "truncated": torch.zeros(2, dtype=torch.bool),
                "sim_context": torch.zeros(2, SIM_CONTEXT_DIM),
            }
        )

    for _ in range(4):
        push(False)
    assert ring.has_window(4)
    # Push 10 steps where every step terminates: newest 8 steps have no legal
    # 4-step window, so has_window must fall back and still find the older one.
    for _ in range(10):
        push(True)
    assert ring.has_window(4)
    sample = ring.sample(1, 16, fields=("state", "motion_time"))
    assert set(sample) == {"state", "motion_time"}
    assert sample["state"].shape == (16, 1, STATE_DIM)


def test_reset_draws_before_the_step_is_recorded_and_keeps_the_action(monkeypatch):
    import rlinf.envs.sim.g1_mimic.g1_mimic_env as g1_mimic_env

    rows = 2
    order = []
    never = torch.zeros(rows, dtype=torch.bool)
    env = object.__new__(G1MimicEnv)
    env.num_envs, env.device, env.action_noise_std = rows, CPU, 0.0
    env._compile_cache = None
    env.prev_action = torch.zeros(rows, ACTION_DIM)
    env._observed = {
        "state": torch.zeros(rows, STATE_DIM),
        "frame": torch.zeros(rows, dtype=torch.long),
    }
    env.sim = SimpleNamespace(
        state=torch.zeros(rows, STATE_DIM),
        sim_context=torch.zeros(rows, SIM_CONTEXT_DIM),
        pd_offset=torch.zeros(ACTION_DIM),
        action_scale=torch.ones(ACTION_DIM),
        motion_time=torch.zeros(rows, dtype=torch.long),
        body_state=lambda: (),
        reset=lambda env_ids, motion_id, time: None,
        step=lambda action: {
            "frame": torch.zeros(rows, dtype=torch.long),
            "failure": never,
            "timeout": torch.tensor([True, False]),
            "clip_end": never,
        },
    )
    env.motion = SimpleNamespace(
        body_state=lambda frame: (),
        locate=lambda frames: (torch.zeros_like(frames), torch.zeros_like(frames)),
    )
    env.reset_sampler = SimpleNamespace(
        record=lambda *args: order.append("record"),
        sample=lambda n: order.append("sample") or torch.zeros(n, dtype=torch.long),
    )
    env._clear_metrics()
    monkeypatch.setattr(
        g1_mimic_env, "run_compiled", lambda name, fn, *args, cache: torch.zeros(rows)
    )

    transition = env.step(torch.full((rows, ACTION_DIM), 0.25))
    # The recorded step may refresh the distribution; the reset must not see it.
    assert order == ["sample", "record"]
    assert torch.equal(transition["action"], torch.full((rows, ACTION_DIM), 0.25))
    assert torch.equal(env.prev_action[0], torch.zeros(ACTION_DIM))
    assert torch.equal(env.prev_action[1], torch.full((ACTION_DIM,), 0.25))


def test_update_spans_evaluate_at_each_boundary_and_sum_their_times(monkeypatch):
    from rlinf.scheduler import Tracer

    monkeypatch.setattr(Tracer, "_unavailable", True)
    runner = object.__new__(SuperMimicRunner)
    runner.global_step, runner.max_steps = 8, 100
    runner.cfg = SimpleNamespace(
        runner=SimpleNamespace(val_check_interval=10, save_interval=-1)
    )
    calls = []
    runner.actor = SimpleNamespace(
        run_updates=lambda count: calls.append(("update", count)) or _Done([{}])
    )
    runner._set_global_step = lambda: None
    runner.update_rollout_weights = lambda: calls.append(("sync", None))
    runner.evaluate = lambda: calls.append(("eval", runner.global_step)) or {"r": 1.0}
    runner.metric_logger = SimpleNamespace(log=lambda data, step: None)

    spans = boundary_update_spans(8, 13, 100, -1, 10)
    *_, eval_metrics, last_step, durations = runner._execute_update_spans(spans)
    assert spans == [2, 10, 1]
    assert [call for call in calls if call[0] == "eval"] == [("eval", 10), ("eval", 20)]
    assert [call[0] for call in calls].count("sync") == len(spans)
    assert (eval_metrics, last_step, runner.global_step) == ({"eval/r": 1.0}, 20, 21)
    assert set(durations) == {"actor_training", "sync_weights", "eval"}


def test_critic_minibatches_can_hold_every_row():
    torch.manual_seed(0)
    seen = set()
    for _ in range(50):
        rows = critic_minibatch_rows(7, 4, CPU)
        assert rows.shape == (4, 1) and rows.unique().numel() == 4
        seen.update(rows.flatten().tolist())
    assert seen == set(range(7))


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


@pytest.mark.parametrize("saved,resumed", [(False, True), (True, False)])
def test_resumed_adam_keeps_the_step_flags_of_the_run(saved, resumed):
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.ones(2)
    before = torch.optim.Adam([param], lr=0.1)
    before.step()
    checkpoint = before.state_dict()
    checkpoint["param_groups"][0]["capturable"] = saved
    after = torch.optim.Adam([param], lr=0.1, capturable=resumed)
    after.register_load_state_dict_pre_hook(keep_step_placement)
    after.load_state_dict(checkpoint)
    assert after.param_groups[0]["capturable"] is resumed
    assert float(after.state[param]["step"]) == 1.0
    torch.testing.assert_close(
        after.state[param]["exp_avg"], before.state[param]["exp_avg"]
    )


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

    # Empty gradients with as_tensor=True must stay on the parameters' device.
    empty = torch.nn.Parameter(torch.zeros(2))
    zero_norm = get_grad_norm([empty], None, as_tensor=True)
    assert torch.is_tensor(zero_norm) and zero_norm.device == empty.device
    assert float(zero_norm) == 0.0


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
    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import (
        asset_default_state,
        g1_kinematics,
        soft_joint_limits,
    )
    from rlinf.envs.sim.g1_mimic.tasks.reward import tracking_reward

    state = asset_default_state().repeat(2, 1)
    bodies = g1_kinematics().forward(state)
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
    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import asset_joint_limits

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
    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import g1_kinematics

    flat = states.reshape(-1, STATE_DIM)
    pos, rot, lin, ang = g1_kinematics().forward(flat)
    full = representation.stored_state(flat, (pos, _rotmat_to_quat(rot), lin, ang))
    return full.reshape(*states.shape[:-1], -1)


@needs_mjlab
@pytest.mark.parametrize("name", ["gencoord", "maxcoord"])
def test_representation_reproduces_the_tracked_bodies(name):
    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import g1_kinematics
    from rlinf.envs.sim.g1_mimic.kinematics.representations import make_representation

    representation = make_representation(name)
    state = _random_states(32, seed=0)
    full = _stored(representation, state)
    enc = representation.encode(full)
    assert full.shape[-1] == representation.state_dim
    assert enc.shape[-1] == representation.encoding_dim
    root = representation.root(full)
    for actual, expected in zip(
        representation.bodies(root, enc), g1_kinematics().forward(state), strict=True
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

    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import asset_default_state
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


def _small_actor(cfg, mesh, motion):
    """The SuperMimic actor of ``cfg`` on one rank, without Ray, over a replay
    of four envs that follow ``motion``'s two clips."""
    from rlinf.hybrid_engines.fsdp import MixedPrecisionPolicy, OffloadPolicy
    from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model
    from rlinf.models.embodiment.supermimic.supermimic_policy import (
        SuperMimicPolicy,
    )
    from rlinf.workers.actor import fsdp_supermimic_policy_worker as module

    def base_init(self, cfg):
        self.cfg = cfg
        self._rank, self._world_size = 0, 1
        self.version = 0

    orig_init = module.EmbodiedFSDPActor.__init__
    module.EmbodiedFSDPActor.__init__ = base_init
    try:
        actor = module.EmbodiedSuperMimicFSDPPolicy(cfg)
    finally:
        module.EmbodiedFSDPActor.__init__ = orig_init
    actor._timer_metrics, actor._accelerator_type = {}, None
    actor.log_info = lambda msg: None
    actor.device = "cuda:0"
    actor._device_mesh, actor._dp_group = mesh, mesh["fsdp"].get_group()
    from rlinf.utils.precision import torch_dtype_from_precision

    mp_cfg = cfg.actor.fsdp_config.mixed_precision
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch_dtype_from_precision(mp_cfg.param_dtype),
        reduce_dtype=torch_dtype_from_precision(mp_cfg.reduce_dtype),
        cast_forward_inputs=mp_cfg.get("cast_forward_inputs", True),
    )
    actor._strategy = SimpleNamespace(
        wrap_model=lambda model, device_mesh: apply_fsdp2_to_model(
            model,
            cfg.actor.fsdp_config,
            device_mesh,
            mp_policy,
            OffloadPolicy(),
            reshard_after_forward=bool(
                cfg.actor.fsdp_config.get("reshard_after_forward", True)
            ),
        )
    )
    actor.model_provider_func = lambda: SuperMimicPolicy.from_config(cfg.actor.model)
    actor.setup_model_and_optimizer()
    actor.motion = motion
    representation = actor.model.representation
    actor.replay = TransitionRing(4, 64, representation.state_dim, motion.frames.device)
    starts = torch.tensor([0, 3, 40, 45], device=motion.frames.device)
    gen = torch.Generator(device=motion.frames.device).manual_seed(1)
    context = 0.3 * torch.randn(4, SIM_CONTEXT_DIM, device=gen.device, generator=gen)
    prev = torch.zeros(4, ACTION_DIM, device=gen.device)
    for t in range(24):
        action = 0.1 * torch.randn(4, ACTION_DIM, device=gen.device, generator=gen)
        actor.replay.add(
            {
                "state": _stored(representation, motion.rows(starts + t)),
                "action": action,
                "prev_action": prev,
                "motion_time": starts + t,
                "terminated": torch.zeros(4, dtype=torch.bool),
                "truncated": torch.zeros(4, dtype=torch.bool),
                "sim_context": context,
            }
        )
        prev = action
    return actor


def _update_with_and_without_graph(
    rank: int, port: int, out: str, precision_mode: str
) -> None:
    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import asset_default_state
    from rlinf.hybrid_engines.fsdp import FSDPModule
    from rlinf.hybrid_engines.fsdp.utils import create_device_mesh

    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=1
    )
    Worker.torch_platform.set_device(0)
    try:
        pose = asset_default_state()
        clips = []
        for length in (40, 30):
            clip = pose.repeat(length, 1)
            clip[:, 0] = torch.linspace(0.0, 1.0, length)
            wave = torch.sin(torch.linspace(0, 3, length))[:, None]
            clip[:, JOINT_POS] += 0.05 * wave
            clips.append(clip.to("cuda:0"))
        motion = MotionLibrary(clips, q0=pose[JOINT_POS])
        mode_overrides = {
            "fp32": [
                "actor.fsdp_config.mixed_precision.param_dtype=null",
                "actor.fsdp_config.mixed_precision.reduce_dtype=null",
                "actor.fsdp_config.amp_autocast.enabled=False",
            ],
            "amp_bf16": [
                "actor.fsdp_config.mixed_precision.param_dtype=null",
                "actor.fsdp_config.mixed_precision.reduce_dtype=null",
                "actor.fsdp_config.amp_autocast.enabled=True",
            ],
            "fsdp_bf16": [
                "actor.fsdp_config.amp_autocast.enabled=False",
                "actor.fsdp_config.mixed_precision.param_dtype=bf16",
                "actor.fsdp_config.mixed_precision.reduce_dtype=fp32",
                "actor.fsdp_config.mixed_precision.cast_forward_inputs=False",
            ],
        }[precision_mode]
        results = {}
        for graph in (False, True):
            cfg = _compose(
                "supermimic_lafan1_gencoord",
                [
                    "env.train.motion_path=/data/clips",
                    "env.train.total_num_envs=4",
                    "algorithm.H=4",
                    "algorithm.wm_batches=2",
                    "algorithm.wm_window_batch=16",
                    "algorithm.imagine_batch=16",
                    "algorithm.critic_epochs=2",
                    "algorithm.critic_minibatches=2",
                    "algorithm.replay_length=64",
                    "actor.model.torch_compile=False",
                    f"actor.cuda_graph={graph}",
                    *mode_overrides,
                ],
            )
            mesh = create_device_mesh(
                1,
                sharding_strategy=cfg.actor.fsdp_config.get(
                    "sharding_strategy", "full_shard"
                ),
            )
            torch.manual_seed(0)
            actor = _small_actor(validate_supermimic_cfg(cfg), mesh, motion)
            # First 4 single updates warm up + record the graph; the final
            # multi-update call exercises side-stream replay prefetching too.
            metrics = [actor.run_updates(1) for _ in range(4)]
            metrics.append(actor.run_updates(3))
            params = {
                name: param.full_tensor().cpu()
                for name, param in actor.model.named_parameters()
            }
            recorded = actor.update_graphs is not None
            units = {
                name
                for name, module in actor.model.named_children()
                if isinstance(module, FSDPModule)
            }
            results[graph] = (
                metrics,
                params,
                dict(actor.phase_optimizer_steps),
                recorded,
                units,
            )
        torch.save(results, f"{out}/updates.pt")
    finally:
        dist.destroy_process_group()


@needs_mjlab
@pytest.mark.skipif(
    not _CUDA_AVAILABLE or not dist.is_nccl_available(),
    reason="the update graph needs a CUDA device and NCCL",
)
@pytest.mark.parametrize("precision_mode", ["fp32", "amp_bf16", "fsdp_bf16"])
def test_recorded_update_replays_the_eager_update(tmp_path, precision_mode):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(
        _update_with_and_without_graph,
        args=(port, str(tmp_path), precision_mode),
        nprocs=1,
    )
    results = torch.load(tmp_path / "updates.pt")
    eager, replayed = results[False], results[True]
    assert not eager[3] and replayed[3]
    assert eager[4] == replayed[4] == {"dynamics", "actor", "critic"}
    assert eager[2] == replayed[2] == {"world_model": 14, "actor": 7, "critic": 28}
    assert eager[0] == replayed[0]
    for name, param in eager[1].items():
        torch.testing.assert_close(replayed[1][name], param, atol=0, rtol=0)


@needs_mjlab
@pytest.mark.parametrize(
    "storage,query",
    [("cpu", "cpu"), ("cpu", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:0", "cpu")],
)
def test_reference_fk_matches_online_fk(storage, query):
    from rlinf.envs.sim.g1_mimic.kinematics.g1_fk import (
        asset_default_state,
        g1_kinematics,
    )

    if "cuda" in storage + query and not _CUDA_AVAILABLE:
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
    result = motion.body_state(motion.global_frame(ids, times))
    expected = g1_kinematics().forward(motion.state_at(ids, times))
    assert all(part.device == motion.frames.device for part in motion._kin_cache)
    for actual, wanted in zip(result, expected, strict=True):
        assert actual.device == ids.device
        torch.testing.assert_close(actual, wanted, atol=2e-5, rtol=2e-5)


@needs_warp
@pytest.mark.parametrize("mode", ["add", "set"])
def test_push_writes_only_the_horizontal_root_velocity(mode):
    from rlinf.envs.sim.g1_mimic.eval.push_recovery import default_standing_state
    from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env

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


@needs_warp
def test_reset_drops_the_previous_episode_from_the_solver():
    from rlinf.envs.sim.g1_mimic.eval.push_recovery import default_standing_state
    from rlinf.envs.sim.g1_mimic.mjlab_env import MjlabG1Env

    device = torch.device("cuda:0")
    pose = default_standing_state()
    motion = MotionLibrary([pose.repeat(30, 1)], q0=pose[JOINT_POS])
    env = MjlabG1Env(2, motion, device, randomize=False)
    for _ in range(5):
        env.step(torch.full((2, ACTION_DIM), 0.5, device=device))
    warm = env.sim.data.qacc_warmstart.clone()
    assert bool((warm.abs().amax(1) > 0).all())
    ids = torch.zeros(1, dtype=torch.long, device=device)
    env.reset(ids, torch.zeros_like(ids), torch.zeros_like(ids))
    # Contact solves start from qacc_warmstart, so a stale one carries the
    # previous episode into the first step of the next.
    assert bool((env.sim.data.qacc_warmstart[0] == 0).all())
    assert float(env.sim.data.time[0]) == 0.0
    torch.testing.assert_close(env.sim.data.qacc_warmstart[1], warm[1], atol=0, rtol=0)
