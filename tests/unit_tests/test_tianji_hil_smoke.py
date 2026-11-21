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

"""CPU-runnable smoke tests for Tianji HIL, DAgger, LoopInfra unification, and Stage2 RLT."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import importlib.util
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_module_direct(module_name: str, rel_path: str):
    """Load a standalone Python file without triggering heavy package __init__ imports."""
    path = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tianji_rlt_core_imports():
    """Verify restored Tianji, RLT, runner, config, and env modules import cleanly."""
    standalone_modules = [
        "rlinf.utils.tianji_startup",
        "rlinf.utils.xense_tactile",
        "rlinf.utils.rtc",
        "rlinf.envs",
    ]
    for name in standalone_modules:
        mod = importlib.import_module(name)
        assert mod is not None

    # Full-stack RLinf modules require optional RLinf runtime dependencies.
    pytest.importorskip("regex", reason="optional RLinf dependency 'regex' not installed")
    pytest.importorskip("ray", reason="optional RLinf dependency 'ray' not installed")
    pytest.importorskip(
        "gymnasium", reason="optional RLinf dependency 'gymnasium' not installed"
    )

    full_stack_modules = [
        "rlinf.algorithms.rlt.action_grid",
        "rlinf.algorithms.rlt.direct_human_bc",
        "rlinf.algorithms.rlt.human_bc",
        "rlinf.algorithms.rlt.route",
        "rlinf.algorithms.rlt.transition",
        "rlinf.models.embodiment.mlp_policy.rlt_mlp_policy",
        "rlinf.config",
        "rlinf.runners.async_embodied_runner",
        "rlinf.envs.realworld.tianji",
        "rlinf.envs.realworld.tianji.tianji_contract",
        "rlinf.envs.realworld.tianji.tianji_env",
        "rlinf.envs.realworld.tianji.tianji_hil",
        "rlinf.envs.realworld.tianji.tianji_joint",
        "rlinf.envs.realworld.tianji.tianji_episode_archive",
        "rlinf.envs.realworld.tianji.car_monitor_tap",
    ]
    for name in full_stack_modules:
        mod = importlib.import_module(name)
        assert mod is not None


def test_supported_env_type_preserves_upstream_and_tianji():
    """Ensure SupportedEnvType retains both upstream sim/real envs and Tianji routing."""
    import warnings
    from rlinf.envs import SupportedEnvType

    values = {member.value for member in SupportedEnvType}
    for expected in ("libero", "maniskill", "real", "tianji"):
        assert expected in values, f"Missing SupportedEnvType value: {expected}"

    # Retired 'realworld' spelling remains accepted via _missing_ alias.
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert SupportedEnvType("realworld") is SupportedEnvType.REAL


def test_default_loopinfra_root_unification(monkeypatch):
    """Ensure default_loopinfra_root() defaults to ~/LoopInfra and honors LOOPINFRA_ROOT."""
    if importlib.util.find_spec("gymnasium") is not None:
        from rlinf.envs.realworld.tianji.tianji_contract import default_loopinfra_root
    else:
        mod = _load_module_direct(
            "_test_tianji_contract",
            "rlinf/envs/realworld/tianji/tianji_contract.py",
        )
        default_loopinfra_root = mod.default_loopinfra_root

    monkeypatch.delenv("LOOPINFRA_ROOT", raising=False)
    assert default_loopinfra_root() == Path.home() / "LoopInfra"

    monkeypatch.setenv("LOOPINFRA_ROOT", "/tmp/custom_loopinfra")
    assert default_loopinfra_root() == Path("/tmp/custom_loopinfra")

    assert default_loopinfra_root("~/other_loopinfra") == Path.home() / "other_loopinfra"


@pytest.mark.parametrize(
    "hil_mode,expect_training_hooks",
    [
        ("training", True),
        ("dagger", False),
        ("inference", False),
    ],
)
def test_tianji_hil_control_modes_standalone(hil_mode, expect_training_hooks):
    """Verify TianjiHILControl mode gating (training vs dagger vs inference) without Gymnasium."""
    tianji_hil = _load_module_direct(
        f"_test_tianji_hil_{hil_mode}",
        "rlinf/envs/realworld/tianji/tianji_hil.py",
    )
    hil = tianji_hil.TianjiHILControl(
        mode=hil_mode, subtask_prompts=["S01", "S02"]
    )
    hil.begin_episode()
    hil.command("start")

    if expect_training_hooks:
        hil.command("step_failure")
        assert hil.consume_step_failure() is True
        hil.command("next_subtask")
        assert hil.consume_milestones() == 1
    else:
        with pytest.raises(ValueError):
            hil.command("step_failure")
        assert hil.consume_step_failure() is False
        hil.command("next_subtask")
        assert hil.consume_milestones() == 0

    if hil_mode == "dagger":
        page = hil.page()
        assert "DAgger" in page
        assert "当前步失败" not in page

    if hil_mode in ("training", "dagger"):
        hil.command("discard")
        assert hil.snapshot()["outcome"] == "discard"
    else:
        with pytest.raises(ValueError):
            hil.command("discard")
        hil.command("success")


def test_car_monitor_tap_subtask_parameterization_standalone(monkeypatch):
    """Verify MonitorTap & HIL banner honor TIANJI_CAR_NUM_SUBTASKS / TIANJI_CAR_BANNER."""
    tap_mod = _load_module_direct(
        "_test_car_monitor_tap",
        "rlinf/envs/realworld/tianji/car_monitor_tap.py",
    )
    tianji_hil = _load_module_direct(
        "_test_tianji_hil_banner",
        "rlinf/envs/realworld/tianji/tianji_hil.py",
    )
    for num_subtasks, banner in (
        (6, "Stage2-30HZ-sub6-1118v1"),
        (15, "Stage2-30HZ-1118v1"),
    ):
        monkeypatch.setenv("TIANJI_CAR_NUM_SUBTASKS", str(num_subtasks))
        monkeypatch.setenv("TIANJI_CAR_BANNER", banner)
        tap = tap_mod.MonitorTap(endpoint=None)
        assert tap._max_stage == num_subtasks
        hil = tianji_hil.TianjiHILControl(
            mode="inference",
            subtask_prompts=[f"S{i:02d}" for i in range(1, num_subtasks + 1)],
            stage_switch_replan=True,
        )
        assert banner in hil.page()


@pytest.mark.parametrize(
    "hil_mode,expect_archive,expect_training_hooks",
    [
        ("training", True, True),
        ("dagger", True, False),
        ("inference", False, False),
    ],
)
def test_tianji_env_dummy_modes(
    tmp_path, monkeypatch, hil_mode, expect_archive, expect_training_hooks
):
    """Check TianjiEnv(backend='dummy') behavior across training, dagger, and inference."""
    pytest.importorskip(
        "gymnasium", reason="optional RLinf dependency 'gymnasium' not installed"
    )
    from rlinf.envs.realworld.tianji.tianji_env import TianjiConfig, TianjiEnv

    monkeypatch.delenv("TIANJI_CAR_1118", raising=False)
    prompt_file = tmp_path / "prompts.json"
    prompts = ["Subtask one.", "Subtask two."]
    prompt_file.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")

    recorder_dir = tmp_path / f"hil_{hil_mode}"
    env = TianjiEnv(
        TianjiConfig(
            backend="dummy",
            action_space="joint",
            hil_control=True,
            hil_port=0,
            hil_mode=hil_mode,
            subtask_prompts_file=str(prompt_file),
            image_hw=(8, 8),
            step_frequency=30,
            manual_episode_control_only=True,
            recorder_dir=str(recorder_dir),
            episode_archive=True,
            tactile_raw_archive=True,
            stream_chunks=True,
        )
    )
    try:
        if expect_archive:
            assert env.episode_archive is not None
        else:
            assert env.episode_archive is None

        assert env._shadow_schema is False

        with ThreadPoolExecutor(max_workers=1) as pool:
            reset_fut = pool.submit(env.reset)
            deadline = time.monotonic() + 3.0
            while env.hil.snapshot()["episode"] != 1:
                assert time.monotonic() < deadline
                time.sleep(0.01)
            env.hil.command("start")
            obs, _ = reset_fut.result(timeout=3.0)
        assert "state" in obs and "frames" in obs

        if expect_training_hooks:
            env.hil.command("step_failure")
            assert env._consume_step_failure() is True
            env.hil.command("next_subtask")
            assert env._consume_milestones() == 1
        else:
            with pytest.raises(ValueError):
                env.mark_step_failure()
            assert env._consume_step_failure() is False
            env.hil.command("next_subtask")
            assert env._consume_milestones() == 0

        if hil_mode == "dagger":
            page = env.hil.page()
            assert "DAgger" in page
            assert "当前步失败" not in page

        if hil_mode in ("training", "dagger"):
            env.hil.command("discard")
        else:
            with pytest.raises(ValueError):
                env.hil.command("discard")
            env.hil.command("success")

        env.step(env.get_hold_action())
    finally:
        env.close()


@pytest.mark.parametrize(
    "num_subtasks,banner",
    [
        (6, "Stage2-30HZ-sub6-1118v1"),
        (15, "Stage2-30HZ-1118v1"),
    ],
)
def test_car_subtask_parameterization(tmp_path, monkeypatch, num_subtasks, banner):
    """Verify 6 vs 15 car subtasks and custom HIL banner parameterization."""
    pytest.importorskip(
        "gymnasium", reason="optional RLinf dependency 'gymnasium' not installed"
    )
    from rlinf.envs.realworld.tianji.tianji_env import TianjiConfig, TianjiEnv

    monkeypatch.setenv("TIANJI_CAR_1118", "1")
    monkeypatch.setenv("TIANJI_CAR_NUM_SUBTASKS", str(num_subtasks))
    monkeypatch.setenv("TIANJI_CAR_BANNER", banner)

    prompt_file = tmp_path / f"prompts_{num_subtasks}.json"
    prompts = [f"S{i:02d} step" for i in range(1, num_subtasks + 1)]
    prompt_file.write_text(json.dumps({"prompts": prompts}), encoding="utf-8")

    env = TianjiEnv(
        TianjiConfig(
            backend="dummy",
            action_space="joint",
            hil_control=True,
            hil_port=0,
            hil_mode="inference",
            subtask_prompts_file=str(prompt_file),
            image_hw=(8, 8),
            step_frequency=30,
            manual_episode_control_only=True,
        )
    )
    try:
        assert len(env.subtask_prompts) == num_subtasks
        assert env.hil._stage_switch_replan is True
        assert banner in env.hil.page()
    finally:
        env.close()

    # Mismatched prompt count must be rejected.
    wrong_count = 15 if num_subtasks == 6 else 6
    wrong_file = tmp_path / f"prompts_wrong_{wrong_count}.json"
    wrong_file.write_text(
        json.dumps({"prompts": [f"S{i:02d}" for i in range(1, wrong_count + 1)]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        TianjiEnv(
            TianjiConfig(
                backend="dummy",
                action_space="joint",
                hil_control=False,
                hil_mode="dagger",
                subtask_prompts_file=str(wrong_file),
                image_hw=(8, 8),
                step_frequency=30,
            )
        )


def test_tianji_dagger_collect_yaml(monkeypatch):
    """Verify tianji_dagger_collect.yaml parses cleanly and disables RL actor updates."""
    cfg_path = REPO_ROOT / "examples/embodiment/config/tianji_dagger_collect.yaml"
    assert cfg_path.is_file()
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert data["runner"]["deploy_actor_updates"] is False
    assert data["env"]["train"]["discard_episodes"] is True
    override = data["env"]["train"]["override_cfg"]
    assert override["hil_mode"] == "dagger"
    assert override["episode_archive"] is True
    assert override["tactile_raw_archive"] is True
    assert data["algorithm"]["replay_buffer"]["auto_save"] is False
    assert data["algorithm"]["rlt_schedule"]["max_updates_per_train_step"] == 0

    hydra = pytest.importorskip("hydra", reason="optional dependency 'hydra' not installed")
    monkeypatch.setenv("EMBODIED_PATH", str(REPO_ROOT / "examples/embodiment"))
    with hydra.initialize_config_dir(
        version_base="1.1",
        config_dir=str(REPO_ROOT / "examples/embodiment/config"),
    ):
        composed = hydra.compose(config_name="tianji_dagger_collect")
    assert composed.env.train.override_cfg.hil_mode == "dagger"
    assert composed.runner.deploy_actor_updates is False
    assert composed.algorithm.rlt_schedule.max_updates_per_train_step == 0


def test_rlt_mlp_policy_cpu_forward():
    """Run a CPU-only forward smoke test for RLTMLPPolicy (joint + tactile)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("ray", reason="optional RLinf dependency 'ray' not installed")
    pytest.importorskip("regex", reason="optional RLinf dependency 'regex' not installed")
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.utils.xense_tactile import empty_tactile

    batch = 2
    z_dim = 64
    proprio_dim = 34
    action_dim = 16
    num_chunks = 12
    ref_chunks = 16

    model = RLTMLPPolicy(
        z_dim=z_dim,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        num_action_chunks=num_chunks,
        ref_num_action_chunks=ref_chunks,
        add_q_head=True,
        q_head_type="default",
        residual_ref=True,
        num_stage_actors=1,
        stage_onehot_dim=4,
        rollout_noise=False,
        tactile_enabled=True,
        tactile_embed_dim=16,
        critic_proprio_dim=42,
        critic_action_dim=20,
        critic_block_encoders=True,
        critic_value_head=True,
        critic_penalty_head=True,
        critic_penalty_only=True,
        critic_block_widths={"z": 16, "proprio": 16, "action": 16, "tactile": 16},
        critic_hidden_dims=[32, 32],
    ).eval()

    obs = {
        "z_rl": torch.zeros((batch, z_dim), dtype=torch.float32),
        "proprio": torch.zeros((batch, proprio_dim), dtype=torch.float32),
        "ref_chunk": torch.zeros((batch, ref_chunks, action_dim), dtype=torch.float32),
    }
    obs["proprio"][:, -4:][:, 0] = 1.0
    for key, arr in empty_tactile().items():
        obs[key] = torch.from_numpy(np.asarray(arr)).unsqueeze(0).repeat(
            (batch,) + (1,) * np.asarray(arr).ndim
        )

    with torch.no_grad():
        actions, _ = model.predict_action_batch(
            env_obs=obs, mode="eval", return_obs=True
        )
    actions = torch.as_tensor(actions)
    assert actions.shape == (batch, num_chunks, action_dim)
    assert torch.isfinite(actions).all()
