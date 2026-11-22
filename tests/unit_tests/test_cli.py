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

"""Unit and integration tests for the thin ``rlinf`` CLI."""

from __future__ import annotations

import io
import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from rlinf.cli import main
from rlinf.cli.catalog import (
    FEATURED_RECIPES,
    RecipeFamily,
    RecipeMode,
    discover_recipes,
    filter_recipes,
    find_repo_root,
    resolve_recipe,
)
from rlinf.cli.doctor import CheckStatus, run_doctor
from rlinf.cli.launcher import LaunchPlan, build_launch_plan, execute_launch_plan

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_find_repo_root_and_featured_recipes_exist():
    root = find_repo_root(REPO_ROOT)
    assert root == REPO_ROOT

    recipes = discover_recipes(root)
    assert len(recipes) >= 200

    # Every discovered recipe must reference a real config file and entry script,
    # and helper configs must be excluded.
    for spec in recipes:
        assert spec.config_file.is_file(), f"Missing config file: {spec.config_file}"
        assert spec.entry_script.is_file(), f"Missing entry script: {spec.entry_script}"
        assert not spec.name.startswith("base_")
        assert spec.name != "tp_comm_overlap_cfg"

    index = {(r.family, r.mode, r.name) for r in recipes}
    for key in FEATURED_RECIPES:
        assert key in index, f"Featured recipe missing from catalog: {key}"


def test_list_default_featured_and_filters():
    out = io.StringIO()
    err = io.StringIO()
    rc = main(["list"], repo_root=REPO_ROOT, stream=out, err_stream=err)
    assert rc == 0
    assert err.getvalue() == ""
    text = out.getvalue()
    assert "featured recipes" in text
    assert "maniskill_ppo_mlp" in text

    out_all = io.StringIO()
    rc = main(
        ["list", "--family", "sft", "--model", "openpi"],
        repo_root=REPO_ROOT,
        stream=out_all,
    )
    assert rc == 0
    assert "libero_sft_openpi" in out_all.getvalue()
    assert "featured recipes" not in out_all.getvalue()

    recipes = discover_recipes(REPO_ROOT)
    libero_evals = filter_recipes(recipes, mode=RecipeMode.EVAL, env="libero")
    assert libero_evals
    assert all(r.mode == RecipeMode.EVAL for r in libero_evals)


@pytest.mark.parametrize(
    ("mode", "target", "expected_family", "expected_script_name"),
    [
        (
            RecipeMode.TRAIN,
            "maniskill_ppo_mlp",
            RecipeFamily.EMBODIED,
            "train_embodied_agent.py",
        ),
        (
            RecipeMode.TRAIN,
            "maniskill_sac_mlp_async",
            RecipeFamily.EMBODIED,
            "train_async.py",
        ),
        (
            RecipeMode.TRAIN,
            "libero_sft_openpi",
            RecipeFamily.SFT,
            "train_vla_sft.py",
        ),
        (
            RecipeMode.TRAIN,
            "qwen3_vl_sft_vlm",
            RecipeFamily.SFT,
            "train_vlm_sft.py",
        ),
        (
            RecipeMode.TRAIN,
            "d4rl_iql_mujoco",
            RecipeFamily.OFFLINE_RL,
            "train_offline_rl.py",
        ),
        (
            RecipeMode.TRAIN,
            "cfg_rl_openpi",
            RecipeFamily.OFFLINE_RL,
            "train_cfg.py",
        ),
        (
            RecipeMode.TRAIN,
            "recap_value_model_sft",
            RecipeFamily.OFFLINE_RL,
            "train_value.py",
        ),
        (
            RecipeMode.TRAIN,
            "steam_value_model_sft",
            RecipeFamily.OFFLINE_RL,
            "train_steam.py",
        ),
        (
            RecipeMode.TRAIN,
            "sd3_nft_ocr",
            RecipeFamily.DIFFUSION,
            "train_embodied_agent.py",
        ),
        (
            RecipeMode.TRAIN,
            "qwen2.5-1.5b-grpo-megatron",
            RecipeFamily.REASONING,
            "main_grpo.py",
        ),
        (
            RecipeMode.TRAIN,
            "searchr1/train_qwen2.5",
            RecipeFamily.AGENT,
            "train.py",
        ),
        (
            RecipeMode.EVAL,
            "libero_spatial_openpi_pi05_eval",
            RecipeFamily.EMBODIED,
            "eval_embodied_agent.py",
        ),
        (
            RecipeMode.EVAL,
            "searchr1/eval_qwen2.5",
            RecipeFamily.AGENT,
            "eval.py",
        ),
        (
            RecipeMode.EVAL,
            "realworld_teleop",
            RecipeFamily.REWARD,
            "eval_realworld_teleop.py",
        ),
        (
            RecipeMode.COLLECT,
            "realworld_collect_data",
            RecipeFamily.EMBODIED,
            "collect_real_data.py",
        ),
        (
            RecipeMode.COLLECT,
            "realworld_collect_dataset",
            RecipeFamily.REWARD,
            "realworld_collect_process_dataset.py",
        ),
    ],
)
def test_resolve_recipe_across_families(
    mode, target, expected_family, expected_script_name
):
    spec = resolve_recipe(target, mode, REPO_ROOT)
    assert spec.family == expected_family
    assert spec.mode == mode
    assert spec.entry_script.name == expected_script_name


@pytest.mark.parametrize(
    ("mode", "rel_yaml", "family", "subgroup", "use_async", "expected_script_name"),
    [
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/embodied/maniskill_sac_mlp.yaml",
            None,
            None,
            False,
            "train_embodied_agent.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/embodied/maniskill_sac_mlp_async.yaml",
            None,
            None,
            True,
            "train_async.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/embodied/d4rl_iql_mujoco.yaml",
            RecipeFamily.OFFLINE_RL,
            None,
            False,
            "train_offline_rl.py",
        ),
        (
            RecipeMode.EVAL,
            "tests/e2e_tests/evaluations/maniskill_sac_mlp_async_decoupled_eval.yaml",
            None,
            None,
            False,
            "eval_embodied_agent.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/sft/maniskill_sft_openpi.yaml",
            RecipeFamily.SFT,
            None,
            False,
            "train_vla_sft.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/sft/robo2vlm_sft_train.yaml",
            RecipeFamily.SFT,
            None,
            False,
            "train_vlm_sft.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/sft/resnet_reward_training.yaml",
            RecipeFamily.REWARD,
            None,
            False,
            "train_reward_model.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/offline_rl/recap_cfg_openpi_libero.yaml",
            RecipeFamily.OFFLINE_RL,
            None,
            False,
            "train_cfg.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/offline_rl/recap_value_model_sft_libero.yaml",
            RecipeFamily.OFFLINE_RL,
            None,
            False,
            "train_value.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/offline_rl/steam_value_model_sft_libero.yaml",
            RecipeFamily.OFFLINE_RL,
            None,
            False,
            "train_steam.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/reasoning/qwen2.5-1.5b-grpo-collocated-mg-sgl.yaml",
            RecipeFamily.REASONING,
            None,
            False,
            "main_grpo.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/agent/searchr1/qwen2.5-1.5b-megatron.yaml",
            RecipeFamily.AGENT,
            "searchr1",
            False,
            "train.py",
        ),
        (
            RecipeMode.EVAL,
            "tests/e2e_tests/agent/searchr1/qwen2.5-1.5b-eval.yaml",
            RecipeFamily.AGENT,
            "searchr1",
            False,
            "eval.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/agent/wideseek/qwen3-train.yaml",
            RecipeFamily.AGENT,
            "wideseek_r1",
            False,
            "train.py",
        ),
        (
            RecipeMode.EVAL,
            "tests/e2e_tests/agent/wideseek/qwen3-eval.yaml",
            RecipeFamily.AGENT,
            "wideseek_r1",
            False,
            "eval.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/agent/rstar2/qwen2.5-1.5b-megatron.yaml",
            RecipeFamily.AGENT,
            "rstar2",
            False,
            "main_rstar2.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/agent/coding_online_rl/qwen2.5-1.5b-ppo.yaml",
            RecipeFamily.AGENT,
            "coding_online_rl",
            False,
            "main_coding_online_rl.py",
        ),
        (
            RecipeMode.TRAIN,
            "tests/e2e_tests/agentlightning/qwen2.5-1.5b-multiturn.yaml",
            RecipeFamily.AGENT,
            "calc_x",
            False,
            "main.py",
        ),
    ],
)
def test_resolve_e2e_yaml_targets_preserve_log_path(
    mode, rel_yaml, family, subgroup, use_async, expected_script_name
):
    yaml_path = REPO_ROOT / rel_yaml
    assert yaml_path.is_file()
    spec = resolve_recipe(
        str(yaml_path),
        mode,
        REPO_ROOT,
        family=family,
        subgroup=subgroup,
        use_async=use_async,
    )
    assert spec.entry_script.name == expected_script_name
    assert spec.inject_log_path is False

    plan = build_launch_plan(spec, REPO_ROOT, base_env={})
    # E2E configs own their runner.logger.log_path unless --log-dir is passed.
    assert not any(a.startswith("runner.logger.log_path=") for a in plan.command)


def test_resolve_async_override_and_eval_embodied_fallback():
    async_spec = resolve_recipe(
        "maniskill_ppo_mlp",
        RecipeMode.TRAIN,
        REPO_ROOT,
        use_async=True,
    )
    assert async_spec.entry_script.name == "train_async.py"

    eval_spec = resolve_recipe("maniskill_ppo_mlp", RecipeMode.EVAL, REPO_ROOT)
    assert eval_spec.mode == RecipeMode.EVAL
    assert eval_spec.entry_script.name == "eval_embodied_agent.py"
    assert eval_spec.config_dir == REPO_ROOT / "examples/embodiment/config"


def test_resolve_external_yaml_adds_hydra_config_dir(tmp_path):
    custom_yaml = tmp_path / "my_custom_exp.yaml"
    custom_yaml.write_text(
        "defaults:\n  - env/maniskill_pick_cube@env.train\nrunner:\n  max_epochs: 5\n",
        encoding="utf-8",
    )

    spec = resolve_recipe(str(custom_yaml), RecipeMode.TRAIN, REPO_ROOT)
    assert spec.config_file == custom_yaml.resolve()
    assert spec.config_dir == tmp_path.resolve()
    assert spec.name == "my_custom_exp"
    assert (
        spec.entry_script == REPO_ROOT / "examples/embodiment/train_embodied_agent.py"
    )
    assert (REPO_ROOT / "examples/embodiment/config") in spec.searchpath

    plan = build_launch_plan(
        spec,
        REPO_ROOT,
        overrides=["runner.max_epochs=2", "+extra.flag=true"],
        log_dir=tmp_path / "logs",
        base_env={"MUJOCO_GL": "osmesa"},
    )
    cfg_dir_idx = plan.command.index("--config-dir")
    assert plan.command[cfg_dir_idx + 1] == str(
        REPO_ROOT / "examples/embodiment/config"
    )
    assert f"runner.logger.log_path={(tmp_path / 'logs').resolve()}" in plan.command
    assert plan.command[-2:] == ["runner.max_epochs=2", "+extra.flag=true"]
    assert plan.env["MUJOCO_GL"] == "osmesa"
    assert plan.env["REPO_PATH"] == str(REPO_ROOT)


def test_resource_and_multinode_env_vars_are_preserved():
    spec = resolve_recipe("libero_spatial_openpi_pi05_eval", RecipeMode.EVAL, REPO_ROOT)
    caller_env = {
        "RLINF_NODE_RANK": "1",
        "RLINF_COMM_NET_DEVICES": "eth0",
        "CUDA_VISIBLE_DEVICES": "2,3",
        "MUJOCO_GL": "osmesa",
        "LIBERO_TYPE": "pro",
        "OMNIGIBSON_DATA_PATH": "/data/og",
        "ISAAC_PATH": "/opt/isaac",
        "RLINF_ROBOT_MOCKS": "1",
        "PYTHONPATH": "/custom/mocks:/other/path",
    }
    plan = build_launch_plan(spec, REPO_ROOT, base_env=caller_env)
    assert plan.env["RLINF_NODE_RANK"] == "1"
    assert plan.env["RLINF_COMM_NET_DEVICES"] == "eth0"
    assert plan.env["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert plan.env["MUJOCO_GL"] == "osmesa"
    assert plan.env["LIBERO_TYPE"] == "pro"
    assert plan.env["LIBERO_PERTURBATION"] == "all"
    assert plan.env["OMNIGIBSON_DATASET_PATH"] == "/data/og/behavior-1k-assets/"
    assert plan.env["EXP_PATH"] == "/opt/isaac/apps"
    assert plan.env["RLINF_ROBOT_MOCKS"] == "1"
    # Existing PYTHONPATH entries are preserved in order after repo root.
    py_parts = plan.env["PYTHONPATH"].split(os.pathsep)
    assert py_parts[0] == str(REPO_ROOT)
    assert py_parts.index("/custom/mocks") < py_parts.index("/other/path")


@pytest.mark.parametrize(
    ("mode", "target"),
    [
        (RecipeMode.TRAIN, "maniskill_ppo_mlp"),
        (RecipeMode.EVAL, "libero_spatial_openpi_pi05_eval"),
        (RecipeMode.TRAIN, "libero_sft_openpi"),
        (RecipeMode.TRAIN, "sd3_nft_ocr"),
        (RecipeMode.TRAIN, "reward_training"),
        (RecipeMode.EVAL, "realworld_teleop"),
        (RecipeMode.COLLECT, "realworld_collect_dataset"),
        (RecipeMode.TRAIN, "d4rl_iql_mujoco"),
        (RecipeMode.TRAIN, "qwen2.5-1.5b-grpo-megatron"),
    ],
)
def test_hydra_compose_resolves_repo_env_interpolations(mode, target, monkeypatch):
    spec = resolve_recipe(target, mode, REPO_ROOT)
    plan = build_launch_plan(spec, REPO_ROOT, base_env={})

    for key in (
        "REPO_PATH",
        "EMBODIED_PATH",
        "EVALUATIONS_PATH",
        "DIFFUSION_PATH",
        "REWARD_PATH",
        "VLM_PATH",
    ):
        if key in plan.env:
            monkeypatch.setenv(key, plan.env[key])
        else:
            monkeypatch.delenv(key, raising=False)

    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base="1.1", config_dir=str(spec.config_dir)):
        cfg = compose(config_name=spec.name)
    assert OmegaConf.to_container(cfg, resolve=False) is not None
    GlobalHydra.instance().clear()


def test_build_launch_plan_preserves_explicit_log_path_override():
    spec = resolve_recipe("maniskill_ppo_mlp", RecipeMode.TRAIN, REPO_ROOT)
    plan = build_launch_plan(
        spec,
        REPO_ROOT,
        overrides=["runner.logger.log_path=/tmp/custom_logs"],
        base_env={},
    )
    log_path_args = [
        arg for arg in plan.command if arg.startswith("runner.logger.log_path=")
    ]
    assert log_path_args == ["runner.logger.log_path=/tmp/custom_logs"]
    assert plan.env["MUJOCO_GL"] == "egl"
    assert plan.env["PYOPENGL_PLATFORM"] == "egl"


def test_cli_dry_run_forwards_hydra_overrides():
    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        [
            "train",
            "maniskill_ppo_mlp",
            "--dry-run",
            "runner.max_epochs=10",
            "+custom.key=1",
            "~unused",
        ],
        repo_root=REPO_ROOT,
        stream=out,
        err_stream=err,
    )
    assert rc == 0
    assert err.getvalue() == ""
    output = out.getvalue()
    assert "train_embodied_agent.py" in output
    assert "--config-name maniskill_ppo_mlp" in output
    assert "runner.max_epochs=10" in output
    assert "+custom.key=1" in output
    assert "~unused" in output


def test_cli_unknown_recipe_reports_suggestions():
    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        ["train", "maniskill_ppo_mlpp"],
        repo_root=REPO_ROOT,
        stream=out,
        err_stream=err,
    )
    assert rc == 2
    assert "Did you mean:" in err.getvalue()
    assert "maniskill_ppo_mlp" in err.getvalue()


def test_execute_launch_plan_streams_and_propagates_exit_code(tmp_path):
    spec = resolve_recipe("maniskill_ppo_mlp", RecipeMode.TRAIN, REPO_ROOT)
    log_dir = tmp_path / "run_log"
    plan = build_launch_plan(spec, REPO_ROOT, log_dir=log_dir, base_env={})

    ok_plan = LaunchPlan(
        recipe=plan.recipe,
        command=[sys.executable, "-c", "import sys; sys.stdout.write('hello-cli\\n')"],
        env=plan.env,
        log_dir=plan.log_dir,
        log_file=plan.log_file,
    )
    out = io.StringIO()
    assert execute_launch_plan(ok_plan, dry_run=False, stream=out) == 0
    assert "hello-cli" in out.getvalue()
    assert plan.log_file is not None and plan.log_file.is_file()
    assert "hello-cli" in plan.log_file.read_text(encoding="utf-8")

    # Non-zero exit codes must never be masked by log streaming.
    fail_plan = LaunchPlan(
        recipe=plan.recipe,
        command=[sys.executable, "-c", "import sys; sys.exit(7)"],
        env=plan.env,
        log_dir=plan.log_dir,
        log_file=plan.log_file,
    )
    assert execute_launch_plan(fail_plan, dry_run=False, stream=io.StringIO()) == 7


def test_execute_launch_plan_forwards_sigterm_to_child(tmp_path):
    marker = tmp_path / "child_sigterm_handled"
    ready = tmp_path / "child_ready"
    child_code = (
        "import signal, sys, time\n"
        f"marker, ready = {str(marker)!r}, {str(ready)!r}\n"
        "def _h(signum, _):\n"
        "    open(marker, 'w').write(str(signum))\n"
        "    sys.exit(42)\n"
        "signal.signal(signal.SIGTERM, _h)\n"
        "open(ready, 'w').write('ok')\n"
        "time.sleep(30)\n"
    )
    runner_code = (
        "import sys\n"
        "from pathlib import Path\n"
        "from rlinf.cli.catalog import RecipeMode, resolve_recipe\n"
        "from rlinf.cli.launcher import LaunchPlan, build_launch_plan, execute_launch_plan\n"
        f"repo = Path({str(REPO_ROOT)!r})\n"
        "spec = resolve_recipe('maniskill_ppo_mlp', RecipeMode.TRAIN, repo)\n"
        "base = build_launch_plan(spec, repo, base_env={})\n"
        f"plan = LaunchPlan(base.recipe, [sys.executable, '-c', {child_code!r}], base.env, None, None)\n"
        "sys.exit(execute_launch_plan(plan))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.Popen([sys.executable, "-c", runner_code], env=env)
    deadline = time.monotonic() + 10.0
    while not ready.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert ready.is_file(), "Child process did not become ready in time"

    proc.send_signal(signal.SIGTERM)
    rc = proc.wait(timeout=10)
    assert rc == 42
    assert marker.read_text(encoding="utf-8") == str(int(signal.SIGTERM))


def test_e2e_shell_wrappers_invoke_rlinf_cli(tmp_path):
    # Provide an executable `rlinf` shim on PATH backed by the repo's CLI module
    # so we test the real e2e shell scripts end-to-end with `--dry-run`.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    rlinf_shim = bin_dir / "rlinf"
    rlinf_shim.write_text(
        f'#!/bin/sh\nexec {sys.executable} -m rlinf.cli "$@"\n',
        encoding="utf-8",
    )
    rlinf_shim.chmod(rlinf_shim.stat().st_mode | stat.S_IEXEC)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    env["REPO_PATH"] = str(REPO_ROOT)

    cases = [
        (
            [
                "bash",
                "tests/e2e_tests/embodied/run.sh",
                "maniskill_sac_mlp",
                "--dry-run",
                "runner.max_epochs=1",
            ],
            "train_embodied_agent.py",
            "maniskill_sac_mlp",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/embodied/run.sh",
                "realworld_dual_franka_mock_sac_cnn",
                "--dry-run",
            ],
            "RLINF_ROBOT_MOCKS=1",
            "realworld_dual_franka_mock_sac_cnn",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/embodied/run_async.sh",
                "maniskill_sac_mlp_async",
                "egl",
                "--dry-run",
            ],
            "train_async.py",
            "maniskill_sac_mlp_async",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/embodied/run_offline.sh",
                "d4rl_iql_mujoco",
                "egl",
                "--dry-run",
            ],
            "train_offline_rl.py",
            "d4rl_iql_mujoco",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/evaluations/run.sh",
                "maniskill_sac_mlp_async_decoupled_eval",
                "egl",
                "--dry-run",
            ],
            "eval_embodied_agent.py",
            "maniskill_sac_mlp_async_decoupled_eval",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/sft/run_vla_sft.sh",
                "maniskill_sft_openpi",
                "egl",
                "--dry-run",
            ],
            "train_vla_sft.py",
            "maniskill_sft_openpi",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/sft/run_vlm_sft.sh",
                "robo2vlm_sft_train",
                "egl",
                "--dry-run",
            ],
            "train_vlm_sft.py",
            "robo2vlm_sft_train",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/sft/run_reward_training.sh",
                "resnet_reward_training",
                "--dry-run",
            ],
            "train_reward_model.py",
            "resnet_reward_training",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/offline_rl/run_cfg_rl.sh",
                "recap_cfg_openpi_libero",
                "egl",
                "--dry-run",
            ],
            "train_cfg.py",
            "recap_cfg_openpi_libero",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/reasoning/run.sh",
                "qwen2.5-1.5b-ppo-collocated-mg-sgl.yaml",
                "--dry-run",
            ],
            "main_grpo.py",
            "qwen2.5-1.5b-ppo-collocated-mg-sgl",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agent/searchr1/run-train.sh",
                "qwen2.5-1.5b-megatron",
                "--dry-run",
            ],
            "examples/agent/searchr1/train.py",
            "qwen2.5-1.5b-megatron",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agent/searchr1/run-eval.sh",
                "qwen2.5-1.5b-eval",
                "--dry-run",
            ],
            "examples/agent/searchr1/eval.py",
            "qwen2.5-1.5b-eval",
        ),
    ]

    for cmd, expected_token, expected_cfg in cases:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, f"{cmd} failed:\n{proc.stderr}"
        assert expected_token in proc.stdout, (
            f"Expected {expected_token!r} in output of {cmd}:\n{proc.stdout}"
        )
        assert f"--config-name {expected_cfg}" in proc.stdout


def test_doctor_runs_cleanly_and_surfaces_resource_env(monkeypatch):
    monkeypatch.setenv("RLINF_NODE_RANK", "2")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    results = run_doctor(repo_root=REPO_ROOT, check_network=False)
    names = [r.name for r in results]
    assert names == [
        "Python",
        "Repository",
        "Accelerators",
        "Rendering",
        "Ray",
        "Hugging Face",
    ]
    by_name = {r.name: r for r in results}
    assert by_name["Python"].status == CheckStatus.OK
    assert by_name["Repository"].status == CheckStatus.OK
    assert "CUDA_VISIBLE_DEVICES=0,1" in by_name["Accelerators"].detail
    assert "RLINF_NODE_RANK=2" in by_name["Ray"].detail

    out = io.StringIO()
    rc = main(["doctor"], repo_root=REPO_ROOT, stream=out)
    assert rc in (0, 1)
    assert "[OK  ] Python" in out.getvalue()
    assert "[OK  ] Repository" in out.getvalue()
