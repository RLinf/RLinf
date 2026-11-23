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
import json
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
import venv
from pathlib import Path

import pytest
import yaml
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
from rlinf.cli.installer import InstallPlan, execute_install_plan, load_install_options
from rlinf.cli.launcher import (
    LaunchPlan,
    build_launch_plan,
    execute_launch_plan,
    resolve_venv_python,
)
from rlinf.cli.shell import (
    ACTIVATE_FILE_ENV,
    HOOK_END,
    HOOK_START,
    SUPPORTED_SHELLS,
    handle_shell_init,
    handle_use,
    install_shell_hook,
    render_shell_hook,
    uninstall_shell_hook,
)

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
                "tests/e2e_tests/embodied/run_eval.sh",
                "maniskill_sac_mlp",
                "egl",
                "--dry-run",
            ],
            "eval_embodied_agent.py",
            "maniskill_sac_mlp",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/offline_rl/run_recap_value_model_sft.sh",
                "recap_value_model_sft_libero",
                "egl",
                "--dry-run",
            ],
            "train_value.py",
            "recap_value_model_sft_libero",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/offline_rl/run_steam_value_model_sft.sh",
                "steam_value_model_sft_libero",
                "egl",
                "--dry-run",
            ],
            "train_steam.py",
            "steam_value_model_sft_libero",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/dynamic_scheduler/run.sh",
                "qwen2.5-1.5b-grpo-dynamic-mg-sgl",
                "--dry-run",
            ],
            "main_grpo.py",
            "qwen2.5-1.5b-grpo-dynamic-mg-sgl",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agent/coding_online_rl/run.sh",
                "--dry-run",
            ],
            "main_coding_online_rl.py",
            "qwen2.5-1.5b-ppo",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agent/rstar2/run.sh",
                "--dry-run",
            ],
            "main_rstar2.py",
            "qwen2.5-1.5b-megatron",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agent/wideseek/run-train.sh",
                "--dry-run",
            ],
            "examples/agent/wideseek_r1/train.py",
            "qwen3-train",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agent/wideseek/run-eval.sh",
                "--dry-run",
            ],
            "examples/agent/wideseek_r1/eval.py",
            "qwen3-eval",
        ),
        (
            [
                "bash",
                "tests/e2e_tests/agentlightning/run.sh",
                "--dry-run",
            ],
            "examples/agent/agentlightning/calc_x/main.py",
            "qwen2.5-1.5b-multiturn",
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

    out_json = io.StringIO()
    rc_json = main(["doctor", "--json"], repo_root=REPO_ROOT, stream=out_json)
    assert rc_json in (0, 1)
    parsed = json.loads(out_json.getvalue())
    assert [entry["name"] for entry in parsed] == names


def test_cli_no_args_and_version(capsys):
    out = io.StringIO()
    assert main([], repo_root=REPO_ROOT, stream=out) == 0
    assert "usage: rlinf" in out.getvalue()

    with pytest.raises(SystemExit) as exc_info:
        main(["--version"], repo_root=REPO_ROOT)
    assert exc_info.value.code == 0
    assert "rlinf " in capsys.readouterr().out


def test_list_verbose_json_and_multi_token_search():
    out_v = io.StringIO()
    assert (
        main(
            ["list", "--search", "libero openpi", "--verbose"],
            repo_root=REPO_ROOT,
            stream=out_v,
        )
        == 0
    )
    text_v = out_v.getvalue()
    assert "SCRIPT" in text_v and "CONFIG" in text_v
    assert "libero_spatial_ppo_openpi_pi05" in text_v

    out_j = io.StringIO()
    assert (
        main(
            ["list", "--family", "sft", "--model", "openpi", "--json"],
            repo_root=REPO_ROOT,
            stream=out_j,
        )
        == 0
    )
    items = json.loads(out_j.getvalue())
    assert items and any(item["name"] == "libero_sft_openpi" for item in items)


def test_show_summary_raw_json_and_env_var_status(monkeypatch):
    monkeypatch.delenv("POLARIS_DATA_PATH", raising=False)
    out = io.StringIO()
    assert (
        main(
            ["show", "polaris_tapeintocontainer_ppo_openpi"],
            repo_root=REPO_ROOT,
            stream=out,
        )
        == 0
    )
    summary = out.getvalue()
    assert "Recipe:    polaris_tapeintocontainer_ppo_openpi (embodied/train)" in summary
    assert "POLARIS_DATA_PATH [UNSET]" in summary
    assert "Run:       rlinf train polaris_tapeintocontainer_ppo_openpi" in summary

    monkeypatch.setenv("POLARIS_DATA_PATH", "/data/polaris")
    out_set = io.StringIO()
    assert (
        main(
            ["show", "polaris_tapeintocontainer_ppo_openpi", "--json"],
            repo_root=REPO_ROOT,
            stream=out_set,
        )
        == 0
    )
    info = json.loads(out_set.getvalue())
    assert info["family"] == "embodied"
    assert "POLARIS_DATA_PATH" in info["required_env_vars"]

    out_raw = io.StringIO()
    assert (
        main(
            ["show", "maniskill_ppo_mlp", "--raw"], repo_root=REPO_ROOT, stream=out_raw
        )
        == 0
    )
    assert "maniskill_pick_cube" in out_raw.getvalue()


def test_wrong_mode_and_typo_flag_and_double_dash_passthrough():
    err_mode = io.StringIO()
    rc = main(
        ["train", "libero_spatial_openpi_pi05_eval"],
        repo_root=REPO_ROOT,
        stream=io.StringIO(),
        err_stream=err_mode,
    )
    assert rc == 2
    assert "rlinf eval libero/libero_spatial_openpi_pi05_eval" in err_mode.getvalue()

    err_flag = io.StringIO()
    rc_flag = main(
        ["train", "maniskill_ppo_mlp", "--dr-run"],
        repo_root=REPO_ROOT,
        stream=io.StringIO(),
        err_stream=err_flag,
    )
    assert rc_flag == 2
    assert "Did you mean '--dry-run'?" in err_flag.getvalue()

    out_pass = io.StringIO()
    rc_pass = main(
        ["train", "maniskill_ppo_mlp", "--dry-run", "--", "--custom-raw-flag"],
        repo_root=REPO_ROOT,
        stream=out_pass,
        err_stream=io.StringIO(),
    )
    assert rc_pass == 0
    assert "--custom-raw-flag" in out_pass.getvalue()


def test_check_flag_validates_hydra_overrides_and_rejects_typos():
    out_ok = io.StringIO()
    err_ok = io.StringIO()
    rc_ok = main(
        ["train", "maniskill_ppo_mlp", "--check", "runner.max_epochs=2"],
        repo_root=REPO_ROOT,
        stream=out_ok,
        err_stream=err_ok,
    )
    assert rc_ok == 0, err_ok.getvalue()
    assert "Config OK: embodied/train (maniskill_ppo_mlp)" in out_ok.getvalue()

    out_bad = io.StringIO()
    err_bad = io.StringIO()
    rc_bad = main(
        ["train", "maniskill_ppo_mlp", "--check", "runer.max_epochs=2"],
        repo_root=REPO_ROOT,
        stream=out_bad,
        err_stream=err_bad,
    )
    assert rc_bad != 0
    assert "runer" in err_bad.getvalue()


def test_preflight_blocks_missing_searchpath_env_and_warns_on_body_env(tmp_path):
    bad_sp = tmp_path / "missing_sp.yaml"
    bad_sp.write_text(
        "hydra:\n  searchpath:\n    - file://${oc.env:RLINF_TEST_MISSING_SP}/config\n"
        "runner:\n  task_type: embodied\n",
        encoding="utf-8",
    )
    err = io.StringIO()
    rc = main(
        ["train", str(bad_sp), "--check"],
        repo_root=REPO_ROOT,
        stream=io.StringIO(),
        err_stream=err,
    )
    assert rc == 2
    assert "RLINF_TEST_MISSING_SP" in err.getvalue()

    body_env = tmp_path / "missing_body.yaml"
    body_env.write_text(
        "runner:\n  task_type: embodied\nmodel_path: ${oc.env:RLINF_TEST_MISSING_BODY}\n",
        encoding="utf-8",
    )
    out_dry = io.StringIO()
    assert (
        main(
            ["train", str(body_env), "--dry-run"],
            repo_root=REPO_ROOT,
            stream=out_dry,
            err_stream=io.StringIO(),
        )
        == 0
    )
    assert "RLINF_TEST_MISSING_BODY" in out_dry.getvalue()


def test_cli_runs_from_outside_repo_and_does_not_import_torch_or_ray(tmp_path):
    env = dict(os.environ)
    env["RLINF_REPO_PATH"] = str(REPO_ROOT)
    env["PYTHONPATH"] = str(REPO_ROOT)
    code = (
        "import io, sys\n"
        "from rlinf.cli import main\n"
        "out = io.StringIO()\n"
        "rc = main(['list'], stream=out)\n"
        "assert rc == 0 and 'maniskill_ppo_mlp' in out.getvalue()\n"
        "assert 'torch' not in sys.modules, 'torch imported eagerly by rlinf list'\n"
        "assert 'ray' not in sys.modules, 'ray imported eagerly by rlinf list'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr


def test_install_list_parses_real_install_sh():
    options = load_install_options(REPO_ROOT)
    assert set(options.targets) == {"embodied", "agentic", "docs"}
    assert "openvla" in options.models and "openpi" in options.models
    assert "maniskill_libero" in options.envs and "libero" in options.envs
    assert {"sglang", "vllm", "phyai"}.issubset(set(options.engines))
    assert {"nvidia", "amd", "ascend", "musa"}.issubset(set(options.platforms))

    out = io.StringIO()
    assert main(["install", "--list"], repo_root=REPO_ROOT, stream=out) == 0
    assert "maniskill_libero" in out.getvalue()
    assert "rlinf install embodied" in out.getvalue()

    out_json = io.StringIO()
    assert (
        main(["install", "--list", "--json"], repo_root=REPO_ROOT, stream=out_json) == 0
    )
    data = json.loads(out_json.getvalue())
    assert "openvla" in data["models"]
    assert "maniskill_libero" in data["envs"]


def test_install_dry_run_and_validation_hints():
    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        [
            "install",
            "embodied",
            "--model",
            "openvla",
            "--env",
            "maniskill_libero",
            "--venv",
            ".venv-vla",
            "--use-mirror",
            "--no-flash-attn",
            "--dry-run",
        ],
        repo_root=REPO_ROOT,
        stream=out,
        err_stream=err,
    )
    assert rc == 0, err.getvalue()
    text = out.getvalue()
    assert (
        "requirements/install.sh embodied --model openvla --env maniskill_libero"
        in text
    )
    assert "--venv .venv-vla" in text
    assert "--use-mirror --no-flash-attn" in text

    out_agent = io.StringIO()
    assert (
        main(
            ["install", "agentic", "--engine", "sglang", "--dry-run"],
            repo_root=REPO_ROOT,
            stream=out_agent,
        )
        == 0
    )
    assert "requirements/install.sh agentic --engine sglang" in out_agent.getvalue()

    err_model = io.StringIO()
    assert (
        main(
            ["install", "embodied", "--model", "opnevla", "--dry-run"],
            repo_root=REPO_ROOT,
            stream=io.StringIO(),
            err_stream=err_model,
        )
        == 2
    )
    assert "Did you mean: openvla" in err_model.getvalue()

    err_env = io.StringIO()
    assert (
        main(
            ["install", "embodied", "--env", "maniskill", "--dry-run"],
            repo_root=REPO_ROOT,
            stream=io.StringIO(),
            err_stream=err_env,
        )
        == 2
    )
    assert "maniskill_libero" in err_env.getvalue()

    err_combo = io.StringIO()
    assert (
        main(
            ["install", "agentic", "--model", "openvla", "--dry-run"],
            repo_root=REPO_ROOT,
            stream=io.StringIO(),
            err_stream=err_combo,
        )
        == 2
    )
    assert "apply only to target 'embodied'" in err_combo.getvalue()


def test_install_works_without_omegaconf_or_torch(tmp_path):
    # Simulates a fresh machine running system python3 before any venv exists.
    system_py = Path("/usr/bin/python3")
    interpreter = str(system_py) if system_py.is_file() else sys.executable
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [
            interpreter,
            "-m",
            "rlinf.cli",
            "install",
            "embodied",
            "--model",
            "openvla",
            "--env",
            "maniskill_libero",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert (
        "requirements/install.sh embodied --model openvla --env maniskill_libero"
        in proc.stdout
    )


def test_execute_install_plan_prints_activation_and_propagates_exit_code(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SHELL", "/bin/bash")
    monkeypatch.delenv(ACTIVATE_FILE_ENV, raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)

    ok_plan = InstallPlan(
        target="embodied",
        venv_dir=".venv-test",
        script_path=REPO_ROOT / "requirements/install.sh",
        repo_root=REPO_ROOT,
        command=[sys.executable, "-c", "pass"],
    )
    out = io.StringIO()
    assert execute_install_plan(ok_plan, stream=out) == 0
    assert "source .venv-test/bin/activate" in out.getvalue()
    assert "rlinf doctor" in out.getvalue()
    assert HOOK_START in (tmp_path / ".bashrc").read_text(encoding="utf-8")

    # When invoked through an active shell hook, writes target venv to RLINF_ACTIVATE_FILE.
    act_file = tmp_path / "activate.target"
    monkeypatch.setenv(ACTIVATE_FILE_ENV, str(act_file))
    out_hook = io.StringIO()
    assert execute_install_plan(ok_plan, stream=out_hook) == 0
    assert "Activating '.venv-test' in the current shell" in out_hook.getvalue()
    assert act_file.read_text(encoding="utf-8").strip() == str(
        (REPO_ROOT / ".venv-test").resolve()
    )

    fail_plan = InstallPlan(
        target="embodied",
        venv_dir=".venv-test",
        script_path=REPO_ROOT / "requirements/install.sh",
        repo_root=REPO_ROOT,
        command=[sys.executable, "-c", "import sys; sys.exit(3)"],
    )
    out_fail = io.StringIO()
    assert execute_install_plan(fail_plan, stream=out_fail) == 3
    assert "Install failed (exit 3)" in out_fail.getvalue()


def test_all_ci_workflow_install_commands_resolve_cleanly():
    workflows_dir = REPO_ROOT / ".github/workflows"
    install_re = re.compile(r"python3\s+-m\s+rlinf\.cli\s+install([^\n]*)")
    checked = 0
    for wf_path in sorted(workflows_dir.glob("*.yml")):
        data = yaml.safe_load(wf_path.read_text(encoding="utf-8"))
        for job_name, job_spec in (data.get("jobs") or {}).items():
            for step in job_spec.get("steps") or []:
                run_block = (step.get("run") or "").replace("\\\n", " ")
                for match in install_re.finditer(run_block):
                    argv = ["install", *shlex.split(match.group(1)), "--dry-run"]
                    out = io.StringIO()
                    err = io.StringIO()
                    rc = main(argv, repo_root=REPO_ROOT, stream=out, err_stream=err)
                    assert rc == 0, (
                        f"{wf_path.name}:{job_name} failed for {argv}: {err.getvalue()}"
                    )
                    assert "requirements/install.sh" in out.getvalue()
                    checked += 1

    assert checked == 89


def test_venv_flag_and_auto_venv_resolution(tmp_path):
    custom_venv = tmp_path / "custom-venv"
    fake_py = custom_venv / "bin" / "python"
    fake_py.parent.mkdir(parents=True)
    fake_py.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_py.chmod(0o755)

    py_path, resolved = resolve_venv_python(REPO_ROOT, venv=custom_venv)
    assert Path(py_path).resolve() == fake_py.resolve()
    assert resolved == custom_venv.resolve()

    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        ["train", "maniskill_ppo_mlp", "--venv", str(custom_venv), "--dry-run"],
        repo_root=REPO_ROOT,
        stream=out,
        err_stream=err,
    )
    assert rc == 0, err.getvalue()
    assert str(fake_py) in out.getvalue()

    err_missing = io.StringIO()
    rc_missing = main(
        [
            "train",
            "maniskill_ppo_mlp",
            "--venv",
            str(tmp_path / "missing"),
            "--dry-run",
        ],
        repo_root=REPO_ROOT,
        stream=io.StringIO(),
        err_stream=err_missing,
    )
    assert rc_missing == 2
    assert "Virtual environment not found" in err_missing.getvalue()

    # Active VIRTUAL_ENV overrides sys.executable when pointing to another venv.
    py_active, active_dir = resolve_venv_python(
        REPO_ROOT, base_env={"VIRTUAL_ENV": str(custom_venv)}
    )
    assert Path(py_active).resolve() == fake_py.resolve()
    assert active_dir == custom_venv.resolve()

    # Auto-detect <repo_root>/.venv when current interpreter lacks runtime deps.
    fake_repo = tmp_path / "repo"
    default_py = fake_repo / ".venv" / "bin" / "python"
    default_py.parent.mkdir(parents=True)
    default_py.write_text("#!/bin/sh\n", encoding="utf-8")
    default_py.chmod(0o755)
    bare_py = Path("/usr/bin/python3")
    if bare_py.is_file():
        code = (
            "from pathlib import Path\n"
            "from rlinf.cli.launcher import resolve_venv_python\n"
            f"py, venv_dir = resolve_venv_python(Path({str(fake_repo)!r}), base_env={{}})\n"
            f"assert Path(py).resolve() == Path({str(default_py)!r}).resolve(), py\n"
            f"assert venv_dir == Path({str(fake_repo / '.venv')!r}).resolve()\n"
        )
        proc = subprocess.run(
            [str(bare_py), "-c", code],
            env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT)},
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


def test_doctor_delegates_to_explicit_venv(tmp_path):
    alt_venv = tmp_path / "alt-venv"
    alt_py = alt_venv / "bin" / "python"
    alt_py.parent.mkdir(parents=True)
    alt_py.symlink_to(Path(sys.executable))

    out = io.StringIO()
    err = io.StringIO()
    rc = main(
        ["doctor", "--venv", str(alt_venv), "--json"],
        repo_root=REPO_ROOT,
        stream=out,
        err_stream=err,
    )
    assert rc in (0, 1), err.getvalue()
    checks = json.loads(out.getvalue())
    py_detail = next(c["detail"] for c in checks if c["name"] == "Python")
    assert str(alt_venv) in py_detail


def test_lightweight_global_cli_package_end_to_end(tmp_path):
    cli_pkg_dir = REPO_ROOT / "cli"
    assert (cli_pkg_dir / "pyproject.toml").is_file()

    # Build/install cli/ into an isolated zero-dependency venv and test the `rlinf` binary.
    tool_venv = tmp_path / "tool-venv"
    venv.create(tool_venv, with_pip=False, symlinks=True)
    tool_py = tool_venv / "bin" / "python"
    uv_bin = shutil.which("uv")
    if uv_bin is not None:
        install_cmd = [
            uv_bin,
            "pip",
            "install",
            "--python",
            str(tool_py),
            "--no-deps",
            "--no-build-isolation",
            str(cli_pkg_dir),
        ]
        # Give setuptools visibility for --no-build-isolation without touching network.
        env_install = dict(os.environ)
        site_paths = [p for p in sys.path if "site-packages" in p]
        if site_paths:
            env_install["PYTHONPATH"] = os.pathsep.join(site_paths)
        proc_install = subprocess.run(
            install_cmd,
            env=env_install,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc_install.returncode != 0:
            pytest.skip(f"uv pip install ./cli unavailable: {proc_install.stderr}")
    else:
        pytest.skip("uv not available to build cli/ package")

    rlinf_bin = tool_venv / "bin" / "rlinf"
    assert rlinf_bin.is_file()

    clean_env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)}
    ver = subprocess.run(
        [str(rlinf_bin), "--version"],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert ver.returncode == 0, ver.stderr
    assert ver.stdout.strip().startswith("rlinf ")

    # Outside a checkout, global rlinf reports a clear actionable error.
    outside = subprocess.run(
        [str(rlinf_bin), "list"],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert outside.returncode == 2
    assert "RLinf repository checkout" in outside.stderr

    # Inside the checkout, global rlinf delegates to rlinf/cli with zero third-party deps.
    dry = subprocess.run(
        [
            str(rlinf_bin),
            "install",
            "embodied",
            "--model",
            "openvla",
            "--env",
            "maniskill_libero",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert dry.returncode == 0, dry.stderr
    assert (
        "requirements/install.sh embodied --model openvla --env maniskill_libero"
        in dry.stdout
    )

    lst = subprocess.run(
        [str(rlinf_bin), "list", "--family", "embodied"],
        cwd=REPO_ROOT,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert lst.returncode == 0, lst.stderr
    assert "maniskill_ppo_mlp" in lst.stdout

    # shell-init works even outside any RLinf checkout.
    hook_out = subprocess.run(
        [str(rlinf_bin), "shell-init", "bash"],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert hook_out.returncode == 0, hook_out.stderr
    assert HOOK_START in hook_out.stdout

    # Real bash end-to-end test: eval shell hook + `rlinf use` switches VIRTUAL_ENV!
    target_venv = tmp_path / "target-venv"
    venv.create(target_venv, with_pip=False, symlinks=True)
    bash_script = (
        f'export PATH="{tool_venv / "bin"}:$PATH"\n'
        'eval "$(rlinf shell-init bash)"\n'
        f'rlinf use "{target_venv}"\n'
        'printf "VIRTUAL_ENV=%s\\n" "${VIRTUAL_ENV:-}"\n'
    )
    bash_proc = subprocess.run(
        ["bash", "-c", bash_script],
        cwd=tmp_path,
        env=clean_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert bash_proc.returncode == 0, bash_proc.stderr
    assert f"VIRTUAL_ENV={target_venv.resolve()}" in bash_proc.stdout


def test_shell_module_stays_synced_between_cli_and_rlinf():
    rlinf_shell = (REPO_ROOT / "rlinf/cli/shell.py").read_text(encoding="utf-8")
    cli_shell = (REPO_ROOT / "cli/rlinf_cli/shell.py").read_text(encoding="utf-8")
    assert rlinf_shell == cli_shell


def test_shell_init_install_update_and_uninstall_all_shells(tmp_path):
    for sh in SUPPORTED_SHELLS:
        snippet = render_shell_hook(sh)
        assert HOOK_START in snippet and HOOK_END in snippet

        res1 = install_shell_hook(sh, home=tmp_path, env={})
        assert res1.status == "installed"
        assert res1.rc_path is not None and res1.rc_path.is_file()
        content = res1.rc_path.read_text(encoding="utf-8")
        assert HOOK_START in content and HOOK_END in content

        # Second install is idempotent.
        res2 = install_shell_hook(sh, home=tmp_path, env={})
        assert res2.status == "already_installed"

        # Uninstall cleans up the block.
        res3 = uninstall_shell_hook(sh, home=tmp_path, env={})
        assert res3.status == "removed"
        assert HOOK_START not in res1.rc_path.read_text(encoding="utf-8")

    # handle_shell_init --install prints reload guidance when hook is not active yet.
    out = io.StringIO()
    err = io.StringIO()
    rc = handle_shell_init(
        shell="zsh",
        install=True,
        stream=out,
        err_stream=err,
        home=tmp_path,
        env={"SHELL": "/bin/zsh"},
    )
    assert rc == 0, err.getvalue()
    assert "Installed zsh shell hook" in out.getvalue()
    assert "source " in out.getvalue()


def test_use_without_hook_prints_manual_source_and_lists_venvs(tmp_path):
    fake_repo = tmp_path / "repo"
    venv_a = fake_repo / ".venv"
    venv_b = fake_repo / ".venv-openpi"
    for v in (venv_a, venv_b):
        py = v / "bin" / "python"
        py.parent.mkdir(parents=True)
        py.write_text("#!/bin/sh\n", encoding="utf-8")
        py.chmod(0o755)

    out_list = io.StringIO()
    rc_list = handle_use(
        list_venvs=True,
        repo_root=fake_repo,
        stream=out_list,
        err_stream=io.StringIO(),
        env={"VIRTUAL_ENV": str(venv_b)},
    )
    assert rc_list == 0
    assert f"* {venv_b.resolve()}" in out_list.getvalue()
    assert f"  {venv_a.resolve()}" in out_list.getvalue()

    # Without shell hook, prints clear manual source command and shell-init hint.
    out_manual = io.StringIO()
    rc_manual = handle_use(
        ".venv-openpi",
        repo_root=fake_repo,
        stream=out_manual,
        err_stream=io.StringIO(),
        env={"SHELL": "/usr/bin/fish"},
    )
    assert rc_manual == 0
    assert "Shell hook is not active" in out_manual.getvalue()
    assert f"source {venv_b.resolve()}/bin/activate.fish" in out_manual.getvalue()
    assert "rlinf shell-init --install" in out_manual.getvalue()

    # With active shell hook, writes target venv to RLINF_ACTIVATE_FILE.
    act_file = tmp_path / "act.txt"
    out_active = io.StringIO()
    rc_active = handle_use(
        ".venv-openpi",
        repo_root=fake_repo,
        stream=out_active,
        err_stream=io.StringIO(),
        env={ACTIVATE_FILE_ENV: str(act_file)},
    )
    assert rc_active == 0
    assert act_file.read_text(encoding="utf-8").strip() == str(venv_b.resolve())

    # Missing venv returns exit 2.
    err_missing = io.StringIO()
    rc_missing = handle_use(
        ".venv-does-not-exist",
        repo_root=fake_repo,
        stream=io.StringIO(),
        err_stream=err_missing,
        env={},
    )
    assert rc_missing == 2
    assert "not found" in err_missing.getvalue()
