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

from pathlib import Path

from toolkits.clean_intermediate_checkpoints import (
    build_cleanup_plans,
    execute_cleanup,
    main,
    plan_checkpoint_cleanup,
)


def _make_step(checkpoints_dir: Path, step: int) -> Path:
    step_dir = checkpoints_dir / f"global_step_{step}"
    (step_dir / "actor" / "dcp_checkpoint").mkdir(parents=True)
    (step_dir / "actor" / "dcp_checkpoint" / ".metadata").write_text(
        "training state", encoding="utf-8"
    )
    (step_dir / "actor" / "model_state_dict").mkdir()
    (step_dir / "actor" / "model_state_dict" / "full_weights.pt").write_text(
        "checkpoint", encoding="utf-8"
    )
    return step_dir


def test_plan_checkpoint_cleanup_orders_steps_numerically(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "run" / "checkpoints"
    for step in (100, 2, 30, 1):
        _make_step(checkpoints_dir, step)
    (checkpoints_dir / "best_model").mkdir()
    (checkpoints_dir / "global_step_200.7z").write_text("archive", encoding="utf-8")

    plan = plan_checkpoint_cleanup(checkpoints_dir, keep_last=2)

    assert [path.name for path in plan.keep] == ["global_step_30", "global_step_100"]
    assert [path.name for path in plan.remove] == ["global_step_1", "global_step_2"]


def test_build_cleanup_plans_scans_each_run(tmp_path: Path) -> None:
    run_a = tmp_path / "run_a" / "checkpoints"
    run_b = tmp_path / "nested" / "run_b" / "checkpoints"
    for step in (1, 2, 3):
        _make_step(run_a, step)
    for step in (10, 20):
        _make_step(run_b, step)

    plans = build_cleanup_plans([tmp_path], keep_last=1)

    assert len(plans) == 2
    assert [[path.name for path in plan.keep] for plan in plans] == [
        ["global_step_20"],
        ["global_step_3"],
    ]


def test_execute_cleanup_removes_whole_old_steps(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    old_step = _make_step(checkpoints_dir, 10)
    middle_step = _make_step(checkpoints_dir, 20)
    latest_step = _make_step(checkpoints_dir, 30)
    archive = checkpoints_dir / "global_step_10.7z"
    archive.write_text("keep unrelated files", encoding="utf-8")
    plan = plan_checkpoint_cleanup(checkpoints_dir, keep_last=2)

    removed = execute_cleanup([plan])

    assert removed == [old_step]
    assert not old_step.exists()
    assert middle_step.is_dir()
    assert latest_step.is_dir()
    assert archive.is_file()


def test_main_is_dry_run_by_default(tmp_path: Path, capsys) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    old_step = _make_step(checkpoints_dir, 1)
    _make_step(checkpoints_dir, 2)

    return_code = main([str(checkpoints_dir), "--keep-last", "1"])

    assert return_code == 0
    assert old_step.is_dir()
    assert "Dry run only" in capsys.readouterr().out


def test_main_execute_deletes_old_steps(tmp_path: Path) -> None:
    checkpoints_dir = tmp_path / "checkpoints"
    old_step = _make_step(checkpoints_dir, 1)
    latest_step = _make_step(checkpoints_dir, 2)

    return_code = main([str(checkpoints_dir), "--keep-last", "1", "--execute"])

    assert return_code == 0
    assert not old_step.exists()
    assert latest_step.is_dir()
