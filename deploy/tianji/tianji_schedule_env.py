"""Print the shell exports that fix one run's chunk timetable and subtasks.

Sourced by tianji_local.env, so Stage1, every Stage2 Ray worker and the
config checks see the same model horizon, control rate, A1 prefetch schedule
(RLT_REF_OFFSET is read at import by RLinf) and subtask count.
"""

import importlib.util
import os
import shlex
import sys
from pathlib import Path

RLINF_ROOT = Path(os.environ.get("RLINF_ROOT", Path(__file__).resolve().parents[2]))
TIANJI = RLINF_ROOT / "rlinf/envs/realworld/tianji"


def load(name):
    # Plain modules (numpy/json only): no package import side effects.
    spec = importlib.util.spec_from_file_location(name, TIANJI / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    contract = load("tianji_contract")
    subtasks = load("tianji_subtasks")
    model = contract.contract_from_env()
    budget = float(
        os.environ.get("TIANJI_STAGE2_INFERENCE_BUDGET_S", contract.DEFAULT_INFERENCE_BUDGET_S)
    )
    schedule = contract.prefetch_schedule(
        model.action_chunk_length, model.control_hz, budget
    )
    prefetch_at, ref_offset = schedule.prefetch_at, schedule.ref_offset
    override = os.environ.get("TIANJI_STAGE2_PREFETCH_AT_OVERRIDE", "").strip()
    if override:
        prefetch_at = int(override)
        ref_offset = schedule.chunk - 1 - prefetch_at
        if not schedule.prefetch_at <= prefetch_at <= schedule.chunk - 2:
            raise SystemExit(
                f"TIANJI_STAGE2_PREFETCH_AT_OVERRIDE={prefetch_at} must lie in "
                f"[{schedule.prefetch_at}, {schedule.chunk - 2}] for chunk {schedule.chunk}"
            )
    chunk = schedule.chunk
    row_scale = contract.row_scale_from_env()
    if row_scale != 1.0:
        if override or os.environ.get("TIANJI_STAGE_TIME_SCALE", "").strip():
            raise SystemExit(
                "TIANJI_ROW_SCALE excludes TIANJI_STAGE2_PREFETCH_AT_OVERRIDE and TIANJI_STAGE_TIME_SCALE"
            )
        if os.environ.get("TIANJI_DEPLOY_ACTOR_UPDATES", "0") == "1" and not contract.dense_rows_from_env():
            raise SystemExit(
                "scaled residual deployment requires TIANJI_DENSE_ROWS=1 and executor_action_grid=true"
            )
        chunk, prefetch_at, ref_offset, _ = contract.resampled_schedule(
            model.action_chunk_length, model.control_hz, budget, row_scale
        )
    prompts_file = os.environ.get("TIANJI_SUBTASK_PROMPTS", "")
    heads = 1
    prompt_count = 0
    if prompts_file:
        prompts_file = str(Path(prompts_file).expanduser().resolve())
        car = os.environ.get("TIANJI_CAR_1118") == "1"
        car_subtasks = int(os.environ.get("TIANJI_CAR_NUM_SUBTASKS", "15"))
        limit = car_subtasks if car else subtasks.MAX_SUBTASKS
        prompt_count = len(subtasks.load_subtask_prompts(prompts_file, limit=limit))
        if car and prompt_count != car_subtasks:
            raise ValueError(f"1118v1 requires all {car_subtasks} car prompts")
        if not car:
            heads = prompt_count
    values = {
        "TIANJI_ACTION_HORIZON": model.action_chunk_length,
        "TIANJI_CONTROL_HZ": f"{model.control_hz:g}",
        "TIANJI_SUBTASK_PROMPTS": prompts_file,
        "TIANJI_STAGE1_PREFETCH_STEP": schedule.request_after_steps,
        "TIANJI_STAGE2_CHUNK": chunk,
        "TIANJI_STAGE2_PREFETCH_AT": prefetch_at,
        "RLT_REF_OFFSET": ref_offset,
        # Ten RL chunks per rollout epoch; deployment epochs are unbounded.
        "TIANJI_STAGE2_EPOCH_STEPS": 10 * chunk,
        "TIANJI_STAGE2_INFER_STEPS": 10000 * chunk,
        "TIANJI_STAGE2_HEADS": heads,
        "RLT_NUM_STAGES": subtasks.MAX_SUBTASKS,
    }
    print(" ".join(f"export {k}={shlex.quote(str(v))};" for k, v in values.items()))
    print(
        f"# TIANJI_SCHEDULE horizon={model.action_chunk_length} hz={model.control_hz:g} "
        f"stage1_request_after={schedule.request_after_steps} stage2_chunk={chunk} "
        f"row_scale={row_scale:g} "
        f"prefetch_at={prefetch_at} ref_offset={ref_offset}{' (override)' if override else ''} "
        f"budget={schedule.inference_budget_s:.3f}s subtasks={prompt_count} heads={heads}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
