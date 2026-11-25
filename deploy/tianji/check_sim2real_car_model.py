"""Offline GPU smoke test for 16D joint Sim2Real toy-car checkpoints.

Loads a Stage1 checkpoint directory (`model.safetensors` or `full_weights.pt`)
plus `norm_stats.json` and optional `prompts.json` (1, 4, 12, or 24 prompts, or
a single global prompt) via `tianji2_joint_stage1`, verifies `use_rlt`
auto-detection, and executes both unguided and RTC-guided 16D joint inference.
Never connects to LoopInfra or robot hardware.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import torch
from hydra import compose, initialize_config_dir

DEFAULT_CAR_GLOBAL_PROMPT = (
    "Install four wheels and screws on a wheel-less toy vehicle using the toy drill."
)
TIANJI_JOINT_CONFIGS = ("tianji_stage1_joint", "tianji_stage1_joint_h60")


def _rlinf_root() -> Path:
    return Path(os.environ.get("RLINF_ROOT", Path(__file__).resolve().parents[2])).resolve()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline GPU smoke test for 16D joint Sim2Real toy-car VLA checkpoints."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Checkpoint directory (defaults to $TIANJI_RLT_STAGE1_MODEL_PATH or $TIANJI_MODEL_DIR).",
    )
    parser.add_argument(
        "--norm-stats",
        default=None,
        help="Path to norm_stats.json (defaults to $TIANJI_RLT_STAGE1_NORM_STATS_PATH or <model>/norm_stats.json).",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Optional prompts.json path (supports 1, 4, 12, 24+ prompts; defaults to $TIANJI_SUBTASK_PROMPTS if set).",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional single global prompt (used when no --prompts file is given).",
    )
    parser.add_argument(
        "--config-name",
        default=os.environ.get("TIANJI_OPENPI_CONFIG", "tianji_stage1_joint_h60"),
        choices=TIANJI_JOINT_CONFIGS,
        help="OpenPI joint config name (default: tianji_stage1_joint_h60).",
    )
    parser.add_argument(
        "--use-rlt",
        default=os.environ.get("TIANJI_USE_RLT", "auto"),
        choices=("auto", "true", "false", "1", "0"),
        help="Whether checkpoint uses RLT module or auto-detect from weights (default: auto).",
    )
    parser.add_argument(
        "--fast",
        default=None,
        choices=("0", "1", "2"),
        help="Set TIANJI_FAST_INFER mode (0=eager, 1=CUDA graph, 2=fused+CUDA graph).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Warmup iterations before latency benchmarking (default: 3).",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=20,
        help="Benchmark iterations for latency percentile reporting on prompt 0 (default: 20).",
    )
    parser.add_argument(
        "--blend-steps",
        type=int,
        default=int(os.environ.get("TIANJI_CHUNK_BLEND_STEPS", "10") or 10),
        help="Control-side Smoothstep blend steps to verify offline (default: 10).",
    )
    parser.add_argument(
        "--v-max",
        type=float,
        default=float(os.environ.get("TIANJI_MAX_STEP_VEL_RAD_S", "0.7854") or 0.7854),
        help="Per-step arm velocity clamp limit in rad/s (default: 0.7854).",
    )
    return parser.parse_args(argv)


def _resolve_prompts(args: argparse.Namespace, rlinf_root: Path) -> tuple[list[str], str]:
    from rlinf.envs.realworld.tianji.tianji_subtasks import load_subtask_prompts

    prompts_arg = args.prompts
    if prompts_arg is None and not args.prompt:
        env_prompts = os.environ.get("TIANJI_SUBTASK_PROMPTS", "").strip()
        if env_prompts:
            prompts_arg = env_prompts
    if prompts_arg:
        max_subtasks = int(os.environ.get("TIANJI_MAX_SUBTASKS", "32"))
        loaded = list(load_subtask_prompts(prompts_arg, limit=max_subtasks))
        return loaded, "subtasks"

    global_prompt = (
        args.prompt
        or os.environ.get("TIANJI_TASK_PROMPT", "").strip()
        or DEFAULT_CAR_GLOBAL_PROMPT
    )
    return [global_prompt], "global"


def _latency_stats_ms(samples_ms: list[float]) -> dict:
    import numpy as np

    arr = np.asarray(samples_ms, dtype=np.float64)
    return {
        "mean_ms": round(float(arr.mean()), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "max_ms": round(float(arr.max()), 2),
        "within_600ms_budget": bool(arr.max() < 600.0),
    }


def run_check(args: argparse.Namespace) -> dict:
    rlinf_root = _rlinf_root()
    if str(rlinf_root) not in sys.path:
        sys.path.insert(0, str(rlinf_root))
    if args.fast is not None:
        os.environ["TIANJI_FAST_INFER"] = args.fast

    import numpy as np

    from rlinf.envs.realworld.tianji.tianji_chunk_monitor import (
        ChunkExecutionMonitor,
        blend_new_chunk,
        clamp_chunk_step_velocity,
    )
    from rlinf.envs.realworld.tianji.tianji_contract import (
        contract_from_env,
        identity_state,
        prefetch_schedule,
    )
    from rlinf.envs.realworld.tianji.tianji_kinematics import (
        JOINT_ACTION_BIAS,
        JOINT_ACTION_SCALE,
    )
    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
        checkpoint_has_rlt,
        resolve_stage1_use_rlt,
    )
    from rlinf.workers.rollout.hf.tianji_vla_eval_worker import joint_reference_actions

    raw_model = (
        args.model
        or os.environ.get("TIANJI_RLT_STAGE1_MODEL_PATH")
        or os.environ.get("TIANJI_MODEL_DIR")
    )
    if not raw_model:
        raise ValueError(
            "Model checkpoint directory is required via --model or TIANJI_RLT_STAGE1_MODEL_PATH."
        )
    model_dir = Path(raw_model).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    norm_stats = Path(
        args.norm_stats
        or os.environ.get("TIANJI_RLT_STAGE1_NORM_STATS_PATH")
        or (model_dir / "norm_stats.json")
    ).expanduser().resolve()
    if not norm_stats.is_file():
        raise FileNotFoundError(f"norm_stats.json not found: {norm_stats}")
    if not isinstance(json.loads(norm_stats.read_text(encoding="utf-8")), dict):
        raise TypeError(f"Invalid norm_stats.json object: {norm_stats}")

    os.environ.setdefault("RLINF_ROOT", str(rlinf_root))
    os.environ.setdefault("EMBODIED_PATH", str(rlinf_root / "examples/embodiment"))
    os.environ["TIANJI_ACTION_SPACE"] = "joint"
    os.environ.setdefault("TIANJI_CONTROL_HZ", "30")
    os.environ.setdefault("TIANJI_ACTION_HORIZON", "60")
    os.environ["TIANJI_OPENPI_CONFIG"] = args.config_name
    os.environ["TIANJI_STAGE1_VLA_ONLY"] = "1"
    os.environ.setdefault("TIANJI_MAX_SUBTASKS", "32")
    os.environ["TIANJI_RLT_STAGE1_MODEL_PATH"] = str(model_dir)
    os.environ["TIANJI_RLT_STAGE1_NORM_STATS_PATH"] = str(norm_stats)

    contract = contract_from_env()
    schedule = prefetch_schedule(contract.action_chunk_length, contract.control_hz)
    chunk = int(os.environ.get("TIANJI_STAGE2_CHUNK", str(schedule.chunk)))
    offset = int(os.environ.get("RLT_REF_OFFSET", str(schedule.ref_offset)))
    horizon = int(contract.action_chunk_length)

    detected_has_rlt = checkpoint_has_rlt(model_dir)
    if detected_has_rlt is None:
        raise FileNotFoundError(
            f"No model.safetensors or full_weights.pt checkpoint found under {model_dir}"
        )
    resolved_use_rlt = resolve_stage1_use_rlt(model_dir, explicit=args.use_rlt)
    if resolved_use_rlt is None:
        raise FileNotFoundError(f"Unable to resolve checkpoint type under {model_dir}")

    # Pass a concrete boolean to Hydra compose, then confirm auto-detection agrees.
    os.environ["TIANJI_USE_RLT"] = "true" if resolved_use_rlt else "false"

    prompts, prompt_mode = _resolve_prompts(args, rlinf_root)

    config_dir = str(Path(os.environ["EMBODIED_PATH"]) / "config")
    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name="tianji2_joint_stage1")

    assert cfg.rollout.rlt_feature_model.openpi.config_name == args.config_name
    assert cfg.rollout.rlt_feature_model.openpi_data.repo_id == args.config_name
    cfg.rollout.rlt_feature_model.openpi.use_rlt = bool(resolved_use_rlt)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required to run check_sim2real_car_model.py")

    model = (
        get_model(cfg.rollout.rlt_feature_model)
        .eval()
        .requires_grad_(False)
        .to("cuda:0")
    )
    assert bool(model.rlt_cfg.use_rlt) is bool(resolved_use_rlt)
    if not resolved_use_rlt:
        assert not hasattr(model, "rlt_module")
    fast_mode = os.environ.get("TIANJI_FAST_INFER", "0").strip() or "0"
    if fast_mode not in ("", "0"):
        from rlinf.models.embodiment.openpi_rlinf.fast_infer import warmup

        warmup(model)

    base_obs = {
        "states": torch.as_tensor(identity_state()[None]),
        "joint_states": torch.zeros((1, 16), dtype=torch.float32),
        "rlt_qvel": torch.zeros((1, 14), dtype=torch.float32),
        "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        "extra_view_images": torch.zeros((1, 2, 224, 224, 3), dtype=torch.uint8),
    }

    rlt_dim = int(model.rlt_cfg.rlt_embed_dim)
    prompt_unguided_ms: list[float] = []
    prompt_rtc_ms: list[float] = []
    bench_unguided_ms: list[float] = []
    bench_rtc_ms: list[float] = []
    last_act_u = None

    with torch.inference_mode():
        # Warmup pass so CUDA kernels/graphs are excluded from prompt timing.
        warm_obs = dict(base_obs)
        warm_obs["task_descriptions"] = [prompts[0]]
        for _ in range(max(1, int(args.warmup))):
            warm_feat = model.extract_rlt_obs(warm_obs)
            _ = joint_reference_actions(warm_feat, chunk, offset)
        torch.cuda.synchronize()

        delay_rows = min(6, chunk)
        for prompt in prompts:
            obs_u = dict(base_obs)
            obs_u["task_descriptions"] = [prompt]
            t0 = time.monotonic()
            feat_u = model.extract_rlt_obs(obs_u)
            act_u = joint_reference_actions(feat_u, chunk, offset)
            torch.cuda.synchronize()
            prompt_unguided_ms.append((time.monotonic() - t0) * 1000.0)
            last_act_u = act_u

            assert feat_u["ref_chunk"].shape == (1, horizon, 16)
            assert torch.isfinite(feat_u["ref_chunk"]).all()
            assert float(feat_u["ref_chunk"].abs().max()) <= 1.00001
            assert act_u.shape == (1, chunk, 16)
            assert torch.isfinite(act_u).all()
            assert float(act_u.abs().max()) <= 1.0
            assert feat_u["z_rl"].shape == (1, rlt_dim)
            assert torch.isfinite(feat_u["z_rl"]).all()
            if not resolved_use_rlt:
                assert torch.count_nonzero(feat_u["z_rl"]) == 0

            obs_r = dict(base_obs)
            obs_r["task_descriptions"] = [prompt]
            obs_r["rtc_prev_valid"] = torch.tensor([True])
            obs_r["rtc_prev_actions"] = act_u[:, :delay_rows, :].clone()
            t1 = time.monotonic()
            feat_r = model.extract_rlt_obs(obs_r)
            act_r = joint_reference_actions(feat_r, chunk, offset)
            torch.cuda.synchronize()
            prompt_rtc_ms.append((time.monotonic() - t1) * 1000.0)

            assert feat_r["ref_chunk"].shape == (1, horizon, 16)
            assert torch.isfinite(feat_r["ref_chunk"]).all()
            assert float(feat_r["ref_chunk"].abs().max()) <= 1.00001
            assert act_r.shape == (1, chunk, 16)
            assert torch.isfinite(act_r).all()
            assert float(act_r.abs().max()) <= 1.0
            assert feat_r["z_rl"].shape == (1, rlt_dim)
            assert torch.isfinite(feat_r["z_rl"]).all()

        # Dedicated multi-iteration latency benchmark on prompt 0.
        bench_obs_u = dict(base_obs)
        bench_obs_u["task_descriptions"] = [prompts[0]]
        bench_obs_r = dict(base_obs)
        bench_obs_r["task_descriptions"] = [prompts[0]]
        bench_obs_r["rtc_prev_valid"] = torch.tensor([True])
        bench_obs_r["rtc_prev_actions"] = last_act_u[:, :delay_rows, :].clone()

        for _ in range(max(1, int(args.iters))):
            t0 = time.monotonic()
            _ = joint_reference_actions(model.extract_rlt_obs(bench_obs_u), chunk, offset)
            torch.cuda.synchronize()
            bench_unguided_ms.append((time.monotonic() - t0) * 1000.0)

            t1 = time.monotonic()
            _ = joint_reference_actions(model.extract_rlt_obs(bench_obs_r), chunk, offset)
            torch.cuda.synchronize()
            bench_rtc_ms.append((time.monotonic() - t1) * 1000.0)

    # Verify control-side Smoothstep blend, step velocity clamp, and ChunkExecutionMonitor.
    act_np = last_act_u[0].detach().cpu().numpy()
    blended = blend_new_chunk(act_np[-1], act_np, skip_m=0, blend_steps=int(args.blend_steps))
    clamped, clamped_steps, preclamp_vmax = clamp_chunk_step_velocity(
        act_np[-1],
        blended,
        dt=1.0 / float(contract.control_hz),
        v_max_rad_s=float(args.v_max),
        joint_scale=JOINT_ACTION_SCALE,
    )
    blended_phys = (blended * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS).astype(np.float32)
    clamped_phys = (clamped * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS).astype(np.float32)
    monitor = ChunkExecutionMonitor(
        dt=1.0 / float(contract.control_hz),
        v_max_rad_s=float(args.v_max) if float(args.v_max) > 0 else 2.0,
    )
    monitor_rec = monitor.analyze_chunk(clamped_phys[0], blended_phys, clamped_phys)

    return {
        "model": str(model_dir),
        "norm_stats": str(norm_stats),
        "config_name": args.config_name,
        "fast_infer_mode": fast_mode,
        "checkpoint_has_rlt": bool(detected_has_rlt),
        "use_rlt": bool(resolved_use_rlt),
        "prompt_mode": prompt_mode,
        "prompt_count": len(prompts),
        "control_hz": contract.control_hz,
        "action_horizon": horizon,
        "chunk": chunk,
        "ref_offset": offset,
        "action_dim": 16,
        "warmup_iters": int(args.warmup),
        "bench_iters": int(args.iters),
        "unguided_latency_ms": _latency_stats_ms(bench_unguided_ms),
        "rtc_guided_latency_ms": _latency_stats_ms(bench_rtc_ms),
        "prompt_unguided_mean_ms": round(sum(prompt_unguided_ms) / len(prompt_unguided_ms), 2),
        "prompt_rtc_mean_ms": round(sum(prompt_rtc_ms) / len(prompt_rtc_ms), 2),
        "control_blend_check": {
            "blend_steps": int(args.blend_steps),
            "v_max_rad_s": float(args.v_max),
            "clamped_steps": int(clamped_steps),
            "preclamp_max_vel_rad_s": round(float(preclamp_vmax), 4),
            "monitor_summary": monitor_rec["summary"],
        },
        "device": torch.cuda.get_device_name(0),
    }


def main(argv: list[str] | None = None) -> int:
    summary = run_check(_parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
