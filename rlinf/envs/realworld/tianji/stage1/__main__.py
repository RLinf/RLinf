# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Stage1 GPU service, manual-start hardware execution, or offline check."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path

from .server import Stage1Server, load_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen Tianji Stage1 GPU inference")
    parser.add_argument(
        "--mode", choices=("serve", "execute", "check"), default="serve"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6110)
    parser.add_argument(
        "--predictor",
        default="gpu",
        help="serve only: gpu, zero (hold), or module:factory",
    )
    parser.add_argument(
        "--model-path", default=os.environ.get("TIANJI_RLT_STAGE1_MODEL_PATH")
    )
    parser.add_argument(
        "--norm-stats-path", default=os.environ.get("TIANJI_RLT_STAGE1_NORM_STATS_PATH")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--loopinfra-url",
        default=os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780"),
    )
    parser.add_argument("--hil-port", type=int, default=8879)
    parser.add_argument(
        "--control-profile",
        choices=("wuji_compat", "loopinfra"),
        default="wuji_compat",
        help="required LoopInfra collector profile for hardware execution",
    )
    from ..tianji_contract import DEFAULT_CONTRACT, prefetch_schedule

    horizon = DEFAULT_CONTRACT.action_chunk_length
    schedule = prefetch_schedule(horizon, DEFAULT_CONTRACT.control_hz)
    parser.add_argument(
        "--execute-steps",
        type=int,
        default=horizon,
        help=f"model chunk steps to execute (TIANJI_ACTION_HORIZON={horizon})",
    )
    parser.add_argument(
        "--prefetch-step",
        type=int,
        default=schedule.request_after_steps,
        help="RTC request point in successful execution receipts (default: "
        "execute this many steps, keeping the inference budget before the chunk ends)",
    )
    parser.add_argument(
        "--subtask-prompts",
        default=os.environ.get("TIANJI_SUBTASK_PROMPTS") or None,
        help="JSON of subtask prompts in order; the HIL panel selects the current one",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="0: unlimited execute; two chunks in check mode",
    )
    parser.add_argument(
        "--no-rtc-guidance",
        action="store_true",
        help="execute/check: start each chunk at the RTC delay index without "
        "guiding it toward the previous chunk",
    )
    parser.add_argument(
        "--read-only",
        action="store_true",
        help="execute mode: observe/infer without policy writes",
    )
    parser.add_argument("--output-dir", default=os.environ.get("TIANJI_STAGE1_OUTPUT"))
    args = parser.parse_args()
    if not 1 <= args.execute_steps <= horizon or args.max_chunks < 0:
        parser.error(f"execute-steps must be 1..{horizon}, max-chunks must be >=0")
    if args.prefetch_step < 0 or args.prefetch_step >= args.execute_steps:
        parser.error("prefetch-step must be 0 or less than execute-steps")
    if args.mode == "execute" and args.prefetch_step < 1:
        parser.error("Tianji hardware execution requires the RTC prefetch schedule")
    if os.environ.get("TIANJI_ACTION_SPACE") == "joint":
        parser.error("Joint Stage1 uses the direct joint queue: run ./launcher/run_tianji_stage1_local.sh")
    if args.mode != "serve" and args.predictor != "gpu":
        parser.error("execute/check require the real GPU predictor")
    if args.subtask_prompts:
        from ..tianji_subtasks import load_subtask_prompts

        prompts = load_subtask_prompts(args.subtask_prompts)
        print(f"STAGE1_SUBTASKS {len(prompts)} from {args.subtask_prompts}", flush=True)
    print(
        f"STAGE1_SCHEDULE horizon={horizon} control_hz={DEFAULT_CONTRACT.control_hz:g} "
        f"execute_steps={args.execute_steps} prefetch_step={args.prefetch_step}",
        flush=True,
    )

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)

    if args.predictor == "gpu":
        import torch

        from .predictor import GPUStage1Predictor

        torch.manual_seed(0)
        print("STAGE1_LOADING model=" + str(args.model_path), flush=True)
        predictor = GPUStage1Predictor(
            model_path=args.model_path,
            norm_stats_path=args.norm_stats_path,
            device=args.device,
        )
        predictor.warmup()
        print("STAGE1_GPU_READY device=" + str(predictor.device), flush=True)
    else:
        from .contract import Stage1Contract

        predictor = load_predictor(args.predictor, Stage1Contract())

    if args.mode == "serve":
        server = Stage1Server(host=args.host, port=args.port, predictor=predictor)
        print(
            f"STAGE1_READY protocol=rlinf.tianji.stage1.v2 port={args.port}", flush=True
        )
        server.serve_forever()
        return

    from ..tianji_env import TianjiConfig, TianjiEnv
    from .runner import run_stage1

    check = args.mode == "check"
    directory = Path(
        args.output_dir
        or (
            "tianji_runs/stage1_"
            + time.strftime("%Y%m%d_%H%M%S")
            + "_"
            + str(os.getpid())
        )
    )
    directory.mkdir(parents=True, exist_ok=False)
    env = TianjiEnv(
        TianjiConfig(
            backend="dummy" if check else "loopinfra",
            server_url=args.loopinfra_url,
            server_token=os.environ.get("LOOPINFRA_TOKEN"),
            read_only=check or args.read_only,
            control_profile=args.control_profile,
            hil_control=not check,
            hil_port=args.hil_port,
            hil_mode="stage1",
            manual_episode_control_only=not check,
            sparse_execution_observation=bool(args.prefetch_step),
            recorder_dir=str(directory / "transitions"),
            recorder_source_jpeg=True,
            subtask_prompts_file=args.subtask_prompts,
        )
    )
    try:
        if not check:
            print(
                f"STAGE1_HIL_READY http://127.0.0.1:{args.hil_port} (waiting for manual Start)",
                flush=True,
            )
        with (directory / "predictions.jsonl").open("w") as stream:
            report = run_stage1(
                env,
                predictor,
                execute_steps=args.execute_steps,
                prefetch_step=args.prefetch_step,
                max_chunks=args.max_chunks or (2 if check else 0),
                predictions=stream,
                inference_frames_dir=directory / "inference_inputs",
                rtc_guidance=not args.no_rtc_guidance,
            )
        report.update(
            control_profile=args.control_profile,
            backend="dummy" if check else "loopinfra",
            read_only=check or args.read_only,
            model_path=predictor.model_path,
            norm_stats_path=predictor.norm_stats_path,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            cuda_device_count=torch.cuda.device_count(),
            gpu_peak_gib=torch.cuda.max_memory_allocated(predictor.device) / 2**30,
        )
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        failure = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "reset_trace": getattr(exc, "reset_trace", []),
            "receipt": getattr(exc, "receipt", None),
            "observation_error": getattr(exc, "observation_error", None),
        }
        try:
            (directory / "error.json").write_text(json.dumps(failure, indent=2) + "\n")
            print(f"STAGE1_ERROR_REPORT {directory / 'error.json'}", flush=True)
        except OSError as log_error:
            print(f"Stage1 error report could not be saved: {log_error}", flush=True)
        raise
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stage1 stopped.", flush=True)
