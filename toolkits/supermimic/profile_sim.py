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

"""Measure SuperMimic collection and update cost on one GPU.

Each env count runs in its own process so simulator memory is released
between sizes. Reported per size: build time, vector steps per second, env
transitions per second, the simulator's share of collection time, optional
seconds per update, peak GPU memory, replay bytes and host RSS.

Example::

    python toolkits/supermimic/profile_sim.py \\
        --motion-path /path/to/lafan1_g1 --num-envs 8 64 256 1024 \\
        --steps 50 --update-iters 2 --out profile.json

Sizes that fail (for example out of memory) are reported with their error
instead of stopping the sweep. ``--estimate-envs`` extrapolates replay and
peak memory linearly from the largest size that ran; it is an estimate,
not a measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ``python toolkits/supermimic/profile_sim.py`` puts only this folder on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
CONFIG_DIR = REPO / "examples" / "embodiment" / "config"


def _compose(config: str, overrides: list[str]):
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        return compose(config_name=config, overrides=overrides)


def _tensor_bytes(obj) -> int:
    import torch

    return sum(
        value.numel() * value.element_size()
        for value in vars(obj).values()
        if isinstance(value, torch.Tensor)
    )


def run_one(args) -> dict:
    import torch

    from rlinf.workers.supermimic.trainer import (
        SuperMimicTrainer,
        _host_rss_bytes,
        gpu_memory_metrics,
    )
    from rlinf.workers.supermimic.validate import validate_supermimic_cfg

    overrides = [
        f"env.num_envs={args.single}",
        f"env.device={args.device}",
        f"env.replay_length={args.replay_length}",
        f"algorithm.warmup_sim_steps={args.warmup}",
    ]
    if args.motion_path:
        overrides.append(f"env.motion_path={args.motion_path}")
    overrides += list(args.override)
    cfg = validate_supermimic_cfg(_compose(args.config, overrides), world_size=1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
    result = {
        "num_envs": args.single,
        "config": args.config,
        "device": str(device),
        "replay_length": args.replay_length,
        "imagine_batch": int(cfg.algorithm.imagine_batch),
        "wm_window_batch": int(cfg.algorithm.wm_window_batch),
    }
    t0 = time.perf_counter()
    trainer = SuperMimicTrainer(cfg, device=device)
    result["build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    trainer.collect_steps(args.warmup, accumulate=False)
    result["warmup_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    metrics = trainer.collect_steps(args.steps, accumulate=False)
    elapsed = time.perf_counter() - t0
    result["collect_s"] = elapsed
    result["vector_steps_per_s"] = args.steps / elapsed
    result["env_transitions_per_s"] = args.steps * args.single / elapsed
    result["sim_share"] = float(metrics["time/sim_s"]) / elapsed
    result["valid_windows"] = float(metrics["buffer/valid_windows"])
    if args.update_iters > 0:
        times = []
        for _ in range(args.update_iters):
            trainer.step_acc = 1.0
            t0 = time.perf_counter()
            update_metrics = trainer.maybe_update()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
        # The first update includes one-time kernel setup.
        result["update_s"] = times
        result["update_s_steady"] = times[-1]
        result["update_breakdown_s"] = {
            key: float(value)
            for key, value in update_metrics.items()
            if key.startswith("time/") and key != "time/sim_s"
        }
    result["replay_bytes"] = _tensor_bytes(trainer.buffer)
    result["motion_bytes"] = int(
        trainer.motion.frames.numel() * trainer.motion.frames.element_size()
    )
    # Torch, Warp and whole-device counters overlap; they are not summed.
    result.update(
        {
            key.split("/", 1)[1]: int(value)
            for key, value in gpu_memory_metrics(device).items()
        }
    )
    result["host_rss"] = int(_host_rss_bytes())
    return result


def estimate(rows: list[dict], target: int) -> dict | None:
    ok = [row for row in rows if "error" not in row]
    if not ok:
        return None
    base = max(ok, key=lambda row: row["num_envs"])
    scale = target / base["num_envs"]
    out = {
        "num_envs": target,
        "estimate_from": base["num_envs"],
        "kind": "linear estimate",
    }
    out["replay_bytes"] = int(base["replay_bytes"] * scale)
    if "torch_peak_allocated" in base:
        fixed = base["torch_peak_allocated"] - base["replay_bytes"]
        out["torch_peak_allocated_lower_bound"] = int(fixed + out["replay_bytes"])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default="supermimic_lafan1")
    parser.add_argument(
        "--motion-path", default=os.environ.get("SUPERMIMIC_MOTION_PATH", "")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, nargs="+", default=[8, 64, 256, 1024])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--replay-length", type=int, default=1000)
    parser.add_argument("--update-iters", type=int, default=0)
    parser.add_argument("--estimate-envs", type=int, nargs="*", default=[4096])
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="extra Hydra override, repeatable",
    )
    parser.add_argument("--single", type=int, default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.motion_path:
        parser.error("--motion-path (or SUPERMIMIC_MOTION_PATH) is required")

    if args.single is not None:
        print(json.dumps(run_one(args)))
        return 0

    rows = []
    for num_envs in args.num_envs:
        cmd = [sys.executable, __file__, "--single", str(num_envs)]
        for name in (
            "config",
            "motion_path",
            "device",
            "steps",
            "warmup",
            "replay_length",
            "update_iters",
        ):
            cmd += [f"--{name.replace('_', '-')}", str(getattr(args, name))]
        for item in args.override:
            cmd += ["--override", item]
        print(f"[profile] num_envs={num_envs}", flush=True)
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=args.timeout, cwd=REPO
            )
        except subprocess.TimeoutExpired:
            rows.append(
                {"num_envs": num_envs, "error": f"timeout after {args.timeout}s"}
            )
            continue
        lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
        if proc.returncode != 0 or not lines:
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-5:]
            rows.append({"num_envs": num_envs, "error": " | ".join(tail)})
        else:
            rows.append(json.loads(lines[-1]))
        print(json.dumps(rows[-1]), flush=True)
    estimates = [e for e in (estimate(rows, n) for n in args.estimate_envs) if e]
    report = {"measured": rows, "estimated": estimates}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if any("error" not in row for row in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
