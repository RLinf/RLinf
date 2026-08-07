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

"""Unified scalar potential-head entrypoint.

Stages:
  extract  Frozen VLM feature extraction (one shard)
  train    Train ScalarPotentialHead on feature shards
  all      Multi-GPU extract for train/eval × potential/progress, then train
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

_HERE = Path(__file__).resolve().parent
_EXTRACT = _HERE / "vlm_trend" / "extract_potential_features.py"
_TRAIN = _HERE / "vlm_trend" / "train_scalar_head.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_all(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        description="Extract features on multiple GPUs, then train the scalar head."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Potential SFT dataset root")
    parser.add_argument("--feat-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cuda-devices",
        default=os.environ.get("CUDA_DEVICES", "0"),
        help="Comma-separated GPU ids for sharded extract.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=4)
    parser.add_argument(
        "--device", default="cuda:0", help="Device for scalar-head train"
    )
    parser.add_argument("--epochs", type=int, default=50)
    args, train_extra = parser.parse_known_args(argv)

    gpus = [g.strip() for g in args.cuda_devices.split(",") if g.strip()]
    if not gpus:
        raise SystemExit("--cuda-devices must list at least one GPU id")
    world_size = len(gpus)
    feat_root = Path(args.feat_root)
    data_root = Path(args.data_root)
    feat_root.mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    python_bin = sys.executable
    for split in ("train", "eval"):
        manifest = data_root / split / "segments.jsonl"
        if not manifest.is_file():
            raise SystemExit(f"missing manifest: {manifest}")
        for sample_type in ("potential", "progress"):
            procs = []
            for rank, gpu in enumerate(gpus):
                output = feat_root / f"{split}_{sample_type}_rank{rank}.pt"
                cmd = [
                    python_bin,
                    str(_EXTRACT),
                    "--model-path",
                    args.model_path,
                    "--checkpoint",
                    args.checkpoint,
                    "--manifest",
                    str(manifest),
                    "--output",
                    str(output),
                    "--sample-type",
                    sample_type,
                    "--device",
                    "cuda:0",
                    "--batch-size",
                    str(args.feature_batch_size),
                    "--rank",
                    str(rank),
                    "--world-size",
                    str(world_size),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu
                procs.append(subprocess.Popen(cmd, env=env))
            for proc in procs:
                if proc.wait() != 0:
                    raise SystemExit(
                        f"feature extract failed with code {proc.returncode}"
                    )

    train_cmd = [
        python_bin,
        str(_TRAIN),
        "--train-pattern",
        str(feat_root / "train_potential_rank*.pt"),
        "--eval-pattern",
        str(feat_root / "eval_potential_rank*.pt"),
        "--progress-pattern",
        str(feat_root / "eval_progress_rank*.pt"),
        "--train-progress-pattern",
        str(feat_root / "train_progress_rank*.pt"),
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--epochs",
        str(args.epochs),
        *train_extra,
    ]
    if subprocess.call(train_cmd) != 0:
        raise SystemExit("scalar-head training failed")
    best = Path(args.output_dir) / "best.pt"
    if not best.is_file():
        raise SystemExit(f"missing {best}")
    print(f"dense scalar head ready at {best}")


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(
        description="Unified VLM Trend scalar potential-head stages.",
        add_help=False,
    )
    pre.add_argument(
        "--stage",
        required=True,
        choices=("extract", "train", "all"),
        help="extract one shard, train from shards, or run the full extract+train pipeline.",
    )
    pre.add_argument("-h", "--help", action="store_true")
    args, remaining = pre.parse_known_args(argv)
    if args.stage == "all":
        if args.help:
            _run_all(["--help"])
            return
        _run_all(remaining)
        return
    module = _load(
        _EXTRACT if args.stage == "extract" else _TRAIN,
        f"vlm_trend_scalar_{args.stage}",
    )
    if args.help:
        pre.print_help()
        module.parse_args(["--help"])
        sys.exit(0)
    stage_args = module.parse_args(remaining)
    if args.stage == "extract":
        module.main(stage_args)
    else:
        module.train(stage_args)


if __name__ == "__main__":
    main()
