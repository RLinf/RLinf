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

"""Record one prepared Search-R1 run, including raw groups and GPU samples."""

import argparse
import json
import os
import runpy
import subprocess
import sys
import threading
import time
from pathlib import Path


def cleanup_owned_ray(directory: Path, started: float) -> list[int]:
    """Stop Ray processes created by this invocation under its private directory."""
    import psutil

    targets = {}
    for process in psutil.process_iter(["pid", "cmdline", "create_time"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if (
                process.info["create_time"] >= started
                and str(directory) in command
                and any(name in command for name in ("raylet", "gcs_server"))
            ):
                targets[process.pid] = process
                targets.update(
                    {child.pid: child for child in process.children(recursive=True)}
                )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for process in targets.values():
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    _, alive = psutil.wait_procs(list(targets.values()), timeout=5)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return sorted(targets)


def main() -> None:
    """Launch the native entry point with diagnostic recording enabled."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--megatron-path", type=Path, required=True)
    parser.add_argument("--driver", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    case = args.case.resolve()
    evidence = case / "evidence"
    evidence.mkdir(exist_ok=True)
    if args.driver:
        from ragged_observer import install_driver

        install_driver()
        entry = root / "examples/agent/searchr1/train.py"
        sys.argv = [
            str(entry),
            "--config-path",
            str(case / "config"),
            "--config-name",
            "run",
        ]
        runpy.run_path(str(entry), run_name="__main__")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join(
            (
                str(root),
                str(Path(__file__).parent.resolve()),
                str(args.megatron_path.resolve()),
                str(root / "examples"),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
        "RLINF_EXT_MODULE": "ragged_observer",
        "RAGGED_RUN_DIR": str(case),
        "RAY_TMPDIR": str(case / "ray"),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "RAY_DEDUP_LOGS": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONHASHSEED": "42",
        "PYTHONUNBUFFERED": "1",
        "HYDRA_FULL_ERROR": "1",
        "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    }
    stop = threading.Event()

    def sample_memory() -> None:
        with (evidence / "gpu_memory.jsonl").open("a", buffering=1) as stream:
            while not stop.is_set():
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                )
                stream.write(
                    json.dumps({"time": time.time(), "gpus": result.stdout}) + "\n"
                )
                stop.wait(1)

    monitor = threading.Thread(target=sample_memory, daemon=True)
    monitor.start()
    started = time.time()
    try:
        with (case / "launcher.log").open("w") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--case",
                    str(case),
                    "--megatron-path",
                    str(args.megatron_path.resolve()),
                    "--driver",
                ],
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    finally:
        stop.set()
        monitor.join(timeout=3)
        cleaned = cleanup_owned_ray(case / "ray", started)
    (case / "status.json").write_text(
        json.dumps(
            {
                "started": started,
                "finished": time.time(),
                "exit_code": process.returncode,
                "cleaned_ray_processes": cleaned,
            },
            indent=2,
        )
        + "\n"
    )
    sys.exit(process.returncode)


if __name__ == "__main__":
    main()
