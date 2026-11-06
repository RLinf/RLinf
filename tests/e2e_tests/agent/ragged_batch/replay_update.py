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

"""Relocate the archived fixed-state GPU update experiment for a fresh replay."""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


def main() -> None:
    """Prepare or launch the archived eight-GPU diagnostic without cached results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--megatron-path", type=Path, required=True)
    parser.add_argument("--control", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    artifact, output = args.artifact.resolve(), args.output.resolve()
    source = artifact / "correctness"
    harness = source / "dense_repeat/harness" if args.control else source / "harness"
    output.mkdir(parents=True, exist_ok=False)
    (output / "harness").mkdir()
    (output / "evidence").mkdir()
    (output / "config").mkdir()
    for name in ("driver.py", "parity_worker.py"):
        shutil.copy2(harness / name, output / "harness" / name)
    config = yaml.safe_load((source / "config/run.yaml").read_text())
    config["rollout"]["model"]["model_path"] = str(args.model.resolve())
    config["data"]["train_data_paths"] = [str(artifact / "performance/questions.jsonl")]
    config["runner"].update(
        output_dir=str(output / "training"), save_interval=-1, val_check_interval=-1
    )
    (output / "config/run.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "SOURCE_RUN.txt").write_text(str(artifact / "performance") + "\n")
    if not args.run:
        return
    env = {
        **os.environ,
        "PARITY_ROOT": str(output),
        "PYTHONPATH": os.pathsep.join(
            (
                str(root),
                str(output / "harness"),
                str(args.megatron_path.resolve()),
                os.environ.get("PYTHONPATH", ""),
            )
        ),
        "RAY_TMPDIR": str(output / "ray"),
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "PYTHONHASHSEED": "42",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }
    env.pop("RLINF_EXT_MODULE", None)
    from record import cleanup_owned_ray

    started = time.time()
    try:
        with (output / "launcher.log").open("w") as stream:
            result = subprocess.run(
                [
                    sys.executable,
                    str(output / "harness/driver.py"),
                    "--config-path",
                    str(output / "config"),
                    "--config-name",
                    "run",
                ],
                cwd=root,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
    finally:
        cleanup_owned_ray(output / "ray", started)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
