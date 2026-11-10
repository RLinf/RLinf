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

"""Search, worker routing, and process-level MPS calibration contracts."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from rlinf.scheduler.hardware.accelerators.mps import (
    SESSION_ENV,
    mps_session,
    worker_environment,
)
from rlinf.scheduler.hardware.accelerators.nvidia_gpu import NvidiaGPUManager
from rlinf.scheduler.worker.worker import Worker
from toolkits.auto_mps_tuning.tuning import (
    launch_config,
    quota_levels,
    search_pair,
    validate_pair,
)


@pytest.mark.parametrize("target", ["env", "rollout"])
@pytest.mark.parametrize(
    "physical_sms,sm_step,expected",
    [
        (100, 1, [100, 97, 94, 96, 98]),
        (108, 2, [100, 93, 88, 91, 95]),
        (120, 4, [100, 87, 74, 80, 90]),
        (256, 2, [100, 97, 94, 96, 98]),
    ],
)
def test_search_steps_follow_hardware(target, physical_sms, sm_step, expected):
    mapping = {
        q: (physical_sms * q // (100 * sm_step)) * sm_step for q in range(10, 101)
    }
    other = "env" if target == "rollout" else "rollout"
    calls = []

    def measure(caps):
        calls.append(caps[target])
        assert caps[other] == 100
        return {
            "window_s": 10 + (caps[target] - expected[1]) ** 2,
            "service_s": {target: 2, other: 5},
        }

    search = search_pair(
        measure,
        mps_baseline=measure({"env": 100, "rollout": 100}),
        levels=quota_levels(mapping),
        min_percentage=10,
        max_trials=24,
        sm_by_percentage=mapping,
    )
    assert len(search["trials"]) == len(expected)
    assert calls == expected
    assert search["best"]["caps"][target] == expected[1]


@pytest.mark.parametrize("delta", [-0.000001, 0, 0.000001])
def test_search_crosses_plateaus_and_compares_times_directly(delta):
    calls = []

    def measure(caps):
        quota = caps["rollout"]
        calls.append(quota)
        times = {100: 10, 97: 10, 94: 10 + delta}
        return {"window_s": times.get(quota, 11), "service_s": {"rollout": 2, "env": 5}}

    mapping = {q: q for q in range(90, 101)}
    search = search_pair(
        measure,
        mps_baseline=measure({"rollout": 100, "env": 100}),
        min_percentage=90,
        max_trials=24,
        sm_by_percentage=mapping,
        levels=quota_levels(mapping),
    )
    assert calls[:3] == [100, 97, 94]
    assert len(calls) == len(set(calls))
    assert 90 not in calls
    expected = 94 if delta < 0 else 100
    assert search["best"]["caps"]["rollout"] == expected
    full = validate_pair(measure, search, levels=quota_levels(mapping))
    assert full["selected_caps"]["rollout"] == expected


@pytest.mark.parametrize("target", ["env", "rollout"])
@pytest.mark.parametrize(
    "anchor,expected", [(80, [80, 86, 100]), (93, [93, 86, 97, 100]), (100, [100, 97])]
)
@pytest.mark.parametrize("improves", [False, True])
def test_validation_checks_only_distinct_neighbors_and_control(
    target, anchor, expected, improves
):
    mapping = {
        q: 40 + 4 * sum(q >= b for b in (86, 93, 97, 99)) for q in range(80, 101)
    }
    levels = quota_levels(mapping)
    assert levels == [80, 86, 93, 97, 100]
    other = "env" if target == "rollout" else "rollout"
    best = {"caps": {target: anchor, other: 100}, "window_s": 8}
    search = {"target": target, "best": best, "trials": [best]}
    calls = []

    def measure(caps):
        calls.append(caps[target])
        assert caps[other] == 100
        optimum = 86 if improves else 100
        return {"window_s": 10 + (caps[target] - optimum) ** 2}

    result = validate_pair(measure, search, levels=levels)
    assert calls == expected
    assert result["selected_caps"][target] == min(
        expected, key=lambda q: abs(q - (86 if improves else 100))
    )
    if not improves:
        assert (result["window_s"], result["relative_window_gain"]) == (10, 0)


def test_search_budget_and_full_window_reuse():
    calls = []

    def measure(caps):
        calls.append(caps["rollout"])
        return {"window_s": caps["rollout"], "service_s": {"rollout": 2, "env": 5}}

    mapping = {80: 98, 85: 98, 90: 102, 100: 108}
    search = search_pair(
        measure,
        mps_baseline=measure({"rollout": 100, "env": 100}),
        min_percentage=80,
        max_trials=1,
        sm_by_percentage=mapping,
        levels=quota_levels(mapping),
    )
    assert calls == [100]
    assert len(search["trials"]) == 1
    result = validate_pair(
        measure, search, levels=quota_levels(mapping), reuse_search=True
    )
    assert calls == [100, 90]
    assert result["selected_caps"] == {"rollout": 90, "env": 100}


@pytest.fixture
def config(tmp_path):
    return OmegaConf.create(
        {
            "cluster": {
                "num_nodes": 1,
                "component_placement": {"actor,env,rollout": "all"},
            },
            "runner": {"logger": {"log_path": str(tmp_path / "run")}},
            "actor": {
                "group_name": "Actor",
                "global_batch_size": 16,
                "model": {"num_action_chunks": 5},
            },
            "env": {
                "group_name": "Env",
                "train": {"total_num_envs": 16, "max_steps_per_rollout_epoch": 20},
            },
            "rollout": {"group_name": "Rollout", "pipeline_stage_num": 2},
            "mps": {
                "enabled": True,
                "auto_tuning": {"min_percentage": 95, "window_chunks": 1},
            },
        }
    )


@pytest.fixture
def fake_mps(monkeypatch, tmp_path):
    """Replace the external NVIDIA CLI services, recording daemon lifetimes."""
    service = (
        f"#!{sys.executable}\n"
        + r"""
import os, sys
from pathlib import Path
if Path(sys.argv[0]).name == "nvidia-smi":
    print("fake-driver" if "--query-gpu=driver_version" in sys.argv else "0, GPU-aaa, 8.0\n1, GPU-bbb, 8.0")
    sys.exit(0)
pipes = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
command = "-d" if "-d" in sys.argv else sys.stdin.read().strip()
with open(os.environ["FAKE_MPS_COMMANDS"], "a") as log:
    log.write(f"{pipes} {command}\n")
if command == "get_server_list":
    print(1234)
elif command.startswith("get_client_list"):
    print((pipes / "client").read_text())
"""
    )
    for name in ("nvidia-smi", "nvidia-cuda-mps-control"):
        executable = tmp_path / name
        executable.write_text(service)
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}:{os.environ['PATH']}")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("FAKE_MPS_COMMANDS", str(tmp_path / "commands.log"))
    monkeypatch.delenv(SESSION_ENV, raising=False)
    for key in tuple(os.environ):
        if key.startswith("CUDA_MPS_"):
            monkeypatch.delenv(key)
    return tmp_path


@pytest.mark.parametrize(
    "group,quota", [("Env", 80), ("Rollout", 100), ("Actor", None)]
)
def test_worker_environment_injection(config, fake_mps, monkeypatch, group, quota):
    previous = dict(os.environ)
    with mps_session(config, fake_mps, caps={"env": 80, "rollout": 100}) as environment:
        assert dict(os.environ) == previous
        session = json.loads(environment[SESSION_ENV])
        pipes = Path(session["pipe_directory"])
        environment.update(GROUP_NAME=group, CUDA_VISIBLE_DEVICES="1,0")
        overrides = worker_environment(environment)
        environment.update(overrides)
        if quota is None:
            assert overrides == {}
            assert not Path(environment["CUDA_MPS_PIPE_DIRECTORY"]).exists()
        else:
            assert environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] == str(quota)
            assert environment["CUDA_VISIBLE_DEVICES"] == "GPU-bbb,GPU-aaa"
            assert environment["CUDA_MPS_PIPE_DIRECTORY"] == str(pipes)
            (pipes / "client").write_text(str(os.getpid()))
        with monkeypatch.context() as worker:
            for key, value in environment.items():
                worker.setenv(key, value)
            assert NvidiaGPUManager.get_visible_devices() == [1, 0]
            if quota is not None:
                Worker.verify_mps_client(None)
    assert dict(os.environ) == previous
    assert not pipes.exists()


@pytest.fixture
def fake_cuda(fake_mps, monkeypatch):
    """Load a fake vendor library in probe subprocesses; keep the RLinf probe real."""
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("The fake CUDA driver requires a C compiler")
    source = fake_mps / "cuda.c"
    source.write_text(r"""
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int cuInit(unsigned flags) { return 0; }
int cuDeviceGetCount(int *count) { *count = 2; return 0; }
int cuDeviceGet(int *device, int ordinal) { *device = ordinal; return 0; }
int cuDeviceGetName(char *name, int size, int device) {
    snprintf(name, size, "Fake GPU"); return 0;
}
int cuDeviceGetAttribute(int *value, int attr, int device) {
    char *quota = getenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE");
    *value = attr == 16 ? (quota ? (108 * atoi(quota) / 200) * 2 : 108) : (attr == 75 ? 8 : 0);
    return 0;
}
int cuCtxCreate_v2(void **context, unsigned flags, int device) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/client", getenv("CUDA_MPS_PIPE_DIRECTORY"));
    FILE *file = fopen(path, "w");
    if (file) { fprintf(file, "%d", getpid()); fclose(file); }
    *context = (void *)1; return 0;
}
int cuCtxDestroy_v2(void *context) { return 0; }
""")
    subprocess.run(
        [
            compiler,
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(fake_mps / "libcuda.so.1"),
        ],
        check=True,
    )
    monkeypatch.setenv(
        "LD_LIBRARY_PATH", f"{fake_mps}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    )


@pytest.mark.parametrize(
    "mode",
    [
        "auto",
        "native",
        "fixed_env",
        "fixed_rollout",
        "signal",
        "trial_failure",
    ],
)
def test_launcher_process_flow(config, fake_mps, fake_cuda, monkeypatch, mode):
    """Exercise orchestration with real subprocesses and a deterministic workload."""
    monkeypatch.setenv("FAKE_TRAINING_MODE", mode)
    inherited_pipe = fake_mps / "inherited-pipes"
    inherited_pipe.mkdir()
    monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", str(inherited_pipe))
    monkeypatch.setenv("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", "40")
    if mode.startswith("fixed_"):
        config.mps[f"{mode.removeprefix('fixed_')}_percentage"] = 80
    if mode == "signal":
        config.mps.env_percentage = 100
    original = OmegaConf.to_container(config, resolve=True)
    previous = dict(os.environ)
    script = fake_mps / "training.py"
    script.write_text(r"""
import json, os, signal, sys
from pathlib import Path
from omegaconf import OmegaConf
args = sys.argv
cfg = OmegaConf.load(Path(args[args.index("--config-path") + 1]) / (args[args.index("--config-name") + 1] + ".yaml"))
session = json.loads(os.environ.get("RLINF_MPS_SESSION", "null"))
mode = os.environ["FAKE_TRAINING_MODE"]
assert "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" not in os.environ
if cfg.mps.get("trial", False):
    if mode == "trial_failure" and session:
        sys.exit(9)
    full = cfg.env.train.max_steps_per_rollout_epoch == 20
    quota = session["groups"]["Rollout"] if session else 100
    window = 10 + (quota - (99 if full else 97)) ** 2 / 100 if session else (9 if mode == "native" else 11)
    result = {"window_s": window, "service_s": {"rollout": 2, "env": 5}}
    (Path(cfg.runner.logger.log_path) / "measurement.json").write_text(json.dumps(result))
else:
    assert not cfg.mps.enabled
    Path(__file__).with_suffix(".json").write_text(json.dumps({"session": session, "pipe": os.environ["CUDA_MPS_PIPE_DIRECTORY"], "config": OmegaConf.to_container(cfg)}))
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGTERM)
    sys.exit(7)
""")
    if mode == "trial_failure":
        with pytest.raises(RuntimeError, match="exited 9.*run.log"):
            launch_config(config, script)
        assert not script.with_suffix(".json").exists()
    else:
        assert launch_config(config, script) == (143 if mode == "signal" else 7)
        formal = json.loads(script.with_suffix(".json").read_text())
        expected = {
            "auto": {"Env": 100, "Rollout": 99},
            "fixed_env": {"Env": 80, "Rollout": 100},
            "fixed_rollout": {"Env": 100, "Rollout": 80},
            "signal": {"Env": 100, "Rollout": 100},
        }.get(mode)
        assert (formal["session"]["groups"] if formal["session"] else None) == expected
        assert not Path(formal["pipe"]).exists()
        assert formal["config"]["env"] == original["env"]
        assert formal["config"]["actor"] == original["actor"]
    assert dict(os.environ) == previous
    assert OmegaConf.to_container(config, resolve=True) == original
    root = Path(config.runner.logger.log_path) / "mps_tuning"
    trials = sorted(root.glob("*/input.yaml"))
    if mode in ("auto", "native", "trial_failure"):
        assert "-native-" in trials[0].parent.name
        assert "-search-" in trials[1].parent.name
        for path in trials:
            trial = OmegaConf.load(path)
            assert trial.actor.global_batch_size == 16
            assert trial.env.train.total_num_envs == 16
            assert trial.rollout.pipeline_stage_num == 2
            assert trial.env.train.max_steps_per_rollout_epoch == (
                20 if "-validation-" in path.parent.name else 5
            )
    if mode == "auto":
        assert (root / "resource_probe/resources.json").exists()
        validation = list(root.glob("*-validation-*/measurement.json"))
        assert len(validation) == 4
        assert len(list(root.glob("*/measurement.json"))) == 9
    else:
        assert len(trials) == (2 if mode in ("native", "trial_failure") else 0)
        assert not (root / "resource_probe").exists()
    commands = fake_mps / "commands.log"
    if commands.exists():
        records = [line.split() for line in commands.read_text().splitlines()]
        started = [path for path, *command in records if command == ["-d"]]
        stopped = [path for path, *command in records if command == ["quit"]]
        assert started == stopped
        assert all(not Path(path).exists() for path in started)
