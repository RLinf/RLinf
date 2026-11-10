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

"""Cold-start MPS pair calibration with native rollout completion windows."""

import json
import math
import select
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.scheduler.hardware.accelerators.mps import (
    disabled_mps_environment,
    mps_daemon,
    mps_session,
    wait_for_clients,
)
from rlinf.utils.logging import get_logger

FULL_CAPS = {"rollout": 100, "env": 100}


def launch_config(cfg: DictConfig, training_script: Path) -> int:
    """Tune a composed configuration and launch training with session-owned quotas.

    Args:
        cfg: Training configuration; only mps.enabled is required to opt in.
        training_script: Entry point used for trials and the final training process.

    Returns:
        The shell exit code (128 + signal for signal termination). The caller's
        configuration is unchanged.
    """
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    log_dir = Path(cfg.runner.logger.log_path).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    caps = tune_before_training(cfg, log_dir, training_script)
    with open_dict(cfg):
        cfg.mps = {"enabled": False}
        cfg.runner.logger.log_path = str(log_dir)
        cfg.hydra = {
            "run": {"dir": str(log_dir)},
            "output_subdir": None,
            "job": {"chdir": False},
        }
    # Freeze the composed configuration so edits during calibration cannot change training.
    OmegaConf.save(cfg, log_dir / "training.yaml")
    command = [
        sys.executable,
        str(training_script),
        "--config-path",
        str(log_dir),
        "--config-name",
        "training",
    ]
    with mps_session(cfg, log_dir, caps=caps) as environment:
        returncode = subprocess.run(command, env=environment).returncode
    return returncode if returncode >= 0 else 128 - returncode


def quota_levels(sm_by_percentage: dict[int, int]) -> list[int]:
    """Return one quota per measured SM count, using 100 for the full-SM limit."""
    levels = []
    previous = None
    for percentage, sms in sorted(sm_by_percentage.items()):
        if sms <= 0 or (previous is not None and sms < previous):
            raise ValueError("MPS SM limits must be positive and nondecreasing")
        if sms != previous:
            levels.append(percentage)
        elif percentage == 100:
            levels[-1] = percentage
        previous = sms
    return levels


def _probe(environment: dict[str, str], *, mps: bool = False) -> dict:
    command = [sys.executable, str(Path(__file__).with_name("probe.py"))]
    with subprocess.Popen(
        command,
        env=environment,
        text=True,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ) as process:
        try:
            if not select.select([process.stdout], [], [], 60)[0]:
                raise subprocess.TimeoutExpired(command, 60)
            result = process.stdout.readline()
            if result and mps:
                wait_for_clients(environment, {process.pid})
            # EOF releases the probe's CUDA contexts after client verification.
            _, stderr = process.communicate(timeout=60)
        except BaseException:
            process.kill()
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"CUDA SM probe failed: {stderr.strip()}")
        return json.loads(result)


def discover_resources(directory: Path, min_percentage: int) -> dict:
    """Measure SM limits using fresh CUDA contexts in probe subprocesses.

    The launcher does not create CUDA contexts. Stop the owned MPS daemon
    after probing, before training starts.
    """
    inventory = _probe(disabled_mps_environment(directory))["devices"]
    drivers = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).splitlines()
    driver = drivers[0].strip()
    signatures = {
        (d["name"], d["sms"], tuple(d["compute_capability"])) for d in inventory
    }
    if len(signatures) != 1:
        raise ValueError("MPS tuning currently requires homogeneous visible GPUs")
    physical_sms = inventory[0]["sms"]
    get_logger().info(
        f"MPS resource probe: {inventory[0]['name']}, "
        f"{physical_sms} SMs/GPU, {len(inventory)} GPUs, driver={driver}",
    )
    mapping = {}
    with mps_daemon(directory) as (environment, _):
        for percentage in range(min_percentage, 101):
            environment["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(percentage)
            devices = _probe(environment, mps=True)["devices"]
            limits = {device["sms"] for device in devices}
            if len(limits) != 1:
                raise RuntimeError("MPS probe returned inconsistent GPU limits")
            mapping[percentage] = limits.pop()
    if mapping[100] != physical_sms:
        raise RuntimeError("MPS probe limits disagree with native GPU SM count")
    levels = quota_levels(mapping)
    if len(levels) < 2:
        raise RuntimeError("MPS probe did not observe the requested SM reduction")
    result = {
        "gpu": inventory[0]["name"],
        "devices": inventory,
        "driver": driver,
        "physical_sms": physical_sms,
        "sm_by_percentage": mapping,
        "quota_levels": levels,
    }
    resource_path = directory / "resources.json"
    resource_path.write_text(json.dumps(result, indent=2))
    get_logger().info(
        f"MPS SM mapping: {min_percentage}% -> {mapping[min_percentage]} ... "
        f"100% -> {mapping[100]} SMs; {len(set(mapping.values()))} distinct limits; "
        f"saved to {resource_path}"
    )
    return result


def _check_measurement(measured: dict) -> None:
    values = [measured["window_s"], *measured["service_s"].values()]
    if any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError("Calibration requires finite positive window/service times")


def search_pair(
    measure: Callable[[dict[str, int]], dict],
    *,
    mps_baseline: dict,
    min_percentage: int,
    max_trials: int,
    sm_by_percentage: dict[int, int],
    levels: list[int],
) -> dict:
    """Search with a measured SM-granularity step and a coarse step three times larger.

    Inputs have already passed configuration and measurement validation.
    The supplied 100/100 baseline counts toward max_trials.
    """
    target = min(("rollout", "env"), key=mps_baseline["service_s"].get)
    baseline = {**mps_baseline, "caps": FULL_CAPS.copy()}
    cache = {100: baseline}
    sms = sorted(set(sm_by_percentage.values()))
    sm_step = min(upper - lower for lower, upper in zip(sms, sms[1:]))
    percentage_step = math.ceil(100 * sm_step / sm_by_percentage[100])

    def evaluate(quota):
        quota = max(level for level in levels if level <= quota)
        if quota in cache:
            return cache[quota]
        if len(cache) >= max_trials:
            return None
        caps = FULL_CAPS.copy()
        caps[target] = quota
        measured = {**measure(caps.copy()), "caps": caps}
        cache[quota] = measured
        return measured

    best_quota, best = 100, baseline
    quota = 100
    while quota > min_percentage:
        quota = max(min_percentage, quota - 3 * percentage_step)
        candidate = evaluate(quota)
        if candidate is None or candidate["window_s"] > best["window_s"]:
            break
        if candidate["window_s"] < best["window_s"]:
            best_quota, best = candidate["caps"][target], candidate

    while True:
        neighbors = []
        for quota in (best_quota - percentage_step, best_quota + percentage_step):
            if min_percentage <= quota <= 100:
                candidate = evaluate(quota)
                if candidate is not None and candidate["window_s"] < best["window_s"]:
                    neighbors.append((candidate["caps"][target], candidate))
        if not neighbors:
            break
        best_quota, best = min(neighbors, key=lambda pair: pair[1]["window_s"])

    return {
        "target": target,
        "best": best,
        "trials": list(cache.values()),
    }


def validate_pair(
    measure: Callable[[dict[str, int]], dict],
    search: dict,
    *,
    levels: list[int],
    reuse_search: bool = False,
) -> dict:
    """Compare the short winner and adjacent measured SM limits against 100/100.

    Full validation measures at most four distinct quotas and does not hill-climb.
    A neighbor is the nearest different measured SM count, not a fixed percentage.
    """
    target = search["target"]
    anchor = search["best"]["caps"][target]
    position = levels.index(anchor)
    candidates = [anchor] + [
        levels[index]
        for index in (position - 1, position + 1)
        if 0 <= index < len(levels)
    ]
    observations = {trial["caps"][target]: trial for trial in search["trials"]}
    cache = {}

    for quota in dict.fromkeys([*candidates, 100]):
        caps = FULL_CAPS.copy()
        caps[target] = quota
        measured = (
            observations[quota]
            if reuse_search and quota in observations
            else measure(caps.copy())
        )
        cache[quota] = {**measured, "caps": caps}

    best = min((cache[quota] for quota in candidates), key=lambda m: m["window_s"])
    baseline = cache[100]
    selected = best if best["window_s"] < baseline["window_s"] else baseline
    return {
        "relative_window_gain": 1 - selected["window_s"] / baseline["window_s"],
        "selected_caps": selected["caps"],
        "window_s": selected["window_s"],
    }


def validate_tuning(cfg: DictConfig) -> tuple[dict[str, int] | None, dict | None]:
    """Validate enabled MPS and return either fixed caps or automatic tuning settings."""
    mps = dict(cfg.get("mps") or {})
    percentages = {side: mps.get(f"{side}_percentage") for side in FULL_CAPS}
    if int(cfg.cluster.get("num_nodes", 1)) != 1:
        raise ValueError("Managed MPS requires a single node")
    names = [(cfg.get(side) or {}).get("group_name") for side in ("env", "rollout")]
    other_names = [
        (cfg.get(side) or {}).get("group_name") for side in ("actor", "reward")
    ]
    if names[0] == names[1] or any(name in other_names for name in names):
        raise ValueError("MPS worker group names must be distinct from other groups")
    if any(value is not None for value in percentages.values()):
        return {
            side: 100 if value is None else value for side, value in percentages.items()
        }, None
    if cfg.runner.get("use_training_pipeline", False) or cfg.runner.get(
        "only_eval", False
    ):
        raise ValueError(
            "Automatic MPS tuning measures synchronous rollouts; training pipeline and only_eval require fixed MPS quotas"
        )
    if (cfg.get("reward") or {}).get("use_reward_model", False):
        raise ValueError(
            "Automatic MPS tuning does not run compute_rewards; reward workers require fixed MPS quotas"
        )
    if cfg.rollout.pipeline_stage_num < 2:
        raise ValueError(
            "MPS pair tuning requires at least two pipeline stages for overlap"
        )
    placements = {}
    for components, placement in cfg.cluster.component_placement.items():
        for component in components.split(","):
            placements[component.strip()] = placement
    if placements.get("rollout") is None or placements.get("rollout") != placements.get(
        "env"
    ):
        raise ValueError(
            "MPS tuning requires matching env/rollout placement for colocated pairs"
        )
    return None, dict(
        OmegaConf.merge(
            OmegaConf.load(Path(__file__).with_name("defaults.yaml")),
            mps.get("auto_tuning") or {},
        )
    )


def trial_config(cfg, directory: Path, settings: dict):
    """Create a disposable, resolved calibration config without mutating training."""
    trial = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
    with open_dict(trial):
        trial.mps = {"enabled": False, "trial": True, "auto_tuning": settings}
        trial.cluster.profiling = None
        trial.cluster.tracer = {"enable": False}
        trial.runner.logger.log_path = str(directory)
        trial.runner.logger.logger_backends = []
        trial.runner.per_worker_log = False
        trial.runner.val_check_interval = -1
        trial.runner.save_interval = -1
        trial.runner.max_steps = (
            settings["warmup_windows"] + settings["measure_windows"]
        )
        if settings.get("window_chunks") is not None:
            trial.env.train.max_steps_per_rollout_epoch = (
                settings["window_chunks"] * trial.actor.model.num_action_chunks
            )
        trial.hydra = {
            "run": {"dir": str(directory)},
            "output_subdir": None,
            "job": {"chdir": False},
        }
    # Resolve output-path interpolations after assigning the private trial path.
    OmegaConf.resolve(trial)
    return trial


def tune_before_training(cfg, log_dir: Path, training_script: Path) -> dict | None:
    """Return worker quotas, or None to train without MPS."""
    caps, settings = validate_tuning(cfg)
    if settings is None:
        get_logger().info(f"Using configured MPS quotas {caps}; skipping calibration")
        return caps
    full_chunks = (
        cfg.env.train.max_steps_per_rollout_epoch // cfg.actor.model.num_action_chunks
    )
    if settings["window_chunks"] is not None:
        settings["window_chunks"] = min(settings["window_chunks"], full_chunks)
    root = log_dir / "mps_tuning"
    root.mkdir(exist_ok=True)
    resources = None

    index = 0

    def measure(caps, *, phase="search", trial_settings=None, use_mps=True):
        nonlocal index
        index += 1
        effective_sms = (
            {side: resources["sm_by_percentage"][cap] for side, cap in caps.items()}
            if resources is not None
            else None
        )
        label = "-".join(f"{side}{cap}" for side, cap in caps.items())
        directory = root / f"{index:02d}-{phase}-{label}"
        directory.mkdir()
        config = trial_config(cfg, directory, trial_settings or settings)
        OmegaConf.save(config, directory / "input.yaml")
        command = [
            sys.executable,
            str(training_script),
            "--config-path",
            str(directory),
            "--config-name",
            "input",
        ]
        with (
            mps_session(
                config, directory, caps=caps if use_mps else None
            ) as environment,
            (directory / "run.log").open("w") as log,
        ):
            with subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
            ) as process:
                try:
                    returncode = process.wait(timeout=settings["trial_timeout_s"])
                except subprocess.TimeoutExpired as error:
                    raise TimeoutError(
                        f"MPS trial exceeded {settings['trial_timeout_s']}s; "
                        f"inspect {directory / 'run.log'}"
                    ) from error
                finally:
                    if process.poll() is None:
                        # Allow cleanup after SIGTERM before forcing the driver to exit.
                        process.terminate()
                        try:
                            process.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.wait()
            if returncode:
                raise RuntimeError(
                    f"MPS trial exited {returncode}; inspect {directory / 'run.log'}"
                )
        measured = json.loads((directory / "measurement.json").read_text())
        _check_measurement(measured)
        get_logger().info(
            f"Calibration trial {index} ({phase}) finished: use_mps={use_mps}, "
            f"caps={caps}, SMs={effective_sms}; window={measured['window_s']:.3f}s, "
            f"service rollout={measured['service_s']['rollout']:.3f}s/"
            f"env={measured['service_s']['env']:.3f}s; log: {directory / 'run.log'}",
        )
        return measured

    native_baseline = measure(FULL_CAPS.copy(), phase="native", use_mps=False)
    baseline = measure(FULL_CAPS.copy())
    if baseline["window_s"] > native_baseline["window_s"]:
        get_logger().info(
            f"MPS 100/100 window {baseline['window_s']:.3f}s exceeds "
            f"native {native_baseline['window_s']:.3f}s; training without MPS."
        )
        return None
    resources = discover_resources(root / "resource_probe", settings["min_percentage"])
    search = search_pair(
        measure,
        mps_baseline=baseline,
        min_percentage=settings["min_percentage"],
        max_trials=settings["max_trials"],
        sm_by_percentage=resources["sm_by_percentage"],
        levels=resources["quota_levels"],
    )
    validation_settings = {
        **settings,
        "window_chunks": None,
        "warmup_windows": settings["validation_warmup_windows"],
        "measure_windows": settings["validation_windows"],
    }

    validation = validate_pair(
        lambda caps: measure(
            caps, phase="validation", trial_settings=validation_settings
        ),
        search,
        levels=resources["quota_levels"],
        reuse_search=(
            settings["window_chunks"] in (None, full_chunks)
            and settings["warmup_windows"] >= settings["validation_warmup_windows"]
            and settings["measure_windows"] >= settings["validation_windows"]
        ),
    )
    caps = validation["selected_caps"]
    get_logger().info(
        f"MPS calibration selected {caps}; window={validation['window_s']:.3f}s, "
        f"gain vs MPS 100/100={validation['relative_window_gain']:.2%}"
    )
    return caps
