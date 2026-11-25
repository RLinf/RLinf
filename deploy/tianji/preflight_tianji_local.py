"""Read-only local deployment checks, including deferred GPU/model requirements."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

LAUNCHER_DIR = Path(__file__).resolve().parent


def _rlinf_root() -> Path:
    return Path(os.environ.get("RLINF_ROOT", LAUNCHER_DIR.parents[1])).resolve()


def compose_configs(rlinf_root: Path) -> dict:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    result = {}
    directory = rlinf_root / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
        for mode, name in (
            ("async", "tianji2_rlt_stage2_async"),
            ("dagger", "tianji_dagger_collect"),
        ):
            if not (directory / f"{name}.yaml").is_file():
                continue
            cfg = compose(config_name=name)
            train = OmegaConf.to_container(cfg.env.train.override_cfg, resolve=True)
            evaluation = OmegaConf.to_container(cfg.env.eval.override_cfg, resolve=True)
            if not (
                train["backend"] == "loopinfra"
                and train["control_profile"] == "wuji_compat"
                and train["read_only"] is False
                and train["hil_control"] is True
                and train["manual_episode_control_only"] is True
                and type(train["hil_port"]) is int
                and evaluation["read_only"] is True
                and (
                    cfg.actor.model.action_dim == 20
                    or (
                        train.get("action_space") == "joint"
                        and evaluation.get("action_space") == "joint"
                        and cfg.actor.model.action_dim == 16
                        and cfg.actor.model.get("critic_action_dim") == 20
                    )
                )
            ):
                raise RuntimeError("Invalid local HIL configuration: " + name)
            chunk = int(cfg.actor.model.num_action_chunks)
            prefetch_at = int(cfg.rollout.prefetch_at)
            offset = int(os.environ.get("RLT_REF_OFFSET", "0"))
            row_scale = float(os.environ.get("TIANJI_ROW_SCALE", "") or 1.0)
            dense = (os.environ.get("TIANJI_DENSE_ROWS") or "").strip()
            if row_scale == 1.0:
                dense = "0"
            if dense not in ("", "0") and not (
                dense == "1"
                and row_scale in (2.0, 1.5)
                and train.get("action_space") == "joint"
                and (
                    os.environ.get("TIANJI_DEPLOY_ACTOR_UPDATES") == "0"
                    or bool(cfg.actor.model.get("executor_action_grid", False))
                )
                and os.environ.get("LOOPINFRA_RTJ_POLICY_LIMITS") == "x2"
            ):
                raise RuntimeError(
                    "TIANJI_DENSE_ROWS=1 needs TIANJI_ROW_SCALE=2 or 1.5, joint Stage2, "
                    "an aligned grid for actor deployment, and LOOPINFRA_RTJ_POLICY_LIMITS=x2: "
                    + name
                )
            sys.path.insert(0, str(rlinf_root))
            ag_spec = importlib.util.spec_from_file_location(
                "tianji_preflight_action_grid",
                rlinf_root / "rlinf/algorithms/rlt/action_grid.py",
            )
            ag_mod = importlib.util.module_from_spec(ag_spec)
            ag_spec.loader.exec_module(ag_mod)
            ActionGrid = ag_mod.ActionGrid
            grid = ActionGrid.from_config(cfg.actor.model)
            if grid.aligned and (
                grid.offset != offset
                or grid.row_scale != row_scale
                or grid.dense != (dense == "1")
            ):
                raise RuntimeError("actor/critic action grid differs from executor schedule")
            if (
                row_scale != 1
                and bool(cfg.runner.get("deploy_actor_updates", True))
                and not (grid.aligned and grid.dense)
            ):
                raise RuntimeError("scaled residual deployment requires aligned dense targets")
            if not (
                train["stream_chunks"] is True
                and offset == round(row_scale * (chunk - 1 - prefetch_at))
                and offset + row_scale * chunk
                <= int(cfg.actor.model.ref_num_action_chunks) + 1e-9
                and float(train["step_frequency"])
                == float(os.environ.get("TIANJI_CONTROL_HZ", "10"))
            ):
                raise RuntimeError(
                    "Stage2 prefetch timetable mismatch; source deploy/tianji/tianji_local.env: "
                    + name
                )
            result[mode] = {
                "config": name,
                "server_url": train["server_url"],
                "hil_port": train["hil_port"],
                "train_backend": train["backend"],
                "control_profile": train["control_profile"],
                "eval_read_only": evaluation["read_only"],
                "component_placement": OmegaConf.to_container(
                    cfg.cluster.component_placement
                ),
                "control_hz": train["step_frequency"],
                "rl_chunk": chunk,
                "target_chunk": grid.target_steps,
                "target_hz": float(train["step_frequency"]) * grid.rate_ratio,
                "request_after_steps": prefetch_at + 1,
                "ref_offset": offset,
                "subtask_prompts_file": train.get("subtask_prompts_file"),
                "subtask_heads": int(cfg.actor.model.num_stage_actors),
                "action_space": train.get("action_space", "tcp"),
            }
    return result


def subtask_status(mode: str | None = None) -> dict:
    """The run's subtask prompts, or the single global prompt."""
    path = os.environ.get("TIANJI_SUBTASK_PROMPTS", "")
    if not path:
        return {
            "subtasks": 0,
            "prompt": os.environ.get("TIANJI_TASK_PROMPT") or "global task_description",
        }
    from rlinf.envs.realworld.tianji.tianji_subtasks import load_subtask_prompts

    car = os.environ.get("TIANJI_CAR_1118") == "1"
    car_subtasks = int(os.environ.get("TIANJI_CAR_NUM_SUBTASKS", "15"))
    raw_max = os.environ.get("TIANJI_MAX_SUBTASKS", "").strip()
    if raw_max:
        limit = int(raw_max)
    elif car:
        limit = car_subtasks
    elif mode == "stage1" or os.environ.get("TIANJI_STAGE1_VLA_ONLY") == "1":
        limit = 32
    else:
        limit = 4
    prompts = load_subtask_prompts(path, limit=limit)
    if car and len(prompts) != car_subtasks:
        raise ValueError(f"1118v1 requires all {car_subtasks} car prompts")
    return {"file": path, "subtasks": len(prompts), "prompts": list(prompts)}


def live_status(url: str, profile: str = "wuji_compat", joint: bool = False) -> dict:
    def get(path):
        token = os.environ.get("LOOPINFRA_TOKEN")
        request = urllib.request.Request(
            url.rstrip("/") + path,
            headers={"Authorization": "Bearer " + token} if token else {},
        )
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    status = get("/api/status?control=1")
    if status.get("execution_metadata_version") != 2:
        raise RuntimeError(
            "Restart the updated LoopInfra collector pair: fast execution metadata v2 unavailable"
        )
    control = status.get("robot_control") or {}
    if (
        status.get("policy_execution_version") != 1
        or control.get("policy_execution_protocol")
        != "loopinfra.policy_execution_observation.v1"
    ):
        raise RuntimeError(
            "Restart both updated LoopInfra peers: measured policy execution snapshots unavailable"
        )
    if (
        status.get("policy_motion_limits_version") != 1
        or control.get("policy_motion_limits_protocol")
        != "loopinfra.policy_motion_limits.v1"
    ):
        raise RuntimeError(
            "Restart both updated LoopInfra peers: teleop policy motion limits unavailable"
        )
    if not (
        status.get("control_status_version") == 1
        and status.get("connected")
        and not status.get("stale", True)
        and status.get("awaiting_generation") is None
        and control.get("mode") == "eef_position"
        and not control.get("transition")
        and not control.get("last_error")
        and not status.get("manual")
        and not (control.get("teleop") or {}).get("active")
    ):
        raise RuntimeError(
            "Collector must be connected, fresh and idle in Position/EEF; release Pico"
        )
    if joint:
        from rlinf.envs.realworld.tianji.tianji_backend import JOINT_ROWS_PROTOCOL

        if control.get("policy_chunk_joint_rows") != JOINT_ROWS_PROTOCOL:
            raise RuntimeError(
                "Joint Stage2 needs joint chunk rows: restart the collector pair from $LOOPINFRA_ROOT"
            )
    from rlinf.envs.realworld.tianji.tianji_backend import LoopInfraPolicyBackend

    backend = LoopInfraPolicyBackend(
        url,
        os.environ.get("LOOPINFRA_TOKEN"),
        read_only=False,
        control_profile=profile,
        reference_windows=True,
    )
    backend.validate_control_profile(status)
    collection = get("/api/collection")
    strategy = collection.get("control_strategy") or {}
    policies = strategy.get("client_policies") or {}
    expected = {
        "speed_profile": "trial45",
        "hold_policy": "sdk_error_only_v1",
        "contact_assist": "sdk_cartesian_impedance_position_v7",
        "policy_pose_contact": "finite_cartesian_impedance_v1",
        "feedback_recovery": "feedback_brake_v1",
        "sync_frequency_hz": int(float(os.environ.get("TIANJI_CONTROL_HZ", "10"))),
    }
    actual = {name: strategy.get(name) for name in ("speed_profile", "hold_policy")}
    actual.update(
        {
            name: policies.get(name)
            for name in ("contact_assist", "policy_pose_contact", "feedback_recovery")
        }
    )
    actual["sync_frequency_hz"] = collection.get("sync_frequency_hz")
    if actual != expected:
        raise RuntimeError(
            f"Collector launch parameters mismatch; no motion sent: {actual}"
        )
    warnings = []
    if control.get("policy_stream_protocol") != "loopinfra.policy_reference_stream.v1":
        warnings.append("collector lacks continuous policy windows; restart both peers")
    if control.get("policy_feedback_gap_policy") != "bounded_pause_100ms_v2":
        warnings.append("collector lacks policy feedback-gap pause; restart both peers")
    return {
        "ready": True,
        "controller_parameters": actual,
        "profile": profile,
        "pose_config_id": control["pose_config_id"],
        "policy_stream": control.get("policy_stream_protocol"),
        "feedback_gap_policy": control.get("policy_feedback_gap_policy"),
        "warnings": warnings,
    }


def xense_status(url: str, timeout: float = 3.0) -> dict:
    """GET-only: every selected Xense role is live and valid in a policy snapshot."""
    from rlinf.envs.realworld.tianji.tianji_backend import (
        LoopInfraStage1ObservationReader,
    )
    from rlinf.utils.xense_tactile import TACTILE_ROLES, WIRE_ROLES

    token = os.environ.get("LOOPINFRA_TOKEN")
    request = urllib.request.Request(
        url.rstrip("/") + "/api/collection",
        headers={"Authorization": "Bearer " + token} if token else {},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        xense = json.load(response).get("xense") or {}
    if not xense.get("enabled"):
        raise RuntimeError(
            "Tactile Stage2 needs Xense: stop the collector normally and start Stage2 "
            "again without --disable-xense"
        )
    sensors = xense.get("sensors")
    if not isinstance(sensors, dict):
        raise RuntimeError(f"Xense owner is not running on the collector: {xense.get('error')}")
    selected = [role for role in WIRE_ROLES if (sensors.get(role) or {}).get("serial")]
    if not selected:
        raise RuntimeError(
            "No Xense sensor is selected; choose the gripper sensors on the collector page"
        )
    if set(selected) != set(TACTILE_ROLES):
        raise RuntimeError(
            "Collector Xense roles "
            + ",".join(selected)
            + " differ from the model's "
            + ",".join(TACTILE_ROLES)
            + "; fix the role selection on the collector page"
        )
    selected = list(TACTILE_ROLES)
    waiting = [
        f"{role}={sensors[role].get('error') or 'not_ready'}"
        for role in selected
        if not (sensors[role].get("connected") and sensors[role].get("valid"))
    ]
    if waiting or not xense.get("ready"):
        raise RuntimeError(
            "Xense sensors not ready: " + (", ".join(waiting) or "stale samples")
        )
    try:
        observation = LoopInfraStage1ObservationReader(
            url, token, timeout, tactile_enabled=True, tactile_raw=True
        ).observe()
    except (KeyError, TypeError) as exc:
        raise RuntimeError(f"tactile snapshot unreadable: {exc!r}") from exc
    valid = observation.tactile["tactile_valid"]
    missing = [role for role in selected if not valid[TACTILE_ROLES.index(role)]]
    if missing:
        raise RuntimeError(
            "Policy tactile snapshot lacks selected roles: " + ", ".join(missing)
        )
    raw = observation.tactile_raw or {}
    if raw.get("error"):
        raise RuntimeError(
            "Full-resolution tactile snapshot unavailable ("
            + raw["error"]
            + "); restart the collector from $LOOPINFRA_ROOT"
        )
    return {
        "roles": selected,
        "observation_id": observation.observation_id,
        "raw_shapes": {r["role"]: r.get("shape") for r in raw.get("sensors", [])},
    }


def pi05_service_status(rlinf_root: Path) -> dict:
    """TIANJI_PI05_SERVICE: the resident VLA service's weights and state (GET-only)."""
    import pickle
    import socket
    import struct

    switch = (os.environ.get("TIANJI_PI05_SERVICE") or "").strip()
    report = {"enabled": switch not in ("", "0")}
    if not report["enabled"]:
        return report
    model_default = os.environ.get(
        "TIANJI_MODEL_DIR", str(rlinf_root / "models/stage1_v1")
    )
    model = Path(
        os.environ.get("TIANJI_RLT_STAGE1_MODEL_PATH", model_default)
    ).expanduser()
    candidates = (
        "actor/model_state_dict/full_weights.pt",
        "model_state_dict/full_weights.pt",
        "full_weights.pt",
    )
    source = next((model / p for p in candidates if (model / p).is_file()), None)
    if source is None:
        raise RuntimeError("Stage1 full_weights.pt missing under " + str(model))
    weights = source.parent / "full_weights.bf16.safetensors"
    if not weights.is_file():
        raise RuntimeError(
            f"{weights} missing: run converter for {model} in a stop window"
        )
    try:
        side = json.loads(Path(str(weights) + ".json").read_text())
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"{weights}.json sidecar unreadable ({exc}); re-run the converter"
        ) from exc
    st = weights.stat()
    if side.get("size") != st.st_size or side.get("mtime") != st.st_mtime:
        raise RuntimeError(
            f"{weights} changed since its sidecar was written; re-run the converter"
        )
    report.update(
        {
            "weights": str(weights),
            "weights_sha256": side.get("sha256"),
            "source": str(source),
        }
    )
    sock_path = os.environ.get("TIANJI_PI05_SOCKET") or os.path.join(
        os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}",
        "tianji-pi05.sock",
    )

    def recv_exact(sock, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = sock.recv(n - len(buf))
            if not chunk:
                raise OSError("service closed the connection")
            buf += chunk
        return bytes(buf)

    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(3.0)
            sock.connect(sock_path)
            data = pickle.dumps({"op": "info"}, protocol=5)
            sock.sendall(struct.pack("!Q", len(data)) + data)
            (length,) = struct.unpack("!Q", recv_exact(sock, 8))
            info = pickle.loads(recv_exact(sock, length))
    except OSError:
        report["service"] = "not running (the Stage2 launcher starts it)"
        return report
    identity = info.get("identity") or {}
    report["service"] = {
        "ready": info.get("ready"),
        "phase": info.get("phase"),
        "error": info.get("error"),
        "fast_mode": info.get("fast_mode"),
        "pid": info.get("pid"),
        "weights_sha256": identity.get("weights_sha256"),
        "code_sha256": identity.get("code_sha256"),
    }
    if info.get("ready") and identity.get("weights_sha256") != side.get("sha256"):
        report["service"]["note"] = "serves other weights; the launcher restarts it"
    return report


def gpu_status(python: str, mode: str) -> dict:
    packages = [
        "torch",
        "openpi",
        "transformers",
        "sentencepiece",
        "safetensors",
        "scipy",
    ]
    if mode.startswith("stage2"):
        packages += ["ray", "torchdata", "accelerate", "tensorboard", "pyarrow"]
    code = (
        "import importlib.util,json; "
        f"missing=[n for n in {packages!r} if importlib.util.find_spec(n) is None]; "
        "assert not missing, 'Missing GPU runtime packages: '+str(missing); "
        f"[__import__(n) for n in {packages!r}]; "
        "import torch; "
        "assert torch.cuda.is_available(), 'CUDA is not available'; "
        "print(json.dumps({'visible_gpus':torch.cuda.device_count(),'torch':torch.__version__}))"
    )
    completed = subprocess.run(
        [python, "-c", code], capture_output=True, text=True, timeout=60, check=False
    )
    if completed.returncode:
        raise RuntimeError(completed.stderr.strip())
    result = json.loads(completed.stdout)
    minimum = 2 if mode == "stage2-async" else 1
    if result["visible_gpus"] < minimum:
        raise RuntimeError(f"{mode} requires {minimum} visible GPU(s)")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("offline", "stage1", "stage2-async", "stage2-dagger", "stage2-infer"),
        default="offline",
    )
    parser.add_argument(
        "--live", action="store_true", help="also GET current collector readiness"
    )
    args = parser.parse_args()
    rlinf_root = _rlinf_root()
    os.environ.setdefault("EMBODIED_PATH", str(rlinf_root / "examples/embodiment"))
    os.environ["RLINF_TIANJI_STAGE1_ONLY"] = "1"
    if str(rlinf_root) not in sys.path:
        sys.path.insert(0, str(rlinf_root))
    report = {
        "mode": args.mode,
        "rlinf_root": str(rlinf_root),
        "checks": {},
        "errors": [],
        "pending": [],
    }

    def check(name, operation):
        try:
            report["checks"][name] = operation()
        except Exception as exc:  # noqa: BLE001 -- collect independent check failures
            report["errors"].append(name + ": " + str(exc))

    check("hydra_configs", lambda: compose_configs(rlinf_root))
    check("subtasks", lambda: subtask_status(args.mode))
    check("pi05_service", lambda: pi05_service_status(rlinf_root))
    check(
        "timetable",
        lambda: {
            key: os.environ.get(key)
            for key in (
                "TIANJI_ACTION_HORIZON",
                "TIANJI_CONTROL_HZ",
                "TIANJI_STAGE1_PREFETCH_STEP",
                "TIANJI_STAGE2_CHUNK",
                "TIANJI_STAGE2_PREFETCH_AT",
                "RLT_REF_OFFSET",
            )
        },
    )
    infra = Path(os.environ.get("LOOPINFRA_ROOT", str(Path.home() / "LoopInfra")))

    def sdk_config():
        python = (
            os.environ.get("LOOPINFRA_PYTHON")
            or (infra / ".loopinfra-python").read_text().strip()
        )
        command = [
            python,
            str(LAUNCHER_DIR / "inspect_loopinfra_local.py"),
            str(infra),
            "--speed",
            os.environ.get("TIANJI_WUJI_SPEED", "trial45"),
        ]
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip())
        return json.loads(completed.stdout)

    check("loopinfra_config", sdk_config)
    if args.live:
        check(
            "live_collector",
            lambda: live_status(
                os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780"),
                joint=any(
                    mode.get("action_space") == "joint"
                    for mode in (report["checks"].get("hydra_configs") or {}).values()
                ),
            ),
        )
        if os.environ.get("TIANJI_TACTILE_REQUIRED") == "1":
            check(
                "live_xense",
                lambda: xense_status(
                    os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780")
                ),
            )
    else:
        report["pending"].append(
            "Live collector, RGB, robot feedback and real motion have not been checked"
        )
    if args.mode == "offline":
        report["pending"].append(
            "GPU environment and real-model/training checks are not run in offline mode"
        )
    else:
        py_bin = (
            os.environ.get("RLINF_PYTHON")
            or os.environ.get("TIANJI_GPU_PYTHON")
            or os.environ.get("PYTHON_BIN")
            or sys.executable
        )
        check("gpu_runtime", lambda: gpu_status(py_bin, args.mode))

        def model_assets():
            model_default = os.environ.get(
                "TIANJI_MODEL_DIR", str(rlinf_root / "models/stage1_v1")
            )
            model = Path(
                os.environ.get("TIANJI_RLT_STAGE1_MODEL_PATH", model_default)
            )
            stats = Path(
                os.environ.get(
                    "TIANJI_RLT_STAGE1_NORM_STATS_PATH", str(model / "norm_stats.json")
                )
            )
            candidates = [
                "model.safetensors",
                "full_weights.pt",
                "model_state_dict/full_weights.pt",
                "actor/model_state_dict/full_weights.pt",
            ]
            if not any((model / p).is_file() for p in candidates):
                raise RuntimeError("Stage1 checkpoint missing under " + str(model))
            if not isinstance(json.loads(stats.read_text()), dict):
                raise TypeError("Invalid Stage1 normalization JSON")
            return {"model": str(model), "norm_stats": str(stats)}

        check("model_assets", model_assets)
    report["ready"] = not report["errors"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
