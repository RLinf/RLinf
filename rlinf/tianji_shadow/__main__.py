"""Monitor sidecar startup and offline verification."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import threading
import time


def supervise(command, variant):
    root = Path(os.environ["TIANJI_RLT_OUTPUT"]).resolve()
    with tempfile.TemporaryDirectory(prefix="tjs1113-", dir="/tmp") as temp:
        env = dict(os.environ, TIANJI_SHADOW_RUNTIME=temp)
        if env.get("TIANJI_SHADOW_DEFAULT") == "on":
            Path(temp, "enabled").touch()
        shadow_env = dict(env, CUDA_VISIBLE_DEVICES=os.environ.get("TIANJI_SHADOW_GPU", "1"))
        log = root / "monitor_shadow_runtime.log"
        process = None
        try:
            with log.open("a") as out:
                assets = os.environ.get("TIANJI_SHADOW_ASSETS", str(Path.home() / "tianji_monitor_assets/shadow1113"))
                process = subprocess.Popen([
                    sys.executable, "-B", "-m", "rlinf.tianji_shadow", "worker",
                    temp, str(root / "monitor_shadow"), assets, env.get("TIANJI_RLT_STAGE1_MODEL_PATH", ""),
                    variant,
                ], env=shadow_env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
        except (OSError, ValueError) as exc:
            print(f"Monitor sidecar could not start: {exc}", file=sys.stderr)
            Path(temp, "status.json").write_text(json.dumps(dict(state="error", error=str(exc))))
        child = None

        def forward(signum, _):
            if child is not None and child.poll() is None:
                try:
                    os.kill(child.pid, signum)
                except ProcessLookupError:
                    pass

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, forward)
        try:
            child = subprocess.Popen(command, env=env)
            while child.poll() is None:
                if process is not None and process.poll() is not None:
                    status = Path(temp, "status.json")
                    try:
                        old = json.loads(status.read_text())
                    except (OSError, ValueError):
                        old = {}
                    if old.get("state") != "error":
                        status.write_text(json.dumps(dict(state="error", error=f"exit {process.returncode}")))
                    process = None
                time.sleep(.1)
            return child.returncode
        finally:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def worker(runtime, output, assets, policy, variant):
    stopped = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stopped.set())
    try:
        from .worker import run
        run(Path(runtime), Path(output), Path(assets), policy, variant, stopped)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        status = Path(runtime, "status.json")
        if status.parent.is_dir():
            status.write_text(json.dumps(dict(state="error", error=f"{type(exc).__name__}: {exc}")))
        return 1
    return 0


def verify(assets, device):
    import numpy as np
    from .inference import Predictor
    from .rules import RISK, THRESHOLDS, TASKS
    from .parallel import PROFILES, Track, VERSION
    predictor = Predictor(assets, device=device)
    profiles = PROFILES if predictor.identity.get("version") == VERSION else {
        "published_two": {"thresholds": THRESHOLDS, "confirm_points": 2}
    }
    risk_threshold = predictor.identity["risk"]
    results = {}
    for path in sorted(p for p in Path(assets).glob("*.npz") if p.stem.startswith(("fixture_", "alarm_"))):
        cond = path.stem
        data = np.load(path, allow_pickle=False)
        out = [predictor({k: data[k][i:i+8] for k in ("rlt","state","valid","task","rgb","refs","meta")})[0]
               for i in range(0,len(data["task"]),8)]
        errors = {}
        for name, expected, tolerance in [
            ("progress", "expect_progress", 1e-4), ("risk", "expect_risk", 1e-5),
            ("logit", "expect_logit", 1e-3), ("hidden", "expect_hidden", 1e-4),
            ("base", "expect_base_logit", 1e-4),
        ]:
            actual = np.concatenate([o[name] for o in out])
            want = data[expected]
            errors[name] = float(np.max(np.abs(actual-want)))
            np.testing.assert_allclose(actual, want, rtol=0, atol=tolerance)
        progress = np.concatenate([o["progress"] for o in out])
        risk = np.concatenate([o["risk"] for o in out])
        completions = dict.fromkeys(profiles, 0)
        alarms = 0
        for offset in range(0, len(progress), 8):
            group = slice(offset, offset+8)
            stage = TASKS.index(int(data["task"][offset])) + 1
            expected_progress = data["expect_progress"][group]
            for name, rule in profiles.items():
                track = Track(rule)
                triggered = [i for i, value in enumerate(progress[group])
                             if track.confirm(stage, stage, float(value))]
                above = expected_progress > rule["thresholds"][stage-1]
                if rule["confirm_points"] == 1:
                    hits = np.flatnonzero(above)
                else:
                    hits = np.flatnonzero(above[1:] & above[:-1]) + 1
                expected = hits[:1].tolist()
                assert triggered == expected, (cond, offset, name, triggered, expected)
                completions[name] += len(triggered)
            def risk_events(values):
                active = False
                found = []
                for i, v in enumerate(values):
                    if v > risk_threshold and not active:
                        found.append(i)
                    if v != risk_threshold:
                        active = v > risk_threshold
                return found
            actual = risk_events(risk[group])
            assert actual == risk_events(data["expect_risk"][group]), (cond, offset, actual)
            alarms += len(actual)
        results[cond] = dict(samples=len(progress), phases=sorted(set(data["phase"].tolist())),
                             max_abs_error=errors, completion=completions, alarms=alarms)
    assert {"fixture_none", "fixture_both"} <= results.keys(), "missing locked fixtures"
    print(json.dumps(results, indent=2))
    return 0


def replay(directory, assets, device):
    import numpy as np
    from .inference import Predictor
    from .rules import Evidence
    predictor = Predictor(assets, device=device)
    report = {}
    fields = ("progress", "risk", "logit")
    for path in sorted(Path(directory).rglob("observations.jsonl")):
        total = 0
        maximum = dict.fromkeys(fields, 0.)
        notes = json.loads((path.parent / "identity.json").read_text())
        if notes["model"]["assets"] != predictor.identity["assets"]:
            raise ValueError(f"Input model identity differs: {path.parent}")
        epochs, won = {}, set()
        actual_events = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["result"] != "ok":
                continue
            with np.load(path.parent / row["input"], allow_pickle=False) as packet:
                for prefix, key in (("a_", "actual"), ("b_", "shadow")):
                    if row[key] is None:
                        continue
                    x = {k[2:]: packet[k] for k in packet.files if k.startswith(prefix)}
                    scores = {n: float(v[0]) for n,v in predictor(x)[0].items() if n in fields}
                    for n in fields:
                        maximum[n] = max(maximum[n], abs(scores[n] - row[key][n]))
                    stage = row["stage"] if key=="actual" else row["shadow_stage"]
                    timeline = "actual" if key=="actual" else "independent"
                    occurrence = row["revision"] if key=="actual" else stage
                    epoch = row.get(key+"_epoch")
                    complete = alarm = False
                    if epoch is not None:
                        if epochs.get(key, (-1,None))[0] != epoch:
                            epochs[key] = (epoch, Evidence())
                        complete, alarm = epochs[key][1].score(stage, scores["progress"], scores["risk"])
                        if complete:
                            prior = (timeline, occurrence) in won
                            won.add((timeline, occurrence))
                            complete = not prior
                    for event, active in (("model_h", alarm), ("complete", complete)):
                        if active:
                            actual_events.append((timeline, event, stage, row["oid"]))
                    print(json.dumps(dict(episode=row["episode"], oid=row["oid"], timeline=timeline,
                                          stage=stage, complete=complete, model_h=alarm, **scores)))
            total += 1
        original = [json.loads(l) for l in (path.parent / "predictions.jsonl").read_text().splitlines()]
        expected = [(x["timeline"], x["name"], x["stage"], x["oid"]) for x in original if "oid" in x]
        report[str(path.parent)] = dict(observations=total,max_abs_error=maximum,
                                        model_events=len(actual_events), same_model_events=expected == actual_events)
    print(json.dumps(report,indent=2),file=sys.stderr)
    if not report:
        return 1
    if device.startswith("cuda"):
        return int(any(not p["same_model_events"] or p["max_abs_error"]["progress"] > 1e-4
                       or p["max_abs_error"]["risk"] > 1e-5 for p in report.values()))
    return 0


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command",required=True)
    sup = sub.add_parser("supervise")
    sup.add_argument("variant")
    sup.add_argument("args",nargs=argparse.REMAINDER)
    w = sub.add_parser("worker")
    for arg in ("runtime","output","assets","policy","variant"):
        w.add_argument(arg)
    for name in ("verify","replay"):
        p = sub.add_parser(name)
        if name == "replay":
            p.add_argument("directory")
        p.add_argument("--assets",default=str(Path.home() / "tianji_monitor_assets/shadow1113"))
        p.add_argument("--device",default="cpu")
    dual = sub.add_parser("replay1116")
    dual.add_argument("--output", required=True)
    dual.add_argument("--thresholds", help="JSON override for two progress profiles")
    dual.add_argument("--assets", default=str(Path.home() / "tianji_monitor_assets/shadow1116"))
    dual.add_argument("--device", default="cuda:0")
    dual.add_argument("sources", nargs="+")
    args = parser.parse_args()
    if args.command == "replay1116":
        from .replay_1116 import replay_1116
        return replay_1116(args.sources, args.output, args.assets, args.device, args.thresholds)
    if args.command == "supervise":
        return supervise(args.args, args.variant)
    if args.command == "worker":
        return worker(args.runtime,args.output,args.assets,args.policy,args.variant)
    if args.command == "verify":
        return verify(args.assets,args.device)
    return replay(args.directory,args.assets,args.device)


if __name__ == "__main__":
    sys.exit(main())
