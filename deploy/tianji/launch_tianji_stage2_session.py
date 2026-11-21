"""Run Stage2 (Online RL or DAgger) and open its HIL panel once Start can be pressed."""

import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid


def panel_status(url: str) -> dict:
    """Read local status without proxying local requests or sending commands."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url + "/api/status", timeout=1) as response:
        return json.load(response)


def vla_and_panel_ready(flag: Path, since: float, launch_id: str, url: str) -> bool:
    """This launch's VLA weights are loaded and its panel waits for Start."""
    try:
        if flag.stat().st_mtime < since:
            return False
        status = panel_status(url)
        return (
            status.get("stage2_launch_id") == launch_id
            and status.get("run_mode") in ("training", "dagger")
            and status.get("phase") == "waiting"
        )
    except (OSError, ValueError, urllib.error.URLError):
        return False


def training_ready(ready_file: Path, launch_id: str) -> bool:
    """Initialization and the first weight sync of this launch are done."""
    try:
        return json.loads(ready_file.read_text()).get("launch_id") == launch_id
    except (OSError, ValueError):
        return False


def xense_ready(collector_url: str) -> str | None:
    """None when every selected Xense role is live in a policy snapshot, else why not."""
    launcher = Path(__file__).resolve().parent
    rlinf_root = Path(os.environ.get("RLINF_ROOT", launcher.parents[1]))
    for path in (str(launcher), str(rlinf_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ["RLINF_TIANJI_STAGE1_ONLY"] = "1"
    from preflight_tianji_local import xense_status

    try:
        xense_status(collector_url)
    except (OSError, RuntimeError, ValueError) as exc:
        return str(exc)
    return None


def open_panel(url: str) -> None:
    """Open in the desktop session inherited from the clickable launcher."""
    try:
        result = subprocess.run(
            ["xdg-open", url],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        if result.returncode:
            print("Browser could not open automatically; open " + url, flush=True)
    except (OSError, subprocess.TimeoutExpired):
        print("Browser could not open automatically; open " + url, flush=True)


def supervise(command: list[str], workdir: Path, env: dict) -> int:
    """Keep training/DAgger output live; never stop collector services or old Ray jobs."""
    runtime = Path(env.get("TIANJI_RUNTIME_DIR", str(workdir / ".runtime/tianji")))
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "stage2_session.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("A Stage2 launcher is already running; use its terminal.")
    port = int(env.get("TIANJI_HIL_PORT", "8879"))
    url = f"http://127.0.0.1:{port}"
    launch_id = uuid.uuid4().hex
    ready_file = runtime / "stage2_ready" / (launch_id + ".json")
    prestart_file = runtime / "stage2_ready" / (launch_id + ".prestart.json")
    tactile = env.get("TIANJI_TACTILE_REQUIRED") == "1"
    collector_url = env.get("LOOPINFRA_URL", "http://127.0.0.1:8780")
    vla_flag = Path(env.get("RLT_READY_FLAG", str(runtime / "rlt_rollout_ready")))
    child = None
    previous_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(signum)

    try:
        with socket.socket() as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(
                    f"HIL port {port} is already in use; stop the old model task first."
                )
        env = dict(
            env,
            TIANJI_STAGE2_READY_FILE=str(ready_file),
            TIANJI_STAGE2_PRESTART_FILE=str(prestart_file),
            TIANJI_STAGE2_LAUNCH_ID=launch_id,
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        started = time.time()
        child = subprocess.Popen(command, cwd=workdir, env=env, start_new_session=True)
        print(
            "STAGE2_LOADING: waiting for the VLA weights, the actor (and checkpoint), the first "
            "weight sync"
            + (", Xense" if tactile else "")
            + " and the HIL panel; the browser will open when Start can be pressed.",
            flush=True,
        )
        opened = announced = False
        xense_note = None
        while child.poll() is None:
            if (
                not opened
                and training_ready(prestart_file, launch_id)
                and vla_and_panel_ready(vla_flag, started, launch_id, url)
            ):
                waiting = xense_ready(collector_url) if tactile else None
                if waiting is not None:
                    if waiting != xense_note:
                        print("STAGE2_WAIT_XENSE: " + waiting, flush=True)
                        xense_note = waiting
                    time.sleep(0.5)
                    continue
                if child.poll() is not None:
                    break
                print(
                    "STAGE2_PANEL_READY "
                    + url
                    + " (VLA, actor, first weight sync"
                    + (", Xense" if tactile else "")
                    + " ready; waiting for manual Start)",
                    flush=True,
                )
                print("After Start the robot resets, then the policy begins.", flush=True)
                open_panel(url)
                opened = True
            if opened and not announced and training_ready(ready_file, launch_id):
                print(
                    "STAGE2_TRAINING_READY: initialization and the first weight sync are done.",
                    flush=True,
                )
                announced = True
            time.sleep(0.5)
        result = child.wait()
        if not opened:
            print("Stage2 exited before readiness; the browser was not opened.", flush=True)
        return result if result >= 0 else 128 - result
    except InterruptedError as exc:
        return 128 + int(exc.args[0])
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid, signal.SIGINT)
                child.wait(timeout=15)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    print(
                        f"Training PID {child.pid} has not exited; inspect its terminal/processes.",
                        flush=True,
                    )
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        for marker in (ready_file, prestart_file):
            marker.unlink(missing_ok=True)
            marker.with_suffix(".pending").unlink(missing_ok=True)
        lock.close()


def main() -> int:
    launcher = Path(__file__).resolve().parent
    rlinf_root = Path(os.environ.get("RLINF_ROOT", launcher.parents[1])).resolve()
    try:
        return supervise(
            ["bash", str(launcher / "run_tianji_stage2_local.sh"), "async", *sys.argv[1:]],
            rlinf_root,
            dict(os.environ),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print("STAGE2_START_ERROR: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
