"""Wait for the existing GET-only policy admission checks; never send motion."""

import argparse
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from preflight_tianji_local import live_status, xense_status


def require_rt_schedule(url: str) -> None:
    """Verify the collector server exposes the Stage2 policy schedule row endpoint."""
    token = os.environ.get("LOOPINFRA_TOKEN")
    request = urllib.request.Request(
        url.rstrip("/") + "/api/policy_schedule/row?schedule_id=probe&row_id=-1&wait_ms=0",
        headers={"Authorization": "Bearer " + token} if token else {},
    )
    try:
        urllib.request.urlopen(request, timeout=3).close()
    except urllib.error.HTTPError as exc:
        if exc.code == 409 and b"unknown_policy_schedule_row" in exc.read():
            return
        raise RuntimeError(
            f"The running collector does not support Stage2 policy schedules (HTTP {exc.code}). "
            "Stop it with $LOOPINFRA_ROOT/scripts/tianji-stop, then restart via deploy/tianji."
        ) from exc
    raise RuntimeError("Unexpected policy schedule probe reply from the collector")


def wait_until_ready(timeout: float) -> None:
    """Wait for a healthy idle collector without changing its control state."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("collector readiness timeout must be positive and finite")
    deadline = time.monotonic() + timeout
    last_note = None
    while True:
        try:
            status = live_status(os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780"))
            if os.environ.get("TIANJI_TACTILE_REQUIRED") == "1":
                xense = xense_status(
                    os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780")
                )
        except (OSError, RuntimeError, ValueError) as exc:
            detail = str(exc)
            if detail != last_note:
                print("STAGE2_WAIT_COLLECTOR: " + detail, flush=True)
                print(
                    "Keep Position/EEF idle and release Pico; collector page: "
                    + os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780")
                    + "/collector",
                    flush=True,
                )
                last_note = detail
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    "Collector not ready; Stage2 was not started: " + detail
                ) from exc
            time.sleep(min(1.0, remaining))
        else:
            if os.environ.get("TIANJI_RT_SCHEDULE") == "1":
                require_rt_schedule(
                    os.environ.get("LOOPINFRA_URL", "http://127.0.0.1:8780")
                )
            for warning in status.get("warnings", []):
                print("STAGE2_COLLECTOR_WARNING: " + warning, flush=True)
            if os.environ.get("TIANJI_TACTILE_REQUIRED") == "1":
                print("STAGE2_XENSE_READY roles=" + ",".join(xense["roles"]), flush=True)
            print("STAGE2_COLLECTOR_READY", flush=True)
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    os.environ["RLINF_TIANJI_STAGE1_ONLY"] = "1"
    rlinf_root = Path(os.environ.get("RLINF_ROOT", Path(__file__).resolve().parents[2]))
    if str(rlinf_root) not in sys.path:
        sys.path.insert(0, str(rlinf_root))
    try:
        wait_until_ready(args.timeout)
    except KeyboardInterrupt:
        return 130
    except (RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
