"""Publish opt-in Tianji Stage2 readiness after initialization and weight sync."""

import json
import os
from pathlib import Path


def publish_stage2_ready(variable: str = "TIANJI_STAGE2_READY_FILE") -> None:
    """Atomically notify the launcher for this invocation, never an older run.

    ``TIANJI_STAGE2_PRESTART_FILE``: actor, resume and first weight sync are
    done while the env's first reset still waits for the operator's Start.
    """
    path = os.environ.get(variable)
    launch_id = os.environ.get("TIANJI_STAGE2_LAUNCH_ID")
    if not path or not launch_id:
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.with_suffix(".pending")
    pending.write_text(json.dumps({"launch_id": launch_id, "driver_pid": os.getpid()}))
    pending.replace(target)
