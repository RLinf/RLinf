"""Find the last saved checkpoint in the newest timestamped Stage2 training run."""

import argparse
from datetime import datetime
from pathlib import Path
import re
import sys

RUN_NAME = re.compile(r"stage2_async_(\d{8}_\d{6})(?:_\d+)?$")
CHECKPOINT_NAME = re.compile(r"global_step_(\d+)$")


def latest_checkpoint(runs_root: Path) -> Path:
    """Use the newest real training output; never fall back to an older run."""
    runs = []
    for path in runs_root.glob("stage2_async_*"):
        match = RUN_NAME.fullmatch(path.name)
        if not path.is_dir() or match is None:
            continue
        try:
            timestamp = datetime.strptime(match[1], "%Y%m%d_%H%M%S")
        except ValueError:
            continue
        runs.append((timestamp, path.stat().st_mtime_ns, path))
    if not runs:
        raise ValueError(f"No Stage2 training runs found under: {runs_root}")
    run = max(runs, key=lambda item: (item[0], item[1]))[2]
    checkpoints = run / "tianji2_rlt_stage2_async/checkpoints"
    candidates = []
    for path in checkpoints.glob("global_step_*"):
        match = CHECKPOINT_NAME.fullmatch(path.name)
        if path.is_dir() and match:
            candidates.append((int(match[1]), path))
    for _, checkpoint in sorted(candidates, reverse=True):
        actor = checkpoint / "actor"
        if (
            actor.is_dir()
            and (actor / "sac_components/runtime_state_rank_0.pt").is_file()
            and (actor / "sac_components/target_model/checkpoint_rank_0.pt").is_file()
            and (actor / "sac_components/replay_buffer/rank_0").is_dir()
        ):
            return checkpoint.resolve()
    raise ValueError(
        f"Newest Stage2 run has no complete checkpoint yet: {run}. "
        "Pass an explicit checkpoint directory instead."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs_root", type=Path)
    args = parser.parse_args()
    try:
        print(latest_checkpoint(args.runs_root))
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
