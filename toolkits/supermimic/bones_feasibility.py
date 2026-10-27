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

"""Add Appendix E geometry to G1 BONES-SEED clips and report filter counts.

EXPERIMENTAL. Input is a folder of ``clip_*.pt`` files in the G1 payload
format (fps, joint_names, q0, state) plus ``name``. For every clip the G1
is posed at each source frame and ``mj_geomDistance`` gives ``zmin`` and
``foot_clearance`` against a z = 0 plane. Clips are then run through the
same name and feasibility filters ``load_bones_library`` applies, and the
counts go to ``bones_stats.json``.

The paper reports 142,220 -> 129,667 (name) -> 129,514 (feasibility) for
its release. Counts from any other dump are that dump's counts.

Write annotated clips the loader can filter::

    python toolkits/supermimic/bones_feasibility.py --clips /data/bones_g1 \\
        --out /data/bones_g1_filtered

Statistics only, for another named or unnamed G1 clip set::

    python toolkits/supermimic/bones_feasibility.py --clips /data/lafan1_g1 \\
        --stats-only --allow-unnamed --stats /tmp/lafan1_feasibility.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# ``python toolkits/supermimic/bones_feasibility.py`` puts only this folder on sys.path.
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

PAPER_COUNTS = {"source": 142220, "after_name": 129667, "after_feasibility": 129514}


def process(
    clips_dir: Path,
    out_dir: Path | None,
    allow_unnamed: bool = False,
    limit: int = 0,
) -> dict:
    import torch

    from rlinf.envs.sim.g1_mimic.bones import feasibility_reason, name_rejected
    from rlinf.envs.sim.g1_mimic.bones_geometry import G1FloorDistance
    from rlinf.envs.sim.g1_mimic.motion import decode_motion_payload

    files = sorted(clips_dir.glob("clip_*.pt"))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no clip_*.pt under {clips_dir}")
    if out_dir is not None:
        if out_dir.resolve() == clips_dir.resolve():
            raise SystemExit(
                "--out must differ from --clips; source clips are not rewritten"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
    geometry = G1FloorDistance()
    reasons: Counter = Counter()
    rows = []
    kept_seconds = 0.0
    for file in files:
        payload = torch.load(file, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise SystemExit(f"{file} is not a G1 clip payload")
        if "name" not in payload:
            if not allow_unnamed:
                raise SystemExit(
                    f"{file} has no name; pass --allow-unnamed to skip the name filter"
                )
            name, check_name = str(payload.get("source", file.stem)), False
        else:
            name, check_name = str(payload["name"]), True
        source_state = torch.as_tensor(payload["state"], dtype=torch.float32)
        zmin, feet = geometry.clip(source_state)
        payload = dict(payload, zmin=zmin, foot_clearance=feet)
        state, _q0 = decode_motion_payload(payload, str(file))
        if check_name and name_rejected(name):
            reason = "name"
        else:
            reason = feasibility_reason(payload, state, str(file)) or "kept"
        reasons[reason] += 1
        seconds = (state.shape[0] - 1) / 50.0
        if reason == "kept":
            kept_seconds += seconds
        rows.append(
            {
                "file": file.name,
                "name": name,
                "result": reason,
                "seconds": seconds,
                "zmin_range": [float(zmin.min()), float(zmin.max())],
                "foot_clearance_min": float(feet.min()),
                "both_feet_off_fraction": float(
                    (feet > 0.02).all(dim=-1).float().mean()
                ),
            }
        )
        if out_dir is not None:
            torch.save(payload, out_dir / file.name)
    named = sum(1 for row in rows if row["result"] != "name")
    return {
        "label": "experimental: Appendix E filters on this dump, not the paper's counts",
        "clips_dir": str(clips_dir),
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "geometry": "mj_geomDistance from every G1 collision geom to a z=0 plane",
        "num_files": len(rows),
        "after_name": named,
        "after_feasibility": reasons["kept"],
        "rejected": {key: value for key, value in reasons.items() if key != "kept"},
        "kept_hours": kept_seconds / 3600.0,
        "paper_reference": PAPER_COUNTS,
        "clips": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--clips", required=True, help="folder of G1 clip_*.pt payloads"
    )
    parser.add_argument("--out", default="", help="write annotated clips here")
    parser.add_argument(
        "--stats", default="", help="statistics JSON (default <out>/bones_stats.json)"
    )
    parser.add_argument("--stats-only", action="store_true", help="do not write clips")
    parser.add_argument(
        "--allow-unnamed",
        action="store_true",
        help="clips without a name skip the name filter",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="process only the first N files"
    )
    args = parser.parse_args()

    if not args.stats_only and not args.out:
        parser.error("pass --out, or --stats-only with --stats")
    if args.stats_only and not args.stats:
        parser.error("--stats-only needs --stats")
    out_dir = None if args.stats_only else Path(args.out)
    stats = process(Path(args.clips), out_dir, args.allow_unnamed, args.limit)
    stats_path = Path(args.stats) if args.stats else out_dir / "bones_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2))
    print(
        f"[bones] {stats['num_files']} files, {stats['after_name']} after name, "
        f"{stats['after_feasibility']} after feasibility ({stats['kept_hours']:.2f} h); "
        f"rejected {stats['rejected']}; {stats['label']}"
    )
    print(f"[bones] statistics: {stats_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
