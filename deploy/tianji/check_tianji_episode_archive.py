#!/usr/bin/env python3
"""Check archived Stage2/DAgger episodes (runs/<run>/hil/episodes) for completeness.

Usage: check_tianji_episode_archive.py PATH [PATH ...]
PATH is an episode directory, an episodes/ directory or a run directory.
Standalone: no RLinf or torch import.
"""

import hashlib
import json
import sys
from pathlib import Path

MATCH_NS = 50_000_000
STEP_KEYS = (
    "state",
    "next_state",
    "action",
    "exec_action",
    "reward",
    "terminated",
    "truncated",
)


def episodes(path: Path):
    if (path / "manifest.json").exists() and (path / "steps.jsonl").exists():
        yield path
        return
    root = path / "hil" / "episodes" if (path / "hil" / "episodes").is_dir() else path
    for child in sorted(root.iterdir()):
        if child.is_dir() and child.name.startswith(("episode_", ".episode_")):
            yield child


def rows(path: Path):
    return (
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if path.exists()
        else []
    )


def check(path: Path) -> dict:
    problems = []
    result = {"episode": str(path), "published": not path.name.startswith(".")}
    if not result["published"] and not (path / "manifest.json").exists():
        return dict(
            result, valid=None, lerobot_ready=False, problems=["unfinished episode"]
        )
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    result.update(
        outcome=manifest.get("outcome"),
        complete=manifest.get("complete"),
        steps=manifest.get("steps"),
        chunks=manifest.get("chunks"),
        issues=manifest.get("issues", []),
    )
    for name, meta in manifest.get("files", {}).items():
        file = path / name
        if not file.exists():
            problems.append(f"missing {name}")
            continue
        data = file.read_bytes()
        if (
            len(data) != meta["bytes"]
            or hashlib.sha256(data).hexdigest() != meta["sha256"]
        ):
            problems.append(f"hash mismatch {name}")
    steps, chunks = rows(path / "steps.jsonl"), rows(path / "chunks.jsonl")
    images = rows(path / "images" / "inputs.jsonl") + rows(
        path / "images" / "stream.jsonl"
    )
    if len(steps) != manifest.get("steps"):
        problems.append(f"{len(steps)} step rows, manifest {manifest.get('steps')}")
    serials = {c["chunk"] for c in chunks}
    if any(s.get("chunk") not in serials for s in steps):
        problems.append("steps outside the listed chunks")
    for c in chunks:
        for kind in ("inference", "labels"):
            if not c.get(kind) or not (path / c[kind]).exists():
                problems.append(f"chunk {c['chunk']} has no {kind}")
    missing_keys = sorted({k for s in steps for k in STEP_KEYS if s.get(k) is None})
    roles = manifest.get("image_roles") or []
    blobs = {
        name: (path / "images" / name).read_bytes()
        for name in ("inputs.bin", "stream.bin")
        if (path / "images" / name).exists()
    }
    bad_jpeg = 0
    for entry in images:
        frames = blobs.get(entry.get("file"), b"")
        for role in roles:
            image = (entry.get("images") or {}).get(role)
            if image is None:
                bad_jpeg += 1
                continue
            data = frames[image["offset"] : image["offset"] + image["length"]]
            if (
                len(data) != image["length"]
                or data[:2] != b"\xff\xd8"
                or b"\xff\xd9" not in data[-16:]
            ):
                bad_jpeg += 1
    if bad_jpeg:
        problems.append(f"{bad_jpeg} camera images missing or not whole JPEGs")
    captures = sorted(
        g["capture_ns"] for g in images if isinstance(g.get("capture_ns"), int)
    )
    nearest = []
    for s in steps:
        t = s.get("capture_ns")
        if not isinstance(t, int) or not captures:
            nearest.append(None)
            continue
        lo, hi = 0, len(captures)
        while lo < hi:
            mid = (lo + hi) // 2
            if captures[mid] < t:
                lo = mid + 1
            else:
                hi = mid
        nearest.append(
            min(abs(captures[j] - t) for j in (lo - 1, lo) if 0 <= j < len(captures))
        )
    far = [n is None or n > MATCH_NS for n in nearest]
    very_far = [n is None or n > 2 * MATCH_NS for n in nearest]
    missing = [n is None or n > 5 * MATCH_NS for n in nearest]
    gaps = [b - a for a, b in zip(captures, captures[1:])]
    result.update(
        image_groups=len(images),
        image_bytes=sum(len(b) for b in blobs.values()),
        image_max_interval_ms=round(max(gaps) / 1e6, 1) if gaps else None,
        steps_without_image_within_50ms=sum(far),
        steps_without_image_within_100ms=sum(very_far),
        steps_without_image_within_250ms=sum(missing),
        step_failures=sum(bool(s.get("step_failure")) for s in steps),
        takeovers=sum(bool(s.get("takeover_start")) for s in steps),
        human_steps=sum(s.get("iv_kind") in (2, 3) for s in steps),
        subtask_completions=sum(int(s.get("milestone") or 0) for s in steps),
        missing_step_fields=missing_keys,
    )
    result["valid"] = not problems
    result["lerobot_ready"] = bool(
        result["valid"]
        and steps
        and not missing_keys
        and not bad_jpeg
        and not any(missing)
        and sum(very_far) <= 0.01 * len(steps)
        and all(
            s.get("subtask_prompt") is not None or not manifest.get("subtask_prompts")
            for s in steps
        )
    )
    result["problems"] = problems
    return result


def main(argv) -> int:
    if not argv:
        print(__doc__)
        return 2
    failed = False
    for arg in argv:
        for path in episodes(Path(arg)):
            try:
                result = check(path)
            except Exception as exc:  # noqa: BLE001
                result = {
                    "episode": str(path),
                    "published": not path.name.startswith("."),
                    "valid": False,
                    "problems": [f"{type(exc).__name__}: {exc}"],
                }
            failed |= bool(result["published"] and not result["valid"])
            print(json.dumps(result, ensure_ascii=False))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
