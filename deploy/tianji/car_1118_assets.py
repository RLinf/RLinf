"""Pin a car NAS checkpoint and stage the serving assets atomically.

Supports both 15-stage (1118v1) and 6-stage (sub6) checkpoints via
TIANJI_CAR_NUM_SUBTASKS (default 15), TIANJI_CAR_NAS_SOURCE, and
TIANJI_CAR_CHECKPOINT (default step_30000).
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile

CHECKPOINT = os.environ.get("TIANJI_CAR_CHECKPOINT", "step_30000")
WEIGHTS = Path("model_state_dict/full_weights.pt")
FILES = (WEIGHTS, Path("norm_stats.json"), Path("prompts.json"))
DEFAULT_WEIGHT_SIZE = 16_401_053_736


def _num_subtasks() -> int:
    return int(os.environ.get("TIANJI_CAR_NUM_SUBTASKS", "15"))


def _expected_weight_size() -> int:
    raw = os.environ.get("TIANJI_CAR_WEIGHT_SIZE", "").strip()
    if raw:
        return int(raw)
    return DEFAULT_WEIGHT_SIZE


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_layout(root):
    expected_size = _expected_weight_size()
    actual_size = (root / WEIGHTS).stat().st_size
    if expected_size > 0:
        assert actual_size == expected_size, (
            f"Unexpected {CHECKPOINT} weight size: {actual_size} != {expected_size}"
        )
    assert isinstance(json.loads((root / "norm_stats.json").read_text()), dict)
    spec = json.loads((root / "prompts.json").read_text())
    prompts = spec["prompts"]
    expected_subtasks = _num_subtasks()
    assert len(prompts) == expected_subtasks and len(set(prompts)) == expected_subtasks, (
        f"Expected {expected_subtasks} unique prompts, got {len(prompts)}"
    )
    assert all(isinstance(p, str) and p.strip() and len(p) <= 4096 for p in prompts)
    assert spec["task_index"] == {prompt: index for index, prompt in enumerate(prompts)}
    return prompts


def verify(cache):
    manifest = json.loads((cache / "identity.json").read_text())
    nas_env = os.environ.get("TIANJI_CAR_NAS_SOURCE", "").strip()
    if nas_env:
        assert manifest["nas_source"] == str(Path(nas_env) / CHECKPOINT)
    else:
        assert str(manifest.get("nas_source", "")).endswith("/" + CHECKPOINT)
    assert set(manifest["sha256"]) == {str(name) for name in FILES}
    check_layout(cache)
    for name, expected in manifest["sha256"].items():
        if sha256(cache / name) != expected:
            raise ValueError(f"Cached model asset hash mismatch: {name}")
    return manifest


def stage(source, cache):
    if cache.exists():
        return verify(cache)
    if source is None:
        raise ValueError(
            "Provide --source NAS_ROOT or set TIANJI_CAR_NAS_SOURCE to stage car assets"
        )
    checkpoint = source / CHECKPOINT
    weights = checkpoint / "full_weights.pt"
    if not weights.is_file():
        weights = checkpoint / WEIGHTS
    roots = {WEIGHTS: weights, FILES[1]: checkpoint / FILES[1], FILES[2]: source / FILES[2]}
    sidecar = weights.with_suffix(".pt.sha256")
    published = (
        sidecar.read_text().split()[0].lower() if sidecar.is_file() else sha256(weights)
    )
    if not re.fullmatch(r"[0-9a-f]{64}", published):
        raise ValueError("Invalid NAS checkpoint SHA-256")
    hashes = {
        str(name): (published if name == WEIGHTS else sha256(path))
        for name, path in roots.items()
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".stage-car-", dir=cache.parent) as tmp:
        bundle = Path(tmp) / CHECKPOINT
        bundle.mkdir()
        for name, path in roots.items():
            target = bundle / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        manifest = {"nas_source": str(checkpoint), "sha256": hashes}
        (bundle / "identity.json").write_text(json.dumps(manifest, indent=2) + "\n")
        verify(bundle)
        os.rename(bundle, cache)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("operation", choices=("stage", "verify"))
    parser.add_argument("cache", type=Path)
    default_source = os.environ.get("TIANJI_CAR_NAS_SOURCE", "").strip()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(default_source) if default_source else None,
    )
    args = parser.parse_args()
    manifest = (
        stage(args.source, args.cache)
        if args.operation == "stage"
        else verify(args.cache)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
