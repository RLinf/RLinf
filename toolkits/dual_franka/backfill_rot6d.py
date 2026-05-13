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
"""Backfill a joint-action dual-Franka LeRobot dataset into rot6d_v1.

The source ``actions`` are joint-space but ``state`` already records TCP
pose (xyz + xyzw quat). No FK is run: the rot6d layout is built by
re-slicing those existing TCP columns and converting the quaternion via
``quat_xyzw_to_rot6d``. ``state[0:20]`` is rewritten in place;
``actions`` widens 16 → 20 using the next frame's TCP as a 10 Hz
commanded-target proxy. Output is a fresh dataset; the source is left
untouched.

Run from the repo root::

    export PYTHONPATH=$(pwd)
    python toolkits/dual_franka/backfill_rot6d.py \\
        --src /path/to/collected_data/rank_0/id_0 \\
        --dst /path/to/lerobot_rot6d_root
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tqdm

from rlinf.utils.rot6d import quat_xyzw_to_rot6d

STATE_DIM = 68
ACTION_DIM_IN = 16
ACTION_DIM_OUT = 20

# Source layout (env ``_wrap_obs`` alphabetical concat).
GRIP_SLICE = slice(0, 2)
L_XYZ_SLICE = slice(36, 39)
L_QUAT_SLICE = slice(39, 43)
R_XYZ_SLICE = slice(43, 46)
R_QUAT_SLICE = slice(46, 50)
L_GRIP_ACT_IDX = 7
R_GRIP_ACT_IDX = 15


# ---------------------------------------------------------------------------
# Pure-numpy transforms
# ---------------------------------------------------------------------------


def _assert_unit_quats(state_68: np.ndarray, slc: slice, name: str) -> None:
    """Fail loud if a slice that's *meant* to be xyzw quats isn't unit-norm.
    Cheap guard against pointing --src at a dataset whose state layout
    differs from the joint-env _wrap_obs alphabetical concat — without
    this, R.from_quat happily normalizes garbage into a plausible-looking
    but wrong rot6d.
    """
    norms = np.linalg.norm(state_68[:, slc], axis=-1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        bad = int(np.argmax(np.abs(norms - 1.0)))
        raise ValueError(
            f"{name} slice {slc} does not look like xyzw quaternions: "
            f"norm range [{norms.min():.4f}, {norms.max():.4f}], "
            f"worst row {bad} norm={norms[bad]:.4f}. The source dataset's "
            f"state layout probably differs from the joint-env _wrap_obs "
            f"alphabetical concat that this script assumes."
        )


def build_rot6d_state(state_68: np.ndarray) -> np.ndarray:
    """Overwrite ``state[:, 0:20]`` with the rot6d layout. Keeps dim 68."""
    if state_68.ndim != 2 or state_68.shape[1] != STATE_DIM:
        raise ValueError(f"expected state shape (T, {STATE_DIM}), got {state_68.shape}")

    _assert_unit_quats(state_68, L_QUAT_SLICE, "left")
    _assert_unit_quats(state_68, R_QUAT_SLICE, "right")

    grip = state_68[:, GRIP_SLICE]
    l_xyz = state_68[:, L_XYZ_SLICE]
    r_xyz = state_68[:, R_XYZ_SLICE]
    l_r6 = quat_xyzw_to_rot6d(state_68[:, L_QUAT_SLICE])
    r_r6 = quat_xyzw_to_rot6d(state_68[:, R_QUAT_SLICE])

    new_prefix = np.concatenate([grip, l_xyz, l_r6, r_xyz, r_r6], axis=-1)
    assert new_prefix.shape == (state_68.shape[0], 20)

    out = state_68.astype(np.float32, copy=True)
    out[:, 0:20] = new_prefix.astype(np.float32)
    return out


def build_rot6d_actions(state_68: np.ndarray, actions_16: np.ndarray) -> np.ndarray:
    """Build 20-d rot6d actions using next-frame TCP as the target proxy."""
    if actions_16.shape != (state_68.shape[0], ACTION_DIM_IN):
        raise ValueError(
            f"expected actions shape (T, {ACTION_DIM_IN}), got {actions_16.shape}"
        )

    _assert_unit_quats(state_68, L_QUAT_SLICE, "left")
    _assert_unit_quats(state_68, R_QUAT_SLICE, "right")

    # Shift state forward by one frame; the final frame repeats itself so the
    # last action means "hold current pose" (task already terminated via `c`).
    nxt = np.empty_like(state_68)
    nxt[:-1] = state_68[1:]
    nxt[-1] = state_68[-1]

    l_xyz = nxt[:, L_XYZ_SLICE]
    r_xyz = nxt[:, R_XYZ_SLICE]
    l_r6 = quat_xyzw_to_rot6d(nxt[:, L_QUAT_SLICE])
    r_r6 = quat_xyzw_to_rot6d(nxt[:, R_QUAT_SLICE])
    l_grip = actions_16[:, L_GRIP_ACT_IDX : L_GRIP_ACT_IDX + 1]
    r_grip = actions_16[:, R_GRIP_ACT_IDX : R_GRIP_ACT_IDX + 1]

    out = np.concatenate([l_xyz, l_r6, l_grip, r_xyz, r_r6, r_grip], axis=-1)
    assert out.shape == (state_68.shape[0], ACTION_DIM_OUT)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Parquet IO
# ---------------------------------------------------------------------------


def _fsl_float32(arr_2d: np.ndarray) -> pa.Array:
    """Wrap a ``(T, D)`` float array as a pyarrow ``FixedSizeList<float>[D]``."""
    if arr_2d.dtype != np.float32:
        arr_2d = arr_2d.astype(np.float32)
    t, d = arr_2d.shape
    flat = pa.array(arr_2d.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, d)


def _patch_hf_metadata(
    metadata: dict[bytes, bytes] | None,
) -> dict[bytes, bytes] | None:
    """Bump ``huggingface`` schema metadata's actions length to 20."""
    if metadata is None:
        return None
    out = dict(metadata)
    raw = out.get(b"huggingface")
    if raw is None:
        return out
    info = json.loads(raw)
    feats = info.get("info", {}).get("features", {})
    if "actions" in feats:
        feats["actions"]["length"] = ACTION_DIM_OUT
    out[b"huggingface"] = json.dumps(info, separators=(",", ":")).encode()
    return out


def _fsl_to_numpy(col: pa.ChunkedArray | pa.Array, dim: int) -> np.ndarray:
    """Pull a ``FixedSizeList<float>[D]`` column into ``(T, D)`` float32."""
    flat = np.asarray(col.combine_chunks().values.to_numpy(zero_copy_only=False))
    return flat.reshape(-1, dim).astype(np.float32, copy=False)


def rewrite_parquet(src_path: Path, dst_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Backfill one episode parquet. Returns the new (state, actions) arrays."""
    src_table = pq.read_table(src_path)

    if "state" not in src_table.column_names or "actions" not in src_table.column_names:
        raise ValueError(f"{src_path}: missing state/actions column")

    actions_type = src_table.schema.field("actions").type
    if getattr(actions_type, "list_size", None) == ACTION_DIM_OUT:
        raise ValueError(
            f"{src_path}: actions are already {ACTION_DIM_OUT}-d; dataset looks "
            "like it was already backfilled. Point --src at the original "
            "joint-space dataset, or pick a fresh --dst."
        )

    state_np = _fsl_to_numpy(src_table.column("state"), STATE_DIM)
    actions_np = _fsl_to_numpy(src_table.column("actions"), ACTION_DIM_IN)

    new_state = build_rot6d_state(state_np)
    new_actions = build_rot6d_actions(state_np, actions_np)

    # Build the replacement schema: actions widens to fixed_size_list<float>[20].
    new_fields = []
    for field in src_table.schema:
        if field.name == "actions":
            new_fields.append(
                pa.field(
                    "actions",
                    pa.list_(pa.float32(), ACTION_DIM_OUT),
                    nullable=field.nullable,
                    metadata=field.metadata,
                )
            )
        else:
            new_fields.append(field)
    new_schema = pa.schema(
        new_fields, metadata=_patch_hf_metadata(src_table.schema.metadata)
    )

    new_columns = []
    for name in src_table.column_names:
        if name == "state":
            new_columns.append(_fsl_float32(new_state))
        elif name == "actions":
            new_columns.append(_fsl_float32(new_actions))
        else:
            new_columns.append(src_table.column(name).combine_chunks())

    new_table = pa.Table.from_arrays(new_columns, schema=new_schema)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return new_state, new_actions


# ---------------------------------------------------------------------------
# Meta rewrite
# ---------------------------------------------------------------------------


def _stats_for(arr: np.ndarray) -> dict:
    """Per-column min/max/mean/std, matching lerobot v2.1 episodes_stats."""
    return {
        "min": arr.min(axis=0).astype(np.float32).tolist(),
        "max": arr.max(axis=0).astype(np.float32).tolist(),
        "mean": arr.mean(axis=0).astype(np.float32).tolist(),
        "std": arr.std(axis=0).astype(np.float32).tolist(),
        "count": [int(arr.shape[0])],
    }


def rewrite_meta(
    src_meta: Path,
    dst_meta: Path,
    new_stats_per_episode: dict[int, tuple[np.ndarray, np.ndarray]],
) -> None:
    """Mirror meta/, updating info.json (actions shape) and episodes_stats.jsonl."""
    dst_meta.mkdir(parents=True, exist_ok=True)

    # info.json: widen actions shape 16 -> 20.
    with (src_meta / "info.json").open() as f:
        info = json.load(f)
    info["features"]["actions"]["shape"] = [ACTION_DIM_OUT]
    with (dst_meta / "info.json").open("w") as f:
        json.dump(info, f, indent=4)

    # episodes_stats.jsonl: patch only state/actions per episode; keep others.
    src_stats = src_meta / "episodes_stats.jsonl"
    dst_stats = dst_meta / "episodes_stats.jsonl"
    if src_stats.exists():
        with src_stats.open() as f_in, dst_stats.open("w") as f_out:
            for line in f_in:
                entry = json.loads(line)
                ep = entry["episode_index"]
                if ep in new_stats_per_episode:
                    new_state, new_actions = new_stats_per_episode[ep]
                    entry["stats"]["state"] = _stats_for(new_state)
                    entry["stats"]["actions"] = _stats_for(new_actions)
                f_out.write(json.dumps(entry) + "\n")

    # Other meta files (episodes.jsonl, tasks.jsonl, etc.) — copy verbatim.
    for fname in ("episodes.jsonl", "tasks.jsonl"):
        src_file = src_meta / fname
        if src_file.exists():
            shutil.copy2(src_file, dst_meta / fname)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _find_parquets(src_data: Path) -> list[Path]:
    return sorted(src_data.rglob("episode_*.parquet"))


def backfill(src: Path, dst: Path, overwrite: bool = False) -> None:
    src_meta = src / "meta"
    src_data = src / "data"
    if not src_meta.is_dir() or not src_data.is_dir():
        raise FileNotFoundError(
            f"{src} does not look like a LeRobot root (expected meta/ and data/)"
        )

    if dst.exists():
        if not overwrite:
            raise FileExistsError(
                f"{dst} already exists. Use --overwrite to rewrite it."
            )
        shutil.rmtree(dst)
    dst_meta = dst / "meta"
    dst_data = dst / "data"

    parquets = _find_parquets(src_data)
    if not parquets:
        raise FileNotFoundError(f"No episode_*.parquet under {src_data}")

    per_episode_new: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for src_pq in tqdm.tqdm(parquets, desc="backfill rot6d"):
        rel = src_pq.relative_to(src_data)
        dst_pq = dst_data / rel
        new_state, new_actions = rewrite_parquet(src_pq, dst_pq)
        # episode_NNNNNN.parquet -> NNNNNN
        ep_idx = int(src_pq.stem.split("_")[-1])
        per_episode_new[ep_idx] = (new_state, new_actions)

    rewrite_meta(src_meta, dst_meta, per_episode_new)

    total_frames = sum(s.shape[0] for s, _ in per_episode_new.values())
    print(
        f"OK  episodes={len(per_episode_new)}  frames={total_frames}  "
        f"state dim={STATE_DIM} (prefix rewritten)  actions dim={ACTION_DIM_OUT}"
    )
    print(f"    src -> dst: {src}  ->  {dst}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill a dual-Franka joint-space LeRobot dataset into rot6d_v1."
    )
    parser.add_argument(
        "--src",
        type=Path,
        required=True,
        help="LeRobot root of the joint-space dataset (contains meta/ and data/).",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        required=True,
        help="Output LeRobot root for the rot6d_v1 dataset (must not exist unless --overwrite).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove --dst before writing. Does NOT touch --src.",
    )
    args = parser.parse_args()

    try:
        backfill(args.src, args.dst, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
