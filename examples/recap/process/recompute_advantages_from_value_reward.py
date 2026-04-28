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

"""
Recompute advantages from existing value_current and reward with a new horizon.

Uses: A = normalize(reward_sum) + gamma^N * v_next - v_curr
- v_curr = value_current at current frame (from parquet)
- v_next = value_current at frame t+N (same episode); out-of-bounds -> 0
- reward_sum = discounted sum over N steps, normalized by global return min/max
- unified_threshold from positive_quantile (top X% positive), then update
  advantage (bool) and advantage_continuous columns and mixture_config.

Usage:
  # Full recompute (provide --advantage_lookahead_step)
  python recompute_advantages_from_value_reward.py \
    --dataset_root /path/to/transformed_advantage_dataset \
    --advantage_lookahead_step 20 \
    --positive_quantile 0.3 \
    --num_workers 4

  # Threshold-only relabeling (omit --advantage_lookahead_step)
  python recompute_advantages_from_value_reward.py \
    --dataset_paths /path/to/ds_a /path/to/ds_b \
    --source_tag old_q20 \
    --new_tag new_q30 \
    --positive_quantile 0.3
"""

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _suppress_video_logging() -> None:
    """Suppress libdav1d/av logging."""
    os.environ["AV_LOG_LEVEL"] = "panic"
    os.environ["LIBDAV1D_LOG_LEVEL"] = "0"
    for name in ("libav", "av", "PIL"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    try:
        import av

        av.logging.set_level(av.logging.PANIC)
    except ImportError:
        pass


_suppress_video_logging()

from rlinf.data.datasets.recap.utils import (  # noqa: E402
    load_return_stats_from_dataset,
)


def _infer_return_range(
    dataset_paths: list[Path],
    root_mixture: dict | None = None,
) -> tuple[float | None, float | None]:
    """Infer global return range from root mixture_config, per-dataset configs, or stats.json."""
    rmin = root_mixture.get("global_return_min") if root_mixture else None
    rmax = root_mixture.get("global_return_max") if root_mixture else None
    if rmin is not None and rmax is not None:
        return float(rmin), float(rmax)

    mins, maxs = [], []
    for dp in dataset_paths:
        dp_min, dp_max = None, None
        mc_path = dp / "mixture_config.yaml"
        if mc_path.exists():
            try:
                mc = yaml.safe_load(mc_path.read_text()) or {}
                dp_min, dp_max = (
                    mc.get("global_return_min"),
                    mc.get("global_return_max"),
                )
            except yaml.YAMLError:
                pass
        if dp_min is None or dp_max is None:
            s_min, s_max = load_return_stats_from_dataset(dp)
            dp_min = dp_min if dp_min is not None else s_min
            dp_max = dp_max if dp_max is not None else s_max
        if dp_min is not None:
            mins.append(float(dp_min))
        if dp_max is not None:
            maxs.append(float(dp_max))

    return (
        float(rmin) if rmin is not None else (min(mins) if mins else None),
        float(rmax) if rmax is not None else (max(maxs) if maxs else None),
    )


def discover_datasets_and_return_range(
    dataset_root: Path,
) -> tuple[list[Path], float | None, float | None]:
    """Discover sub-datasets under root and get global return range."""
    mixture_path = dataset_root / "mixture_config.yaml"
    mixture: dict = {}
    dataset_paths: list[Path] = []

    if mixture_path.exists():
        with open(mixture_path, "r") as f:
            mixture = yaml.safe_load(f) or {}
        for d in mixture.get("datasets", []):
            name = (
                d["name"]
                if isinstance(d, dict) and "name" in d
                else (d if isinstance(d, str) else None)
            )
            if name:
                path = dataset_root / name
                if path.is_dir() and (path / "meta" / "info.json").exists():
                    dataset_paths.append(path)

    if not dataset_paths:
        dataset_paths = [
            c
            for c in sorted(dataset_root.iterdir())
            if c.is_dir() and (c / "meta" / "info.json").exists()
        ]
    if not dataset_paths:
        raise ValueError(f"No datasets found under {dataset_root}")

    rmin, rmax = _infer_return_range(dataset_paths, root_mixture=mixture)
    return dataset_paths, rmin, rmax


def load_existing_advantages(
    dataset_path: Path,
    source_tag: str | None = None,
) -> pd.DataFrame:
    """Load an existing advantages parquet for threshold reapplication."""
    adv_filename = (
        f"advantages_{source_tag}.parquet" if source_tag else "advantages.parquet"
    )
    adv_path = dataset_path / "meta" / adv_filename
    if not adv_path.exists():
        raise FileNotFoundError(
            f"Advantage file not found: {adv_path}. "
            "Pass --source_tag for a tagged file or run compute_advantages.py first."
        )

    df = pd.read_parquet(adv_path)
    required_cols = {"episode_index", "frame_index"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Advantage file {adv_path} missing required columns: {sorted(missing_cols)}"
        )

    if "advantage_continuous" not in df.columns:
        if "advantage" not in df.columns:
            raise ValueError(
                f"Advantage file {adv_path} must contain either "
                "'advantage_continuous' or 'advantage'."
            )
        df["advantage_continuous"] = df["advantage"].astype(np.float64)

    if "advantage" not in df.columns:
        df["advantage"] = df["advantage_continuous"] >= 0.0

    return df


def compute_advantages_for_dataset(
    dataset_path: Path,
    advantage_lookahead_step: int,
    gamma: float,
    global_return_min: float,
    global_return_max: float,
    num_workers: int = 1,
    returns_tag: str | None = None,
) -> pd.DataFrame:
    """
    Compute advantage_continuous, reward_sum, value_next for one dataset.

    Reads directly from parquet files (avoids video decoding for speed).
    Per-episode computation is vectorized using numpy stride tricks.
    """
    _suppress_video_logging()

    data_dir = dataset_path / "data"
    if not data_dir.exists():
        raise ValueError(f"No data/ directory at {dataset_path}")

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_dir}")

    sample_schema = pq.read_schema(str(parquet_files[0]))
    sample_cols = set(sample_schema.names)

    columns_needed = ["episode_index", "frame_index"]
    for col in ["value_current", "reward", "return"]:
        if col in sample_cols:
            columns_needed.append(col)

    if num_workers > 1 and len(parquet_files) > 1:
        read_workers = min(num_workers, len(parquet_files))
        with ThreadPoolExecutor(max_workers=read_workers) as executor:
            dfs = list(
                executor.map(
                    lambda f: pd.read_parquet(f, columns=columns_needed),
                    parquet_files,
                )
            )
    else:
        dfs = [pd.read_parquet(f, columns=columns_needed) for f in parquet_files]
    full_df = pd.concat(dfs, ignore_index=True)
    full_df = full_df.sort_values(["episode_index", "frame_index"]).reset_index(
        drop=True
    )

    sidecar_filename = (
        f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    )
    sidecar_path = dataset_path / "meta" / sidecar_filename
    if (
        "reward" not in full_df.columns or "return" not in full_df.columns
    ) and sidecar_path.exists():
        sidecar_df = pd.read_parquet(
            sidecar_path, columns=["episode_index", "frame_index", "reward", "return"]
        )
        sidecar_df = sidecar_df.sort_values(
            ["episode_index", "frame_index"]
        ).reset_index(drop=True)
        merge_cols = [
            c
            for c in ["reward", "return"]
            if c not in full_df.columns and c in sidecar_df.columns
        ]
        if merge_cols:
            full_df = full_df.merge(
                sidecar_df[["episode_index", "frame_index"] + merge_cols],
                on=["episode_index", "frame_index"],
                how="left",
            )
            logger.info(f"Loaded {merge_cols} from sidecar: {sidecar_path}")

    if "value_current" in full_df.columns:
        full_df["value_current"] = full_df["value_current"].fillna(0.0)
    else:
        raise ValueError(
            f"Dataset {dataset_path} missing 'value_current' column in episode parquets. "
            "Run compute_advantages.py first to generate value predictions."
        )
    if "reward" not in full_df.columns:
        raise ValueError(
            f"Dataset {dataset_path} missing 'reward' column and no returns sidecar found. "
            "Run compute_returns.py first."
        )
    full_df["reward"] = full_df["reward"].fillna(0.0)
    has_return = "return" in full_df.columns

    ret_range = global_return_max - global_return_min
    if ret_range <= 0:
        raise ValueError(
            f"Invalid return range: [{global_return_min}, {global_return_max}] "
            f"(range={ret_range}). Ensure global_return_max > global_return_min."
        )
    gamma_powers = np.array(
        [gamma**i for i in range(advantage_lookahead_step)], dtype=np.float32
    )
    gamma_k = gamma**advantage_lookahead_step

    adv_cols = [
        "episode_index",
        "frame_index",
        "advantage",
        "reward_sum",
        "value_current",
        "value_next",
        "return",
    ]
    results: list[pd.DataFrame] = []
    for ep_idx, ep_group in tqdm(
        full_df.groupby("episode_index"),
        desc=f"Advantages ({dataset_path.name})",
        leave=False,
    ):
        ep_df = ep_group.sort_values("frame_index").reset_index(drop=True)
        ep_len = len(ep_df)
        if ep_len == 0:
            continue

        rewards = ep_df["reward"].values.astype(np.float32)
        values = ep_df["value_current"].values.astype(np.float64)

        v_next_arr = np.zeros(ep_len, dtype=np.float64)
        cut = ep_len - advantage_lookahead_step
        if cut > 0:
            v_next_arr[:cut] = values[advantage_lookahead_step:]

        padded = np.ascontiguousarray(
            np.concatenate(
                [rewards, np.zeros(advantage_lookahead_step - 1, dtype=np.float32)]
            )
        )
        windowed = np.lib.stride_tricks.as_strided(
            padded,
            shape=(ep_len, advantage_lookahead_step),
            strides=(padded.strides[0], padded.strides[0]),
        )
        reward_sums = (windowed @ gamma_powers).astype(np.float64)
        reward_sums = (reward_sums - global_return_min) / ret_range - 1.0

        results.append(
            pd.DataFrame(
                {
                    "episode_index": np.full(ep_len, int(ep_idx), dtype=np.int64),
                    "frame_index": ep_df["frame_index"].values.astype(np.int64),
                    "advantage": reward_sums + gamma_k * v_next_arr - values,
                    "reward_sum": reward_sums,
                    "value_current": values,
                    "value_next": v_next_arr,
                    "return": ep_df["return"].values.astype(np.float64)
                    if has_return
                    else np.zeros(ep_len),
                }
            )
        )

    if not results:
        return pd.DataFrame(columns=adv_cols)
    return pd.concat(results, ignore_index=True)


def modify_episodes_with_advantage(
    source_episodes_path: Path,
    output_episodes_path: Path,
    advantages_df: pd.DataFrame,
    threshold: float,
) -> None:
    """Set is_success per episode: any frame has advantage (continuous) >= threshold."""
    adv_col = (
        "advantage_continuous"
        if "advantage_continuous" in advantages_df.columns
        else "advantage"
    )
    episode_has_positive = (
        advantages_df.groupby("episode_index")[adv_col]
        .apply(lambda x: (x >= threshold).any())
        .to_dict()
    )
    episodes = []
    with open(source_episodes_path, "r") as f:
        for line in f:
            episode = json.loads(line.strip())
            ep_idx = episode.get("episode_index", len(episodes))
            episode["is_success"] = episode_has_positive.get(ep_idx, False)
            episodes.append(episode)
    with open(output_episodes_path, "w") as f:
        for episode in episodes:
            f.write(json.dumps(episode) + "\n")
    num_ok = sum(1 for ep in episodes if ep.get("is_success", False))
    logger.info(
        f"Modified episodes.jsonl: {num_ok}/{len(episodes)} episodes marked successful"
    )


def _update_single_parquet_file(
    pq_file: Path,
    advantage_lookup: dict[tuple[int, int], int],
    adv_cont_arr: np.ndarray,
    threshold: float,
    vc_arr: np.ndarray | None = None,
    vn_arr: np.ndarray | None = None,
    rs_arr: np.ndarray | None = None,
    ret_arr: np.ndarray | None = None,
) -> tuple[int, int]:
    """Update a single parquet file with advantage columns. Returns (updated, missing)."""
    from datasets import Dataset

    _suppress_video_logging()
    ds = Dataset.from_parquet(str(pq_file))
    if len(ds) == 0:
        return 0, 0

    # Vectorized lookup
    indices = np.array(
        [
            advantage_lookup.get((int(e), int(f)), -1)
            for e, f in zip(ds["episode_index"], ds["frame_index"])
        ]
    )
    found = indices >= 0
    safe_idx = np.maximum(indices, 0)

    adv_values = np.where(found, adv_cont_arr[safe_idx], threshold - 0.1)
    columns_to_add: dict[str, list] = {
        "advantage": (adv_values >= threshold).tolist(),
        "advantage_continuous": adv_values.tolist(),
    }
    if vc_arr is not None:
        for col, arr in [
            ("value_current", vc_arr),
            ("value_next", vn_arr),
            ("reward_sum", rs_arr),
            ("return", ret_arr),
        ]:
            columns_to_add[col] = np.where(found, arr[safe_idx], 0.0).tolist()

    existing = set(ds.column_names)
    for col in columns_to_add:
        if col in existing:
            ds = ds.remove_columns([col])
    for col, data in columns_to_add.items():
        ds = ds.add_column(col, data)
    ds.to_parquet(str(pq_file))
    return int(found.sum()), int((~found).sum())


def add_advantages_to_parquet_files(
    data_dir: Path,
    advantages_df: pd.DataFrame,
    threshold: float,
    num_workers: int = 1,
    adv_only: bool = False,
) -> tuple[int, int]:
    """Write advantage columns to parquet files.

    adv_only=True writes only advantage/advantage_continuous (threshold relabeling).
    """
    advantages_df = advantages_df.copy()
    if adv_only:
        if "advantage_continuous" not in advantages_df.columns:
            raise ValueError(
                "advantages_df must contain 'advantage_continuous' for threshold-only relabeling."
            )
    else:
        advantages_df["advantage_continuous"] = advantages_df["advantage"]

    ep_arr = advantages_df["episode_index"].values.astype(np.int64)
    fr_arr = advantages_df["frame_index"].values.astype(np.int64)
    adv_cont_arr = advantages_df["advantage_continuous"].values
    extra = {}
    if not adv_only:
        extra = {
            k: advantages_df[v].values
            for k, v in {
                "vc_arr": "value_current",
                "vn_arr": "value_next",
                "rs_arr": "reward_sum",
                "ret_arr": "return",
            }.items()
        }

    lookup = {(int(ep_arr[i]), int(fr_arr[i])): i for i in range(len(advantages_df))}
    parquet_files = list(data_dir.rglob("*.parquet"))
    if not parquet_files:
        return 0, 0
    logger.info(f"  Found {len(parquet_files)} parquet files to update")

    def _process(pq_file):
        return _update_single_parquet_file(
            pq_file, lookup, adv_cont_arr, threshold, **extra
        )

    workers = max(min(num_workers, len(parquet_files)), 1)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            tqdm(
                executor.map(_process, parquet_files),
                total=len(parquet_files),
                desc="  Updating parquet",
                leave=False,
            )
        )

    total_updated = sum(u for u, _ in results)
    total_missing = sum(m for _, m in results)
    logger.info(f"  Updated {total_updated} samples with advantages")
    if total_missing > 0:
        logger.warning(f"  {total_missing} samples missing advantage (set to negative)")
    return total_updated, total_missing


def build_save_advantages_df(
    advantages_df: pd.DataFrame,
    threshold: float,
) -> pd.DataFrame:
    """Return a DataFrame with bool and continuous advantage columns ready to save."""
    save_df = advantages_df.copy()
    if "advantage_continuous" not in save_df.columns:
        if "advantage" not in save_df.columns:
            raise ValueError(
                "advantages_df must contain either 'advantage_continuous' or 'advantage'."
            )
        save_df.rename(columns={"advantage": "advantage_continuous"}, inplace=True)

    save_df["advantage"] = save_df["advantage_continuous"] >= threshold
    return save_df


def update_mixture_config(
    mixture_root: Path,
    dataset_paths: list[Path],
    positive_quantile: float,
    unified_threshold: float,
    advantage_tag: str | None,
    global_return_min: float | None,
    global_return_max: float | None,
) -> None:
    """Update mixture_config.yaml for the current relabeling run."""
    mixture_path = mixture_root / "mixture_config.yaml"
    if mixture_path.exists():
        with open(mixture_path, "r") as f:
            mixture = yaml.safe_load(f) or {}
    else:
        mixture = {}

    mixture["datasets"] = [{"name": p.name, "weight": 1.0} for p in dataset_paths]
    if global_return_min is not None:
        mixture["global_return_min"] = global_return_min
    if global_return_max is not None:
        mixture["global_return_max"] = global_return_max

    tag_stats = {
        "unified_threshold": unified_threshold,
        "positive_quantile": positive_quantile,
    }
    if advantage_tag:
        if "tags" not in mixture:
            mixture["tags"] = {}
        mixture["tags"][advantage_tag] = tag_stats
        mixture["advantage_tag"] = advantage_tag

    mixture["unified_threshold"] = unified_threshold
    mixture["positive_quantile"] = positive_quantile

    with open(mixture_path, "w") as f:
        yaml.dump(mixture, f, default_flow_style=False)

    logger.info(f"  Saved mixture_config.yaml to: {mixture_path}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="Recompute advantages from value_current and reward with new horizon."
    )
    parser.add_argument(
        "--dataset_root",
        type=Path,
        help="Root path containing multiple LeRobot datasets (e.g. transformed_advantage_dataset_*)",
    )
    parser.add_argument(
        "--dataset_paths",
        type=Path,
        nargs="+",
        default=None,
        help="Explicit dataset paths to process. Use this instead of --dataset_root.",
    )
    parser.add_argument(
        "--advantage_lookahead_step",
        "--advantage_horizon",
        type=int,
        dest="advantage_lookahead_step",
        default=None,
        help="N steps for reward sum and v_next (value_current at t+N). "
        "If omitted, only reapply quantile threshold on existing advantages.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Discount factor. Default 1.0.",
    )
    parser.add_argument(
        "--positive_quantile",
        type=float,
        default=0.3,
        help="Top fraction of advantages treated as positive (e.g. 0.3 = top 30%%). Default 0.3.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="If set, write outputs here; else update datasets in place under dataset_root.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel threads for parquet file read/write within each dataset. Default 1.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Legacy single tag. If set, behaves like --new_tag for outputs and "
        "like --source_tag for optional sidecar inputs.",
    )
    parser.add_argument(
        "--source_tag",
        type=str,
        default=None,
        help="Source tag to read existing meta/advantages_{source_tag}.parquet or "
        "meta/returns_{source_tag}.parquet from.",
    )
    parser.add_argument(
        "--new_tag",
        type=str,
        default=None,
        help="Output tag for meta/advantages_{new_tag}.parquet.",
    )
    parser.add_argument(
        "--skip_embed",
        action="store_true",
        help="Skip embedding advantages into data parquet files. Only save meta parquet.",
    )
    args = parser.parse_args()

    if bool(args.dataset_root) == bool(args.dataset_paths):
        parser.error("Specify exactly one of --dataset_root or --dataset_paths.")
    if not 0.0 < args.positive_quantile < 1.0:
        parser.error("--positive_quantile must be in the open interval (0, 1).")
    if args.tag and args.new_tag and args.tag != args.new_tag:
        parser.error("--tag and --new_tag disagree; pass only one output tag.")

    recompute = args.advantage_lookahead_step is not None
    source_tag = args.source_tag or args.tag
    output_tag = args.new_tag or args.tag

    if args.dataset_root:
        dataset_root = args.dataset_root.resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
        dataset_paths, global_return_min, global_return_max = (
            discover_datasets_and_return_range(dataset_root)
        )
    else:
        dataset_root = None
        dataset_paths = [p.resolve() for p in args.dataset_paths]
        for dataset_path in dataset_paths:
            if not dataset_path.is_dir():
                raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            if not (dataset_path / "meta" / "info.json").exists():
                raise ValueError(
                    f"Dataset path does not look like a LeRobot dataset: {dataset_path}"
                )
        global_return_min, global_return_max = _infer_return_range(dataset_paths)

    logger.info(
        f"Found {len(dataset_paths)} datasets: {[p.name for p in dataset_paths]}"
    )
    if global_return_min is not None and global_return_max is not None:
        logger.info(f"Global return range: [{global_return_min}, {global_return_max}]")
    else:
        logger.info(
            "Global return range unavailable from metadata; threshold-only relabeling can still proceed."
        )
    if recompute:
        logger.info(
            f"Mode: full recompute (lookahead={args.advantage_lookahead_step}, gamma={args.gamma})"
        )
    else:
        logger.info(
            "Mode: threshold-only relabeling (--advantage_lookahead_step not set)"
        )
    logger.info(f"Positive quantile: {args.positive_quantile}")
    if source_tag:
        logger.info(f"Source tag: {source_tag}")
    if output_tag:
        logger.info(f"Output tag: {output_tag}")

    all_advantages: list[np.ndarray] = []
    dataset_results: list[tuple[Path, pd.DataFrame]] = []

    num_workers = args.num_workers

    logger.info(
        f"Processing {len(dataset_paths)} datasets with {num_workers} workers "
        f"(parallelized within each dataset for parquet read/write)"
    )
    for ds_path in tqdm(dataset_paths, desc="Processing datasets"):
        if recompute:
            if global_return_min is None or global_return_max is None:
                raise ValueError(
                    "Cannot recompute from value/reward without a known global return range. "
                    "Provide datasets with mixture_config.yaml or meta/stats.json return metadata, "
                    "or omit --advantage_lookahead_step for threshold-only relabeling."
                )
            df = compute_advantages_for_dataset(
                ds_path,
                args.advantage_lookahead_step,
                args.gamma,
                global_return_min,
                global_return_max,
                num_workers=num_workers,
                returns_tag=source_tag,
            )
            all_advantages.append(df["advantage"].values)
            logger.info(
                f"  {ds_path.name}: {len(df)} advantages, mean={df['advantage'].mean():.4f}"
            )
        else:
            df = load_existing_advantages(ds_path, source_tag=source_tag)
            all_advantages.append(df["advantage_continuous"].values)
            logger.info(
                f"  {ds_path.name}: {len(df)} existing advantages, "
                f"mean={df['advantage_continuous'].mean():.4f}"
            )
        dataset_results.append((ds_path, df))

    combined_advantages = np.concatenate(all_advantages)
    unified_threshold = float(
        np.percentile(combined_advantages, (1.0 - args.positive_quantile) * 100)
    )
    logger.info(
        f"Unified threshold ({(1 - args.positive_quantile) * 100:.0f}th percentile): {unified_threshold:.4f}"
    )
    logger.info(
        f"Combined advantage range: [{combined_advantages.min():.4f}, {combined_advantages.max():.4f}]"
    )
    logger.info(
        f"Total samples with positive advantage: {(combined_advantages >= unified_threshold).sum()}"
    )

    for i, (ds_path, df) in enumerate(dataset_results):
        pos = (all_advantages[i] >= unified_threshold).sum()
        logger.info(
            f"  {ds_path.name}: {pos}/{len(df)} ({100 * pos / len(df):.1f}% positive)"
        )

    for ds_path, df in tqdm(dataset_results, desc="Writing results"):
        if args.output_dir:
            out_ds = args.output_dir.resolve() / ds_path.name
            out_ds.mkdir(parents=True, exist_ok=True)
            import shutil

            subdirs_to_copy = ["meta"] if args.skip_embed else ["meta", "data"]
            for sub in subdirs_to_copy:
                src = ds_path / sub
                if src.exists():
                    dst = out_ds / sub
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            src_mixture = ds_path / "mixture_config.yaml"
            if src_mixture.exists():
                shutil.copy2(src_mixture, out_ds / "mixture_config.yaml")
            target_path = out_ds
        else:
            target_path = ds_path

        meta_dir = target_path / "meta"
        data_dir = target_path / "data"
        meta_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_embed:
            if not data_dir.exists():
                logger.warning(
                    f"  No data/ dir at {target_path}, skipping parquet update"
                )
            else:
                add_advantages_to_parquet_files(
                    data_dir,
                    build_save_advantages_df(df, unified_threshold)
                    if not recompute
                    else df,
                    unified_threshold,
                    num_workers=num_workers,
                    adv_only=not recompute,
                )
        else:
            logger.info("  Skipping data parquet embedding (--skip_embed)")

        save_df = build_save_advantages_df(df, unified_threshold)
        adv_filename = (
            f"advantages_{output_tag}.parquet" if output_tag else "advantages.parquet"
        )
        save_df.to_parquet(meta_dir / adv_filename, index=False)
        logger.info(f"  Saved {adv_filename} ({len(save_df)} entries)")
        episodes_path = meta_dir / "episodes.jsonl"
        if episodes_path.exists():
            modify_episodes_with_advantage(
                episodes_path,
                episodes_path,
                df.copy(),
                unified_threshold,
            )
        else:
            logger.warning(f"  No episodes.jsonl at {meta_dir}")
        update_mixture_config(
            mixture_root=target_path,
            dataset_paths=dataset_paths,
            positive_quantile=args.positive_quantile,
            unified_threshold=unified_threshold,
            advantage_tag=output_tag,
            global_return_min=global_return_min,
            global_return_max=global_return_max,
        )

    logger.info("Recompute advantages complete.")


if __name__ == "__main__":
    main()
