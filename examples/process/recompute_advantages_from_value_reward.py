# Copyright 2025 The RLinf Authors.
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
  python recompute_advantages_from_value_reward.py \
    --dataset_root /path/to/transformed_advantage_dataset \
    --advantage_horizon 20 \
    --positive_quantile 0.3 \
    --num_workers 4
"""

# Suppress video decoder (libdav1d) logging — must be before any av/video imports
import os

os.environ["AV_LOG_LEVEL"] = "panic"
os.environ["LIBDAV1D_LOG_LEVEL"] = "0"

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _suppress_video_logging() -> None:
    """Suppress libdav1d/av logging (call in each worker process)."""
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


def _load_return_stats_from_dataset(
    dataset_path: Path,
) -> tuple[float | None, float | None]:
    """Load return min/max from dataset's meta/stats.json."""
    stats_path = dataset_path / "meta" / "stats.json"
    if not stats_path.exists():
        return None, None
    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return_stats = stats.get("return", {})
        return return_stats.get("min"), return_stats.get("max")
    except (json.JSONDecodeError, KeyError):
        return None, None


def discover_datasets_and_return_range(
    dataset_root: Path,
) -> tuple[list[Path], float, float]:
    """
    Discover sub-datasets under root and get global return range.

    Returns:
        (list of dataset paths, global_return_min, global_return_max)
    """
    mixture_path = dataset_root / "mixture_config.yaml"
    dataset_paths: list[Path] = []
    global_return_min: float | None = None
    global_return_max: float | None = None

    if mixture_path.exists():
        with open(mixture_path, "r") as f:
            mixture = yaml.safe_load(f) or {}
        names = []
        for d in mixture.get("datasets", []):
            if isinstance(d, dict) and "name" in d:
                names.append(d["name"])
            elif isinstance(d, str):
                names.append(d)
        global_return_min = mixture.get("global_return_min")
        global_return_max = mixture.get("global_return_max")
        for name in names:
            path = dataset_root / name
            if path.is_dir() and (path / "meta" / "info.json").exists():
                dataset_paths.append(path)
    if not dataset_paths:
        for child in dataset_root.iterdir():
            if child.is_dir() and (child / "meta" / "info.json").exists():
                dataset_paths.append(child)

    if not dataset_paths:
        raise ValueError(f"No datasets found under {dataset_root}")

    if global_return_min is None or global_return_max is None:
        all_mins, all_maxs = [], []
        for dp in dataset_paths:
            rmin, rmax = _load_return_stats_from_dataset(dp)
            if rmin is not None:
                all_mins.append(rmin)
            if rmax is not None:
                all_maxs.append(rmax)
        if all_mins:
            global_return_min = (
                min(all_mins) if global_return_min is None else global_return_min
            )
        if all_maxs:
            global_return_max = (
                max(all_maxs) if global_return_max is None else global_return_max
            )
    global_return_min = global_return_min if global_return_min is not None else -700.0
    global_return_max = global_return_max if global_return_max is not None else 0.0

    return dataset_paths, float(global_return_min), float(global_return_max)


def compute_advantages_for_dataset(
    dataset_path: Path,
    advantage_horizon: int,
    gamma: float,
    global_return_min: float,
    global_return_max: float,
    num_workers: int = 1,
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

    # Check available columns efficiently (schema only)
    sample_schema = pq.read_schema(str(parquet_files[0]))
    sample_cols = set(sample_schema.names)
    columns_needed = ["episode_index", "frame_index", "value_current", "reward"]
    has_return = "return" in sample_cols
    if has_return:
        columns_needed.append("return")

    # Read all parquet files directly (much faster than dataset[i] which decodes video)
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
    full_df["value_current"] = full_df["value_current"].fillna(0.0)
    full_df["reward"] = full_df["reward"].fillna(0.0)

    ret_range = global_return_max - global_return_min
    gamma_powers = np.array(
        [gamma**i for i in range(advantage_horizon)], dtype=np.float32
    )
    gamma_k = gamma**advantage_horizon

    results: list[pd.DataFrame] = []
    for ep_idx, ep_group in tqdm(
        full_df.groupby("episode_index"),
        desc=f"Advantages ({dataset_path.name})",
        leave=False,
    ):
        ep_df = ep_group.sort_values("frame_index").reset_index(drop=True)
        rewards = ep_df["reward"].values.astype(np.float32)
        values = ep_df["value_current"].values.astype(np.float64)
        frame_indices = ep_df["frame_index"].values
        returns_col = (
            ep_df["return"].values.astype(np.float64)
            if has_return
            else np.zeros(len(ep_df))
        )
        ep_len = len(ep_df)
        if ep_len == 0:
            continue

        # v_next: value at position i + advantage_horizon, else 0
        v_next_arr = np.zeros(ep_len, dtype=np.float64)
        cut = ep_len - advantage_horizon
        if cut > 0:
            v_next_arr[:cut] = values[advantage_horizon:]

        # Discounted reward sums via sliding window (vectorized)
        padded = np.concatenate(
            [rewards, np.zeros(advantage_horizon - 1, dtype=np.float32)]
        )
        padded = np.ascontiguousarray(padded)
        windowed = np.lib.stride_tricks.as_strided(
            padded,
            shape=(ep_len, advantage_horizon),
            strides=(padded.strides[0], padded.strides[0]),
        )
        reward_sums_raw = (windowed @ gamma_powers).astype(np.float64)

        # Normalize reward sums
        if ret_range <= 0:
            reward_sums = np.full(ep_len, -0.5)
        else:
            reward_sums = (reward_sums_raw - global_return_min) / ret_range - 1.0

        # advantage = normalized_reward_sum + gamma^N * v_next - v_curr
        # Note: when v_next=0 (near episode end), gamma_k * v_next = 0 regardless
        advantages = reward_sums + gamma_k * v_next_arr - values

        ep_result = pd.DataFrame(
            {
                "episode_index": np.full(ep_len, int(ep_idx), dtype=np.int64),
                "frame_index": frame_indices.astype(np.int64),
                "advantage": advantages,
                "reward_sum": reward_sums,
                "value_current": values,
                "value_next": v_next_arr,
                "return": returns_col,
            }
        )
        results.append(ep_result)

    if not results:
        return pd.DataFrame(
            columns=[
                "episode_index",
                "frame_index",
                "advantage",
                "reward_sum",
                "value_current",
                "value_next",
                "return",
            ]
        )
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
    vc_arr: np.ndarray,
    vn_arr: np.ndarray,
    rs_arr: np.ndarray,
    ret_arr: np.ndarray,
    threshold: float,
) -> tuple[int, int]:
    """Update a single parquet file with advantage columns. Returns (updated, missing)."""
    from datasets import Dataset

    _suppress_video_logging()

    ds = Dataset.from_parquet(str(pq_file))
    if len(ds) == 0:
        return 0, 0

    ds_ep_indices = ds["episode_index"]
    ds_frame_indices = ds["frame_index"]

    advantages = []
    advantage_continuous = []
    value_current_list = []
    value_next_list = []
    reward_sum_list = []
    return_list = []
    updated = 0
    missing = 0

    for ep_idx, frame_idx in zip(ds_ep_indices, ds_frame_indices):
        key = (int(ep_idx), int(frame_idx))
        if key in advantage_lookup:
            i = advantage_lookup[key]
            advantages.append(float(adv_cont_arr[i]) >= threshold)
            advantage_continuous.append(float(adv_cont_arr[i]))
            value_current_list.append(float(vc_arr[i]))
            value_next_list.append(float(vn_arr[i]))
            reward_sum_list.append(float(rs_arr[i]))
            return_list.append(float(ret_arr[i]))
            updated += 1
        else:
            advantages.append(False)
            advantage_continuous.append(threshold - 0.1)
            value_current_list.append(0.0)
            value_next_list.append(0.0)
            reward_sum_list.append(0.0)
            return_list.append(0.0)
            missing += 1

    columns_to_add = {
        "advantage": advantages,
        "advantage_continuous": advantage_continuous,
        "value_current": value_current_list,
        "value_next": value_next_list,
        "reward_sum": reward_sum_list,
        "return": return_list,
    }
    existing = set(ds.column_names)
    for col in columns_to_add:
        if col in existing:
            ds = ds.remove_columns([col])
    for col, data in columns_to_add.items():
        ds = ds.add_column(col, data)
    ds.to_parquet(str(pq_file))
    return updated, missing


def add_advantages_to_parquet_files(
    data_dir: Path,
    advantages_df: pd.DataFrame,
    threshold: float,
    num_workers: int = 1,
) -> tuple[int, int]:
    """Write advantage (bool), advantage_continuous, reward_sum, value_current, value_next to parquet."""
    adv_continuous_col = advantages_df["advantage"]
    advantages_df = advantages_df.copy()
    advantages_df["advantage_continuous"] = adv_continuous_col

    # Build lookup using vectorized column access (much faster than iterrows)
    ep_arr = advantages_df["episode_index"].values.astype(np.int64)
    fr_arr = advantages_df["frame_index"].values.astype(np.int64)
    adv_cont_arr = advantages_df["advantage_continuous"].values
    vc_arr = advantages_df["value_current"].values
    vn_arr = advantages_df["value_next"].values
    rs_arr = advantages_df["reward_sum"].values
    ret_arr = advantages_df["return"].values

    advantage_lookup: dict[tuple[int, int], int] = {}
    for i in range(len(advantages_df)):
        advantage_lookup[(int(ep_arr[i]), int(fr_arr[i]))] = i

    parquet_files = list(data_dir.rglob("*.parquet"))
    logger.info(f"  Found {len(parquet_files)} parquet files to update")
    total_updated = 0
    total_missing = 0

    effective_workers = min(num_workers, len(parquet_files))

    if effective_workers > 1:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = {
                executor.submit(
                    _update_single_parquet_file,
                    pq_file,
                    advantage_lookup,
                    adv_cont_arr,
                    vc_arr,
                    vn_arr,
                    rs_arr,
                    ret_arr,
                    threshold,
                ): pq_file
                for pq_file in parquet_files
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="  Updating parquet",
                leave=False,
            ):
                updated, missing = future.result()
                total_updated += updated
                total_missing += missing
    else:
        for pq_file in tqdm(parquet_files, desc="  Updating parquet", leave=False):
            updated, missing = _update_single_parquet_file(
                pq_file,
                advantage_lookup,
                adv_cont_arr,
                vc_arr,
                vn_arr,
                rs_arr,
                ret_arr,
                threshold,
            )
            total_updated += updated
            total_missing += missing

    logger.info(f"  Updated {total_updated} samples with advantages")
    if total_missing > 0:
        logger.warning(f"  {total_missing} samples missing advantage (set to negative)")
    return total_updated, total_missing


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
        required=True,
        help="Root path containing multiple LeRobot datasets (e.g. transformed_advantage_dataset_*)",
    )
    parser.add_argument(
        "--advantage_horizon",
        type=int,
        default=10,
        help="N steps for reward sum and v_next (value_current at t+N). Default 10.",
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
        "--advantage_tag",
        type=str,
        default=None,
        help="Tag for this advantage computation. Saves to meta/advantages_{tag}.parquet. Default: None (saves as advantages.parquet).",
    )
    parser.add_argument(
        "--skip_embed",
        action="store_true",
        help="Skip embedding advantages into data parquet files. Only save meta parquet.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset_paths, global_return_min, global_return_max = (
        discover_datasets_and_return_range(dataset_root)
    )
    logger.info(
        f"Found {len(dataset_paths)} datasets: {[p.name for p in dataset_paths]}"
    )
    logger.info(f"Global return range: [{global_return_min}, {global_return_max}]")
    logger.info(f"Advantage horizon: {args.advantage_horizon}, gamma: {args.gamma}")
    logger.info(f"Positive quantile: {args.positive_quantile}")

    all_advantages: list[np.ndarray] = []
    dataset_results: list[tuple[Path, pd.DataFrame]] = []

    num_workers = args.num_workers

    logger.info(
        f"Processing {len(dataset_paths)} datasets with {num_workers} workers "
        f"(parallelized within each dataset for parquet read/write)"
    )
    for ds_path in tqdm(dataset_paths, desc="Processing datasets"):
        df = compute_advantages_for_dataset(
            ds_path,
            args.advantage_horizon,
            args.gamma,
            global_return_min,
            global_return_max,
            num_workers=num_workers,
        )
        dataset_results.append((ds_path, df))
        all_advantages.append(df["advantage"].values)
        logger.info(
            f"  {ds_path.name}: {len(df)} advantages, mean={df['advantage'].mean():.4f}"
        )

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

    output_root = args.output_dir.resolve() if args.output_dir else dataset_root
    if args.output_dir:
        output_root.mkdir(parents=True, exist_ok=True)

    for ds_path, df in tqdm(dataset_results, desc="Writing results"):
        if args.output_dir:
            out_ds = output_root / ds_path.name
            out_ds.mkdir(parents=True, exist_ok=True)
            import shutil

            for sub in ["meta", "data"]:
                src = ds_path / sub
                if src.exists():
                    dst = out_ds / sub
                    if dst.exists():
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
            target_path = out_ds
        else:
            target_path = ds_path

        meta_dir = target_path / "meta"
        data_dir = target_path / "data"
        if not args.skip_embed:
            if not data_dir.exists():
                logger.warning(
                    f"  No data/ dir at {target_path}, skipping parquet update"
                )
            else:
                add_advantages_to_parquet_files(
                    data_dir, df, unified_threshold, num_workers=num_workers
                )
        else:
            logger.info("  Skipping data parquet embedding (--skip_embed)")

        save_df = df.copy()
        save_df.rename(columns={"advantage": "advantage_continuous"}, inplace=True)
        save_df["advantage"] = save_df["advantage_continuous"] >= unified_threshold
        adv_filename = (
            f"advantages_{args.advantage_tag}.parquet"
            if args.advantage_tag
            else "advantages.parquet"
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

    mixture_path = output_root / "mixture_config.yaml"
    if mixture_path.exists():
        with open(mixture_path, "r") as f:
            mixture = yaml.safe_load(f) or {}
    else:
        mixture = {}
    mixture["unified_threshold"] = unified_threshold
    mixture["positive_quantile"] = args.positive_quantile
    mixture["global_return_min"] = global_return_min
    mixture["global_return_max"] = global_return_max
    if "datasets" not in mixture:
        mixture["datasets"] = [{"name": p.name, "weight": 1.0} for p in dataset_paths]
    with open(output_root / "mixture_config.yaml", "w") as f:
        yaml.dump(mixture, f, default_flow_style=False)
    logger.info(
        f"Saved mixture_config.yaml with unified_threshold={unified_threshold:.4f}"
    )

    logger.info("Recompute advantages complete.")


if __name__ == "__main__":
    main()
