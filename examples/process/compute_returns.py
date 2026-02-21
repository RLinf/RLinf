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
Compute returns for LeRobot datasets.

This preprocessing script adds `return`, `reward`, and `prompt` columns to
LeRobot datasets. After running this script, the dataset can be used with
DynamicReturnDataset which reads the precomputed returns and normalizes them
using global min/max across all datasets at training time.

Key features:
- SFT datasets: reward=-1 per step, last step=0, all episodes successful
- Rollout datasets: reward=-1 per step, last step=0 (success) or failure_reward (failure)
- Returns computed via backward iteration: G_t = r_t + gamma * G_{t+1}
- Updates meta/stats.json with return/reward statistics (mean, std, min, max)
- Updates meta/info.json with new feature definitions

Note: Normalization (target_values) is NOT done here. It's done at training time
using global return_min/return_max computed from all datasets' stats.json.

Usage:
    python compute_returns.py --config-name compute_returns \
        data.dataset_path=/path/to/dataset \
        data.dataset_type=sft \
        data.gamma=1.0
"""

import json
import logging
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import hydra
import numpy as np
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


def compute_returns_for_episode(
    episode_length: int,
    is_success: bool,
    gamma: float,
    failure_reward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute returns and rewards for a single episode.

    Uses the DynamicReturn approach:
    - Each step reward = -1
    - Success at episode end: last step reward = 0
    - Failure at episode end: last step reward = failure_reward

    Returns are computed via backward iteration: G_t = r_t + gamma * G_{t+1}

    Args:
        episode_length: Total length of episode
        is_success: Whether episode was successful
        gamma: Discount factor (default 1.0 for undiscounted)
        failure_reward: Penalty for failure (default -300.0)

    Returns:
        Tuple of (returns array, rewards array) for all steps
    """
    rewards = np.full(episode_length, -1.0, dtype=np.float32)
    rewards[-1] = 0.0 if is_success else failure_reward

    # Compute returns via backward iteration: G_t = r_t + gamma * G_{t+1}
    returns = np.zeros(episode_length, dtype=np.float32)
    returns[-1] = rewards[-1]
    for t in range(episode_length - 2, -1, -1):
        returns[t] = rewards[t] + gamma * returns[t + 1]

    return returns, rewards


def get_episode_boundaries(episode_indices: np.ndarray) -> list[tuple[int, int, int]]:
    """Extract episode boundaries from episode_index array.

    Args:
        episode_indices: 1-D numpy array of episode indices.

    Returns:
        List of (episode_index, start_idx, end_idx) tuples
    """
    if len(episode_indices) == 0:
        return []

    # Find positions where episode_index changes
    change_mask = np.diff(episode_indices) != 0
    change_positions = np.where(change_mask)[0] + 1  # +1: index of new ep

    # Build boundaries: [0, pos1, pos2, ..., len]
    starts = np.concatenate([[0], change_positions])
    ends = np.concatenate([change_positions, [len(episode_indices)]])
    ep_ids = episode_indices[starts]

    return list(zip(ep_ids.tolist(), starts.tolist(), ends.tolist()))


def _process_single_parquet(
    pq_file: str,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    tasks: dict[int, str],
) -> tuple[str, np.ndarray, np.ndarray]:
    """Process a single parquet file: compute returns/rewards/prompts and save.

    Designed to run in a worker process.

    Returns:
        (file_path, returns_array, rewards_array)
    """
    ds = Dataset.from_parquet(pq_file)
    n = len(ds)
    if n == 0:
        return pq_file, np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # ---- columnar access (fast) instead of per-sample random access ----
    ep_indices = np.array(ds["episode_index"], dtype=np.int64)
    episodes = get_episode_boundaries(ep_indices)

    # Batch-load is_success column once (if exists)
    has_is_success = "is_success" in ds.column_names
    is_success_col = None
    if has_is_success:
        is_success_col = ds["is_success"]
        if hasattr(is_success_col, "to_pylist"):
            is_success_col = is_success_col.to_pylist()

    # ---- compute returns & rewards per episode ----
    returns_arr = np.empty(n, dtype=np.float32)
    rewards_arr = np.empty(n, dtype=np.float32)

    for _, ep_start, ep_end in episodes:
        ep_length = ep_end - ep_start

        if dataset_type == "sft":
            is_success = True
        elif is_success_col is not None:
            is_success = bool(is_success_col[ep_end - 1])
        else:
            is_success = False

        ep_returns, ep_rewards = compute_returns_for_episode(
            episode_length=ep_length,
            is_success=is_success,
            gamma=gamma,
            failure_reward=failure_reward,
        )
        returns_arr[ep_start:ep_end] = ep_returns
        rewards_arr[ep_start:ep_end] = ep_rewards

    # ---- build prompts via columnar access ----
    if "task" in ds.column_names:
        prompts_list = [str(t) for t in ds["task"]]
    elif "task_index" in ds.column_names:
        task_indices = ds["task_index"]
        if hasattr(task_indices, "to_pylist"):
            task_indices = task_indices.to_pylist()
        prompts_list = [tasks.get(int(idx), "perform the task") for idx in task_indices]
    else:
        prompts_list = ["perform the task"] * n

    # ---- write back to parquet ----
    for col_name, col_data in [
        ("return", returns_arr.tolist()),
        ("reward", rewards_arr.tolist()),
        ("prompt", prompts_list),
    ]:
        if col_name in ds.column_names:
            ds = ds.remove_columns(col_name)
        ds = ds.add_column(col_name, col_data)

    ds.to_parquet(pq_file)
    return pq_file, returns_arr, rewards_arr


def process_dataset(
    dataset_path: Path,
    output_path: Path | None,
    dataset_type: str,
    gamma: float,
    failure_reward: float,
    num_workers: int = 8,
) -> dict:
    """Process a LeRobot dataset and add return/reward/prompt columns.

    Args:
        dataset_path: Path to input dataset
        output_path: Path to output dataset (or None to modify in-place)
        dataset_type: "sft" or "rollout"
        gamma: Discount factor
        failure_reward: Penalty for failed episodes
        num_workers: Number of parallel workers for parquet processing

    Returns:
        Statistics dict with return/reward stats
    """
    logger.info(f"Processing dataset: {dataset_path}")
    logger.info(
        f"  Type: {dataset_type}, Gamma: {gamma}, Failure reward: {failure_reward}"
    )

    # Determine output location
    if output_path is None:
        output_path = dataset_path
    else:
        # Copy dataset to output location
        if output_path.exists():
            logger.warning(f"Removing existing output: {output_path}")
            shutil.rmtree(output_path)
        shutil.copytree(dataset_path, output_path)
        logger.info(f"Copied dataset to: {output_path}")

    # Find all parquet files
    data_dir = output_path / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    parquet_files = sorted(str(p) for p in data_dir.rglob("*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files")

    # Load tasks for prompt conversion
    tasks: dict[int, str] = {}
    tasks_path = output_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                task_idx = entry.get("task_index", len(tasks))
                task_desc = entry.get("task", "")
                tasks[task_idx] = task_desc

    # ---- parallel processing of parquet files ----
    all_returns: list[np.ndarray] = []
    all_rewards: list[np.ndarray] = []

    effective_workers = min(num_workers, len(parquet_files))
    if effective_workers <= 1:
        # Single-worker fast path (no process overhead)
        for pq_file in tqdm(parquet_files, desc="Processing parquet files"):
            _, ret, rew = _process_single_parquet(
                pq_file, dataset_type, gamma, failure_reward, tasks
            )
            if len(ret) > 0:
                all_returns.append(ret)
                all_rewards.append(rew)
    else:
        logger.info(f"Using {effective_workers} parallel workers")
        futures = {}
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            for pq_file in parquet_files:
                fut = pool.submit(
                    _process_single_parquet,
                    pq_file,
                    dataset_type,
                    gamma,
                    failure_reward,
                    tasks,
                )
                futures[fut] = pq_file

            with tqdm(total=len(futures), desc="Processing parquet files") as pbar:
                for fut in as_completed(futures):
                    _, ret, rew = fut.result()
                    if len(ret) > 0:
                        all_returns.append(ret)
                        all_rewards.append(rew)
                    pbar.update(1)

    # Compute statistics
    returns_arr = np.concatenate(all_returns) if all_returns else np.array([])
    rewards_arr = np.concatenate(all_rewards) if all_rewards else np.array([])

    stats = {
        "return": {
            "mean": float(returns_arr.mean()),
            "std": float(returns_arr.std()),
            "min": float(returns_arr.min()),
            "max": float(returns_arr.max()),
        },
        "reward": {
            "mean": float(rewards_arr.mean()),
            "std": float(rewards_arr.std()),
            "min": float(rewards_arr.min()),
            "max": float(rewards_arr.max()),
        },
    }

    logger.info("\nStatistics:")
    logger.info(
        f"  Return: mean={stats['return']['mean']:.4f}, std={stats['return']['std']:.4f}, "
        f"min={stats['return']['min']:.4f}, max={stats['return']['max']:.4f}"
    )
    logger.info(
        f"  Reward: mean={stats['reward']['mean']:.4f}, std={stats['reward']['std']:.4f}, "
        f"min={stats['reward']['min']:.4f}, max={stats['reward']['max']:.4f}"
    )

    # Update meta/stats.json
    stats_path = output_path / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            existing_stats = json.load(f)
    else:
        existing_stats = {}

    existing_stats["return"] = stats["return"]
    existing_stats["reward"] = stats["reward"]

    with open(stats_path, "w") as f:
        json.dump(existing_stats, f, indent=2)
    logger.info("Updated stats.json")

    # Update meta/info.json with new feature definitions
    info_path = output_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

        info["features"]["return"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        info["features"]["reward"] = {
            "dtype": "float32",
            "shape": [1],
            "names": None,
        }
        info["features"]["prompt"] = {
            "dtype": "string",
            "shape": [1],
            "names": None,
        }

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        logger.info("Updated info.json with new features")

    return stats


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="compute_returns",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for return computation."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting return computation...")
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get global defaults
    data_root = cfg.data.get("data_root", None)
    default_type = cfg.data.get("dataset_type", "sft")
    default_gamma = cfg.data.get("gamma", 1.0)
    default_failure_reward = cfg.data.get("failure_reward", -300.0)
    num_workers = cfg.data.get("num_workers", 8)

    # Get datasets list
    datasets_list = cfg.data.get("datasets", None)

    # Build list of datasets to process
    datasets_to_process = []

    if datasets_list is not None and len(datasets_list) > 0:
        # Process multiple datasets from list
        for entry in datasets_list:
            entry = dict(entry)
            ds_path = entry.get("dataset_path")
            if ds_path is None:
                raise ValueError("Each dataset entry must have 'dataset_path'")

            # Resolve relative path with data_root
            if data_root and not Path(ds_path).is_absolute():
                ds_path = str(Path(data_root) / ds_path)

            output_path = entry.get("output_path", None)
            if output_path and data_root and not Path(output_path).is_absolute():
                output_path = str(Path(data_root) / output_path)

            datasets_to_process.append(
                {
                    "dataset_path": ds_path,
                    "output_path": output_path,
                    "dataset_type": entry.get("type", default_type),
                    "gamma": entry.get("gamma", default_gamma),
                    "failure_reward": entry.get(
                        "failure_reward", default_failure_reward
                    ),
                }
            )
    else:
        # Backward compatibility: single dataset_path
        dataset_path = cfg.data.get("dataset_path", None)
        if dataset_path is None:
            raise ValueError(
                "No datasets specified. Either set data.datasets list or data.dataset_path"
            )

        if data_root and not Path(dataset_path).is_absolute():
            dataset_path = str(Path(data_root) / dataset_path)

        output_path = cfg.data.get("output_path", None)
        if output_path and data_root and not Path(output_path).is_absolute():
            output_path = str(Path(data_root) / output_path)

        datasets_to_process.append(
            {
                "dataset_path": dataset_path,
                "output_path": output_path,
                "dataset_type": default_type,
                "gamma": default_gamma,
                "failure_reward": default_failure_reward,
            }
        )

    logger.info(f"Processing {len(datasets_to_process)} dataset(s)...")

    # Process each dataset
    all_stats = []
    for i, ds_config in enumerate(datasets_to_process):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Dataset {i + 1}/{len(datasets_to_process)}")
        logger.info(f"{'=' * 60}")

        dataset_path = Path(ds_config["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        output_path = ds_config["output_path"]
        if output_path is not None:
            output_path = Path(output_path)

        stats = process_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            dataset_type=ds_config["dataset_type"],
            gamma=ds_config["gamma"],
            failure_reward=ds_config["failure_reward"],
            num_workers=num_workers,
        )
        all_stats.append(
            {
                "path": str(output_path if output_path else dataset_path),
                "stats": stats,
            }
        )

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("Return computation complete!")
    logger.info(f"{'=' * 60}")
    logger.info(f"Processed {len(all_stats)} dataset(s):")
    for ds_stats in all_stats:
        ret_stats = ds_stats["stats"]["return"]
        logger.info(f"  {ds_stats['path']}")
        logger.info(
            f"    return: min={ret_stats['min']:.2f}, max={ret_stats['max']:.2f}"
        )


if __name__ == "__main__":
    main()
