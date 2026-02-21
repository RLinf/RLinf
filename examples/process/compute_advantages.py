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
Compute advantages for CFG-RL training using a trained ValuePolicy.

Advantage formula: A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t)

This script:
1. Loads a trained ValuePolicy from checkpoint (with input transforms)
2. Loads LeRobot datasets with delta_timestamps for multi-step data
3. Computes N-step discounted reward sum and values
4. Creates independent output datasets with is_success = (advantage >= threshold)

The ValuePolicy encapsulates all input transforms (LiberoInputs, Normalize, ResizeImages, etc.)
so inference uses the same data preprocessing as training.

Supports multi-GPU parallel processing via torchrun:
    torchrun --nproc_per_node=N compute_advantages.py --config-name compute_advantages

Usage:
    # Single GPU
    python compute_advantages.py --config-name compute_advantages \
        advantage.value_checkpoint=/path/to/checkpoint

    # Multi-GPU (via torchrun)
    torchrun --nproc_per_node=4 compute_advantages.py --config-name compute_advantages \
        advantage.value_checkpoint=/path/to/checkpoint
"""

import gc
import json
import logging
import os

# Disable tokenizers parallelism to avoid warning when using multiprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure rlinf is importable
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from rlinf for value policy with batch inference support
from rlinf.models.embodiment.vla_lib_value_model import value_policy_config

logger = logging.getLogger(__name__)


# =============================================================================
# Distributed Utilities
# =============================================================================


def setup_distributed(cfg: DictConfig) -> tuple[int, int, str]:
    """Initialize torch.distributed for torchrun-launched processes.

    Args:
        cfg: Configuration with distributed settings

    Returns:
        Tuple of (rank, world_size, device_string)
    """
    # Check if we're running under torchrun (or similar launcher)
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Get distributed config settings
        dist_cfg = cfg.get("distributed", {})
        backend = dist_cfg.get("backend", "nccl")
        timeout_seconds = dist_cfg.get("timeout", 1800)

        # Initialize process group
        if not dist.is_initialized():
            from datetime import timedelta

            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_seconds),
            )

        # Set device for this process
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        if rank == 0:
            logger.info(f"Distributed mode enabled: {world_size} GPUs")
            logger.info(f"  Backend: {backend}, Timeout: {timeout_seconds}s")

        return rank, world_size, device

    # Single GPU fallback
    return 0, 1, "cuda"


def cleanup_distributed():
    """Clean up distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_shard_indices(
    total_samples: int, rank: int, world_size: int
) -> tuple[int, int]:
    """Calculate start/end indices for this rank's shard.

    Distributes samples as evenly as possible, with earlier ranks
    getting one extra sample if there's a remainder.

    Args:
        total_samples: Total number of samples
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Tuple of (start_index, end_index) where end is exclusive
    """
    base_count = total_samples // world_size
    remainder = total_samples % world_size

    if rank < remainder:
        start = rank * (base_count + 1)
        end = start + base_count + 1
    else:
        start = remainder * (base_count + 1) + (rank - remainder) * base_count
        end = start + base_count

    return start, end


def gather_all_advantages(
    local_df: pd.DataFrame,
    rank: int,
    world_size: int,
) -> pd.DataFrame:
    """Gather advantages from all ranks using all_gather_object.

    Args:
        local_df: Local DataFrame with advantages for this rank's shard
        rank: Current process rank
        world_size: Total number of processes

    Returns:
        Merged DataFrame with all advantages, sorted by (episode_index, frame_index)
    """
    if world_size == 1:
        return local_df

    # Gather DataFrames from all ranks
    all_dfs = [None] * world_size
    dist.all_gather_object(all_dfs, local_df.to_dict("records"))

    # Merge all records
    all_records = []
    for df_records in all_dfs:
        if df_records:
            all_records.extend(df_records)

    # Create merged DataFrame and sort
    merged_df = pd.DataFrame(all_records)
    if len(merged_df) > 0:
        merged_df = merged_df.sort_values(["episode_index", "frame_index"]).reset_index(
            drop=True
        )

    return merged_df


# =============================================================================
# Key Mappings for Different Robot Types
# =============================================================================

# Observation-like keys that need delta_timestamps (both prefixed and non-prefixed)
OBSERVATION_LIKE_KEYS = {
    # Non-prefixed (collected_data format)
    "image",
    "wrist_image",
    "front_image",
    "state",
    # Prefixed (standard LeRobot format)
    "observation.image",
    "observation.wrist_image",
    "observation.state",
    "observation.images.front_cam",
    "observation.images.wrist_cam",
    "observation.images.left_cam",
    "observation.images.right_cam",
    "observation.state.tcp_pose",
    "observation.state.gripper_pose",
    "observation.exterior_image_1_left",
    "observation.wrist_image_left",
    "observation.joint_position",
    "observation.gripper_position",
}

# Key mappings for building raw observations for ValuePolicy
# Maps LeRobot dataset keys to vla_lib observation format
KEY_MAPPINGS = {
    "franka": {
        "observation.images.front_cam": "observation/images/front_cam",
        "observation.images.wrist_cam": "observation/images/wrist_cam",
        "observation.state.tcp_pose": "observation/state/tcp_pose",
        "observation.state.gripper_pose": "observation/state/gripper_pose",
        "task": "prompt",
    },
    "franka_3cam": {
        "observation.images.left_cam": "observation/images/left_cam",
        "observation.images.right_cam": "observation/images/right_cam",
        "observation.images.wrist_cam": "observation/images/wrist_cam",
        "observation.state.tcp_pose": "observation/state/tcp_pose",
        "observation.state.gripper_pose": "observation/state/gripper_pose",
        "task": "prompt",
    },
    "libero": {
        # Prefixed format (standard LeRobot)
        "observation.image": "observation/image",
        "observation.wrist_image": "observation/wrist_image",
        "observation.state": "observation/state",
        # Non-prefixed format (collected_data)
        "image": "observation/image",
        "wrist_image": "observation/wrist_image",
        "state": "observation/state",
        "task": "prompt",
    },
    "droid": {
        "observation.exterior_image_1_left": "observation/exterior_image_1_left",
        "observation.wrist_image_left": "observation/wrist_image_left",
        "observation.joint_position": "observation/joint_position",
        "observation.gripper_position": "observation/gripper_position",
        "task": "prompt",
    },
}


# =============================================================================
# Utility Functions
# =============================================================================


def to_numpy(x):
    """Convert tensor or array to numpy."""
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.array(x)


def to_scalar(x):
    """Convert to Python scalar."""
    if hasattr(x, "item"):
        return x.item()
    return x


class RunningStats:
    """Online statistics using Welford's algorithm (memory-efficient).

    Computes mean, std, min, max incrementally without storing all values.
    This avoids OOM when processing millions of samples.
    """

    def __init__(self, name: str = ""):
        self.name = name
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")

    def update(self, x: float):
        """Update statistics with a single value."""
        self.n += 1
        delta = x - self._mean
        self._mean += delta / self.n
        delta2 = x - self._mean
        self._m2 += delta * delta2
        if x < self._min:
            self._min = x
        if x > self._max:
            self._max = x

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return (self._m2 / (self.n - 1)) ** 0.5

    @property
    def min(self) -> float:
        return self._min

    @property
    def max(self) -> float:
        return self._max

    def summary(self) -> str:
        return (
            f"mean={self.mean:.4f}, std={self.std:.4f}, "
            f"min={self.min:.4f}, max={self.max:.4f}"
        )


def _load_return_stats_from_dataset(
    dataset_path: Path,
) -> tuple[float | None, float | None]:
    """Load return min/max from dataset's stats.json.

    Args:
        dataset_path: Path to LeRobot dataset

    Returns:
        Tuple of (return_min, return_max), or (None, None) if not found
    """
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


# =============================================================================
# Model Loading
# =============================================================================


def load_value_policy(cfg: DictConfig, device: str = "cuda"):
    """Load trained ValuePolicy from checkpoint.

    The ValuePolicy encapsulates all input transforms (LiberoInputs, Normalize,
    ResizeImages, etc.) so inference uses the same data preprocessing as training.

    Args:
        cfg: Config with checkpoint path and model settings
        device: Target device

    Returns:
        Configured ValuePolicy instance
    """
    checkpoint_path = cfg.advantage.value_checkpoint
    if checkpoint_path is None:
        raise ValueError("advantage.value_checkpoint must be specified")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading value policy from: {checkpoint_path}")

    # Extract configuration
    adv_cfg = cfg.advantage
    data_cfg = cfg.data
    model_cfg = adv_cfg.get("model", {})

    # Get robot_type (env_type for vla_lib)
    robot_type = data_cfg.get("robot_type", "libero")
    # If datasets are specified, use robot_type from first dataset
    if "datasets" in data_cfg and len(data_cfg.datasets) > 0:
        robot_type = data_cfg.datasets[0].get("robot_type", robot_type)

    # Get model_type (pi0, pi05)
    model_type = data_cfg.get("model_type", "pi05")

    # Get value_mode (expert_categorical, expert_mse, continuous, discrete)
    value_mode = adv_cfg.get("value_mode", "expert_categorical")

    # Get value head settings
    num_return_bins = model_cfg.get("num_bins", 201)
    return_min = model_cfg.get("v_min", -1.0)
    return_max = model_cfg.get("v_max", 0.0)
    critic_expert_variant = model_cfg.get("critic_expert_variant", "gemma_100m")

    logger.info(f"  env_type (robot_type): {robot_type}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  value_mode: {value_mode}")
    logger.info(f"  num_return_bins: {num_return_bins}")
    logger.info(f"  return_range: [{return_min}, {return_max}]")
    logger.info(f"  critic_expert_variant: {critic_expert_variant}")

    # Create ValuePolicy using vla_lib's factory function
    # This properly sets up all input transforms (LiberoInputs, Normalize, ResizeImages, etc.)
    value_policy = value_policy_config.create_trained_value_policy(
        checkpoint_dir=checkpoint_path,
        env_type=robot_type,
        model_type=model_type,
        device=device,
        value_mode=value_mode,
        num_return_bins=num_return_bins,
        return_min=return_min,
        return_max=return_max,
        critic_expert_variant=critic_expert_variant,
    )

    logger.info(f"Loaded ValuePolicy: {value_policy.metadata}")
    return value_policy


# =============================================================================
# Dataset Loading with delta_timestamps
# =============================================================================


def get_observation_keys(meta: LeRobotDatasetMetadata) -> list[str]:
    """Get observation keys that need delta_timestamps.

    Supports both prefixed (observation.image) and non-prefixed (image) formats.

    Args:
        meta: Dataset metadata

    Returns:
        List of observation key names
    """
    obs_keys = []
    for k in meta.features:
        if k.startswith("observation."):
            obs_keys.append(k)
        elif k in OBSERVATION_LIKE_KEYS:
            obs_keys.append(k)
    return obs_keys


def load_lerobot_dataset(
    dataset_path: Path,
    action_horizon: int,
) -> tuple[LeRobotDataset, dict, LeRobotDatasetMetadata]:
    """Load a LeRobot dataset with delta_timestamps for multi-step data.

    Args:
        dataset_path: Path to dataset
        action_horizon: Number of steps (N) for reward and next observation

    Returns:
        Tuple of (dataset, tasks_dict, metadata)
    """
    meta = LeRobotDatasetMetadata(str(dataset_path))

    # Log dataset features for debugging
    logger.info(f"Dataset features: {list(meta.features.keys())}")

    # Get observation keys (supports both prefixed and non-prefixed formats)
    obs_keys = get_observation_keys(meta)

    if not obs_keys:
        logger.warning(
            f"No observation keys found in dataset! "
            f"Features: {list(meta.features.keys())}"
        )

    # Build delta_timestamps: get t and t+N for observations
    delta_timestamps = {
        k: [0.0, action_horizon / meta.fps]  # [t, t+N]
        for k in obs_keys
    }

    # Add reward timestamps: get reward from t to t+N-1
    # NOTE: Dataset must be preprocessed with compute_returns.py to have reward column
    has_reward = "reward" in meta.features
    has_return = "return" in meta.features

    if not has_reward:
        raise ValueError(
            f"Dataset {dataset_path} missing 'reward' column. "
            "Run compute_returns.py to preprocess the dataset first."
        )

    if not has_return:
        raise ValueError(
            f"Dataset {dataset_path} missing 'return' column. "
            "Run compute_returns.py to preprocess the dataset first."
        )

    delta_timestamps["reward"] = [i / meta.fps for i in range(action_horizon)]

    logger.info("Loading dataset with delta_timestamps:")
    logger.info(f"  Dataset path: {dataset_path}")
    logger.info(f"  FPS: {meta.fps}")
    logger.info(f"  Observation keys: {obs_keys}")
    logger.info(f"  Has precomputed reward: {has_reward}")
    logger.info(f"  Has precomputed return: {has_return}")
    logger.info(f"  delta_timestamps: {delta_timestamps}")

    dataset = LeRobotDataset(
        str(dataset_path),
        delta_timestamps=delta_timestamps,
        download_videos=False,
    )

    # Load tasks
    tasks = {}
    tasks_path = dataset_path / "meta" / "tasks.jsonl"
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                entry = json.loads(line.strip())
                task_idx = entry.get("task_index", len(tasks))
                task_desc = entry.get("task", "")
                tasks[task_idx] = task_desc

    logger.info(
        f"Loaded dataset: {len(dataset)} samples, {meta.total_episodes} episodes"
    )

    # Debug: log first sample keys
    if len(dataset) > 0:
        first_sample = dataset[0]
        logger.debug(f"First sample keys: {list(first_sample.keys())}")
        for k, v in first_sample.items():
            if hasattr(v, "shape"):
                logger.debug(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            elif hasattr(v, "__len__") and not isinstance(v, str):
                logger.debug(f"  {k}: len={len(v)}, type={type(v).__name__}")
            else:
                logger.debug(f"  {k}: {v}")

    return dataset, tasks, meta


# =============================================================================
# Observation Building for Value Policy
# =============================================================================


def build_obs(
    sample: dict,
    robot_type: str,
    tasks: dict,
    use_next: bool,
) -> dict[str, Any]:
    """Build raw observation dict for ValuePolicy.

    The ValuePolicy internally handles all input transforms (LiberoInputs,
    Normalize, ResizeImages, etc.), so we only need to build the raw observation
    in the format expected by vla_lib's input processing.

    Args:
        sample: Sample from LeRobot dataset
        robot_type: Robot type for key mapping
        tasks: Task descriptions dict
        use_next: If True, use t+N observation; if False, use t observation

    Returns:
        Raw observation dict compatible with ValuePolicy.infer()
    """
    key_map = KEY_MAPPINGS.get(robot_type, KEY_MAPPINGS["libero"])
    obs = {}
    obs_idx = 1 if use_next else 0  # delta_timestamps: [t, t+N]

    for src_key, dst_key in key_map.items():
        if src_key == "task":
            # Handle prompt
            if "task" in sample:
                obs[dst_key] = str(to_scalar(sample["task"]))
            elif "task_index" in sample and tasks:
                task_idx = int(to_scalar(sample["task_index"]))
                obs[dst_key] = tasks.get(task_idx, "")
            else:
                obs[dst_key] = ""
        elif src_key in sample:
            val = to_numpy(sample[src_key])
            # Check if this is a multi-timestep observation
            if val.ndim >= 1 and val.shape[0] >= 2:
                val = val[obs_idx]
            obs[dst_key] = val

    return obs


def get_episode_info(
    dataset: LeRobotDataset,
    meta: LeRobotDatasetMetadata,
) -> dict[int, dict]:
    """Extract episode information from dataset metadata.

    NOTE: In distributed mode, all ranks call this function with the same dataset.
    Since episode_data_index is deterministic, all ranks get identical episode_info.
    This consistency is important for correct advantage computation.

    Args:
        dataset: LeRobot dataset
        meta: Dataset metadata

    Returns:
        Dict mapping episode_index to {length, is_success}
    """
    episode_info = {}

    # Get episode data indices from dataset (not metadata)
    # episode_data_index is an attribute of LeRobotDataset
    for ep_idx in range(meta.total_episodes):
        ep_start = dataset.episode_data_index["from"][ep_idx].item()
        ep_end = dataset.episode_data_index["to"][ep_idx].item()
        ep_length = ep_end - ep_start

        # Try to get is_success from episodes.jsonl
        is_success = False  # Default to False

        episode_info[ep_idx] = {
            "length": ep_length,
            "is_success": is_success,  # Will be updated from sample
        }

    return episode_info


# =============================================================================
# Advantage Computation
# =============================================================================


def check_is_next_pad(sample: dict) -> bool:
    """Check if the next observation (t+N) is padded (beyond episode end).

    Args:
        sample: Sample from LeRobot dataset

    Returns:
        True if next observation is padded, False otherwise
    """
    for key in sample:
        if key.endswith("_is_pad") and (
            "observation" in key or key.replace("_is_pad", "") in OBSERVATION_LIKE_KEYS
        ):
            is_pad_array = to_numpy(sample[key])
            if len(is_pad_array) >= 2:
                return bool(is_pad_array[1])
    return False


class AdvantageDataset(torch.utils.data.Dataset):
    """Wrapper dataset for DataLoader-based advantage computation.

    Preprocesses each sample from LeRobotDataset into observations
    ready for value model inference. Used with DataLoader for efficient
    multi-process data loading that bypasses the Python GIL.
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        robot_type: str,
        tasks: dict,
        action_horizon: int,
    ):
        self.dataset = lerobot_dataset
        self.robot_type = robot_type
        self.tasks = tasks
        self.action_horizon = action_horizon

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        ep_idx = int(to_scalar(sample["episode_index"]))
        frame_idx = int(to_scalar(sample["frame_index"]))

        obs_current = build_obs(sample, self.robot_type, self.tasks, use_next=False)
        is_next_pad = check_is_next_pad(sample)

        obs_next = None
        if not is_next_pad:
            obs_next = build_obs(sample, self.robot_type, self.tasks, use_next=True)

        # Extract reward
        reward = to_numpy(sample["reward"]).flatten()
        num_valid = self.action_horizon
        if "reward_is_pad" in sample:
            is_pad = to_numpy(sample["reward_is_pad"]).flatten().astype(bool)
            num_valid = int(np.sum(~is_pad[: self.action_horizon]))
            reward = reward.copy()
            reward[is_pad[: len(reward)]] = 0.0

        true_return = float(to_scalar(sample["return"]))

        return {
            "obs_current": obs_current,
            "obs_next": obs_next,
            "is_next_pad": is_next_pad,
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "reward": reward,
            "num_valid": num_valid,
            "true_return": true_return,
        }


def advantage_collate_fn(
    batch: list[dict],
) -> tuple[list[dict], list[tuple[int, dict]], list[tuple]]:
    """Custom collate function for AdvantageDataset.

    Keeps observations as lists of dicts (not batched tensors) since
    value_policy.infer_batch() expects this format. Produces the same
    output signature as the old collect_observations_batch() so that
    process_batch_results() works without changes.

    Returns:
        current_obs_list: List of current observation dicts
        next_obs_with_idx: List of (local_idx, next_obs) for non-padded samples
        sample_info_list: List of (ep_idx, frame_idx, is_next_pad, reward, num_valid, true_return)
    """
    current_obs_list = [item["obs_current"] for item in batch]
    next_obs_with_idx = [
        (i, item["obs_next"]) for i, item in enumerate(batch) if not item["is_next_pad"]
    ]
    sample_info_list = [
        (
            item["episode_index"],
            item["frame_index"],
            item["is_next_pad"],
            item["reward"],
            item["num_valid"],
            item["true_return"],
        )
        for item in batch
    ]
    return current_obs_list, next_obs_with_idx, sample_info_list


@torch.no_grad()
def compute_advantages_for_dataset(
    value_policy,
    dataset: LeRobotDataset,
    tasks: dict,
    cfg: DictConfig,
    dataset_cfg: dict,
    meta: LeRobotDatasetMetadata,
    rank: int = 0,
    world_size: int = 1,
    global_return_min: float = -700.0,
    global_return_max: float = 0.0,
) -> pd.DataFrame:
    """Compute advantages for dataset (or shard in distributed mode).

    Uses batch inference with ValuePolicy for efficient processing.
    The ValuePolicy internally handles all input transforms
    (LiberoInputs, Normalize, ResizeImages, etc.).

    In distributed mode, each rank processes a shard of the dataset.
    The results are local to each rank and should be gathered afterwards.

    Args:
        value_policy: Trained ValuePolicy with input transforms and batch inference
        dataset: LeRobot dataset with delta_timestamps
        tasks: Task descriptions
        cfg: Full config
        dataset_cfg: Dataset-specific config
        meta: Dataset metadata
        rank: Current process rank (0 for single-GPU)
        world_size: Total number of processes (1 for single-GPU)
        global_return_min: Global minimum return value for normalization
        global_return_max: Global maximum return value for normalization

    Returns:
        DataFrame with advantages and related values (local shard in distributed mode)
    """
    gamma = cfg.data.gamma
    # Use advantage_horizon if set (must match dataset's reward/next_obs layout); else action_horizon
    action_horizon = cfg.data.get("advantage_horizon", None) or cfg.data.action_horizon
    robot_type = dataset_cfg.get("robot_type", "libero")
    discount_next_value = cfg.advantage.get("discount_next_value", True)
    batch_size = cfg.advantage.get("batch_size", 64)

    # Use global return range (passed from main)
    ret_min = global_return_min
    ret_max = global_return_max

    if rank == 0:
        logger.info(f"  Using global return_range: [{ret_min}, {ret_max}]")
        logger.info(f"  Using batch inference with batch_size: {batch_size}")

    # Normalization function: maps [ret_min, ret_max] -> [-1, 0]
    ret_range = ret_max - ret_min

    def normalize(x):
        if ret_range <= 0:
            return -0.5
        return (x - ret_min) / ret_range - 1.0

    # Gamma powers for discounted reward sum
    gamma_powers = np.array([gamma**i for i in range(action_horizon)], dtype=np.float32)

    # Limit samples for testing
    max_samples = cfg.advantage.get("max_samples", None)
    total_samples = (
        len(dataset) if max_samples is None else min(len(dataset), max_samples)
    )

    # Calculate shard indices for this rank
    shard_start, shard_end = get_shard_indices(total_samples, rank, world_size)
    shard_size = shard_end - shard_start

    if rank == 0:
        logger.info(
            f"Computing advantages for {total_samples} samples (total in dataset: {len(dataset)})..."
        )
        logger.info(f"  gamma: {gamma}, action_horizon: {action_horizon}")
        logger.info(f"  return_range: [{ret_min}, {ret_max}]")
        logger.info("  Using ValuePolicy with batch inference")
        logger.info("  Using precomputed reward/return from dataset")
        if world_size > 1:
            logger.info(f"  Distributed mode: {world_size} GPUs")

    if world_size > 1:
        logger.info(
            f"  [Rank {rank}] Processing samples {shard_start} to {shard_end} ({shard_size} samples)"
        )

    # DataLoader configuration for multi-process data loading
    pipeline_batch_size = cfg.advantage.get("pipeline_batch_size", 2048)
    num_dataloader_workers = cfg.advantage.get("num_dataloader_workers", 8)
    prefetch_factor = cfg.advantage.get("prefetch_factor", 2)

    if rank == 0:
        logger.info(
            f"  Using DataLoader: workers={num_dataloader_workers}, "
            f"prefetch_factor={prefetch_factor}, batch_size={pipeline_batch_size}"
        )

    # Results storage (periodically flushed to disk to prevent OOM)
    results = {
        "episode_index": [],
        "frame_index": [],
        "advantage": [],
        "return": [],
        "value_current": [],
        "value_next": [],
        "reward_sum": [],
        "reward_sum_raw": [],
        "num_valid_rewards": [],
    }

    # Online statistics (memory-efficient, replaces full-history lists)
    v_curr_stats = RunningStats("V(o_t)")
    v_next_stats = RunningStats("V(o_N)")
    reward_sum_raw_stats = RunningStats("R_raw")

    # Temporary file management for periodic flushing
    flush_interval = cfg.advantage.get("flush_interval", 5)
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix=f"adv_rank{rank}_"))
    temp_files = []
    flushed_sample_count = 0

    if rank == 0:
        logger.info(
            f"  Memory management: flush to disk every {flush_interval} pipeline batches "
            f"(~{flush_interval * pipeline_batch_size} samples)"
        )

    def flush_results_to_disk():
        """Flush current results to a temporary parquet file and clear memory."""
        nonlocal flushed_sample_count
        if not results["episode_index"]:
            return
        chunk_size = len(results["episode_index"])
        temp_df = pd.DataFrame(results)
        temp_file = temp_dir / f"chunk_{len(temp_files):04d}.parquet"
        temp_df.to_parquet(temp_file, index=False)
        temp_files.append(temp_file)
        flushed_sample_count += chunk_size
        # Clear results to free memory
        for k in results:
            results[k] = []
        del temp_df
        gc.collect()
        if rank == 0:
            logger.info(
                f"  Flushed chunk {len(temp_files)} ({chunk_size} samples) to disk. "
                f"Total flushed: {flushed_sample_count}"
            )

    # Create wrapper dataset and DataLoader for efficient multi-process loading
    # DataLoader workers are persistent subprocesses that decode images in parallel,
    # bypassing the GIL. prefetch_factor controls how many batches each worker
    # prepares ahead of time (total prefetch depth = num_workers * prefetch_factor).
    advantage_dataset = AdvantageDataset(dataset, robot_type, tasks, action_horizon)
    shard_indices = list(range(shard_start, shard_end))
    shard_dataset = torch.utils.data.Subset(advantage_dataset, shard_indices)

    dataloader = torch.utils.data.DataLoader(
        shard_dataset,
        batch_size=pipeline_batch_size,
        num_workers=num_dataloader_workers,
        prefetch_factor=prefetch_factor if num_dataloader_workers > 0 else None,
        persistent_workers=num_dataloader_workers > 0,
        collate_fn=advantage_collate_fn,
        shuffle=False,
    )

    if rank == 0:
        logger.info(
            f"Processing {len(shard_indices)} samples in {len(dataloader)} batches"
        )

    def process_batch_results(
        current_obs_list: list[dict],
        next_obs_with_idx: list[tuple[int, dict]],
        sample_info_list: list[tuple],
    ):
        """Run GPU inference and compute advantages for a collected batch."""
        # Batch inference for V(o_t)
        v_curr_results = value_policy.infer_batch(
            current_obs_list, batch_size=batch_size
        )
        v_curr_values = [r["value"] for r in v_curr_results]

        # Batch inference for V(o_{t+N}) - only for non-padded samples
        next_local_indices = [idx for idx, _ in next_obs_with_idx]
        next_obs_only = [obs for _, obs in next_obs_with_idx]

        if next_obs_only:
            v_next_results = value_policy.infer_batch(
                next_obs_only, batch_size=batch_size
            )
            v_next_map = {
                next_local_indices[i]: v_next_results[i]["value"]
                for i in range(len(next_local_indices))
            }
        else:
            v_next_map = {}

        # Compute advantages for this batch
        for i, (
            ep_idx,
            frame_idx,
            is_next_pad,
            reward,
            num_valid,
            true_return,
        ) in enumerate(sample_info_list):
            v_curr = v_curr_values[i]
            v_next = 0.0 if is_next_pad else v_next_map.get(i, 0.0)

            # Compute discounted reward sum
            reward_sum_raw = float(
                np.sum(gamma_powers[:num_valid] * reward[:num_valid])
            )
            reward_sum = normalize(reward_sum_raw)

            # Compute advantage: A = reward_sum + gamma^N * V(o_{t+N}) - V(o_t)
            gamma_k = gamma**num_valid if discount_next_value else 1.0
            advantage = reward_sum + gamma_k * v_next - v_curr

            # Update online statistics (memory-efficient, no list accumulation)
            v_curr_stats.update(v_curr)
            v_next_stats.update(v_next)
            reward_sum_raw_stats.update(reward_sum_raw)

            # Store results
            results["episode_index"].append(ep_idx)
            results["frame_index"].append(frame_idx)
            results["advantage"].append(advantage)
            results["return"].append(true_return)
            results["value_current"].append(v_curr)
            results["value_next"].append(v_next)
            results["reward_sum"].append(reward_sum)
            results["reward_sum_raw"].append(reward_sum_raw)
            results["num_valid_rewards"].append(num_valid)

    # DataLoader handles multi-process prefetching automatically:
    # - Each worker is a persistent process that loads/decodes data in parallel
    # - prefetch_factor controls how many batches each worker prepares ahead
    # - Total prefetch depth = num_workers * prefetch_factor
    for batch_idx, (current_obs_list, next_obs_with_idx, sample_info_list) in enumerate(
        tqdm(dataloader, desc="Processing batches", disable=rank != 0)
    ):
        # GPU inference and advantage computation
        process_batch_results(current_obs_list, next_obs_with_idx, sample_info_list)

        # Free observation data to reduce memory pressure
        del current_obs_list, next_obs_with_idx, sample_info_list

        # Periodic flush to disk to prevent OOM
        if (batch_idx + 1) % flush_interval == 0:
            flush_results_to_disk()

    # Final flush for any remaining results
    flush_results_to_disk()

    # Log statistics (using online RunningStats, no full-history arrays needed)
    if v_curr_stats.n > 0:
        rank_prefix = f"[Rank {rank}] " if world_size > 1 else ""
        logger.info(
            f"\n{rank_prefix}Value and reward Statistics (local shard, {v_curr_stats.n} samples):"
        )
        logger.info(
            f"  {rank_prefix}V(o_t):    mean={v_curr_stats.mean:.4f}, std={v_curr_stats.std:.4f}, "
            f"min={v_curr_stats.min:.4f}, max={v_curr_stats.max:.4f}"
        )
        logger.info(
            f"  {rank_prefix}V(o_N):    mean={v_next_stats.mean:.4f}, std={v_next_stats.std:.4f}, "
            f"min={v_next_stats.min:.4f}, max={v_next_stats.max:.4f}"
        )
        logger.info(
            f"  {rank_prefix}R_raw:     mean={reward_sum_raw_stats.mean:.4f}, std={reward_sum_raw_stats.std:.4f}, "
            f"min={reward_sum_raw_stats.min:.4f}, max={reward_sum_raw_stats.max:.4f}"
        )
    else:
        logger.warning(f"[Rank {rank}] No samples processed in this shard")

    # Merge all temporary chunks into final DataFrame
    if temp_files:
        if rank == 0:
            logger.info(
                f"Merging {len(temp_files)} temporary chunks ({flushed_sample_count} total samples)..."
            )
        merged_df = pd.concat(
            [pd.read_parquet(f) for f in temp_files], ignore_index=True
        )
        # Clean up temporary files
        for f in temp_files:
            f.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
            temp_dir.parent.rmdir()  # Remove .temp_advantages if empty
        except OSError:
            pass
        return merged_df
    else:
        return pd.DataFrame(results)


# =============================================================================
# Output Dataset Creation
# =============================================================================


def save_advantages_to_dataset(
    dataset_path: Path,
    advantages_df: pd.DataFrame,
    threshold: float,
    dataset_type: str | None = None,
    rank: int = 0,
    world_size: int = 1,
    advantage_tag: str | None = None,
):
    """Save advantages parquet directly into the source dataset's meta/ directory.

    Only writes meta/advantages_{tag}.parquet (or meta/advantages.parquet).
    Does NOT modify info.json, episodes.jsonl, or any data parquet files.
    Training code loads advantages from this parquet via (episode_index, frame_index) lookup.

    In distributed mode, only rank 0 writes the file.

    Args:
        dataset_path: Source LeRobot dataset path (writes into its meta/)
        advantages_df: DataFrame with advantage values
        threshold: Threshold for positive advantage
        dataset_type: Dataset type ("sft" forces all-True advantage labels)
        rank: Current process rank
        world_size: Total number of processes
        advantage_tag: Optional tag for advantages parquet filename
    """
    if rank == 0:
        meta_dir = dataset_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        # Build advantages parquet with boolean advantage column
        save_df = advantages_df.copy()
        save_df.rename(columns={"advantage": "advantage_continuous"}, inplace=True)
        if (dataset_type or "").lower() == "sft":
            save_df["advantage"] = True
        else:
            save_df["advantage"] = save_df["advantage_continuous"] >= threshold

        adv_filename = (
            f"advantages_{advantage_tag}.parquet"
            if advantage_tag
            else "advantages.parquet"
        )
        save_df.to_parquet(meta_dir / adv_filename, index=False)
        if (dataset_type or "").lower() == "sft":
            logger.info(
                f"  Dataset type is sft, forcing all advantage labels to True ({len(save_df)} entries)"
            )
        logger.info(f"  Saved {adv_filename} to {meta_dir} ({len(save_df)} entries)")

    # Synchronize after writing
    if world_size > 1:
        dist.barrier()


# =============================================================================
# Main
# =============================================================================


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="compute_advantages",
)
def main(cfg: DictConfig) -> None:
    """Main entry point for advantage computation.

    Supports both single-GPU and multi-GPU (via torchrun) execution.
    In multi-GPU mode:
    1. Each GPU processes its shard of samples in parallel
    2. Advantages are gathered across all GPUs
    3. Unified threshold is computed from combined advantages
    4. Output datasets are created in parallel
    """
    # Setup distributed (if running under torchrun)
    rank, world_size, device = setup_distributed(cfg)

    # Override device in config
    cfg.advantage.device = device

    # Setup logging (only rank 0 shows full config)
    logging.basicConfig(level=logging.INFO)
    if rank == 0:
        logger.info("Starting advantage computation...")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    else:
        # Reduce logging verbosity for non-rank-0 processes
        logging.getLogger().setLevel(logging.WARNING)

    try:
        # Load value policy (each rank loads its own copy)
        # ValuePolicy encapsulates all input transforms (LiberoInputs, Normalize,
        # ResizeImages, etc.) so inference uses same preprocessing as training
        value_policy = load_value_policy(cfg, device)

        # Process all datasets and collect advantages
        all_advantages = []
        dataset_results = {}

        # ---- Compute global return_min/return_max ----
        # Priority: 1) global config  2) compute from all datasets' stats.json
        data_cfg = cfg.data
        global_return_min = data_cfg.get("return_min", None)
        global_return_max = data_cfg.get("return_max", None)

        if global_return_min is None or global_return_max is None:
            # Compute from all datasets' stats.json
            all_mins = []
            all_maxs = []
            for ds_cfg in cfg.data.datasets:
                ds_path = Path(ds_cfg.dataset_path)
                ds_min, ds_max = _load_return_stats_from_dataset(ds_path)
                if ds_min is not None:
                    all_mins.append(ds_min)
                if ds_max is not None:
                    all_maxs.append(ds_max)

            if all_mins and all_maxs:
                global_return_min = (
                    min(all_mins) if global_return_min is None else global_return_min
                )
                global_return_max = (
                    max(all_maxs) if global_return_max is None else global_return_max
                )
                if rank == 0:
                    logger.info(
                        f"Computed global return range from stats.json: "
                        f"[{global_return_min}, {global_return_max}]"
                    )
            else:
                # Fallback to defaults
                global_return_min = (
                    global_return_min if global_return_min is not None else -700.0
                )
                global_return_max = (
                    global_return_max if global_return_max is not None else 0.0
                )
                if rank == 0:
                    logger.warning(
                        f"No stats.json found, using default return range: "
                        f"[{global_return_min}, {global_return_max}]"
                    )
        else:
            if rank == 0:
                logger.info(
                    f"Using global return range from config: "
                    f"[{global_return_min}, {global_return_max}]"
                )

        for ds_cfg in cfg.data.datasets:
            ds_path = Path(ds_cfg.dataset_path)
            if rank == 0:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing dataset: {ds_path.name}")
                logger.info(f"{'=' * 60}")

            # Load dataset (each rank loads full dataset but processes shard)
            horizon = cfg.data.get("advantage_horizon", None) or cfg.data.action_horizon
            if rank == 0:
                logger.info("  Horizon N=%s for reward and V(o_t+N)", horizon)
            dataset, tasks, meta = load_lerobot_dataset(ds_path, horizon)

            # Compute advantages for this rank's shard
            local_df = compute_advantages_for_dataset(
                value_policy=value_policy,
                dataset=dataset,
                tasks=tasks,
                cfg=cfg,
                dataset_cfg=OmegaConf.to_container(ds_cfg),
                meta=meta,
                rank=rank,
                world_size=world_size,
                global_return_min=global_return_min,
                global_return_max=global_return_max,
            )

            # Synchronize and gather advantages from all ranks
            if world_size > 1:
                dist.barrier()
                df = gather_all_advantages(local_df, rank, world_size)
            else:
                df = local_df

            # Store results
            df["dataset_name"] = ds_path.name
            all_advantages.append(df["advantage"].values)
            dataset_results[ds_path] = {
                "df": df,
                "config": OmegaConf.to_container(ds_cfg),
            }

            # Print statistics (only rank 0)
            if rank == 0 and len(df) > 0:
                logger.info(f"\nAdvantage Statistics for {ds_path.name}:")
                logger.info(f"  Mean: {df['advantage'].mean():.4f}")
                logger.info(f"  Std: {df['advantage'].std():.4f}")
                logger.info(f"  Min: {df['advantage'].min():.4f}")
                logger.info(f"  Max: {df['advantage'].max():.4f}")
                logger.info(f"  V(o_t) mean: {df['value_current'].mean():.4f}")
                logger.info(f"  V(o_N) mean: {df['value_next'].mean():.4f}")
                logger.info(f"  reward_sum mean: {df['reward_sum'].mean():.4f}")

        # Compute unified threshold across all datasets
        positive_quantile = cfg.advantage.get("positive_quantile", 0.3)
        combined_advantages = np.concatenate(all_advantages)
        unified_threshold = float(
            np.percentile(combined_advantages, (1 - positive_quantile) * 100)
        )

        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info("Unified Advantage Threshold (across ALL datasets)")
            logger.info(f"{'=' * 60}")
            logger.info(f"  Number of datasets: {len(all_advantages)}")
            logger.info(f"  Samples per dataset: {[len(a) for a in all_advantages]}")
            logger.info(f"  Total samples: {len(combined_advantages)}")
            logger.info(
                f"  Combined advantage range: [{combined_advantages.min():.4f}, {combined_advantages.max():.4f}]"
            )
            logger.info(f"  Combined advantage mean: {combined_advantages.mean():.4f}")
            logger.info(
                f"  Positive quantile: {positive_quantile} (top {positive_quantile * 100:.0f}% positive)"
            )
            logger.info(
                f"  Unified threshold (at {(1 - positive_quantile) * 100:.0f}th percentile): {unified_threshold:.4f}"
            )
            logger.info(
                f"  Total samples with positive advantage: {(combined_advantages >= unified_threshold).sum()}"
            )

            # Show per-dataset positive rates using unified threshold
            logger.info("\n  Per-dataset positive rates (using unified threshold):")
            for i, (ds_path, result) in enumerate(dataset_results.items()):
                ds_advantages = all_advantages[i]
                positive_count = (ds_advantages >= unified_threshold).sum()
                positive_rate = positive_count / len(ds_advantages) * 100
                logger.info(
                    f"    {ds_path.name}: {positive_count}/{len(ds_advantages)} ({positive_rate:.1f}%)"
                )

        # Save advantages parquet and mixture_config.yaml to each source dataset
        advantage_tag = cfg.advantage.get("tag", None)

        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info("Saving Advantages")
            logger.info(f"{'=' * 60}")
            if advantage_tag:
                logger.info(f"  Advantage tag: {advantage_tag}")

        # Build mixture_config content (shared across all datasets)
        tag_stats = {
            "unified_threshold": unified_threshold,
            "positive_quantile": positive_quantile,
        }

        for ds_path, result in dataset_results.items():
            df = result["df"]
            dataset_type = result["config"].get("dataset_type")
            save_advantages_to_dataset(
                dataset_path=ds_path,
                advantages_df=df,
                threshold=unified_threshold,
                dataset_type=dataset_type,
                rank=rank,
                world_size=world_size,
                advantage_tag=advantage_tag,
            )

            # Save mixture_config.yaml to each dataset root (only rank 0)
            if rank == 0:
                import yaml

                mixture_config_path = ds_path / "mixture_config.yaml"

                # Load existing to preserve other tags
                if mixture_config_path.exists():
                    with open(mixture_config_path, "r") as f:
                        mixture_config = yaml.safe_load(f) or {}
                else:
                    mixture_config = {}

                # Common fields (always update)
                mixture_config["global_return_min"] = global_return_min
                mixture_config["global_return_max"] = global_return_max
                mixture_config["datasets"] = [
                    {
                        "name": p.name,
                        "weight": r["config"].get("weight", 1.0),
                        "return_min": r["config"].get("return_min"),
                        "return_max": r["config"].get("return_max"),
                    }
                    for p, r in dataset_results.items()
                ]

                if advantage_tag:
                    if "tags" not in mixture_config:
                        mixture_config["tags"] = {}
                    mixture_config["tags"][advantage_tag] = tag_stats
                else:
                    mixture_config["unified_threshold"] = unified_threshold
                    mixture_config["positive_quantile"] = positive_quantile

                with open(mixture_config_path, "w") as f:
                    yaml.dump(mixture_config, f, default_flow_style=False)
                logger.info(f"  Saved mixture_config.yaml to: {ds_path}")

        if rank == 0:
            logger.info("\nAdvantage computation complete!")

    finally:
        # Clean up distributed
        cleanup_distributed()


if __name__ == "__main__":
    main()
