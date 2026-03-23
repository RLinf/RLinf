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
Compute advantages for CFG-RL training using a trained ValueCritic.

Advantage formula: A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t)

This script:
1. Loads a trained ValueCritic from checkpoint (with input transforms)
2. Loads LeRobot datasets (without delta_timestamps for ~50x faster access)
3. Computes N-step discounted reward sum and values
4. Creates independent output datasets with is_success = (advantage >= threshold)

The ValueCritic encapsulates all input transforms (LiberoInputs, Normalize, ResizeImages, etc.)
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

# Import ValueCritic for value inference
from rlinf.models.embodiment.value_model.modeling_critic import ValueCritic

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

# Key mappings for building raw observations for ValueCritic
# Maps LeRobot dataset keys to value model observation format
KEY_MAPPINGS = {
    "franka": {
        # Multi-cam format (front_cam + wrist_cam)
        "observation.images.front_cam": "observation/images/front_cam",
        "observation.images.wrist_cam": "observation/images/wrist_cam",
        "observation.state.tcp_pose": "observation/state/tcp_pose",
        "observation.state.gripper_pose": "observation/state/gripper_pose",
        # Single-cam format (push_button style: image + state)
        "observation.images.image": "observation/image",
        "image": "observation/image",
        "state": "observation/state",
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


def _load_returns_sidecar(
    dataset_path: Path,
    returns_tag: str | None = None,
) -> dict[int, dict[str, np.ndarray]] | None:
    """Load ``meta/returns_{tag}.parquet`` sidecar written by compute_returns.py.

    Falls back to ``meta/returns.parquet`` when *returns_tag* is None.

    Returns:
        ``{episode_index: {"return": np.array, "reward": np.array}}``
        or None if sidecar does not exist.
    """
    import pyarrow.parquet as pq

    sidecar_filename = (
        f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    )
    sidecar_path = dataset_path / "meta" / sidecar_filename
    if not sidecar_path.exists():
        return None

    table = pq.read_table(str(sidecar_path))
    ep_col = table.column("episode_index").to_numpy()
    frame_col = table.column("frame_index").to_numpy()
    ret_col = table.column("return").to_numpy()
    rew_col = table.column("reward").to_numpy()

    sidecar: dict[int, dict[str, np.ndarray]] = {}
    for ep in np.unique(ep_col):
        mask = ep_col == ep
        frames = frame_col[mask]
        order = np.argsort(frames)
        sidecar[int(ep)] = {
            "return": ret_col[mask][order].astype(np.float32),
            "reward": rew_col[mask][order].astype(np.float32),
        }

    logger.info(f"Loaded returns sidecar: {sidecar_path} ({len(sidecar)} episodes)")
    return sidecar


# =============================================================================
# Model Loading
# =============================================================================


def load_value_model(cfg: DictConfig, device: str = "cuda"):
    """Load trained ValueCritic from checkpoint.

    The ValueCritic encapsulates all input transforms (LiberoInputs, Normalize,
    ResizeImages, etc.) so inference uses the same data preprocessing as training.

    Args:
        cfg: Config with checkpoint path and model settings
        device: Target device

    Returns:
        Configured ValueCritic instance with inference methods
    """
    checkpoint_path = cfg.advantage.value_checkpoint
    if checkpoint_path is None:
        raise ValueError("advantage.value_checkpoint must be specified")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading value model from: {checkpoint_path}")

    # Extract configuration
    adv_cfg = cfg.advantage
    data_cfg = cfg.data
    model_cfg = adv_cfg.get("model", {})

    # Get robot_type (env_type)
    robot_type = data_cfg.get("robot_type", "libero")
    if "train_data_paths" in data_cfg and len(data_cfg.train_data_paths) > 0:
        robot_type = data_cfg.train_data_paths[0].get("robot_type", robot_type)

    model_type = data_cfg.get("model_type", "pi05")

    num_return_bins = model_cfg.get("num_bins", 201)
    return_min = model_cfg.get("v_min", -1.0)
    return_max = model_cfg.get("v_max", 0.0)
    critic_expert_variant = model_cfg.get("critic_expert_variant", "gemma_100m")
    tokenizer_path = model_cfg.get("tokenizer_path", None)
    backbone_variant = model_cfg.get("backbone_variant", "paligemma")
    siglip_path = model_cfg.get("siglip_path", None)
    gemma3_path = model_cfg.get("gemma3_path", None)

    logger.info(f"  env_type (robot_type): {robot_type}")
    logger.info(f"  model_type: {model_type}")
    logger.info(f"  backbone_variant: {backbone_variant}")
    logger.info(f"  num_return_bins: {num_return_bins}")
    logger.info(f"  return_range: [{return_min}, {return_max}]")
    logger.info(f"  critic_expert_variant: {critic_expert_variant}")
    if tokenizer_path:
        logger.info(f"  tokenizer_path: {tokenizer_path}")
    if siglip_path:
        logger.info(f"  siglip_path: {siglip_path}")
    if gemma3_path:
        logger.info(f"  gemma3_path: {gemma3_path}")

    model = ValueCritic.from_checkpoint(
        checkpoint_dir=checkpoint_path,
        env_type=robot_type,
        model_type=model_type,
        device=device,
        num_return_bins=num_return_bins,
        return_min=return_min,
        return_max=return_max,
        critic_expert_variant=critic_expert_variant,
        tokenizer_path=tokenizer_path,
        backbone_variant=backbone_variant,
        siglip_path=siglip_path,
        gemma3_path=gemma3_path,
    )

    logger.info("Loaded ValueCritic for inference")
    return model


# =============================================================================
# Dataset Loading
# =============================================================================


def load_lerobot_dataset(
    dataset_path: Path,
    returns_tag: str | None = None,
) -> tuple[LeRobotDataset, dict, LeRobotDatasetMetadata, dict | None]:
    """Load a LeRobot dataset WITHOUT delta_timestamps for fast single-row access.

    Loading without delta_timestamps is ~50x faster (6ms vs 580ms per sample)
    because it avoids expensive multi-timestep parquet reads and image decoding.
    The AdvantageDataset handles multi-timestep access via separate dataset[idx]
    and dataset[idx+N] calls instead.

    Also loads ``meta/returns_{tag}.parquet`` sidecar if present.

    Args:
        dataset_path: Path to dataset
        returns_tag: Optional tag for the returns sidecar filename

    Returns:
        Tuple of (dataset, tasks_dict, metadata, returns_sidecar)
    """
    meta = LeRobotDatasetMetadata(str(dataset_path))

    # Log dataset features for debugging
    logger.info(f"Dataset features: {list(meta.features.keys())}")

    # Load sidecar written by compute_returns.py (if present)
    returns_sidecar = _load_returns_sidecar(dataset_path, returns_tag=returns_tag)

    # Validate required columns: accept either features in parquets OR sidecar
    has_reward = "reward" in meta.features
    has_return = "return" in meta.features
    has_sidecar = returns_sidecar is not None

    sidecar_name = (
        f"returns_{returns_tag}.parquet" if returns_tag else "returns.parquet"
    )
    if not has_reward and not has_sidecar:
        raise ValueError(
            f"Dataset {dataset_path} missing 'reward' column and no "
            f"meta/{sidecar_name} sidecar found. "
            "Run compute_returns.py to preprocess the dataset first."
        )

    if not has_return and not has_sidecar:
        raise ValueError(
            f"Dataset {dataset_path} missing 'return' column and no "
            f"meta/{sidecar_name} sidecar found. "
            "Run compute_returns.py to preprocess the dataset first."
        )

    logger.info("Loading dataset (no delta_timestamps for ~50x faster access):")
    logger.info(f"  Dataset path: {dataset_path}")
    logger.info(f"  FPS: {meta.fps}")

    dataset = LeRobotDataset(
        str(dataset_path),
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

    return dataset, tasks, meta, returns_sidecar


# =============================================================================
# Observation Building for Value Policy
# =============================================================================


def build_obs(
    sample: dict,
    robot_type: str,
    tasks: dict,
) -> dict[str, Any]:
    """Build raw observation dict for ValueCritic from a single-timestep sample.

    The ValueCritic internally handles all input transforms (LiberoInputs,
    Normalize, ResizeImages, etc.), so we only need to build the raw observation
    in the format expected by the value model's input processing.

    Args:
        sample: Single-timestep sample from LeRobot dataset (no delta_timestamps)
        robot_type: Robot type for key mapping
        tasks: Task descriptions dict

    Returns:
        Raw observation dict compatible with ValueCritic.infer()
    """
    key_map = KEY_MAPPINGS.get(robot_type, KEY_MAPPINGS["libero"])
    obs = {}

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
            obs[dst_key] = val

    return obs


# =============================================================================
# Advantage Computation
# =============================================================================


class AdvantageDataset(torch.utils.data.Dataset):
    """Wrapper dataset for DataLoader-based advantage computation.

    Builds observation at the current timestep only. The caller can later
    reuse V(o_t) values by index lookup to obtain V(o_{t+N}) without
    a second forward pass.
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        robot_type: str,
        tasks: dict,
        input_transform=None,
        prepare_observation_cpu=None,
        returns_sidecar: dict[int, dict[str, np.ndarray]] | None = None,
    ):
        self.dataset = lerobot_dataset
        self.robot_type = robot_type
        self.tasks = tasks
        self.input_transform = input_transform
        self.prepare_observation_cpu = prepare_observation_cpu
        self.returns_sidecar = returns_sidecar

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        ep_idx = int(to_scalar(sample["episode_index"]))
        frame_idx = int(to_scalar(sample["frame_index"]))

        # Build current observation from single-timestep sample
        obs = build_obs(sample, self.robot_type, self.tasks)

        # Apply CPU transforms in worker (input_transform + prepare_observation_cpu)
        if self.input_transform is not None:
            obs = self.input_transform(
                {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
            )
        if self.prepare_observation_cpu is not None:
            obs = self.prepare_observation_cpu(obs)

        # Look up return/reward from sidecar or fall back to sample columns
        if self.returns_sidecar is not None and ep_idx in self.returns_sidecar:
            ep_data = self.returns_sidecar[ep_idx]
            true_return = float(ep_data["return"][frame_idx])
            reward = float(ep_data["reward"][frame_idx])
        else:
            true_return = float(to_scalar(sample["return"]))
            reward = (
                float(to_scalar(sample["reward"]))
                if "reward" in sample
                else float("nan")
            )

        return {
            "obs": obs,
            "global_idx": idx,
            "episode_index": ep_idx,
            "frame_index": frame_idx,
            "true_return": true_return,
            "reward": reward,
        }


def advantage_collate_fn(
    batch: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Custom collate function for AdvantageDataset.

    Keeps observations as lists of dicts (not batched tensors), since
    value_model.infer_batch() expects this format.

    Returns:
        obs_list: List of current observation dicts
        meta_list: List of metadata dicts aligned with obs_list
    """
    obs_list = [item["obs"] for item in batch]
    meta_list = [
        {
            "global_idx": item["global_idx"],
            "episode_index": item["episode_index"],
            "frame_index": item["frame_index"],
            "true_return": item["true_return"],
            "reward": item["reward"],
        }
        for item in batch
    ]
    return obs_list, meta_list


@torch.no_grad()
def compute_advantages_for_dataset(
    value_model,
    dataset: LeRobotDataset,
    tasks: dict,
    cfg: DictConfig,
    dataset_cfg: dict,
    meta: LeRobotDatasetMetadata,
    rank: int = 0,
    world_size: int = 1,
    global_return_min: float = -700.0,
    global_return_max: float = 0.0,
    global_pbar: tqdm | None = None,
    returns_sidecar: dict[int, dict[str, np.ndarray]] | None = None,
) -> pd.DataFrame:
    """Compute advantages for dataset (or shard in distributed mode).

    Uses batch inference with ValueCritic for efficient processing.
    The ValueCritic internally handles all input transforms
    (LiberoInputs, Normalize, ResizeImages, etc.).

    In distributed mode, each rank processes a shard of the dataset.
    The results are local to each rank and should be gathered afterwards.

    Args:
        value_model: Trained ValueCritic with input transforms and batch inference
        dataset: LeRobot dataset
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
    action_horizon = cfg.data.advantage_lookahead_step
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
    gamma_powers = np.array([gamma**i for i in range(action_horizon)], dtype=np.float64)

    # Limit samples for testing
    max_samples = cfg.advantage.get("max_samples", None)
    total_samples = (
        len(dataset) if max_samples is None else min(len(dataset), max_samples)
    )

    # Calculate shard indices for this rank
    shard_start, shard_end = get_shard_indices(total_samples, rank, world_size)
    shard_size = shard_end - shard_start

    # Include an extension window so idx + lookahead can be looked up.
    extended_end = (
        shard_start
        if shard_size == 0
        else min(shard_end + action_horizon, len(dataset))
    )
    extended_size = extended_end - shard_start

    # Precompute episode boundaries for fast lookup
    ep_ends = {}
    for ep_idx in range(len(dataset.episode_data_index["to"])):
        ep_ends[ep_idx] = int(dataset.episode_data_index["to"][ep_idx].item())

    if rank == 0:
        logger.info(
            f"Computing advantages for {total_samples} samples (total in dataset: {len(dataset)})..."
        )
        logger.info(f"  gamma: {gamma}, advantage_lookahead_step: {action_horizon}")
        logger.info(f"  return_range: [{ret_min}, {ret_max}]")
        logger.info("  Using ValueCritic with batch inference")
        logger.info("  Using precomputed reward/return from dataset")
        if world_size > 1:
            logger.info(f"  Distributed mode: {world_size} GPUs")

    if world_size > 1:
        logger.info(
            f"  [Rank {rank}] Processing samples {shard_start} to {shard_end} ({shard_size} samples)"
        )
        logger.info(
            f"  [Rank {rank}] Extended inference range: {shard_start} to {extended_end} ({extended_size} samples)"
        )

    # DataLoader configuration for multi-process data loading
    num_dataloader_workers = cfg.advantage.get("num_dataloader_workers", 8)
    prefetch_factor = cfg.advantage.get("prefetch_factor", 2)

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
    flush_interval = max(1, int(cfg.advantage.get("flush_interval", 5)))
    flush_every_samples = max(1, flush_interval * batch_size)
    import tempfile

    temp_dir = Path(tempfile.mkdtemp(prefix=f"adv_rank{rank}_"))
    temp_files = []
    flushed_sample_count = 0

    if rank == 0:
        logger.info(
            f"  Memory management: flush to disk every ~{flush_every_samples} samples"
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

    # Workers run _input_transform AND _prepare_observation_cpu (image resize + tokenize).
    # Workers only do CPU work (video decode, image resize, tokenize), so fork is safe
    # even after CUDA init — same pattern used by VLA lib SFT training.
    from functools import partial

    processor = getattr(value_model, "processor", None)
    worker_cpu_prep = partial(
        value_model.__class__._prepare_observation_cpu, processor=processor
    )

    cpu_prep_in_workers = num_dataloader_workers > 0

    if rank == 0:
        logger.info(
            f"  Using DataLoader: workers={num_dataloader_workers}, "
            f"prefetch_factor={prefetch_factor}, batch_size={batch_size}, "
            f"cpu_prep_in_workers={cpu_prep_in_workers}"
        )

    advantage_dataset = AdvantageDataset(
        dataset,
        robot_type,
        tasks,
        input_transform=value_model._input_transform if cpu_prep_in_workers else None,
        prepare_observation_cpu=worker_cpu_prep if cpu_prep_in_workers else None,
        returns_sidecar=returns_sidecar,
    )
    extended_indices = list(range(shard_start, extended_end))
    extended_dataset = torch.utils.data.Subset(advantage_dataset, extended_indices)

    dataloader = torch.utils.data.DataLoader(
        extended_dataset,
        batch_size=batch_size,
        num_workers=num_dataloader_workers,
        prefetch_factor=prefetch_factor if num_dataloader_workers > 0 else None,
        persistent_workers=num_dataloader_workers > 0,
        collate_fn=advantage_collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(
            f"Phase 1: inferring V(o_t) for {extended_size} samples in {len(dataloader)} batches"
        )

    # Phase 1 storage: values + metadata for [shard_start, extended_end)
    v_values = np.full(extended_size, np.nan, dtype=np.float64)
    meta_ep_idx = np.full(extended_size, -1, dtype=np.int64)
    meta_frame_idx = np.full(extended_size, -1, dtype=np.int64)
    meta_return = np.full(extended_size, np.nan, dtype=np.float64)
    meta_reward = np.full(extended_size, np.nan, dtype=np.float64)
    filled_mask = np.zeros(extended_size, dtype=bool)

    def process_value_batch(obs_list: list[dict], meta_list: list[dict]):
        """Run GPU inference for V(o_t) only and store batch results."""
        batch_results = value_model.infer_batch(
            obs_list,
            batch_size=batch_size,
            pretransformed=cpu_prep_in_workers,
            already_cpu_prepared=cpu_prep_in_workers,
        )
        if len(batch_results) != len(meta_list):
            raise RuntimeError(
                "Mismatch between inference outputs and metadata: "
                f"{len(batch_results)} vs {len(meta_list)}"
            )

        for result, meta_info in zip(batch_results, meta_list):
            local_idx = int(meta_info["global_idx"]) - shard_start
            if local_idx < 0 or local_idx >= extended_size:
                raise RuntimeError(
                    f"local_idx out of range: {local_idx}, extended_size={extended_size}"
                )

            v_values[local_idx] = float(result["value"])
            meta_ep_idx[local_idx] = int(meta_info["episode_index"])
            meta_frame_idx[local_idx] = int(meta_info["frame_index"])
            meta_return[local_idx] = float(meta_info["true_return"])
            meta_reward[local_idx] = float(meta_info["reward"])
            filled_mask[local_idx] = True

    # Pipeline: prefetch next batch from DataLoader while GPU processes current batch.
    # DataLoader workers do ALL CPU work (image decode + transform + image_processor + tokenize),
    # so the main process only needs to stack tensors -> GPU forward.
    import time as _time
    from concurrent.futures import ThreadPoolExecutor

    _t_fetch_total = 0.0
    _t_infer_total = 0.0

    ds_name = Path(dataset_cfg.get("dataset_path", "")).name
    if global_pbar is not None:
        pbar = global_pbar
        pbar.set_description(f"[Rank {rank}] {ds_name}")
    else:
        pbar = tqdm(
            total=shard_size,
            desc=f"[Rank {rank}] {ds_name}",
            unit="samples",
            disable=rank != 0,
            dynamic_ncols=True,
            file=sys.stdout,
        )
    _owns_pbar = global_pbar is None

    dataloader_iter = iter(dataloader)
    batch_count = len(dataloader)

    def _fetch_next():
        return next(dataloader_iter)

    with ThreadPoolExecutor(max_workers=1) as prefetch_pool:
        # Pre-submit first batch
        pending_future = prefetch_pool.submit(_fetch_next) if batch_count > 0 else None

        for batch_idx in range(batch_count):
            _t0 = _time.perf_counter()
            batch = pending_future.result()
            _t_fetch = _time.perf_counter() - _t0
            _t_fetch_total += _t_fetch

            # Submit prefetch for next batch BEFORE processing current one
            if batch_idx + 1 < batch_count:
                pending_future = prefetch_pool.submit(_fetch_next)

            _t0 = _time.perf_counter()
            process_value_batch(*batch)
            _t_infer = _time.perf_counter() - _t0
            _t_infer_total += _t_infer

            # Progress reflects output shard only (not extended tail)
            n_samples = sum(
                1 for item in batch[1] if int(item["global_idx"]) < shard_end
            )
            pbar.update(n_samples)
            if rank == 0:
                pbar.set_postfix(
                    fetch=f"{_t_fetch * 1000:.0f}ms",
                    infer=f"{_t_infer * 1000:.0f}ms",
                    GPU=f"{_t_infer * 100 / ((_t_fetch + _t_infer) or 1e-9):.0f}%",
                )

            del batch

    if _owns_pbar:
        pbar.close()

    if rank == 0 and batch_count > 0:
        avg_fetch = _t_fetch_total / batch_count * 1000
        avg_infer = _t_infer_total / batch_count * 1000
        logger.info(
            f"[Timing] avg_fetch={avg_fetch:.0f}ms  avg_infer={avg_infer:.0f}ms  "
            f"GPU_busy≈{avg_infer * 100 / ((avg_fetch + avg_infer) or 1e-9):.0f}%  "
            f"batches={batch_count}"
        )

    if extended_size > 0:
        missing_count = int(np.size(filled_mask) - np.count_nonzero(filled_mask))
        if missing_count > 0:
            raise RuntimeError(
                f"Phase 1 incomplete: {missing_count}/{extended_size} entries were not filled."
            )

    # Phase 2: compute advantages from precomputed V(o_t) values with table lookup.
    for i in range(shard_size):
        gidx = shard_start + i
        ep_idx = int(meta_ep_idx[i])
        frame_idx = int(meta_frame_idx[i])
        true_return = float(meta_return[i])

        ep_end = int(ep_ends.get(ep_idx, gidx + 1))
        ep_end = max(ep_end, gidx + 1)
        next_gidx = gidx + action_horizon
        is_next_pad = next_gidx >= ep_end
        num_valid = min(action_horizon, ep_end - gidx)

        v_curr = float(v_values[i])
        if is_next_pad:
            v_next = 0.0
            next_local_idx = None
        else:
            next_local_idx = next_gidx - shard_start
            if next_local_idx < 0 or next_local_idx >= extended_size:
                raise RuntimeError(
                    "next_local_idx out of range: "
                    f"{next_local_idx}, extended_size={extended_size}, "
                    f"gidx={gidx}, next_gidx={next_gidx}"
                )
            v_next = float(v_values[next_local_idx])

        if abs(gamma - 1.0) < 1e-8:
            if is_next_pad:
                reward_sum_raw = true_return
            else:
                reward_sum_raw = true_return - float(meta_return[next_local_idx])
        else:
            reward_slice = meta_reward[i : i + num_valid]
            if len(reward_slice) != num_valid:
                raise RuntimeError(
                    f"Invalid reward slice size {len(reward_slice)} for num_valid={num_valid}"
                )
            if np.isnan(reward_slice).any():
                raise ValueError(
                    "Reward values are required when gamma != 1.0, but missing reward was found."
                )
            reward_sum_raw = float(np.sum(gamma_powers[:num_valid] * reward_slice))

        reward_sum = normalize(reward_sum_raw)
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

        if (i + 1) % flush_every_samples == 0:
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
    tag: str | None = None,
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
        tag: Optional tag for advantages parquet filename
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

        adv_filename = f"advantages_{tag}.parquet" if tag else "advantages.parquet"
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
    config_name="compute_advantages_paligemma",
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
        # ValueCritic encapsulates all input transforms (LiberoInputs, Normalize,
        # ResizeImages, etc.) so inference uses same preprocessing as training
        value_model = load_value_model(cfg, device)

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
            for ds_cfg in cfg.data.train_data_paths:
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

        # Pre-compute grand total samples across all datasets (for this rank)
        max_samples = cfg.advantage.get("max_samples", None)
        grand_total = 0
        for ds_cfg in cfg.data.train_data_paths:
            ds_meta = LeRobotDatasetMetadata(str(ds_cfg.dataset_path))
            n = ds_meta.total_frames
            if max_samples is not None:
                n = min(n, max_samples)
            shard_start, shard_end = get_shard_indices(n, rank, world_size)
            grand_total += shard_end - shard_start

        if rank == 0:
            logger.info(
                f"Grand total: {grand_total} samples across "
                f"{len(cfg.data.train_data_paths)} datasets (this rank's shard)"
            )

        global_pbar = tqdm(
            total=grand_total,
            desc=f"[Rank {rank}] total",
            unit="samples",
            disable=rank != 0,
            dynamic_ncols=True,
            file=sys.stdout,
        )

        tag = cfg.advantage.get("tag", None)
        returns_tag = cfg.advantage.get("returns_tag", tag)

        for ds_cfg in cfg.data.train_data_paths:
            ds_path = Path(ds_cfg.dataset_path)
            if rank == 0:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing dataset: {ds_path.name}")
                logger.info(f"{'=' * 60}")

            # Load dataset (each rank loads full dataset but processes shard)
            dataset, tasks, meta, returns_sidecar = load_lerobot_dataset(
                ds_path, returns_tag=returns_tag
            )

            # Compute advantages for this rank's shard
            local_df = compute_advantages_for_dataset(
                value_model=value_model,
                dataset=dataset,
                tasks=tasks,
                cfg=cfg,
                dataset_cfg=OmegaConf.to_container(ds_cfg),
                meta=meta,
                rank=rank,
                world_size=world_size,
                global_return_min=global_return_min,
                global_return_max=global_return_max,
                global_pbar=global_pbar,
                returns_sidecar=returns_sidecar,
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

        global_pbar.close()

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
        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info("Saving Advantages")
            logger.info(f"{'=' * 60}")
            if tag:
                logger.info(f"  Tag: {tag}")

        # Build mixture_config content (shared across all datasets)
        tag_stats = {
            "unified_threshold": unified_threshold,
            "positive_quantile": positive_quantile,
        }

        import yaml

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
                tag=tag,
            )

            # Save mixture_config.yaml to each dataset root (only rank 0)
            if rank == 0:
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

                if tag:
                    if "tags" not in mixture_config:
                        mixture_config["tags"] = {}
                    mixture_config["tags"][tag] = tag_stats
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
