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
Compute advantages for CFG-RL training using a trained ValueCriticModel.

Advantage formula: A = normalize(r_{t:t+N}) + gamma^N * V(o_{t+N}) - V(o_t)

Usage:
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
from lerobot.common.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from rlinf.data.datasets.recap.utils import (
    decode_image_struct_batch,
    load_return_stats_from_dataset,
    load_returns_sidecar,
)
from rlinf.models.embodiment.value_model.modeling_critic import ValueCriticModel

logger = logging.getLogger(__name__)


def setup_distributed(cfg: DictConfig) -> tuple[int, int, str]:
    """Initialize torch.distributed for torchrun-launched processes.

    Args:
        cfg: Configuration with distributed settings

    Returns:
        Tuple of (rank, world_size, device_string)
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        dist_cfg = cfg.get("distributed", {})
        backend = dist_cfg.get("backend", "nccl")
        timeout_seconds = dist_cfg.get("timeout", 1800)

        if not dist.is_initialized():
            from datetime import timedelta

            dist.init_process_group(
                backend=backend,
                timeout=timedelta(seconds=timeout_seconds),
            )

        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"

        if rank == 0:
            logger.info(f"Distributed mode enabled: {world_size} GPUs")
            logger.info(f"  Backend: {backend}, Timeout: {timeout_seconds}s")

        return rank, world_size, device

    # Single GPU fallback
    return 0, 1, "cuda"


def cleanup_distributed():
    """Clean up distributed process group if initialized."""
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

    all_dfs = [None] * world_size
    dist.all_gather_object(all_dfs, local_df.to_dict("records"))

    all_records = []
    for df_records in all_dfs:
        if df_records:
            all_records.extend(df_records)

    merged_df = pd.DataFrame(all_records)
    if len(merged_df) > 0:
        merged_df = merged_df.sort_values(["episode_index", "frame_index"]).reset_index(
            drop=True
        )

    return merged_df


# Maps LeRobot dataset keys to value model observation format
KEY_MAPPINGS = {
    "franka": {
        # Multi-cam format (front_cam + wrist_cam)
        "observation.images.front_cam": "observation/images/front_cam",
        "observation.images.wrist_cam": "observation/images/wrist_cam",
        "observation.state.tcp_pose": "observation/state/tcp_pose",
        "observation.state.gripper_pose": "observation/state/gripper_pose",
        # Single-cam format (image + state)
        "observation.images.image": "observation/image",
        "image": "observation/image",
        "state": "observation/state",
        "task": "prompt",
    },
    "franka_co_train": {
        # Single-cam + wrist format (image + wrist_image + state)
        "image": "observation/image",
        "wrist_image": "observation/wrist_image",
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
        "observation.image": "observation/image",
        "observation.wrist_image": "observation/wrist_image",
        "observation.state": "observation/state",
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
        """Update with a single value (Welford's online algorithm)."""
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


def _parse_value_model_kwargs(cfg: DictConfig) -> dict:
    """Extract ValueCriticModel.from_checkpoint kwargs from Hydra config."""
    checkpoint_path = cfg.advantage.value_checkpoint
    if checkpoint_path is None:
        raise ValueError("advantage.value_checkpoint must be specified")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    data_cfg = cfg.data
    model_cfg = cfg.advantage.get("model", {})

    robot_type = data_cfg.get("robot_type", "libero")
    if "train_data_paths" in data_cfg and len(data_cfg.train_data_paths) > 0:
        robot_type = data_cfg.train_data_paths[0].get("robot_type", robot_type)

    return {
        "checkpoint_dir": checkpoint_path,
        "env_type": robot_type,
        "model_type": data_cfg.get("model_type", "pi05"),
        "num_return_bins": model_cfg.get("num_bins", 201),
        "return_min": model_cfg.get("v_min", -1.0),
        "return_max": model_cfg.get("v_max", 0.0),
        "critic_expert_variant": model_cfg.get("critic_expert_variant", "gemma_100m"),
        "tokenizer_path": model_cfg.get("tokenizer_path", None),
        "siglip_path": model_cfg.get("siglip_path", None),
        "gemma3_path": model_cfg.get("gemma3_path", None),
    }


def load_lerobot_dataset(
    dataset_path: Path,
    returns_tag: str | None = None,
) -> tuple[LeRobotDataset, dict, LeRobotDatasetMetadata, dict | None]:
    """Load a LeRobot dataset WITHOUT delta_timestamps for fast single-row access.

    Loading without delta_timestamps is ~50x faster (6ms vs 580ms per sample)
    because it avoids expensive multi-timestep parquet reads and image decoding.
    The ValueInferenceDataset handles multi-timestep access via separate dataset[idx]
    and dataset[idx+N] calls instead.

    Also loads ``meta/returns_{tag}.parquet`` sidecar if present.

    Args:
        dataset_path: Path to dataset
        returns_tag: Optional tag for the returns sidecar filename

    Returns:
        Tuple of (dataset, tasks_dict, metadata, returns_sidecar)
    """
    meta = LeRobotDatasetMetadata(str(dataset_path))

    logger.info(f"Dataset features: {list(meta.features.keys())}")

    returns_sidecar = load_returns_sidecar(dataset_path, returns_tag=returns_tag)

    # Validate: accept either features in parquets OR sidecar
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
    dataset.hf_dataset.set_transform(decode_image_struct_batch)

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


def build_obs(
    sample: dict,
    robot_type: str,
    tasks: dict,
) -> dict[str, Any]:
    """Build raw observation dict for ValueCriticModel from a single-timestep sample.

    Args:
        sample: Single-timestep sample from LeRobot dataset (no delta_timestamps)
        robot_type: Robot type for key mapping
        tasks: Task descriptions dict

    Returns:
        Raw observation dict compatible with ValueCriticModel.infer()
    """
    if robot_type not in KEY_MAPPINGS:
        raise ValueError(
            f"Unknown robot_type {robot_type!r}. "
            f"Supported types: {list(KEY_MAPPINGS.keys())}. "
            "Add a new entry to KEY_MAPPINGS if this is a new robot type."
        )
    key_map = KEY_MAPPINGS[robot_type]
    obs = {}

    for src_key, dst_key in key_map.items():
        if src_key == "task":
            if "task" in sample:
                obs[dst_key] = str(to_scalar(sample["task"]))
            elif "task_index" in sample and tasks:
                task_idx = int(to_scalar(sample["task_index"]))
                if task_idx not in tasks:
                    raise ValueError(
                        f"task_index {task_idx} not found in tasks dict. "
                        f"Available task indices: {list(tasks.keys())}. "
                        "Check that meta/tasks.jsonl is complete."
                    )
                obs[dst_key] = tasks[task_idx]
            else:
                raise ValueError(
                    "Sample has neither 'task' nor 'task_index' field. "
                    "Cannot determine task prompt for value model inference."
                )
        elif src_key in sample:
            val = to_numpy(sample[src_key])
            obs[dst_key] = val

    return obs


class ValueInferenceDataset(torch.utils.data.Dataset):
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

        obs = build_obs(sample, self.robot_type, self.tasks)

        if self.input_transform is not None:
            obs = self.input_transform(
                {
                    k: v.copy() if isinstance(v, np.ndarray) else v
                    for k, v in obs.items()
                }
            )
        if self.prepare_observation_cpu is not None:
            obs = self.prepare_observation_cpu(obs)

        if self.returns_sidecar is not None and ep_idx in self.returns_sidecar:
            ep_data = self.returns_sidecar[ep_idx]
            true_return = float(ep_data["return"][frame_idx])
            reward = float(ep_data["reward"][frame_idx])
        else:
            if "return" not in sample:
                raise ValueError(
                    f"Sample (episode={ep_idx}, frame={frame_idx}) missing 'return' field "
                    "and no returns sidecar available. "
                    "Run compute_returns.py first."
                )
            true_return = float(to_scalar(sample["return"]))
            if "reward" not in sample:
                raise ValueError(
                    f"Sample (episode={ep_idx}, frame={frame_idx}) missing 'reward' field "
                    "and no returns sidecar available. "
                    "Run compute_returns.py first."
                )
            reward = float(to_scalar(sample["reward"]))

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
    """Custom collate function for ValueInferenceDataset.

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

    Args:
        value_model: Trained ValueCriticModel with input transforms and batch inference
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

    ret_min = global_return_min
    ret_max = global_return_max

    if rank == 0:
        logger.info(f"  Using global return_range: [{ret_min}, {ret_max}]")
        logger.info(f"  Using batch inference with batch_size: {batch_size}")

    # Maps [ret_min, ret_max] -> [-1, 0]
    ret_range = ret_max - ret_min

    def normalize(x):
        if ret_range <= 0:
            return -0.5
        return (x - ret_min) / ret_range - 1.0

    gamma_powers = np.array([gamma**i for i in range(action_horizon)], dtype=np.float64)

    max_samples = cfg.advantage.get("max_samples", None)
    total_samples = (
        len(dataset) if max_samples is None else min(len(dataset), max_samples)
    )

    shard_start, shard_end = get_shard_indices(total_samples, rank, world_size)
    shard_size = shard_end - shard_start

    # Extend range so idx + lookahead can be looked up for V(o_{t+N})
    extended_end = (
        shard_start
        if shard_size == 0
        else min(shard_end + action_horizon, len(dataset))
    )
    extended_size = extended_end - shard_start

    ep_ends = {}
    for ep_idx in range(len(dataset.episode_data_index["to"])):
        ep_ends[ep_idx] = int(dataset.episode_data_index["to"][ep_idx].item())

    if rank == 0:
        logger.info(
            f"Computing advantages for {total_samples} samples (total in dataset: {len(dataset)})..."
        )
        logger.info(f"  gamma: {gamma}, advantage_lookahead_step: {action_horizon}")
        logger.info(f"  return_range: [{ret_min}, {ret_max}]")
        logger.info("  Using ValueCriticModel with batch inference")
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

    num_dataloader_workers_per_gpu = cfg.advantage.get(
        "num_dataloader_workers_per_gpu",
        cfg.advantage.get("num_dataloader_workers", 8),
    )
    prefetch_factor = cfg.advantage.get("prefetch_factor", 2)

    # Periodically flushed to disk to prevent OOM
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

    v_curr_stats = RunningStats("V(o_t)")
    v_next_stats = RunningStats("V(o_N)")
    reward_sum_raw_stats = RunningStats("R_raw")

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
        for k in results:
            results[k] = []
        del temp_df
        gc.collect()
        if rank == 0:
            logger.info(
                f"  Flushed chunk {len(temp_files)} ({chunk_size} samples) to disk. "
                f"Total flushed: {flushed_sample_count}"
            )

    from functools import partial

    processor = getattr(value_model, "processor", None)
    worker_cpu_prep = partial(
        value_model.__class__._prepare_observation_cpu, processor=processor
    )

    cpu_prep_in_workers = num_dataloader_workers_per_gpu > 0

    if rank == 0:
        logger.info(
            f"  Using DataLoader: workers_per_gpu={num_dataloader_workers_per_gpu}, "
            f"prefetch_factor={prefetch_factor}, batch_size={batch_size}, "
            f"cpu_prep_in_workers={cpu_prep_in_workers}"
        )

    advantage_dataset = ValueInferenceDataset(
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
        num_workers=num_dataloader_workers_per_gpu,
        prefetch_factor=prefetch_factor if num_dataloader_workers_per_gpu > 0 else None,
        persistent_workers=num_dataloader_workers_per_gpu > 0,
        collate_fn=advantage_collate_fn,
        shuffle=False,
        pin_memory=True,
    )

    if rank == 0:
        logger.info(
            f"Phase 1: inferring V(o_t) for {extended_size} samples in {len(dataloader)} batches"
        )

    v_values = np.full(extended_size, np.nan, dtype=np.float64)
    meta_ep_idx = np.full(extended_size, -1, dtype=np.int64)
    meta_frame_idx = np.full(extended_size, -1, dtype=np.int64)
    meta_return = np.full(extended_size, np.nan, dtype=np.float64)
    meta_reward = np.full(extended_size, np.nan, dtype=np.float64)
    filled_mask = np.zeros(extended_size, dtype=bool)

    def process_value_batch(obs_list: list[dict], meta_list: list[dict]):
        """Run GPU inference for V(o_t) and store batch results."""
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

    # Prefetch next batch while GPU processes current batch
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

            if batch_idx + 1 < batch_count:
                pending_future = prefetch_pool.submit(_fetch_next)

            _t0 = _time.perf_counter()
            process_value_batch(*batch)
            _t_infer = _time.perf_counter() - _t0
            _t_infer_total += _t_infer

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

    # Phase 2: compute advantages using precomputed V(o_t) values
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

        v_curr_stats.update(v_curr)
        v_next_stats.update(v_next)
        reward_sum_raw_stats.update(reward_sum_raw)

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

    flush_results_to_disk()

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

    if temp_files:
        if rank == 0:
            logger.info(
                f"Merging {len(temp_files)} temporary chunks ({flushed_sample_count} total samples)..."
            )
        merged_df = pd.concat(
            [pd.read_parquet(f) for f in temp_files], ignore_index=True
        )
        for f in temp_files:
            f.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass
        return merged_df
    else:
        return pd.DataFrame(results)


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

    if world_size > 1:
        dist.barrier()


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
    rank, world_size, device = setup_distributed(cfg)

    logging.basicConfig(level=logging.INFO)
    if rank == 0:
        logger.info("Starting advantage computation...")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    else:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        value_model = ValueCriticModel.from_checkpoint(
            **_parse_value_model_kwargs(cfg), device=device
        )

        all_advantages = []
        dataset_results = {}

        # Priority: 1) global config  2) compute from all datasets' stats.json
        data_cfg = cfg.data
        global_return_min = data_cfg.get("return_min", None)
        global_return_max = data_cfg.get("return_max", None)

        if global_return_min is None or global_return_max is None:
            all_mins = []
            all_maxs = []
            for ds_cfg in cfg.data.train_data_paths:
                ds_path = Path(ds_cfg.dataset_path)
                ds_min, ds_max = load_return_stats_from_dataset(ds_path)
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
                missing = []
                if global_return_min is None:
                    missing.append("data.return_min")
                if global_return_max is None:
                    missing.append("data.return_max")
                raise ValueError(
                    "Cannot determine return range for normalization. "
                    "No stats.json found in any dataset and the following config "
                    f"fields are not set: {missing}. Either run compute_returns.py "
                    "first (generates stats.json) or set data.return_min and "
                    "data.return_max explicitly in the config."
                )
        else:
            if rank == 0:
                logger.info(
                    f"Using global return range from config: "
                    f"[{global_return_min}, {global_return_max}]"
                )

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

        if rank == 0:
            if returns_tag != tag:
                logger.info(f"Returns tag: {returns_tag}, Advantages tag: {tag}")

        for ds_cfg in cfg.data.train_data_paths:
            ds_path = Path(ds_cfg.dataset_path)
            if rank == 0:
                logger.info(f"\n{'=' * 60}")
                logger.info(f"Processing dataset: {ds_path.name}")
                logger.info(f"{'=' * 60}")

            dataset, tasks, meta, returns_sidecar = load_lerobot_dataset(
                ds_path, returns_tag=returns_tag
            )

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

            if world_size > 1:
                dist.barrier()
                df = gather_all_advantages(local_df, rank, world_size)
            else:
                df = local_df

            df["dataset_name"] = ds_path.name
            all_advantages.append(df["advantage"].values)
            dataset_results[ds_path] = {
                "df": df,
                "config": OmegaConf.to_container(ds_cfg),
            }

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

            logger.info("\n  Per-dataset positive rates (using unified threshold):")
            for i, (ds_path, result) in enumerate(dataset_results.items()):
                ds_advantages = all_advantages[i]
                positive_count = (ds_advantages >= unified_threshold).sum()
                positive_rate = positive_count / len(ds_advantages) * 100
                logger.info(
                    f"    {ds_path.name}: {positive_count}/{len(ds_advantages)} ({positive_rate:.1f}%)"
                )

        if rank == 0:
            logger.info(f"\n{'=' * 60}")
            logger.info("Saving Advantages")
            logger.info(f"{'=' * 60}")
            if tag:
                logger.info(f"  Tag: {tag}")

        tag_stats = {
            "unified_threshold": unified_threshold,
            "positive_quantile": positive_quantile,
        }

        import yaml

        for ds_path, result in dataset_results.items():
            df = result["df"]
            dataset_type = result["config"].get("type")
            save_advantages_to_dataset(
                dataset_path=ds_path,
                advantages_df=df,
                threshold=unified_threshold,
                dataset_type=dataset_type,
                rank=rank,
                world_size=world_size,
                tag=tag,
            )

            if rank == 0:
                mixture_config_path = ds_path / "mixture_config.yaml"

                if mixture_config_path.exists():
                    with open(mixture_config_path, "r") as f:
                        mixture_config = yaml.safe_load(f) or {}
                else:
                    mixture_config = {}

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
        cleanup_distributed()


if __name__ == "__main__":
    main()
