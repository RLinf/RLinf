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

"""Debug FSDP Actor for CFG training from LeRobot datasets.

This module provides DebugCFGFSDPActor, a simplified worker for training
CFG models directly from LeRobot datasets without environment rollout.

Key features:
- Uses AdvantageMixtureDataset for data loading with weighted sampling
- Pre-computed advantages from datasets (computed by compute_advantages.py)
- Positive/negative guidance selection based on advantage (bool)

Example config:
    data:
      balance_dataset_weights: true
      seed: 42
      datasets:
        - path: "/path/to/collected_data_with_advantages"
          episodes: null
          weight: 1.0
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import jax
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.datasets import TokenizePromptWithGuidance
from rlinf.datasets.vla_lib.advantage_mixture_dataset import AdvantageMixtureDataset
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models import get_model
from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
    Observation as CFGObservation,
)
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import clear_memory

# =============================================================================
# Helper Functions
# =============================================================================


def _cast_image_features(hf_dataset):
    """Cast image columns from struct to Image type for proper decoding.

    When parquet files store images as struct<bytes: binary, path: string>,
    we need to cast them to datasets.Image type for automatic decoding.

    Args:
        hf_dataset: HuggingFace dataset with struct-type image columns.

    Returns:
        Dataset with image columns cast to Image type.
    """
    from datasets import Image

    # Check if casting is needed
    features = hf_dataset.features
    needs_cast = False
    new_features = features.copy()

    for key, feat in features.items():
        # Check if this is a struct-type image (dict feature with 'bytes' field)
        # The feature type will be a dict like {'bytes': Value(...), 'path': Value(...)}
        if isinstance(feat, dict) and "bytes" in feat:
            new_features[key] = Image()
            needs_cast = True

    if needs_cast:
        from lerobot.datasets.utils import hf_transform_to_torch

        hf_dataset = hf_dataset.cast(new_features)
        hf_dataset.set_transform(hf_transform_to_torch)

    return hf_dataset


# =============================================================================
# Helper Classes
# =============================================================================


class DatasetWithAdvantage:
    """Wrapper to preserve advantage through OpenPI transform pipeline.

    OpenPI's RepackTransform removes all keys except required ones, which drops
    the advantage field. This wrapper pre-builds an index-to-advantage mapping
    at init time using efficient HF dataset column access (no image loading),
    avoiding the need to load each sample twice.

    Attributes:
        _transformed_dataset: Dataset after applying OpenPI transforms.
        _advantage_by_index: Pre-built mapping from sample index to advantage value.
        _base_dataset: Kept only as fallback when pre-building fails.
    """

    def __init__(
        self,
        base_dataset: Any,
        transformed_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None = None,
    ):
        """Initialize DatasetWithAdvantage.

        Pre-builds index-to-advantage mapping to avoid loading each sample twice
        (once from base_dataset for advantage, once from transformed_dataset).

        Args:
            base_dataset: Base dataset with advantage field (from compute_advantages.py).
            transformed_dataset: Dataset after applying OpenPI transforms.
            advantages_lookup: Optional pre-loaded advantage lookup from
                meta/advantages_{tag}.parquet. If provided, advantage is read
                from this lookup instead of from the data parquet.
        """
        self._transformed_dataset = transformed_dataset
        self._advantage_by_index = self._build_advantage_index(
            base_dataset, advantages_lookup
        )
        # Keep base_dataset only as fallback when pre-building fails
        self._base_dataset = base_dataset if self._advantage_by_index is None else None

    @staticmethod
    def _get_hf_dataset(dataset: Any) -> Any:
        """Extract the underlying HuggingFace dataset from wrapped datasets.

        Traverses TransformedDataset wrappers to find the LeRobotDataset's
        hf_dataset, which allows efficient column access without image loading.
        """
        current = dataset
        while current is not None:
            if hasattr(current, "hf_dataset"):
                return current.hf_dataset
            elif hasattr(current, "_dataset"):
                current = current._dataset
            else:
                return None
        return None

    def _build_advantage_index(
        self,
        base_dataset: Any,
        advantages_lookup: dict[tuple[int, int], bool] | None,
    ) -> dict[int, bool] | None:
        """Build mapping from sample index to advantage value.

        Uses efficient column access on the underlying HF dataset to read
        episode_index/frame_index or advantage columns without loading images.

        Returns:
            Dict mapping sample index -> advantage (bool), or None if
            the HF dataset is not accessible (falls back to slow path).
        """
        hf_dataset = self._get_hf_dataset(base_dataset)
        if hf_dataset is None:
            print(
                "[DatasetWithAdvantage] WARNING: Cannot access underlying HF dataset, "
                "falling back to per-sample advantage loading (slower)."
            )
            return None

        if advantages_lookup is not None:
            # Efficient path: read episode_index and frame_index columns directly
            # (no image decoding, just integer columns)
            ep_indices = hf_dataset["episode_index"]
            frame_indices = hf_dataset["frame_index"]
            advantage_by_index = {}
            missing_keys = []
            for i in range(len(hf_dataset)):
                key = (int(ep_indices[i]), int(frame_indices[i]))
                if key in advantages_lookup:
                    advantage_by_index[i] = advantages_lookup[key]
                else:
                    missing_keys.append(key)
            if missing_keys:
                raise ValueError(
                    f"[DatasetWithAdvantage] {len(missing_keys)} samples not found "
                    f"in advantages lookup (first 5: {missing_keys[:5]}). "
                    f"The advantages parquet does not match this dataset. "
                    f"Re-run compute_advantages.py."
                )
            return advantage_by_index

        elif "advantage" in hf_dataset.column_names:
            # Fallback: read advantage column directly (no image decoding)
            advantages = hf_dataset["advantage"]
            return {i: bool(v) for i, v in enumerate(advantages)}

        else:
            raise ValueError(
                "[DatasetWithAdvantage] No advantage data found: "
                "advantages_lookup is None, and 'advantage' column not in dataset. "
                "Run compute_advantages.py first."
            )

    def __len__(self) -> int:
        return len(self._transformed_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get sample with advantage added.

        Only loads from _transformed_dataset once (no double data loading).

        Args:
            idx: Sample index.

        Returns:
            Transformed sample dict with 'advantage' field added.
        """
        sample = self._transformed_dataset[idx]

        if self._advantage_by_index is not None:
            if idx not in self._advantage_by_index:
                raise KeyError(
                    f"[DatasetWithAdvantage] Index {idx} not found in advantage index. "
                    f"Dataset size: {len(self._transformed_dataset)}, "
                    f"advantage index size: {len(self._advantage_by_index)}."
                )
            sample["advantage"] = self._advantage_by_index[idx]
        else:
            # Slow fallback: load from base dataset (only when HF dataset not accessible)
            base_sample = self._base_dataset[idx]
            if "advantage" not in base_sample:
                raise KeyError(
                    f"[DatasetWithAdvantage] 'advantage' key not found in base_sample "
                    f"at index {idx}. Run compute_advantages.py first."
                )
            advantage = base_sample["advantage"]
            if isinstance(advantage, torch.Tensor):
                advantage = bool(advantage.item())
            sample["advantage"] = advantage

        return sample


class CFGDataLoaderImpl:
    """DataLoader wrapper for CFG training.

    Yields (observation, actions, advantage) tuples for CFG model training.
    The advantage field is used to select positive or negative guidance.

    Attributes:
        _data_config: OpenPI data configuration.
        _data_loader: Underlying PyTorch DataLoader.
    """

    def __init__(self, data_config: Any, data_loader: Any):
        """Initialize CFGDataLoaderImpl.

        Args:
            data_config: OpenPI data configuration.
            data_loader: Underlying PyTorch DataLoader.
        """
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> Any:
        """Get data configuration."""
        return self._data_config

    def __len__(self) -> int:
        """Return number of batches."""
        # TorchDataLoader stores underlying torch.utils.data.DataLoader in torch_loader property
        if hasattr(self._data_loader, "torch_loader"):
            return len(self._data_loader.torch_loader)
        # Fallback for standard DataLoader
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        """Forward set_epoch to sampler and dataset for proper shuffling each epoch."""
        # TorchDataLoader stores underlying torch.utils.data.DataLoader in torch_loader property
        torch_loader = getattr(self._data_loader, "torch_loader", self._data_loader)

        if hasattr(torch_loader, "sampler") and hasattr(
            torch_loader.sampler, "set_epoch"
        ):
            torch_loader.sampler.set_epoch(epoch)
        # AdvantageMixtureDataset has set_epoch method
        if hasattr(torch_loader, "dataset") and hasattr(
            torch_loader.dataset, "set_epoch"
        ):
            torch_loader.dataset.set_epoch(epoch)

    def __iter__(self):
        """Iterate over batches.

        Yields:
            Tuple of (observation, actions, advantage) for each batch.
        """
        for batch in self._data_loader:
            observation = CFGObservation.from_dict(batch)
            actions = batch["actions"]

            advantage = batch["advantage"]
            if not isinstance(advantage, torch.Tensor):
                advantage = torch.tensor(advantage, dtype=torch.bool)

            yield observation, actions, advantage


# =============================================================================
# Main Worker Class
# =============================================================================


class DebugCFGFSDPActor(FSDPModelManager, Worker):
    """Debug FSDP Actor for CFG training from LeRobot datasets.

    This worker loads data directly from LeRobot datasets and trains
    CFG models without environment rollout. It's designed for debugging
    and offline training scenarios.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self.enable_offload = self.cfg.actor.get("enable_offload", False)

        # For weight sync to rollout worker (for evaluation)
        self._rollout_group_name = (
            cfg.rollout.group_name if hasattr(cfg, "rollout") else None
        )
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

    def model_provider_func(self):
        model = get_model(self.cfg.actor.model)
        if model is not None:
            return model
        return super().model_provider_func()

    def init_worker(self) -> None:
        """Initialize the actor worker."""
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

        # Setup rollout weight sync ranks (for evaluation)
        self._setup_rollout_weight_dst_ranks()

    # =========================================================================
    # Weight Sync Methods (for evaluation)
    # =========================================================================

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for weight communication to rollout workers."""
        if self._rollout_group_name is None:
            self._weight_dst_rank_in_rollout = []
            return

        rollout_world_size = self._component_placement.get_world_size("rollout")
        actor_world_size = self._world_size
        rank = self._rank
        self._weight_dst_rank_in_rollout = []
        rollout_ranks_per_actor = (
            rollout_world_size + actor_world_size - 1
        ) // actor_world_size
        for i in range(rollout_ranks_per_actor):
            if i * actor_world_size + rank < rollout_world_size:
                self._weight_dst_rank_in_rollout.append(i * actor_world_size + rank)

    def sync_model_to_rollout(self) -> None:
        """Sync the model's full state dict to the rollout worker."""
        if self._rollout_group_name is None:
            return

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.enable_offload and self.is_weight_offloaded:
            self.load_param_and_grad(self.device)

        state_dict = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        for rank in self._weight_dst_rank_in_rollout:
            self.send(
                state_dict,
                self._rollout_group_name,
                rank,
                async_op=True,
            )
        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

    # =========================================================================
    # CFG DataLoader Methods
    # =========================================================================

    @staticmethod
    def _load_advantages_lookup(
        data_path: str,
        advantage_tag: str | None = None,
    ) -> dict[tuple[int, int], bool]:
        """Load advantage lookup from meta/advantages_{tag}.parquet or meta/advantages.parquet.

        Args:
            data_path: Path to LeRobot dataset.
            advantage_tag: Advantage tag name. If None, loads meta/advantages.parquet.

        Returns:
            Dict mapping (episode_index, frame_index) -> bool.
        """
        import pandas as pd

        if advantage_tag:
            meta_path = Path(data_path) / "meta" / f"advantages_{advantage_tag}.parquet"
        else:
            meta_path = Path(data_path) / "meta" / "advantages.parquet"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Advantage file not found: {meta_path}. "
                f"Run compute_advantages.py first."
            )

        adv_df = pd.read_parquet(meta_path)

        lookup = dict(
            zip(
                zip(
                    adv_df["episode_index"].values.astype(int).tolist(),
                    adv_df["frame_index"].values.astype(int).tolist(),
                ),
                adv_df["advantage"].values.astype(bool).tolist(),
            )
        )
        return lookup

    def build_cfg_dataloader(self, cfg: DictConfig):
        """Build CFG dataloader using AdvantageMixtureDataset.

        Args:
            cfg: Configuration with data.datasets section.
        """
        import lerobot.datasets.lerobot_dataset as lerobot_dataset
        import openpi.training.data_loader as openpi_data_loader
        import openpi.transforms as transforms

        from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

        # Parse config
        data_cfg = cfg.get("data", {})
        openpi_cfg = cfg.actor.model.openpi
        advantage_tag = data_cfg.get("advantage_tag", None)

        # Parse datasets from config
        datasets_config = data_cfg.get("datasets", [])
        if not datasets_config:
            raise ValueError(
                "At least one dataset must be provided in data.datasets. "
                "Each dataset should have 'path' and optionally 'episodes' and 'weight' fields."
            )

        # Get OpenPI config using first dataset path
        first_path = datasets_config[0]["path"]
        config = get_openpi_config(
            openpi_cfg.config_name,
            model_path=cfg.actor.model.model_path,
            batch_size=cfg.actor.micro_batch_size * self._world_size,
            repo_id=first_path,
        )
        data_config = config.data.create(config.assets_dirs, config.model)

        # Build transforms with TokenizePromptWithGuidance
        model_transforms = self._build_model_transforms(data_config)
        norm_stats = data_config.norm_stats or {}

        # Load datasets sequentially
        datasets_with_weights = []
        for ds_config in datasets_config:
            data_path = ds_config["path"]
            episodes = ds_config.get("episodes")
            weight = ds_config.get("weight", 1.0)

            # 1. Create LeRobotDataset
            dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(data_path)
            base_dataset = lerobot_dataset.LeRobotDataset(
                data_path,
                episodes=episodes,
                delta_timestamps={
                    key: [
                        t / dataset_meta.fps for t in range(config.model.action_horizon)
                    ]
                    for key in data_config.action_sequence_keys
                },
            )

            # Cast image features for backward compatibility with old data
            base_dataset.hf_dataset = _cast_image_features(base_dataset.hf_dataset)

            # Fix episode_data_index if using specific episodes
            if episodes is not None:
                self._fix_episode_data_index(base_dataset, episodes)

            # Apply prompt transform if needed
            if data_config.prompt_from_task:
                base_dataset = openpi_data_loader.TransformedDataset(
                    base_dataset,
                    [transforms.PromptFromLeRobotTask(dataset_meta.tasks)],
                )

            # 2. Apply OpenPI transforms
            # Note: RepackTransform strips all keys except OpenPI required ones,
            # so DatasetWithAdvantage is needed to restore advantage field
            transforms_list = [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=data_config.use_quantile_norm
                ),
                *model_transforms,
            ]
            transformed_dataset = openpi_data_loader.TransformedDataset(
                base_dataset, transforms_list
            )

            # 3. Load advantage lookup from meta parquet (required)
            advantages_lookup = self._load_advantages_lookup(data_path, advantage_tag)
            if self._rank == 0:
                adv_filename = (
                    f"advantages_{advantage_tag}.parquet"
                    if advantage_tag
                    else "advantages.parquet"
                )
                print(
                    f"[DebugCFGFSDPActor] Loaded advantages from "
                    f"meta/{adv_filename} ({len(advantages_lookup)} entries)"
                )

            # 4. Wrap with DatasetWithAdvantage (restores advantage after transforms)
            final_dataset = DatasetWithAdvantage(
                base_dataset=base_dataset,
                transformed_dataset=transformed_dataset,
                advantages_lookup=advantages_lookup,
            )

            datasets_with_weights.append((final_dataset, weight))

            if self._rank == 0:
                print(
                    f"[DebugCFGFSDPActor] Loaded dataset: {data_path} "
                    f"({len(final_dataset)} samples, weight={weight})"
                )

        # Use AdvantageMixtureDataset for weighted sampling
        combined_dataset = AdvantageMixtureDataset(
            datasets=datasets_with_weights,
            mode="train",
            balance_dataset_weights=data_cfg.get("balance_dataset_weights", True),
            seed=data_cfg.get("seed", 42),
        )

        # Create DataLoader
        torch_data_loader = self._create_torch_dataloader(
            combined_dataset, config, openpi_data_loader
        )

        self.cfg_data_loader = CFGDataLoaderImpl(data_config, torch_data_loader)
        self._current_epoch = -1  # Will be set properly by set_global_step

    def _build_model_transforms(self, data_config: Any) -> list:
        """Build model transforms with TokenizePromptWithGuidance."""
        tokenizer = None
        for t in data_config.model_transforms.inputs:
            if hasattr(t, "tokenizer"):
                tokenizer = t.tokenizer
                break

        if tokenizer is None:
            raise ValueError("Cannot find tokenizer in model_transforms")

        model_transforms = []
        for t in data_config.model_transforms.inputs:
            if type(t).__name__ == "TokenizePrompt":
                model_transforms.append(
                    TokenizePromptWithGuidance(
                        tokenizer=tokenizer,
                        discrete_state_input=getattr(t, "discrete_state_input", False),
                    )
                )
            else:
                model_transforms.append(t)

        return model_transforms

    def _fix_episode_data_index(self, dataset: Any, episodes: list) -> None:
        """Fix LeRobotDataset episode_data_index when using specific episodes."""
        ep_idx_mapping = {ep: i for i, ep in enumerate(sorted(episodes))}
        max_ep_idx = max(episodes) + 1

        old_from = dataset.episode_data_index["from"]
        old_to = dataset.episode_data_index["to"]

        new_from = torch.full((max_ep_idx,), -1, dtype=old_from.dtype)
        new_to = torch.full((max_ep_idx,), -1, dtype=old_to.dtype)

        for orig_ep, new_idx in ep_idx_mapping.items():
            new_from[orig_ep] = old_from[new_idx]
            new_to[orig_ep] = old_to[new_idx]

        dataset.episode_data_index["from"] = new_from
        dataset.episode_data_index["to"] = new_to

    def _create_torch_dataloader(
        self,
        dataset: Any,
        config: Any,
        openpi_data_loader: Any,
        shuffle: bool = True,
    ) -> Any:
        """Create PyTorch DataLoader with distributed sampler."""
        batch_size = config.batch_size
        sampler = None

        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // self._world_size
        else:
            local_batch_size = batch_size

        return openpi_data_loader.TorchDataLoader(
            dataset,
            local_batch_size=local_batch_size,
            sharding=None,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_batches=None,
            num_workers=config.num_workers,
            seed=config.seed,
            framework="pytorch",
        )

    # =========================================================================
    # Training
    # =========================================================================

    def run_training(self):
        """Run one training step using CFG DataLoader.

        Returns:
            Dictionary of training metrics.
        """
        if self.enable_offload:
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        self.model.train()

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        metrics = {}

        for idx in range(self.gradient_accumulation):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
            )

            # Get data from CFG DataLoader (iterator managed by set_global_step)
            observation, actions, advantage = next(self.cfg_data_iter)

            observation = jax.tree.map(
                lambda x: torch.as_tensor(x, device=self.device).contiguous().clone(),
                observation,
            )
            actions = actions.to(torch.float32).to(self.device)
            advantage = advantage.to(self.device)

            with self.amp_context:
                result = self.model(
                    data={
                        "observation": observation,
                        "actions": actions,
                        "advantage": advantage,
                    },
                )
                if isinstance(result, tuple) and len(result) == 2:
                    losses, metrics_data = result
                else:
                    losses = result
                    metrics_data = None
                if isinstance(losses, (list, tuple)):
                    losses = torch.stack(losses)
                elif not isinstance(losses, torch.Tensor):
                    losses = torch.tensor(
                        losses, device=self.device, dtype=torch.float32
                    )
                loss = losses.mean()

            loss = loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            batch_metrics = {"loss": loss.detach().item()}
            if metrics_data is not None:
                batch_metrics.update(metrics_data)
            append_to_dict(metrics, batch_metrics)

        torch.cuda.empty_cache()
        grad_norm, lr_list = self.optimizer_step()

        data = {
            "grad_norm": grad_norm,
            "lr": lr_list[0],
        }
        if len(lr_list) > 1:
            data["critic_lr"] = lr_list[1]
        append_to_dict(metrics, data)

        # LR scheduler step
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        # Handle CFG-specific metrics
        special_keys = [
            "conditional_count",
            "unconditional_count",
            "conditional_loss",
            "unconditional_loss",
            "positive_guidance_count",
            "negative_guidance_count",
        ]

        sum_metric_dict = {
            key: np.sum(value) for key, value in metrics.items() if key in special_keys
        }
        sum_metric_dict = all_reduce_dict(
            sum_metric_dict, op=torch.distributed.ReduceOp.SUM
        )

        mean_metric_dict = {
            key: np.mean(value)
            for key, value in metrics.items()
            if key not in special_keys
        }
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        # Calculate ratios
        total = sum_metric_dict.get("conditional_count", 0) + sum_metric_dict.get(
            "unconditional_count", 0
        )
        if total > 0:
            mean_metric_dict.update(
                {
                    "conditional_ratio": sum_metric_dict.get("conditional_count", 0)
                    / total,
                    "unconditional_ratio": sum_metric_dict.get("unconditional_count", 0)
                    / total,
                }
            )
        for key in ["conditional", "unconditional"]:
            count = sum_metric_dict.get(f"{key}_count", 0)
            if count > 0:
                mean_metric_dict[f"{key}_loss"] = (
                    sum_metric_dict.get(f"{key}_loss", 0) / count
                )

        # Calculate positive/negative guidance ratio
        guidance_total = sum_metric_dict.get(
            "positive_guidance_count", 0
        ) + sum_metric_dict.get("negative_guidance_count", 0)
        if guidance_total > 0:
            mean_metric_dict["positive_guidance_ratio"] = (
                sum_metric_dict.get("positive_guidance_count", 0) / guidance_total
            )
            mean_metric_dict["negative_guidance_ratio"] = (
                sum_metric_dict.get("negative_guidance_count", 0) / guidance_total
            )

        return mean_metric_dict

    def set_global_step(self, global_step: int) -> None:
        """Set global step and handle epoch-based iterator reset.

        This method calculates the current epoch based on global_step and
        resets the data iterator when epoch changes, ensuring proper data
        shuffling across epochs.

        Args:
            global_step: Current global training step.
        """
        self.global_step = global_step

        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

        # Skip if dataloader not initialized
        if not hasattr(self, "cfg_data_loader"):
            return

        loader_len = len(self.cfg_data_loader)
        if loader_len == 0:
            return

        # Calculate current epoch
        grad_accum = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        steps_per_epoch = max(1, loader_len // grad_accum)
        new_epoch = global_step // steps_per_epoch

        # Initialize or reset iterator only when epoch changes
        current_epoch = getattr(self, "_current_epoch", -1)
        if current_epoch != new_epoch:
            self._current_epoch = new_epoch
            self.cfg_data_loader.set_epoch(new_epoch)
            self.cfg_data_iter = iter(self.cfg_data_loader)
