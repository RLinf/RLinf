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

"""Reward Workers for RLinf.

This module provides:
- RewardWorker: For inference/computing rewards during RL training
- FSDPRewardWorker: For training reward models with FSDP (like FSDPSftWorker)
"""

import re as _re

import os
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.models.embodiment.reward import ResNetRewardModel
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import clear_memory
from rlinf.utils.logging import get_logger

from rlinf.data.datasets.reward_model import (
    load_episodes_with_labels,
    balance_and_split_by_episode,
    RewardBinaryDataset,
)

from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.utils.down_sampling import down_sample_batch
from rlinf.utils.comm_mapping import CommMapper

class RewardWorker(Worker):
    """Reward Worker for inference during RL training."""

    def __init__(self, cfg: DictConfig, placement: HybridComponentPlacement = None):
        Worker.__init__(self)
        self.cfg = cfg

        self.placement = placement

        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )
        self.do_down_sampling = self.cfg.algorithm.get("down_sampling", {}).get("do_down_sampling", False)
        if self.do_down_sampling:
            self.down_sampling_config = self.cfg.algorithm.get("down_sampling", {}).get("down_sampling_config", {})

    def init_worker(self):
        if self.cfg.reward.use_reward_model:
            self.reward_model = get_reward_model_class(self.cfg.reward.model.model_type)(self.cfg.reward.model)
        else:
            self.rule_based_reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

        if self.cfg.reward.get("tokenizer", None) is not None:
            self.tokenizer = hf_tokenizer(self.cfg.reward.tokenizer.tokenizer_model)

    @Worker.timer("compute_rewards")
    def compute_rewards(
        self, input_channel: Channel, output_channel: Channel, total_batch_size=None
    ):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        if total_batch_size is None:
            total_batch_size_per_dp = self.total_batch_size_per_dp
        else:
            assert total_batch_size % self._world_size == 0, (
                f"Total batch size {total_batch_size} is not divisible by world size {self._world_size}"
            )
            total_batch_size_per_dp = total_batch_size // self._world_size
        while recv_batch_size < total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            if rollout_result.rewards is None:
                if self.cfg.reward.use_reward_model:
                    raise NotImplementedError
                else:
                    rollout_result.rewards = self._compute_rule_based_rewards(
                        rollout_result
                    )
            if self.do_down_sampling:
                if rollout_result.response_texts is None:
                    rollout_result.response_texts = [
                        self.tokenizer.decode(ids, skip_special_tokens=True)
                        for ids in rollout_result.response_ids
                    ]
                rollout_result = down_sample_batch(rollout_result, self.down_sampling_config)
            # answer is not needed in training
            rollout_result.answers = None

            output_channel.put(rollout_result, async_op=True)

        assert recv_batch_size == total_batch_size_per_dp, (
            f"Expected {total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def _compute_rule_based_rewards(self, rollout_result: RolloutResult):
        # Decode only the generated tokens; response_ids are already the post-prompt tokens
        texts = rollout_result.response_texts
        if texts is None:
            texts = self.tokenizer.batch_decode(
                rollout_result.response_ids, skip_special_tokens=True
            )

        kwargs = {}
        if getattr(self.cfg.reward, "use_prompt", False):
            prompts = rollout_result.prompt_texts
            if prompts is None:
                prompts = self.tokenizer.batch_decode(
                    rollout_result.prompt_ids, skip_special_tokens=True
                )
            kwargs["prompts"] = prompts
        scores = self.rule_based_reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )


# =============================================================================
# FSDP Reward Worker for Training (like FSDPSftWorker)
# =============================================================================

class FSDPRewardWorker(FSDPModelManager, Worker):
    """FSDP-based worker for reward model training.

    This follows the same pattern as FSDPSftWorker:
    - Inherits FSDPModelManager for optimizer, scheduler, FSDP setup
    - Implements model_provider_func() to return ResNetRewardModel
    - Implements build_dataloader() for data loading
    - Implements run_training() for training loop
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self._component_placement = HybridComponentPlacement(cfg, Cluster())
        self.data_loader, self.val_loader = self.build_dataloader()
        self.data_iter = iter(self.data_loader) if self.data_loader else None

        # Training step counter for validation interval
        self._training_step = 0
        self._val_interval = cfg.runner.get("val_check_interval", 50)

    def init_worker(self):
        """Initialize model and optimizer using base class."""
        self.tokenizer = None
        self.is_lora = False
        self.setup_model_and_optimizer()

        self.logger.info(
            f"Initialized FSDPRewardWorker with "
            f"{sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def _save_dataset_images(
        self, dataset: RewardBinaryDataset, save_dir: str, split: str
    ) -> None:
        """Save all images from dataset for debugging.

        Args:
            dataset: The dataset to save images from.
            save_dir: Base directory to save images.
            split: 'train' or 'val' to indicate which split.
        """
        from PIL import Image
        from tqdm import tqdm

        success_dir = os.path.join(save_dir, split, "success")
        fail_dir = os.path.join(save_dir, split, "fail")
        os.makedirs(success_dir, exist_ok=True)
        os.makedirs(fail_dir, exist_ok=True)

        success_count = 0
        fail_count = 0

        self.logger.info(f"Saving {len(dataset)} {split} images to {save_dir}...")

        for idx in tqdm(range(len(dataset)), desc=f"Saving {split} images"):
            img_tensor, label = dataset[idx]

            # Convert tensor to numpy image
            # img_tensor is (C, H, W) float [0, 1]
            img_np = (img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8")
            img_pil = Image.fromarray(img_np)

            is_success = label.item() > 0.5
            if is_success:
                img_pil.save(os.path.join(success_dir, f"{split}_{idx:05d}.png"))
                success_count += 1
            else:
                img_pil.save(os.path.join(fail_dir, f"{split}_{idx:05d}.png"))
                fail_count += 1

        self.logger.info(f"Saved {split} images: {success_count} success, {fail_count} fail")

    def model_provider_func(self) -> torch.nn.Module:
        """Provide the ResNet reward model."""
        model_cfg = self.cfg.actor.model
        model = ResNetRewardModel(model_cfg)
        return model

    def build_dataloader(self) -> tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Build training and validation dataloaders.

        Uses per-frame 'is_obj_placed' labels from infos instead of episode-level labels.
        Splits by EPISODE to prevent data leakage (adjacent frames are too similar).
        Uses sparse sampling (start/middle/end) to reduce redundancy.
        """
        data_cfg = self.cfg.get("data", {})
        data_path = data_cfg.get("data_path")

        if not data_path or not os.path.exists(data_path):
            self.logger.warning(f"Data path not found: {data_path}")
            return None, None

        # Load episodes with configurable sampling
        num_samples = data_cfg.get("num_samples_per_episode", 5)
        self.logger.info(
            f"Loading episodes from {data_path} with {num_samples} samples per episode..."
        )
        episodes = load_episodes_with_labels(data_path, num_samples)

        if len(episodes) == 0:
            self.logger.warning("No episodes loaded from dataset")
            return None, None

        # Split by EPISODE (prevents data leakage), sample with fail:success ratio
        val_split = data_cfg.get("val_split", 0.2)
        fail_success_ratio = data_cfg.get("fail_success_ratio", 2.0)
        train_dataset, val_dataset = balance_and_split_by_episode(
            episodes, val_split, fail_success_ratio
        )

        if len(train_dataset) == 0:
            self.logger.warning("Training dataset is empty after balancing")
            return None, None

        # Debug: save training images to verify data pipeline
        debug_save_dir = data_cfg.get("debug_save_dir", None)
        if debug_save_dir and self._rank == 0:
            self._save_dataset_images(train_dataset, debug_save_dir, "train")
            self._save_dataset_images(val_dataset, debug_save_dir, "val")

        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=True,
        )

        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=False,
        )

        batch_size = self.cfg.actor.micro_batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=data_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=False,
        )

        self.logger.info(
            f"Created dataloaders: {len(train_dataset)} train, {len(val_dataset)} val"
        )

        return train_loader, val_loader

    def run_training(self) -> dict[str, float]:
        """Run one training iteration with gradient accumulation.

        Follows the same pattern as FSDPSftWorker.run_training().
        """
        # Check if data loader is available
        if self.data_iter is None or self.data_loader is None:
            raise RuntimeError(
                "Data loader is not available. Please check that:\n"
                f"  1. Data path exists: {self.cfg.get('data', {}).get('data_path')}\n"
                "  2. Data path is an absolute path (not relative)\n"
                "  3. Data files (*_success.pkl and *_fail.pkl) are present"
            )

        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
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

                # Get batch (image, label)
                try:
                    images, labels = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    images, labels = next(self.data_iter)

                # Move to device: images shape is (B, C, H, W), labels shape is (B,)
                images = images.to(self.device)
                labels = labels.to(self.device)

                with self.amp_context:
                    # Forward pass - loss computed inside model
                    outputs = self.model(images, labels)
                    loss = outputs["loss"]

                loss = loss / self.gradient_accumulation
                with backward_ctx:
                    self.grad_scaler.scale(loss).backward()

                # Accumulate metrics
                append_to_dict(
                    metrics,
                    {
                        "loss": outputs["loss"].item(),
                        "accuracy": outputs["accuracy"].item(),
                        "probabilities_mean": outputs["probabilities"].mean().item(),
                    },
                )

            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # Collect stats
            lr_value = (
                lr_list[0] if len(lr_list) > 0 else self.optimizer.param_groups[0]["lr"]
            )
            grad_norm_value = (
                float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            append_to_dict(
                metrics,
                {
                    "learning_rate": lr_value,
                    "grad_norm": grad_norm_value,
                },
            )

            self.lr_scheduler.step()

            clear_memory()
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            # Increment step counter and run validation at interval
            self._training_step += 1
            if self._training_step % self._val_interval == 0:
                val_metrics = self.run_validation()
                train_metrics.update(val_metrics)

            return train_metrics

    def run_validation(self) -> dict[str, float]:
        """Run validation over the entire validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        metrics = {}

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                with self.amp_context:
                    outputs = self.model(images, labels)

                append_to_dict(
                    metrics,
                    {
                        "val_loss": outputs["loss"].item(),
                        "val_accuracy": outputs["accuracy"].item(),
                        "val_probabilities_mean": outputs["probabilities"]
                        .mean()
                        .item(),
                    },
                )

        val_metrics = {key: np.mean(value) for key, value in metrics.items()}
        val_metrics = all_reduce_dict(val_metrics, op=torch.distributed.ReduceOp.AVG)

        return val_metrics


# =============================================================================
# Image Reward Worker for Inference (with Channel Communication)
# =============================================================================


class ImageRewardWorker(RewardWorker):

    def __init__(self, cfg: DictConfig, placement: HybridComponentPlacement):
        super().__init__(cfg, placement)

        # Device setup
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        self.enable_offload = self.cfg.reward.get("enable_offload", True)

        self.reward_threshold = 0.6

    def init_worker(self):
        """Initialize the reward model from checkpoint."""
        model_cfg = self.cfg.reward.get("model", {})

        # build model
        self.model = ResNetRewardModel(model_cfg)

        model_path = model_cfg.get("model_path", None)
        if model_path is not None:
            self._load_model(model_path)

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        self.dst_ranks = {
            "train": self._setup_dst_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }
        self.src_ranks = {
            "train": self._setup_src_ranks(
                self.total_num_train_envs // self.num_pipeline_stages
            ),
        }

    def _load_model(self, model_path):
        # load state dict
        if model_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            state_dict = load_file(model_path)
        else:
            state_dict = torch.load(
                model_path, map_location="cpu", weights_only=False
            )

        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            for prefix in ["module.", "_orig_mod.", "model."]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix) :]
            # Skip mean/std buffers (they are persistent=False, auto-created)
            if new_key in ["mean", "std", "_mean", "_std"]:
                continue
            new_state_dict[new_key] = v
        state_dict = new_state_dict

        self.model.load_state_dict(state_dict, strict=True)

    @Worker.timer("compute_rewards")
    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.model.to(self.device)

        while True:
            data = input_channel.get()
            images = data.get("images")
            last_run = data.get("last_run", None)

            rewards = self._compute_image_rewards(images=images)
            self.send_reward_output(output_channel, rewards)

            if last_run:
                if self.enable_offload:
                    self.model.to("cpu")
                break

    def _compute_image_rewards(self, images: torch.Tensor):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        images = images.to(self.device)

        with torch.no_grad():
            outputs = self.model(images)
            probs = outputs["probabilities"]
            rewards = torch.where(
                probs > self.reward_threshold,
                probs,
                torch.zeros_like(probs),
            )

        return rewards

    def run_inference_loop(self, input_channel: Channel, output_channel: Channel):
        while True:
            data = input_channel.get()
            images = data.get("images")

            images = images.to(self.device)

            with torch.no_grad():
                rewards = self.model.compute_reward(images)

            if rewards.dim() == 1:
                rewards = rewards.unsqueeze(1)

            self.send_reward_output(output_channel, rewards)

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env peer ranks for this reward worker.

        This mapping supports both one-to-many and many-to-one env/reward layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this reward worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        reward_world_size = self.placement.get_world_size("reward")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=reward_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        reward_world_size = self.placement.get_world_size("reward")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=reward_world_size,
            dst_rank=self._rank,
        )
    
    def send_reward_output(
        self,
        output_channel: Channel,
        reward_tensor: torch.Tensor | np.ndarray,
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            reward_tensor: Predicted rewards (tensor or ndarray).
        """

        dst_ranks_and_sizes = self.dst_ranks["train"]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        reward_tensor_split = list(torch.split(reward_tensor, split_sizes, dim=0))
        for (dst_rank, _), reward_i in zip(
            dst_ranks_and_sizes, reward_tensor_split
        ):
            if isinstance(reward_i, torch.Tensor):
                reward_i = reward_i.cpu().contiguous()
            output_channel.put(
                reward_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"reward_output"
                ),
                async_op=True,
            )