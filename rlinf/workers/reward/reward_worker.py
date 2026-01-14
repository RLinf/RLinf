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

"""Reward Worker with FSDP support for distributed training and inference."""

import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import HybridComponentPlacement, ModelParallelComponentPlacement
from rlinf.utils.utils import clear_memory

logger = logging.getLogger(__name__)


class RewardWorker(FSDPModelManager, Worker):
    """Reward Worker with FSDP support for distributed training and inference.
    
    This worker supports two modes:
    1. Rule-based rewards: Uses predefined reward functions (e.g., math, code)
    2. Model-based rewards: Uses trained reward models (e.g., ResNet, VLM)
    
    For model-based rewards, it supports:
    - FSDP distributed training
    - Distributed data loading with DistributedSampler
    - Mixed precision training
    - Gradient accumulation
    """

    def __init__(
        self, 
        cfg: DictConfig, 
        placement: Optional[ModelParallelComponentPlacement] = None
    ):
        """Initialize the RewardWorker.
        
        Args:
            cfg: Configuration containing reward settings.
            placement: Optional placement configuration for model parallelism.
        """
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        
        # Check if using reward model (embodied RL) or rule-based reward (LLM RL)
        self.use_reward_model = cfg.reward.get("use_reward_model", False)
        
        if self.use_reward_model:
            # Initialize FSDP for model-based rewards
            # Use reward config for FSDP setup
            reward_fsdp_cfg = self._build_fsdp_config()
            super().__init__(reward_fsdp_cfg, self._world_size, self._rank)
            
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
            self.device = torch.cuda.current_device()
            
            self._component_placement = HybridComponentPlacement(cfg, Cluster())
        else:
            # Rule-based rewards don't need FSDP
            pass
        
        # Tokenizer is only needed for LLM-based rewards, not embodied RL
        self.tokenizer = None
        if hasattr(cfg.reward, 'tokenizer') and cfg.reward.tokenizer is not None:
            self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
        
        # Batch size for LLM rewards (may not exist for embodied RL)
        if hasattr(cfg, 'data') and hasattr(cfg.data, 'rollout_batch_size'):
            self.total_batch_size_per_dp = (
                self.cfg.data.rollout_batch_size
                * self.cfg.algorithm.get("group_size", 1)
                // self._world_size
            )
        else:
            self.total_batch_size_per_dp = 1
        
        # Reward manager for model-based rewards
        self.reward_manager = None
        
        # Data loader for training
        self.data_loader = None
        self.data_iter = None
        
        # Training state
        self.gradient_accumulation = 1

    def _build_fsdp_config(self) -> DictConfig:
        """Build FSDP configuration from reward config.
        
        Returns:
            DictConfig suitable for FSDPModelManager initialization.
        """
        reward_cfg = self.cfg.reward
        
        # Build a config structure similar to actor config for FSDP
        fsdp_cfg = DictConfig({
            "model": {
                "model_type": reward_cfg.get("reward_model_type", "resnet"),
                "precision": reward_cfg.get("precision", "bf16"),
                "model_path": reward_cfg.get(reward_cfg.get("reward_model_type", "resnet"), {}).get("checkpoint_path", ""),
            },
            "fsdp_config": reward_cfg.get("fsdp_config", {
                "strategy": "fsdp",
                "sharding_strategy": "no_shard",
                "amp": {"enabled": True, "precision": "bf16"},
            }),
            "optim": reward_cfg.get("optim", {
                "lr": 1e-4,
                "weight_decay": 1e-5,
                "clip_grad": 1.0,
            }),
            "micro_batch_size": reward_cfg.get("micro_batch_size", 32),
            "global_batch_size": reward_cfg.get("global_batch_size", 64),
            "seed": self.cfg.get("seed", 1234),
        })
        
        return fsdp_cfg

    def init_worker(self):
        """Initialize the worker based on reward type."""
        if self.use_reward_model:
            self._init_model_based_reward()
        else:
            self._init_rule_based_reward()

    def _init_rule_based_reward(self):
        """Initialize rule-based reward function."""
        self.reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

    def _init_model_based_reward(self):
        """Initialize model-based reward with RewardManager."""
        from rlinf.algorithms.rewards.embodiment import RewardManager
        
        # Initialize RewardManager (creates the model)
        self.reward_manager = RewardManager(self.cfg.reward)
        self.reward_manager.to_device(self.device)
        
        logger.info(
            f"Initialized reward manager with model type: "
            f"{self.reward_manager.model_type}"
        )
        
        # Build data loader if training data path is configured
        training_cfg = self.cfg.get("reward_training", {})
        if training_cfg.get("enabled", False) and training_cfg.get("data_path"):
            self.data_loader = self.build_dataloader()
            self.data_iter = iter(self.data_loader)
            
            # Setup optimizer and scheduler for training
            self._setup_training()

    def _setup_training(self):
        """Setup optimizer and scheduler for reward model training."""
        if self.reward_manager is None or self.reward_manager.model is None:
            return
            
        training_cfg = self.cfg.get("reward_training", {})
        
        # Get model parameters
        model = self.reward_manager.model
        if hasattr(model, 'model'):
            params = model.model.parameters()
        else:
            params = model.parameters()
        
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            params,
            lr=training_cfg.get("lr", 1e-4),
            weight_decay=training_cfg.get("weight_decay", 1e-5),
        )
        
        # Create scheduler
        epochs = training_cfg.get("epochs", 100)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Gradient accumulation
        micro_batch = training_cfg.get("micro_batch_size", 32)
        global_batch = training_cfg.get("global_batch_size", 64)
        self.gradient_accumulation = max(1, global_batch // (micro_batch * self._world_size))

    def build_dataloader(self) -> DataLoader:
        """Build distributed data loader for reward model training.
        
        Returns:
            DataLoader with DistributedSampler for multi-GPU training.
        """
        from rlinf.algorithms.rewards.embodiment.reward_model_trainer import RewardDataset
        
        training_cfg = self.cfg.get("reward_training", {})
        data_path = training_cfg.get("data_path")
        
        if not data_path:
            raise ValueError("reward_training.data_path must be specified for training")
        
        # Get image size from reward config
        reward_cfg = self.cfg.reward
        model_type = reward_cfg.get("reward_model_type", "resnet")
        model_cfg = reward_cfg.get(model_type, {})
        image_size = model_cfg.get("image_size", [3, 224, 224])
        
        # Create dataset
        dataset = RewardDataset(
            data_path,
            image_size=(image_size[1], image_size[2]),
            augment=True,
        )
        
        # Create distributed sampler
        sampler = DistributedSampler(
            dataset,
            num_replicas=self._world_size,
            rank=self._rank,
            shuffle=True,
        )
        
        # Create data loader
        batch_size = training_cfg.get("micro_batch_size", 32)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=training_cfg.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
        )
        
        logger.info(
            f"Built data loader: {len(dataset)} samples, "
            f"batch_size={batch_size}, world_size={self._world_size}"
        )
        
        return data_loader

    def run_training(self):
        """Run one training step with FSDP support.
        
        This method follows the same pattern as FSDPSftWorker.run_training():
        - Gradient accumulation
        - Mixed precision training
        - Distributed gradient synchronization
        - Metrics aggregation
        
        Returns:
            Dictionary of training metrics.
        """
        if not self.use_reward_model or self.reward_manager is None:
            return {"loss": 0.0}
        
        if self.data_loader is None or self.data_iter is None:
            logger.warning("Data loader not initialized, skipping training")
            return {"loss": 0.0}
        
        with self.worker_timer():
            # Get model
            model = self.reward_manager.model
            if hasattr(model, 'model'):
                train_model = model.model
            else:
                train_model = model
            
            train_model.train()
            
            metrics = {}
            total_loss = 0.0
            correct = 0
            total = 0
            
            for idx in range(self.gradient_accumulation):
                try:
                    images, labels = next(self.data_iter)
                except StopIteration:
                    # Reset iterator at epoch end
                    if hasattr(self.data_loader.sampler, 'set_epoch'):
                        self.data_loader.sampler.set_epoch(
                            getattr(self, '_current_epoch', 0) + 1
                        )
                    self.data_iter = iter(self.data_loader)
                    images, labels = next(self.data_iter)
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                if hasattr(model, 'compute_reward'):
                    # Use RewardManager interface
                    outputs = model.compute_reward({"images": images})
                else:
                    outputs = train_model(images)
                
                # Ensure outputs are proper shape
                if outputs.dim() > 1:
                    outputs = outputs.squeeze(-1)
                
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation
                
                # Backward pass
                loss.backward()
                
                # Metrics
                total_loss += loss.item() * self.gradient_accumulation
                predictions = (outputs > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
            
            # Optimizer step
            if hasattr(self, 'optimizer'):
                torch.nn.utils.clip_grad_norm_(
                    train_model.parameters(),
                    self.cfg.get("reward_training", {}).get("clip_grad", 1.0)
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
            
            # Scheduler step
            if hasattr(self, 'lr_scheduler'):
                self.lr_scheduler.step()
            
            # Collect metrics
            lr_value = (
                self.optimizer.param_groups[0]["lr"]
                if hasattr(self, 'optimizer')
                else 0.0
            )
            
            append_to_dict(
                metrics,
                {
                    "loss": total_loss / self.gradient_accumulation,
                    "accuracy": correct / max(total, 1),
                    "learning_rate": lr_value,
                },
            )
            
            clear_memory()
            
            # Aggregate metrics across processes
            train_metrics = {key: np.mean(value) for key, value in metrics.items()}
            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )
            
            # Increment epoch counter
            self._current_epoch = getattr(self, '_current_epoch', 0) + 1
            
            return train_metrics

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        """Get a batch from the channel.
        
        Args:
            channel: Input channel to read from.
            
        Returns:
            Tuple of (batch dict, RolloutResult).
        """
        result: RolloutResult = channel.get()
        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards for rollout results.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        if self.use_reward_model:
            self._compute_model_rewards(input_channel, output_channel)
        else:
            self._compute_rule_rewards(input_channel, output_channel)

    def _compute_rule_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rule-based rewards (original logic for LLM RL)."""
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            with self.worker_timer():
                if rollout_result.rewards is None:
                        rollout_result.rewards = self._compute_rule_based_rewards(
                            rollout_result
                        )

            output_channel.put(rollout_result, async_op=True)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, "
            f"but got {recv_batch_size}"
        )

    def _compute_model_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute model-based rewards using RewardManager."""
        if self.reward_manager is None:
            raise RuntimeError("RewardManager not initialized for model-based rewards")
        
        # For embodied RL, process observations with reward model
        while True:
            try:
                data = input_channel.get(timeout=1.0)
            except Exception:
                break
                
            with self.worker_timer():
                if isinstance(data, dict) and "observations" in data:
                    # Compute rewards using reward manager
                    observations = data["observations"]
                    task_descriptions = data.get("task_descriptions")
                    
                    with self.device_lock:
                        rewards = self.compute_batch_rewards_with_model(
                            observations, task_descriptions
                        )
                    
                    data["rewards"] = rewards
                
            output_channel.put(data, async_op=True)

    def _compute_rule_based_rewards(self, rollout_result: RolloutResult):
        """Compute rule-based rewards for LLM RL tasks."""
        # Decode only the generated tokens
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
            
        scores = self.reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )

    def compute_batch_rewards(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
        mini_batch_size: int = 256,
    ) -> torch.Tensor:
        """Compute rewards using the reward model (direct call interface).
        
        Uses mini-batching to avoid OOM for large batches.
        
        Args:
            observations: Dictionary containing observation data (images, states).
            task_descriptions: Optional task descriptions for VLM models.
            mini_batch_size: Size of mini-batches for processing (default 256).
            
        Returns:
            Reward tensor of shape matching input batch.
        """
        if self.reward_manager is None:
            raise RuntimeError("RewardManager not initialized")
        
        images = observations.get("main_images") or observations.get("images")
        if images is None:
            raise ValueError("Observations must contain 'main_images' or 'images'")
        
        total_size = images.shape[0]
        
        # If batch is small enough, process in one go
        if total_size <= mini_batch_size:
            with self.worker_timer():
                with torch.no_grad():
                    return self.reward_manager.compute_rewards(observations, task_descriptions)
        
        # Mini-batch processing for large batches
        all_rewards = []
        with self.worker_timer():
            with torch.no_grad():
                for start_idx in range(0, total_size, mini_batch_size):
                    end_idx = min(start_idx + mini_batch_size, total_size)
                    mini_obs = {"main_images": images[start_idx:end_idx]}
                    mini_rewards = self.reward_manager.compute_rewards(mini_obs, task_descriptions)
                    all_rewards.append(mini_rewards)
        
        return torch.cat(all_rewards, dim=0)

    def compute_batch_rewards_with_model(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards using the reward model.
        
        Args:
            observations: Dictionary containing observation data (images, states).
            task_descriptions: Optional task descriptions for VLM models.
            
        Returns:
            Reward tensor of shape [B].
        """
        if self.reward_manager is None:
            raise RuntimeError("RewardManager not initialized")
        
        return self.reward_manager.compute_rewards(observations, task_descriptions)

    def save_checkpoint(self, save_path: str, step: int = 0):
        """Save reward model checkpoint.
        
        Args:
            save_path: Path to save the checkpoint.
            step: Current training step.
        """
        if self.reward_manager is None or self.reward_manager.model is None:
            logger.warning("No reward model to save")
            return
        
        os.makedirs(save_path, exist_ok=True)
        
        model = self.reward_manager.model
        if hasattr(model, 'model'):
            state_dict = model.model.state_dict()
        else:
            state_dict = model.state_dict()
        
        checkpoint = {
            "model_state_dict": state_dict,
            "step": step,
        }
        
        if hasattr(self, 'optimizer'):
            checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
        if hasattr(self, 'lr_scheduler'):
            checkpoint["scheduler_state_dict"] = self.lr_scheduler.state_dict()
        
        torch.save(checkpoint, os.path.join(save_path, f"reward_model_step_{step}.pt"))
        logger.info(f"Saved reward model checkpoint to {save_path}")

    def load_checkpoint(self, load_path: str):
        """Load reward model checkpoint.
        
        Args:
            load_path: Path to the checkpoint file.
        """
        if self.reward_manager is None or self.reward_manager.model is None:
            logger.warning("No reward model to load into")
            return
        
        checkpoint = torch.load(load_path, map_location=self.device)
        
        model = self.reward_manager.model
        if hasattr(model, 'model'):
            model.model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        
        if hasattr(self, 'optimizer') and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if hasattr(self, 'lr_scheduler') and "scheduler_state_dict" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        logger.info(f"Loaded reward model checkpoint from {load_path}")

    def set_global_step(self, global_step: int):
        """Set the global training step.
        
        Args:
            global_step: Current global step.
        """
        self._global_step = global_step
        if self.reward_manager is not None and hasattr(self.reward_manager.model, 'set_global_step'):
            self.reward_manager.model.set_global_step(global_step)
