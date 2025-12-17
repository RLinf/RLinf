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


import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.datasets.reward import RewardDataset
from rlinf.data.io_struct import EmbodiedRolloutResult, RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.models.reward_model.reward_classifier import BinaryRewardClassifier
from rlinf.scheduler import Channel, Worker
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger


class RewardWorker(Worker):
    def __init__(self, cfg: DictConfig, placement=None):
        Worker.__init__(self)
        self.cfg = cfg

        # Check if this is training mode (has reward.training_backend config)
        self.is_training_mode = (
            hasattr(cfg, "reward")
            and hasattr(cfg.reward, "training_backend")
            and cfg.reward.training_backend is not None
        )

        if self.is_training_mode:
            # Training mode: regular PyTorch training
            if placement is None:
                raise ValueError("placement is required for training mode")
            self.component_placement = placement
            self.placement = placement
            # Initialize dataset for training
            import sys

            print("Initializing dataset...", flush=True)
            sys.stdout.flush()
            self.dataset = RewardDataset(cfg.reward.data, device="cpu")
            print(f"Dataset initialized with {len(self.dataset)} samples", flush=True)
            sys.stdout.flush()

            # Initialize dataloader (simple mode, single GPU)
            # Use default collate_fn like train_reward_model.py
            # Default collate_fn handles tuple returns (images_dict, label_tensor) correctly
            self.dataloader = DataLoader(
                dataset=self.dataset,
                batch_size=cfg.reward.global_batch_size,
                shuffle=getattr(cfg.reward.data, "shuffle", True),
                num_workers=getattr(cfg.reward.data, "num_workers", 0),
                pin_memory=getattr(cfg.reward.data, "pin_memory", False),
                persistent_workers=getattr(
                    cfg.reward.data, "persistent_workers", False
                ),
                prefetch_factor=getattr(cfg.reward.data, "prefetch_factor", 2),
            )

            # MetricLogger expects the full cfg object, not just log_dir
            self.metric_logger = MetricLogger(cfg)

            # Training state
            self.global_step = 0
            self.epoch = 0
            self.timer = ScopedTimer(reduction="max", sync_cuda=False)

            self.tokenizer = None
            self.reward = None
            self.reward_model = None
        else:
            # Inference mode: original logic
            if placement is None:
                raise ValueError("placement is required for inference mode")
            self.component_placement = placement

            # Determine if this is an embodied task
            self.is_embodied_task = self._is_embodied_task()

            if self.is_embodied_task:
                # For embodied tasks, batch size is based on env config
                if hasattr(cfg, "env") and hasattr(cfg.env, "train"):
                    env_batch_size = cfg.env.train.num_group * cfg.env.train.group_size
                elif hasattr(cfg, "algorithm") and "num_group_envs" in cfg.algorithm:
                    env_batch_size = cfg.algorithm.num_group_envs
                else:
                    env_batch_size = cfg.get("data", {}).get("rollout_batch_size", 256)
                    if env_batch_size == 256:
                        print(
                            f"Warning: Could not find batch size config for embodied task, using default {env_batch_size}"
                        )

                self.total_batch_size_per_dp = (
                    env_batch_size
                    * cfg.algorithm.get("group_size", 1)
                    // self._world_size
                )
                self.tokenizer = None  # Not needed for embodied tasks
                self.reward = None  # Will be set in init_worker if needed
                self.reward_model = None
            else:
                # For text-based reasoning tasks
                self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
                self.total_batch_size_per_dp = (
                    self.cfg.data.rollout_batch_size
                    * self.cfg.algorithm.get("group_size", 1)
                    // self._world_size
                )
                self.reward = None  # Will be set in init_worker if needed
                self.reward_model = None

    def _is_embodied_task(self) -> bool:
        """Check if this is an embodied task based on config."""
        # Check for runner type
        if hasattr(self.cfg, "runner") and hasattr(self.cfg.runner, "task_type"):
            return self.cfg.runner.task_type == "embodied"

        # Check for env config (embodied tasks have env config)
        if hasattr(self.cfg, "env"):
            return True

        # Check if reward config specifies embodied reward model
        if hasattr(self.cfg, "reward"):
            reward_cfg = self.cfg.reward
            if reward_cfg.get("use_reward_model", False):
                reward_model_cfg = reward_cfg.get("reward_model", {})
                # If reward_model config has image_keys or image_size, it's likely embodied
                if "image_keys" in reward_model_cfg or "image_size" in reward_model_cfg:
                    return True

        return False

    def init_worker(self):
        if self.is_training_mode:
            # Training mode: setup model and optimizer
            import sys

            print("Initializing model and optimizer...", flush=True)
            sys.stdout.flush()

            # Simple mode: create model and optimizer directly
            # Get GPU device from config, default to 0
            gpu_id = getattr(self.cfg.reward, "gpu_id", 0)
            if torch.cuda.is_available() and gpu_id is not None:
                self.device = torch.device(f"cuda:{gpu_id}")
                torch.cuda.set_device(gpu_id)
            else:
                self.device = torch.device("cpu")

            self.model = BinaryRewardClassifier(
                image_keys=self.cfg.reward.model.image_keys,
                image_size=self.cfg.reward.model.image_size,
                hidden_dim=self.cfg.reward.model.hidden_dim,
                num_spatial_blocks=self.cfg.reward.model.num_spatial_blocks,
                pretrained_encoder_path=self.cfg.reward.model.get(
                    "pretrained_encoder_path", None
                ),
                use_pretrain=self.cfg.reward.model.get("use_pretrain", True),
                freeze_encoder=self.cfg.reward.model.get("freeze_encoder", True),
            )
            self.model = self.model.to(self.device)

            # Create optimizer
            optim_cfg = self.cfg.reward.optim
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=optim_cfg.lr,
                betas=(optim_cfg.adam_beta1, optim_cfg.adam_beta2),
                eps=optim_cfg.adam_eps,
                weight_decay=optim_cfg.weight_decay,
            )

            # Create learning rate scheduler if needed
            lr_sched_cfg = self.cfg.reward.get("lr_sched", {})
            if lr_sched_cfg.get("lr_decay_style") != "constant":
                from torch.optim.lr_scheduler import LambdaLR

                num_warmup_steps = int(lr_sched_cfg.get("lr_warmup_iters", 0))
                min_lr = lr_sched_cfg.get("min_lr", 0.0)
                max_lr = lr_sched_cfg.get("max_lr", optim_cfg.lr)

                def lr_lambda(current_step):
                    if current_step < num_warmup_steps:
                        return float(current_step) / float(max(1, num_warmup_steps))
                    else:
                        return max(min_lr / max_lr, 0.0)

                self.lr_scheduler = LambdaLR(self.optimizer, lr_lambda)
            else:
                self.lr_scheduler = None

            import sys

            print("Model and optimizer initialized", flush=True)
            sys.stdout.flush()
            return

        # Inference mode: original logic
        # Set device first (needed before using self.device)
        self.device = torch.cuda.current_device()

        if self.is_embodied_task:
            # Embodied task initialization
            if self.cfg.reward.use_reward_model:
                # Get model configuration
                reward_cfg = self.cfg.reward.reward_model
                image_keys = reward_cfg.get(
                    "image_keys", self.cfg.actor.model.image_keys
                )
                image_size = reward_cfg.get(
                    "image_size", self.cfg.actor.model.image_size
                )
                hidden_dim = reward_cfg.get("hidden_dim", 256)
                num_spatial_blocks = reward_cfg.get("num_spatial_blocks", 8)
                pretrained_encoder_path = reward_cfg.get(
                    "pretrained_encoder_path", None
                )
                use_pretrain = reward_cfg.get("use_pretrain", True)
                freeze_encoder = reward_cfg.get("freeze_encoder", True)

                # Load checkpoint if specified and check configuration compatibility
                checkpoint_path = reward_cfg.get("checkpoint_path", None)
                checkpoint_state = None
                if checkpoint_path and os.path.exists(checkpoint_path):
                    checkpoint = torch.load(
                        checkpoint_path, map_location="cpu", weights_only=False
                    )
                    checkpoint_state = (
                        checkpoint.get("model_state_dict")
                        or checkpoint.get("state_dict")
                        or checkpoint
                    )

                    # Try to infer use_pretrain from checkpoint if possible
                    pooling_key = None
                    for key in checkpoint_state.keys():
                        if "pooling_layer.kernel" in key:
                            pooling_key = key
                            break

                    if pooling_key is not None:
                        kernel_shape = checkpoint_state[pooling_key].shape
                        if kernel_shape[1] == 1 and kernel_shape[2] == 1:
                            inferred_use_pretrain = False
                            print(
                                f"Inferred use_pretrain=False from checkpoint (pooling kernel shape: {kernel_shape})"
                            )
                        else:
                            inferred_use_pretrain = True
                            print(
                                f"Inferred use_pretrain=True from checkpoint (pooling kernel shape: {kernel_shape})"
                            )

                        if inferred_use_pretrain != use_pretrain:
                            print(
                                f"Warning: use_pretrain mismatch! Config says {use_pretrain}, but checkpoint suggests {inferred_use_pretrain}. Using inferred value."
                            )
                            use_pretrain = inferred_use_pretrain

                # Create reward model
                self.reward_model = BinaryRewardClassifier(
                    image_keys=image_keys,
                    image_size=image_size,
                    hidden_dim=hidden_dim,
                    num_spatial_blocks=num_spatial_blocks,
                    pretrained_encoder_path=pretrained_encoder_path,
                    use_pretrain=use_pretrain,
                    freeze_encoder=freeze_encoder,
                )

                # Load checkpoint state dict if available
                if checkpoint_state is not None:
                    self.reward_model.load_state_dict(checkpoint_state, strict=True)
                    print(f"Loaded reward model from {checkpoint_path}")
                elif checkpoint_path:
                    print(
                        f"Warning: Reward model checkpoint not found at {checkpoint_path}"
                    )

                # Move to device
                self.reward_model = self.reward_model.to(self.device)
                self.reward_model.eval()
            else:
                raise NotImplementedError(
                    "Only reward model is supported for embodiment tasks. Set reward.use_reward_model=True"
                )
        else:
            # Text-based reasoning task initialization
            if self.cfg.reward.use_reward_model:
                raise NotImplementedError(
                    "Reward model for text-based tasks is not implemented yet."
                )
            else:
                self.reward = get_reward_class(self.cfg.reward.reward_type)(
                    self.cfg.reward
                )

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()
        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        This method supports both text-based reasoning tasks (RolloutResult)
        and embodied tasks (EmbodiedRolloutResult).

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result = input_channel.get()

            # Determine result type and count batch size accordingly
            if isinstance(rollout_result, EmbodiedRolloutResult):
                recv_batch_size += len(rollout_result.forward_inputs)
            elif isinstance(rollout_result, RolloutResult):
                recv_batch_size += rollout_result.num_sequence
            else:
                raise ValueError(
                    f"Unexpected rollout result type: {type(rollout_result)}"
                )

            with self.worker_timer():
                if isinstance(rollout_result, EmbodiedRolloutResult):
                    # Handle embodied task rewards
                    if (
                        rollout_result.rewards is None
                        or len(rollout_result.rewards) == 0
                    ):
                        if self.cfg.reward.use_reward_model:
                            with self.device_lock:
                                rewards = self._compute_embodied_rewards_with_model(
                                    rollout_result
                                )
                                rollout_result.rewards = rewards
                        else:
                            raise NotImplementedError(
                                "Rule-based rewards not implemented for embodiment tasks. "
                                "Set reward.use_reward_model=True"
                            )
                elif isinstance(rollout_result, RolloutResult):
                    # Handle text-based reasoning task rewards
                    if rollout_result.rewards is None:
                        if self.tokenizer is None:
                            raise ValueError(
                                "Tokenizer is required for text-based reasoning tasks. "
                                "This should not happen if _is_embodied_task() is working correctly."
                            )

                        # use_reward_model=True for text tasks would have raised in init_worker
                        rollout_result.rewards = self._compute_rule_based_rewards(
                            rollout_result
                        )

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
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
        scores = self.reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )

    def _compute_reward_from_images(
        self, images: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute reward from images using the reward model.

        Args:
            images: Dictionary of image tensors.

        Returns:
            Reward tensor.
        """
        with torch.no_grad():
            logits = self.reward_model(images, train=False)
            probs = torch.sigmoid(logits)
            reward_type = self.cfg.reward.reward_model.get("reward_type", "binary")
            if reward_type == "binary":
                rewards = (probs > 0.5).float()
            else:  # continuous
                rewards = probs

            # Handle batch dimension
            if rewards.dim() == 2 and rewards.shape[1] == 1:
                rewards = rewards.squeeze(1)
            return rewards.cpu()

    def _compute_embodied_rewards_with_model(
        self, rollout_result: EmbodiedRolloutResult
    ) -> list:
        """Compute rewards using the reward model for frame-based classification.

        Args:
            rollout_result: The rollout result containing observations.

        Returns:
            List of reward tensors, one per transition/step.
        """
        all_rewards = []

        # Process forward_inputs (current observations)
        for forward_input in rollout_result.forward_inputs:
            obs = self._extract_obs_from_forward_inputs(forward_input)
            images = self._extract_images_from_obs(obs)
            rewards = self._compute_reward_from_images(images)
            all_rewards.append(rewards)

        # Concatenate all rewards
        if len(all_rewards) > 0:
            all_rewards_tensor = torch.cat(all_rewards, dim=0)
            # Convert to list of tensors (one per step)
            return [all_rewards_tensor[i] for i in range(len(all_rewards_tensor))]
        else:
            return []

    def _extract_obs_from_forward_inputs(self, forward_input: dict) -> dict:
        """Extract observation dictionary from forward_inputs.

        Args:
            forward_input: Dictionary with keys like "obs/images/base_camera", "obs/states", etc.

        Returns:
            Dictionary with keys like "images/base_camera", "states", etc.
        """
        obs = {}
        for key, value in forward_input.items():
            if key.startswith("obs/"):
                new_key = key[len("obs/") :]
                obs[new_key] = value
        return obs

    def _extract_images_from_obs(self, obs: dict) -> dict[str, torch.Tensor]:
        """Extract and format images from observation dictionary.

        Args:
            obs: Observation dictionary with image keys.

        Returns:
            Dictionary of image tensors in [B, C, H, W] format.
        """
        images = {}
        image_keys = self.cfg.reward.reward_model.get(
            "image_keys", self.cfg.actor.model.image_keys
        )

        for key in image_keys:
            # Try different possible keys
            possible_keys = [
                f"images/{key}",
                key,
                f"obs/images/{key}",
            ]

            image_tensor = None
            for possible_key in possible_keys:
                if possible_key in obs:
                    image_tensor = obs[possible_key]
                    break

            if image_tensor is None:
                raise ValueError(
                    f"Image key {key} not found in observation. "
                    f"Available keys: {list(obs.keys())}"
                )

            # Ensure tensor format [B, C, H, W]
            if isinstance(image_tensor, torch.Tensor):
                if image_tensor.dim() == 3:
                    # Add batch dimension
                    image_tensor = image_tensor.unsqueeze(0)
                elif image_tensor.dim() == 4:
                    pass  # Already in correct format
                else:
                    raise ValueError(
                        f"Unexpected image tensor shape: {image_tensor.shape}"
                    )

                # Move to device if needed
                if image_tensor.device != self.device:
                    image_tensor = image_tensor.to(self.device)

                images[key] = image_tensor
            else:
                raise ValueError(
                    f"Image {key} is not a torch.Tensor: {type(image_tensor)}"
                )

        return images

    def fit(self):
        """Main training loop for reward model"""
        if not self.is_training_mode:
            raise RuntimeError("fit() can only be called in training mode")

        print(f"\n{'=' * 60}", flush=True)
        print("Starting training...", flush=True)
        print(f"Total epochs: {getattr(self.cfg.reward, 'num_epochs', 1)}", flush=True)
        print(f"Global batch size: {self.cfg.reward.global_batch_size}", flush=True)
        print(f"Device: {self.device}", flush=True)
        print(f"{'=' * 60}\n", flush=True)
        sys.stdout.flush()

        num_epochs = getattr(self.cfg.reward, "num_epochs", 1)

        for epoch in range(num_epochs):
            self.epoch = epoch

            for step, batch in enumerate(self.dataloader):
                m_batch = self._prepare_batch(batch)
                images_dict, labels = m_batch
                labels = labels.unsqueeze(1)  # [B] -> [B, 1]

                # Forward pass
                logits = self.model(images_dict, train=True)
                loss = F.binary_cross_entropy_with_logits(logits, labels)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Gradient clipping if configured
                if (
                    hasattr(self.cfg.reward.optim, "clip_grad")
                    and self.cfg.reward.optim.clip_grad > 0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.cfg.reward.optim.clip_grad
                    )

                # Optimizer step
                self.optimizer.step()
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Compute accuracy
                probs = torch.sigmoid(logits.detach())
                preds = (probs > 0.5).float()
                correct = (preds == labels).sum().item()
                total = labels.size(0)
                accuracy = correct / total if total > 0 else 0.0

                # Compute grad norm for logging
                total_norm = 0.0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norm = total_norm ** (1.0 / 2)
                lr_list = [group["lr"] for group in self.optimizer.param_groups]

                # Aggregate metrics
                mean_metric_dict = {
                    "loss": loss.detach().item(),
                    "accuracy": accuracy,
                    "grad_norm": float(grad_norm),
                    "lr": lr_list[0],
                }

                self.global_step += 1

                # Logging
                if (
                    self.global_step % getattr(self.cfg.reward, "log_interval", 100)
                    == 0
                ):
                    print(
                        f"Epoch {epoch}, Step {step}, Loss: {mean_metric_dict['loss']:.4f}, Acc: {mean_metric_dict['accuracy']:.4f}",
                        flush=True,
                    )
                    sys.stdout.flush()

                # Save checkpoint
                if (
                    self.global_step % getattr(self.cfg.reward, "save_interval", 1000)
                    == 0
                ):
                    save_base_path = getattr(
                        self.cfg.reward, "save_dir", "./checkpoints"
                    )
                    save_path = f"{save_base_path}/step_{self.global_step}"

                    os.makedirs(save_path, exist_ok=True)
                    # Get model state dict
                    checkpoint = {
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "loss": mean_metric_dict["loss"],
                        "accuracy": mean_metric_dict["accuracy"],
                    }
                    if self.lr_scheduler is not None:
                        checkpoint["lr_scheduler_state_dict"] = (
                            self.lr_scheduler.state_dict()
                        )
                    torch.save(checkpoint, f"{save_path}/checkpoint.pt")
                    print(f"Saved checkpoint to {save_path}/checkpoint.pt", flush=True)

                time_metrics = self.timer.consume_durations()
                if self.metric_logger is not None:
                    self.metric_logger.log(time_metrics, self.global_step)
                    self.metric_logger.log(mean_metric_dict, self.global_step)

    def _prepare_batch(self, batch):
        """Prepare batch for training"""
        # Default collate_fn behavior for tuple returns (images_dict, label_tensor):
        # - Collects all first elements (images_dict) and merges them into a dict
        # - Collects all second elements (label_tensor) and stacks them into a tensor
        # So batch can be either:
        #   1. tuple: (merged_images_dict, labels_tensor) where:
        #      - merged_images_dict[key] = tensor (already stacked) or list of tensors
        #      - labels_tensor = tensor (already stacked) or list of tensors
        #   2. list: [images_dict, labels_tensor] (same as tuple but as list)

        # Handle both tuple and list formats
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            images_part, labels_part = batch

            # Process images_dict
            if isinstance(images_part, dict):
                images_dict = {}
                for key, value in images_part.items():
                    if isinstance(value, list):
                        # List of tensors from default collate_fn, stack them
                        images_dict[key] = torch.stack(value, dim=0)
                    elif isinstance(value, torch.Tensor):
                        # Already stacked tensor (default collate_fn may stack dict values)
                        images_dict[key] = value
                    else:
                        raise ValueError(
                            f"Unexpected value type for key {key}: {type(value)}"
                        )
            else:
                raise ValueError(
                    f"Expected images_part to be dict, got {type(images_part)}"
                )

            # Process labels
            if isinstance(labels_part, list):
                if len(labels_part) > 0 and isinstance(labels_part[0], torch.Tensor):
                    labels = torch.stack(labels_part, dim=0)
                else:
                    labels = torch.tensor(labels_part, dtype=torch.float32)
            elif isinstance(labels_part, torch.Tensor):
                # Already stacked tensor
                labels = labels_part
            else:
                raise ValueError(
                    f"Expected labels_part to be list or tensor, got {type(labels_part)}"
                )
        else:
            raise ValueError(
                f"Unexpected batch format: {type(batch)}, expected tuple or list of length 2. "
                f"Batch type: {type(batch)}, length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}"
            )

        # Move to device
        # images_dict values should already be stacked tensors [B, C, H, W]
        images_dict = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in images_dict.items()
        }
        labels = labels.to(self.device) if isinstance(labels, torch.Tensor) else labels

        return images_dict, labels
