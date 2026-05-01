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

import asyncio
import os
from typing import Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from rlinf.config import torch_dtype_from_precision
from rlinf.data.datasets.reward_model import RewardBinaryDataset
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.scheduler import (
    Channel,
    Cluster,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    Worker,
)
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.down_sampling import down_sample_batch
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import clear_memory


class RewardWorker(Worker):
    """Reward Worker for inference during reasoning and agentic RL training."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg

        self.placement = HybridComponentPlacement(cfg, Cluster())

    def init_worker(self):
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )
        self.do_down_sampling = self.cfg.algorithm.get("down_sampling", {}).get(
            "do_down_sampling", False
        )
        if self.do_down_sampling:
            self.down_sampling_config = self.cfg.algorithm.get("down_sampling", {}).get(
                "down_sampling_config", {}
            )

        if self.cfg.reward.use_reward_model:
            raise NotImplementedError
        else:
            from rlinf.algorithms.rewards import get_rule_based_reward_class

            self.rule_based_reward = get_rule_based_reward_class(
                self.cfg.reward.reward_type
            )(self.cfg.reward)

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
                rollout_result = down_sample_batch(
                    rollout_result, self.down_sampling_config
                )
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
        scores = self.rule_based_reward.get_reward(
            texts, rollout_result.answers, **kwargs
        )
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )


class EmbodiedRewardWorker(Worker):
    """Reward Worker for inference during embodied RL training."""

    @staticmethod
    def launch_for_realworld(
        reward_cfg: dict,
        node_rank: int,
        node_group_label: Optional[str] = None,
        hardware_rank: Optional[int] = None,
        env_idx: int = 0,
        worker_rank: int = 0,
    ):
        """Launch a single-rank reward worker for real-world env inference."""
        cluster = Cluster()
        if hardware_rank is not None:
            placement = FlexiblePlacementStrategy(
                [[hardware_rank]],
                node_group_label=node_group_label,
            )
        elif node_group_label is not None:
            node_group = cluster.get_node_group(node_group_label)
            assert node_group is not None, (
                f"Node group {node_group_label} not found in cluster."
            )
            assert node_rank in node_group.node_ranks, (
                f"Node rank {node_rank} is not in node group {node_group_label}: "
                f"{node_group.node_ranks}"
            )
            placement_node_rank = node_group.node_ranks.index(node_rank)
            placement = NodePlacementStrategy(
                [placement_node_rank], node_group_label=node_group_label
            )
        else:
            placement = NodePlacementStrategy([node_rank])

        standalone_reward_cfg = dict(reward_cfg)
        standalone_reward_cfg["standalone_realworld"] = True
        standalone_cfg = OmegaConf.create({"reward": standalone_reward_cfg})
        return EmbodiedRewardWorker.create_group(standalone_cfg).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"EmbodiedRewardWorker-{worker_rank}-{env_idx}",
        )

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        self.cfg = cfg

        self._standalone_realworld = self.cfg.reward.get("standalone_realworld", False)
        self.placement = (
            HybridComponentPlacement(cfg, Cluster())
            if not self._standalone_realworld
            else None
        )

        # Device setup
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        if not self._standalone_realworld:
            self.total_num_train_envs = cfg.env.train.total_num_envs
            self.total_num_eval_envs = cfg.env.eval.total_num_envs
            self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        else:
            self.total_num_train_envs = 1
            self.total_num_eval_envs = 1
            self.num_pipeline_stages = 1

        self.enable_offload = self.cfg.reward.get("enable_offload", False)
        self._interact_task = None

        self.reward_threshold = self.cfg.reward.get("reward_threshold", 0.6)
        self._use_reward_prob = self.cfg.reward.get("use_reward_prob", False)

    def model_provider_func(self):
        from rlinf.models.embodiment.reward import get_reward_model_class

        reward_cls = get_reward_model_class(self.cfg.reward.model.model_type)

        model_cfg = self.cfg.reward.model
        torch_dtype = torch_dtype_from_precision(model_cfg.precision)

        model = reward_cls(model_cfg)

        model.to(torch_dtype)

        return model

    def init_worker(self):
        """Initialize the reward worker for inference."""
        # build model
        self.model = self.model_provider_func()

        # Move to device and set eval mode
        self.model = self.model.to(self.device)
        self.model.eval()

        if self._standalone_realworld:
            return

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

    async def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.model.to(self.device)

        local_num_train_envs = sum(size for _, size in self.src_ranks["train"])
        total_last_run_count = 0
        while True:
            merged_images, last_run_count = await self.recv_merged_reward_input(
                input_channel, mode="train"
            )
            rewards = self._compute_image_rewards(images=merged_images)
            self.send_reward_output(output_channel, rewards)
            total_last_run_count += last_run_count
            if total_last_run_count >= local_num_train_envs:
                break

        if self.enable_offload:
            self.model.to("cpu")

    async def recv_merged_reward_input(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor | np.ndarray, int]:
        """Receive all mapped reward inputs, merge images on batch dim."""
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        image_batches: list[torch.Tensor | np.ndarray] = []
        last_run_count = 0
        for src_rank, expected_size in src_ranks_and_sizes:
            data = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_reward_input"
                ),
                async_op=True,
            ).async_wait()
            images = data.get("images")
            actual_size = self._infer_reward_batch_size(images)
            assert actual_size == expected_size, (
                f"Expected reward input batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            image_batches.append(images)
            last_run = data.get("last_run", None)
            last_run_count += int(last_run.sum().item()) if last_run is not None else 0

        merged_images = self._merge_image_batches(image_batches)
        return merged_images, last_run_count

    @staticmethod
    def _merge_image_batches(
        image_batches: list[torch.Tensor | np.ndarray],
    ) -> torch.Tensor | np.ndarray:
        if len(image_batches) == 0:
            raise ValueError("No image batches received for reward inference.")
        if all(isinstance(images, torch.Tensor) for images in image_batches):
            return torch.cat(image_batches, dim=0)
        if all(isinstance(images, np.ndarray) for images in image_batches):
            return np.concatenate(image_batches, axis=0)
        # Fallback for mixed types: cast ndarray to tensor and merge as torch.Tensor.
        tensor_batches = [
            images if isinstance(images, torch.Tensor) else torch.from_numpy(images)
            for images in image_batches
        ]
        return torch.cat(tensor_batches, dim=0)

    @staticmethod
    def _infer_reward_batch_size(images: torch.Tensor | np.ndarray) -> int:
        if isinstance(images, torch.Tensor) or isinstance(images, np.ndarray):
            return images.shape[0]
        raise ValueError(f"Unsupported reward input image type: {type(images)}")

    @Worker.timer("compute_image_rewards")
    def _compute_image_rewards(self, images: torch.Tensor):
        if isinstance(images, np.ndarray):
            images = torch.from_numpy(images)

        model_dtype = next(self.model.parameters()).dtype
        images = images.to(device=self.device, dtype=model_dtype)

        with torch.no_grad():
            outputs = self.model(images)
            probs = outputs["probabilities"]
            if self._use_reward_prob:
                self.log_info(
                    f"[reward_model/probs] shape={probs.shape} values={probs.cpu().tolist()}"
                )
            rewards = (probs > self.reward_threshold).to(probs.dtype)

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)

        return rewards

    def compute_image_rewards(
        self, images: torch.Tensor | np.ndarray
    ) -> torch.Tensor | np.ndarray:
        """Run one-shot reward inference and return CPU results."""
        rewards = self._compute_image_rewards(images)
        if isinstance(rewards, torch.Tensor):
            return rewards.detach().cpu()
        return rewards

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
        for (dst_rank, _), reward_i in zip(dst_ranks_and_sizes, reward_tensor_split):
            if isinstance(reward_i, torch.Tensor):
                reward_i = reward_i.cpu().contiguous()
            output_channel.put(
                reward_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra="reward_output"
                ),
                async_op=True,
            )

    async def compute_rewards_async(
        self, input_channel: Channel, output_channel: Channel
    ):
        assert self._interact_task is None or self._interact_task.done(), (
            "Previous interact task is still running while a new interact call is made."
        )
        self._interact_task = asyncio.create_task(
            self._compute_rewards(input_channel, output_channel)
        )
        try:
            await self._interact_task
        except asyncio.CancelledError:
            pass

    async def _compute_rewards(self, input_channel: Channel, output_channel: Channel):
        while True:
            merged_images, _ = await self.recv_merged_reward_input(
                input_channel, mode="train"
            )
            rewards = self._compute_image_rewards(images=merged_images)
            self.send_reward_output(output_channel, rewards)

    async def stop(self):
        if self._interact_task is not None and not self._interact_task.done():
            self._interact_task.cancel()


class FSDPRewardWorker(FSDPModelManager, Worker):
    """FSDP-based worker for reward model training."""

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg

        self.data_loader, self.val_loader = self.build_dataloader()

        # Training step counter for validation interval
        self._training_step = 0

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

    def model_provider_func(self):
        from rlinf.models.embodiment.reward import get_reward_model_class

        reward_cls = get_reward_model_class(self.cfg.actor.model.model_type)

        model_cfg = self.cfg.actor.model
        torch_dtype = torch_dtype_from_precision(model_cfg.precision)

        model = reward_cls(model_cfg)

        model.to(torch_dtype)

        return model

    def init_worker(self):
        """Initialize model and optimizer using base class."""

        if self.data_loader is None:
            raise ValueError("data_loader is not set")
        self.data_iter = iter(self.data_loader)

        self.setup_model_and_optimizer()

        self.logger.info(
            f"Initialized FSDPRewardWorker with "
            f"{sum(p.numel() for p in self.model.parameters())} parameters"
        )

    def build_dataloader(self) -> tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Build dataloaders from preprocessed train/val dataset files."""
        data_cfg = self.cfg.get("data", {})
        train_data_paths = data_cfg.get("train_data_paths")
        val_data_paths = data_cfg.get("val_data_paths")

        self.logger.info(
            f"Loading preprocessed reward datasets from "
            f"{train_data_paths} and {val_data_paths}"
        )
        train_dataset = RewardBinaryDataset(train_data_paths)
        val_dataset = RewardBinaryDataset(val_data_paths)

        if len(train_dataset) == 0:
            self.logger.warning("Training dataset is empty")
            return None, None

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

    @Worker.timer("run_training")
    def run_training(self) -> dict[str, float]:
        """Run one training iteration with gradient accumulation."""
        self.model.train()

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

        return train_metrics

    @Worker.timer("run_eval")
    def run_eval(self) -> dict[str, float]:
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

    def get_max_steps_per_epoch(self):
        if self.data_loader is not None:
            return max(1, len(self.data_loader) // self.gradient_accumulation)
        return 0
