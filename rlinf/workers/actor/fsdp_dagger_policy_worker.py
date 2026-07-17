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

import asyncio
import os
import threading
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.config import SupportedModel
from rlinf.data.datasets.dagger import (
    RollingLeRobotDataset,
    build_dataloader_from_dataset,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Worker
from rlinf.utils import drq
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict, compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedDAGGERFSDPPolicy(EmbodiedFSDPActor):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.replay_buffer = None
        self.update_step = 0
        self.enable_drq = bool(getattr(self.cfg.actor, "enable_drq", False))
        self.dataset = None
        self._lerobot_loader = None
        self._lerobot_iter = None
        self.enable_online_lerobot = bool(
            OmegaConf.select(
                cfg, "algorithm.dagger.online_lerobot.enabled", default=False
            )
        )
        self._data_epoch = 0
        # Actor-side LeRobot archive state for online DAgger training.
        self._next_lerobot_archive_id = 0
        self._pending_archive_path: str | None = None
        self._pending_archive_episodes: list[list[dict]] = []
        self._pending_archive_lock = threading.Lock()
        self._lerobot_loader_lock = threading.Lock()
        self._lerobot_resume_done = True
        self._lerobot_resume_thread: threading.Thread | None = None
        self._lerobot_resume_error: Exception | None = None

    def _online_lerobot_cfg(self) -> DictConfig:
        return OmegaConf.select(
            self.cfg, "algorithm.dagger.online_lerobot", default=OmegaConf.create({})
        )

    def _build_lerobot_dataset(self):
        online_lerobot_cfg = self._online_lerobot_cfg()
        lerobot_num_workers = online_lerobot_cfg.get("lerobot_num_workers", 0)
        self._lerobot_resume_num_workers = int(lerobot_num_workers)
        if self._lerobot_resume_num_workers < 0:
            raise ValueError(
                "algorithm.dagger.online_lerobot.lerobot_num_workers must be non-negative."
            )
        self.dataset = RollingLeRobotDataset(
            root_dir=self.cfg.algorithm.dagger.online_lerobot.data_path,
            chunk_size=self.cfg.actor.model.num_action_chunks,
            min_frames=online_lerobot_cfg.get("min_frames", 1),
            wait_interval_s=online_lerobot_cfg.get("wait_interval_s", 10.0),
            require_all_intervene=self.cfg.algorithm.dagger.get(
                "only_save_expert", False
            ),
            window_size=online_lerobot_cfg.get("rolling_lerobot_window_size", None),
            in_memory_mode=True,
            fps=int(
                OmegaConf.select(
                    self.cfg, "algorithm.dagger.online_lerobot.fps", default=10
                )
            ),
        )

    def _discover_lerobot_resume_shards(self) -> list[dict]:
        rank_dir = (
            Path(self.cfg.algorithm.dagger.online_lerobot.data_path)
            / f"rank_{self._rank}"
        )
        if not rank_dir.is_dir():
            self.log_info(
                f"No LeRobot resume directory found for actor rank {self._rank}: {rank_dir}"
            )
            return []

        valid_shards = []
        for child in rank_dir.iterdir():
            if not child.is_dir() or not child.name.startswith("id_"):
                continue
            try:
                shard_id = int(child.name.removeprefix("id_"))
            except ValueError:
                continue
            shard_info = self.dataset.archived_shard_info(child)
            if shard_info is None:
                self.log_info(f"Skipping invalid LeRobot resume shard: {child}")
                continue
            valid_shards.append(
                {
                    "id": shard_id,
                    "path": child,
                    "num_episodes": int(shard_info["num_episodes"]),
                    "num_frames": int(shard_info["num_frames"]),
                }
            )

        if not valid_shards:
            self.log_info(f"No valid LeRobot resume shards found under {rank_dir}")
            return []

        valid_shards.sort(key=lambda item: item["id"])
        latest_shard = valid_shards[-1]
        self._next_lerobot_archive_id = max(
            self._next_lerobot_archive_id,
            latest_shard["id"] + latest_shard["num_episodes"],
        )

        return self._select_lerobot_resume_shards(valid_shards)

    def _resume_lerobot_dataset(self) -> None:
        shards_to_load = self._discover_lerobot_resume_shards()
        if not shards_to_load:
            self._lerobot_resume_done = True
            return

        self._lerobot_resume_done = False
        self._lerobot_resume_error = None
        self._logger.info(
            "Starting background LeRobot resume: shards=%d, ids=%s, workers=%d",
            len(shards_to_load),
            [shard["id"] for shard in shards_to_load],
            self._lerobot_resume_num_workers,
        )

        self._lerobot_resume_thread = threading.Thread(
            target=self._resume_lerobot_dataset_background,
            args=(shards_to_load,),
            daemon=True,
            name=f"lerobot-resume-rank-{self._rank}",
        )
        self._lerobot_resume_thread.start()

    def _resume_lerobot_dataset_background(self, shards_to_load: list[dict]) -> None:
        try:
            staged_stores = self.dataset.load_archived_shards_staged(
                [shard["path"] for shard in shards_to_load],
                num_workers=self._lerobot_resume_num_workers,
            )
            total_episodes, total_frames = self.dataset.publish_staged_resume_shards(
                staged_stores
            )
            self._refresh_lerobot_loader_after_resume()
            self._lerobot_resume_done = True
        except Exception as exc:  # noqa: BLE001
            self._lerobot_resume_error = exc
            self._logger.exception("Background LeRobot resume failed.")
            return

        self._logger.info(
            "Resumed LeRobot data for actor rank "
            f"{self._rank}: shards={len(shards_to_load)}, "
            f"ids={[shard['id'] for shard in shards_to_load]}, "
            f"episodes={total_episodes}, frames={total_frames}, "
            f"resume_workers={self._lerobot_resume_num_workers}, "
            f"next_archive_id={self._next_lerobot_archive_id}"
        )

    def _refresh_lerobot_loader_after_resume(self) -> None:
        with self._lerobot_loader_lock:
            if not self.dataset.is_ready():
                return
            if self._lerobot_loader is not None:
                self._data_epoch += 1
            self._build_lerobot_data_loader()

    def _select_lerobot_resume_shards(self, valid_shards: list[dict]) -> list[dict]:
        total_frames = sum(shard["num_frames"] for shard in valid_shards)
        window_size = self._online_lerobot_cfg().get(
            "rolling_lerobot_window_size", None
        )
        if (
            window_size is None
            or int(window_size) <= 0
            or total_frames <= int(window_size)
        ):
            return valid_shards

        selected = []
        selected_frames = 0
        for shard in reversed(valid_shards):
            selected.append(shard)
            selected_frames += shard["num_frames"]
            if selected_frames >= int(window_size):
                break
        selected.reverse()
        return selected

    def _build_lerobot_data_loader(self):
        self.data_loader = build_dataloader_from_dataset(
            dataset=self.dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            world_size=self._world_size,
            rank=self._rank,
            use_random_replacement=True,
            num_samples_per_epoch=self.cfg.actor.global_batch_size,
            seed=self.cfg.actor.get("seed", 42),
            num_workers=0,
        )

        if hasattr(self.data_loader.sampler, "set_epoch"):
            self.data_loader.sampler.set_epoch(self._data_epoch)
        self._logger.info(
            "in _build_lerobot_data_loader: len(data_loader)=%d, "
            "len(dataset)=%d, num_samples_per_epoch=%d",
            len(self.data_loader),
            len(self.dataset),
            self.cfg.actor.global_batch_size,
        )
        # Point the unified loader/iter at the new DataLoader.
        self._lerobot_loader = self.data_loader
        self._lerobot_iter = iter(self.data_loader)

    def init_worker(self):
        super().setup_model_and_optimizer()
        self.setup_dagger_components()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()
        if self.cfg.actor.get("compile_model", False):
            self.model = torch.compile(self.model, mode="default")

    def setup_dagger_components(self):
        """Initialize DAgger-specific replay buffer state."""
        seed = self.cfg.actor.get("seed", 1234)
        if not self.enable_online_lerobot:
            auto_save_path = self.cfg.algorithm.replay_buffer.get(
                "auto_save_path", None
            )
            if auto_save_path is None:
                auto_save_path = os.path.join(
                    self.cfg.runner.logger.log_path, f"replay_buffer/rank_{self._rank}"
                )
            else:
                auto_save_path = os.path.join(auto_save_path, f"rank_{self._rank}")
            self.replay_buffer = TrajectoryReplayBuffer(
                seed=seed,
                enable_cache=self.cfg.algorithm.replay_buffer.enable_cache,
                cache_size=self.cfg.algorithm.replay_buffer.cache_size,
                sample_window_size=self.cfg.algorithm.replay_buffer.sample_window_size,
                auto_save=self.cfg.algorithm.replay_buffer.get("auto_save", False),
                auto_save_path=auto_save_path,
                trajectory_format=self.cfg.algorithm.replay_buffer.get(
                    "trajectory_format", "pt"
                ),
            )
        else:
            self._build_lerobot_dataset()
            self._resume_lerobot_dataset()

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        clear_memory(sync=False)

        if not self.enable_online_lerobot:
            from rlinf.workers.trajectory.channel import TrajectoryChannel

            split_num = 1
            if not isinstance(input_channel, TrajectoryChannel):
                send_num = (
                    self._component_placement.get_world_size("env") * self.stage_num
                )
                recv_num = self._component_placement.get_world_size("actor")
                split_num = compute_split_num(send_num, recv_num)
            recv_list = []
            for _ in range(split_num):
                trajectory: Trajectory = await input_channel.get(
                    async_op=True
                ).async_wait()
                recv_list.append(trajectory)
            return self.recv_buffer_rollout_trajectories(recv_list)
        else:
            return self.recv_lerobot_rollout_trajectories(input_channel)

    def recv_buffer_rollout_trajectories(self, recv_list: list[Trajectory]) -> None:
        intervene_traj_list = []
        for traj in recv_list:
            assert isinstance(traj, Trajectory)
            intervene_trajs = traj.extract_intervene_traj(mode="all")
            if intervene_trajs is not None:
                intervene_traj_list.extend(intervene_trajs)
        if intervene_traj_list:
            self.replay_buffer.add_trajectories(intervene_traj_list)

    def _recv_lerobot_episodes_from_channel(self, input_channel: Channel) -> bool:
        """Receive up to one actor-side split from the shared Actor channel.

        Each rank pulls at most ``split_num`` messages per call so multi-actor
        recv stays balanced like the buffer trajectory path. ``get_nowait`` is
        used because env ranks with no completed episodes send nothing.
        """
        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)
        received_any = False
        for _ in range(split_num):
            try:
                episodes: list[list[dict]] = input_channel.get_nowait()
            except asyncio.QueueEmpty:
                break
            received_any = True
            for ep_frames in episodes:
                if ep_frames:
                    self._append_lerobot_episode(ep_frames)
        return received_any

    def recv_lerobot_rollout_trajectories(self, input_channel: Channel) -> None:
        """Receive episodes from EnvWorker and append them to the memory dataset.

        EnvWorkers collect completed episodes via ``EmbodiedLerobotRolloutResult``
        and send them here each interact round. Empty batches are not sent by env;
        if the dataset is still below ``min_frames``, training is skipped later.
        """
        self._recv_lerobot_episodes_from_channel(input_channel)
        if self.dataset.is_ready():
            self._ensure_lerobot_loader()

    @staticmethod
    def _collect_lerobot_image_keys(
        frame: dict, prefix: str
    ) -> dict[str, tuple[int, ...]]:
        """Return ``{key: shape}`` for all frame keys matching *prefix*."""
        return {
            k: tuple(frame[k].shape)
            for k in frame
            if (k == prefix or k.startswith(f"{prefix}-"))
            and isinstance(frame[k], np.ndarray)
            and frame[k].ndim == 3
        }

    def _ensure_lerobot_loader(self) -> None:
        with self._lerobot_loader_lock:
            if self._lerobot_loader is not None:
                return
            self._build_lerobot_data_loader()

    def _current_archive_path(self) -> str:
        if self._pending_archive_path is None:
            self._pending_archive_path = os.path.join(
                self.cfg.algorithm.dagger.online_lerobot.data_path,
                f"rank_{self._rank}",
                f"id_{self._next_lerobot_archive_id}",
            )
        return self._pending_archive_path

    @Worker.timer("append_lerobot_episode")
    def _append_lerobot_episode(self, ep_frames: list[dict]) -> None:
        """Append one received episode to memory and queue it for archive output.
        The memory append is the training path. The pending archive buffer is
        flushed to disk every
        ``algorithm.dagger.online_lerobot.finalize_interval`` episodes.

        Args:
            ep_frames: List of per-step frame dicts as produced by
                ``CollectEpisode._buffer_to_lerobot_ep``.
        """
        if not ep_frames:
            return
        with self._pending_archive_lock:
            archive_path = self._current_archive_path()
        self.dataset.append_episode_to_memory(archive_path, ep_frames)
        should_archive = False
        with self._pending_archive_lock:
            self._pending_archive_episodes.append(ep_frames)
            self._next_lerobot_archive_id += 1
            finalize_interval = OmegaConf.select(
                self.cfg, "algorithm.dagger.online_lerobot.finalize_interval", default=8
            )
            if finalize_interval > 0 and len(self._pending_archive_episodes) >= int(
                finalize_interval
            ):
                should_archive = True
        if should_archive:
            self._archive_pending_lerobot_episodes()

    @Worker.timer("archive_lerobot_episodes")
    def _archive_pending_lerobot_episodes(self) -> None:
        with self._pending_archive_lock:
            if not self._pending_archive_episodes or self._pending_archive_path is None:
                return
            archive_path = self._pending_archive_path
            pending_episodes = self._pending_archive_episodes
            self._pending_archive_path = None
            self._pending_archive_episodes = []

        from rlinf.data.lerobot_writer import LeRobotDatasetWriter

        writer = LeRobotDatasetWriter()
        first = pending_episodes[0][0]
        wrist_image_keys = self._collect_lerobot_image_keys(first, "wrist_image")
        extra_view_image_keys = self._collect_lerobot_image_keys(
            first, "extra_view_image"
        )
        writer.create(
            repo_id=archive_path,
            robot_type=OmegaConf.select(
                self.cfg, "algorithm.dagger.online_lerobot.robot_type", default="panda"
            ),
            fps=int(
                OmegaConf.select(
                    self.cfg, "algorithm.dagger.online_lerobot.fps", default=10
                )
            ),
            image_shape=first["image"].shape if "image" in first else None,
            state_dim=int(first["state"].shape[-1]),
            action_dim=int(first["actions"].shape[-1]),
            has_image="image" in first,
            wrist_image_keys=wrist_image_keys,
            extra_view_image_keys=extra_view_image_keys,
            has_intervene_flag="intervene_flag" in first,
            has_segment_id="segment_id" in first,
        )
        for ep_frames in pending_episodes:
            writer.add_episode(ep_frames)
        writer.finalize()

    def _prepare_sft_batch(self, batch):
        """Prepare model-specific DAgger training inputs."""
        if not self.enable_online_lerobot:
            # Replay-buffer samples store model inputs under forward_inputs.
            if "forward_inputs" in batch:
                batch = batch["forward_inputs"]
            return self.model.prepare_dagger_sft_batch(batch)
        return self.model.prepare_lerobot_sft_batch(batch)

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        """Run one supervised forward pass for DAgger."""
        data = self._prepare_sft_batch(batch)
        use_action_chunk_loss = (
            SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI
        )
        return self.model(
            forward_type=ForwardType.SFT,
            data=data,
            use_action_chunk_loss=use_action_chunk_loss,
        )

    @Worker.timer("update_one_epoch")
    def update_one_epoch(self):
        if not self.enable_online_lerobot:
            return self.update_buffer_one_epoch()
        return self.update_lerobot_one_epoch()

    def update_buffer_one_epoch(self):
        """Run one replay-buffer update epoch for DAgger."""
        global_batch_size_per_rank = (
            self.cfg.actor.global_batch_size // self._world_size
        )
        with self.worker_timer("sample"):
            global_batch = self.replay_buffer.sample(
                num_chunks=global_batch_size_per_rank
            )

        train_micro_batch_list = split_dict_to_chunk(
            global_batch,
            global_batch_size_per_rank // self.cfg.actor.micro_batch_size,
        )
        for idx, batch in enumerate(train_micro_batch_list):
            batch = put_tensor_device(batch, device=self.device)
            if self.enable_drq:
                drq.apply_drq(batch["curr_obs"], pad=4)
                drq.apply_drq(batch["next_obs"], pad=4)
            train_micro_batch_list[idx] = batch

        self.optimizer.zero_grad()
        gbs_actor_loss = []
        for mb_idx, batch in enumerate(train_micro_batch_list):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(mb_idx + 1) == self.gradient_accumulation,
            )
            with self.amp_context:
                actor_loss = self.forward_actor(batch["forward_inputs"])
            actor_loss = actor_loss / self.gradient_accumulation
            with backward_ctx:
                self.grad_scaler.scale(actor_loss).backward()
            gbs_actor_loss.append(actor_loss.item() * self.gradient_accumulation)

        actor_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.optim.clip_grad
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        return {
            "dagger/actor_loss": np.mean(gbs_actor_loss),
            "actor/lr": self.optimizer.param_groups[0]["lr"],
            "actor/grad_norm": actor_grad_norm,
        }

    def update_lerobot_one_epoch(self):
        """Run one LeRobot update epoch."""

        with self.worker_timer("prepare_micro_batches"):
            with self._lerobot_loader_lock:
                num_batches = len(self._lerobot_loader)
                train_micro_batch_list = [
                    next(self._lerobot_iter) for _ in range(num_batches)
                ]
            for idx, batch in enumerate(train_micro_batch_list):
                batch = put_tensor_device(batch, device=self.device)
                if self.enable_drq:
                    drq.apply_drq(batch["curr_obs"], pad=4)
                train_micro_batch_list[idx] = batch

        self.optimizer.zero_grad()
        gbs_actor_loss = []

        for idx, batch in enumerate(train_micro_batch_list):
            # set the gradient accumulation backward_ctx
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == num_batches,
            )

            with self.amp_context:
                actor_loss = self.forward_actor(batch)

            actor_loss = actor_loss / num_batches
            with backward_ctx:
                self.grad_scaler.scale(actor_loss).backward()
            gbs_actor_loss.append(actor_loss.item() * num_batches)

        actor_grad_norm = self.model.clip_grad_norm_(
            max_norm=self.cfg.actor.optim.clip_grad
        )
        self.optimizer.step()
        self.lr_scheduler.step()

        with self._lerobot_loader_lock:
            self._data_epoch += 1
            if hasattr(self._lerobot_loader.sampler, "set_epoch"):
                self._lerobot_loader.sampler.set_epoch(self._data_epoch)
            self._lerobot_iter = iter(self._lerobot_loader)
        return {
            "dagger/actor_loss": np.mean(gbs_actor_loss),
            "actor/lr": self.optimizer.param_groups[0]["lr"],
            "actor/grad_norm": actor_grad_norm,
        }

    def process_train_metrics(self, metrics):
        """Aggregate DAgger training and replay-buffer metrics."""

        if not self.enable_online_lerobot:
            replay_buffer_stats = self.replay_buffer.get_stats()
            replay_buffer_stats = {
                f"replay_buffer/{key}": value
                for key, value in replay_buffer_stats.items()
            }
            append_to_dict(metrics, replay_buffer_stats)
        else:
            lerobot_dataset_stats = self.dataset.get_stats()
            lerobot_dataset_stats = {
                f"lerobot_dataset/{key}": value
                for key, value in lerobot_dataset_stats.items()
            }
            resume_thread = self._lerobot_resume_thread
            lerobot_dataset_stats.update(
                {
                    "lerobot_dataset/resume_done": int(self._lerobot_resume_done),
                    "lerobot_dataset/resume_loading": int(
                        resume_thread is not None and resume_thread.is_alive()
                    ),
                    "lerobot_dataset/resume_error": int(
                        self._lerobot_resume_error is not None
                    ),
                }
            )
            append_to_dict(metrics, lerobot_dataset_stats)

        mean_metric_dict = {}
        for key, value in metrics.items():
            if isinstance(value, list) and value:
                cpu_values = [
                    v.detach().cpu().item() if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
                mean_metric_dict[key] = np.mean(cpu_values)
            else:
                mean_metric_dict[key] = (
                    value.detach().cpu().item()
                    if isinstance(value, torch.Tensor)
                    else value
                )

        return all_reduce_dict(mean_metric_dict, op=torch.distributed.ReduceOp.AVG)

    def _lerobot_all_ranks_ready(self) -> bool:
        """Return True only when every actor rank can train from LeRobot data."""
        local_ready = int(self.dataset.is_ready() and self._lerobot_loader is not None)
        ready_tensor = torch.tensor(
            [local_ready], device=self.device, dtype=torch.int32
        )
        torch.distributed.all_reduce(ready_tensor, op=torch.distributed.ReduceOp.MIN)
        return ready_tensor.item() == 1

    def _skip_lerobot_training(self) -> bool:
        if self._lerobot_all_ranks_ready():
            return False
        self.log_on_first_rank(
            f"LeRobot dataset not ready (len={len(self.dataset)}), skipping training"
        )
        return True

    @Worker.timer("run_training")
    def run_training(self):
        """Run DAgger updates with replay-buffer samples."""
        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)
        if not self.enable_online_lerobot:
            min_buffer_size = self.cfg.algorithm.replay_buffer.get(
                "min_buffer_size", 100
            )
            if not self.replay_buffer.is_ready(min_buffer_size):
                self.log_on_first_rank(
                    f"Replay buffer size {len(self.replay_buffer)} < {min_buffer_size}, skipping training"
                )
                return {}
        elif self._skip_lerobot_training():
            return {}
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            metrics_data = self.update_one_epoch()
            append_to_dict(metrics, metrics_data)
            self.update_step += 1

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return self.process_train_metrics(metrics)

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self):
        """Skip advantage computation for supervised DAgger updates."""
        return {}

    def save_checkpoint(self, save_base_path, step):
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
            self.is_weight_offloaded = False
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)
            self.is_optimizer_offloaded = False

        if self.enable_online_lerobot:
            self._archive_pending_lerobot_episodes()

        self._strategy.save_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            save_path=save_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )
        if not self.enable_online_lerobot:
            buffer_save_path = os.path.join(
                save_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.save_checkpoint(buffer_save_path)

    def load_checkpoint(self, load_base_path):
        self._strategy.load_checkpoint(
            model=self.model,
            optimizers=[self.optimizer],
            lr_schedulers=[self.lr_scheduler],
            load_path=load_base_path,
            checkpoint_format="local_shard"
            if self.cfg.actor.fsdp_config.use_orig_params
            else "dcp",
        )

        if not self.enable_online_lerobot:
            buffer_load_path = os.path.join(
                load_base_path, f"dagger_components/replay_buffer/rank_{self._rank}"
            )
            self.replay_buffer.load_checkpoint(buffer_load_path)
