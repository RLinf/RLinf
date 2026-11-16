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

"""DAgger supervised accumulation and optimizer contracts on CPU."""

import asyncio
import contextlib
import logging
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.scheduler import Worker
from rlinf.scheduler.hardware.accelerators import AcceleratorType
from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
    AsyncEmbodiedDAGGERFSDPPolicy,
)
from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


class SmallPolicy(torch.nn.Module):
    def __init__(self, dictionary_output):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.dictionary_output = dictionary_output
        self.steps = []

    def forward(self, batch):
        loss = (self.weight - batch["actions"]).square().mean()
        if self.dictionary_output:
            return {"loss": loss, "flow_matching_loss_action": loss}
        return loss

    def set_global_step(self, step, **kwargs):
        self.steps.append((step, kwargs))


class CPUStrategy:
    """Replace distributed hardware while using the real manager/scaler step."""

    def clip_grad_norm_(self, *, model):
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 1000)

    def get_model_state_dict(self, model, cpu_offload, full_state_dict):
        return model.state_dict()

    def save_checkpoint(self, *, model, optimizers, lr_schedulers, save_path, **kwargs):
        Path(save_path).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizers[0].state_dict(),
                "scheduler": lr_schedulers[0].state_dict(),
            },
            Path(save_path) / "state.pt",
        )

    def load_checkpoint(self, *, model, optimizers, lr_schedulers, load_path, **kwargs):
        state = torch.load(Path(load_path) / "state.pt", weights_only=True)
        model.load_state_dict(state["model"])
        optimizers[0].load_state_dict(state["optimizer"])
        lr_schedulers[0].load_state_dict(state["scheduler"])


class CPUWorker(EmbodiedDAGGERFSDPPolicy):
    @property
    def device(self):
        return torch.device("cpu")

    def worker_timer(self, *args, **kwargs):
        return contextlib.nullcontext()

    @contextlib.contextmanager
    def before_micro_batch(self, model, *, is_last_micro_batch):
        self.context_flags.append(is_last_micro_batch)
        self.inside_backward_context = True
        try:
            yield
        finally:
            self.inside_backward_context = False

    def forward_actor(self, batch):
        if not self.inside_backward_context:
            raise AssertionError(
                "Forward must be inside the FSDP accumulation context."
            )
        return self.model(batch)


class CPUAsyncWorker(AsyncEmbodiedDAGGERFSDPPolicy, CPUWorker):
    def broadcast(self, data, **kwargs):
        return self.transport.send(data)


class CPUPlatform:
    @staticmethod
    def current_device():
        return torch.device("cpu")

    @staticmethod
    def synchronize():
        pass

    @staticmethod
    def empty_cache():
        pass


class TrainingDataset(torch.utils.data.Dataset):
    def __init__(self, targets):
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {"actions": torch.tensor(self.targets[index])}

    def is_ready(self):
        return bool(self.targets)

    def get_stats(self):
        return {"total_frames": len(self)}


class DelayedBroadcast:
    """An in-flight transport keeps its tensor payload until send completes."""

    def __init__(self):
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.sent = []

    def send(self, data):
        async def async_wait():
            self.started.set()
            await self.release.wait()
            self.sent.append({key: value.clone() for key, value in data.items()})

        return SimpleNamespace(async_wait=async_wait)


def make_worker(
    *,
    model_type="cosmos3",
    dictionary_output=True,
    targets=(1.0, 3.0),
    worker_cls=CPUWorker,
):
    worker = object.__new__(worker_cls)
    worker.cfg = SimpleNamespace(
        actor=SimpleNamespace(model=SimpleNamespace(model_type=model_type))
    )
    worker.model = SmallPolicy(dictionary_output)
    worker.optimizer = torch.optim.SGD(worker.model.parameters(), lr=0.1)
    worker.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        worker.optimizer, lambda _: 1
    )
    worker.grad_scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=128)
    worker.amp_context = contextlib.nullcontext()
    worker._strategy = CPUStrategy()
    worker.optimizer_steps = 0
    worker.critic_warmup_steps = 0
    worker.update_step = 0
    worker._logger = logging.getLogger(__name__)
    worker._stacklevel = 1
    worker.enable_drq = False
    worker._data_epoch = 0
    worker._lerobot_loader_lock = threading.Lock()
    worker._lerobot_loader = torch.utils.data.DataLoader(
        [{"actions": torch.tensor(value)} for value in targets], batch_size=1
    )
    worker._lerobot_iter = iter(worker._lerobot_loader)
    worker.inside_backward_context = False
    worker.context_flags = []
    return worker


def make_async_worker(*, targets=(1.0, 3.0), update_epoch=2):
    worker = make_worker(targets=targets, worker_cls=CPUAsyncWorker)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "model": {"model_type": "cosmos3"},
                "global_batch_size": len(targets),
                "micro_batch_size": 1,
                "fsdp_config": {"use_orig_params": True},
            },
            "algorithm": {"update_epoch": update_epoch},
        }
    )
    worker._world_size = 1
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker.enable_online_lerobot = True
    worker.enable_offload = False
    worker.is_weight_offloaded = False
    worker.is_optimizer_offloaded = False
    worker.dataset = TrainingDataset(targets)
    worker._lerobot_resume_thread = None
    worker._lerobot_resume_done = True
    worker._lerobot_resume_error = None
    worker._model_lock = asyncio.Lock()
    worker._lerobot_ingest_lock = threading.RLock()
    worker._pending_archive_lock = threading.Lock()
    worker._pending_archive_episodes = []
    worker._pending_archive_path = None
    worker._recv_error = None
    worker.should_stop = False
    worker.torch_platform = CPUPlatform
    worker.version = 0
    return worker


class DaggerWorkerTests(unittest.TestCase):
    def test_scaled_accumulation_matches_global_batch_gradient(self):
        worker = make_worker()
        metrics = worker.update_lerobot_one_epoch()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.4))
        self.assertEqual(worker.context_flags, [False, True])
        self.assertEqual(worker.update_step, 1)
        self.assertEqual(worker.optimizer_steps, 1)
        self.assertEqual(worker.lr_scheduler.last_epoch, 1)
        self.assertEqual(worker.model.steps, [(1, {})])
        self.assertEqual(metrics["dagger/actor_loss"], 5)
        self.assertEqual(metrics["dagger/flow_matching_loss_action"], 5)
        self.assertIsNone(worker.model.weight.grad)
        # The iterator is reset for the next DAgger update epoch.
        worker.update_lerobot_one_epoch()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.72))
        self.assertEqual(worker.update_step, 2)

    def test_tensor_loss_preserves_openpi_updates(self):
        worker = make_worker(
            model_type="openpi", dictionary_output=False, targets=(1.0,)
        )
        metrics = worker.update_lerobot_one_epoch()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.2))
        self.assertEqual(worker.model.steps, [])
        self.assertEqual(metrics["dagger/actor_loss"], 1)

    def test_nonfinite_gradients_skip_parameters_scheduler_and_ema(self):
        worker = make_worker(targets=(float("nan"),))
        metrics = worker.update_lerobot_one_epoch()
        self.assertEqual(worker.model.weight.item(), 0)
        self.assertEqual(worker.update_step, 0)
        self.assertEqual(worker.lr_scheduler.last_epoch, 0)
        self.assertEqual(worker.model.steps, [])
        self.assertEqual(metrics["dagger/update_skipped"], 1)
        self.assertIsNone(worker.model.weight.grad)

    def test_rollout_version_does_not_advance_cosmos_optimizer_iteration(self):
        worker = make_worker()
        worker.update_step = 7
        worker.set_global_step(100)
        self.assertEqual(worker.version, 100)
        self.assertEqual(worker.model.steps, [(7, {"update_ema": False})])

    def test_loss_contract_rejects_missing_detached_and_vector_losses(self):
        for output in (
            {},
            {"loss": torch.tensor(1.0)},
            torch.ones(2, requires_grad=True),
        ):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    EmbodiedDAGGERFSDPPolicy._sft_loss_and_metrics(output)

    def test_training_horizon_is_independent_of_execution_chunk(self):
        with tempfile.TemporaryDirectory() as root:
            worker = make_worker()
            worker.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {"model_type": "openpi", "num_action_chunks": 10}
                    },
                    "algorithm": {
                        "dagger": {
                            "online_lerobot": {"data_path": root, "min_frames": 1024}
                        }
                    },
                }
            )
            worker._build_lerobot_dataset()
            self.assertEqual(worker.dataset.chunk_size, 10)
            worker.cfg.algorithm.dagger.online_lerobot.training_action_horizon = 16
            worker._build_lerobot_dataset()
            self.assertEqual(worker.dataset.chunk_size, 16)
            self.assertEqual(worker.cfg.actor.model.num_action_chunks, 10)
            self.assertEqual(worker.dataset.min_frames, 1024)
            self.assertFalse(worker.dataset.is_ready())
            self.assertFalse(worker.dataset.require_full_window)
            worker.cfg.actor.model.model_type = "cosmos3"
            worker.cfg.actor.model.sft_objective = "joint"
            worker._build_lerobot_dataset()
            self.assertTrue(worker.dataset.require_full_window)
            self.assertEqual(worker.dataset._vision_horizon, 16)
            worker.cfg.actor.model.sft_objective = "action_only"
            worker._build_lerobot_dataset()
            self.assertFalse(worker.dataset.require_full_window)
            for invalid in (0, -1, 1.5, True, None, "16"):
                worker.cfg.algorithm.dagger.online_lerobot.training_action_horizon = (
                    invalid
                )
                with self.subTest(horizon=invalid):
                    with self.assertRaisesRegex(ValueError, "positive integer"):
                        worker._build_lerobot_dataset()

    def test_incomplete_archives_reserve_their_directory_names(self):
        with tempfile.TemporaryDirectory() as root:
            worker = make_worker()
            worker.cfg = OmegaConf.create(
                {
                    "actor": {
                        "model": {"model_type": "openpi", "num_action_chunks": 10}
                    },
                    "algorithm": {"dagger": {"online_lerobot": {"data_path": root}}},
                }
            )
            worker._rank = 3
            worker._next_lerobot_archive_id = 0
            worker._pending_archive_path = None
            worker._build_lerobot_dataset()
            rank_dir = Path(root) / "rank_3"
            for archive_id in (0, 7):
                (rank_dir / f"id_{archive_id}").mkdir(parents=True)
            self.assertEqual(worker._discover_lerobot_resume_shards(), [])
            next_path = Path(worker._current_archive_path())
            self.assertEqual(next_path, rank_dir / "id_8")
            self.assertFalse(next_path.exists())
            self.assertTrue((rank_dir / "id_0").is_dir())
            self.assertTrue((rank_dir / "id_7").is_dir())


class AsyncDaggerWorkerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.hardware = contextlib.ExitStack()
        self.hardware.enter_context(patch.object(Worker, "torch_platform", CPUPlatform))
        self.hardware.enter_context(
            patch.object(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
        )
        self.hardware.enter_context(
            patch.object(torch.distributed, "barrier", lambda *args, **kwargs: None)
        )
        self.addCleanup(self.hardware.close)

    async def test_async_updates_match_sync_optimizer_and_cosmos_iterations(self):
        worker = make_async_worker()
        await worker.run_training()
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.72))
        self.assertEqual(worker.update_step, 2)
        self.assertEqual(worker.optimizer_steps, 2)
        self.assertEqual(worker.lr_scheduler.last_epoch, 2)
        self.assertEqual(worker.model.steps, [(1, {}), (2, {})])

        with tempfile.TemporaryDirectory() as checkpoint:
            await worker.save_checkpoint(checkpoint, 1)
            restored = make_async_worker()
            await restored.load_checkpoint(checkpoint)
            self.assertEqual(restored.update_step, 2)
            self.assertEqual(restored.optimizer_steps, 2)
            torch.testing.assert_close(restored.model.weight, worker.model.weight)
            await restored.set_global_step(123)
            self.assertEqual(restored.version, 123)
            self.assertEqual(restored.model.steps[-1], (2, {"update_ema": False}))

    async def test_async_nonfinite_update_does_not_advance_cosmos_iteration(self):
        worker = make_async_worker(targets=(float("nan"),))
        metrics = await worker.run_training()
        self.assertEqual(worker.model.weight.item(), 0)
        self.assertEqual(worker.update_step, 0)
        self.assertEqual(worker.lr_scheduler.last_epoch, 0)
        self.assertEqual(worker.model.steps, [])
        self.assertEqual(metrics["dagger/update_skipped"], 1)

    async def test_memory_training_is_ready_while_episode_archive_is_blocked(self):
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:
            try:
                from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            except ModuleNotFoundError:
                self.skipTest("LeRobot is required for the online archive boundary.")

        archive_started = threading.Event()
        release_archive = threading.Event()
        save_episode = LeRobotDataset.save_episode

        def delayed_save_episode(dataset, *args, **kwargs):
            archive_started.set()
            if not release_archive.wait(20):
                raise TimeoutError("The test did not release the episode archive.")
            return save_episode(dataset, *args, **kwargs)

        with tempfile.TemporaryDirectory() as root:
            worker = make_async_worker(update_epoch=1)
            worker.cfg.actor.model.num_action_chunks = 1
            worker.cfg.algorithm.dagger = {
                "online_lerobot": {
                    "data_path": root,
                    "min_frames": 2,
                    "finalize_interval": 1,
                }
            }
            worker._rank = 0
            worker.stage_num = 1
            worker._component_placement = SimpleNamespace(get_world_size=lambda _: 1)
            worker._next_lerobot_archive_id = 0
            worker._lerobot_loader = None
            worker._lerobot_iter = None
            worker._build_lerobot_dataset()
            messages = queue.Queue()
            messages.put(
                [
                    [
                        {
                            "state": np.array([target], dtype=np.float32),
                            "actions": np.array([target], dtype=np.float32),
                            "done": np.array([index == 1], dtype=bool),
                            "is_success": np.array([True], dtype=bool),
                            "intervene_flag": np.array([True], dtype=bool),
                            "task": "pick up the cube",
                        }
                        for index, target in enumerate((1.0, 3.0))
                    ]
                ]
            )

            class EpisodeChannel:
                def get_nowait(self):
                    try:
                        return messages.get_nowait()
                    except queue.Empty as error:
                        raise asyncio.QueueEmpty from error

            with patch.object(LeRobotDataset, "save_episode", delayed_save_episode):
                try:
                    await worker.recv_rollout_trajectories(EpisodeChannel())
                    started = await asyncio.to_thread(archive_started.wait, 20)
                    self.assertTrue(started, worker._recv_error)
                    self.assertEqual(len(worker.dataset), 2)
                    self.assertIsNotNone(worker._lerobot_loader)
                    self.assertFalse(release_archive.is_set())
                    metrics = await worker.run_training()
                    self.assertGreater(worker.model.weight.item(), 0)
                    self.assertEqual(worker.update_step, 1)
                    self.assertEqual(metrics["dagger/update_skipped"], 0)
                finally:
                    release_archive.set()
                    await worker.stop()

    async def test_weight_send_finishes_before_training_can_mutate_its_payload(self):
        worker = make_async_worker(update_epoch=1)
        worker.weight_syncer = BucketWeightSyncer(1024, None, "cpu")
        worker._is_weight_sender = True
        worker._group_name = "actor"
        worker._rollout_group_name = "rollout"
        worker._rollout_all_ranks = [0]
        worker._sync_weight_comm_options = None
        worker.param_names_need_sync = ["weight"]
        worker.transport = DelayedBroadcast()

        sync = asyncio.create_task(worker.sync_model_to_rollout())
        await asyncio.wait_for(worker.transport.started.wait(), 1)
        training = asyncio.create_task(worker.run_training())
        await asyncio.sleep(0)
        self.assertFalse(training.done())
        worker.transport.release.set()
        await asyncio.wait_for(asyncio.gather(sync, training), 1)
        self.assertEqual(worker.transport.sent[0]["weight"].item(), 0)
        torch.testing.assert_close(worker.model.weight, torch.tensor(0.4))

    async def test_empty_replay_receiver_stops_without_another_trajectory(self):
        worker = make_async_worker()
        worker.enable_online_lerobot = False
        worker.stage_num = 1
        worker._component_placement = SimpleNamespace(get_world_size=lambda _: 1)

        class EmptyChannel:
            def get_nowait(self):
                raise asyncio.QueueEmpty

        await worker.recv_rollout_trajectories(EmptyChannel())
        await asyncio.wait_for(worker.stop(), 1)
        self.assertFalse(worker._recv_rollout_thread.is_alive())

    async def test_receiver_failure_reaches_training_instead_of_waiting_forever(self):
        worker = make_async_worker()
        worker.enable_online_lerobot = False
        worker.stage_num = 1
        worker._component_placement = SimpleNamespace(get_world_size=lambda _: 1)
        worker._recv_queue = queue.Queue()

        class FailedChannel:
            def get_nowait(self):
                raise ValueError("channel is unavailable")

        await worker.recv_rollout_trajectories(FailedChannel())
        await asyncio.to_thread(worker._recv_rollout_thread.join, 1)
        with self.assertRaisesRegex(
            RuntimeError, "trajectory receiver failed"
        ) as error:
            await worker.run_training()
        self.assertIsInstance(error.exception.__cause__, ValueError)


if __name__ == "__main__":
    unittest.main()
