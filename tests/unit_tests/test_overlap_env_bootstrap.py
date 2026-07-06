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

import sys
import unittest
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EnvOutput

# Mock gymnasium and rlinf.envs.wrappers before importing EnvWorker
# to avoid ModuleNotFoundError when gymnasium is not installed.
# We do this here at the very top to satisfy functional requirements
# while using # noqa for linter satisfaction if needed.
if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = MagicMock()

if "rlinf.envs.wrappers" not in sys.modules:
    sys.modules["rlinf.envs.wrappers"] = MagicMock()

from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.workers.env.env_worker import EnvWorker  # noqa: E402


class TestOverlapEnvBootstrap(unittest.TestCase):
    def setUp(self):
        self.cfg = OmegaConf.create(
            {
                "env": {
                    "train": {
                        "total_num_envs": 2,
                        "max_steps_per_rollout_epoch": 8,
                        "env_type": "dummy",
                        "auto_reset": True,
                        "video_cfg": {"save_video": False},
                        "max_episode_steps": 10,
                    },
                    "eval": {
                        "total_num_envs": 2,
                        "max_steps_per_rollout_epoch": 8,
                        "env_type": "dummy",
                        "video_cfg": {"save_video": False},
                        "max_episode_steps": 10,
                    },
                },
                "actor": {
                    "model": {
                        "model_type": "dummy",
                        "num_action_chunks": 4,
                        "action_dim": 7,
                    }
                },
                "rollout": {
                    "group_name": "RolloutGroup",
                    "pipeline_stage_num": 1,
                    "collect_transitions": False,
                },
                "runner": {
                    "val_check_interval": -1,
                },
                "algorithm": {
                    "rollout_epoch": 1,
                },
                "cluster": {},
            }
        )

        # Create EnvWorker instance without calling __init__
        self.worker = object.__new__(EnvWorker)

        # Manually set required attributes
        self.worker.cfg = self.cfg
        self.worker._rank = 0
        self.worker._world_size = 1
        self.worker._group_name = "EnvGroup"
        self.worker._timer_metrics = {}
        self.worker.stage_num = 1
        self.worker.train_num_envs_per_stage = 2
        self.worker.n_train_chunk_steps = 2
        self.worker.rollout_epoch = 1
        self.worker.enable_online_lerobot = False
        self.worker.enable_offload = False
        self.worker.train_enable_offload = False
        self.worker.use_training_pipeline = False
        self.worker.collect_transitions = False
        self.worker.collect_prev_infos = True
        self.worker.reward_mode = self.cfg.get("reward", {}).get(
            "reward_mode", "per_step"
        )
        self.worker.history_reward_assign = self.cfg.get("reward", {}).get(
            "history_reward_assign", True
        )
        self.worker._accelerator_type = AcceleratorType.NO_ACCEL
        self.worker._prefetched_train_bootstrap = None

        # Mock env_list
        mock_env = MagicMock()
        self.worker.env_list = [mock_env]

        # Initialize last_obs_list for auto_reset=True
        self.worker.last_obs_list = [{"main_images": torch.zeros(2, 3, 224, 224)}]
        self.worker.last_intervened_info_list = [(None, None)]
        self.worker.only_eval = False
        self.worker.model_cfg = self.cfg.actor.model
        self.worker.train_batch_size = (
            self.cfg.env.train.total_num_envs // self.worker.stage_num
        )
        self.worker.env_decoupled_mode = False
        self.worker.send_to = MagicMock()

    def test_prefetch_consumption(self):
        """Test that prefetched bootstrap is correctly consumed in interact()."""
        rollout_channel = MagicMock()
        input_channel = MagicMock()

        # Mock recv_from to return a dummy RolloutResult
        mock_rollout_result = MagicMock()
        mock_rollout_result.actions = torch.zeros(2, 28)
        mock_rollout_result.bootstrap_values = None
        mock_rollout_result.forward_inputs = {"action": torch.zeros(2, 28)}
        mock_rollout_result.versions = torch.zeros(2, 1)
        mock_rollout_result.save_flags = None

        # Patch methods on the instance
        self.worker.recv_from = MagicMock(return_value=mock_rollout_result)
        self.worker.env_interact_step = MagicMock(
            return_value=(
                EnvOutput(
                    obs={"main_images": torch.zeros(2, 3, 224, 224)},
                    dones=torch.zeros(2, 4, dtype=torch.bool),
                    truncations=torch.zeros(2, 4, dtype=torch.bool),
                    terminations=torch.zeros(2, 4, dtype=torch.bool),
                ),
                {},
                {},
            )
        )
        self.worker.send_env_batch = MagicMock()
        self.worker.store_last_obs_and_intervened_info = MagicMock()
        self.worker.finish_rollout = MagicMock()
        self.worker.compute_bootstrap_rewards = MagicMock(
            return_value=torch.zeros(2, 4)
        )
        self.worker.record_env_metrics = MagicMock()

        # 1. Prefetch
        # We need to mock _bootstrap_and_send_train as it's called by prefetch_train_bootstrap
        dummy_bootstrap = [
            EnvOutput(obs={"m": torch.zeros(1)}, dones=torch.zeros(1, 4))
        ]
        self.worker._bootstrap_and_send_train = MagicMock(return_value=dummy_bootstrap)

        self.worker.prefetch_train_bootstrap(rollout_channel)
        self.assertEqual(self.worker._prefetched_train_bootstrap, dummy_bootstrap)

        # 2. Interact (should consume the prefetch)
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Mock send_rollout_trajectories as it's awaited
            self.worker.send_rollout_trajectories = MagicMock(
                return_value=asyncio.Future()
            )
            self.worker.send_rollout_trajectories.return_value.set_result(None)

            loop.run_until_complete(
                self.worker.interact(input_channel, rollout_channel, None, None)
            )
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        self.assertIsNone(self.worker._prefetched_train_bootstrap)
        # Verify that _bootstrap_and_send_train was NOT called during interact
        # (it was only called once during prefetch)
        self.worker._bootstrap_and_send_train.assert_called_once()
        self.assertEqual(self.worker.record_env_metrics.call_count, 2)

    def test_duplicate_prefetch_protection(self):
        """Test that multiple prefetch calls raise RuntimeError."""
        rollout_channel = MagicMock()
        self.worker._bootstrap_and_send_train = MagicMock()

        # First prefetch
        self.worker.prefetch_train_bootstrap(rollout_channel)

        # Second prefetch should raise RuntimeError
        with self.assertRaises(RuntimeError) as cm:
            self.worker.prefetch_train_bootstrap(rollout_channel)

        self.assertIn("A prefetched train bootstrap already exists", str(cm.exception))

    def test_record_env_metrics_appends_values(self):
        """record_env_metrics should append env info tensors as-is."""
        env_metrics = {}

        self.worker.record_env_metrics(
            env_metrics, {"episode_len": torch.tensor([5, 6])}
        )
        self.worker.record_env_metrics(
            env_metrics, {"episode_len": torch.tensor([7, 8])}
        )

        self.assertEqual(len(env_metrics["episode_len"]), 2)
        self.assertTrue(
            torch.equal(env_metrics["episode_len"][0], torch.tensor([5, 6]))
        )
        self.assertTrue(
            torch.equal(env_metrics["episode_len"][1], torch.tensor([7, 8]))
        )

    def test_interact_records_metrics_only_on_final_chunk_when_not_auto_reset(self):
        """Non-auto-reset training should record episode metrics only once per rollout epoch."""
        self.worker.cfg.env.train.auto_reset = False
        self.worker.cfg.env.train.ignore_terminations = False
        self.worker.env_list[0].reset.return_value = (
            {"main_images": torch.zeros(2, 3, 224, 224)},
            {},
        )
        self.worker.record_env_metrics = MagicMock()

        rollout_channel = MagicMock()
        input_channel = MagicMock()

        mock_rollout_result = MagicMock()
        mock_rollout_result.actions = torch.zeros(2, 28)
        mock_rollout_result.bootstrap_values = None
        mock_rollout_result.forward_inputs = {"action": torch.zeros(2, 28)}
        mock_rollout_result.versions = torch.zeros(2, 1)
        mock_rollout_result.save_flags = None

        self.worker.recv_from = MagicMock(return_value=mock_rollout_result)
        self.worker.env_interact_step = MagicMock(
            return_value=(
                EnvOutput(
                    obs={"main_images": torch.zeros(2, 3, 224, 224)},
                    dones=torch.zeros(2, 4, dtype=torch.bool),
                    truncations=torch.zeros(2, 4, dtype=torch.bool),
                    terminations=torch.zeros(2, 4, dtype=torch.bool),
                ),
                {"episode_len": torch.tensor([1, 2])},
                {},
            )
        )
        self.worker.send_env_batch = MagicMock()
        self.worker.store_last_obs_and_intervened_info = MagicMock()
        self.worker.finish_rollout = MagicMock()
        self.worker.compute_bootstrap_rewards = MagicMock(
            return_value=torch.zeros(2, 4)
        )
        self.worker._bootstrap_and_send_train = MagicMock(
            return_value=[EnvOutput(obs={"m": torch.zeros(1)}, dones=torch.zeros(1, 4))]
        )
        self.worker.send_rollout_trajectories = MagicMock(
            return_value=MagicMock(wait=MagicMock(return_value=None))
        )

        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                self.worker.interact(input_channel, rollout_channel, None, None)
            )
        finally:
            asyncio.set_event_loop(None)
            loop.close()

        self.assertEqual(self.worker.record_env_metrics.call_count, 1)


if __name__ == "__main__":
    unittest.main()
