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

"""Integration tests for EnvWorker → HistoryVLMRewardModel reward path."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

import torch
from omegaconf import OmegaConf

# Avoid optional env deps when importing EnvWorker in unit tests.
if "gymnasium" not in sys.modules:
    sys.modules["gymnasium"] = MagicMock()
if "rlinf.envs.wrappers" not in sys.modules:
    sys.modules["rlinf.envs.wrappers"] = MagicMock()

from rlinf.data.embodied_io_struct import EnvOutput  # noqa: E402
from rlinf.models.embodiment.reward.vlm_reward_model import (  # noqa: E402
    HistoryVLMRewardModel,
)
from rlinf.workers.env.env_worker import EnvWorker  # noqa: E402
from rlinf.workers.env.history_manager import HistoryManager  # noqa: E402


def _make_env_worker(num_envs: int = 1) -> EnvWorker:
    from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType

    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "reward": {
                "group_name": "RewardGroup",
                "reward_mode": "history_buffer",
                "model": {
                    "history_buffers": {
                        "history_window": {
                            "history_size": 2,
                            "min_history_size": 1,
                            "input_interval": 1,
                            "history_keys": ["main_images"],
                            "input_on_done": False,
                        }
                    }
                },
            }
        }
    )
    worker.reward_mode = "history_buffer"
    worker.train_num_envs_per_stage = num_envs
    worker.train_batch_size = num_envs
    worker.env_decoupled_mode = False
    worker.env_infos_reward_keys = ("success", "episode", "final_info")
    worker._timer_metrics = {}
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker.train_history_managers = [
        HistoryManager(worker.cfg.reward, num_envs=num_envs)
    ]
    worker.history_lengths = [{}]
    return worker


def _capture_reward_input(worker: EnvWorker, env_output: EnvOutput) -> dict:
    """Run get_reward_model_output and return the reward_input it would send."""
    captured: dict = {}

    def _send_to(**kwargs):
        captured["reward_input"] = kwargs["data"]

    worker.send_to = _send_to
    worker.recv_from = lambda **_kwargs: torch.zeros(
        worker.train_num_envs_per_stage, dtype=torch.float32
    )
    worker.get_reward_model_output(
        env_output,
        send_channel=MagicMock(),
        recv_channel=MagicMock(),
        stage_id=0,
    )
    return captured["reward_input"]


def test_get_reward_model_output_attaches_1d_dones():
    worker = _make_env_worker(num_envs=2)
    dones = torch.tensor([True, False])
    env_output = EnvOutput(
        obs={"main_images": torch.zeros(2, 3, 4, 4)},
        dones=dones,
    )

    reward_input = _capture_reward_input(worker, env_output)

    assert "dones" in reward_input
    torch.testing.assert_close(reward_input["dones"], dones)


def test_get_reward_model_output_reduces_2d_dones_with_any():
    """Mid-chunk termination must not be dropped by last-column slicing."""
    worker = _make_env_worker(num_envs=2)
    dones = torch.tensor(
        [
            [False, False, True, False, False],
            [False, False, False, False, False],
        ]
    )
    env_output = EnvOutput(
        obs={"main_images": torch.zeros(2, 3, 4, 4)},
        dones=dones,
    )

    reward_input = _capture_reward_input(worker, env_output)

    assert "dones" in reward_input
    torch.testing.assert_close(reward_input["dones"], torch.tensor([True, False]))


def _make_history_reward_model(
    monkeypatch, score_queue: list[tuple]
) -> HistoryVLMRewardModel:
    """HistoryVLMRewardModel that uses real compute_reward / state reset paths."""
    model = object.__new__(HistoryVLMRewardModel)
    model.interval_reward = 0.0
    model.infer_micro_batch_size = 0
    model.inference_mode = "scalar_head"
    model.potential_scale = 1.0
    model.potential_gamma = 1.0
    model.potential_ema_alpha = 1.0
    model.potential_clip = 0.0
    model._previous_potentials = None
    model.success_threshold = 0.8
    model.success_bonus = 1.0
    model.success_confirmation_windows = 1
    model._success_fired = None
    model._success_streak = None
    model.success_input_builder = object()  # non-None enables success bonus path
    model.gt_success_bonus = 0.0

    scores = iter(score_queue)

    def _score_micro_batch(_obs, _history, batch_size: int):
        potential, success = next(scores)
        assert batch_size == 1
        return (
            torch.tensor([potential], dtype=torch.float32),
            torch.tensor([True]),
            torch.tensor([success], dtype=torch.float32),
        )

    monkeypatch.setattr(model, "_score_micro_batch", _score_micro_batch)
    monkeypatch.setattr(model, "apply_gt_success_bonus", lambda rewards, _: rewards)
    return model


def test_reward_computation_across_two_episodes_through_env_worker_path(monkeypatch):
    """Integration requested by reviewer: two episodes via the env-worker path.

    Flow: EnvWorker builds ``reward_input`` (including 1-D ``dones``) →
    ``HistoryVLMRewardModel.compute_reward`` consumes it. Without attaching
    1-D dones, potential / success state would leak across auto-reset.
    """
    worker = _make_env_worker(num_envs=1)
    # Scores fed into compute_reward via mocked _score_micro_batch:
    # ep1 step1 potential=0.2 success=0.9 → fire bonus once
    # ep1 step2 potential=0.9 success=0.9 + done → reset state
    # ep2 step1 potential=0.1 success=0.9 → potential delta 0, bonus can fire again
    model = _make_history_reward_model(
        monkeypatch,
        score_queue=[
            (0.2, 0.9),
            (0.9, 0.9),
            (0.1, 0.9),
        ],
    )

    def _obs() -> dict:
        return {"main_images": torch.zeros(1, 3, 4, 4)}

    # Episode 1, step 1 (not done): establish potential + fire success bonus.
    ep1_step1 = _capture_reward_input(
        worker, EnvOutput(obs=_obs(), dones=torch.tensor([False]))
    )
    assert "dones" in ep1_step1
    assert ep1_step1["dones"].tolist() == [False]
    assert any(ep1_step1["history_input"].values())
    rewards_1 = model.compute_reward(ep1_step1)
    torch.testing.assert_close(rewards_1, torch.tensor([1.0]))  # first ΔΦ=0 + bonus

    # Episode 1, step 2 (done via 1-D dones): shaping + state reset.
    ep1_done = _capture_reward_input(
        worker, EnvOutput(obs=_obs(), dones=torch.tensor([True]))
    )
    assert ep1_done["dones"].tolist() == [True]
    rewards_done = model.compute_reward(ep1_done)
    # ΔΦ = 0.9 - 0.2 = 0.7; success already fired → no second bonus
    torch.testing.assert_close(rewards_done, torch.tensor([0.7]))

    # Episode 2, step 1: after reset, first potential delta is 0 and bonus can fire.
    ep2_step1 = _capture_reward_input(
        worker, EnvOutput(obs=_obs(), dones=torch.tensor([False]))
    )
    assert ep2_step1["dones"].tolist() == [False]
    rewards_2 = model.compute_reward(ep2_step1)
    torch.testing.assert_close(rewards_2, torch.tensor([1.0]))  # ΔΦ=0 + bonus again
