# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import pickle
from types import SimpleNamespace

import numpy as np
import torch

from examples.reward.preprocess_vlm_trend_reward_dataset import (
    build_terminal_success_rows,
)
from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.models.embodiment.reward.vlm_reward_model import BufferedVLMRewardModel
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    VLMTrendRewardInputBuilder,
    VLMTrendSuccessPotentialInputBuilder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    VLMTrendBinaryDigitRewardParser,
)
from rlinf.models.embodiment.reward.vlm_trend_success_potential_reward_model import (
    ScalarPotentialHead,
    VLMTrendSuccessPotentialRewardModel,
)


def test_success_potential_input_builder_preserves_standard_prompt() -> None:
    standard = VLMTrendRewardInputBuilder(history_buffer_names=["history_window"])
    specialized = VLMTrendSuccessPotentialInputBuilder(
        history_buffer_names=["history_window"],
        prompt_template="Task:{task}; bins:{num_bins_max}",
        num_bins=10,
    )

    assert "Judge whether the action trend is positive" in standard._render_prompt(
        "PickCube"
    )
    assert standard._video_fps() == 24.0
    assert specialized._render_prompt("PickCube") == "Task:PickCube; bins:9"
    assert specialized._video_fps() == 24.0


def test_success_potential_model_has_dedicated_registry_entry() -> None:
    assert (
        get_reward_model_class("vlm_trend_success_potential")
        is VLMTrendSuccessPotentialRewardModel
    )
    assert get_reward_model_class("buffered_vlm") is BufferedVLMRewardModel


class _HiddenModel:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden
        self.device = torch.device("cpu")

    def __call__(self, **_kwargs):
        return SimpleNamespace(hidden_states=[self.hidden])


class _IdentityHead:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        return features.squeeze(-1)


def test_terminal_success_rows_use_online_windows_without_balancing(tmp_path) -> None:
    observations = [
        {
            "states": torch.zeros(2),
            "main_images": np.zeros((2, 2, 3), dtype=np.uint8),
            "extra_view_images": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        }
        for _ in range(8)
    ]
    infos = [{"success": index == 6} for index in range(8)]
    episode = {
        "observations": observations,
        "actions": [torch.zeros(1) for _ in range(7)],
        "infos": infos,
        "terminated": [False] * 7 + [True],
        "truncated": [False] * 8,
        "success": True,
    }
    with (tmp_path / "episode.pkl").open("wb") as stream:
        pickle.dump(episode, stream)

    rows, stats = build_terminal_success_rows(
        [str(tmp_path)],
        tmp_path / "output",
        window_size=5,
        interval=3,
        val_split=0,
        workers=1,
        seed=0,
        task_description="test task",
    )

    samples = sorted(
        (row["segment_metadata"]["end_step"], row["answer"]) for row in rows["train"]
    )
    assert samples == [(5, "0"), (6, "1")]
    assert stats["splits"]["train"] == {
        "positive": 1,
        "negative": 1,
        "interval": 3,
    }


def test_scalar_head_uses_last_attended_prompt_token() -> None:
    model = object.__new__(VLMTrendSuccessPotentialRewardModel)
    torch.nn.Module.__init__(model)
    hidden = torch.zeros(2, 4, 1)
    hidden[0, 1, 0], hidden[1, 3, 0] = -2.0, 2.0
    model._model = _HiddenModel(hidden)
    model.scalar_head = _IdentityHead()

    potentials = model.compute_scalar_potential(
        {
            "input_ids": torch.zeros(2, 4, dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0, 0], [0, 1, 1, 1]]),
        }
    )

    torch.testing.assert_close(potentials, torch.sigmoid(torch.tensor([-2.0, 2.0])))


def test_scalar_head_loads_auxiliary_checkpoint(tmp_path) -> None:
    source = ScalarPotentialHead(4, 3, 0.1)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model_state_dict": source.state_dict(),
            "config": {"input_dim": 4, "hidden_dim": 3, "dropout": 0.1},
        },
        checkpoint,
    )
    model = object.__new__(VLMTrendSuccessPotentialRewardModel)
    torch.nn.Module.__init__(model)
    model.scalar_head_path = str(checkpoint)
    model._model = SimpleNamespace(device=torch.device("cpu"))

    model.setup_scalar_head()

    assert model.scalar_head is not None
    model.scalar_head.eval()
    source.eval()
    torch.testing.assert_close(
        model.scalar_head(torch.ones(2, 4)), source(torch.ones(2, 4))
    )


def test_potential_and_success_state_reset_at_episode_end() -> None:
    model = object.__new__(VLMTrendSuccessPotentialRewardModel)
    model.potential_scale = 1.0
    model.potential_gamma = 1.0
    model.potential_ema_alpha = 1.0
    model.potential_clip = 0.0
    model._previous_potentials = None
    model.success_threshold = 0.5
    model.success_bonus = 2.0
    model.success_confirmation_windows = 2
    model._success_fired = None
    model._success_streak = None
    valid = torch.tensor([True, True])

    torch.testing.assert_close(
        model.potential_differences(torch.tensor([0.2, 0.8]), valid), torch.zeros(2)
    )
    terminal = model.potential_differences(
        torch.tensor([0.9, 0.7]), valid, dones=torch.tensor([True, False])
    )
    torch.testing.assert_close(terminal, torch.tensor([0.7, -0.1]))
    torch.testing.assert_close(
        model.potential_differences(torch.tensor([0.1, 0.6]), valid),
        torch.tensor([0.0, -0.1]),
    )

    zeros = torch.zeros(2)
    first = model.apply_model_success_bonus(zeros, torch.tensor([0.9, 0.1]), valid)
    second = model.apply_model_success_bonus(zeros, torch.tensor([0.9, 0.9]), valid)
    third = model.apply_model_success_bonus(
        zeros, torch.tensor([0.9, 0.9]), valid, dones=torch.tensor([False, True])
    )
    after_reset = model.apply_model_success_bonus(
        zeros, torch.tensor([0.9, 0.9]), valid
    )
    torch.testing.assert_close(first, zeros)
    torch.testing.assert_close(second, torch.tensor([2.0, 0.0]))
    torch.testing.assert_close(third, torch.tensor([0.0, 2.0]))
    torch.testing.assert_close(after_reset, zeros)


def test_binary_success_parser_uses_last_standalone_digit() -> None:
    parser = VLMTrendBinaryDigitRewardParser()
    torch.testing.assert_close(
        parser.parse_rewards(["answer: 1", "0", "invalid", "0 then 1"]),
        torch.tensor([1.0, 0.0, 0.0, 1.0]),
    )
