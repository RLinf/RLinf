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

import torch

from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    QwentrendTerminalSuccessInputBuilder,
)
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    QwentrendBinaryDigitRewardParser,
)


def test_binary_digit_parser_uses_sparse_rewards():
    parser = QwentrendBinaryDigitRewardParser(unclear_reward=-0.2)

    rewards = parser.parse_rewards(["1", "0", "answer: 1", "unclear", "10"])

    torch.testing.assert_close(
        rewards,
        torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
    )


def test_terminal_success_builder_matches_sft_prompt(monkeypatch):
    builder = QwentrendTerminalSuccessInputBuilder(
        history_buffer_names=["history_window"],
        default_task_description="fallback task",
        _processor=None,
    )
    videos = [[["main frames"], ["extra frames"]]]
    monkeypatch.setattr(builder, "extract_videos", lambda *_: videos)
    observations = {"task_descriptions": ["Pick up the cube."]}
    history_input = {"history_window": {}}

    prepared = builder.prepare_inputs(observations, history_input, [0])

    assert prepared["videos_list"] == videos
    assert prepared["prompt_texts_list"] == [
        [
            "Estimate task-conditioned success potential for this robot "
            "manipulation state. Task: Pick up the cube.. The two synchronized "
            "videos show the same 5-frame history from two camera views."
        ]
    ]


def test_history_vlm_returns_zero_before_first_window(monkeypatch):
    from rlinf.models.embodiment.reward.vlm_reward_model import HistoryVLMRewardModel

    model = object.__new__(HistoryVLMRewardModel)
    model.interval_reward = 0.0
    monkeypatch.setattr(model, "apply_gt_success_bonus", lambda rewards, _: rewards)

    rewards = model.compute_reward(
        {
            "dones": torch.zeros(3, dtype=torch.bool),
            "history_input": {"history_window": {}},
        }
    )

    torch.testing.assert_close(rewards, torch.zeros(3))
