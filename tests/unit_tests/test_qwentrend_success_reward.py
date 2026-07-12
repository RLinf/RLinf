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
