import sys
import types

import pytest
import torch

from rlinf.algorithms.rewards.top_reward.top_reward import TOPReward


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeTokenizer:
    eos_token = "<|im_end|>"


class _FakeProcessor:
    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.last_text = None
        self.last_messages = None
        self.last_images = None
        self.last_videos = None
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_messages = messages
        return "<|im_start|>user\nPROMPT<|im_end|>\n"

    def __call__(
        self,
        *,
        text,
        images,
        videos,
        padding,
        return_tensors,
        **kwargs,
    ):
        self.last_text = text[0]
        self.last_images = images
        self.last_videos = videos
        self.calls.append(text[0])
        input_ids = torch.tensor([[10, 11, 12, 2]], dtype=torch.long)
        attention_mask = torch.tensor([[1, 1, 1, 1]], dtype=torch.long)
        return _FakeBatch(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        logits = torch.zeros((1, 4, 16), dtype=torch.float32)
        return types.SimpleNamespace(logits=logits)


class _ScoringFakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        logits = torch.zeros((1, 4, 16), dtype=torch.float32)
        logits[0, 2, 2] = 9.0
        return types.SimpleNamespace(logits=logits)


def test_top_reward_matches_reference_qwen_prompt_and_video_path(monkeypatch):
    fake_qwen_utils = types.SimpleNamespace(
        process_vision_info=lambda messages: (None, ["dummy-video"])
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_utils)

    processor = _FakeProcessor()
    reward = TOPReward(
        {"reward_scale": 1.0, "top_reward_max_frames": 16},
        model=_FakeModel(),
        processor=processor,
    )

    frames = [torch.zeros((8, 8, 3), dtype=torch.uint8).numpy() for _ in range(4)]
    score = reward.compute_score(frames, "pick and place")

    assert processor.last_messages[0]["content"][0]["type"] == "video"
    assert processor.last_images is None
    assert processor.last_videos == ["dummy-video"]
    assert processor.last_text == "<|im_start|>user\nPROMPTpick and place Decide whether the above statement is True or not. The answer is: True"
    assert isinstance(score, float)


def test_top_reward_scores_final_token_like_reference(monkeypatch):
    fake_qwen_utils = types.SimpleNamespace(
        process_vision_info=lambda messages: (None, ["dummy-video"])
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_utils)

    processor = _FakeProcessor()
    reward = TOPReward(
        {"reward_scale": 1.0, "top_reward_max_frames": 16},
        model=_ScoringFakeModel(),
        processor=processor,
    )

    frames = [torch.zeros((8, 8, 3), dtype=torch.uint8).numpy() for _ in range(4)]
    score = reward.compute_score(frames, "pick and place")

    expected = torch.log_softmax(torch.tensor([0.0] * 2 + [9.0] + [0.0] * 13), dim=0)[2].item()
    assert score == pytest.approx(expected, abs=1e-6)


def test_top_reward_trims_chat_eos_like_reference(monkeypatch):
    fake_qwen_utils = types.SimpleNamespace(
        process_vision_info=lambda messages: (None, ["dummy-video"])
    )
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_utils)

    processor = _FakeProcessor()
    reward = TOPReward(
        {"reward_scale": 1.0, "top_reward_max_frames": 16},
        model=_FakeModel(),
        processor=processor,
    )

    frames = [torch.zeros((8, 8, 3), dtype=torch.uint8).numpy() for _ in range(4)]
    reward.compute_score(frames, "pick and place")

    assert "<|im_end|>" not in processor.calls[0]
