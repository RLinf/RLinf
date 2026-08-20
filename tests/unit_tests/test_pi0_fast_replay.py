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

import math

import pytest
import torch

import rlinf.models.embodiment.pi0_fast.pi0_fast_action_model as action_model_module
from rlinf.models.embodiment.pi0_fast.fast_replay import (
    _preprocess_images,
    _sample_next_token,
    build_action_sequence_metadata,
    compute_token_logprobs,
    replay_action_logits,
    safe_detokenize_actions,
)
from rlinf.models.embodiment.pi0_fast.pi0_fast_action_model import (
    PI0FastForRLActionPrediction,
)


class _FakeReplayPolicy(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self._logits = logits
        self.model = type(
            "FakeModel",
            (),
            {"config": type("FakeConfig", (), {"hidden_size": 4})()},
        )()


class _FakeImageFeaturePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)
        self.config = type(
            "FakeConfig",
            (),
            {
                "image_features": {
                    "observation.images.base_0_rgb": object(),
                    "observation.images.left_wrist_0_rgb": object(),
                    "observation.images.empty_camera_0": object(),
                },
                "image_resolution": (224, 224),
            },
        )()
        self.model = type(
            "FakeModel",
            (),
            {"config": type("FakeModelConfig", (), {"hidden_size": 4})()},
        )()


class _FakeTokenizer:
    bos_token_id = 1

    def __call__(
        self,
        prompts,
        *,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        del padding, truncation, return_tensors
        return {
            "input_ids": torch.zeros(len(prompts), max_length, dtype=torch.long),
            "attention_mask": torch.ones(len(prompts), max_length, dtype=torch.long),
        }


class _FakeNativePolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(1, 1)
        self.config = type(
            "FakeConfig",
            (),
            {
                "image_features": {},
                "max_state_dim": 8,
                "tokenizer_max_length": 4,
            },
        )()
        self.model = type(
            "FakeModel",
            (),
            {"config": type("FakeModelConfig", (), {"hidden_size": 4})()},
        )()
        self._paligemma_tokenizer = _FakeTokenizer()
        self.predict_calls = 0

    def predict_action_chunk(self, batch):
        self.predict_calls += 1
        batch_size = batch["observation.state"].shape[0]
        return torch.ones(batch_size, 10, 7)


class _FakeActionBpeTokenizer:
    def get_vocab_size(self):
        return 16

    def decode(self, action_ids):
        if any(action_id < 0 or action_id >= 16 for action_id in action_ids):
            raise ValueError("invalid action token")
        return "decoded action"


class _FakeActionSequenceTokenizer:
    vocab_size = 1000

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        if text == "Action: ":
            return [10, 11]
        if text == "|":
            return [12]
        raise AssertionError(f"unexpected text: {text}")


class _FakeActionSequencePolicy:
    def __init__(self):
        self._paligemma_tokenizer = _FakeActionSequenceTokenizer()
        self.action_tokenizer = type(
            "FakeActionTokenizer",
            (),
            {"bpe_tokenizer": _FakeActionBpeTokenizer()},
        )()
        self.config = type("FakeConfig", (), {"fast_skip_tokens": 0})()
        self.detokenize_calls = 0

    def detokenize_actions(self, tokens, *, action_horizon, action_dim):
        self.detokenize_calls += tokens.shape[0]
        return torch.ones(tokens.shape[0], action_horizon, action_dim)


class _ReplayLMHead(torch.nn.Module):
    def forward(self, hidden_states):
        return torch.cat([hidden_states + offset for offset in range(4)], dim=-1)


class _ReplayLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        attention = type(
            "FakeAttention",
            (),
            {"q_proj": torch.nn.Linear(1, 1, bias=False)},
        )()
        self.layers = [type("FakeLayer", (), {"self_attn": attention})()]


class _ReplayPaliGemma(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = _ReplayLanguageModel()
        self.lm_head = _ReplayLMHead()


class _ReplayPaliGemmaWithExpert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma = _ReplayPaliGemma()
        self.forward_calls = 0

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        self.forward_calls += 1
        return (inputs_embeds[0], None), None


class _TeacherForcingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = _ReplayPaliGemmaWithExpert()

    def embed_prefix_fast(
        self,
        images,
        image_masks,
        tokens,
        token_masks,
        *,
        fast_action_tokens,
        fast_action_masks,
    ):
        del images, image_masks
        embeddings = tokens.float().unsqueeze(-1)
        pad_masks = token_masks
        num_fast_embs = 0
        if fast_action_tokens is not None:
            fast_embeddings = fast_action_tokens.float().unsqueeze(-1)
            embeddings = torch.cat([embeddings, fast_embeddings], dim=1)
            pad_masks = torch.cat([pad_masks, fast_action_masks], dim=1)
            num_fast_embs = fast_action_tokens.shape[1]
        attention_masks = pad_masks[:, None, :] & pad_masks[:, :, None]
        return embeddings, pad_masks, attention_masks, None, num_fast_embs

    def _prepare_attention_masks_4d(self, attention_masks, *, dtype):
        del dtype
        return attention_masks


class _TeacherForcingPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.model = _TeacherForcingModel()
        self.config = type(
            "FakeConfig",
            (),
            {
                "image_features": {"observation.images.image": object()},
                "image_resolution": (2, 2),
            },
        )()
        self._paligemma_tokenizer = type("FakeTokenizer", (), {"bos_token_id": 9})()


def test_compute_token_logprobs_skips_entropy_when_disabled():
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
    action_tokens = torch.tensor([[2, 0]])
    action_token_mask = torch.tensor([[True, True]])

    logprobs, entropy = compute_token_logprobs(
        logits,
        action_tokens,
        action_token_mask,
        compute_entropy=False,
    )

    expected = (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, action_tokens.unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(logprobs, expected)
    assert entropy is None


def test_sample_next_token_uses_argmax_when_sampling_is_disabled(monkeypatch):
    logits = torch.tensor([[1.0, 3.0, 2.0]])

    def fail_multinomial(*args, **kwargs):
        raise AssertionError("greedy decoding must not call torch.multinomial")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    token, logprob = _sample_next_token(
        logits,
        temperature=0.3,
        do_sample=False,
    )

    assert token.item() == 1
    assert torch.allclose(logprob, torch.log_softmax(logits, dim=-1)[:, 1])


def test_native_action_sequence_mask_stops_after_first_end_marker():
    policy = _FakeActionSequencePolicy()
    tokens = torch.tensor(
        [
            [10, 11, 996, 12, 400, 401],
            [10, 11, 996, 995, 994, 993],
        ]
    )

    metadata = build_action_sequence_metadata(policy, tokens)

    assert metadata["prefix_valid"].tolist() == [True, True]
    assert metadata["end_marker_present"].tolist() == [True, False]
    assert metadata["body_decode_valid"].tolist() == [True, False]
    assert metadata["action_logprob_mask"].tolist() == [
        [True, True, True, True, False, False],
        [True, True, True, True, True, True],
    ]


def test_replay_action_logits_uses_one_teacher_forced_forward_without_shift():
    policy = _TeacherForcingPolicy()
    forward_inputs = {
        "observation.images.image": torch.zeros(1, 3, 2, 2),
        "observation.language.tokens": torch.tensor([[7, 8]]),
        "observation.language.attention_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    action_tokens = torch.tensor([[1, 2, 3]])
    action_token_mask = torch.ones_like(action_tokens, dtype=torch.bool)

    logits, final_hidden = replay_action_logits(
        policy, forward_inputs, action_tokens, action_token_mask
    )

    expected_hidden = torch.tensor([[[9.0], [1.0], [2.0]]])
    expected_logits = torch.cat(
        [expected_hidden + offset for offset in range(4)], dim=-1
    )
    assert policy.model.paligemma_with_expert.forward_calls == 1
    assert torch.equal(logits, expected_logits)
    assert torch.equal(final_hidden, torch.tensor([[2.0]]))


def test_native_action_sequence_mask_keeps_invalid_samples_for_failure_signal():
    policy = _FakeActionSequencePolicy()
    tokens = torch.tensor(
        [
            [10, 11, 500, 12, 0, 0],
            [10, 11, 996, 995, 994, 0],
        ]
    )
    generation_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )

    metadata = build_action_sequence_metadata(
        policy, tokens, generation_mask=generation_mask
    )

    assert metadata["body_decode_valid"].tolist() == [False, False]
    assert torch.equal(metadata["action_logprob_mask"], generation_mask)


def test_safe_detokenize_executes_zero_action_for_invalid_sequences():
    policy = _FakeActionSequencePolicy()
    tokens = torch.tensor(
        [
            [10, 11, 996, 12, 400],
            [9, 11, 996, 12, 400],
            [10, 11, 996, 995, 994],
            [10, 11, 500, 12, 400],
        ]
    )

    actions, metadata = safe_detokenize_actions(
        policy, tokens, action_horizon=2, action_dim=3
    )

    assert policy.detokenize_calls == 1
    assert metadata["prefix_valid"].tolist() == [True, False, True, True]
    assert metadata["end_marker_present"].tolist() == [True, True, False, True]
    assert metadata["decode_valid"].tolist() == [True, False, False, False]
    assert torch.equal(actions[0], torch.ones(2, 3))
    assert torch.count_nonzero(actions[1:]) == 0


def test_default_forward_replays_cached_action_tokens(monkeypatch):
    logits = torch.zeros(2, 5, 16)
    model = PI0FastForRLActionPrediction(
        _FakeReplayPolicy(logits),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=5,
    )
    monkeypatch.setattr(
        model,
        "_replay_action_logits",
        lambda *args: (logits, torch.zeros(logits.shape[0], 4)),
    )
    forward_inputs = {
        "action_tokens": torch.tensor(
            [[1, 2, 3, 0, 0], [4, 5, 0, 0, 0]], dtype=torch.long
        ),
        "action_token_mask": torch.tensor(
            [
                [True, True, True, False, False],
                [True, True, False, False, False],
            ]
        ),
    }

    out = model.default_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
        compute_values=True,
    )

    assert out["logprobs"].shape == (2, 5)
    assert out["entropy"].shape == (2, 5)
    assert torch.equal(out["logprob_mask"], forward_inputs["action_token_mask"])
    assert out["values"] is None


def test_generate_action_tokens_forwards_temperature_to_fallback(monkeypatch):
    captured = {}

    def fake_generate_action_tokens_with_logprobs(
        policy,
        batch,
        *,
        max_action_tokens,
        num_action_chunks,
        action_dim,
        temperature,
        do_sample,
        compute_logprobs,
    ):
        captured.update(
            {
                "policy": policy,
                "batch": batch,
                "max_action_tokens": max_action_tokens,
                "num_action_chunks": num_action_chunks,
                "action_dim": action_dim,
                "temperature": temperature,
                "do_sample": do_sample,
                "compute_logprobs": compute_logprobs,
            }
        )
        return {
            "actions": torch.zeros(1, num_action_chunks, action_dim),
            "action_tokens": torch.zeros(1, max_action_tokens, dtype=torch.long),
            "action_token_mask": torch.ones(1, max_action_tokens, dtype=torch.bool),
            "action_logprob_mask": torch.ones(1, max_action_tokens, dtype=torch.bool),
        }

    monkeypatch.setattr(
        action_model_module,
        "generate_action_tokens_with_logprobs",
        fake_generate_action_tokens_with_logprobs,
    )
    policy = torch.nn.Linear(1, 1)
    model = PI0FastForRLActionPrediction(
        policy,
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=5,
    )
    batch = {"observation.state": torch.zeros(1, 8)}

    out = model._generate_action_tokens_with_logprobs(batch, temperature=0.42)

    assert out["actions"].shape == (1, 10, 7)
    assert captured == {
        "policy": policy,
        "batch": batch,
        "max_action_tokens": 5,
        "num_action_chunks": 10,
        "action_dim": 7,
        "temperature": 0.42,
        "do_sample": True,
        "compute_logprobs": True,
    }


def test_default_forward_uses_action_logprob_mask_when_present(monkeypatch):
    logits = torch.zeros(1, 3, 4)
    model = PI0FastForRLActionPrediction(
        _FakeReplayPolicy(logits),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=3,
    )
    monkeypatch.setattr(
        model,
        "_replay_action_logits",
        lambda *args: (logits, torch.zeros(logits.shape[0], 4)),
    )
    forward_inputs = {
        "action_tokens": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "action_token_mask": torch.tensor([[True, True, True]]),
        "action_logprob_mask": torch.tensor([[False, True, False]]),
    }

    out = model.default_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        compute_entropy=True,
    )

    assert torch.allclose(out["logprobs"], torch.tensor([[0.0, -math.log(4), 0.0]]))
    assert torch.allclose(out["entropy"], torch.tensor([[0.0, math.log(4), 0.0]]))
    assert torch.equal(out["logprob_mask"], forward_inputs["action_logprob_mask"])


def test_default_forward_computes_logprobs_with_sampling_temperature(monkeypatch):
    logits = torch.tensor([[[0.0, 2.0], [2.0, 0.0]]])
    model = PI0FastForRLActionPrediction(
        _FakeReplayPolicy(logits),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=2,
    )
    monkeypatch.setattr(
        model,
        "_replay_action_logits",
        lambda *args: (logits, torch.zeros(logits.shape[0], 4)),
    )
    forward_inputs = {
        "action_tokens": torch.tensor([[1, 0]], dtype=torch.long),
        "action_token_mask": torch.tensor([[True, True]]),
    }

    out = model.default_forward(
        forward_inputs=forward_inputs,
        compute_logprobs=True,
        temperature=2.0,
    )

    expected = (
        torch.log_softmax(logits / 2.0, dim=-1)
        .gather(-1, forward_inputs["action_tokens"].unsqueeze(-1))
        .squeeze(-1)
    )
    assert torch.allclose(out["logprobs"], expected)


def test_predict_action_batch_preserves_generated_token_logprobs(monkeypatch):
    model = PI0FastForRLActionPrediction(
        _FakeNativePolicy(),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=4,
    )
    expected_logprobs = torch.tensor([[-0.1, -0.2, 0.0, 0.0]])

    def fake_generate(
        batch,
        *,
        temperature,
        do_sample,
        max_action_tokens,
        compute_logprobs,
    ):
        del batch, temperature, do_sample, max_action_tokens, compute_logprobs
        return {
            "actions": torch.zeros(1, 10, 7),
            "action_tokens": torch.tensor([[11, 12, 0, 0]]),
            "action_token_mask": torch.tensor([[True, True, False, False]]),
            "action_logprob_mask": torch.tensor([[True, True, False, False]]),
            "token_logprobs": expected_logprobs,
        }

    monkeypatch.setattr(model, "_generate_action_tokens_with_logprobs", fake_generate)
    env_obs = {
        "main_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 8),
        "task_descriptions": ["pick up the object"],
    }

    _, result = model.predict_action_batch(
        env_obs,
        mode="train",
        compute_values=False,
        calculate_logprobs=True,
    )

    assert torch.equal(result["prev_logprobs"], expected_logprobs)


def test_predict_action_batch_honors_greedy_sampling_and_token_limit(monkeypatch):
    model = PI0FastForRLActionPrediction(
        _FakeNativePolicy(),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=4,
    )
    captured = {}

    def fake_generate(
        batch,
        *,
        temperature,
        do_sample,
        max_action_tokens,
        compute_logprobs,
    ):
        del batch, compute_logprobs
        captured.update(
            temperature=temperature,
            do_sample=do_sample,
            max_action_tokens=max_action_tokens,
        )
        return {
            "actions": torch.zeros(1, 10, 7),
            "action_tokens": torch.tensor([[11, 12, 13]]),
            "action_token_mask": torch.ones(1, 3, dtype=torch.bool),
            "action_logprob_mask": torch.ones(1, 3, dtype=torch.bool),
            "token_logprobs": torch.zeros(1, 3),
        }

    monkeypatch.setattr(model, "_generate_action_tokens_with_logprobs", fake_generate)
    env_obs = {
        "main_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 8),
        "task_descriptions": ["pick up the object"],
    }

    model.predict_action_batch(
        env_obs,
        mode="train",
        calculate_logprobs=True,
        do_sample=False,
        temperature=1.0,
        max_new_tokens=3,
    )

    assert captured == {
        "temperature": 1.0,
        "do_sample": False,
        "max_action_tokens": 3,
    }


@pytest.mark.parametrize(
    ("sampling_kwargs", "parameter_name"),
    [
        ({"top_k": 10}, "top_k"),
        ({"top_p": 0.9}, "top_p"),
    ],
)
def test_predict_action_batch_rejects_unsupported_sampling_truncation(
    monkeypatch,
    sampling_kwargs,
    parameter_name,
):
    model = PI0FastForRLActionPrediction(
        _FakeNativePolicy(),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=4,
    )
    monkeypatch.setattr(
        model,
        "_generate_action_tokens_with_logprobs",
        lambda *args, **kwargs: {
            "actions": torch.zeros(1, 10, 7),
            "action_tokens": torch.zeros(1, 4, dtype=torch.long),
            "action_token_mask": torch.ones(1, 4, dtype=torch.bool),
            "action_logprob_mask": torch.ones(1, 4, dtype=torch.bool),
            "token_logprobs": torch.zeros(1, 4),
        },
    )
    env_obs = {
        "main_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 8),
        "task_descriptions": ["pick up the object"],
    }

    with pytest.raises(ValueError, match=parameter_name):
        model.predict_action_batch(
            env_obs,
            mode="train",
            calculate_logprobs=True,
            **sampling_kwargs,
        )


def test_predict_action_batch_keeps_invalid_action_zero_after_postprocessing(
    monkeypatch,
):
    model = PI0FastForRLActionPrediction(
        _FakeNativePolicy(),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=4,
        postprocessor=lambda actions: actions + 5,
    )

    def fake_generate(
        batch,
        *,
        temperature,
        do_sample,
        max_action_tokens,
        compute_logprobs,
    ):
        del batch, temperature, do_sample, max_action_tokens, compute_logprobs
        return {
            "actions": torch.zeros(1, 10, 7),
            "action_tokens": torch.tensor([[11, 12, 13, 14]]),
            "action_token_mask": torch.ones(1, 4, dtype=torch.bool),
            "action_logprob_mask": torch.tensor([[True, True, True, False]]),
            "prefix_valid": torch.tensor([True]),
            "end_marker_present": torch.tensor([True]),
            "decode_valid": torch.tensor([False]),
            "token_logprobs": torch.zeros(1, 4),
        }

    monkeypatch.setattr(model, "_generate_action_tokens_with_logprobs", fake_generate)
    env_obs = {
        "main_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "wrist_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
        "states": torch.zeros(1, 8),
        "task_descriptions": ["pick up the object"],
    }

    actions, result = model.predict_action_batch(
        env_obs,
        mode="train",
        compute_values=False,
        calculate_logprobs=True,
    )

    assert torch.count_nonzero(actions) == 0
    assert "pi0_fast_prefix_valid" not in result["forward_inputs"]
    assert "pi0_fast_end_marker_present" not in result["forward_inputs"]
    assert "pi0_fast_decode_valid" not in result["forward_inputs"]


def test_policy_image_keys_are_aliased_from_generic_batch_keys():
    model = PI0FastForRLActionPrediction(
        _FakeImageFeaturePolicy(),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=5,
    )
    main = torch.zeros(1, 3, 224, 224)
    wrist = torch.ones(1, 3, 224, 224)
    batch = {
        "observation.images.image": main,
        "observation.images.image2": wrist,
    }

    out = model._ensure_policy_image_keys(batch)

    assert out["observation.images.base_0_rgb"] is main
    assert out["observation.images.left_wrist_0_rgb"] is wrist
    assert "observation.images.empty_camera_0" not in out


def test_prepare_lerobot_batch_casts_float_inputs_to_model_dtype():
    model = PI0FastForRLActionPrediction(
        _FakeImageFeaturePolicy(),
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=5,
    ).to(dtype=torch.bfloat16)
    batch = {
        "observation.images.image": torch.zeros(1, 3, 224, 224),
        "observation.state": torch.zeros(1, 8),
        "observation.language.tokens": torch.zeros(1, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(1, 4, dtype=torch.bool),
    }

    out = model._prepare_lerobot_batch(batch)

    assert out["observation.images.image"].dtype == torch.bfloat16
    assert out["observation.images.base_0_rgb"].dtype == torch.bfloat16
    assert out["observation.state"].dtype == torch.bfloat16
    assert out["observation.language.tokens"].dtype == torch.long
    assert out["observation.language.attention_mask"].dtype == torch.bool


def test_preprocess_images_casts_to_policy_dtype_after_normalization():
    policy = _FakeImageFeaturePolicy().to(dtype=torch.bfloat16)
    batch = {
        "observation.images.base_0_rgb": torch.zeros(1, 3, 224, 224),
    }

    images, masks = _preprocess_images(policy, batch)

    assert [image.dtype for image in images] == [
        torch.bfloat16,
        torch.bfloat16,
        torch.bfloat16,
    ]
    assert [mask.dtype for mask in masks] == [
        torch.bool,
        torch.bool,
        torch.bool,
    ]


def test_eval_without_logprobs_uses_native_policy_path():
    policy = _FakeNativePolicy()
    model = PI0FastForRLActionPrediction(
        policy,
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=5,
    )
    env_obs = {
        "main_images": torch.zeros(2, 64, 64, 3),
        "wrist_images": torch.zeros(2, 64, 64, 3),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["task a", "task b"],
    }

    actions, result = model.predict_action_batch(
        env_obs,
        mode="eval",
    )

    assert policy.predict_calls == 1
    assert torch.equal(actions, torch.ones(2, 10, 7))
    assert result == {
        "prev_logprobs": None,
        "prev_values": None,
        "forward_inputs": {},
    }


def test_eval_decode_failure_falls_back_to_safe_native_generation(monkeypatch):
    policy = _FakeNativePolicy()

    def fail_native_decode(batch):
        policy.predict_calls += 1
        raise AssertionError("Token sequence does not start with ['Action', ':']")

    monkeypatch.setattr(policy, "predict_action_chunk", fail_native_decode)
    model = PI0FastForRLActionPrediction(
        policy,
        action_dim=7,
        num_action_chunks=10,
        max_action_tokens=5,
    )
    captured = {}

    def safe_generate(
        batch,
        *,
        temperature,
        do_sample,
        max_action_tokens,
        compute_logprobs,
    ):
        captured.update(
            {
                "batch_size": batch["observation.state"].shape[0],
                "temperature": temperature,
                "do_sample": do_sample,
                "max_action_tokens": max_action_tokens,
                "compute_logprobs": compute_logprobs,
            }
        )
        return {
            "actions": torch.ones(2, 10, 7),
            "decode_valid": torch.tensor([True, False]),
        }

    monkeypatch.setattr(model, "_generate_action_tokens_with_logprobs", safe_generate)
    env_obs = {
        "main_images": torch.zeros(2, 64, 64, 3),
        "wrist_images": torch.zeros(2, 64, 64, 3),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["task a", "task b"],
    }

    actions, result = model.predict_action_batch(env_obs, mode="eval")

    assert policy.predict_calls == 1
    assert torch.equal(actions[0], torch.ones(10, 7))
    assert torch.count_nonzero(actions[1]) == 0
    assert captured == {
        "batch_size": 2,
        "temperature": 0.0,
        "do_sample": False,
        "max_action_tokens": 5,
        "compute_logprobs": False,
    }
    assert result == {
        "prev_logprobs": None,
        "prev_values": None,
        "forward_inputs": {},
    }
