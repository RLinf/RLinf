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

"""Cosmos LeRobot adapter contracts with native action utilities (CPU only)."""

import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("cosmos_framework")
from cosmos_framework.data.generator.action.utils import action_processing
from cosmos_framework.data.generator.action.utils.transforms import (
    build_sequence_plan_from_mode,
)

from rlinf.data.datasets.cosmos3.lerobot_batch import prepare_lerobot_batch


@pytest.fixture
def batch():
    actions = torch.tensor([0.02, -0.03, 0.01, 0, 0, math.pi / 2, -1]).repeat(2, 16, 1)
    actions[1, :, :3] *= -1
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[1, 12:] = True
    return {
        "image": torch.full((2, 3, 32, 32), 25, dtype=torch.uint8),
        "wrist_image": torch.full((2, 3, 32, 32), 200, dtype=torch.uint8),
        "state": torch.zeros(2, 8),
        "task": ["pick the cup", "move the cup"],
        "actions": actions,
        "actions_is_pad": pad,
    }


@pytest.fixture
def normalizer():
    stats = (
        Path(action_processing.__file__).parents[1]
        / "normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
    )
    return action_processing.load_action_normalizer(
        "quantile_rot",
        stats_path=stats,
        stats_key="global_raw",
        apply_forward_clamp=False,
        expected_dim=10,
    )


class ObservationBuilder:
    """Replace only observation preprocessing and tokenizer, outside this adapter."""

    def __init__(self, video_key="video"):
        self.video_key = video_key
        self.observations = None
        self.data = None
        self.inputs = None

    def __call__(self, observations):
        self.observations = observations
        images = observations["main_images"]
        videos = torch.cat([images, observations["wrist_images"]], dim=-1)
        record = action_processing.ActionProcessingRecord(
            raw_action_dim=10, action_normalizer=None
        )
        self.data = {
            self.video_key: [[video.unsqueeze(1)] for video in videos],
            "mode": ["policy"] * len(images),
            "sequence_plan": [
                build_sequence_plan_from_mode("policy", 1, 16) for _ in images
            ],
            **action_processing.make_batched_action_processing_fields(
                record, len(images)
            ),
            "conditioning_fps": [torch.tensor(20)] * len(images),
        }
        self.inputs = {
            "tokenized_prompt": torch.tensor([[11, 12, 13], [21, 22, 0]]),
            "tokenized_prompt_mask": torch.tensor(
                [[True, True, True], [True, True, False]]
            ),
        }
        return self.data, self.inputs


def prepare(batch, normalizer, builder=None, **kwargs):
    if builder is None:
        builder = ObservationBuilder()
    return prepare_lerobot_batch(
        batch,
        horizon=16,
        max_action_dim=64,
        normalizer=normalizer,
        prepare_observation_batch=builder,
        **kwargs,
    )


def test_physical_labels_use_rot6d_stats_once_and_preserve_input(batch, normalizer):
    original = batch["actions"].clone()
    builder = ObservationBuilder()
    data = prepare(batch, normalizer, builder)
    actual = torch.stack([item[0] for item in data["action"]])
    # 90 degrees about z: first two rotation-matrix columns, in Cosmos order.
    rotation = torch.tensor([0, 1, 0, -1, 0, 0]).repeat(2, 16, 1)
    raw10 = torch.cat([original[..., :3], rotation, original[..., 6:7]], dim=-1)
    expected = (raw10 - normalizer.offset) / normalizer.scale
    torch.testing.assert_close(actual[0, :, :10], expected[0], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(
        actual[1, :12, :10], expected[1, :12], atol=1e-6, rtol=1e-5
    )
    assert actual.shape == (2, 16, 64)
    assert actual.dtype == torch.float32 and not actual.requires_grad
    assert torch.count_nonzero(actual[..., 10:]) == 0
    torch.testing.assert_close(batch["actions"], original)
    assert builder.observations["main_images"] is batch["image"]
    assert builder.observations["wrist_images"] is batch["wrist_image"]
    assert "states" not in builder.observations
    assert builder.observations["task_descriptions"] is batch["task"]
    assert data["action_processing_record"][0].raw_action_dim == 10
    assert [int(dim) for dim in data["raw_action_dim"]] == [10, 10]
    assert "action_valid_mask" not in data


def test_padding_is_clean_zero_and_does_not_change_vision_or_channel_masks(
    batch, normalizer
):
    builder = ObservationBuilder()
    data = prepare(batch, normalizer, builder)
    assert torch.count_nonzero(data["action"][1][0][12:]) == 0
    assert data["sequence_plan"][0].condition_frame_indexes_action == []
    assert data["sequence_plan"][1].condition_frame_indexes_action == [12, 13, 14, 15]
    assert data["sequence_plan"][1].condition_frame_indexes_vision == [0]
    assert data["sequence_plan"][1].action_start_frame_offset == 1
    # The adapter leaves builder-owned plans alone and adds actions to its copy.
    assert builder.data["sequence_plan"][1].condition_frame_indexes_action == []
    assert "action" not in builder.data
    batch["actions"][1, 12:] = float("nan")
    changed = prepare(batch, normalizer)
    torch.testing.assert_close(changed["action"][1][0], data["action"][1][0])


def test_text_is_ragged_grouped_and_current_video_is_preserved(batch, normalizer):
    builder = ObservationBuilder("image")
    data = prepare(batch, normalizer, builder, video_key="image")
    assert [item[0].tolist() for item in data["text_token_ids"]] == [
        [11, 12, 13],
        [21, 22],
    ]
    assert all(item[0].dtype == torch.int64 for item in data["text_token_ids"])
    assert data["image"] is builder.data["image"]
    assert all(item[0].shape == (3, 1, 32, 64) for item in data["image"])
    assert [int(fps) for fps in data["conditioning_fps"]] == [20, 20]


def test_out_of_range_labels_are_not_clamped(batch, normalizer):
    batch["actions"][..., 0] = 10
    data = prepare(batch, normalizer)
    assert data["action"][0][0][:, 0].min() > 1


@pytest.mark.parametrize("invalid", ["horizon", "width", "missing", "nonfinite"])
def test_bad_labels_fail_before_observation_preprocessing(batch, normalizer, invalid):
    builder = ObservationBuilder()
    if invalid == "horizon":
        batch["actions"] = batch["actions"][:, :10]
    elif invalid == "width":
        batch["actions"] = batch["actions"][..., :6]
    elif invalid == "missing":
        del batch["actions_is_pad"]
    else:
        batch["actions"][0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        prepare(batch, normalizer, builder)
    assert builder.observations is None


@pytest.mark.parametrize("invalid", ["dtype", "shape", "hole", "empty"])
def test_bad_temporal_masks_fail_before_preprocessing(batch, normalizer, invalid):
    builder = ObservationBuilder()
    if invalid == "dtype":
        batch["actions_is_pad"] = batch["actions_is_pad"].float()
    elif invalid == "shape":
        batch["actions_is_pad"] = batch["actions_is_pad"][:, :10]
    elif invalid == "hole":
        batch["actions_is_pad"][0, 2] = True
    else:
        batch["actions_is_pad"][0] = True
    with pytest.raises(ValueError):
        prepare(batch, normalizer, builder)
    assert builder.observations is None


@pytest.mark.parametrize(
    "invalid", ["wam", "future", "noisy", "history", "size", "dim"]
)
def test_invalid_policy_layout_is_rejected(batch, normalizer, invalid):
    builder = ObservationBuilder()

    def invalid_builder(observations):
        data, tokens = builder(observations)
        if invalid == "wam":
            data["mode"] = ["wam", "wam"]
        elif invalid == "future":
            data["video"][0][0] = data["video"][0][0].repeat(1, 17, 1, 1)
        elif invalid in ("noisy", "history"):
            data["sequence_plan"][0] = replace(
                data["sequence_plan"][0],
                **(
                    {"condition_frame_indexes_vision": []}
                    if invalid == "noisy"
                    else {"condition_frame_indexes_action": [0]}
                ),
            )
        elif invalid == "size":
            data["sequence_plan"].pop()
        else:
            data["raw_action_dim"] = [torch.tensor(7)] * 2
        return data, tokens

    with pytest.raises(ValueError):
        prepare(batch, normalizer, invalid_builder)


@pytest.mark.parametrize("invalid", ["size", "mask_dtype", "empty", "negative"])
def test_invalid_prompt_tokens_are_rejected(batch, normalizer, invalid):
    builder = ObservationBuilder()

    def invalid_builder(observations):
        data, inputs = builder(observations)
        if invalid == "size":
            inputs["tokenized_prompt"] = inputs["tokenized_prompt"][:1]
        elif invalid == "mask_dtype":
            inputs["tokenized_prompt_mask"] = inputs["tokenized_prompt_mask"].int()
        elif invalid == "empty":
            inputs["tokenized_prompt_mask"][0] = False
        else:
            inputs["tokenized_prompt"][0, 0] = -1
        return data, inputs

    with pytest.raises(ValueError):
        prepare(batch, normalizer, invalid_builder)


def test_native_flow_loss_excludes_temporal_and_channel_padding():
    from types import SimpleNamespace

    from cosmos_framework.model.generator.algorithm.loss.flow_matching import (
        compute_flow_matching_loss,
    )

    prediction = torch.zeros(16, 64, requires_grad=True)
    target = torch.ones_like(prediction)
    target[12:] = 1000
    target[:, 10:] = 1000
    condition = torch.zeros(16, 1)
    condition[12:] = 1
    unit_weight = SimpleNamespace(
        train_time_weight=lambda time, kwargs: torch.ones_like(time).to(**kwargs)
    )
    loss, per_instance = compute_flow_matching_loss(
        pred=[prediction],
        target=[target],
        condition_mask=[condition],
        timesteps=torch.zeros(1, 1),
        has_valid_tokens=True,
        rectified_flow=unit_weight,
        tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
        raw_action_dim=[torch.tensor(10)],
        normalize_by_active=True,
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))
    torch.testing.assert_close(per_instance, torch.ones(1))
    loss.backward()
    assert torch.count_nonzero(prediction.grad[12:]) == 0
    assert torch.count_nonzero(prediction.grad[:, 10:]) == 0
    torch.testing.assert_close(
        prediction.grad[:12, :10],
        torch.full((12, 10), -2 / 120),
    )


class CaptionRecorder(torch.nn.Module):
    """Capture the string passed to the external tokenizer without model weights."""

    def __init__(self):
        super().__init__()
        from types import SimpleNamespace

        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_action_dim=64)
        self.input_video_key = "video"
        self.captions = []

    def tokenize_text(self, captions, **kwargs):
        self.captions = list(captions)
        self.tokenizer_kwargs = kwargs
        return torch.tensor([[1, 2]] * len(captions), dtype=torch.long)


def make_joint_policy(dropout_rate):
    from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy

    model = CaptionRecorder()
    policy = Cosmos3Policy(
        model,
        ema_enabled=False,
        rollout_config={
            "num_frames": 17,
            "fps": 20,
            "image_size": 256,
            "action_model_mode": "wam",
            "sft_objective": "joint",
            "sft_cfg_dropout_rate": dropout_rate,
        },
    )
    return policy, model


def joint_batch():
    pixels = torch.arange(256, dtype=torch.uint8).view(1, 1, 1, 256)
    pixels = pixels.expand(1, 3, 256, 256).contiguous()
    return {
        "image": pixels,
        "wrist_image": pixels.flip(-1),
        "vision_images": pixels[:, None].expand(-1, 17, -1, -1, -1),
        "vision_wrist_images": pixels.flip(-1)[:, None].expand(-1, 17, -1, -1, -1),
        "vision_valid": torch.ones(1, dtype=torch.bool),
        "task": ["pick the cup"],
        "actions": torch.zeros(1, 16, 7),
        "actions_is_pad": torch.zeros(1, 16, dtype=torch.bool),
    }


def test_joint_preprocessing_matches_native_sft_video_caption_and_actions():
    import json

    from cosmos_framework.data.generator.action.datasets.libero_lerobot_dataset import (
        LIBEROLeRobotDataset,
    )
    from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation
    from cosmos_framework.data.generator.action.utils.transforms import (
        ActionTransformPipeline,
    )

    batch = joint_batch()
    policy, tokenizer = make_joint_policy(0)
    actual = policy.prepare_lerobot_sft_batch(batch)
    raw7 = batch["actions"][0]
    raw10 = torch.cat(
        [
            raw7[:, :3],
            convert_rotation(raw7[:, 3:6], "axisangle", "rot6d"),
            raw7[:, 6:],
        ],
        dim=-1,
    )
    # The native idle detector needs only the physical action spec and fps.
    native_dataset = object.__new__(LIBEROLeRobotDataset)
    native_dataset._fps = 20
    native_dataset._rotation_space = "6d"
    idle = native_dataset._compute_idle_frames(raw10)
    assert idle > 0
    native_video = torch.cat(
        [batch["vision_images"][0], batch["vision_wrist_images"][0]], dim=-1
    ).permute(1, 0, 2, 3)
    sample = {
        "video": native_video,
        "action": policy._get_action_normalizer().normalize_action(raw10),
        "ai_caption": batch["task"][0],
        "conditioning_fps": torch.tensor(20),
        "domain_id": torch.tensor(5),
        "viewpoint": "concat_view",
        "mode": "wam",
        "idle_frames": idle,
        "additional_view_description": (
            "The left half shows the third-person view; the right half "
            "shows the wrist-mounted camera."
        ),
    }
    expected = ActionTransformPipeline(
        tokenizer_config=None, max_action_dim=64, format_prompt_as_json=True
    )(sample, None)
    assert tokenizer.captions == [json.dumps(expected["ai_caption"])]
    assert tokenizer.tokenizer_kwargs["is_video"] is False
    assert tokenizer.tokenizer_kwargs["use_system_prompt"] is False
    torch.testing.assert_close(actual["video"][0][0], expected["video"], rtol=0, atol=0)
    torch.testing.assert_close(
        actual["action"][0][0], expected["action"], rtol=0, atol=0
    )
    torch.testing.assert_close(actual["image_size"][0], expected["image_size"])
    assert actual["sequence_plan"][0] == expected["sequence_plan"]


@pytest.mark.parametrize(
    "dropout_rate,random_value,dropped",
    [(0.1, 0.05, True), (0.1, 0.15, False), (1.0, 0.99, True), (0.0, 0.0, False)],
)
def test_sft_caption_dropout_matches_native_probability(
    monkeypatch, dropout_rate, random_value, dropped
):
    import random

    monkeypatch.setattr(random, "random", lambda: random_value)
    policy, tokenizer = make_joint_policy(dropout_rate)
    prepared = policy.prepare_lerobot_sft_batch(joint_batch())
    assert (tokenizer.captions[0] == "") is dropped
    assert prepared["mode"] == ["wam"]
    assert prepared["video"][0][0].shape == (3, 17, 192, 320)
    assert prepared["action"][0][0].shape == (16, 64)


def test_sft_tokens_use_native_truncation_without_rollout_length_limit(monkeypatch):
    from cosmos_framework.data.generator.augmentors.text_tokenizer import (
        _MAX_NUM_TOKENS,
    )

    policy, tokenizer = make_joint_policy(0)
    policy._rollout_config["max_prompt_length"] = 16
    ids = torch.arange(_MAX_NUM_TOKENS + 10, dtype=torch.long).unsqueeze(0)
    monkeypatch.setattr(tokenizer, "tokenize_text", lambda *args, **kwargs: ids)
    prepared = policy.prepare_lerobot_sft_batch(joint_batch())
    assert prepared["text_token_ids"][0][0].tolist() == list(range(_MAX_NUM_TOKENS))
