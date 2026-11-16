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

"""Native Cosmos rollout contracts, decoding and replay labels (CPU only)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("cosmos_framework")
from cosmos_framework.data.generator.action.utils import action_processing
from cosmos_framework.data.generator.action.utils.pose_utils import convert_rotation

from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy


class _FakeOmni(torch.nn.Module):
    """Replace the external multi-billion-parameter network, keeping native I/O."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.child = torch.nn.Linear(1, 1)
        self.child.eval()
        self.config = SimpleNamespace(max_action_dim=64, ema=None)
        self.input_video_key = "video"
        self.input_caption_key = "ai_caption"
        self.calls = []
        self.predictions = None
        self.fail = False

    def tokenize_text(
        self, captions, *, pad_token_id, device, is_video=False, use_system_prompt=None
    ):
        assert not is_video
        assert use_system_prompt in (None, False)
        # Unequal prompt lengths exercise replay padding and its validity mask.
        return torch.tensor([[11, 12, 13], [21, 22, pad_token_id]], device=device)[
            : len(captions)
        ]

    def generate_samples_from_batch(self, batch, **kwargs):
        assert not torch.is_grad_enabled()
        assert all(not module.training for module in self.modules())
        self.calls.append((batch, kwargs))
        if self.fail:
            raise RuntimeError("sampler failed")
        return {"action": self.predictions}


@pytest.fixture
def rollout():
    omni = _FakeOmni()
    policy = Cosmos3Policy(omni, rollout_config={"num_action_chunks": 10})
    stats = (
        Path(action_processing.__file__).parents[1]
        / "normalizer_stats/libero_native_frame_wise_relative_rot6d.json"
    )
    normalizer = action_processing.load_action_normalizer(
        "quantile_rot", stats_path=stats, apply_forward_clamp=False, expected_dim=10
    )
    # Non-identity rotations catch row/column rot6d or axis-angle convention bugs.
    raw7 = torch.tensor([0.02, -0.03, 0.01, 0.1, -0.2, 0.3, -1.0]).repeat(2, 16, 1)
    raw7[1, :, :3] *= -1
    raw10 = torch.cat(
        [
            raw7[..., :3],
            convert_rotation(raw7[..., 3:6], "axisangle", "rot6d"),
            raw7[..., 6:],
        ],
        dim=-1,
    )
    normalized = normalizer.normalize_action(raw10)
    omni.predictions = [normalized[0], normalized[1].unsqueeze(0)]
    observations = {
        "main_images": torch.full((2, 256, 256, 3), 25, dtype=torch.uint8),
        "wrist_images": torch.full((2, 256, 256, 3), 200, dtype=torch.uint8),
        "states": torch.zeros(2, 8),
        "task_descriptions": ["pick the cup", "put the red cup on the table"],
    }
    return policy, observations, raw7, normalized


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_native_rollout_decodes_actions_and_keeps_model_labels(rollout, mode):
    policy, observations, raw7, normalized = rollout
    original_modes = [module.training for module in policy.omni.modules()]
    actions, result = policy.predict_action_batch(observations, mode=mode, seed=[7, 9])

    torch.testing.assert_close(actions, raw7[:, :10], atol=2e-6, rtol=2e-5)
    assert actions.shape == (2, 10, 7)
    assert not actions.requires_grad
    assert result["prev_logprobs"] is None and result["prev_values"] is None
    model_actions = result["model_actions"]
    assert model_actions.shape == (2, 16, 64)
    torch.testing.assert_close(model_actions[..., :10], normalized)
    assert torch.count_nonzero(model_actions[..., 10:]) == 0
    inputs = result["forward_inputs"]
    assert all(isinstance(value, torch.Tensor) for value in inputs.values())
    assert inputs["tokenized_prompt"].shape == (2, 1024)
    assert inputs["tokenized_prompt_mask"].sum(dim=1).tolist() == [3, 2]
    assert torch.count_nonzero(inputs["tokenized_prompt"][1, 2:]) == 0
    torch.testing.assert_close(inputs["model_action"], model_actions.flatten(1))
    torch.testing.assert_close(inputs["action"], actions.flatten(1))
    torch.testing.assert_close(inputs["observation/image"], observations["main_images"])
    assert [module.training for module in policy.omni.modules()] == original_modes

    batch, settings = policy.omni.calls[0]
    assert settings == {
        "seed": [7, 9],
        "num_steps": 30,
        "guidance": 1.0,
        "has_negative_prompt": False,
    }
    assert batch["mode"] == ["wam", "wam"]
    assert [int(fps) for fps in batch["conditioning_fps"]] == [20, 20]
    for index, item in enumerate(batch["video"]):
        video = item[0]
        assert video.shape[1] == 17
        assert video.dtype == torch.uint8
        # Native bucket resizing/padding keeps the left-to-right camera order.
        assert video[:, :, 0, 0].eq(25).all()
        assert video[:, :, 0, video.shape[-1] * 3 // 4].eq(200).all()
        assert batch["action_processing_record"][index].action_normalizer is None
        assert batch["action_processing_record"][index].raw_action_dim == 10
        caption = json.loads(batch["ai_caption"][index])
        assert observations["task_descriptions"][index] in json.dumps(caption)
        assert "wrist-mounted camera" in json.dumps(caption)

    # The executed chunk and observations cannot mutate the stored expert labels.
    actions.zero_()
    observations["main_images"].zero_()
    torch.testing.assert_close(
        inputs["action"], raw7[:, :10].flatten(1), atol=2e-6, rtol=2e-5
    )
    assert inputs["observation/image"].eq(25).all()


def test_single_camera_and_shared_seed(rollout):
    policy, observations, _, _ = rollout
    observations.pop("wrist_images")
    observations["main_images"] = (
        observations["main_images"].permute(0, 3, 1, 2).float() / 255
    )
    observations["task_descriptions"] = "pick the cup"
    _, result = policy.predict_action_batch(
        observations, seed=42, num_steps=2, guidance=1.5
    )
    batch, settings = policy.omni.calls[0]
    assert settings["seed"] == [42, 42]
    assert settings["num_steps"] == 2 and settings["guidance"] == 1.5
    assert "observation/wrist_image" not in result["forward_inputs"]
    assert result["forward_inputs"]["observation/image"].shape == (2, 256, 256, 3)
    assert "wrist-mounted camera" not in batch["ai_caption"][0]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"seed": [1]}, "integer seeds"),
        ({"num_steps": 0}, "steps > 0"),
        ({"mode": "invalid"}, "train or eval"),
    ],
)
def test_invalid_inference_settings_fail_before_sampling(rollout, change, message):
    policy, observations, _, _ = rollout
    with pytest.raises(ValueError, match=message):
        policy.predict_action_batch(observations, **change)
    assert not policy.omni.calls


def test_missing_or_mismatched_observations_fail_before_sampling(rollout):
    policy, observations, _, _ = rollout
    observations["wrist_images"] = observations["wrist_images"][:1]
    with pytest.raises(ValueError, match="same batch size"):
        policy.predict_action_batch(observations)
    observations.pop("wrist_images")
    observations.pop("task_descriptions")
    with pytest.raises(ValueError, match="task descriptions"):
        policy.predict_action_batch(observations)
    assert not policy.omni.calls


def test_sampler_failure_restores_training_modes(rollout):
    policy, observations, _, _ = rollout
    policy.omni.fail = True
    original_modes = [module.training for module in policy.omni.modules()]
    with pytest.raises(RuntimeError, match="sampler failed"):
        policy.predict_action_batch(observations)
    assert [module.training for module in policy.omni.modules()] == original_modes


@pytest.mark.parametrize("invalid", ["width", "nan"])
def test_invalid_sampler_output_is_rejected(rollout, invalid):
    policy, observations, _, _ = rollout
    if invalid == "width":
        policy.omni.predictions[0] = torch.zeros(16, 7)
    else:
        policy.omni.predictions[0][0, 0] = float("nan")
    with pytest.raises(RuntimeError, match="normalized action|non-finite"):
        policy.predict_action_batch(observations)


@pytest.mark.parametrize("invalid_key", [None, "missing", "unexpected"])
def test_inference_checkpoint_loads_main_weights_and_checks_them(tmp_path, invalid_key):
    from rlinf.models.embodiment.cosmos3 import load_inference_checkpoint

    source = Cosmos3Policy(_FakeOmni(), ema_enabled=False)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(3.0)
    checkpoint = dict(source.state_dict())
    checkpoint["omni.net_ema.anchor"] = torch.full((1,), 7.0)
    if invalid_key == "missing":
        checkpoint.pop("omni.anchor")
    elif invalid_key == "unexpected":
        checkpoint["omni.unexpected"] = torch.zeros(1)
    path = tmp_path / "full_weights.pt"
    torch.save(checkpoint, path)
    target = Cosmos3Policy(_FakeOmni(), ema_enabled=False)
    if invalid_key is not None:
        with pytest.raises(RuntimeError, match="Missing key|Unexpected key"):
            load_inference_checkpoint(target, str(path))
    else:
        load_inference_checkpoint(target, str(path))
        for key, value in source.state_dict().items():
            torch.testing.assert_close(target.state_dict()[key], value)


@pytest.mark.parametrize("mode", ["wam", "policy"])
def test_rollout_layout_keeps_student_policy_and_expert_wam_distinct(rollout, mode):
    policy, observations, _, _ = rollout
    policy._rollout_config["action_model_mode"] = mode
    policy.predict_action_batch(observations)
    data, _ = policy.omni.calls[0]
    for videos, plan in zip(data["video"], data["sequence_plan"], strict=True):
        assert videos[0].shape[1] == (1 if mode == "policy" else 17)
        assert plan.condition_frame_indexes_vision == [0]
        assert plan.condition_frame_indexes_action == []
    assert data["mode"] == [mode, mode]


@pytest.mark.parametrize(
    "mode, objective", [("wam", "joint"), ("policy", "action_only")]
)
def test_sft_objective_preserves_joint_and_disables_only_vision(mode, objective):
    from rlinf.models.embodiment.cosmos3 import _configure_sft_objective

    rf = SimpleNamespace(
        loss_scale=10.0,
        image_loss_scale=None,
        action_loss_weight=7.0,
        normalize_loss_by_active=False,
    )
    _configure_sft_objective(
        SimpleNamespace(rectified_flow_training_config=rf),
        {"action_model_mode": mode, "sft_objective": objective},
    )
    assert rf.action_loss_weight == 7.0
    if objective == "action_only":
        assert rf.loss_scale == rf.image_loss_scale == 0.0
        assert rf.normalize_loss_by_active
    else:
        assert rf.loss_scale == 10.0 and rf.image_loss_scale is None
        assert not rf.normalize_loss_by_active


@pytest.mark.parametrize("weight", [0.0, -1.0, float("nan"), float("inf")])
def test_action_only_rejects_invalid_action_weight(weight):
    from rlinf.models.embodiment.cosmos3 import _configure_sft_objective

    config = SimpleNamespace(
        rectified_flow_training_config=SimpleNamespace(
            action_loss_weight=weight,
            normalize_loss_by_active=False,
        )
    )
    with pytest.raises(ValueError, match="positive action_loss_weight"):
        _configure_sft_objective(
            config,
            {"action_model_mode": "policy", "sft_objective": "action_only"},
        )


def test_action_only_rejects_wam_before_changing_vision_weights():
    from rlinf.models.embodiment.cosmos3 import _configure_sft_objective

    rf = SimpleNamespace(loss_scale=10.0, image_loss_scale=None)
    with pytest.raises(ValueError, match="policy"):
        _configure_sft_objective(
            SimpleNamespace(rectified_flow_training_config=rf),
            {"action_model_mode": "wam", "sft_objective": "action_only"},
        )
    assert rf.loss_scale == 10.0 and rf.image_loss_scale is None


def test_restoring_optimizer_step_does_not_update_ema():
    policy = Cosmos3Policy(_FakeOmni())
    policy._ema_enabled = True
    updated = []
    policy._update_ema = updated.append
    policy.set_global_step(1, update_ema=False)
    assert updated == [] and policy._global_step == 1
    policy.set_global_step(2)
    assert updated == [1] and policy._global_step == 2


class _TrainingOmni(_FakeOmni):
    """Replace the external network while exercising native flow loss and RLinf I/O."""

    def _encode_vision_item(self, state, *args, **kwargs):
        return state

    def training_step(self, data, iteration):
        from cosmos_framework.model.generator.algorithm.loss.flow_matching import (
            compute_flow_matching_loss,
        )

        self.last_iteration = iteration
        actions = [items[0] for items in data["action"]]
        masks = []
        for action, plan in zip(actions, data["sequence_plan"], strict=True):
            mask = action.new_zeros(action.shape[0], 1)
            mask[plan.condition_frame_indexes_action] = 1
            masks.append(mask)
        prediction = [self.anchor.expand_as(action) for action in actions]
        target = [torch.ones_like(action) - action for action in actions]
        unit_weight = SimpleNamespace(
            train_time_weight=lambda time, kwargs: torch.ones_like(time).to(**kwargs)
        )
        loss, _ = compute_flow_matching_loss(
            pred=prediction,
            target=target,
            condition_mask=masks,
            timesteps=torch.zeros(len(actions), 1),
            has_valid_tokens=True,
            rectified_flow=unit_weight,
            tensor_kwargs_fp32={"device": "cpu", "dtype": torch.float32},
            raw_action_dim=data["raw_action_dim"],
            normalize_by_active=True,
        )
        return {"flow_matching_loss_action": loss}, loss


def test_lerobot_sft_forward_keeps_native_loss_graph_and_updates_parameters(rollout):
    _, observations, raw7, _ = rollout
    omni = _TrainingOmni()
    omni.config.rectified_flow_training_config = SimpleNamespace(
        loss_scale=0.0,
        image_loss_scale=0.0,
        action_loss_weight=1.0,
        normalize_loss_by_active=True,
    )
    policy = Cosmos3Policy(
        omni,
        rollout_config={
            "num_action_chunks": 10,
            "action_model_mode": "policy",
            "sft_objective": "action_only",
        },
    )
    pad = torch.zeros(2, 16, dtype=torch.bool)
    pad[1, 12:] = True
    batch = {
        "image": observations["main_images"],
        "wrist_image": observations["wrist_images"],
        "task": observations["task_descriptions"],
        "actions": raw7,
        "actions_is_pad": pad,
    }
    data = policy.prepare_lerobot_sft_batch(batch)
    policy.set_global_step(7, update_ema=False)
    output = policy.sft_forward(data)
    assert output["loss"].requires_grad
    assert torch.isfinite(output["loss"])
    assert not output["flow_matching_loss_action"].requires_grad
    assert omni.last_iteration == 7
    output["loss"].backward()
    assert omni.anchor.grad is not None and omni.anchor.grad.abs().sum() > 0
    before = omni.anchor.detach().clone()
    torch.optim.SGD(policy.parameters(), lr=0.01).step()
    assert not torch.equal(omni.anchor.detach(), before)


@pytest.mark.parametrize("mode", ["wam", "policy"])
@pytest.mark.parametrize("reduction_api", ["legacy", "sample_aware"])
def test_joint_sft_loss_reduction_is_independent_of_inference_mode(mode, reduction_api):
    from rlinf.models.embodiment.cosmos3 import _configure_sft_objective

    rf = SimpleNamespace(
        loss_scale=10.0,
        image_loss_scale=None,
        action_loss_weight=10.0,
        normalize_loss_by_active=True,
    )
    if reduction_api == "legacy":
        rf.sample_level_loss_averaging = True
    else:
        rf.loss_reduction_mode = "global_sample_mean"
    _configure_sft_objective(
        SimpleNamespace(rectified_flow_training_config=rf),
        {"action_model_mode": mode, "sft_objective": "joint"},
    )
    assert not rf.normalize_loss_by_active
    if reduction_api == "legacy":
        assert not rf.sample_level_loss_averaging
    else:
        assert rf.loss_reduction_mode == "local_item_mean"
    assert rf.loss_scale == rf.action_loss_weight == 10.0
    assert rf.image_loss_scale is None
