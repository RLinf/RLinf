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
from pathlib import Path

import numpy as np
import pytest
import torch

from rlinf.algorithms.pear_score_centering import score_centering_correction
from rlinf.algorithms.utils import preprocess_loss_inputs
from rlinf.envs.sim.libero.vlm_controller import (
    ActionProgress,
    apply_deltas,
    gripper_command,
    matrix_to_rpy,
    native_grip_direction,
    pd_offsets,
    rpy_to_matrix,
    vector_to_action,
)
from rlinf.envs.sim.libero.vlm_controller import (
    execute_action as execute_libero_action,
)
from rlinf.envs.sim.robodojo.vlm_action import execute_action, validate_action
from rlinf.models.embodiment.qwen_vlm.policy import (
    QwenVLMPolicy,
    response_logprobs_from_logits,
)
from rlinf.models.embodiment.qwen_vlm.protocol import (
    InvalidAction,
    action_to_vector,
    parse_action,
)
from rlinf.utils.nested_dict_process import pad_stack_list_of_dict_tensor
from rlinf.utils.utils import preprocess_embodied_batch


def _action_text(**arm):
    values = {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0, **arm}
    return (
        "thinking</think>\n```json\n"
        '{"arm": {"x": %s, "y": %s, "z": %s, "roll": %s, "pitch": %s, "yaw": %s}, '
        '"gripper": {"width": 1.0}}\n```'
        % tuple(values[name] for name in ("x", "y", "z", "roll", "pitch", "yaw"))
    )


def test_json_action_uses_the_last_block_after_thinking():
    text = '```json\n{"arm": {}}\n```</think>\n' + _action_text(z=0.3)
    action = parse_action(text)
    assert action["arm"]["z"] == 0.3
    assert action_to_vector(action)[-1] == 1.0


@pytest.mark.parametrize(
    "text",
    [
        "no json here",
        '```json\n{"arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, '
        '"gripper": {"width": NaN}}\n```',
        '```json\n{"arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0, '
        '"extra": 1}, "gripper": {"width": 0}}\n```',
        '```json\n{"arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, '
        '"gripper": {"width": true}}\n```',
        '```json\n{"arm": {"x": 0, "y": 0, "z": 0, "roll": 0, "pitch": 0, "yaw": 0}, '
        '"gripper": {}}\n```',
    ],
)
def test_invalid_json_is_not_a_zero_motion(text):
    with pytest.raises(InvalidAction):
        parse_action(text)
    assert action_to_vector(None)[-1] == 0.0
    assert vector_to_action(action_to_vector(None)) is None


def test_relative_pose_and_gripper_semantics():
    measured = {"x": 1.0, "y": 2.0, "z": 3.0, "roll": 0.1, "pitch": 0.2, "yaw": 0.3}
    action = parse_action(_action_text(x=0.5, yaw=-0.1))
    targets = apply_deltas(measured, action)
    assert targets["x"] == pytest.approx(1.5)
    assert targets["yaw"] == pytest.approx(0.2)
    assert gripper_command(1.0) == 1.0
    assert gripper_command(-0.2) == -1.0
    assert gripper_command(0.0) is None
    matrix = rpy_to_matrix(torch.tensor([0.2, -0.3, 0.4]).numpy())
    assert matrix_to_rpy(matrix).tolist() == pytest.approx([0.2, -0.3, 0.4])


def test_pd_grip_and_stall():
    offsets = pd_offsets(
        np.array([0.015, 0.0, 0.0]),
        np.zeros(3),
        np.zeros(3),
        np.zeros(3),
    )
    assert offsets[0] == pytest.approx(1.0)
    braking = pd_offsets(
        np.array([0.04, 0.0, 0.0]),
        np.zeros(3),
        np.array([1.0, 0.0, 0.0]),
        np.zeros(3),
    )
    assert braking[0] == pytest.approx(0.0)
    assert native_grip_direction(0.0, gripper_command(1.0)) == -1.0
    assert native_grip_direction(0.0, gripper_command(-1.0)) == 1.0
    assert native_grip_direction(0.0, gripper_command(0.0)) == 0.0
    reached = ActionProgress()
    assert (
        reached.update(
            pos_err=0.0,
            rot_err=0.0,
            is_slow=True,
            fingers_done=True,
            opening_reached=True,
        )
        == "reached"
    )
    blocked = ActionProgress()
    for _ in range(2):
        assert (
            blocked.update(
                pos_err=0.1,
                rot_err=0.0,
                is_slow=True,
                fingers_done=True,
                opening_reached=True,
            )
            is None
        )
    assert (
        blocked.update(
            pos_err=0.1,
            rot_err=0.0,
            is_slow=True,
            fingers_done=True,
            opening_reached=True,
        )
        == "stopped"
    )
    assert execute_libero_action(None, None, native_budget=10)["outcome"] == "parse"


def test_replayed_response_logprobs_match_and_mask_prompt_gradients():
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7, requires_grad=True)
    input_ids = torch.randint(0, 7, (1, 5))
    response_index = torch.tensor([[3, 4, 0]])
    logprobs = response_logprobs_from_logits(logits, input_ids, response_index)
    shifted = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    assert logprobs[0, 0] == shifted[0, 2, input_ids[0, 3]]
    assert logprobs[0, 1] == shifted[0, 3, input_ids[0, 4]]
    assert logprobs[0, 2].item() == 0.0
    (logprobs * torch.tensor([[1.0, 1.0, 0.0]])).sum().backward()
    assert torch.count_nonzero(logits.grad[:, :2]) == 0
    assert torch.count_nonzero(logits.grad[:, 2:4]) > 0


def test_token_advantage_overfits_the_rewarded_response():
    torch.manual_seed(0)
    token_logits = torch.nn.Parameter(torch.zeros(2))
    optimizer = torch.optim.SGD([token_logits], lr=1.0)
    for _ in range(8):
        logprobs = token_logits.view(1, 2)
        processed = preprocess_loss_inputs(
            logprobs=logprobs,
            old_logprobs=logprobs.detach(),
            advantages=torch.tensor([1.0]),
            logprob_type="sequence_token_level",
            logprob_mask=torch.tensor([[True, False]]),
            loss_agg_func=lambda values, loss_mask, ratio: values,
        )
        mask = processed["loss_mask"].to(dtype=logprobs.dtype)
        loss = -(processed["logprobs"] * processed["advantages"] * mask).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    assert torch.softmax(token_logits.detach(), dim=-1)[0] > 0.9


def test_response_mask_removes_prompt_and_padding_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 6, 11, requires_grad=True)
    input_ids = torch.randint(0, 11, (2, 6))
    response_index = torch.tensor([[4, 5, 0], [3, 4, 5]])
    logprobs = response_logprobs_from_logits(logits, input_ids, response_index)
    assert logprobs[0, 2].item() == 0.0
    logprobs[0, 0].backward()
    prompt_positions = logits.grad[:, :3]
    assert prompt_positions is not None
    assert torch.count_nonzero(logprobs[0, 2]) == 0


def test_variable_turn_lengths_stack_without_changing_real_tokens():
    short = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "response_mask": torch.tensor([[True, False]]),
        "response_index": torch.tensor([[2, 0]]),
    }
    long = {
        "input_ids": torch.tensor([[4, 5, 6, 7]]),
        "response_mask": torch.tensor([[True, True, True]]),
        "response_index": torch.tensor([[2, 3, 0]]),
    }
    stacked = pad_stack_list_of_dict_tensor([short, long])
    assert stacked["input_ids"].shape == (2, 1, 4)
    assert stacked["input_ids"][0, 0, :3].tolist() == [1, 2, 3]
    assert stacked["input_ids"][0, 0, 3].item() == 0
    assert stacked["response_mask"][0, 0].tolist() == [True, False, False]
    assert stacked["response_index"][1, 0].tolist() == [2, 3, 0]


def test_history_resets_for_the_selected_environment():
    policy = QwenVLMPolicy.__new__(QwenVLMPolicy)
    policy._histories = [[{"role": "assistant", "content": "old"}]]
    policy._reset_histories({"reset_mask": [True]}, 1)
    assert policy._histories == [[]]


def test_episode_advantage_is_broadcast_to_response_tokens():
    logprobs = torch.zeros(2, 3)
    old = torch.zeros(2, 3)
    advantages = torch.tensor([2.0, -1.0])
    mask = torch.tensor([[True, True, False], [True, False, False]])
    processed = preprocess_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old,
        advantages=advantages,
        logprob_type="sequence_token_level",
        logprob_mask=mask,
        loss_agg_func=lambda values, loss_mask, ratio: values,
    )
    broadcast = processed["advantages"] * torch.ones_like(logprobs)
    assert broadcast.tolist() == [
        [2.0, 2.0, 2.0],
        [-1.0, -1.0, -1.0],
    ]
    assert processed["loss_mask"].tolist() == mask.tolist()
    chunk_mask = preprocess_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old,
        advantages=torch.tensor([[2.0], [-1.0]]),
        logprob_type="sequence_token_level",
        loss_mask=torch.tensor([[True], [False]]),
        logprob_mask=mask,
        loss_agg_func=lambda values, loss_mask, ratio: values,
    )
    assert chunk_mask["loss_mask"].tolist() == [
        [True, True, False],
        [False, False, False],
    ]


def test_all_success_and_all_failure_groups_are_filtered():
    rewards = torch.tensor([[[0.0], [0.0], [1.0], [0.0]]])
    dones = torch.zeros(2, 4, 1)
    dones[-1] = 1
    batch = preprocess_embodied_batch(
        {"rewards": rewards, "dones": dones},
        rollout_epoch=1,
        auto_reset=False,
        ignore_terminations=False,
        reward_type="chunk_level",
        filter_rewards=True,
        group_size=2,
        rewards_lower_bound=0.01,
        rewards_upper_bound=0.99,
    )
    kept = batch["loss_mask"][0, :, 0]
    assert kept.tolist() == [False, False, True, True]


def test_score_centering_is_zero_when_the_heads_match():
    logprobs = torch.log(torch.tensor([[0.2, 0.3], [0.4, 0.1]]))
    correction, _, _ = score_centering_correction(logprobs, logprobs)
    assert correction.tolist() == pytest.approx([0.0, 0.0])


def test_robodojo_schema_is_validated_but_not_executed():
    action = {
        part: dict.fromkeys(("x", "y", "z", "roll", "pitch", "yaw"), 0.0)
        if "arm" in part
        else {"width": 0.0}
        for part in ("left_arm", "right_arm", "left_gripper", "right_gripper")
    }
    assert validate_action(action)["left_gripper"]["width"] == 0.0
    with pytest.raises(NotImplementedError):
        execute_action(action)


def test_qwen3_8_install_target_pins_the_supported_stack():
    root = Path(__file__).resolve().parents[2]
    script = (root / "requirements" / "install.sh").read_text()
    requirements = (
        root / "requirements" / "embodied" / "models" / "qwen3_8.txt"
    ).read_text()
    assert "qwen3_8" in script
    assert "install_qwen3_8_model" in script
    assert "torch==2.13.0" in script or 'expected_torch="2.13.0"' in script
    assert "sglang==0.5.19" in script or 'expected_sglang="0.5.19"' in script
    assert (
        "transformers==5.12.1" in script or 'expected_transformers="5.12.1"' in script
    )
    assert "qwen-vl-utils==0.0.14" in requirements
    assert "flash-linear-attention==0.5.2" in requirements
    assert math.isfinite(1.0)


def test_vlm_policy_stage_configs_compose():
    from hydra import compose, initialize_config_dir

    config_dir = (
        Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"
    )
    expected = {
        "libero_gso_grpo_pear_sglang": ("libero_gso", 8, "libero"),
        "libero_10_grpo_pear_stage2": ("libero_10", 8, "libero"),
        "pear_stage3_libero_robodojo": ("libero_10", 8, "robodojo"),
    }
    for name, values in expected.items():
        with initialize_config_dir(config_dir=str(config_dir), version_base=None):
            cfg = compose(config_name=name)
        assert (
            cfg.env.train.task_suite_name,
            cfg.algorithm.group_size,
            cfg.actor.model.embodiment,
        ) == values
        assert cfg.rollout.rollout_backend == "sglang"
        assert cfg.algorithm.loss_type == "pear_reinforce"
        assert cfg.actor.global_batch_size == 64
