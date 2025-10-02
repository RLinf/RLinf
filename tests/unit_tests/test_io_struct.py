# Copyright 2025 The RLinf Authors.
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

from rlinf.data.io_struct import RolloutRequest, RolloutResult


def test_rollout_request_repeat_preserves_multimodal():
    request = RolloutRequest(
        n=2,
        input_ids=[[1, 2, 3], [4, 5]],
        image_data=[[b"img1-1", b"img1-2"], []],
        answers=["ans1", "ans2"],
        multi_modal_inputs=[{"pixels": [1, 2]}, {"pixels": [3]}],
    )

    repeated = request.repeat()

    assert repeated.n == 2
    assert repeated.input_ids == [[1, 2, 3], [1, 2, 3], [4, 5], [4, 5]]
    assert repeated.answers == ["ans1", "ans1", "ans2", "ans2"]
    assert repeated.image_data == [
        [b"img1-1", b"img1-2"],
        [b"img1-1", b"img1-2"],
        [],
        [],
    ]
    assert repeated.multi_modal_inputs == [
        {"pixels": [1, 2]},
        {"pixels": [1, 2]},
        {"pixels": [3]},
        {"pixels": [3]},
    ]


def _make_rollout_result():
    num_sequence = 4
    group_size = 2
    return RolloutResult(
        num_sequence=num_sequence,
        group_size=group_size,
        prompt_lengths=[3, 3, 4, 4],
        prompt_ids=[[11, 12, 13], [11, 12, 13], [21, 22, 23, 24], [21, 22, 23, 24]],
        response_lengths=[2, 2, 2, 2],
        response_ids=[[101, 102], [201, 202], [301, 302], [401, 402]],
        is_end=[True, False, True, True],
        answers=[{"answer": "a"}, {"answer": "b"}, {"answer": "c"}, {"answer": "d"}],
        image_data=[[b"a"], [b"b"], [b"c"], [b"d"]],
        multi_modal_inputs=[
            {"vision": "img-a"},
            {"vision": "img-b"},
            {"vision": "img-c"},
            {"vision": "img-d"},
        ],
        prompt_texts=["prompt-a", "prompt-a", "prompt-b", "prompt-b"],
        response_texts=["resp-a1", "resp-a2", "resp-b1", "resp-b2"],
        rollout_logprobs=[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
        rewards=torch.tensor([[1.0], [0.5], [0.2], [0.1]]),
        advantages=[0.1, 0.2, 0.3, 0.4],
        prev_logprobs=torch.tensor(
            [
                [0.01, 0.02],
                [0.03, 0.04],
                [0.05, 0.06],
                [0.07, 0.08],
            ]
        ),
        ref_logprobs=torch.tensor(
            [
                [0.11, 0.12],
                [0.13, 0.14],
                [0.15, 0.16],
                [0.17, 0.18],
            ]
        ),
    )


def test_rollout_result_split_and_merge_roundtrip():
    result = _make_rollout_result()

    split_results = RolloutResult.split_result_list_by_group([result])

    assert len(split_results) == result.num_sequence // result.group_size
    first, second = split_results

    assert first.num_sequence == result.group_size
    assert second.num_sequence == result.group_size
    assert first.prompt_ids == result.prompt_ids[: result.group_size]
    assert second.prompt_ids == result.prompt_ids[result.group_size :]
    assert first.response_ids == result.response_ids[: result.group_size]
    assert second.response_ids == result.response_ids[result.group_size :]
    assert first.prompt_texts == result.prompt_texts[: result.group_size]
    assert second.prompt_texts == result.prompt_texts[result.group_size :]
    assert first.response_texts == result.response_texts[: result.group_size]
    assert second.response_texts == result.response_texts[result.group_size :]
    assert first.image_data == result.image_data[: result.group_size]
    assert second.image_data == result.image_data[result.group_size :]
    assert first.multi_modal_inputs == result.multi_modal_inputs[: result.group_size]
    assert second.multi_modal_inputs == result.multi_modal_inputs[result.group_size :]
    assert first.rollout_logprobs == result.rollout_logprobs[: result.group_size]
    assert second.rollout_logprobs == result.rollout_logprobs[result.group_size :]
    assert torch.equal(first.rewards, result.rewards[: result.group_size])
    assert torch.equal(second.rewards, result.rewards[result.group_size :])
    assert first.advantages == result.advantages[: result.group_size]
    assert second.advantages == result.advantages[result.group_size :]

    merged = RolloutResult.merge_result_list(split_results)

    assert merged.num_sequence == result.num_sequence
    assert merged.group_size == result.group_size
    assert merged.prompt_ids == result.prompt_ids
    assert merged.prompt_lengths == result.prompt_lengths
    assert merged.response_ids == result.response_ids
    assert merged.response_lengths == result.response_lengths
    assert merged.is_end == result.is_end
    assert merged.answers == result.answers
    assert merged.rollout_logprobs == result.rollout_logprobs
    assert merged.advantages == result.advantages
    assert torch.equal(merged.rewards, result.rewards)
    assert torch.equal(merged.prev_logprobs, result.prev_logprobs)
    assert torch.equal(merged.ref_logprobs, result.ref_logprobs)
