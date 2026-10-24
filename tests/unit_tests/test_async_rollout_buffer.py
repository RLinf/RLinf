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

"""Channel-backed completion ordering, critic targets, and policy staleness."""

import copy

import pytest
import torch

from rlinf.data.schema.agent_trajectory import (
    BufferedTrajectory,
    construct_rollout_batch,
)
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Channel
from rlinf.scheduler.channel.channel_worker import LocalChannel


def channel(capacity=0):
    result = Channel()
    result._initialize(
        "async-completions", None, None, None, local_channel=LocalChannel(capacity)
    )
    return result


def trajectory(sample_id=0, versions=None):
    if versions is None:
        versions = [[0, 1], [2]]
    result = DynamicRolloutResult(
        num_sequence=2,
        group_size=1,
        idx_to_traj=[0, 0],
        input_ids=[[1, 2, 3], [1, 2, 3, 4, 5]],
        prompt_lengths=[1, 4],
        response_lengths=[2, 1],
        rollout_logprobs=[[-0.2, -0.3], [-0.4]],
        rollout_weight_versions=versions,
        rollout_ids=[sample_id, sample_id],
        is_end=[False, True],
        rewards=[1.0, 1.0],
    )
    return BufferedTrajectory(sample_id, [1], "answer", result, {"__sum__/turns": 2})


def consume(input_channel, output_channel, retry_channel, **kwargs):
    return construct_rollout_batch(
        input_channel=input_channel,
        output_channel=output_channel,
        retry_channel=retry_channel,
        batch_size=kwargs.pop("batch_size", 1),
        seq_length=6,
        gamma=0.5,
        published_version=kwargs.pop("published_version", 2),
        **kwargs,
    )


def test_completed_trajectories_form_a_batch_without_waiting_for_prompt_order():
    incoming, output, retry = channel(3), channel(), channel()
    for sample_id in (2, 0, 1):
        incoming.put(trajectory(sample_id))
    state = consume(incoming, output, retry, batch_size=2)
    assert state == {
        "sample_ids": [2, 0],
        "oldest_version": 0,
        "stale_dropped": 0,
        "agent_metrics": [{"__sum__/turns": 2}, {"__sum__/turns": 2}],
    }
    assert incoming.qsize() == 1 and retry.empty()
    for sample_id in (2, 0):
        result = output.get_nowait()
        assert result.rollout_ids == [sample_id, sample_id]
        assert result.values is None
        assert result.rollout_logprobs == [[-0.2, -0.3], [-0.4]]
        assert result.rollout_weight_versions == [[0, 1], [2]]
        torch.testing.assert_close(
            result.returns,
            torch.tensor([[0, 0.25, 0.5, 0, 0, 0], [0, 0, 0, 0, 1.0, 0]]),
        )
    assert consume(incoming, output, retry)["sample_ids"] == [1]


@pytest.mark.parametrize(
    "staleness,stale", [(None, False), (0, True), (2, True), (3, False)]
)
def test_staleness_is_checked_at_consumption_and_retries_keep_the_prompt(
    staleness, stale
):
    incoming, output, retry = channel(2), channel(), channel()
    incoming.put(trajectory(9, versions=[[4, 5], [6]]))
    incoming.put(trajectory(9, versions=[[7, 7], [7]]))
    state = consume(
        incoming, output, retry, published_version=7, max_weight_staleness=staleness
    )
    assert state["sample_ids"] == [9]
    assert state["stale_dropped"] == int(stale)
    assert state["oldest_version"] == (7 if stale else 4)
    assert output.qsize() == 1
    if stale:
        assert retry.get_nowait() == (9, [1], "answer")
        assert incoming.empty()
    else:
        assert retry.empty() and incoming.qsize() == 1


def test_zero_staleness_accepts_the_current_published_version():
    incoming, output, retry = channel(1), channel(), channel()
    incoming.put(trajectory(versions=[[5, 5], [5]]))
    state = consume(
        incoming, output, retry, published_version=5, max_weight_staleness=0
    )
    assert state["stale_dropped"] == 0 and state["oldest_version"] == 5
    assert retry.empty()


def test_checkpoint_snapshot_preserves_fifo_and_behavior_probabilities():
    incoming, restored, output, retry = channel(2), channel(2), channel(), channel()
    for sample_id in (10, 5):
        incoming.put(trajectory(sample_id))
    snapshot = copy.deepcopy(incoming.peek_all())
    assert incoming.qsize() == 2
    for entry in snapshot:
        restored.put(entry)
    for sample_id in (10, 5):
        state = consume(restored, output, retry, published_version=8)
        assert state["sample_ids"] == [sample_id]
        result = output.get_nowait()
        assert result.rollout_logprobs == [[-0.2, -0.3], [-0.4]]
        assert result.rollout_weight_versions == [[0, 1], [2]]
    assert restored.empty() and incoming.qsize() == 2


@pytest.mark.parametrize("versions", [[[1], [2]], [[1, -1], [2]], [[1, 2], None]])
def test_consumption_rejects_missing_or_misaligned_behavior_provenance(versions):
    incoming, output, retry = channel(1), channel(), channel()
    incoming.put(trajectory(versions=versions))
    with pytest.raises(ValueError, match="Each generated token"):
        consume(incoming, output, retry)
    assert incoming.empty() and output.empty() and retry.empty()


def test_consumption_rejects_unpublished_weights():
    incoming, output, retry = channel(1), channel(), channel()
    incoming.put(trajectory(versions=[[2, 2], [2]]))
    with pytest.raises(ValueError, match="newer than the published version"):
        consume(incoming, output, retry, published_version=1)
    assert output.empty() and retry.empty()
