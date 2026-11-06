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

"""Actor-channel publication of routed embodied policy and environment parts."""

import asyncio
import inspect
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.schema.embodied_types import (
    EnvPart,
    EnvTransition,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    TrajectoryKey,
    TrajectorySource,
)
from rlinf.workers.rollout.embodied_protocol import (
    publish_env_parts,
    publish_policy_part,
)


class _RecordingChannel:
    def __init__(self):
        self.items = []

    def put(self, item, *, async_op):
        assert async_op
        self.items.append(item)


def _source(rank, size, offset=0):
    return TrajectorySource(TrajectoryKey(3, 1, rank, 2, 7), size, offset=offset)


def _unexpected_terminal(next_obs, current_features):
    raise AssertionError("This request must not perform terminal inference.")


def _unexpected_features():
    raise AssertionError("Features must not be requested before part validation.")


@pytest.mark.parametrize("collect_raw_obs", [False, True])
def test_policy_publication_preserves_routed_rows_and_statistics(collect_raw_obs):
    sources = [_source(1, 1, offset=2), _source(0, 2, offset=4)]
    states = torch.arange(12).reshape(3, 4)
    actions = torch.arange(18).reshape(3, 2, 3)
    output = PolicyOutput(
        forward_inputs={"action": actions.reshape(3, 6)},
        prev_logprobs=torch.arange(3).reshape(3, 1).float(),
        prev_values=torch.arange(3, 6).reshape(3, 1).float(),
        versions=torch.full((3, 1), 9.0),
    )
    request = PolicyInput(obs={"states": states}, sources=sources)
    channel = _RecordingChannel()

    publish_policy_part(
        request,
        actions,
        output,
        channel,
        collect_raw_transition_obs=collect_raw_obs,
    )

    [part] = channel.items
    assert isinstance(part, PolicyPart)
    assert part.inferred
    assert part.sources == sources
    assert part.external_actions is None
    assert torch.equal(part.output.prev_logprobs, output.prev_logprobs)
    assert torch.equal(part.output.prev_values, output.prev_values)
    assert torch.equal(part.output.versions, output.versions)
    if collect_raw_obs:
        assert torch.equal(part.obs["states"], states)
    else:
        assert part.obs == {}
    fragments = part.split()
    assert fragments[1].sources == [sources[1]]
    assert torch.equal(
        fragments[1].output.forward_inputs["action"], actions[1:].reshape(2, 6)
    )
    assert torch.equal(request.obs["states"], states)


@pytest.mark.parametrize("collect_raw_obs", [False, True])
def test_external_actions_publish_without_policy_statistics(collect_raw_obs):
    actions = torch.arange(12).reshape(2, 2, 3)
    states = torch.arange(8).reshape(2, 4)
    request = PolicyInput(
        obs={"states": states},
        sources=[_source(0, 2, offset=3)],
        external_actions=actions,
    )
    channel = _RecordingChannel()

    publish_policy_part(
        request,
        actions,
        None,
        channel,
        collect_raw_transition_obs=collect_raw_obs,
    )

    [part] = channel.items
    assert not part.inferred
    assert part.output is None
    assert part.sources == request.sources
    assert torch.equal(part.external_actions, actions)
    if collect_raw_obs:
        assert torch.equal(part.obs["states"], states)
    else:
        assert part.obs == {}


@pytest.mark.parametrize("collect_raw_obs", [False, True])
def test_environment_publication_splits_features_across_empty_request_slots(
    collect_raw_obs,
):
    first = EnvPart(sources=[_source(0, 2, offset=1)], transition=EnvTransition())
    last = EnvPart(sources=[_source(1, 1, offset=5)], transition=EnvTransition())
    states = torch.arange(16).reshape(4, 4)
    features = {"z_rl": torch.arange(8).reshape(4, 2)}
    request = PolicyInput(
        obs={"states": states},
        env_parts=[first, None, last],
        request_sizes=[2, 1, 1],
    )
    channel = _RecordingChannel()
    feature_calls = []

    def get_features():
        feature_calls.append(True)
        return features

    publish_env_parts(
        request,
        channel,
        collect_raw_transition_obs=collect_raw_obs,
        infer_terminal=_unexpected_terminal,
        get_next_rlt_obs=get_features,
    )

    assert feature_calls == [True]
    assert len(channel.items) == 2
    for part, original, rows in zip(
        channel.items, [first, last], [slice(0, 2), slice(3, 4)]
    ):
        assert part.sources == original.sources
        assert part.transition is original.transition
        assert not part.requires_inference
        assert torch.equal(part.next_rlt_obs["z_rl"], features["z_rl"][rows])
        if collect_raw_obs:
            assert torch.equal(part.next_obs["states"], states[rows])
        else:
            assert part.next_obs is None
    assert first.next_rlt_obs is None
    assert last.next_rlt_obs is None


@pytest.mark.parametrize("collect_raw_obs", [False, True])
def test_terminal_observation_overrides_current_request_and_resolves_values(
    collect_raw_obs,
):
    terminal_obs = {"states": torch.full((2, 4), 17.0)}
    terminal = EnvPart(
        sources=[_source(1, 2, offset=4)],
        transition=EnvTransition(),
        next_obs=terminal_obs,
        requires_inference=True,
    )
    request = PolicyInput(
        obs={"states": torch.zeros(3, 4)},
        env_parts=[None, terminal],
        request_sizes=[1, 2],
    )
    current_features = {"z_rl": torch.arange(6).reshape(3, 2)}
    resolved_features = {"z_rl": torch.full((2, 2), 23.0)}
    values = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    calls = []
    channel = _RecordingChannel()

    def infer_terminal(next_obs, features):
        calls.append((next_obs, features))
        return values, resolved_features

    publish_env_parts(
        request,
        channel,
        collect_raw_transition_obs=collect_raw_obs,
        infer_terminal=infer_terminal,
        get_next_rlt_obs=lambda: current_features,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0][0]["states"], terminal_obs["states"])
    assert torch.equal(calls[0][1]["z_rl"], current_features["z_rl"][1:])
    [part] = channel.items
    assert torch.equal(part.final_prev_values, values)
    assert torch.equal(part.bootstrap_values, values[:, :1])
    assert torch.equal(part.next_rlt_obs["z_rl"], resolved_features["z_rl"])
    assert not part.requires_inference
    assert terminal.requires_inference
    if collect_raw_obs:
        assert torch.equal(part.next_obs["states"], terminal_obs["states"])
    else:
        assert part.next_obs is None


def test_terminal_inference_uses_current_observation_when_raw_collection_is_disabled():
    terminal = EnvPart(
        sources=[_source(0, 1, offset=2)],
        transition=EnvTransition(),
        requires_inference=True,
    )
    states = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    request = PolicyInput(
        obs={"states": states},
        env_parts=[None, terminal],
        request_sizes=[1, 1],
    )
    calls = []
    channel = _RecordingChannel()

    def infer_terminal(next_obs, features):
        calls.append((next_obs, features))
        return None, {"z_rl": torch.ones(1, 2)}

    publish_env_parts(
        request,
        channel,
        collect_raw_transition_obs=False,
        infer_terminal=infer_terminal,
    )

    assert len(calls) == 1
    assert torch.equal(calls[0][0]["states"], states[1:])
    assert calls[0][1] is None
    [part] = channel.items
    assert part.next_obs is None
    assert part.bootstrap_values is None
    assert part.final_prev_values is None
    assert torch.equal(part.next_rlt_obs["z_rl"], torch.ones(1, 2))


def test_final_request_only_publishes_its_environment_completion():
    completion = EnvPart(
        sources=[_source(0, 1, offset=3)],
        transition=EnvTransition(rewards=torch.tensor([[2.0]])),
        next_obs={"states": torch.ones(1, 2)},
    )
    request = PolicyInput(
        obs={},
        env_parts=[completion],
        request_sizes=[1],
        is_last=True,
    )
    channel = _RecordingChannel()

    publish_env_parts(
        request,
        channel,
        collect_raw_transition_obs=True,
        infer_terminal=_unexpected_terminal,
    )

    [part] = channel.items
    assert isinstance(part, EnvPart)
    assert part.sources == completion.sources
    assert torch.equal(part.transition.rewards, completion.transition.rewards)
    assert torch.equal(part.next_obs["states"], completion.next_obs["states"])


def test_no_attached_parts_do_not_request_backend_features():
    request = PolicyInput(obs={"states": torch.ones(1, 2)}, request_sizes=[1])
    channel = _RecordingChannel()

    publish_env_parts(
        request,
        channel,
        collect_raw_transition_obs=False,
        infer_terminal=_unexpected_terminal,
        get_next_rlt_obs=_unexpected_features,
    )

    assert channel.items == []


@pytest.mark.parametrize("request_sizes", [[], [1, 1]])
def test_mismatched_parts_fail_before_backend_callbacks_or_publication(request_sizes):
    request = PolicyInput(
        obs={"states": torch.ones(2, 2)},
        env_parts=[EnvPart(sources=[_source(0, 1)], transition=EnvTransition())],
        request_sizes=request_sizes,
    )
    channel = _RecordingChannel()

    with pytest.raises(ValueError, match="parts and request sizes must align"):
        publish_env_parts(
            request,
            channel,
            collect_raw_transition_obs=False,
            infer_terminal=_unexpected_terminal,
            get_next_rlt_obs=_unexpected_features,
        )

    assert channel.items == []


def test_phyai_worker_completes_terminal_values_in_training_mode():
    from rlinf.workers.rollout.phyai.phyai_rollout_worker import PhyAIRolloutWorker

    values = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    terminal_obs = {"states": torch.full((2, 4), 17.0)}
    source = _source(1, 2, offset=4)
    completion = EnvPart(
        sources=[source],
        transition=EnvTransition(),
        next_obs=terminal_obs,
        requires_inference=True,
    )
    request = PolicyInput(
        obs={"states": torch.zeros(2, 4)},
        env_parts=[completion],
        request_sizes=[2],
    )
    calls = []

    def predict(obs, *, mode):
        calls.append((obs, mode))
        return torch.zeros(2, 1, 3), {"prev_values": values}

    rollout = object.__new__(PhyAIRolloutWorker)
    rollout.collect_transitions = False
    rollout.enable_rlt = False
    rollout.predict = predict
    channel = _RecordingChannel()

    rollout._publish_env_parts(request, channel)

    assert len(calls) == 1
    assert calls[0][1] == "train"
    assert torch.equal(calls[0][0]["states"], terminal_obs["states"])
    [part] = channel.items
    assert part.sources == [source]
    assert part.transition is completion.transition
    assert torch.equal(part.final_prev_values, values)
    assert torch.equal(part.bootstrap_values, values[:, :1])
    assert part.next_obs is None
    assert part.next_rlt_obs is None
    assert not part.requires_inference
    assert completion.requires_inference


@pytest.mark.parametrize("external", [False, True])
def test_phyai_epoch_sends_actions_before_policy_and_terminal_publication(external):
    from rlinf.workers.rollout.phyai.phyai_rollout_worker import PhyAIRolloutWorker

    events = []
    policy_sources = [_source(0, 1, offset=2), _source(1, 1, offset=4)]
    terminal_source = TrajectorySource(TrajectoryKey(3, 1, 0, 2, 6), 1, offset=2)
    terminal = EnvPart(
        sources=[terminal_source],
        transition=EnvTransition(rewards=torch.tensor([[3.0]])),
        next_obs={"states": torch.full((1, 2), 9.0)},
        requires_inference=True,
    )
    final = EnvPart(
        sources=policy_sources,
        transition=EnvTransition(rewards=torch.full((2, 1), 4.0)),
        next_obs={"states": torch.full((2, 2), 11.0)},
        requires_inference=True,
    )
    external_actions = torch.arange(4).reshape(2, 1, 2).float()
    policy_requests = [
        PolicyInput(
            obs={"states": torch.full((1, 2), float(index + 1))},
            sources=[source],
            env_parts=[terminal if index == 0 else None],
            request_sizes=[1],
            external_actions=external_actions[index : index + 1] if external else None,
        )
        for index, source in enumerate(policy_sources)
    ]
    final_request = PolicyInput(
        obs={"states": torch.zeros(2, 2)},
        env_parts=[final],
        request_sizes=[2],
        is_last=True,
    )
    requests = [
        ("train_policy", policy_requests),
        ("train_policy_final", [final_request]),
    ]
    input_channel = object()
    output_channel = object()
    sent = []

    class Receipt:
        def __init__(self, merge_fn, parts):
            self.merge_fn = merge_fn
            self.parts = parts

        async def async_wait(self):
            return self.merge_fn(self.parts)

    class ActorChannel(_RecordingChannel):
        def put(self, part, *, async_op):
            events.append(("publish", type(part).__name__))
            super().put(part, async_op=async_op)

    def recv_from(
        *,
        group_name,
        channel,
        tag,
        route_key,
        async_op,
        batch_size,
        merge_fn,
        infer_batch_size_fn,
    ):
        assert group_name == "env"
        assert channel is input_channel
        assert route_key == 0
        assert async_op
        assert batch_size == 2
        expected_tag, parts = requests.pop(0)
        assert tag == expected_tag
        assert sum(infer_batch_size_fn(part) for part in parts) == batch_size
        events.append(("receive", tag))
        return Receipt(merge_fn, parts)

    def send_to(
        *,
        group_name,
        channel,
        data,
        tag,
        route_key,
        async_op,
        batch_size,
        split_fn,
    ):
        assert group_name == "env"
        assert channel is output_channel
        assert tag == "policy"
        assert route_key == 0
        assert async_op
        assert batch_size == 2
        assert data.device.type == "cpu"
        assert data.is_contiguous()
        shards = split_fn(data, [1, 1])
        assert all(shard.shape == (1, 1, 2) for shard in shards)
        sent.append(torch.cat(shards))
        events.append(("send", tag))

    def predict(obs, *, mode):
        assert mode == "train"
        states = obs["states"]
        marker = int(states[0, 0])
        events.append(("predict", marker))
        actions = states.unsqueeze(1) + 100
        values = torch.stack([states[:, 0], states[:, 0] + 1], dim=1)
        return actions, {
            "forward_inputs": {"action": actions.flatten(1)},
            "prev_logprobs": torch.zeros(len(states), 1),
            "prev_values": values,
        }

    rollout = object.__new__(PhyAIRolloutWorker)
    rollout.cfg = SimpleNamespace(env=SimpleNamespace(group_name="env"))
    rollout.n_train_chunk_steps = 1
    rollout.num_pipeline_stages = 1
    rollout.train_batch_size = 2
    rollout.collect_transitions = True
    rollout.enable_rlt = False
    rollout.collect_prev_infos = True
    rollout.version = 7
    rollout.recv_from = recv_from
    rollout.send_to = send_to
    rollout.predict = predict
    actor_channel = ActorChannel()

    # Bypass scheduler decorators; all protocol methods remain real.
    asyncio.run(
        inspect.unwrap(PhyAIRolloutWorker.generate_one_epoch)(
            rollout, input_channel, output_channel, actor_channel
        )
    )

    expected_events = [("receive", "train_policy")]
    if not external:
        expected_events.append(("predict", 1))
    expected_events.extend(
        [
            ("send", "policy"),
            ("publish", "PolicyPart"),
            ("predict", 9),
            ("publish", "EnvPart"),
            ("receive", "train_policy_final"),
            ("predict", 11),
            ("publish", "EnvPart"),
        ]
    )
    assert events == expected_events
    assert requests == []
    policy_part, terminal_part, final_part = actor_channel.items
    assert policy_part.sources == policy_sources
    assert policy_part.inferred is not external
    [actions] = sent
    if external:
        assert policy_part.output is None
        assert torch.equal(policy_part.external_actions, external_actions)
        assert torch.equal(actions, external_actions)
    else:
        assert torch.equal(actions, torch.tensor([[[101.0, 101.0]], [[102.0, 102.0]]]))
        assert torch.equal(policy_part.output.versions, torch.full((2, 1), 7.0))
        assert torch.equal(
            policy_part.output.prev_values, torch.tensor([[1.0, 2.0], [2.0, 3.0]])
        )
    assert terminal_part.sources == [terminal_source]
    assert torch.equal(terminal_part.bootstrap_values, torch.tensor([[9.0]]))
    assert torch.equal(terminal_part.final_prev_values, torch.tensor([[9.0, 10.0]]))
    assert final_part.sources == policy_part.sources
    assert torch.equal(final_part.bootstrap_values, torch.full((2, 1), 11.0))
    assert not terminal_part.requires_inference
    assert not final_part.requires_inference
