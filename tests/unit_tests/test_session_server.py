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

"""Integration test of the session server's write and read sides.

A mock sglang server (this directory's ``mock_sglang_server``,
thread-hosted) stands in for the router's backend; the real
``SessionServerWorker`` runs in a scheduler worker and forwards to it,
and the payload channels are real (two shards on this node). Downstream,
the real ``RolloutBufferWorker`` consumes the server's output. This
driver plays the two ends the buffer connects: the agent loop (it takes
the buffer's admitted requests and drives the HTTP protocol for each
group) and the actor's fetch side (it takes the delivered payload refs,
gets the payloads by key, and assembles them).

Covered: a group completing normally end to end — buffered, released as
a payload ref, fetched by key and assembled; a generate response
arriving after an abort leaving no payload; a group whose equal rewards
make the buffer drop it and release its payload; an under-reported
truncation failing loud at close; a failed background payload put voiding
the group; and the finalize part's put failing after all earlier puts
settled — the group is discarded, not delivered. Plus a pure assembly
check: constructed call records (one turn kept shorter than the engine
produced) assemble to exactly what ``get_rollout_result`` produces from
the matching agent outputs.

The scenario classes run in file order, and the buffer issues group ids
0, 1, 2, ... in admission order, so scenario k drives group id k: the two
shard-killing scenarios depend on the id parity that picks the payload
shard, and the last one kills a shard that every earlier scenario still
needs.
"""

import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("fastapi")

import ray  # noqa: E402
import requests  # noqa: E402
from mock_sglang_server import MockSGLangServer, find_available_port  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from rlinf.config import validate_session_server_cfg  # noqa: E402
from rlinf.data.schema.reasoning_results import GroupDelivery  # noqa: E402
from rlinf.scheduler import (  # noqa: E402
    Channel,
    Cluster,
    NodePlacementStrategy,
)
from rlinf.scheduler.channel.channel import DEFAULT_KEY  # noqa: E402
from rlinf.workers.agent.agent_loop import (  # noqa: E402
    AgentLoopOutput,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)
from rlinf.workers.rollout.session_server.assembly import (  # noqa: E402
    assemble_group_payload,
)
from rlinf.workers.rollout.session_server.payload import (  # noqa: E402
    GroupPayload,
    PayloadPart,
    create_payload_channels,
    payload_channel_for,
    payload_shard_for,
)
from rlinf.workers.rollout.session_server.server_worker import (  # noqa: E402
    SessionServerWorker,
)
from rlinf.workers.rollout_buffer.rollout_buffer_worker import (  # noqa: E402
    RolloutBufferWorker,
)

TASK_NAME = "searchr1"
GROUP_SIZE = 2
LOADED_VERSION = 3
MOCK_WEIGHT_VERSION = 5
# The buffer's dataloader forms batches of rollout_batch_size *
# max_num_gen_batches samples and admits one group per sample; 16 samples
# keep it issuing while the scenarios consume the first ones.
NUM_DATASET_SAMPLES = 16


def fake_dataset():
    """Dataset items shaped for the buffer's reasoning collate."""
    return [
        SimpleNamespace(
            prompt=[100 + sample, 200 + sample, 300 + sample],
            length=3,
            idx=sample,
            answer=f"answer {sample}",
            solution=None,
            image_data=None,
            prompt_text=None,
            meta=None,
            multi_modal_inputs=None,
        )
        for sample in range(NUM_DATASET_SAMPLES)
    ]


@pytest.fixture(scope="module")
def cluster():
    c = Cluster(num_nodes=1)
    yield c


@pytest.fixture(scope="module")
def env(cluster):
    """One session server, one real rollout buffer, two payload shards,
    and a mock sglang server."""
    server_cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": GROUP_SIZE,
                "sampling_params": {"do_sample": False, "max_new_tokens": 8},
            },
            "rollout": {
                "return_logprobs": True,
                "router": {"max_idle_secs": 600.0},
            },
            # Only the values this test needs to pick explicitly; the rest
            # of the session_server block is filled by config below, so
            # keys added there reach the worker without this file
            # drifting. Validation runs with enable still false (defaults
            # only), then the flag flips for the worker itself.
            "session_server": {
                "enable": False,
                "host": "127.0.0.1",
                "port": find_available_port(),
                "payload_put_timeout_s": 300.0,
                "max_pending_payload_bytes": 512 * 1024 * 1024,
                "generate_timeout_s": 600.0,
                "per_task": {},
                "verify_assembly": False,
            },
        }
    )
    server_cfg = validate_session_server_cfg(server_cfg)
    server_cfg.session_server.enable = True
    # The buffer's config: one group per training batch, the dataset
    # under admission, equal-reward rejection on (the normal scenario
    # scores its trajectories differently to survive it), and no actor
    # staleness bound so drops come only from the scenarios themselves.
    buffer_cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": GROUP_SIZE,
                "max_num_gen_batches": 8,
                "down_sampling": {
                    "do_down_sampling": True,
                    "down_sampling_config": {"reject_equal_reward": True},
                },
            },
            "data": {
                "rollout_batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
            },
            "agentloop": {"task_name": TASK_NAME},
            "rollout_buffer": {
                "max_inflight_groups": 8,
                "max_buffered_groups": 16,
                "max_dropped_groups_per_step": 8,
            },
        }
    )

    mock = MockSGLangServer()
    mock.start()

    agent_output = Channel.create(name="AgentOutput")
    dataloader = Channel.create(name="DataLoader")
    rollout = Channel.create(name="Rollout")
    payload_channels = create_payload_channels("Payload", num_shards=2, node_rank=0)

    server = SessionServerWorker.create_group(server_cfg, 0).launch(
        cluster, name="SessionServer", placement_strategy=NodePlacementStrategy([0])
    )
    # serve() is resident: dispatch it and never wait on the handle.
    server.serve(mock.url, agent_output, payload_channels)
    base_url = f"http://{server.get_server_address().wait()[0]}"

    for _ in range(100):
        try:
            if requests.get(f"{base_url}/health", timeout=1).status_code == 200:
                break
        except requests.RequestException:
            pass
        time.sleep(0.1)
    else:
        pytest.fail("the session server did not come up")

    server.set_loaded_versions({0: LOADED_VERSION}).wait()

    buffer = RolloutBufferWorker.create_group(buffer_cfg, fake_dataset()).launch(
        cluster, name="RolloutBuffer", placement_strategy=NodePlacementStrategy([0])
    )
    buffer.set_loaded_versions({0: LOADED_VERSION}).wait()
    # Resident like serve(): dispatch and never wait on the handle.
    buffer.run_continuously(agent_output, dataloader, rollout, payload_channels)

    yield SimpleNamespace(
        base_url=base_url,
        server=server,
        mock=mock,
        buffer=buffer,
        dataloader=dataloader,
        rollout=rollout,
        payload_channels=payload_channels,
    )

    buffer.stop().wait()
    server.stop().wait()
    mock.stop()


def create_session(env, group_id, traj_index):
    resp = requests.post(
        f"{env.base_url}/sessions",
        json={
            "group_id": group_id,
            "traj_index": traj_index,
            "group_size": GROUP_SIZE,
            "task_name": TASK_NAME,
        },
        timeout=5,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def generate(env, session_id, input_ids, sampling_params=None, extra=None):
    body = {"input_ids": input_ids}
    if sampling_params is not None:
        body["sampling_params"] = sampling_params
    if extra is not None:
        body.update(extra)
    return requests.post(
        f"{env.base_url}/sessions/{session_id}/generate", json=body, timeout=30
    )


def close(env, session_id, kept_lens=None, extra_fields_train=None):
    body = {} if kept_lens is None else {"kept_lens": kept_lens}
    if extra_fields_train is not None:
        body["verdict"] = {"extra_fields_train": extra_fields_train}
    return requests.post(
        f"{env.base_url}/sessions/{session_id}/close", json=body, timeout=30
    )


def reward(env, session_id, value):
    return requests.post(
        f"{env.base_url}/sessions/{session_id}/reward",
        json={"reward": value},
        timeout=30,
    )


def poll_channel(channel, key=DEFAULT_KEY, timeout_s=5.0):
    """Return the next item for ``key``, or None if nothing arrives in time.

    Polls with non-blocking gets: a blocking get whose wait times out here
    would stay alive inside the channel worker and silently consume
    whatever a later scenario puts for the same key.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return channel.get_nowait(key=key)
        except asyncio.QueueEmpty:
            time.sleep(0.05)
    return None


def next_request(env, timeout_s=10.0):
    """Pop the next rollout request the buffer admitted."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return env.dataloader.get_nowait()
        except asyncio.QueueEmpty:
            time.sleep(0.05)
    pytest.fail("the rollout buffer admitted no request in time")


def wait_for_drops(env, reason, timeout_s=15.0):
    """Poll the buffer's step metrics until it counts a drop for ``reason``.

    Each poll drains its own metrics window, but a drop lands in exactly
    one window — the one spanning the moment the buffer processed it — so
    repeated polling cannot miss it. Returns the window that saw it.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        metrics = env.buffer.get_metrics().wait()[0]
        if metrics.get(f"dropped_{reason}", 0) >= 1:
            return metrics
        time.sleep(0.1)
    pytest.fail(f"the buffer did not drop a group for reason {reason!r} in time")


def run_group(env, request, rewards):
    """Drive one admitted group as the agent loop would.

    One session per trajectory; trajectory 0 makes two generate calls
    (the second continues the first and overrides max_new_tokens,
    exercising the sampling-param merge), trajectory 1 one call; then
    close and reward. Returns the group id, the per-turn (prompt, output)
    token pairs in call order, and the session ids.
    """
    group_id = request.group_ids[0]
    prompt = list(request.input_ids[0])
    sids = [create_session(env, group_id, index) for index in range(GROUP_SIZE)]
    # The junk rlinf block must be stripped before forwarding.
    first_output = generate(env, sids[0], prompt, extra={"rlinf": {"junk": 1}}).json()[
        "output_ids"
    ]
    second_input = prompt + first_output
    second_output = generate(
        env, sids[0], second_input, sampling_params={"max_new_tokens": 5}
    ).json()["output_ids"]
    other_output = generate(env, sids[1], [10, 11]).json()["output_ids"]

    assert (
        close(env, sids[0], extra_fields_train={"idx_to_sub_traj": [0, 0]}).status_code
        == 200
    )
    assert (
        close(env, sids[1], extra_fields_train={"idx_to_sub_traj": [0]}).status_code
        == 200
    )
    assert reward(env, sids[0], rewards[0]).json()["group_complete"] is False
    assert reward(env, sids[1], rewards[1]).json()["group_complete"] is True

    turns = [
        (prompt, first_output),
        (second_input, second_output),
        ([10, 11], other_output),
    ]
    return group_id, turns, sids


def payload_channel_for_group(env, group_id):
    """The payload shard serving one group, as the shard mapping picks it."""
    return env.payload_channels[payload_shard_for(group_id, len(env.payload_channels))]


class TestAssembly:
    def test_assembled_result_matches_get_rollout_result(self):
        """Constructed call records assemble to the agent loop's result.

        Trajectory 0 has two turns, the first kept shorter than the engine
        produced (the agent truncated at a stop token); trajectory 1 one
        full turn. The parts are handed to the assembly out of order.
        """
        loaded, engine = 3, 5

        traj0_turn0 = PayloadPart(
            group_id=0,
            traj_index=0,
            turn_index=0,
            input_ids=[1, 2, 3],
            output_ids=[10, 11, 12, 13, 14, 15, 16, 17],
            logprobs=[-0.1 * i for i in range(8)],
            version_start=loaded,
            weight_version=engine,
            finish_reason={"type": "stop"},
        )
        # The agent kept 4 tokens and built the next prompt on them; the
        # extra token 99 stands for the appended tool response.
        traj0_turn1 = PayloadPart(
            group_id=0,
            traj_index=0,
            turn_index=1,
            input_ids=[1, 2, 3, 10, 11, 12, 13, 99],
            output_ids=[20, 21, 22, 23, 24],
            logprobs=[-0.2 * i for i in range(5)],
            version_start=loaded,
            weight_version=engine,
            finish_reason={"type": "stop"},
        )
        traj1_turn0 = PayloadPart(
            group_id=0,
            traj_index=1,
            turn_index=0,
            input_ids=[7, 8],
            output_ids=[30, 31, 32, 33, 34, 35, 36, 37],
            logprobs=[-0.3 * i for i in range(8)],
            version_start=loaded,
            weight_version=engine,
            finish_reason={"type": "stop"},
        )
        finalize = PayloadPart(
            group_id=0,
            traj_index=-1,
            turn_index=-1,
            input_ids=[],
            output_ids=[],
            logprobs=None,
            version_start=loaded,
            weight_version=engine,
            finish_reason={},
            last=True,
            rewards=[0.7, 0.4],
            verdicts={
                0: {
                    "kept_lens": {0: 4, 1: 5},
                    "continues_prev": [False, True],
                    "extra_fields_train": {"idx_to_sub_traj": [0, 0]},
                },
                1: {
                    "kept_lens": {0: 8},
                    "continues_prev": [False],
                    "extra_fields_train": {"idx_to_sub_traj": [0]},
                },
            },
        )

        def agent_output(part, kept, reward_score):
            # The agent view of the same call: the same truncation the
            # verdict declares, versions the engine stamped.
            return AgentLoopOutput(
                prompt_ids=list(part.input_ids),
                response_ids=part.output_ids[:kept],
                response_logprobs=part.logprobs[:kept],
                reward_score=reward_score,
                version_start=part.version_start,
                version_end=part.weight_version,
            )

        task_results = [
            MultiAgentLoopOutput(
                single_turn_outputs=[
                    agent_output(traj0_turn0, 4, 0.7),
                    agent_output(traj0_turn1, 5, 0.7),
                ]
            ),
            MultiAgentLoopOutput(
                single_turn_outputs=[agent_output(traj1_turn0, 8, 0.4)]
            ),
        ]
        # get_rollout_result only reads these two attributes.
        worker = MultiAgentLoopWorker.__new__(MultiAgentLoopWorker)
        worker.print_outputs = False
        worker.return_logprobs = True
        reference = worker.get_rollout_result(
            task_results, None, None, None, {"idx_to_sub_traj": [0, 0, 0]}
        )

        assembled = assemble_group_payload(
            GroupPayload(
                group_id=0,
                parts=[traj0_turn1, finalize, traj1_turn0, traj0_turn0],
            ),
            return_logprobs=True,
        )

        for field in (
            "num_sequence",
            "group_size",
            "idx_to_traj",
            "input_ids",
            "prompt_lengths",
            "response_lengths",
            "rollout_logprobs",
            "is_end",
            "rewards",
            "extra_fields_train",
            "version_start",
            "version_end",
        ):
            assert getattr(assembled, field) == getattr(reference, field), field


class TestNormalCompletion:
    def test_group_is_released_fetched_and_assembled(self, env):
        request = next_request(env)
        assert request.group_ids == [0]
        assert request.task_names == [TASK_NAME]
        assert request.n == GROUP_SIZE
        group_id, turns, sids = run_group(env, request, rewards=(1.0, 0.5))

        # What was forwarded: routing key header, stripped rlinf block,
        # forced return_logprob, merged sampling params.
        generates = [
            record
            for record in env.mock.request_log
            if record["path"] == "/generate"
            and record["headers"]["x-smg-routing-key"] == str(group_id)
        ]
        assert len(generates) == 3
        for record in generates:
            assert "rlinf" not in record["body"]
            assert record["body"]["return_logprob"] is True
        assert generates[0]["body"]["sampling_params"] == {
            "temperature": 0,
            "max_new_tokens": 8,
        }
        assert generates[1]["body"]["sampling_params"] == {
            "temperature": 0,
            "max_new_tokens": 5,
        }

        env.buffer.release_batch(version=0).wait()

        # The buffer delivered the group as a payload ref; the actor side
        # resolves it, fetches by key, and assembles.
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        assert delivery.group_id == group_id
        assert delivery.task_name == TASK_NAME
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        assert isinstance(payload, GroupPayload)
        result = assemble_group_payload(payload, return_logprobs=True)
        assert result.num_sequence == 3
        assert result.group_size == GROUP_SIZE
        assert result.idx_to_traj == [0, 0, 1]
        assert result.input_ids == [prompt + output for prompt, output in turns]
        assert result.prompt_lengths == [len(prompt) for prompt, _ in turns]
        assert result.response_lengths == [len(output) for _, output in turns]
        assert result.rollout_logprobs is not None
        assert [len(lp) for lp in result.rollout_logprobs] == result.response_lengths
        assert result.rewards == [1.0, 1.0, 0.5]
        assert result.extra_fields_train == {"idx_to_sub_traj": [0, 0, 0]}
        assert result.version_start == [LOADED_VERSION] * 3
        assert result.version_end == [MOCK_WEIGHT_VERSION] * 3

        # The group is taken: nothing is left under its payload key.
        assert poll_channel(channel, key=group_id, timeout_s=1) is None
        # The group is spent: its sessions answer unknown from now on.
        assert generate(env, sids[0], [1]).status_code == 404


class TestAbort:
    def test_late_response_after_abort_is_dropped_by_the_buffer(self, env):
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)

        env.mock.latency_s = 0.5
        outcome = {}

        def slow_generate():
            outcome["response"] = generate(env, sid, [1, 2, 3])

        thread = threading.Thread(target=slow_generate)
        thread.start()
        time.sleep(0.15)
        abort = requests.post(
            f"{env.base_url}/groups/{group_id}/abort",
            json={"reason": "agent_reset"},
            timeout=5,
        )
        thread.join(timeout=30)
        env.mock.latency_s = 0.0

        assert abort.status_code == 200
        # The generate was still in flight when the group was aborted: its
        # late response is rejected and writes nothing.
        assert outcome["response"].status_code == 410
        # Close and reward after the abort are dropped, not errors.
        assert close(env, sid).status_code == 200
        assert reward(env, sid, 0.0).status_code == 200

        # The buffer consumed the discard record: the group is dropped and
        # returns its admission capacity.
        metrics = wait_for_drops(env, "agent_reset")
        assert metrics["dropped_groups"] == 1
        assert (
            poll_channel(
                payload_channel_for_group(env, group_id), key=group_id, timeout_s=2
            )
            is None
        )
        assert poll_channel(env.rollout, timeout_s=2) is None


class TestEqualRewardDrop:
    def test_equal_reward_group_is_dropped_and_payload_released(self, env):
        request = next_request(env)
        group_id = request.group_ids[0]
        run_group(env, request, rewards=(0.5, 0.5))

        # A group whose trajectories all scored the same carries no group
        # signal: the buffer drops it and releases its payload — the actor
        # would never fetch it, so the queue must not hold it forever.
        metrics = wait_for_drops(env, "equal_reward")
        assert metrics["dropped_groups"] == 1
        assert (
            poll_channel(
                payload_channel_for_group(env, group_id), key=group_id, timeout_s=2
            )
            is None
        )
        assert poll_channel(env.rollout, timeout_s=2) is None


class TestUnderreportedTruncation:
    def test_close_fails_loud_on_under_reported_truncation(self, env):
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)

        first_input = [7, 8, 9]
        first_output = generate(env, sid, first_input).json()["output_ids"]
        assert len(first_output) == 8

        # The agent kept only 4 tokens of the first output but declares the
        # full 8 at close: the break inside the declared kept output must
        # fail loud instead of assembling wrong training data.
        second_input = first_input + first_output[:4] + [424242]
        assert generate(env, sid, second_input).status_code == 200

        resp = close(env, sid, kept_lens={"0": 8})
        assert resp.status_code == 500
        assert "declared" in resp.json()["error"]

        # Nothing left the session server, so the buffer never heard of the
        # group: nothing delivered, nothing dropped.
        assert poll_channel(env.rollout, timeout_s=2) is None
        assert (
            poll_channel(
                payload_channel_for_group(env, group_id), key=group_id, timeout_s=2
            )
            is None
        )
        metrics = env.buffer.get_metrics().wait()[0]
        assert metrics["dropped_groups"] == 0


class TestVerdictValidation:
    def test_close_rejects_unsupported_verdict_fields(self, env):
        """A verdict the session server does not support fails at close."""
        # A group id outside the buffer worker's sequence: this scenario
        # never finalizes, so it must not shift the ids the drop-path
        # scenarios pin their shard choice on.
        group_id = 4242
        sid = create_session(env, group_id, 0)
        generate(env, sid, [7, 8, 9])

        # An unknown verdict key, a field that does not cover the turn,
        # and a non-scalar value are all rejected; nothing is committed,
        # so the same session still closes cleanly afterwards.
        resp = requests.post(
            f"{env.base_url}/sessions/{sid}/close",
            json={"verdict": {"per_turn_rewards": [0.5]}},
            timeout=30,
        )
        assert resp.status_code == 400
        assert "unsupported verdict fields" in resp.json()["error"]

        resp = close(env, sid, extra_fields_train={"idx_to_sub_traj": [0, 0]})
        assert resp.status_code == 400
        assert "idx_to_sub_traj" in resp.json()["error"]

        resp = close(env, sid, extra_fields_train={"idx_to_sub_traj": ["x"]})
        assert resp.status_code == 400
        assert "idx_to_sub_traj" in resp.json()["error"]

        resp = close(env, sid, extra_fields_train={"idx_to_sub_traj": [0]})
        assert resp.status_code == 200


class TestPutFailure:
    def test_failed_background_put_voids_group(self, env):
        request = next_request(env)
        group_id = request.group_ids[0]
        # This scenario's group id is even, so its parts go to shard 0;
        # kill that shard's channel worker so every payload put fails.
        assert group_id % 2 == 0
        ray.kill(env.payload_channels[0]._main_channel_worker_actor)

        sid = create_session(env, group_id, 0)
        assert generate(env, sid, [1, 2]).status_code == 200

        # The failed put voids the group server-side; the buffer consumes
        # the discard record.
        metrics = wait_for_drops(env, "payload_write_failed")
        assert metrics["dropped_groups"] == 1

        server_metrics = env.server.get_metrics().wait()[0]
        assert server_metrics["session_server/payload_write_failed"] == 1


class TestFinalizePutFailure:
    def test_finalize_put_failure_discards_not_delivers(self, env):
        # Must stay last: this scenario kills the second payload shard,
        # and the scenarios above still need both shards alive.
        request = next_request(env)
        group_id = request.group_ids[0]
        assert group_id % 2 == 1
        sid0 = create_session(env, group_id, 0)
        sid1 = create_session(env, group_id, 1)
        assert generate(env, sid0, [1, 2]).status_code == 200
        assert generate(env, sid1, [3, 4]).status_code == 200

        # Wait until the background settlements are done, then kill the
        # shard so only the finalize part's put fails.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            metrics = env.server.get_metrics().wait()[0]
            if metrics["session_server/pending_payload_bytes"] == 0:
                break
            time.sleep(0.1)
        else:
            pytest.fail("payload settlements did not drain")
        ray.kill(env.payload_channels[1]._main_channel_worker_actor)

        assert close(env, sid0).status_code == 200
        assert close(env, sid1).status_code == 200
        assert reward(env, sid0, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is False

        # The group is discarded, not delivered: the GroupDelivery would
        # promise a payload the collector can never emit.
        metrics = wait_for_drops(env, "payload_write_failed")
        assert metrics["dropped_groups"] == 1
        assert poll_channel(env.rollout, timeout_s=2) is None

        server_metrics = env.server.get_metrics().wait()[0]
        # get_metrics drains per window: the failure above landed after
        # the last read (the pending-drain poll), so this window's count
        # is the one failure of this scenario alone.
        assert server_metrics["session_server/payload_write_failed"] == 1
