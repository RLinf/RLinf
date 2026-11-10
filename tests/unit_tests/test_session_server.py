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

"""Integration test of the session server write side.

A mock sglang server (this directory's ``mock_sglang_server``, thread-hosted)
stands in for the router's backend; the real ``SessionServerWorker`` runs in
a scheduler worker and forwards to it. The payload channels are real (two
shards on this node); the buffer worker's position is taken by this driver,
which gets from the channels directly.

Covered: a group completing normally end to end (payload parts plus the
finalize part, then the completion metadata), a generate response arriving
after an abort leaving no payload, a failed background payload put voiding
the group, an under-reported truncation failing loud at close, and the
finalize part's put failing after all earlier puts settled — the group is
discarded, not reported complete.
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

from rlinf.data.schema.reasoning_results import (  # noqa: E402
    DiscardedGroup,
    GroupMetadata,
)
from rlinf.scheduler import (  # noqa: E402
    Channel,
    Cluster,
    NodePlacementStrategy,
)
from rlinf.scheduler.channel.channel import DEFAULT_KEY  # noqa: E402
from rlinf.workers.rollout.session_server.payload import (  # noqa: E402
    GroupPayload,
    create_payload_channels,
)
from rlinf.workers.rollout.session_server.server_worker import (  # noqa: E402
    SessionServerWorker,
)

TASK_NAME = "searchr1"
GROUP_SIZE = 2
LOADED_VERSION = 3
MOCK_WEIGHT_VERSION = 5
# Even group ids land on payload shard 0, odd ones on shard 1.
GROUP_NORMAL = 10
GROUP_LATE_RESPONSE = 14
GROUP_PUT_FAILURE = 15
GROUP_UNDERREPORT = 12
# Must stay last: this scenario kills shard 0's channel worker, which the
# even group ids of the scenarios above also use.
GROUP_FINALIZE_PUT_FAILURE = 16


@pytest.fixture(scope="module")
def cluster():
    c = Cluster(num_nodes=1)
    yield c


@pytest.fixture(scope="module")
def env(cluster):
    """One session server, two payload shards, and a mock sglang server."""
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "group_size": GROUP_SIZE,
                "sampling_params": {"do_sample": False, "max_new_tokens": 8},
            },
            "rollout": {"return_logprobs": True},
            "session_server": {"host": "127.0.0.1", "port": find_available_port()},
        }
    )

    mock = MockSGLangServer()
    mock.start()

    agent_output = Channel.create(name="AgentOutput")
    payload_channels = create_payload_channels("Payload", num_shards=2, node_rank=0)

    server = SessionServerWorker.create_group(cfg, 0).launch(
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

    yield SimpleNamespace(
        base_url=base_url,
        server=server,
        mock=mock,
        agent_output=agent_output,
        payload_channels=payload_channels,
    )

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


def close(env, session_id, kept_lens=None):
    body = {} if kept_lens is None else {"kept_lens": kept_lens}
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


class TestNormalCompletion:
    def test_group_completes_end_to_end(self, env):
        sid0 = create_session(env, GROUP_NORMAL, 0)
        sid1 = create_session(env, GROUP_NORMAL, 1)

        # Trajectory 0, turn 0: defaults only. The junk rlinf block must be
        # stripped before forwarding.
        first_input = [1, 2, 3, 4, 5]
        resp = generate(env, sid0, first_input, extra={"rlinf": {"junk": 1}}).json()
        assert len(resp["output_ids"]) == 8
        assert resp["weight_version"] == MOCK_WEIGHT_VERSION
        assert resp["finish_reason"] == {"type": "stop", "length": 8}
        first_output = resp["output_ids"]

        # Trajectory 0, turn 1: continues the previous turn and overrides
        # max_new_tokens, exercising the sampling-param merge.
        second_input = first_input + first_output
        resp = generate(
            env, sid0, second_input, sampling_params={"max_new_tokens": 5}
        ).json()
        assert len(resp["output_ids"]) == 5

        # Trajectory 1: one turn, defaults only.
        resp = generate(env, sid1, [10, 11]).json()
        assert len(resp["output_ids"]) == 8

        assert close(env, sid0).status_code == 200
        assert close(env, sid1).status_code == 200
        assert reward(env, sid0, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is True

        # What was forwarded: routing key header, stripped rlinf block,
        # forced return_logprob, merged sampling params.
        generates = [r for r in env.mock.request_log if r["path"] == "/generate"]
        assert len(generates) == 3
        for record in generates:
            assert record["headers"]["x-smg-routing-key"] == str(GROUP_NORMAL)
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

        metadata = poll_channel(env.agent_output)
        assert isinstance(metadata, GroupMetadata)
        assert metadata.group_id == GROUP_NORMAL
        assert metadata.task_name == TASK_NAME
        assert metadata.rewards == [1.0, 0.5]
        assert metadata.earliest_version == LOADED_VERSION
        assert metadata.completion_version == MOCK_WEIGHT_VERSION
        assert metadata.num_gen_tokens == 8 + 5 + 8
        assert metadata.num_generate_calls == 3

        payload = poll_channel(env.payload_channels[0], key=GROUP_NORMAL)
        assert isinstance(payload, GroupPayload)
        assert payload.group_id == GROUP_NORMAL
        assert len(payload.parts) == 4
        part0, part1, part2, finalize = payload.parts
        assert (part0.traj_index, part0.turn_index) == (0, 0)
        assert part0.input_ids == first_input
        assert part0.output_ids == first_output
        assert part0.logprobs is not None and len(part0.logprobs) == 8
        assert part0.version_start == LOADED_VERSION
        assert part0.weight_version == MOCK_WEIGHT_VERSION
        assert (part1.traj_index, part1.turn_index) == (0, 1)
        assert part1.prev_kept_len is None
        assert (part2.traj_index, part2.turn_index) == (1, 0)
        assert finalize.last is True
        assert finalize.traj_index == -1
        assert finalize.turn_index == -1
        assert finalize.rewards == [1.0, 0.5]
        assert finalize.version_start == LOADED_VERSION
        assert finalize.weight_version == MOCK_WEIGHT_VERSION
        assert finalize.verdicts == {
            0: {"kept_lens": {0: 8, 1: 5}, "continues_prev": [False, True]},
            1: {"kept_lens": {0: 8}, "continues_prev": [False]},
        }

        # The group is spent: its sessions answer unknown from now on.
        assert generate(env, sid0, [1]).status_code == 404


class TestAbort:
    def test_late_response_after_abort_leaves_no_payload(self, env):
        sid = create_session(env, GROUP_LATE_RESPONSE, 0)

        env.mock.latency_s = 0.5
        outcome = {}

        def slow_generate():
            outcome["response"] = generate(env, sid, [1, 2, 3])

        thread = threading.Thread(target=slow_generate)
        thread.start()
        time.sleep(0.15)
        abort = requests.post(
            f"{env.base_url}/groups/{GROUP_LATE_RESPONSE}/abort",
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

        discarded = poll_channel(env.agent_output)
        assert isinstance(discarded, DiscardedGroup)
        assert discarded.group_id == GROUP_LATE_RESPONSE
        assert discarded.reason == "agent_reset"
        assert (
            poll_channel(env.payload_channels[0], key=GROUP_LATE_RESPONSE, timeout_s=2)
            is None
        )


class TestPutFailure:
    def test_failed_background_put_voids_group(self, env):
        # GROUP_PUT_FAILURE is odd, so its parts go to shard 1; kill that
        # shard's channel worker so every payload put for it fails.
        ray.kill(env.payload_channels[1]._main_channel_worker_actor)

        sid = create_session(env, GROUP_PUT_FAILURE, 0)
        assert generate(env, sid, [1, 2]).status_code == 200

        discarded = poll_channel(env.agent_output, timeout_s=15)
        assert isinstance(discarded, DiscardedGroup)
        assert discarded.group_id == GROUP_PUT_FAILURE
        assert discarded.reason == "payload_write_failed"

        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/payload_write_failed"] == 1


class TestUnderreportedTruncation:
    def test_close_fails_loud_on_under_reported_truncation(self, env):
        sid = create_session(env, GROUP_UNDERREPORT, 0)

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

        assert poll_channel(env.agent_output, timeout_s=2) is None
        assert (
            poll_channel(env.payload_channels[0], key=GROUP_UNDERREPORT, timeout_s=2)
            is None
        )


class TestFinalizePutFailure:
    def test_finalize_put_failure_discards_not_reports(self, env):
        # GROUP_FINALIZE_PUT_FAILURE is even, so its parts go to shard 0.
        # All earlier puts settle fine; only the finalize part's put finds
        # the shard's channel worker dead. The group must then be discarded,
        # not reported: the GroupMetadata would promise a payload the
        # collector can never emit.
        sid0 = create_session(env, GROUP_FINALIZE_PUT_FAILURE, 0)
        sid1 = create_session(env, GROUP_FINALIZE_PUT_FAILURE, 1)
        assert generate(env, sid0, [1, 2]).status_code == 200
        assert generate(env, sid1, [3, 4]).status_code == 200

        # Wait until the background settlements are done, then kill the
        # shard so only the finalize part fails.
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            metrics = env.server.get_metrics().wait()[0]
            if metrics["session_server/pending_payload_bytes"] == 0:
                break
            time.sleep(0.1)
        else:
            pytest.fail("payload settlements did not drain")
        ray.kill(env.payload_channels[0]._main_channel_worker_actor)

        assert close(env, sid0).status_code == 200
        assert close(env, sid1).status_code == 200
        assert reward(env, sid0, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is False

        discarded = poll_channel(env.agent_output, timeout_s=15)
        assert isinstance(discarded, DiscardedGroup)
        assert discarded.group_id == GROUP_FINALIZE_PUT_FAILURE
        assert discarded.reason == "payload_write_failed"

        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/payload_write_failed"] == 2
