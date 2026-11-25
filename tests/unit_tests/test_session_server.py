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
0, 1, 2, ... in admission order, so scenario k drives group id k. The two
payload put-failure scenarios use their own stacks with a failing payload
collector, so they inject isolated write failures without taking down a
channel worker or the shared stack.
"""

import asyncio
import contextvars
import importlib.util
import json
import threading
import time
from types import SimpleNamespace
from typing import Optional

import aiohttp  # noqa: E402
import pytest

# The session server's HTTP layer, its mock engine, and the SGLang module its
# worker imports (for the default sampling params) need these packages. The
# whole file skips without them, the config cases included: they build no
# server, but they live with the rest of the component.
_HAS_SESSION_SERVER_DEPS = all(
    importlib.util.find_spec(name) is not None for name in ("fastapi", "sglang")
)
pytestmark = pytest.mark.skipif(
    not _HAS_SESSION_SERVER_DEPS,
    reason="the session server needs fastapi and SGLang",
)

import requests  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from ray.exceptions import ActorDiedError  # noqa: E402

if _HAS_SESSION_SERVER_DEPS:
    from mock_sglang_server import (  # noqa: E402
        MockSGLangServer,
        find_available_port,
    )

from rlinf.config import (  # noqa: E402
    validate_reasoning_eval_cfg,
    validate_session_server_cfg,
)
from rlinf.data.schema.reasoning_results import GroupDelivery  # noqa: E402
from rlinf.scheduler import (  # noqa: E402
    Channel,
    Cluster,
    NodePlacementStrategy,
)
from rlinf.scheduler.channel.channel import DEFAULT_KEY  # noqa: E402
from rlinf.scheduler.collective import PeerCommunicationError  # noqa: E402
from rlinf.utils.http_client import no_proxy_env  # noqa: E402
from rlinf.workers.agent.agent_loop import (  # noqa: E402
    AgentLoopOutput,
    GroupDiscarded,
    MultiAgentLoopOutput,
    MultiAgentLoopWorker,
)
from rlinf.workers.agent.session_server_client import (  # noqa: E402
    SessionServerClient,
)
from rlinf.workers.rollout.session_server.assembly import (  # noqa: E402
    assemble_group_payload,
    verify_group_assembly,
)
from rlinf.workers.rollout.session_server.chat_completions import (  # noqa: E402
    ChatConversation,
)
from rlinf.workers.rollout.session_server.payload import (  # noqa: E402
    GroupPayload,
    PayloadCollector,
    PayloadPart,
    PayloadWriter,
    create_payload_channels,
    payload_channel_for,
)

if _HAS_SESSION_SERVER_DEPS:
    from rlinf.workers.rollout.session_server.server_worker import (  # noqa: E402
        SessionServerWorker,
        _GroupState,
        _SessionState,
    )
from rlinf.workers.rollout.session_server.turns import (  # noqa: E402
    UnderreportedTruncation,
    classify_turns,
    new_turn_record,
)
from rlinf.workers.rollout_buffer.rollout_buffer_worker import (  # noqa: E402
    RolloutBufferWorker,
)

TASK_NAME = "searchr1"
GROUP_SIZE = 2
# The training sequence length of the chat test stack, and the sampling
# default the entry caps against it. It has to leave room for the tool block
# the chat template renders into a prompt, which the other chat tests use.
SEQ_LENGTH = 1024
DEFAULT_MAX_NEW_TOKENS = 8
LOADED_VERSION = 3
MOCK_WEIGHT_VERSION = 5
# The tool list the chat tests send, shaped like Search-R1's search tool.
CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search Wikipedia.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
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


@pytest.fixture(scope="module", autouse=True)
def _no_proxy():
    """Keep loopback HTTP calls off any ambient HTTP proxy."""
    with no_proxy_env():
        yield


@pytest.fixture(scope="module")
def cluster():
    c = Cluster(num_nodes=1)
    yield c


_CHAT_FAMILIES = {
    "qwen": ("RLINF_TEST_QWEN25_MODEL_DIR", "qwen"),
    "glm47": ("RLINF_TEST_GLM47_MODEL_DIR", "glm47"),
}


def _chat_model_dir(family="qwen") -> Optional[str]:
    """The local tokenizer directory one chat family renders with.

    None when the family's environment variable is unset, which is the
    usual case off CI. A variable naming no directory fails the test: the
    chat tests would otherwise never run where the variable is set.
    """
    import os

    env_var = _CHAT_FAMILIES[family][0]
    model_dir = os.environ.get(env_var)
    if model_dir and not os.path.isdir(model_dir):
        pytest.fail(f"{env_var} is set to {model_dir!r}, which is not a directory")
    return model_dir


def _build_env(
    cluster,
    verify_assembly,
    suffix,
    chat=False,
    family="qwen",
    reject_equal_reward=True,
    payload_collector=None,
):
    """One session server, one real rollout buffer, two payload shards,
    and a mock sglang server. ``verify_assembly`` picks the session
    server's response mode; ``suffix`` keeps the channel and group names
    of a second stack from colliding with the first; ``chat`` adds the
    tokenizer config the chat entry needs."""
    server_cfg = OmegaConf.create(
        {
            # The chat entry caps a turn at runner.seq_length; a small value
            # keeps both the cap and the rejection reachable.
            "runner": {"seq_length": SEQ_LENGTH},
            "algorithm": {
                "group_size": GROUP_SIZE,
                "sampling_params": {"do_sample": False, "max_new_tokens": 8},
            },
            "rollout": {
                "return_logprobs": True,
                "router": {"max_idle_secs": 600.0},
                "model": {"model_path": _chat_model_dir(family)},
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
                "verify_assembly": verify_assembly,
            },
        }
    )
    if chat:
        server_cfg.agentloop = {
            "agentic_tokenizer": _CHAT_FAMILIES[family][1],
        }
        server_cfg.session_server.chat_template_kwargs = {}
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
                    "down_sampling_config": {
                        "reject_equal_reward": reject_equal_reward
                    },
                },
            },
            "data": {
                "rollout_batch_size": 1,
                "num_workers": 0,
                "shuffle": False,
            },
            "agentloop": {"task_name": TASK_NAME},
            "rollout_buffer": {
                # Every test in the module shares this stack, and a test that
                # leaves its group undelivered keeps that group's in-flight
                # slot, so the limit leaves room for all of them.
                "max_inflight_groups": 32,
                "max_buffered_groups": 16,
                "max_dropped_groups_per_step": 8,
            },
        }
    )

    mock = MockSGLangServer()
    mock.start()

    agent_output = Channel.create(name=f"AgentOutput{suffix}")
    dataloader = Channel.create(name=f"DataLoader{suffix}")
    rollout = Channel.create(name=f"Rollout{suffix}")
    if payload_collector is None:
        payload_channels = create_payload_channels(
            f"Payload{suffix}", num_shards=2, node_rank=0
        )
    else:
        payload_channels = [
            Channel.create(
                name=f"Payload{suffix}_{shard}",
                collector=payload_collector,
                node_rank=0,
            )
            for shard in range(2)
        ]

    server = SessionServerWorker.create_group(server_cfg, 0).launch(
        cluster,
        name=f"SessionServer{suffix}",
        placement_strategy=NodePlacementStrategy([0]),
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
        cluster,
        name=f"RolloutBuffer{suffix}",
        placement_strategy=NodePlacementStrategy([0]),
    )
    buffer.set_loaded_versions({0: LOADED_VERSION}).wait()
    # Resident like serve(): dispatch and never wait on the handle.
    buffer.run_continuously(agent_output, dataloader, rollout, payload_channels)

    return SimpleNamespace(
        base_url=base_url,
        server=server,
        server_cfg=server_cfg,
        mock=mock,
        buffer=buffer,
        dataloader=dataloader,
        rollout=rollout,
        payload_channels=payload_channels,
    )


def _stop_env(env):
    env.buffer.stop().wait()
    env.server.stop().wait()
    env.mock.stop()


@pytest.fixture(scope="module")
def env(cluster):
    """The default stack, assembly verification off."""
    built = _build_env(cluster, verify_assembly=False, suffix="")
    yield built
    _stop_env(built)


@pytest.fixture(scope="module")
def chat_env(cluster):
    """The same stack with the chat entry configured; skipped when the
    local Qwen tokenizer directory is not available."""
    if not _chat_model_dir("qwen"):
        pytest.skip("set RLINF_TEST_QWEN25_MODEL_DIR to run the chat tests")
    built = _build_env(
        cluster, verify_assembly=False, suffix="Chat", chat=True, family="qwen"
    )
    yield built
    _stop_env(built)


@pytest.fixture(scope="module")
def chat_env_glm(cluster):
    """The chat entry with the GLM family; skipped without its tokenizer."""
    if not _chat_model_dir("glm47"):
        pytest.skip("set RLINF_TEST_GLM47_MODEL_DIR to run the GLM chat tests")
    built = _build_env(
        cluster, verify_assembly=False, suffix="ChatGlm", chat=True, family="glm47"
    )
    yield built
    _stop_env(built)


@pytest.fixture(scope="module")
def verify_env(cluster):
    """The same stack with assembly verification on, so generate
    responses carry logprobs."""
    built = _build_env(cluster, verify_assembly=True, suffix="Verify")
    yield built
    _stop_env(built)


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


def payload_group_item(env, group_id, timeout_s=2.0):
    """Poll every payload shard for one group's parts, or return None.

    The session server picks a group's shard, so look in every shard: a
    group's parts are in exactly one of them, and a group whose payload was
    released leaves nothing in any.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        for channel in env.payload_channels:
            if channel.qsize(group_id):
                return channel.get_nowait(key=group_id)
        time.sleep(0.05)
    return None


def _make_worker(worker_cls, verify, base_url):
    """A loop worker built without Worker.__init__, set up against the
    session server at ``base_url``.

    return_logprobs starts as AgentLoopWorker.__init__ sets it, and
    init_worker applies the session server rule: with ``verify`` off, the
    worker must not ask for agent-side logprobs.
    """
    cfg = OmegaConf.create(
        {
            "algorithm": {"group_size": GROUP_SIZE, "recompute_logprobs": False},
            "session_server": {"enable": False, "verify_assembly": verify},
        }
    )
    # Validate with the path disabled: that fills the session_server
    # defaults without the enabled-path checks, which read the runner,
    # actor and rollout config these tests do not build.
    cfg = validate_session_server_cfg(cfg)
    cfg.session_server.enable = True
    worker = worker_cls.__new__(worker_cls)
    worker.print_outputs = False
    worker.cfg = cfg
    worker.return_logprobs = not cfg.algorithm.recompute_logprobs
    worker.extra_keys_turn = None
    worker.extra_keys_traj = None
    worker._session_binding = contextvars.ContextVar(
        "rlinf_agent_session_binding", default=None
    )
    worker.init_worker(
        None,
        None,
        None,
        None,
        None,
        session_server_urls=[base_url],
        session_server_generate_timeout_s=30.0,
        session_server_keepalive_timeout_s=4.0,
    )
    return worker


class TestSessionServerConfig:
    """The config checks, which build no environment."""

    @staticmethod
    def _session_server_cfg(**session_server):
        return OmegaConf.create({"session_server": {"enable": False, **session_server}})

    @pytest.mark.parametrize("enable", [False, True])
    def test_chat_template_kwargs_must_be_a_mapping(self, enable):
        """A non-mapping chat_template_kwargs is refused by name, with the
        flag off as well: the check runs before the value is converted."""
        cfg = self._session_server_cfg(enable=enable, chat_template_kwargs=123)
        with pytest.raises(ValueError, match="chat_template_kwargs must be a mapping"):
            validate_session_server_cfg(cfg)

    def test_reasoning_eval_rejects_an_enabled_session_server(self):
        """Only the asynchronous training runner drives the session servers:
        an eval run with them enabled would wait for closes no agent sends."""
        cfg = OmegaConf.create(
            {
                "runner": {"task_type": "reasoning_eval"},
                "session_server": {"enable": True},
            }
        )
        with pytest.raises(ValueError, match="is not supported for runner.task_type"):
            validate_reasoning_eval_cfg(cfg)


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
                    "is_end": {0: False, 1: False},
                    "extra_fields_train": {"idx_to_sub_traj": [0, 0]},
                },
                1: {
                    "kept_lens": {0: 8},
                    "continues_prev": [False],
                    "is_end": {0: False},
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

    def test_retried_close_and_reward_after_delivery_answer_success(self, env):
        """The client resends a close or reward whose response was lost.

        The group is delivered by then and its sessions are gone, so the
        retried call has nothing left to do; it must still succeed, and it
        must not make the buffer see a second record for the group.
        """
        request = next_request(env)
        group_id, _turns, sids = run_group(env, request, rewards=(1.0, 0.5))
        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        assert delivery.group_id == group_id

        # run_group's reward on the last session completed the group: that
        # is the call whose response the client can lose.
        retried_reward = reward(env, sids[1], 0.5)
        assert retried_reward.status_code == 200, retried_reward.text
        assert retried_reward.json() == {"group_complete": True}
        retried_close = close(env, sids[0])
        assert retried_close.status_code == 200, retried_close.text
        assert retried_close.json() == {}

        # One record for the group, and only one.
        assert poll_channel(env.rollout, timeout_s=1) is None


class TestAgentLoopSessionTransport:
    """The agent loop over the session server, driven end to end.

    With verify_assembly off, the session server keeps logprobs out of its
    responses (training reads them from the payload), so the agent loop
    must not ask for agent-side logprobs.
    """

    @pytest.mark.parametrize("verify", [False, True])
    def test_group_runs_and_delivers_with_logprob_rule(self, request, cluster, verify):
        env = request.getfixturevalue("verify_env" if verify else "env")
        rollout_request = next_request(env)
        group_id = rollout_request.group_ids[0]
        prompt_ids = rollout_request.input_ids[0]

        worker = _make_worker(TwoTurnLogprobLoopWorker, verify, env.base_url)

        asyncio.run(
            worker.run_agentloop_rollout_group(
                prompt_ids,
                rollout_request.answers[0],
                GROUP_SIZE,
                env.rollout,
                group_id,
                TASK_NAME,
            )
        )
        # The session-server rule under test: agent-side logprobs are
        # collected only for the assembly check.
        assert worker.return_logprobs is verify

        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        assert delivery.group_id == group_id
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        assert isinstance(payload, GroupPayload)
        result = assemble_group_payload(payload, return_logprobs=True)
        # Every turn of both trajectories carries per-token logprobs.
        assert result.rollout_logprobs is not None
        assert [len(lp) for lp in result.rollout_logprobs] == result.response_lengths
        if verify:
            # This agent marks every turn's answer as ended; the assembled
            # result must carry those flags, not a default.
            assert result.is_end == [True] * result.num_sequence
            verify_record = next(part.verify for part in payload.parts if part.last)
            verify_group_assembly(result, verify_record, env.server_cfg)


class TestSessionServerClientRetry:
    """Connection failures on the idempotent lifecycle calls are resent.

    A pooled connection the server has just closed surfaces as a reset on
    the next request; without the resend, one such connection would fail
    the trajectory and, through the agent loop, the whole run.
    """

    class _StubClient:
        """An HTTP client that fails its first ``failures`` calls."""

        def __init__(self, failures, error, response):
            self.calls = 0
            self._failures = failures
            self._error = error
            self._response = response

        async def async_post(self, path, body, timeout_s=None):
            self.calls += 1
            if self.calls <= self._failures:
                raise self._error
            return self._response

    @staticmethod
    def _transport(stub):
        transport = SessionServerClient(
            ["http://127.0.0.1:1"], generate_timeout_s=30.0, keepalive_timeout_s=4.0
        )
        # The pool of real clients is unused: every call goes to the stub.
        transport._clients[0] = stub
        return transport

    @pytest.mark.parametrize(
        "error",
        [aiohttp.ClientOSError("reset by peer"), aiohttp.ServerDisconnectedError()],
        ids=["reset", "disconnected"],
    )
    def test_close_is_resent(self, error):
        stub = self._StubClient(failures=1, error=error, response={})
        asyncio.run(self._transport(stub).close(1, "sid", {}))
        assert stub.calls == 2

    def test_reward_and_abort_are_resent(self):
        for path in ("reward", "abort"):
            stub = self._StubClient(
                failures=1, error=aiohttp.ClientOSError("x"), response={}
            )
            transport = self._transport(stub)
            if path == "reward":
                asyncio.run(transport.reward(1, "sid", 1.0))
            else:
                asyncio.run(transport.abort(1, "reason"))
            assert stub.calls == 2

    def test_create_and_generate_are_not_resent(self):
        """A resent create opens a second session and a resent generate
        samples the turn again, so neither may retry."""
        stub = self._StubClient(
            failures=1, error=aiohttp.ClientOSError("x"), response={}
        )
        with pytest.raises(aiohttp.ClientOSError):
            asyncio.run(
                self._transport(stub).create_session(1, 0, GROUP_SIZE, TASK_NAME)
            )
        assert stub.calls == 1

        stub = self._StubClient(
            failures=1, error=aiohttp.ClientOSError("x"), response={}
        )
        with pytest.raises(aiohttp.ClientOSError):
            asyncio.run(self._transport(stub).generate(1, "sid", [1, 2]))
        assert stub.calls == 1


class TwoTurnLogprobLoopWorker(MultiAgentLoopWorker):
    """Two generate turns per trajectory, with the per-token logprob
    check of Search-R1.

    Turn one continues into turn two (the tool response becomes the next
    turn's input); the second generate ends the trajectory. The rewards
    differ per trajectory so the equal-reward rejection lets the group
    through.
    """

    async def pre_process_query(self, prompt_ids, answer):
        return list(prompt_ids), {}

    async def generate_llm_response(
        self, generate_context, trace_prints, problem_prompt_ids, turn_prompt_ids
    ):
        result = await self.generate(turn_prompt_ids)
        response_ids = result["output_ids"]
        logprobs = result.get("logprobs") if self.return_logprobs else None
        if self.return_logprobs and (
            logprobs is None or len(logprobs) != len(response_ids)
        ):
            raise ValueError("one rollout logprob per generated token is required")
        binding = self._session_binding.get()
        output = AgentLoopOutput(
            prompt_ids=turn_prompt_ids,
            response_ids=response_ids,
            response_logprobs=logprobs,
            is_end=True,
            reward_score=1.0 if binding.traj_index == 0 else 0.5,
        )
        generate_context["turns"] = generate_context.get("turns", 0) + 1
        return (
            generate_context["turns"] < 2,
            response_ids,
            "",
            output,
        )

    async def generate_tool_response(
        self,
        generate_context,
        trace_prints,
        problem_prompt_ids,
        turn_prompt_ids,
        llm_response_ids,
        llm_response_text,
    ):
        return True, turn_prompt_ids + llm_response_ids + [77]

    async def post_process_query(self, generate_context, output):
        return output


class TestAbort:
    def test_late_response_after_abort_is_dropped_by_the_buffer(self, env):
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)

        env.mock.latency_s = 0.5
        outcome = {}

        def slow_generate():
            outcome["response"] = generate(env, sid, [1, 2, 3])

        before = len(env.mock.request_log)
        thread = threading.Thread(target=slow_generate)
        thread.start()
        # Wait until the generate reached the engine: the mock logs a request
        # before it starts its latency, so the request is in flight.
        deadline = time.monotonic() + 10
        while len(env.mock.request_log) == before:
            assert time.monotonic() < deadline, "the generate never reached the engine"
            time.sleep(0.01)
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
        assert payload_group_item(env, group_id) is None
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
        assert payload_group_item(env, group_id) is None
        assert poll_channel(env.rollout, timeout_s=2) is None


class TestTurnClassification:
    """The turn-by-turn prefix classification of one trajectory."""

    @staticmethod
    def _turns(third_tail):
        """Three turns: 10 tokens in and 5 out, then 20 in and 10 out.

        ``third_tail`` is what the third turn's input adds after the second
        turn's input, so a test picks whether the third turn kept the whole
        second output or broke inside it.
        """
        first_input = list(range(10))
        first_output = list(range(100, 105))
        second_input = first_input + first_output + list(range(200, 205))
        second_output = list(range(300, 310))
        return [
            new_turn_record(first_input, None, None, None),
            new_turn_record(second_input, first_input, first_output, None),
            new_turn_record(
                second_input + third_tail, second_input, second_output, None
            ),
        ]

    def test_third_turn_that_holds_the_second_output_continues(self):
        """A third turn whose prompt holds the second turn's whole output
        continues the second turn."""
        turns = self._turns(list(range(300, 310)) + list(range(400, 402)))
        continues, counts = classify_turns(turns, [5, 10, 2], set(), None, 0, 0)
        assert continues == [False, True, True]
        assert counts.continuous == 2

    def test_third_turn_that_breaks_inside_the_second_output_is_underreported(self):
        """A third turn that breaks inside the second turn's kept output is
        an under-reported truncation.

        The third prompt holds only 5 of the 10 output tokens that the close
        declares as kept for the second turn. The check compares the third
        turn with the second; compared with the first turn, the same prompt
        would pass as a continuation.
        """
        turns = self._turns(list(range(300, 305)) + list(range(400, 403)))
        with pytest.raises(UnderreportedTruncation):
            classify_turns(turns, [5, 10, 2], set(), None, 0, 0)

    def test_third_turn_that_breaks_inside_the_second_output_rewrites_for_chat(self):
        """For a chat agent the same break counts as a rewritten history.

        A chat agent that shortened the second answer sends such a prompt,
        so the turn starts a new segment instead of failing the close.
        """
        turns = self._turns(list(range(300, 305)) + list(range(400, 403)))
        continues, counts = classify_turns(turns, [5, 10, 2], set(), "chat", 0, 0)
        assert continues == [False, True, False]
        assert counts.rewritten == 1


class TestRewriteMetric:
    """prefix_rewritten counts the turns that are recorded."""

    def test_a_rewrite_whose_forward_fails_is_counted_once(self, env, monkeypatch):
        """A request that rewrites history counts only when its turn is
        recorded: a failed forward writes no turn, and the retry that
        succeeds counts the rewrite once."""
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)
        assert generate(env, sid, [1, 2, 3]).status_code == 200
        env.server.get_metrics().wait()

        # A request that breaks inside the first turn's input is a rewrite.
        abort = {"value": True}
        response_body = env.mock._generate_response

        def response_with_abort(body):
            response = response_body(body)
            if abort["value"]:
                response["meta_info"]["finish_reason"] = {"type": "abort"}
            return response

        monkeypatch.setattr(env.mock, "_generate_response", response_with_abort)
        assert generate(env, sid, [7, 8, 9]).status_code == 503
        metrics = env.server.get_metrics().wait()[0]
        assert metrics.get("session_server/prefix_rewritten", 0) == 0

        abort["value"] = False
        assert generate(env, sid, [7, 8, 9]).status_code == 200
        # The rewrite is counted with the turn it belongs to.
        metrics = env.server.get_metrics().wait()[0]
        assert metrics.get("session_server/prefix_rewritten", 0) == 1
        assert close(env, sid).status_code == 200


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
        assert payload_group_item(env, group_id) is None
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

    def test_close_rejects_a_non_bool_is_end(self, env):
        """is_end must map a turn index to a bool: it becomes each sequence's
        ended flag, so another shape is rejected, and the session still
        closes cleanly afterwards."""
        sid = create_session(env, 4241, 0)
        generate(env, sid, [7, 8, 9])

        resp = requests.post(
            f"{env.base_url}/sessions/{sid}/close",
            json={"is_end": {"0": 1}},
            timeout=30,
        )
        assert resp.status_code == 400
        assert "is_end[0] must be a bool" in resp.json()["error"]

        resp = requests.post(
            f"{env.base_url}/sessions/{sid}/close",
            json={"is_end": [True]},
            timeout=30,
        )
        assert resp.status_code == 400
        assert "is_end must map" in resp.json()["error"]

        assert close(env, sid).status_code == 200


class TestNotTraining:
    def test_dropped_turn_is_assembled_out_and_next_turn_pairs_back(self):
        """A not_training turn stays in the payload but yields no sequence,
        and the turn after it pairs with the last kept turn.

        The flag is set server-side by the chat entry's rollback, never by
        a close verdict; this exercises the assembly contract directly.
        """
        loaded, engine = 3, 5
        first_input = [1, 2, 3]
        first_output = [10, 11]
        retry_output = [40, 41]
        third_output = [20, 21]

        traj0_turn0 = PayloadPart(
            group_id=0,
            traj_index=0,
            turn_index=0,
            input_ids=first_input,
            output_ids=first_output,
            logprobs=[-0.1, -0.2],
            version_start=loaded,
            weight_version=engine,
            finish_reason={"type": "stop"},
        )
        traj0_turn1 = PayloadPart(
            group_id=0,
            traj_index=0,
            turn_index=1,
            input_ids=first_input + first_output,
            output_ids=retry_output,
            logprobs=[-0.3, -0.4],
            version_start=loaded,
            weight_version=engine,
            finish_reason={"type": "stop"},
        )
        # Continues turn 0, not the dropped turn 1.
        traj0_turn2 = PayloadPart(
            group_id=0,
            traj_index=0,
            turn_index=2,
            input_ids=first_input + first_output,
            output_ids=third_output,
            logprobs=[-0.5, -0.6],
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
            rewards=[1.0],
            verdicts={
                0: {
                    "kept_lens": {0: 2, 1: 2, 2: 2},
                    "continues_prev": [False, False, True],
                    "not_training": {1: True},
                    "is_end": {0: False, 1: False, 2: False},
                    "extra_fields_train": {},
                },
            },
        )
        payload = GroupPayload(
            group_id=0,
            parts=[traj0_turn2, finalize, traj0_turn0, traj0_turn1],
        )

        result = assemble_group_payload(payload, return_logprobs=True)
        assert result.num_sequence == 2
        assert result.idx_to_traj == [0, 0]
        assert result.input_ids == [
            first_input + first_output,
            first_input + first_output + third_output,
        ]
        assert result.response_lengths == [2, 2]

    def test_extra_fields_follow_the_trained_turns(self):
        """An extra training field covers the trained turns, not the dropped
        one, and rides the payload in turn order.

        Turn 1 (the retry) is dropped, so the two values belong to turns 0
        and 2.
        """
        loaded, engine = 3, 5
        first_input = [1, 2, 3]
        first_output = [10, 11]

        def turn(index, output_ids, input_ids):
            return PayloadPart(
                group_id=0,
                traj_index=0,
                turn_index=index,
                input_ids=input_ids,
                output_ids=output_ids,
                logprobs=[-0.1, -0.2],
                version_start=loaded,
                weight_version=engine,
                finish_reason={"type": "stop"},
            )

        turns = [
            turn(0, first_output, first_input),
            turn(1, [40, 41], first_input + first_output),
            turn(2, [20, 21], first_input + first_output),
        ]
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
            rewards=[1.0],
            verdicts={
                0: {
                    "kept_lens": {0: 2, 1: 2, 2: 2},
                    "continues_prev": [False, False, True],
                    "not_training": {1: True},
                    "is_end": {0: False, 1: False, 2: False},
                    "extra_fields_train": {"idx_to_sub_traj": [0, 7]},
                },
            },
        )
        payload = GroupPayload(group_id=0, parts=[*turns, finalize])

        result = assemble_group_payload(payload, return_logprobs=True)
        assert result.num_sequence == 2
        assert result.extra_fields_train["idx_to_sub_traj"] == [0, 7]


class TestChatCompletions:
    """The chat entry: TITO, parsing, errors, concurrency, rollback."""

    @staticmethod
    def _chat(env, sid, body):
        return requests.post(
            f"{env.base_url}/sessions/{sid}/v1/chat/completions",
            json=body,
            timeout=30,
        )

    @staticmethod
    def _chat_in_background(env, sid, body):
        """Start a chat request the mock holds open, and return its thread.

        Returns once the mock has logged the forwarded generate, which means
        the session server holds the session: a second request now sees the
        session busy.
        """
        env.mock.latency_s = 1.0
        responses = []

        def _send():
            responses.append(TestChatCompletions._chat(env, sid, body))

        before = len(env.mock.request_log)
        thread = threading.Thread(target=_send)
        thread.start()
        deadline = time.monotonic() + 10
        while len(env.mock.request_log) == before:
            assert time.monotonic() < deadline, (
                "the chat request never reached the engine"
            )
            time.sleep(0.01)
        return thread, responses

    @staticmethod
    def _messages(question="What is RL?", tool_answer="RL is ..."):
        messages = [
            {"role": "system", "content": "Answer using the search tool."},
            {"role": "user", "content": question},
        ]
        return messages

    def test_two_turns_with_tool_message_continue(self, chat_env):
        env = chat_env
        env.mock.scripted_texts = [
            '<tool_call>\n{"name": "search", "arguments": {"query": "RL"}}\n</tool_call>',
            "Reinforcement learning is a learning paradigm.",
        ]
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)

        tools = CHAT_TOOLS
        messages = self._messages()
        first = self._chat(env, sid, {"messages": messages, "tools": tools})
        assert first.status_code == 200, first.text
        body = first.json()
        assistant = body["choices"][0]["message"]
        assert body["choices"][0]["finish_reason"] == "tool_calls"
        assert assistant["tool_calls"][0]["function"]["name"] == "search"
        assert body["usage"]["prompt_tokens"] > 0

        tool_messages = [
            *messages,
            assistant,
            {"role": "tool", "name": "search", "content": "RL is a paradigm."},
        ]
        second = self._chat(env, sid, {"messages": tool_messages, "tools": tools})
        assert second.status_code == 200, second.text
        second_message = second.json()["choices"][0]["message"]
        assert second_message["content"].startswith("Reinforcement learning")
        assert "tool_calls" not in second_message

        # Both turns ride the payload, the second continuing the first.
        assert close(env, sid).status_code == 200
        sid1 = create_session(env, group_id, 1)
        generate(env, sid1, [10, 11])
        assert close(env, sid1).status_code == 200
        assert reward(env, sid, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is True
        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        result = assemble_group_payload(payload, return_logprobs=True)
        assert result.num_sequence == 3  # two chat turns + one plain turn
        # The turn classification is counted at close.
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/prefix_continuous"] >= 1
        assert metrics["session_server/chat_tool_call_responses"] >= 1

    def test_unreadable_tool_call_counts_and_keeps_the_text(self, chat_env):
        """A tool-call block the parser cannot read is a parse failure, and
        the whole text answers as content.

        The parser only warns about the broken block and reports no call;
        the content it returns stops before that block, so keeping its text
        would drop what the model wrote.
        """
        env = chat_env
        broken = (
            '<tool_call>\n{"name": "search", "arguments": {"query": "RL"\n</tool_call>'
        )
        env.mock.scripted_texts = [broken]
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)

        response = self._chat(
            env, sid, {"messages": self._messages(), "tools": CHAT_TOOLS}
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message["content"] == broken
        assert "tool_calls" not in message
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/chat_tool_call_parse_failed"] == 1

    def test_two_calls_of_one_tool_get_distinct_ids(self, chat_env):
        env = chat_env
        # Two calls of the same tool: the agent answers both by tool_call_id,
        # so the ids must differ even though the tool is the same.
        call = '<tool_call>\n{"name": "search", "arguments": {"query": "RL"}}\n</tool_call>'
        env.mock.scripted_texts = [call + call]
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)

        response = self._chat(
            env, sid, {"messages": self._messages(), "tools": CHAT_TOOLS}
        )
        assert response.status_code == 200, response.text
        calls = response.json()["choices"][0]["message"]["tool_calls"]
        assert len(calls) == 2
        assert len({call["id"] for call in calls}) == 2

    def test_retry_of_last_request_rolls_back_and_delivers_once(self, chat_env):
        env = chat_env
        env.mock.scripted_texts = ["First answer."]
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)
        messages = self._messages()

        first = self._chat(env, sid, {"messages": messages})
        assert first.status_code == 200, first.text
        # The same request again: the agent never used the first answer.
        retry = self._chat(env, sid, {"messages": messages})
        assert retry.status_code == 200, retry.text

        # One value per trained turn: the abandoned first turn is not one.
        assert (
            close(env, sid, extra_fields_train={"idx_to_sub_traj": [0]}).status_code
            == 200
        )
        sid1 = create_session(env, group_id, 1)
        generate(env, sid1, [10, 11])
        assert (
            close(env, sid1, extra_fields_train={"idx_to_sub_traj": [0]}).status_code
            == 200
        )
        assert reward(env, sid, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is True
        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        result = assemble_group_payload(payload, return_logprobs=True)
        # Three turns in the payload (first attempt, retry, plain) but the
        # rolled-back one trains nothing.
        assert result.num_sequence == 2
        # One declared value per trained turn, in trajectory order.
        assert result.extra_fields_train["idx_to_sub_traj"] == [0, 0]

    def test_stored_history_keeps_reasoning_the_request_dropped(self, chat_env_glm):
        """What later requests are matched against is the server's own copy
        of the assistant message: a client that drops reasoning_content when
        it replays that message does not erase it, so a later request that
        sends the original message back still continues the conversation
        instead of re-rendering it.
        """
        env = chat_env_glm
        env.mock.scripted_texts = [
            "first thoughts</think>First answer.",
            "second thoughts</think>Second answer.",
            "third thoughts</think>Third answer.",
        ]
        sid = create_session(env, 4321, 0)
        messages = self._messages()

        first = self._chat(env, sid, {"messages": messages})
        assert first.status_code == 200, first.text
        assistant = first.json()["choices"][0]["message"]
        assert assistant["reasoning_content"] == "first thoughts"

        # The client clears the reasoning when it replays the message.
        dropped = {k: v for k, v in assistant.items() if k != "reasoning_content"}
        tool = {"role": "tool", "name": "search", "content": "RL is a paradigm."}
        second = self._chat(env, sid, {"messages": [*messages, dropped, tool]})
        assert second.status_code == 200, second.text
        # Drained here: the close below then counts this session's turns
        # alone.
        env.server.get_metrics().wait()

        # A later request brings the reasoning back, as a client that kept
        # the original message would.
        third = self._chat(
            env,
            sid,
            {
                "messages": [
                    *messages,
                    assistant,
                    tool,
                    second.json()["choices"][0]["message"],
                    {"role": "user", "content": "And which algorithms?"},
                ]
            },
        )
        assert third.status_code == 200, third.text
        assert close(env, sid).status_code == 200
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/prefix_rewritten"] == 0

    def test_replayed_messages_match_with_empty_content_and_tool_calls(self, chat_env):
        """A replayed assistant message matches whether the client sends an
        empty field or leaves it out: None, missing and "" all mean "no
        content", and a missing tool_calls list means no tool calls.
        """
        env = chat_env
        tool_call = (
            '<tool_call>\n{"name": "search", "arguments": {"query": "RL"}}\n'
            "</tool_call>"
        )
        messages = self._messages()
        tool = {"role": "tool", "name": "search", "content": "RL is a paradigm."}

        def replay(group_id, scripted, mutate):
            env.mock.scripted_texts = list(scripted)
            sid = create_session(env, group_id, 0)
            first = self._chat(env, sid, {"messages": messages, "tools": CHAT_TOOLS})
            assert first.status_code == 200, first.text
            replayed = mutate(first.json()["choices"][0]["message"])
            # Drained here: the close below then counts this session's turns
            # alone.
            env.server.get_metrics().wait()
            second = self._chat(
                env,
                sid,
                {"messages": [*messages, replayed, tool], "tools": CHAT_TOOLS},
            )
            assert second.status_code == 200, second.text
            assert close(env, sid).status_code == 200
            metrics = env.server.get_metrics().wait()[0]
            assert metrics["session_server/prefix_rewritten"] == 0
            assert metrics["session_server/prefix_continuous"] == 1

        # A tool-call answer carries no content: replayed as an explicit
        # null, then with the field left out. One scripted text per case:
        # the mock repeats its last text once a script runs out.
        replay(4322, [tool_call], lambda m: {**m, "content": None})
        replay(
            4323,
            [tool_call],
            lambda m: {k: v for k, v in m.items() if k != "content"},
        )
        # An answer with no tool calls: replayed with an empty list.
        replay(
            4324,
            ["Plain answer."],
            lambda m: {**m, "tool_calls": []},
        )

    def test_changing_the_tools_rerenders_the_conversation(self, chat_env):
        """The stored tokens were rendered under the tool list the session
        started with, so a request that changes that list cannot continue
        them: the conversation is rendered again from its start.
        """
        from rlinf.models.tokenization.hf import hf_tokenizer

        env = chat_env
        env.mock.scripted_texts = ["First answer.", "Second answer."]
        sid = create_session(env, 4325, 0)
        messages = self._messages()
        first = self._chat(env, sid, {"messages": messages, "tools": CHAT_TOOLS})
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]
        follow_up = {"role": "user", "content": "And which algorithms?"}
        extended_tools = [
            *CHAT_TOOLS,
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look a term up.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]

        second = self._chat(
            env,
            sid,
            {"messages": [*messages, reply, follow_up], "tools": extended_tools},
        )
        assert second.status_code == 200, second.text

        # What was forwarded is the whole conversation rendered under the
        # new tool list, not the stored tokens plus an increment.
        tokenizer = hf_tokenizer(_chat_model_dir("qwen"), trust_remote_code=False)
        expected = tokenizer.apply_chat_template(
            [*messages, reply, follow_up],
            tools=extended_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        forward = [r for r in env.mock.request_log if r["path"] == "/generate"][-1]
        assert forward["body"]["input_ids"] == list(expected)

        assert close(env, sid).status_code == 200
        metrics = env.server.get_metrics().wait()[0]
        # The re-render starts a new segment instead of pairing with the
        # turn rendered under the old list.
        assert metrics["session_server/prefix_rewritten"] == 1

    def test_a_rolled_back_turn_keeps_the_tools_of_the_restored_tokens(self, chat_env):
        """A rollback restores the tool list the checkpoint's tokens were
        rendered with, so the retried request is rendered under the tools it
        sends, not under the ones those tokens saw.

        Without that, the retry appends its messages to tokens rendered under
        another tool list, and the engine is asked with tool definitions that
        do not match the prompt it continues.
        """
        from rlinf.models.tokenization.hf import hf_tokenizer

        env = chat_env
        env.mock.scripted_texts = ["First answer.", "Second answer."]
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)
        messages = self._messages()
        extended_tools = [
            *CHAT_TOOLS,
            {
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "Look a term up.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
        ]
        first = self._chat(env, sid, {"messages": messages, "tools": CHAT_TOOLS})
        assert first.status_code == 200, first.text
        reply = first.json()["choices"][0]["message"]
        body = {
            "messages": [
                *messages,
                reply,
                {"role": "user", "content": "And which algorithms?"},
            ],
            "tools": extended_tools,
        }

        second = self._chat(env, sid, body)
        assert second.status_code == 200, second.text
        forwarded = [r for r in env.mock.request_log if r["path"] == "/generate"][-1]
        tokenizer = hf_tokenizer(_chat_model_dir("qwen"), trust_remote_code=False)
        expected = tokenizer.apply_chat_template(
            body["messages"],
            tools=extended_tools,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=False,
        )
        # The changed tool list governs this turn: the whole conversation is
        # rendered under it.
        assert forwarded["body"]["input_ids"] == list(expected)

        # The same request again, which the chat entry rolls back: the retry
        # must be rendered exactly as the first attempt was.
        retry = self._chat(env, sid, body)
        assert retry.status_code == 200, retry.text
        retried = [r for r in env.mock.request_log if r["path"] == "/generate"][-1]
        assert retried["body"]["input_ids"] == forwarded["body"]["input_ids"]

    def test_a_consumed_checkpoint_rolls_back_once(self, chat_env, monkeypatch):
        """A rollback consumes the checkpoint, so a request that repeats an
        earlier turn afterwards is a rewrite: no turn is dropped a second
        time."""
        env = chat_env
        env.mock.scripted_texts = [
            "First answer.",
            "Second answer.",
            "Third answer.",
        ]
        sid = create_session(env, 4326, 0)
        messages = self._messages()
        first = self._chat(env, sid, {"messages": messages})
        assert first.status_code == 200, first.text
        turn_one = [
            *messages,
            first.json()["choices"][0]["message"],
            {"role": "user", "content": "And which algorithms?"},
        ]
        second = self._chat(env, sid, {"messages": turn_one})
        assert second.status_code == 200, second.text

        # The retry of the second turn rolls that turn back, and then its
        # forward fails.
        abort = {"value": True}
        response_body = env.mock._generate_response

        def response_with_abort(body):
            response = response_body(body)
            if abort["value"]:
                response["meta_info"]["finish_reason"] = {"type": "abort"}
            return response

        monkeypatch.setattr(env.mock, "_generate_response", response_with_abort)
        retry = self._chat(env, sid, {"messages": turn_one})
        assert retry.status_code == 503, retry.text

        # The opening turn's request again: the checkpoint is gone, so this
        # turn is rendered as a rewrite, and the only dropped turn is still
        # the one the client abandoned.
        abort["value"] = False
        rewrite = self._chat(env, sid, {"messages": messages})
        assert rewrite.status_code == 200, rewrite.text
        # One declared value per trained turn: the two that trained.
        assert (
            close(env, sid, extra_fields_train={"idx_to_sub_traj": [0, 0]}).status_code
            == 200
        )

    def test_a_retry_after_a_failed_retry_continues(self, chat_env, monkeypatch):
        """A failed retry drops its turn; the next request repeats it and
        continues the conversation, so the first and third turns train and
        the abandoned second one does not."""
        env = chat_env
        env.mock.scripted_texts = ["First answer.", "Second answer.", "Third answer."]
        sid = create_session(env, 4327, 0)
        messages = self._messages()
        first = self._chat(env, sid, {"messages": messages})
        assert first.status_code == 200, first.text
        turn_one = [
            *messages,
            first.json()["choices"][0]["message"],
            {"role": "user", "content": "And which algorithms?"},
        ]
        assert self._chat(env, sid, {"messages": turn_one}).status_code == 200

        abort = {"value": True}
        response_body = env.mock._generate_response

        def response_with_abort(body):
            response = response_body(body)
            if abort["value"]:
                response["meta_info"]["finish_reason"] = {"type": "abort"}
            return response

        monkeypatch.setattr(env.mock, "_generate_response", response_with_abort)
        assert self._chat(env, sid, {"messages": turn_one}).status_code == 503

        # The same request again, served this time: the abandoned turn is
        # not retrained, the new one is.
        abort["value"] = False
        env.server.get_metrics().wait()
        again = self._chat(env, sid, {"messages": turn_one})
        assert again.status_code == 200, again.text
        assert (
            close(env, sid, extra_fields_train={"idx_to_sub_traj": [0, 0]}).status_code
            == 200
        )
        metrics = env.server.get_metrics().wait()[0]
        # The new turn continued the first one; the abandoned turn is not
        # part of the chain.
        assert metrics["session_server/prefix_continuous"] == 1
        assert metrics["session_server/prefix_rewritten"] == 0

    def test_rewritten_history_marks_prefix_rewritten(self, chat_env):
        env = chat_env
        env.mock.scripted_texts = [
            "First answer.",
            "Second answer.",
        ]
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)
        messages = self._messages()

        assert self._chat(env, sid, {"messages": messages}).status_code == 200
        rewritten = [
            {"role": "system", "content": "A different instruction."},
            {"role": "user", "content": "Another question?"},
        ]
        resp = self._chat(env, sid, {"messages": rewritten})
        assert resp.status_code == 200, resp.text

        assert close(env, sid).status_code == 200
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/prefix_rewritten"] >= 1

    def test_answer_is_capped_to_the_room_the_prompt_leaves(self, chat_env):
        """A turn cannot grow past runner.seq_length: the entry lowers
        max_new_tokens to what the prompt leaves.
        """
        env = chat_env
        env.mock.scripted_texts = ["Answer."]
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)
        # The longest filler that still fits, sized with the tokenizer the
        # server renders with, so the prompt crowds the limit whatever the
        # tokenizer makes of the words.
        from rlinf.models.tokenization.hf import hf_tokenizer

        tokenizer = hf_tokenizer(_chat_model_dir("qwen"), trust_remote_code=False)

        def prompt_len(words: int) -> int:
            return len(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": "word " * words}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=False,
                )
            )

        fits, too_long = 1, SEQ_LENGTH
        while fits < too_long:
            middle = (fits + too_long + 1) // 2
            if prompt_len(middle) < SEQ_LENGTH:
                fits = middle
            else:
                too_long = middle - 1
        filler = "word " * fits

        response = self._chat(
            env, sid, {"messages": [{"role": "user", "content": filler}]}
        )
        assert response.status_code == 200, response.text
        forward = [r for r in env.mock.request_log if r["path"] == "/generate"][-1]
        room = SEQ_LENGTH - len(forward["body"]["input_ids"])
        # The cap is what was asserted: the prompt leaves less than the
        # configured default, and the forwarded limit is exactly the room.
        assert 0 < room < DEFAULT_MAX_NEW_TOKENS
        assert forward["body"]["sampling_params"]["max_new_tokens"] == room

    def test_null_fields_count_as_unset(self, chat_env):
        """A client that serializes every optional parameter as null is
        served the configured sampling instead of a 400 or a 500.

        The group id is outside the buffer worker's sequence, like the
        rejected-request cases below: this scenario admits nothing.
        """
        env = chat_env
        env.mock.scripted_texts = ["Fine."]
        sid = create_session(env, 4311, 0)

        response = self._chat(
            env,
            sid,
            {
                "messages": self._messages(),
                "temperature": None,
                "top_p": None,
                "top_k": None,
                "tool_choice": None,
                "response_format": None,
                "stop": None,
                "max_completion_tokens": None,
                "max_tokens": 4,
            },
        )
        assert response.status_code == 200, response.text
        forward = [r for r in env.mock.request_log if r["path"] == "/generate"][-1]
        # The null max_completion_tokens does not shadow max_tokens, and
        # the null stop adds no stop string.
        assert forward["body"]["sampling_params"]["max_new_tokens"] == 4
        assert "stop" not in forward["body"]["sampling_params"]

    def test_output_ids_without_text_are_decoded(self, chat_env):
        """An engine started without a tokenizer answers ids only.

        The session server decodes them itself, so a tool call parses out of the
        response exactly as it does when the engine sends its own text.
        This scenario admits nothing either; its two sessions share one
        group so both halves compare under the same conditions.
        """
        env = chat_env
        from rlinf.models.tokenization.hf import hf_tokenizer

        text = (
            '<tool_call>\n{"name": "search", "arguments": {"query": "RL"}}\n'
            "</tool_call>"
        )

        env.mock.scripted_texts = [text]
        with_text = self._chat(
            env,
            create_session(env, 4312, 0),
            {"messages": self._messages(), "tools": CHAT_TOOLS},
        )
        assert with_text.status_code == 200, with_text.text

        tokenizer = hf_tokenizer(_chat_model_dir("qwen"), trust_remote_code=False)
        env.mock.scripted_texts = []
        env.mock.scripted_output_ids = tokenizer.encode(text, add_special_tokens=False)
        with_ids = self._chat(
            env,
            create_session(env, 4312, 1),
            {"messages": self._messages(), "tools": CHAT_TOOLS},
        )
        env.mock.scripted_output_ids = []
        assert with_ids.status_code == 200, with_ids.text

        def calls(response):
            message = response.json()["choices"][0]["message"]
            return [
                (call["function"]["name"], call["function"]["arguments"])
                for call in message["tool_calls"]
            ]

        assert with_ids.json()["choices"][0]["finish_reason"] == "tool_calls"
        assert calls(with_ids) == calls(with_text) == [("search", '{"query": "RL"}')]

    def test_prompt_that_does_not_fit_is_rejected(self, chat_env):
        """A prompt that alone exceeds the training length gets a 400 and
        nothing is forwarded or recorded."""
        env = chat_env
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)
        before = len(env.mock.request_log)

        response = self._chat(
            env,
            sid,
            {"messages": [{"role": "user", "content": "word " * (SEQ_LENGTH * 2)}]},
        )
        assert response.status_code == 400, response.text
        error = response.json()["error"]
        assert error["code"] == "context_length_exceeded"
        assert error["type"] == "invalid_request_error"
        # No turn was forwarded, so the group gained no turn.
        assert len(env.mock.request_log) == before

    def test_unclosed_reasoning_marker_keeps_the_whole_answer(self, chat_env):
        """A model without reasoning markers keeps its text whole.

        Qwen2.5 writes ``<think>`` as ordinary text when the prompt asks
        for it; splitting there would put the rest of the turn, tool calls
        included, into reasoning_content and leave the agent an empty
        answer.
        """
        env = chat_env
        tool_call = '<tool_call>\n{"name": "search", "arguments": {"query": "RL"}}\n</tool_call>'
        env.mock.scripted_texts = [f"<think> reasoning without a closer\n{tool_call}"]
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)

        response = self._chat(
            env, sid, {"messages": self._messages(), "tools": CHAT_TOOLS}
        )
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message["tool_calls"][0]["function"]["name"] == "search"
        assert "reasoning_content" not in message
        assert message["content"].startswith("<think> reasoning without a closer")
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/chat_reasoning_responses"] == 0

    def test_reasoning_parsed_from_the_engine_text(self, chat_env_glm):
        """GLM's reasoning is split out; a model without reasoning markers
        keeps its text whole (see the unclosed-marker test)."""
        env = chat_env_glm
        # GLM's generation prompt ends with the reasoning opener, so the
        # engine text starts inside the block: the parser must be built
        # with force_reasoning, as the entry does.
        env.mock.scripted_texts = ["thinking out loud</think>The answer."]
        rollout_request = next_request(env)
        group_id = rollout_request.group_ids[0]
        sid = create_session(env, group_id, 0)
        response = self._chat(env, sid, {"messages": self._messages()})
        assert response.status_code == 200, response.text
        message = response.json()["choices"][0]["message"]
        assert message["content"] == "The answer."
        assert message["reasoning_content"] == "thinking out loud"
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/chat_reasoning_responses"] >= 1

    def test_stream_answers_one_chunk_and_done(self, chat_env):
        env = chat_env
        env.mock.scripted_texts = ["Streamed."]
        request = next_request(env)
        group_id = request.group_ids[0]
        sid = create_session(env, group_id, 0)
        response = self._chat(env, sid, {"messages": self._messages(), "stream": True})
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        text = response.text
        assert text.endswith("data: [DONE]\n\n")
        payload = json.loads(text.split("data: ", 1)[1].split("\n", 1)[0])
        assert payload["object"] == "chat.completion.chunk"
        assert payload["choices"][0]["delta"]["content"].startswith("Streamed")

    @pytest.mark.parametrize(
        "group_id,body_patch",
        [
            (4301, {"n": 2}),
            (4302, {"temperature": 0.5}),
            (4303, {"tool_choice": "required"}),
            (4304, {"response_format": {"type": "json_object"}}),
            (4305, {"chat_template_kwargs": {"clear_thinking": True}}),
        ],
    )
    def test_invalid_requests_answer_400(self, chat_env, group_id, body_patch):
        env = chat_env
        # Group ids outside the buffer worker's sequence: this scenario
        # admits nothing, so it neither finalizes nor shifts the ids the
        # delivery scenarios pin.
        sid = create_session(env, group_id, 0)
        body = {"messages": self._messages(), **body_patch}
        response = self._chat(env, sid, body)
        assert response.status_code == 400, response.text
        assert response.json()["error"]["type"] == "invalid_request_error"

    def test_concurrent_request_on_one_session_conflicts(self, chat_env):
        env = chat_env
        env.mock.scripted_texts = ["Slow answer."]
        # Not admitted by the buffer: the scenario leaves no group pending.
        sid = create_session(env, 4310, 0)
        body = {"messages": self._messages()}

        thread, responses = self._chat_in_background(env, sid, body)
        second = self._chat(env, sid, body)
        env.mock.latency_s = 0.0
        thread.join(timeout=30)

        assert responses[0].status_code == 200
        assert second.status_code == 409, second.text


class ChatCompletionsTestLoopWorker(MultiAgentLoopWorker):
    """The minimal agent loop the chat completions endpoint needs.

    It overrides the hook and nothing else: the group runner opens each
    session, calls :meth:`run_session_trajectory`, and closes the session
    with what it returns. Two chat turns: the first gets a tool call from
    the engine, the second its answer.
    """

    _SEARCH_TOOL = {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search Wikipedia for information about a query.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    _MAX_TURNS = 2

    async def run_session_trajectory(self, binding, input_ids, answer):
        """One trajectory: two chat turns, closed with its extra fields."""
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            base_url=self.session_server_client.session_base_url(
                binding.group_id, binding.session_id
            ),
            api_key="EMPTY",
            max_retries=0,  # a retried request is sampled again
            timeout=30.0,
        )
        messages = [
            {"role": "system", "content": "Answer with the search tool."},
            {"role": "user", "content": "Question: what is RL?"},
        ]
        num_responses = 0
        final_text = ""
        try:
            for _ in range(self._MAX_TURNS):
                response = await client.chat.completions.create(
                    model="policy", messages=messages, tools=[self._SEARCH_TOOL]
                )
                num_responses += 1
                message = response.choices[0].message
                messages.append(message.model_dump(exclude_none=True))
                final_text = message.content or final_text
                if not message.tool_calls:
                    break
                for call in message.tool_calls:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.id,
                            "content": "Reinforcement learning is a paradigm.",
                        }
                    )
        finally:
            await client.close()
        # Different rewards per trajectory: the buffer's equal-reward
        # rejection would otherwise drop the group before delivery.
        reward = 0.5 + 0.25 * binding.traj_index
        close_fields = {
            # One value per response received: one agent per trajectory.
            "extra_fields_train": {"idx_to_sub_traj": [0] * num_responses}
        }
        return None, close_fields, reward


class TestChatCompletionsAgentLoop:
    """The hook drives the chat endpoint; the runner does the rest."""

    def test_two_turns_are_delivered_with_extra_fields(self, chat_env):
        env = chat_env
        # Every conversation opens with a tool call and closes with the
        # answer, whichever order the trajectories interleave in.
        env.mock.scripted_by_turn = (
            '<tool_call>\n{"name": "search", "arguments": {"query": "RL"}}\n</tool_call>',
            "<answer>Reinforcement learning</answer>",
        )
        request = next_request(env)
        group_id = request.group_ids[0]

        worker = ChatCompletionsTestLoopWorker.__new__(ChatCompletionsTestLoopWorker)
        worker.print_outputs = False
        worker.return_logprobs = False
        worker.extra_keys_turn = None
        worker.extra_keys_traj = None
        worker.cfg = validate_session_server_cfg(
            OmegaConf.create(
                {
                    "agentloop": {"max_turns": 2},
                    "session_server": {"enable": False},
                    "data": {"max_prompt_length": 512},
                    "runner": {"seq_length": 1024},
                    "rollout": {"return_logprobs": True},
                }
            )
        )
        worker._session_binding = contextvars.ContextVar(
            "rlinf_agent_session_binding", default=None
        )
        worker.init_worker(
            None,
            None,
            {},
            {},
            None,
            session_server_urls=[env.base_url],
            session_server_generate_timeout_s=30.0,
            session_server_keepalive_timeout_s=4.0,
        )

        asyncio.run(
            worker.run_agentloop_rollout_group(
                request.input_ids[0],
                request.answers[0],
                GROUP_SIZE,
                env.rollout,
                group_id,
                TASK_NAME,
            )
        )

        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        result = assemble_group_payload(payload, return_logprobs=True)
        # Two trajectories, two chat turns each, and the second turn of each
        # continues the first.
        assert result.num_sequence == 4
        assert sorted(set(result.idx_to_traj)) == [0, 1]
        # The extra field covers the trained turns, one per response.
        assert result.extra_fields_train["idx_to_sub_traj"] == [0] * 4
        metrics = env.server.get_metrics().wait()[0]
        assert metrics["session_server/prefix_continuous"] >= 2


def _minimal_server_cfg(agentic_tokenizer: str):
    """A session server config, with the given agentic tokenizer family."""
    cfg = OmegaConf.create(
        {
            "runner": {"seq_length": SEQ_LENGTH},
            "algorithm": {
                "group_size": GROUP_SIZE,
                "sampling_params": {"do_sample": False, "max_new_tokens": 8},
            },
            "rollout": {
                "return_logprobs": True,
                "router": {"max_idle_secs": 600.0},
                "model": {"model_path": _chat_model_dir()},
            },
            "session_server": {
                "enable": False,
                "host": "127.0.0.1",
                "port": find_available_port(),
                "payload_put_timeout_s": 300.0,
                "max_pending_payload_bytes": 512 * 1024 * 1024,
                "generate_timeout_s": 600.0,
                "per_task": {},
                "verify_assembly": False,
                "chat_template_kwargs": {},
            },
            "agentloop": {"agentic_tokenizer": agentic_tokenizer},
        }
    )
    cfg = validate_session_server_cfg(cfg)
    cfg.session_server.enable = True
    return cfg


class TestStartupFailure:
    """A server whose chat entry cannot be built fails while starting.

    The worker object is driven directly, with the serving task and the
    waiter in one event loop: in a worker group, a worker method that raises
    signals the main process, which shuts the cluster down.
    """

    def test_a_bad_chat_entry_fails_serving(self):
        """A chat entry that cannot be built ends serving with that error.

        wait_until_serving() raises it, so the runner's liveness monitor ends
        the run instead of the agent loop waiting for a server that never
        listened; the cause is the startup error itself.
        """
        if not _chat_model_dir("qwen"):
            pytest.skip("set RLINF_TEST_QWEN25_MODEL_DIR to run the chat tests")
        server = SessionServerWorker.__new__(SessionServerWorker)
        server._cfg = _minimal_server_cfg("no_such_family")
        server._server_index = 0
        server._host = "127.0.0.1"
        server._port = find_available_port()
        server._chat_template_kwargs = {}
        server._http_server = SimpleNamespace(started=False)
        server._http_server_task = None
        server._serve_error = None
        server._serve_ended = False
        server._fatal_error = None

        async def scenario():
            serving = asyncio.create_task(
                server.serve("http://127.0.0.1:1", None, [None])
            )
            try:
                with pytest.raises(RuntimeError) as excinfo:
                    await asyncio.wait_for(server.wait_until_serving(), timeout=30)
                return excinfo.value
            finally:
                await asyncio.gather(serving, return_exceptions=True)

        error = asyncio.run(scenario())
        assert isinstance(error.__cause__, ValueError)
        assert server._serve_ended
        assert server._serve_error is error.__cause__


class TestOneRequestAtATime:
    """A session serves one request at a time, on either endpoint."""

    @staticmethod
    def _generate_in_background(env, sid, input_ids):
        """Start a generate that the mock holds open, and return its thread.

        Returns once the mock has logged the request, which means the
        session server holds the session and is forwarding: a second
        request now sees the session busy.
        """
        env.mock.latency_s = 1.0
        responses = []

        def _send():
            responses.append(generate(env, sid, input_ids))

        before = len(env.mock.request_log)
        thread = threading.Thread(target=_send)
        thread.start()
        deadline = time.monotonic() + 10
        while len(env.mock.request_log) == before:
            assert time.monotonic() < deadline, "the generate never reached the engine"
            time.sleep(0.01)
        return thread, responses

    @staticmethod
    def _finish(env, thread):
        env.mock.latency_s = 0.0
        thread.join(timeout=30)

    def test_a_second_generate_conflicts(self, env):
        """A second request on a busy session gets a 409 instead of queueing
        behind the first, which the engine would interleave."""
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)
        thread, responses = self._generate_in_background(env, sid, [1, 2])
        try:
            second = generate(env, sid, [3, 4])
            assert second.status_code == 409
        finally:
            self._finish(env, thread)
        assert responses[0].status_code == 200

    def test_a_close_while_a_generate_runs_conflicts(self, env):
        """A close mid-generation would classify a turn that is still being
        written, so it gets a 409; afterwards it closes and the group
        finishes normally."""
        request = next_request(env)
        group_id = request.group_ids[0]
        sid0 = create_session(env, group_id, 0)
        sid1 = create_session(env, group_id, 1)
        thread, responses = self._generate_in_background(env, sid0, [1, 2])
        try:
            during = close(env, sid0)
            assert during.status_code == 409
        finally:
            self._finish(env, thread)
        assert responses[0].status_code == 200

        assert close(env, sid0).status_code == 200
        assert generate(env, sid1, [3, 4]).status_code == 200
        assert close(env, sid1).status_code == 200
        assert reward(env, sid0, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is True
        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        assert delivery.group_id == group_id

    def test_a_generate_after_close_adds_no_turn(self, env):
        """A closed trajectory takes no more turns: the request gets a 410 and
        the training data keeps only the turns the close classified."""
        request = next_request(env)
        group_id = request.group_ids[0]
        sid0 = create_session(env, group_id, 0)
        sid1 = create_session(env, group_id, 1)
        assert generate(env, sid0, [1, 2]).status_code == 200
        assert close(env, sid0).status_code == 200

        late = generate(env, sid0, [3, 4])
        assert late.status_code == 410

        assert generate(env, sid1, [5, 6]).status_code == 200
        assert close(env, sid1).status_code == 200
        assert reward(env, sid0, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is True
        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        result = assemble_group_payload(payload, return_logprobs=True)
        # One turn per trajectory: the rejected request added none.
        assert result.num_sequence == 2

    def test_a_chat_retry_on_a_closed_trajectory_changes_nothing(self, chat_env):
        """A closed trajectory's training data is frozen for the chat entry
        too: a retry that the chat entry would roll back is refused before it
        touches the session, so the group still assembles.

        The retry repeats the request the server answered last, which the
        chat entry reads as a rollback; the trajectory is closed, so its
        turns are already classified and its declared fields already
        validated.
        """
        env = chat_env
        chat = TestChatCompletions._chat
        messages = TestChatCompletions._messages()
        env.mock.scripted_texts = ["First answer.", "Second answer."]
        request = next_request(env)
        group_id = request.group_ids[0]
        sid0 = create_session(env, group_id, 0)
        body = {"messages": messages}

        assert chat(env, sid0, body).status_code == 200
        assert (
            close(env, sid0, extra_fields_train={"idx_to_sub_traj": [0]}).status_code
            == 200
        )

        # The same request again: the chat entry would roll the closed
        # trajectory's last turn back.
        retry = chat(env, sid0, body)
        assert retry.status_code == 410

        # The group finishes, and the actor assembles both trajectories'
        # turns; the refused retry neither added nor dropped one.
        sid1 = create_session(env, group_id, 1)
        assert chat(env, sid1, body).status_code == 200
        assert (
            close(env, sid1, extra_fields_train={"idx_to_sub_traj": [0]}).status_code
            == 200
        )
        assert reward(env, sid0, 1.0).json()["group_complete"] is False
        assert reward(env, sid1, 0.5).json()["group_complete"] is True
        env.buffer.release_batch(version=0).wait()
        delivery = poll_channel(env.rollout)
        assert isinstance(delivery, GroupDelivery)
        channel = payload_channel_for(
            delivery.payload_ref.locator, env.payload_channels
        )
        payload = poll_channel(channel, key=delivery.payload_ref.key)
        result = assemble_group_payload(payload, return_logprobs=True)
        assert result.num_sequence == 2
        assert result.extra_fields_train["idx_to_sub_traj"] == [0, 0]


class TestNoTrainedTurn:
    """A trajectory without a trained turn cannot be closed."""

    def test_a_close_without_a_turn_is_rejected(self, env):
        """Closing a trajectory the agent never sampled would deliver a group
        with nothing to train on, so it gets a 400."""
        request = next_request(env)
        sid = create_session(env, request.group_ids[0], 0)
        resp = close(env, sid)
        assert resp.status_code == 400
        assert "has no trained turn" in resp.json()["error"]

    def test_a_first_turn_that_cannot_be_sampled_aborts_the_group(
        self, request, cluster
    ):
        """An agent that gives up on its first turn aborts the group, so the
        buffer records a discard and no half-built trajectory is delivered."""
        env = request.getfixturevalue("env")
        rollout_request = next_request(env)
        group_id = rollout_request.group_ids[0]
        prompt_ids = rollout_request.input_ids[0]

        worker = _make_worker(_DiscardingLoopWorker, False, env.base_url)

        asyncio.run(
            worker.run_agentloop_rollout_group(
                prompt_ids,
                rollout_request.answers[0],
                GROUP_SIZE,
                env.rollout,
                group_id,
                TASK_NAME,
            )
        )

        wait_for_drops(env, "context_length_exceeded")
        # Nothing of the group reaches the actor.
        assert poll_channel(env.rollout, timeout_s=2) is None


class _DiscardingLoopWorker(TwoTurnLogprobLoopWorker):
    """A loop whose first turn cannot be sampled: it gives the group up."""

    async def run_session_trajectory(self, binding, input_ids, answer):
        raise GroupDiscarded("context_length_exceeded")


class TestPayloadShards:
    """The payload shards are handed to groups in creation order."""

    def test_groups_take_the_shards_in_turn(self, env):
        """Four groups in a row use both shards twice, whatever the number of
        session servers on the node: a session server only receives the
        groups whose id picks it, so an id-based shard choice leaves shards
        unused."""
        group_ids = []
        for _ in range(4):
            request = next_request(env)
            group_id, _turns, _sids = run_group(env, request, rewards=(1.0, 0.5))
            group_ids.append(group_id)
            env.buffer.release_batch(version=0).wait()

        shards = []
        for group_id in group_ids:
            delivery = poll_channel(env.rollout)
            assert isinstance(delivery, GroupDelivery)
            assert delivery.group_id == group_id
            channel = payload_channel_for(
                delivery.payload_ref.locator, env.payload_channels
            )
            shards.append(env.payload_channels.index(channel))

        # Both shards are used, two groups each.
        assert len(set(shards)) == len(env.payload_channels)
        for shard in set(shards):
            assert shards.count(shard) == 2


class _FailingPayloadCollector(PayloadCollector):
    """A PayloadCollector that raises on either part or finalize writes.

    Runs inside the real ChannelWorker after ``self.recv`` completes, so the
    collective stream stays healthy and the error reaches PayloadWriter as an
    isolated write failure rather than a dead channel peer.
    """

    def __init__(self, *, fail_on_last: bool) -> None:
        super().__init__()
        self._fail_on_last = fail_on_last

    def collect(self, item, key):
        if isinstance(item, PayloadPart) and item.last is self._fail_on_last:
            raise RuntimeError("simulated payload write failure")
        yield from PayloadCollector.collect(self, item, key)


class TestFinalizePutFailure:
    """The finalize part's put fails after every earlier part has settled."""

    def test_finalize_put_failure_discards_not_delivers(self, cluster):
        fail_env = _build_env(
            cluster,
            verify_assembly=False,
            suffix="FinalizeFail",
            payload_collector=_FailingPayloadCollector(fail_on_last=True),
        )
        try:
            request = next_request(fail_env)
            group_id = request.group_ids[0]
            sid0 = create_session(fail_env, group_id, 0)
            sid1 = create_session(fail_env, group_id, 1)
            assert generate(fail_env, sid0, [1, 2]).status_code == 200
            assert generate(fail_env, sid1, [3, 4]).status_code == 200

            # Wait until the background settlements are done; the collector
            # accepts non-last parts and raises only on the finalize part.
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                metrics = fail_env.server.get_metrics().wait()[0]
                if metrics["session_server/pending_payload_bytes"] == 0:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("payload settlements did not drain")

            assert close(fail_env, sid0).status_code == 200
            assert close(fail_env, sid1).status_code == 200
            assert reward(fail_env, sid0, 1.0).json()["group_complete"] is False
            assert reward(fail_env, sid1, 0.5).json()["group_complete"] is False

            # The group is discarded, not delivered: the GroupDelivery would
            # promise a payload the collector can never emit.
            metrics = wait_for_drops(fail_env, "payload_write_failed")
            assert metrics["dropped_groups"] == 1
            assert poll_channel(fail_env.rollout, timeout_s=2) is None

            server_metrics = fail_env.server.get_metrics().wait()[0]
            # get_metrics drains per window: the failure above landed after
            # the last read (the pending-drain poll), so this window's count
            # is the one failure of this scenario alone.
            assert server_metrics["session_server/payload_write_failed"] == 1
        finally:
            _stop_env(fail_env)


class TestPutFailure:
    """A background part put fails while the channel worker stays alive."""

    def test_failed_background_put_voids_group(self, cluster):
        fail_env = _build_env(
            cluster,
            verify_assembly=False,
            suffix="PutFail",
            payload_collector=_FailingPayloadCollector(fail_on_last=False),
        )
        try:
            request = next_request(fail_env)
            group_id = request.group_ids[0]
            sid = create_session(fail_env, group_id, 0)
            assert generate(fail_env, sid, [1, 2]).status_code == 200

            # The failed put voids the group server-side; the buffer consumes
            # the discard record and the server keeps running.
            metrics = wait_for_drops(fail_env, "payload_write_failed")
            assert metrics["dropped_groups"] == 1

            server_metrics = fail_env.server.get_metrics().wait()[0]
            assert server_metrics["session_server/payload_write_failed"] == 1
        finally:
            _stop_env(fail_env)


class TestVoidCleanup:
    """A voided group keeps its record, not the state it had buffered.

    No env stack: the state is the point, and the voiding path is driven
    directly on the worker.
    """

    def test_voiding_clears_the_buffered_state(self):
        """Voiding drops each session's turns, prefix, chat conversation and
        declared verdict state, and the group's verification reference, so a
        discarded group holds no token buffers until its record expires."""
        worker = SessionServerWorker.__new__(SessionServerWorker)
        recorded = []

        async def record_discard(group_id, reason):
            recorded.append((group_id, reason))

        class _Writer:
            async def put(self, *args, **kwargs):
                return True

        worker._groups = {}
        worker._metrics = {}
        worker._report_discard = record_discard
        worker._payload_writer = _Writer()

        group = _GroupState(group_id=4501, task_name=TASK_NAME, group_size=1)
        session = _SessionState(session_id="sid", group_id=4501, traj_index=0)
        session.turns = [new_turn_record([1, 2, 3], None, None, None)]
        session.last_input = [1, 2]
        session.last_output = [3]
        session.extra_fields_train = {"idx_to_sub_traj": [0]}
        session.not_training_turns = {1}
        session.chat = ChatConversation(
            messages=[{"role": "user", "content": "u0"}],
            tools=None,
            checkpoint=None,
        )
        group.sessions[0] = session
        group.verify_result = {"kept_lens": [3]}
        worker._groups[group.group_id] = group

        asyncio.run(worker._void_group(group.group_id, "context_length_exceeded"))

        assert recorded == [(group.group_id, "context_length_exceeded")]
        assert group.discard_reason == "context_length_exceeded"
        assert session.turns == []
        assert session.last_input is None
        assert session.last_output is None
        assert session.chat is None
        assert session.extra_fields_train == {}
        assert session.not_training_turns == set()
        assert group.verify_result is None


class _FakePutWork:
    """A payload put that settles the way the test asks it to."""

    def __init__(self, error=None, delay_s=0.0):
        self._error = error
        self._delay_s = delay_s

    async def async_wait(self):
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        if self._error is not None:
            raise self._error


class _FakePayloadChannel:
    """A payload channel whose puts settle as the test asks."""

    def __init__(self, work):
        self._work = work

    def put(self, item, **kwargs):
        return self._work


class TestPayloadWriterFailures:
    """How the writer tells the two kinds of put failure apart."""

    @staticmethod
    def _writer(work, **kwargs):
        written, peers = [], []
        writer = PayloadWriter(
            [_FakePayloadChannel(work)],
            max_pending_bytes=1024 * 1024,
            put_timeout_s=kwargs.get("put_timeout_s", 5.0),
            on_write_failure=lambda group_id: _append_async(written, group_id),
            on_peer_failure=peers.append,
        )
        return writer, written, peers

    def test_a_failed_channel_peer_stops_the_caller(self):
        """A put refused because the channel's collective direction failed is
        not one group's failure: the writer reports it to the caller, which
        stops serving, instead of voiding group after group."""
        error = PeerCommunicationError("an earlier send in collective group c failed")
        writer, written, peers = self._writer(_FakePutWork(error=error))

        success = asyncio.run(
            writer.put(0, group_id=7, item="part", await_completion=True)
        )

        assert success is False
        assert peers == [error]
        assert written == []

    def test_a_dead_channel_worker_stops_the_caller(self):
        """A dead payload channel worker cannot accept any later write either,
        so the writer reports it through on_peer_failure instead of voiding
        group after group."""
        error = ActorDiedError()
        writer, written, peers = self._writer(_FakePutWork(error=error))

        success = asyncio.run(
            writer.put(0, group_id=10, item="part", await_completion=True)
        )

        assert success is False
        assert peers == [error]
        assert written == []

    def test_a_timed_out_put_voids_its_group(self):
        """A put that timed out is one group's failure: the group is voided
        and serving continues."""
        writer, written, peers = self._writer(
            _FakePutWork(delay_s=0.2), put_timeout_s=0.05
        )

        success = asyncio.run(
            writer.put(0, group_id=8, item="part", await_completion=True)
        )

        assert success is False
        assert written == [8]
        assert peers == []

    def test_another_failure_voids_its_group(self):
        """Any other put failure voids the group, as it did before."""
        writer, written, peers = self._writer(
            _FakePutWork(error=ValueError("channel down"))
        )

        success = asyncio.run(
            writer.put(0, group_id=9, item="part", await_completion=True)
        )

        assert success is False
        assert written == [9]
        assert peers == []


async def _append_async(target, value):
    """Record a group id the way the writer's callback does."""
    target.append(value)
