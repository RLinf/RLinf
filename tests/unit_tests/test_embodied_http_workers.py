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

import asyncio
import importlib.util
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.scheduler import Worker
from rlinf.utils import http_client as http_client_module
from rlinf.workers.rollout.embodied_http_worker import EmbodiedHTTPRolloutWorker


class FakeHTTPWorker(EmbodiedHTTPRolloutWorker):
    http_config_key = "http"
    server_url_attr = "fake_server_url"
    server_urls_attr = "_fake_server_urls"
    action_path = "/actions"
    missing_urls_error = "fake URLs are required"

    def _build_http_request(self, env_obs, mode="eval"):
        return {"observation": env_obs, "mode": mode}, {"mode": mode}

    def _parse_http_response(self, response, state):
        return torch.as_tensor(response["actions"], dtype=torch.float32), state


@pytest.fixture
def fake_scheduler(monkeypatch):
    def init_scheduler(worker):
        worker._rank = 0
        worker._world_size = 1
        worker._group_name = "RolloutGroup"
        worker._logger = logging.getLogger("worker-test")
        worker._stacklevel = 1

    monkeypatch.setattr(Worker, "__init__", init_scheduler)


def cfg(only_eval=True, chunks=2, http=None):
    return OmegaConf.create(
        {
            "runner": {"only_eval": only_eval, "enable_decoupled_mode": False},
            "env": {
                "group_name": "EnvGroup",
                "eval": {
                    "total_num_envs": 4,
                    "rollout_epoch": 1,
                    "max_steps_per_rollout_epoch": 4,
                },
            },
            "rollout": {
                "pipeline_stage_num": 2,
                "model": {
                    "model_type": "fake",
                    "num_action_chunks": chunks,
                    "action_dim": 1,
                },
                "http": http or {},
            },
        }
    )


def test_base_override_and_validation(fake_scheduler):
    worker = FakeHTTPWorker(
        cfg(chunks=1),
        config_rollout=OmegaConf.create(
            {"model": {"num_action_chunks": 2, "action_dim": 3}}
        ),
    )
    assert worker.model_cfg.num_action_chunks == 2
    assert worker.eval_batch_size == 2
    assert worker.n_eval_chunk_steps == 2
    with pytest.raises(ValueError, match="only_eval=True"):
        FakeHTTPWorker(cfg(only_eval=False))


def test_base_url_config_and_transport(monkeypatch, fake_scheduler):
    worker = FakeHTTPWorker(
        cfg(
            http={
                "http_timeout_s": 7.5,
                "http_max_retries": 3,
                "http_retry_backoff_s": 0.25,
            }
        )
    )
    worker._rank = 3
    worker._set_http_server_urls(["http://one", "http://two"])
    seen = []

    class Client:
        def __init__(self, url):
            seen.append(url)

    monkeypatch.setattr(http_client_module, "InferenceHTTPClient", Client)
    asyncio.run(worker.init_worker())
    assert worker.fake_server_url == "http://two"
    assert worker._fake_server_urls == ["http://one", "http://two"]
    assert seen == ["http://two"]
    assert (
        worker._http_timeout_s,
        worker._http_max_retries,
        worker._http_retry_backoff_s,
    ) == (7.5, 3, 0.25)
    with pytest.raises(RuntimeError, match="fake URLs are required"):
        asyncio.run(FakeHTTPWorker(cfg()).init_worker())


def test_base_post_predict_and_merge(fake_scheduler):
    worker = FakeHTTPWorker(cfg())
    calls = []

    class Client:
        def post(self, path, payload, **kwargs):
            calls.append((path, payload, kwargs))
            return {"actions": [[[1.0]]]}

    worker.http_client = Client()
    actions, state = worker.predict({"ids": torch.tensor([0])})
    assert actions.shape == (1, 1, 1) and state == {"mode": "eval"}
    assert calls[0][0] == "/actions" and calls[0][2]["msgpack"] is True
    merged = FakeHTTPWorker._merge_obs_batches(
        [
            {"obs": {"x": torch.tensor([[1]]), "t": ["a"]}},
            {
                "obs": {"x": torch.tensor([[2]]), "t": ["b"]},
                "final_obs": {},
                "eval_epoch_done": True,
            },
        ]
    )
    assert torch.equal(merged["obs"]["x"], torch.tensor([[1], [2]]))
    assert (
        merged["obs"]["t"] == ["a", "b"]
        and merged["reset"]
        and merged["eval_epoch_done"]
    )


def test_shared_worker_evaluate_preserves_channels_and_stops(
    monkeypatch, fake_scheduler
):
    worker = FakeHTTPWorker(cfg())
    worker.n_eval_chunk_steps = 2
    received, sent, predicted = [], [], []
    actions = torch.tensor([[[1.0]]])

    class ReceiveResult:
        def __init__(self, payload):
            self.payload = payload

        async def async_wait(self):
            return self.payload

    def recv_from(**kwargs):
        received.append(kwargs)
        payload = {
            "obs": {"ids": torch.tensor([0])},
            "eval_epoch_done": len(received) == 2,
        }
        return ReceiveResult(kwargs["merge_fn"]([payload]))

    def predict(obs, mode):
        predicted.append((obs, mode))
        return actions, {}

    monkeypatch.setattr(worker, "recv_from", recv_from)
    monkeypatch.setattr(worker, "predict", predict)
    monkeypatch.setattr(worker, "send_to", lambda **kwargs: sent.append(kwargs))
    asyncio.run(worker.evaluate("input", "output"))
    assert len(received) == 2 and len(predicted) == len(sent) == 1
    assert received[0]["group_name"] == "EnvGroup"
    assert received[0]["channel"] == "input" and received[0]["route_key"] == 0
    assert received[0]["batch_size"] == 2
    assert predicted[0][0]["_rlinf_stage_id"] == 0
    assert sent[0]["channel"] == "output" and sent[0]["route_key"] == 0
    assert sent[0]["batch_size"] == 2
    assert torch.equal(sent[0]["data"], actions)


def test_sglang_worker_grouped_transport_without_sglang(fake_scheduler):
    path = (
        Path(__file__).resolve().parents[2]
        / "rlinf/workers/rollout/sglang/sglang_embodied_worker.py"
    )
    spec = importlib.util.spec_from_file_location("sglang_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    worker = module.SGLangEmbodiedWorker(cfg(), SimpleNamespace())

    class Adapter:
        action_path = "/actions"

        def request_groups(self, obs):
            return [[2], [0, 1]]

        def build_request(self, obs, mode="eval"):
            return {"ids": obs["ids"]}, {"ids": obs["ids"]}

        def parse_response(self, response, state):
            return state["ids"].float().view(-1, 1, 1), {"ids": state["ids"]}

    worker.sglang_adapter = Adapter()
    calls = []

    def post(path, payload):
        calls.append(payload["ids"].tolist())
        return {}

    worker._post_msgpack = post
    actions, info = worker.predict({"ids": torch.tensor([0, 1, 2])})
    assert calls == [[2], [0, 1]]
    assert actions[:, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert info["ids"].tolist() == [0, 1, 2]
