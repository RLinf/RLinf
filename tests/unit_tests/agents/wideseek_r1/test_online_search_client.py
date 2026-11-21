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

"""Unit tests for AsyncOnlineSearchClient search providers (Serper / You.com)."""

import asyncio
from unittest import mock

import pytest
from aioresponses import aioresponses
from omegaconf import OmegaConf

from rlinf.agents.wideseek_r1.tools import AsyncOnlineSearchClient

SERPER_RESPONSE = {
    "organic": [
        {
            "title": "Serper Title",
            "snippet": "Serper snippet text.",
            "link": "https://example.com/serper",
        },
        {
            "title": "Serper Title 2",
            "snippet": "Second snippet.",
            "link": "https://example.com/serper2",
        },
    ]
}

YOUCOM_RESPONSE = {
    "hits": [
        {
            "title": "You.com Title",
            "description": "You.com snippet text.",
            "url": "https://example.com/you",
            "snippets": ["You.com snippet text."],
        },
        {
            "title": "You.com Title 2",
            "description": "Second snippet.",
            "url": "https://example.com/you2",
        },
    ]
}


def make_cfg(search_provider: str):
    return OmegaConf.create(
        {
            "tools": {
                "online": True,
                "search_provider": search_provider,
                "search": {"topk": 5},
                "use_jina": False,
                "enable_cache": False,
                "max_retries": 2,
                "retry_delay_base": 0,
            }
        }
    )


@pytest.fixture(autouse=True)
def reset_shared_session():
    AsyncOnlineSearchClient._shared_session = None
    yield
    if AsyncOnlineSearchClient._shared_session is not None:
        asyncio.get_event_loop_policy()
        loop = asyncio.new_event_loop()
        loop.run_until_complete(AsyncOnlineSearchClient._shared_session.close())
        loop.close()
        AsyncOnlineSearchClient._shared_session = None


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make retry backoff instantaneous in tests."""

    async def _instant_sleep(_delay):
        pass

    monkeypatch.setattr(
        "rlinf.agents.wideseek_r1.tools.asyncio.sleep", _instant_sleep
    )


class TestProviderInit:
    def test_serper_provider_requires_key(self, monkeypatch):
        monkeypatch.delenv("SERPER_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="SERPER_API_KEY"):
            AsyncOnlineSearchClient(cfg=make_cfg("serper"))

    def test_you_provider_requires_key(self, monkeypatch):
        monkeypatch.delenv("YDC_API_KEY", raising=False)
        with pytest.raises(RuntimeError, match="YDC_API_KEY"):
            AsyncOnlineSearchClient(cfg=make_cfg("you"))

    def test_unknown_provider_rejected(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sk-test-1234")
        with pytest.raises(ValueError, match="Unsupported search provider"):
            AsyncOnlineSearchClient(cfg=make_cfg("bing"))

    def test_default_provider_is_serper(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sk-test-1234")
        cfg = make_cfg("serper")
        cfg.tools.pop("search_provider")
        client = AsyncOnlineSearchClient(cfg=cfg)
        assert client.search_provider == "serper"

    def test_you_provider_init(self, monkeypatch):
        monkeypatch.setenv("YDC_API_KEY", "ydc-test-5678")
        client = AsyncOnlineSearchClient(cfg=make_cfg("you"))
        assert client.search_provider == "you"
        assert client.you_headers["X-API-Key"] == "ydc-test-5678"


class TestQueryAsync:
    def test_serper_results_structure(self, monkeypatch):
        monkeypatch.setenv("SERPER_API_KEY", "sk-test-1234")
        client = AsyncOnlineSearchClient(cfg=make_cfg("serper"))
        with aioresponses() as m:
            m.post(
                "https://google.serper.dev/search",
                payload=SERPER_RESPONSE,
            )
            results = run(client.query_async({"queries": ["q1"], "topk": 5}))

        assert len(results) == 1
        assert set(results[0].keys()) == {"documents", "urls", "server_type"}
        assert results[0]["server_type"] == "async-online-search"
        assert results[0]["documents"] == [
            "Serper Title Serper snippet text.",
            "Serper Title 2 Second snippet.",
        ]
        assert results[0]["urls"] == [
            "https://example.com/serper",
            "https://example.com/serper2",
        ]

    def test_youcom_results_match_serper_structure(self, monkeypatch):
        """You.com results must be normalized to the same structure as Serper's."""
        monkeypatch.setenv("YDC_API_KEY", "ydc-test-5678")
        client = AsyncOnlineSearchClient(cfg=make_cfg("you"))
        with aioresponses() as m:
            m.post(
                "https://api.ydc-index.io/search",
                payload=YOUCOM_RESPONSE,
            )
            results = run(client.query_async({"queries": ["q1"], "topk": 5}))

        assert len(results) == 1
        # Same normalized structure as the Serper provider.
        assert set(results[0].keys()) == {"documents", "urls", "server_type"}
        assert results[0]["server_type"] == "async-online-search"
        assert results[0]["documents"] == [
            "You.com Title You.com snippet text.",
            "You.com Title 2 Second snippet.",
        ]
        assert results[0]["urls"] == [
            "https://example.com/you",
            "https://example.com/you2",
        ]

    def test_youcom_request_payload(self, monkeypatch):
        """The You.com request must carry query + num_web_results and the API key."""
        monkeypatch.setenv("YDC_API_KEY", "ydc-test-5678")
        client = AsyncOnlineSearchClient(cfg=make_cfg("you"))
        with aioresponses() as m:
            m.post(
                "https://api.ydc-index.io/search",
                payload={"hits": []},
            )
            run(client.query_async({"queries": ["what is rl?"], "topk": 3}))

            (_, request), *_ = m.requests.items()
            request = request[0]
            assert request.kwargs["json"] == {
                "query": "what is rl?",
                "num_web_results": 3,
            }
            assert request.kwargs["headers"]["X-API-Key"] == "ydc-test-5678"

    def test_youcom_retry_then_success(self, monkeypatch):
        """Transient failures must be retried before returning a result."""
        monkeypatch.setenv("YDC_API_KEY", "ydc-test-5678")
        cfg = make_cfg("you")
        cfg.tools["max_retries"] = 3
        client = AsyncOnlineSearchClient(cfg=cfg)
        with aioresponses() as m:
            m.post("https://api.ydc-index.io/search", status=500)
            m.post("https://api.ydc-index.io/search", status=503)
            m.post("https://api.ydc-index.io/search", payload=YOUCOM_RESPONSE)
            results = run(client.query_async({"queries": ["q1"], "topk": 5}))

        assert results[0]["documents"][0].startswith("You.com Title")

    def test_youcom_all_retries_exhausted_returns_empty(self, monkeypatch):
        """After exhausting retries, results must degrade gracefully to empty."""
        monkeypatch.setenv("YDC_API_KEY", "ydc-test-5678")
        client = AsyncOnlineSearchClient(cfg=make_cfg("you"))
        with aioresponses() as m:
            for _ in range(2):
                m.post("https://api.ydc-index.io/search", status=500)
            results = run(client.query_async({"queries": ["q1"], "topk": 5}))

        assert results[0] == {
            "documents": [],
            "urls": [],
            "server_type": "async-online-search",
        }

    def test_serper_unchanged_by_youcom_addition(self, monkeypatch):
        """Existing Serper path still works when provider is left at default."""
        monkeypatch.setenv("SERPER_API_KEY", "sk-test-1234")
        cfg = make_cfg("serper")
        cfg.tools.pop("search_provider")
        client = AsyncOnlineSearchClient(cfg=cfg)
        with aioresponses() as m:
            m.post("https://google.serper.dev/search", payload=SERPER_RESPONSE)
            results = run(client.query_async({"queries": ["q1"], "topk": 5}))

        assert results[0]["urls"][0] == "https://example.com/serper"
