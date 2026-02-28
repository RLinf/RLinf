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


import asyncio
import json
import logging
import os
import random
import threading
from typing import Any

import aiohttp
from omegaconf import DictConfig

from rlinf.agents.wideseek_r1.utils.webpage import WebPageCache
from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker

SERPER_STATS = dict(num_requests=0)


class AsyncOnlineSearchClient:
    """Online search client using Serper API and Jina API for web access."""

    # Class-level shared session for connection pooling
    _shared_session = None
    _session_lock = threading.Lock()
    _search_semaphore = None
    _access_semaphore = None

    @classmethod
    async def get_session(cls):
        """Get or create shared aiohttp session with connection pooling."""
        if cls._shared_session is None or cls._shared_session.closed:
            with cls._session_lock:
                if cls._shared_session is None or cls._shared_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=1000,  # Max total connections
                        limit_per_host=500,  # Max connections per host
                        ttl_dns_cache=600,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=300, sock_connect=100),
                        trust_env=True,
                    )
        return cls._shared_session

    @classmethod
    def _get_search_semaphore(cls):
        """Return a shared semaphore limiting concurrent search requests."""
        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore

    @classmethod
    def _get_access_semaphore(cls):
        """Return a shared semaphore limiting concurrent access requests."""
        if cls._access_semaphore is None:
            cls._access_semaphore = asyncio.Semaphore(10)
        return cls._access_semaphore

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        # Retry configuration
        self.max_retries = self.cfg.tools.get("max_retries", 15)
        self.retry_delay_base = self.cfg.tools.get("retry_delay_base", 5)

        # Serper API
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = os.environ.get("SERPER_API_KEY", "")
        if not self.serper_api_key:
            raise RuntimeError(
                "Serper API key is not set. Please set the SERPER_API_KEY environment variable."
            )
        self.serper_headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json",
        }
        self.logger.info(
            f"Initialized Serper API client with key: {self.serper_api_key[:8]}..."
        )

        # Jina API
        self.use_jina = self.cfg.tools.get("use_jina", False)
        self.jina_api_key = os.environ.get("JINA_API_KEY", "")
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError(
                "Jina is enabled but the API key is not set. Please set the JINA_API_KEY environment variable."
            )
        if self.use_jina:
            self.logger.info(
                f"Initialized Jina API client with key: {self.jina_api_key[:8]}..."
            )

        # Web page cache
        cache_enabled = self.cfg.tools.get("enable_cache", True)
        cache_size = self.cfg.tools.get("cache_size", 10000)
        cache_file = self.cfg.tools.get("cache_file", "./webpage_cache.json")

        if cache_enabled:
            self.webpage_cache = WebPageCache(cache_size, cache_file, save_interval=5)
            self.logger.info(
                f"Web page cache enabled: size={cache_size}, file={cache_file}"
            )
        else:
            self.webpage_cache = None
            self.logger.info("Web page cache disabled")

    async def _do_serper_search(self, session, query: str, topk: int) -> dict:
        """
        Execute a single Serper API search request (low-level network call).

        Args:
            session: aiohttp session
            query: Search query string (already truncated)
            topk: Number of results to return

        Returns:
            Dict with 'success' bool and either 'data' or 'error'

        Raises:
            Exception: If the request fails (to trigger retry)
        """
        async with self._get_search_semaphore():
            payload = {"q": query, "num": topk}
            await asyncio.sleep(0.1)  # Rate limiting
            SERPER_STATS["num_requests"] += 1

            async with session.post(
                f"{self.serper_server_addr}/search",
                headers=self.serper_headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"success": True, "data": data}
                else:
                    response_text = await response.text()
                    raise Exception(f"HTTP {response.status}: {response_text[:100]}")

    async def _single_serper_query(self, session, query: str, topk: int) -> dict:
        """
        Execute a single Serper API search query with retry logic.

        Args:
            session: aiohttp session
            query: Search query string
            topk: Number of results to return

        Returns:
            Dict with 'success' bool and either 'data' or 'error'
        """
        query = query[:2000]  # Truncate long queries

        # Retry with backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay_base * (
                        2 ** (attempt - 1)
                    ) + random.uniform(0, 20)
                    delay = min(delay, 300) + random.uniform(0, 20)
                    error_type = type(last_error).__name__ if last_error else "Unknown"
                    error_msg = str(last_error)[:100] if last_error else ""
                    if attempt > 5:
                        self.logger.warning(
                            f"Retrying search query '{query[:50]}...' "
                            f"(attempt {attempt + 1}/{self.max_retries}, delay {delay}s) "
                            f"- Last error: {error_type}: {error_msg}"
                        )
                    await asyncio.sleep(delay)

                return await self._do_serper_search(session, query, topk)

            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    error_msg = f"{type(e).__name__}: {str(e)[:200]}"
                    return {"success": False, "error": error_msg}

        return {"success": False, "error": "Unknown error after all retries"}

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict]:
        """
        Query using Serper API with retry logic.

        Args:
            req_meta: Dict containing 'queries' list and 'topk' int

        Returns:
            List of dicts with 'documents', 'urls', and 'server_type'
        """
        queries = req_meta.get("queries", [])
        topk = req_meta.get("topk", 5)

        if not queries:
            return []

        session = await self.get_session()
        tasks = [self._single_serper_query(session, query, topk) for query in queries]
        serper_results = await asyncio.gather(*tasks)

        # Format results
        formatted_results = []
        for query, serper_result in zip(queries, serper_results):
            if serper_result and serper_result.get("success", False):
                data = serper_result.get("data", {})
                organic_results = data.get("organic", [])[:topk]

                documents = [
                    result.get("title", "") + " " + result.get("snippet", "")
                    for result in organic_results
                ]
                urls = [result.get("link", "") for result in organic_results]

                formatted_results.append(
                    {
                        "documents": documents,
                        "urls": urls,
                        "server_type": "async-online-search",
                    }
                )
            else:
                formatted_results.append(
                    {"documents": [], "urls": [], "server_type": "async-online-search"}
                )

        return formatted_results

    async def _do_jina_access(self, session, url: str) -> dict:
        """
        Execute a single Jina API access request (low-level network call).

        Args:
            session: aiohttp session
            url: URL to access

        Returns:
            Dict with 'page' and 'type'

        Raises:
            Exception: If the request fails (to trigger retry)
        """
        headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "Content-Type": "application/json",
        }
        async with session.get(
            f"https://r.jina.ai/{url}",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=200),
        ) as response:
            if response.status == 200:
                content = await response.text()
                return dict(page=content, type="jina")
            elif response.status != 429:
                return dict(
                    page="The current URL cannot be searched. Please switch to a different URL and try again.",
                    type="jina",
                )
            # elif response.status == 422:
            #     content = await response.text()
            #     return dict(page=content, type="jina")
            else:
                raise Exception(f"HTTP {response.status}")

    async def _single_jina_access(self, session, url: str) -> dict:
        """
        Access a single URL via Jina API with retry logic.

        Args:
            session: aiohttp session
            url: URL to access

        Returns:
            Dict with 'page', 'type'
        """
        # Retry with backoff
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if attempt > 0:
                    delay = self.retry_delay_base * (
                        2 ** (attempt - 1)
                    ) + random.uniform(0, 20)
                    delay = min(delay, 300) + random.uniform(0, 20)
                    error_type = type(last_error).__name__ if last_error else "Unknown"
                    error_msg = str(last_error)[:100] if last_error else ""
                    if attempt > 5:
                        self.logger.warning(
                            f"Retrying URL access '{url}' "
                            f"(attempt {attempt + 1}/{self.max_retries}, delay {delay}s) "
                            f"- Last error: {error_type}: {error_msg}"
                        )
                    await asyncio.sleep(delay)

                return await self._do_jina_access(session, url)

            except Exception as e:
                last_error = e
                if attempt == self.max_retries - 1:
                    return dict(
                        page="The current URL cannot be searched. Please switch to a different URL and try again.",
                        type="access",
                    )

        return dict(
            page="The current URL cannot be searched. Please switch to a different URL and try again.",
            type="access",
        )

    async def access_async(self, urls: list[str]) -> list[dict]:
        """
        Access URLs using Jina API with caching and retry logic.

        Args:
            urls: List of URLs to access

        Returns:
            List of dicts with 'page', 'type', and 'server_type'
        """
        if not urls:
            return []

        results = []
        urls_to_fetch = []

        # Check cache first
        for url in urls:
            if self.webpage_cache and self.webpage_cache.has(url):
                cached_content = self.webpage_cache.get(url)
                if cached_content:
                    results.append(dict(page=cached_content, type="access"))
                else:
                    urls_to_fetch.append(url)
                    results.append(None)
            else:
                urls_to_fetch.append(url)
                results.append(None)

        # Fetch uncached URLs
        if urls_to_fetch and self.use_jina and self.jina_api_key:
            session = await self.get_session()
            async with self._get_access_semaphore():
                tasks = [
                    self._single_jina_access(session, url) for url in urls_to_fetch
                ]
                fetched_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge fetched results back
            fetch_index = 0
            for i, result in enumerate(results):
                if result is None:
                    fetched_result = (
                        fetched_results[fetch_index]
                        if fetch_index < len(fetched_results)
                        else dict(
                            page="The current URL cannot be searched. Please switch to a different URL and try again.",
                            type="access",
                        )
                    )

                    # Handle exceptions
                    if isinstance(fetched_result, Exception):
                        fetched_result = dict(
                            page="The current URL cannot be searched. Please switch to a different URL and try again.",
                            type="access",
                        )

                    results[i] = fetched_result

                    # Cache successful fetches
                    if self.webpage_cache and fetched_result.get("page"):
                        self.webpage_cache.put(urls[i], fetched_result["page"])

                    fetch_index += 1

        # Fill in any remaining None values
        for i, result in enumerate(results):
            if result is None:
                results[i] = dict(
                    page="The current URL cannot be searched. Please switch to a different URL and try again.",
                    type="access",
                )

        # Add server_type to all results
        for result in results:
            result["server_type"] = "async-online-search"

        return results

    def get_cache_stats(self) -> dict[str, Any]:
        """Return cache statistics for observability."""
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}

    def clear_cache(self):
        """Clear in-memory and persisted cache entries when cache is enabled."""
        if self.webpage_cache:
            self.webpage_cache.clear()

    def force_save_cache(self):
        """Force a cache snapshot to disk when cache is enabled."""
        if self.webpage_cache:
            self.webpage_cache.force_save()


class AsyncSearchClient:
    """Local/offline search client that connects to a local RAG server."""

    # Class-level shared session for connection pooling
    _shared_session = None
    _session_lock = threading.Lock()

    @classmethod
    async def get_session(cls):
        """Get or create shared aiohttp session with connection pooling."""
        if cls._shared_session is None or cls._shared_session.closed:
            with cls._session_lock:
                if cls._shared_session is None or cls._shared_session.closed:
                    # Use Unix domain socket if localhost, otherwise TCP
                    connector = aiohttp.TCPConnector(
                        limit=2000,  # Max total connections
                        limit_per_host=1000,  # Max connections per host
                        ttl_dns_cache=1000,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=300, sock_connect=100),
                        trust_env=False,
                    )
        return cls._shared_session

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.server_addr = self.cfg.tools.search.server_addr
        print(f"[INFO] AsyncSearchClient: Using local server at {self.server_addr}")

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict]:
        """Query local search server."""
        cnt = 0
        last_exception = None
        session = await self.get_session()

        while cnt < 10:
            try:
                async with session.post(
                    f"http://{self.server_addr}/retrieve",
                    json=req_meta,
                ) as response:
                    response.raise_for_status()
                    res = await response.json()
                    return [
                        dict(
                            documents=[r["contents"] for r in result],
                            urls=[r["url"] for r in result],
                            server_type="async-search-browser",
                        )
                        for result in res["result"]
                    ]
            except Exception as e:
                last_exception = e
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post search query to RAG server"
        ) from last_exception

    async def access_async(self, urls: list[str]) -> list[dict]:
        """Access URLs via local server following ASearcher's AsyncSearchBrowserClient logic."""
        cnt = 0
        last_exception = None
        session = await self.get_session()

        while cnt < 10:
            try:
                async with session.post(
                    f"http://{self.server_addr}/access",
                    json={"urls": urls},
                ) as response:
                    response.raise_for_status()
                    res = await response.json()
                    return [
                        dict(
                            page=result["contents"] if result is not None else "",
                            type="access",
                            server_type="async-search-browser",
                        )
                        for result in res["result"]
                    ]
            except Exception as e:
                last_exception = e
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post access request to RAG server"
        ) from last_exception


class WideSeekR1ToolWorker(ToolWorker):
    """Tool worker that serves `search` and `access` requests for WideSeek-R1."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.topk = self.cfg.tools.search.topk
        self.request_processor_task = None

        # Determine whether to use online or local search
        self.use_online_search = self.cfg.tools.get("online", False)

        if self.use_online_search:
            self.log_info(
                "[INFO] WideSeekR1ToolWorker: Using online search (Serper API)"
            )
            self.search_client = AsyncOnlineSearchClient(cfg=self.cfg)
        else:
            self.log_info("[INFO] WideSeekR1ToolWorker: Using local search server")
            self.search_client = AsyncSearchClient(cfg=self.cfg)

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        """Bind input/output channels used for async tool request serving."""
        self.input_channel = input_channel
        self.output_channel = output_channel

    def start_server(self):
        """Start the background coroutine that consumes tool requests."""
        loop = asyncio.get_running_loop()
        self.request_processor_task = loop.create_task(self._process_requests())

    def stop_server(self):
        """Stop request processing by cancelling the background task."""
        # Cancel request processor task
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()

    async def _process_requests(self):
        """Continuously consume tool requests and respond on per-session keys."""

        def process_tool_result(response, tool_type, access_token=None):
            """Process tool results following ASearcher's consume_tool_response logic.

            Args:
                response: The response from search client
                tool_type: Either 'search' or 'access'

            Returns:
                Formatted text for the agent
            """
            if tool_type == "search":
                # Process search results following ASearcher's logic
                documents = response[0]["documents"]
                urls = response[0]["urls"]

                if len(documents) > 0:
                    doc_id_template = "[Doc {doc_id}]({url}):\n"
                    text = "\n\n".join(
                        [
                            doc_id_template.format(doc_id=str(k + 1), url=url)
                            + doc[:5000]
                            for k, (doc, url) in enumerate(zip(documents, urls))
                        ]
                    )
                else:
                    text = "No search results are found."

                return text

            elif tool_type == "access":
                # Process webpage access following ASearcher's logic
                page = response[0].get("page", "")
                assert access_token is not None
                if page is not None and page.strip() != "":
                    # # Limit page content to 250k characters
                    # page = page[:250000]

                    # # Split into chunks of 25k characters, max 10 pages
                    # page_chunks = []
                    # chunk_idx = 0
                    # while len(page) > 0 and chunk_idx < 10:
                    #     chunk_len = min(25000, len(page))
                    #     page_chunks.append(
                    #         f">>>> Page {chunk_idx + 1} >>>>\n\n" + page[:chunk_len]
                    #     )
                    #     page = page[chunk_len:]
                    #     chunk_idx += 1

                    # # Return first page chunk (agent will need to handle pagination)
                    # return page_chunks[0] if page_chunks else "No More Information is Found for this URL."
                    return page[:access_token]
                else:
                    return "No More Information is Found for this URL."

            else:
                raise ValueError(f"Unknown tool type: {tool_type}")

        async def generate_and_send(channel_key: str, tool_name: str, tool_args: dict):
            """Handle both search and access tool requests."""
            try:
                if tool_name == "search":
                    # Handle search query
                    query = tool_args.get("query", "")
                    topk = tool_args.get("topk", self.topk)
                    req_meta = {
                        "queries": [query],
                        "topk": topk,
                        "return_scores": False,
                    }
                    response = await self.search_client.query_async(req_meta)
                    full_text = process_tool_result(response, "search")

                elif tool_name == "access":
                    # Handle webpage access
                    url = tool_args.get("url", "")
                    access_token = tool_args.get("access_token", "5000")
                    response = await self.search_client.access_async([url])
                    full_text = process_tool_result(response, "access", access_token)

                else:
                    raise ValueError(f"Unknown tool name: {tool_name}")

                result = ToolChannelResponse(
                    success=True,
                    result=full_text,
                )
                await self.output_channel.put(
                    result, key=channel_key, async_op=True
                ).async_wait()

            except Exception as e:
                self.log_error(
                    f"[ERROR] WideSeekR1ToolWorker: Tool execution failed for {tool_name}: {e}, tool name is {tool_name}, tool args is {tool_args}"
                )
                result = ToolChannelResponse(
                    success=False,
                    result=f"Tool execution failed: {str(e)}",
                )
                await self.output_channel.put(
                    result, key=channel_key, async_op=True
                ).async_wait()

        while True:
            # Each tool request is handled in an independent task to keep throughput high.
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            assert request.request_type == "execute"
            assert request.tool_name in ["search", "access"], (
                f"Unknown tool: {request.tool_name}"
            )
            asyncio.create_task(
                generate_and_send(
                    request.session_id, request.tool_name, request.tool_args
                )
            )


if __name__ == "__main__":
    """Test WideSeekR1ToolWorker with both online and offline modes."""
    import argparse

    from omegaconf import OmegaConf

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test WideSeekR1ToolWorker")
    parser.add_argument(
        "--is_online",
        type=lambda x: x.lower() == "true",
        default=False,
        help="Whether to test online mode (True) or offline mode (False)",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Create test configuration
    cfg_dict = {
        "tools": {
            "online": args.is_online,
            "search": {
                "server_addr": "127.0.0.1:8000",
                "topk": 3,
            },
            "enable_cache": True,
            "cache_size": 10000,
            "cache_file": "./test_webpage_cache.json",
            "use_jina": True,
            "max_retries": 10,
            "retry_delay_base": 1.0,
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    async def test_search_and_access(is_online: bool):
        """Test search and access functionality."""
        mode = "ONLINE" if is_online else "OFFLINE"
        print(f"[INFO] Testing {mode} mode...")
        print("[INFO] Creating search client...")

        if is_online:
            client = AsyncOnlineSearchClient(cfg=cfg)
            search_query = "what is google?"
            access_url = "https://www.google.com"
        else:
            client = AsyncSearchClient(cfg=cfg)
            search_query = "What is the capital of France?"
            access_url = "https://en.wikipedia.org/w/index.php?title=List%20of%20capitals%20of%20France"

        # Test 1: Search query
        print(f"\n[TEST 1] Testing search query: '{search_query}'...")
        req_meta = {"queries": [search_query], "topk": 3, "return_scores": False}

        try:
            results = await client.query_async(req_meta)
            print(f"[SUCCESS] Search returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Query {i + 1}:")
                print(f"    Documents: {len(result.get('documents', []))}")
                print(f"    URLs: {len(result.get('urls', []))}")
                if result.get("documents"):
                    print(
                        f"    First document preview: {result['documents'][0][:200] if result['documents'] else 'N/A'}..."
                    )
                if result.get("urls"):
                    print(
                        f"    First URL: {result['urls'][0] if result['urls'] else 'N/A'}"
                    )
        except Exception as e:
            print(f"[ERROR] Search test failed: {e}")
            import traceback

            traceback.print_exc()

        # Test 2: Access URL
        print(f"\n[TEST 2] Testing webpage access: '{access_url}'...")
        try:
            access_results = await client.access_async([access_url])
            print(f"[SUCCESS] Access returned {len(access_results)} results")
            for i, result in enumerate(access_results):
                page_content = result.get("page", "")
                print(f"  URL {i + 1}: {access_url}")
                print(f"    Page length: {len(page_content)} chars")
                print(f"    Type: {result.get('type', 'unknown')}")
                if page_content:
                    print(f"    Preview: {page_content[:500]}...")
        except Exception as e:
            print(f"[ERROR] Access test failed: {e}")
            import traceback

            traceback.print_exc()

        # Test 3: Cache statistics (only for online mode)
        if is_online and hasattr(client, "get_cache_stats"):
            print("\n[TEST 3] Cache statistics...")
            stats = client.get_cache_stats()
            print(f"[INFO] Cache stats: {json.dumps(stats, indent=2)}")

        print(f"\n[INFO] All {mode} mode tests completed!")

    # Run the test
    print("=" * 60)
    print(
        f"Testing WideSeekR1ToolWorker - {'ONLINE' if args.is_online else 'OFFLINE'} mode"
    )
    print("=" * 60)
    asyncio.run(test_search_and_access(args.is_online))
