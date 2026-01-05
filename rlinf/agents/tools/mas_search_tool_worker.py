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
import atexit
from collections import OrderedDict
import hashlib
import threading
import logging

from omegaconf import DictConfig

import requests
import random
import time
import json
import html
import os
import aiohttp

from typing import Dict, List, Any, Optional
from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


class WebPageCache:
    """Web page cache for storing accessed web pages."""

    def __init__(self, max_size: int = 100000, cache_file: str = "./webpage_cache.json", save_interval: int = 10):
        self.max_size = max_size
        self.cache_file = cache_file
        self.cache = OrderedDict()
        self.lock = threading.Lock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
        self.save_interval = save_interval
        self.operations_since_save = 0

        self.load_from_file()

        atexit.register(self.save_to_file)

    def _generate_cache_key(self, url: str) -> str:
        return hashlib.md5(url.encode()).hexdigest()

    def put(self, url: str, content: str):
        if not url or not content:
            return

        cache_key = self._generate_cache_key(url)

        with self.lock:
            if cache_key in self.cache:
                del self.cache[cache_key]

            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
                self.stats["evictions"] += 1

            self.cache[cache_key] = {
                "url": url,
                "content": content,
                "timestamp": time.time()
            }

            self.operations_since_save += 1
            if self.operations_since_save >= self.save_interval:
                self.operations_since_save = 0
                threading.Thread(target=self._background_save, daemon=True).start()

    def get(self, url: str) -> Optional[str]:
        cache_key = self._generate_cache_key(url)

        with self.lock:
            if cache_key in self.cache:
                entry = self.cache.pop(cache_key)
                self.cache[cache_key] = entry
                self.stats["hits"] += 1
                return entry["content"]
            else:
                self.stats["misses"] += 1
                return None

    def has(self, url: str) -> bool:
        cache_key = self._generate_cache_key(url)
        with self.lock:
            return cache_key in self.cache

    def clear(self):
        with self.lock:
            self.cache.clear()
            self.stats = {"hits": 0, "misses": 0, "evictions": 0}
            self.operations_since_save = 0

    def force_save(self):
        self.save_to_file()
        self.operations_since_save = 0

    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total_requests = self.stats["hits"] + self.stats["misses"]
            hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0

            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.stats["hits"],
                "misses": self.stats["misses"],
                "evictions": self.stats["evictions"],
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }

    def _background_save(self):
        try:
            self.save_to_file()
        except Exception as e:
            print(f"[ERROR] WebPageCache: Background save failed: {e}")

    def save_to_file(self):
        try:
            with self.lock:
                ordered_cache = []
                for key, value in self.cache.items():
                    ordered_cache.append((key, value))

                cache_data = {
                    "cache_ordered": ordered_cache,
                    "stats": self.stats,
                    "max_size": self.max_size,
                    "saved_at": time.time()
                }

            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2, ensure_ascii=False)

            print(f"[DEBUG] WebPageCache: Saved {len(self.cache)} entries to {self.cache_file}")

        except Exception as e:
            print(f"[ERROR] WebPageCache: Failed to save cache to {self.cache_file}: {e}")

    def load_from_file(self):
        """Load cache from JSON file."""
        if not os.path.exists(self.cache_file):
            print(f"[DEBUG] WebPageCache: No existing cache file {self.cache_file}, starting fresh")
            return

        try:
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            with self.lock:
                if "cache_ordered" in cache_data:
                    ordered_cache = cache_data["cache_ordered"]
                    self.cache = OrderedDict(ordered_cache)
                    print(f"[DEBUG] WebPageCache: Loaded ordered cache format")
                else:
                    loaded_cache = cache_data.get("cache", {})
                    self.cache = OrderedDict(loaded_cache)
                    print(f"[DEBUG] WebPageCache: Loaded legacy cache format (LRU order may be lost)")

                self.stats = cache_data.get("stats", {"hits": 0, "misses": 0, "evictions": 0})

                while len(self.cache) > self.max_size:
                    self.cache.popitem(last=False)
                    self.stats["evictions"] += 1

            saved_at = cache_data.get("saved_at", 0)
            saved_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(saved_at))

            print(f"[DEBUG] WebPageCache: Loaded {len(self.cache)} entries from {self.cache_file} (saved at {saved_time})")

        except Exception as e:
            print(f"[ERROR] WebPageCache: Failed to load cache from {self.cache_file}: {e}")
            with self.lock:
                self.cache = OrderedDict()
                self.stats = {"hits": 0, "misses": 0, "evictions": 0}


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
                        limit=100,  # Max total connections
                        limit_per_host=50,  # Max connections per host
                        ttl_dns_cache=300,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=30),
                        trust_env=True
                    )
        return cls._shared_session

    @classmethod
    def _get_search_semaphore(cls):
        if cls._search_semaphore is None:
            cls._search_semaphore = asyncio.Semaphore(20)
        return cls._search_semaphore

    @classmethod
    def _get_access_semaphore(cls):
        if cls._access_semaphore is None:
            cls._access_semaphore = asyncio.Semaphore(10)
        return cls._access_semaphore

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(self.__class__.__name__)

        # Retry configuration
        self.max_retries = self.cfg.tools.get('max_retries', 15)
        self.retry_delay_base = self.cfg.tools.get('retry_delay_base', 5)

        # Serper API
        self.serper_server_addr = "https://google.serper.dev"
        self.serper_api_key = os.environ.get('SERPER_API_KEY', '')
        if not self.serper_api_key:
            raise RuntimeError("Serper API key is not set. Please set the SERPER_API_KEY environment variable.")
        self.serper_headers = {
            'X-API-KEY': self.serper_api_key,
            'Content-Type': 'application/json'
        }
        self.logger.info(f"Initialized Serper API client with key: {self.serper_api_key[:8]}...")

        # Jina API
        self.use_jina = self.cfg.tools.get('use_jina', False)
        self.jina_api_key = os.environ.get('JINA_API_KEY', '')
        if self.use_jina and not self.jina_api_key:
            raise RuntimeError("Jina is enabled but the API key is not set. Please set the JINA_API_KEY environment variable.")
        if self.use_jina:
            self.logger.info(f"Initialized Jina API client with key: {self.jina_api_key[:8]}...")

        # Web page cache
        cache_enabled = self.cfg.tools.get('enable_cache', True)
        cache_size = self.cfg.tools.get('cache_size', 10000)
        cache_file = self.cfg.tools.get('cache_file', './webpage_cache.json')

        if cache_enabled:
            self.webpage_cache = WebPageCache(cache_size, cache_file, save_interval=5)
            self.logger.info(f"Web page cache enabled: size={cache_size}, file={cache_file}")
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
                timeout=aiohttp.ClientTimeout(total=120)
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
                    delay = self.retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 20)
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

    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
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

                documents = [result.get("title", "") + " " + result.get("snippet", "") for result in organic_results]
                urls = [result.get("link", "") for result in organic_results]

                formatted_results.append({
                    "documents": documents,
                    "urls": urls,
                    "server_type": "async-online-search"
                })
            else:
                formatted_results.append({
                    "documents": [],
                    "urls": [],
                    "server_type": "async-online-search"
                })

        return formatted_results

    async def _do_jina_access(self, session, url: str) -> Dict:
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
            'Authorization': f'Bearer {self.jina_api_key}',
            'Content-Type': 'application/json',
        }

        async with session.get(f'https://r.jina.ai/{url}', headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            if response.status == 200:
                content = await response.text()
                return dict(page=content, type="jina")
            elif response.status != 429:
                return dict(page="The current URL cannot be searched. Please switch to a different URL and try again.", type="jina")    
            # elif response.status == 422:
            #     content = await response.text()
            #     return dict(page=content, type="jina")                           
            else:
                raise Exception(f"HTTP {response.status}")

    async def _single_jina_access(self, session, url: str) -> Dict:
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
                    delay = self.retry_delay_base * (2 ** (attempt - 1)) + random.uniform(0, 20)
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
                    return dict(page="The current URL cannot be searched. Please switch to a different URL and try again.", type="access")

        return dict(page="The current URL cannot be searched. Please switch to a different URL and try again.", type="access")

    async def access_async(self, urls: List[str]) -> List[Dict]:
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
                tasks = [self._single_jina_access(session, url) for url in urls_to_fetch]
                fetched_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Merge fetched results back
            fetch_index = 0
            for i, result in enumerate(results):
                if result is None:
                    fetched_result = fetched_results[fetch_index] if fetch_index < len(fetched_results) else dict(page="The current URL cannot be searched. Please switch to a different URL and try again.", type="access")

                    # Handle exceptions
                    if isinstance(fetched_result, Exception):
                        fetched_result = dict(page="The current URL cannot be searched. Please switch to a different URL and try again.", type="access")

                    results[i] = fetched_result

                    # Cache successful fetches
                    if self.webpage_cache and fetched_result.get("page"):
                        self.webpage_cache.put(urls[i], fetched_result["page"])

                    fetch_index += 1

        # Fill in any remaining None values
        for i, result in enumerate(results):
            if result is None:
                results[i] = dict(page="The current URL cannot be searched. Please switch to a different URL and try again.", type="access")

        # Add server_type to all results
        for result in results:
            result["server_type"] = "async-online-search"

        return results

    def get_cache_stats(self) -> Dict[str, Any]:
        if self.webpage_cache:
            return self.webpage_cache.get_stats()
        else:
            return {"cache_disabled": True}

    def clear_cache(self):
        if self.webpage_cache:
            self.webpage_cache.clear()

    def force_save_cache(self):
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
                        limit=100,  # Max total connections
                        limit_per_host=50,  # Max connections per host
                        ttl_dns_cache=300,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=30),
                        trust_env=False
                    )
        return cls._shared_session

    def __init__(self, cfg:DictConfig):
        self.cfg = cfg
        self.server_addr = self.cfg.tools.search.server_addr
        print(f"[INFO] AsyncSearchClient: Using local server at {self.server_addr}")

    async def query_async(self, req_meta: Dict[str, Any]) -> List[Dict]:
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
                        ) for result in res["result"]
                    ]
            except Exception as e:
                last_exception = e
                print(f"[WARNING] AsyncSearchClient: Search query error {e}. Retry {cnt} times.")
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError("Fail to post search query to RAG server") from last_exception

    async def access_async(self, urls: List[str]) -> List[Dict]:
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
                        ) for result in res["result"]
                    ]
            except Exception as e:
                last_exception = e
                print(f"[WARNING] AsyncSearchClient: Access request error {e}. Retry {cnt} times.")
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError("Fail to post access request to RAG server") from last_exception


class MASToolWorker(ToolWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.topk = self.cfg.tools.search.topk
        self.request_processor_task = None

        # Determine whether to use online or local search
        self.use_online_search = self.cfg.tools.get('online', False)

        if self.use_online_search:
            self.log_info("[INFO] MASToolWorker: Using online search (Serper API)")
            self.search_client = AsyncOnlineSearchClient(cfg=self.cfg)
        else:
            self.log_info("[INFO] MASToolWorker: Using local search server")
            self.search_client = AsyncSearchClient(cfg=self.cfg)

    def init_worker(self, input_channel: Channel, output_channel: Channel):
        self.input_channel = input_channel
        self.output_channel = output_channel

    def start_server(self):
        loop = asyncio.get_running_loop()
        self.request_processor_task = loop.create_task(self._process_requests())

    def stop_server(self):
        # Cancel request processor task
        if self.request_processor_task and not self.request_processor_task.done():
            self.request_processor_task.cancel()

    async def _process_requests(self):
        def process_tool_result(response, tool_type):
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
                    text = "\n\n".join([
                        doc_id_template.format(doc_id=str(k+1), url=url) + doc[:5000]
                        for k, (doc, url) in enumerate(zip(documents, urls))
                    ])
                else:
                    text = "No search results are found."

                return text

            elif tool_type == "access":
                # Process webpage access following ASearcher's logic
                page = response[0].get("page", "")

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
                    return page[:25000]
                else:
                    return "No More Information is Found for this URL."

            else:
                raise ValueError(f"Unknown tool type: {tool_type}")


        async def generate_and_send(channel_key: str, tool_name: str, tool_args: dict):
            """Handle both search and access tool requests."""
            try:
                if tool_name == "search":
                    # Handle search query
                    query = tool_args.get('query', '')
                    topk = tool_args.get('topk', self.topk)
                    req_meta = {
                        "queries": [query],
                        "topk": topk,
                        "return_scores": False
                    }
                    response = await self.search_client.query_async(req_meta)
                    full_text = process_tool_result(response, "search")

                elif tool_name == "access":
                    # Handle webpage access
                    url = tool_args.get('url', '')
                    response = await self.search_client.access_async([url])
                    full_text = process_tool_result(response, "access")

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
                self.log_error(f"[ERROR] MASToolWorker: Tool execution failed for {tool_name}: {e}, tool name is {tool_name}, tool args is {tool_args}")
                result = ToolChannelResponse(
                    success=False,
                    result=f"Tool execution failed: {str(e)}",
                )
                await self.output_channel.put(
                    result, key=channel_key, async_op=True
                ).async_wait()

        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            assert request.request_type == "execute"
            assert request.tool_name in ["search", "access"], f"Unknown tool: {request.tool_name}"
            asyncio.create_task(
                generate_and_send(request.session_id, request.tool_name, request.tool_args)
            )


if __name__ == "__main__":
    """Test AsyncOnlineSearchClient independently."""
    from omegaconf import OmegaConf

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create test configuration
    cfg_dict = {
        "tools": {
            "online": True,
            "search": {
                "server_addr": "127.0.0.1:8000",
                "topk": 3,
            },
            "enable_cache": True,
            "cache_size": 10000,
            "cache_file": "./test_webpage_cache.json",
            "use_jina": True,
            "max_retries": 10,  # Fewer retries for testing
            "retry_delay_base": 1.0,
        }
    }
    cfg = OmegaConf.create(cfg_dict)

    async def test_online_search():
        """Test online search functionality."""
        print("[INFO] Creating AsyncOnlineSearchClient...")
        client = AsyncOnlineSearchClient(cfg=cfg)

        # Test 1: Search query
        print("\n[TEST 1] Testing search query...")
        req_meta = {
            "queries": ["苹果是什么"],
            "topk": 3,
            "return_scores": False
        }

        try:
            results = await client.query_async(req_meta)
            print(f"[SUCCESS] Search returned {len(results)} results")
            for i, result in enumerate(results):
                print(f"  Query {i+1}:")
                print(f"    Documents: {len(result.get('documents', []))}")
                print(f"    URLs: {len(result.get('urls', []))}")
                if result.get('documents'):
                    print(f"    First document preview: {result['documents']}...")
        except Exception as e:
            print(f"[ERROR] Search test failed: {e}")
            import traceback
            traceback.print_exc()

        # Test 2: Access URL (if Jina is enabled)
        if cfg.tools.use_jina:
            print("\n[TEST 2] Testing webpage access with Jina...")
            test_urls = ["https://www.baidu.com"]

            try:
                access_results = await client.access_async(test_urls)
                print(f"[SUCCESS] Access returned {len(access_results)} results")
                for i, result in enumerate(access_results):
                    page_content = result.get('page', '')
                    print(f"  URL {i+1}:")
                    print(f"    Page length: {len(page_content)} chars")
                    print(f"    Type: {result.get('type', 'unknown')}")
                    if page_content:
                        print(f"    Preview: {page_content[:2000]}...")
            except Exception as e:
                print(f"[ERROR] Access test failed: {e}")
                import traceback
                traceback.print_exc()

        # Test 3: Cache statistics
        print("\n[TEST 3] Cache statistics...")
        stats = client.get_cache_stats()
        print(f"[INFO] Cache stats: {json.dumps(stats, indent=2)}")

        print("\n[INFO] All tests completed!")

    # Run the test
    print("=" * 60)
    print("Testing AsyncOnlineSearchClient with session pooling")
    print("=" * 60)
    asyncio.run(test_online_search())
