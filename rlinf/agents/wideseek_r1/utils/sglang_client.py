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
import threading
import aiohttp

class SGLangClient:
    """SGLang API client with connection pooling."""

    # Class-level shared session for connection pooling
    _shared_session = None
    _session_lock = threading.Lock()

    @classmethod
    async def get_session(cls):
        """Get or create shared aiohttp session with connection pooling."""
        if cls._shared_session is None or cls._shared_session.closed:
            with cls._session_lock:
                if cls._shared_session is None or cls._shared_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=2000,  # Max total connections
                        limit_per_host=1000,  # Max connections per host
                        ttl_dns_cache=1000,  # DNS cache TTL
                        enable_cleanup_closed=True,
                    )
                    cls._shared_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=aiohttp.ClientTimeout(total=1000, sock_connect=500),
                        trust_env=False,
                    )
        return cls._shared_session

    def __init__(self, llm_ip: str, llm_port: str, llm_type: str):
        self.llm_ip = llm_ip
        self.llm_port = llm_port
        self.llm_type = llm_type

    async def call_sglang_api(self, messages: list) -> str:
        """
        Call SGLang API with connection pooling.

        Args:
            llm_ip: LLM server IP
            llm_port: LLM server port
            llm_type: LLM model type
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Response text from the API, or None if failed
        """
        url = f"http://{self.llm_ip}:{self.llm_port}/v1/chat/completions"
        data = {
            "model": self.llm_type,
            "messages": messages,
        }

        max_retries = 10
        retry_count = 0
        session = await self.get_session()

        while retry_count < max_retries:
            try:
                async with session.post(
                    url, json=data, timeout=aiohttp.ClientTimeout(total=500)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_text = result["choices"][0]["message"]["content"]
                        return result_text
                    else:
                        response_text = await response.text()
                        print(
                            f"[ERROR] SGLangClient: Failed calling sglang: {response.status}, response: {response_text}, Retry {retry_count}/{max_retries}"
                        )
            except Exception as e:
                print(
                    f"[ERROR] SGLangClient: Exception error in calling sglang: {e}, Retry {retry_count}/{max_retries}"
                )

            retry_count += 1
            await asyncio.sleep(10)

        print(f"[ERROR] SGLangClient: Failed after {max_retries} retries")
        return None
