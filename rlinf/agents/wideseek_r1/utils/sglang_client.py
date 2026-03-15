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


import argparse
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

    @classmethod
    async def close_session(cls) -> None:
        """Close the shared aiohttp session when the process exits."""
        if cls._shared_session is not None and not cls._shared_session.closed:
            await cls._shared_session.close()
        cls._shared_session = None

    def __init__(self, llm_ip: str, llm_port: str, llm_type: str):
        """Store endpoint and model metadata for SGLang chat completions.

        Args:
            llm_ip: Host/IP of the SGLang server.
            llm_port: Port of the SGLang server.
            llm_type: Model identifier expected by the API.
        """
        self.llm_ip = llm_ip
        self.llm_port = llm_port
        self.llm_type = llm_type

    async def call_sglang_api(self, messages: list) -> str:
        """
        Call SGLang API with connection pooling.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Response text from the API, or None if all retries fail.
        """
        url = f"http://{self.llm_ip}:{self.llm_port}/v1/chat/completions"
        data = {
            "model": self.llm_type,
            "messages": messages,
        }

        max_retries = 10
        retry_count = 0
        session = await self.get_session()

        # Retry transient failures with a fixed backoff to avoid flakiness.
        while retry_count < max_retries:
            try:
                async with session.post(
                    url, json=data, timeout=aiohttp.ClientTimeout(total=500)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        result_text = result["choices"][0]["message"]["content"]
                        return result_text

            except Exception:
                pass

            retry_count += 1
            await asyncio.sleep(10)

        print(f"[ERROR] SGLangClient: Failed after {max_retries} retries")
        return None


async def _main() -> None:
    """Run a simple manual test against an SGLang endpoint."""
    parser = argparse.ArgumentParser(description="Simple SGLang client test")
    parser.add_argument("--llm-ip", required=True, help="SGLang server host or IP")
    parser.add_argument(
        "--llm-port",
        default="30000",
        help="SGLang server port, defaults to 30000",
    )
    parser.add_argument(
        "--llm-type",
        default="qwen3",
        help="Model name, defaults to qwen3",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, introduce yourself briefly.",
        help="Prompt sent to the model",
    )
    args = parser.parse_args()

    try:
        client = SGLangClient(
            llm_ip=args.llm_ip,
            llm_port=args.llm_port,
            llm_type=args.llm_type,
        )
        messages = [{"role": "user", "content": args.prompt}]
        response = await client.call_sglang_api(messages)
        print(response)
    finally:
        await SGLangClient.close_session()


if __name__ == "__main__":
    asyncio.run(_main())
