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
from typing import Any

import aiohttp
from omegaconf import DictConfig

from rlinf.data.tool_call.tool_io_struct import ToolChannelRequest, ToolChannelResponse
from rlinf.scheduler import Channel
from rlinf.workers.agent.tool_worker import ToolWorker


class AsyncSearchClient:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.session = None
        self.server_addr = self.cfg.tools.search.server_addr
        print(self.server_addr)

    async def query_async(self, req_meta: dict[str, Any]) -> list[dict]:
        cnt = 0
        last_exception = None
        while cnt < 10:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"http://{self.server_addr}/retrieve",
                        json=req_meta,
                        timeout=aiohttp.ClientTimeout(total=120, sock_connect=120),
                    ) as response:
                        response.raise_for_status()
                        res = await response.json()
                        return [
                            {
                                "documents": [r["contents"] for r in result],
                                "urls": [r["url"] for r in result],
                                "server_type": "async-search-browser",
                            }
                            for result in res["result"]
                        ]
            except Exception as e:
                last_exception = e
                print(f"Search Engine error {e}. Retry for {cnt} times.")
                cnt += 1
                await asyncio.sleep(10)

        raise RuntimeError(
            "Fail to post search query to RAG server"
        ) from last_exception


class SearchToolWorker(ToolWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.topk = self.cfg.tools.search.topk
        self.request_processor_task = None
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
        def process_tool_result(response):
            res = {}
            res["documents"] = response[0]["documents"]
            res["urls"] = response[0]["urls"]
            res["type"] = "search"

            documents = response[0]["documents"][: self.topk]
            urls = res["urls"][: self.topk]
            if len(documents) > 0:
                doc_id_template = "[Doc {doc_id}]({url}):\n"
                text = (
                    "<information>\n"
                    + "\n\n".join(
                        [
                            doc_id_template.format(doc_id=str(k + 1), url=url)
                            + doc[:5000]
                            for k, (doc, url) in enumerate(zip(documents, urls))
                        ]
                    )
                    + "\n</information>"
                )
            else:
                text = "<information>\nNo search results are found.\n</information>"

            full_text = "\n\n" + text + "\n<think>\n"
            return full_text

        async def generate_and_send(channel_key: str, tool_args: dict):
            try:
                req_meta = {
                    "queries": [tool_args["keyword"]],
                    "topk": self.topk,
                    "return_scores": False,
                }
                response = await self.search_client.query_async(req_meta)
                full_text = process_tool_result(response)

                result = ToolChannelResponse(
                    success=True,
                    result=full_text,
                )
            except Exception as e:
                result = ToolChannelResponse(
                    success=False,
                    result=e,
                )
            await self.output_channel.put(
                result, key=channel_key, async_op=True
            ).async_wait()
            # self.logger.info("SearchToolWorker._process_requests: sent response")

        while True:
            request: ToolChannelRequest = await self.input_channel.get(
                async_op=True
            ).async_wait()
            # self.logger.info("SearchToolWorker._process_requests: got request")
            assert request.request_type == "execute", (
                "SearchToolWorker has no session, so only get 'execute' request_type"
            )
            assert request.tool_name == "search", (
                "SearchToolWorker only execute tool_name 'search'"
            )
            asyncio.create_task(
                generate_and_send(request.session_id, request.tool_args)
            )
