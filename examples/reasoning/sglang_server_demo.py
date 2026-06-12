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
import json
import sys
import time

import hydra
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Cluster
from rlinf.scheduler.placement import ComponentPlacement
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.utils.logging import get_logger
from rlinf.workers.rollout.sglang_server import launch_sglang_router_and_server

logger = get_logger()


def _prompts_to_chat_messages(prompts: list[str]) -> list[list[dict]]:
    return [[{"role": "user", "content": p}] for p in prompts]


def test_sync_generate(
    client: InferenceHTTPClient,
    prompts: list[str],
    sampling_params: dict,
) -> None:
    logger.info("running sync /generate pass")
    start = time.time()
    for i, prompt in enumerate(prompts):
        logger.info(f"sync /generate request {i + 1}/{len(prompts)}")
        client.generate(prompt=prompt, sampling_params=sampling_params)
    logger.info(f"sync /generate pass took {time.time() - start:.2f} seconds")


def test_async_generate(
    router_url: str,
    prompts: list[str],
    sampling_params: dict,
) -> None:
    async def _run() -> None:
        async with InferenceHTTPClient(router_url) as aclient:
            tasks = [
                aclient.async_generate(prompt=p, sampling_params=sampling_params)
                for p in prompts
            ]
            await asyncio.gather(*tasks)

    logger.info("running async /generate pass")
    start = time.time()
    asyncio.run(_run())
    logger.info(f"async /generate pass took {time.time() - start:.2f} seconds")


def test_sync_chat(
    client: InferenceHTTPClient,
    model: str,
    prompts: list[str],
    max_new_tokens: int,
) -> None:
    logger.info("running sync /v1/chat/completions pass")
    start = time.time()
    messages_list = _prompts_to_chat_messages(prompts)
    for i, messages in enumerate(messages_list):
        logger.info(f"sync chat request {i + 1}/{len(messages_list)}")
        client.chat_completion(
            messages=messages,
            model=model,
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
    logger.info(
        f"sync /v1/chat/completions pass took {time.time() - start:.2f} seconds"
    )


def test_async_chat(
    router_url: str,
    model: str,
    prompts: list[str],
    max_new_tokens: int,
) -> None:
    messages_list = _prompts_to_chat_messages(prompts)

    async def _run() -> None:
        async with InferenceHTTPClient(router_url) as aclient:
            tasks = [
                aclient.async_chat_completion(
                    messages=messages,
                    model=model,
                    temperature=0.0,
                    max_tokens=max_new_tokens,
                )
                for messages in messages_list
            ]
            await asyncio.gather(*tasks)

    logger.info("running async /v1/chat/completions pass")
    start = time.time()
    asyncio.run(_run())
    logger.info(
        f"async /v1/chat/completions pass took {time.time() - start:.2f} seconds"
    )


@hydra.main(version_base="1.1", config_path="config", config_name="sglang_server_demo")
def main(cfg: DictConfig) -> None:
    logger.info(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ComponentPlacement(cfg, cluster)
    router_server_args = cfg.rollout

    server_group, router_group = launch_sglang_router_and_server(
        cfg,
        cluster,
        rollout_hardware_ranks=placement.get_hardware_ranks("llm"),
        router_server_args=router_server_args,
    )
    logger.info("launch_sglang_router_and_server returned")

    router_url = router_group.get_router_url().wait()[0]
    logger.info(f"router ready at {router_url}")

    prompts = [
        "The capital of France is",
        "RLinf is a framework that",
    ]
    max_new_tokens = int(cfg.get("demo_max_new_tokens", 64))
    sampling_params = {
        "temperature": 0.0,
        "max_new_tokens": max_new_tokens,
    }
    chat_model = router_server_args.model_path

    client = InferenceHTTPClient(router_url)

    test_sync_generate(client, prompts, sampling_params)
    test_async_generate(router_url, prompts, sampling_params)
    test_sync_chat(client, chat_model, prompts, max_new_tokens)
    test_async_chat(router_url, chat_model, prompts, max_new_tokens)

    logger.info("shutting down router + servers")
    router_group.shutdown().wait()
    server_group.shutdown().wait()
    logger.info("done")


if __name__ == "__main__":
    # Force unbuffered stdio even if -u wasn't passed.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    main()
