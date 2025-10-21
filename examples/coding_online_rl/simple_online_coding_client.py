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
import uuid
from datetime import datetime

import httpx

batch_size = 16
epoch = 10 * 2


async def agenerate(prefix, suffix):
    TARGET_URL = "http://127.0.0.1:8081/v1/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }
    body = {
        "model": "test-model",
        "prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
        "max_tokens": 50,
        "temperature": 0.7,
        "stream": False,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TARGET_URL,
            headers=headers,
            json=body,
            timeout=15.0,
        )
        print(f"agenerate get response: {response.json()}")
        return response.json()["choices"][0]["text"]


async def atrack(prefix, suffix, completion, accepted):
    TARGET_URL = "http://127.0.0.1:8082/api/training/submit"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-token",
    }

    body = {
        "completionId": str(uuid.uuid4()),
        "filepath": "file:///Users/qurakchin/.vscode/extensions/continue.continue-1.2.3-darwin-arm64/continue_tutorial.py",
        "prefix": prefix,
        "suffix": suffix,
        "prompt": f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>",
        "completion": completion,
        "modelProvider": "openai",
        "modelName": "Qwen2.5-Coder-1.5B-Q8_0.gguf",
        "accepted": accepted,
        "timestamp": datetime.now().isoformat(),
        "time": 4294,
        "uniqueId": str(uuid.uuid4()),
        "numLines": 1,
        "cacheHit": False,
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TARGET_URL,
            headers=headers,
            json=body,
            timeout=15.0,
        )
        print(f"atrack get response: {response.json()}")


async def single_iteration(prefix, suffix):
    await asyncio.sleep(0.001)
    completion = await agenerate(prefix=prefix, suffix=suffix)
    await asyncio.sleep(0.001)
    await atrack(prefix=prefix, suffix=suffix, completion=completion, accepted=True)


async def loop():
    prefix = "if x[j] > x[j + 1]:\n                x[j], x[j + 1] = x[j + 1], x[j]\n    return x\n\ndef han"
    suffix = '\n# —————————————————————————————————————————————————     Agent      ————————————————————————————————————————————————— #\n#           Agent equips the Chat model with the tools needed to handle a wide range of coding tasks, allowing\n#           the model to make decisions and save you the work of manually finding context and performing actions.\n\n# 1. Switch from "Chat" to "Agent" mode using the dropdown in the bottom left of the input box'

    tasks = []
    for i in range(batch_size * epoch):
        task = asyncio.create_task(single_iteration(prefix, suffix))
        tasks.append(task)

        if i % batch_size == 0:
            await asyncio.gather(*tasks)
            tasks = []

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(loop())
