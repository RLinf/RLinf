import asyncio
import json
from typing import Callable, Literal, Optional

import aiohttp


async def _http_post_json(session: aiohttp.ClientSession, url: str, payload: dict, timeout: Optional[float] = None):
    async with session.post(url, json=payload, timeout=timeout) as resp:
        resp.raise_for_status()
        return await resp.json()


async def run_tool_calls_on_server_async(
    tool_calls: list[dict],
    session: aiohttp.ClientSession,
    *,
    host_addr: str = "localhost",
    host_port: str | int = "8000",
) -> list[str]:
    """Send a batch of tool calls to a code-judge compatible server and return text results.

    Each tool_call should be a dict like {"name": str, "arguments": {...}}.
    Currently supported names:
      - "python_code_with_standard_io": executes python with input/output
      - "jupyter_code": executes code in a simulated jupyter-like environment (mapped to python exec)
    """
    submissions = []
    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("arguments", {})
        if name == "python_code_with_standard_io":
            code = args.get("code", "")
            std_input = args.get("input", "")
            submissions.append({
                "type": "python",
                "solution": code,
                "input": std_input,
            })
        elif name == "jupyter_code":
            code = args.get("code", "")
            submissions.append({
                "type": "python",
                "solution": code,
            })
        else:
            submissions.append({
                "type": "python",
                "solution": f"print('Unsupported tool: {name}')",
            })

    url = f"http://{host_addr}:{host_port}/run/long-batch"
    payload = {"type": "batch", "submissions": submissions}
    result = await _http_post_json(session, url, payload)
    results = result.get("results", [])
    texts: list[str] = []
    for r in results:
        text = r.get("stdout") or r.get("stderr") or ""
        texts.append(text)
    return texts


