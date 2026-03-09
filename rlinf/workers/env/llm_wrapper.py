import asyncio
import base64
import sys
import time
from pathlib import Path
project_root = Path("/mnt/project_rlinf/yingcheng/mobile-agent")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

rlinf_path = project_root / "RLinf"
if str(rlinf_path) not in sys.path:
    sys.path.insert(0, str(rlinf_path))

import os

current_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = []
if current_pythonpath:
    pythonpath_parts.append(current_pythonpath)
pythonpath_parts.append(str(project_root))
pythonpath_parts.append(str(rlinf_path))
os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)

from rlinf.scheduler import Channel
import numpy as np
from typing import Optional, Any
from PIL import Image
import io
from uuid import uuid4

def encode_image(image: np.ndarray) -> str:
    pil = Image.fromarray(image)
    buf = io.BytesIO()
    pil.save(buf, format="JPEG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

class LLMWrapper:
    def __init__(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel
    ):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel

    def predict_mm(self, text_prompt: str, images: list[np.ndarray]) -> tuple[str, Optional[bool], Any]:
        t_encode = time.perf_counter()
        messages = self._build_messages(text_prompt, images)
        image_encode_s = time.perf_counter() - t_encode

        result = self._vl_generate(messages)

        if isinstance(result, dict):
            result["image_encode_s"] = image_encode_s

        return self._parse_result(result)
    
    def _build_messages(self, text_prompt: str, images: list[np.ndarray]) -> list[dict]:
        content = []

        for image in images:
            content.append({
                "type": "image",
                "image": f"data:image/jpeg;base64,{encode_image(image)}",
            })
        content.append({
            "type": "text",
            "text": text_prompt
        })  
        return [{
            "role": "user",
            "content": content
        }]
    
    def _vl_generate(self, messages: list[dict]) -> dict:
        channel_key = uuid4().hex
        request = {
            "channel_key": channel_key,
            "messages": messages,
            "sampling_params": {
                "temperature": 0.0,
                "top_p": 0.95,
                "max_new_tokens": 2000,
            },
        }

        async def _generate():
            t0 = time.perf_counter()
            await self.generate_input_channel.put(request, async_op=True).async_wait()
            agent_put_s = time.perf_counter() - t0
            t1 = time.perf_counter()
            result = await self.generate_output_channel.get(channel_key, async_op=True).async_wait()
            agent_get_s = time.perf_counter() - t1
            if isinstance(result, dict):
                result["agent_put_s"] = agent_put_s
                result["agent_get_s"] = agent_get_s
            return result

        try:
            return asyncio.run(_generate())
        except RuntimeError as e:
            raise RuntimeError(
                f"EngineLLMWrapper: asyncio.run failed (maybe already in async context). {e}"
            ) from e

    def _parse_result(self, result: dict) -> tuple[str, Optional[bool], Any]:
        text = result["text"]
        return (text, None, result)