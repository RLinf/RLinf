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

"""SGLang text generation for a synchronous Qwen VLM policy.

Each environment keeps one session so a later turn sends only the new
observation. A weight sync closes those sessions: the radix cache belongs to
the previous policy.
"""

from __future__ import annotations

import base64
import io
import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

from rlinf.models.embodiment.qwen_vlm.protocol import (
    DEFAULT_REASONING_EFFORT,
    chat_template_kwargs,
)


class QwenVLMSGLangClient:
    """HTTP client for SGLang ``/generate`` with sticky per-env sessions."""

    def __init__(
        self,
        base_url: str = "",
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        transport: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        engine: Any = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self._transport = transport
        self.engine = engine
        self._sessions: dict[int, str] = {}
        self._committed: dict[int, str] = {}
        self.weight_version = 0
        self.synced_versions: list[int] = []
        self.memory_released = False
        chat_template_kwargs(reasoning_effort)

    def generate(
        self,
        env_index: int,
        *,
        full_text: str,
        max_new_tokens: int,
        images: list[Any] | None = None,
    ) -> dict[str, Any]:
        """Generate one turn. A matching session sends only the new suffix."""
        if self.memory_released:
            raise RuntimeError("SGLang memory is released; resume it before generate")
        committed = self._committed.get(env_index, "")
        continuation = (
            env_index in self._sessions
            and bool(committed)
            and full_text.startswith(committed)
            and len(full_text) > len(committed)
        )
        if continuation:
            text = full_text[len(committed) :]
        else:
            if env_index in self._sessions:
                self.close_session(env_index)
            self._sessions[env_index] = self._open_session(env_index)
            text = full_text
        payload = {
            "text": text,
            "session_params": {"id": self._sessions[env_index]},
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": int(max_new_tokens),
            },
            "rid": f"qwen-env-{env_index}-v{self.weight_version}",
            "image_data": list(images or []),
            "return_logprob": False,
        }
        response = self._post("/generate", payload)
        if "text" not in response or "output_ids" not in response:
            raise RuntimeError("SGLang /generate did not return text and output_ids")
        self._committed[env_index] = full_text + str(response["text"])
        return response

    def close_session(self, env_index: int) -> None:
        self._committed.pop(env_index, None)
        session_id = self._sessions.pop(env_index, None)
        if session_id is not None:
            self._post("/close_session", {"session_id": session_id})

    def close_sessions(self) -> None:
        for env_index in list(self._sessions):
            self.close_session(env_index)

    def release_gpu_memory(self) -> None:
        """Return SGLang weights and KV cache to the host before FSDP training."""
        self.close_sessions()
        self._post("/release_memory_occupation", {})
        self.memory_released = True

    def resume_gpu_memory(self) -> None:
        """Bring SGLang back onto the GPU for the next rollout."""
        self._post("/resume_memory_occupation", {})
        self.memory_released = False

    def sync_weights(
        self, version: int, parameters: list[tuple[str, Any]] | None = None
    ) -> None:
        """Drop prefix caches, then push the new policy into SGLang if an engine is bound."""
        self.weight_version = int(version)
        self.synced_versions.append(self.weight_version)
        self.close_sessions()
        self._post("/flush_cache", {"version": self.weight_version})
        if self.engine is not None and parameters is not None:
            updater = getattr(self.engine, "update_weights_from_tensor", None)
            if updater is None:
                raise RuntimeError("SGLang engine has no update_weights_from_tensor")
            updater(parameters)

    def _open_session(self, env_index: int) -> str:
        # capacity_of_str_len is required by SGLang 0.5.19 and is not enforced.
        response = self._post("/open_session", {"capacity_of_str_len": 1_000_000})
        if isinstance(response, str) and response:
            return response
        session_id = None
        if isinstance(response, dict):
            session_id = response.get("session_id") or response.get("id")
        if not session_id:
            raise RuntimeError("SGLang /open_session did not return a session_id")
        return str(session_id)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._transport is not None:
            return self._transport(path, payload)
        if not self.base_url:
            raise RuntimeError(
                "Qwen VLM generation needs rollout.sglang.base_url or a transport"
            )
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(self._jsonable(payload)).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise RuntimeError(f"SGLang request {path} failed: {exc}") from exc
        parsed = json.loads(body) if body else {}
        if isinstance(parsed, str):
            return parsed
        if not isinstance(parsed, dict):
            raise RuntimeError(f"SGLang response for {path} is not an object")
        return parsed

    @staticmethod
    def _jsonable(payload: dict[str, Any]) -> dict[str, Any]:
        encoded = dict(payload)
        images = []
        for image in payload.get("image_data", []):
            if isinstance(image, str):
                images.append(image)
                continue
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            images.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        encoded["image_data"] = images
        return encoded
