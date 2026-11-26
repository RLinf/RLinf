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

"""Record the Qwen3.5 image inputs used by RPent planner turns."""

import math
from functools import cached_property
from itertools import groupby
from pathlib import Path

from rlinf.hybrid_engines.sglang.multimodal.replay import generate_with_capture
from rlinf.workers.rollout.session_server.payload import routing_key_for
from rlinf.workers.rollout.session_server.server_worker import SessionServerWorker


class RPentSessionServerWorker(SessionServerWorker):
    """Forward planner image turns and keep their native SGLang inputs for training."""

    async def _chat_completions(self, sid, body):
        # RPent sends its interactive top-k/top-p defaults on every request.
        # Training must sample from the configured behavior distribution.
        body = {
            key: value
            for key, value in body.items()
            if key not in ("temperature", "top_k", "top_p")
        }
        if body.get("tool_choice") == "required":
            # The batch-action planner validates tool use after unconstrained sampling.
            # A decode grammar would change the behavior policy used for GRPO.
            body["tool_choice"] = "auto"
        return await super()._chat_completions(sid, body)

    @cached_property
    def _image_token_id(self) -> int:
        from transformers import AutoConfig

        return AutoConfig.from_pretrained(
            self._cfg.rollout.model.model_path
        ).image_token_id

    async def _sample_turn(self, session, forwarded, task_name, chat_messages):
        result, entry, history = await self._generate_planner_turn(
            session, forwarded, task_name, chat_messages
        )
        meta = result["meta_info"]
        if meta.get("finish_reason", {}).get("type") != "abort":
            count = len(result["output_ids"])
            if self._cfg.rollout.get("record_token_weight_versions", False):
                versions = meta.get("rollout_weight_versions") or []
                if len(versions) != count or any(
                    type(v) is not int or not 0 <= v <= int(meta["weight_version"])
                    for v in versions
                ):
                    raise ValueError(
                        "The engine must return one installed policy version per sampled token"
                    )
            if forwarded["return_logprob"]:
                logprobs = meta.get("output_token_logprobs") or []
                if len(logprobs) != count or not all(
                    math.isfinite(float(item[0])) for item in logprobs
                ):
                    raise ValueError(
                        "The engine must return one finite logprob per sampled token"
                    )
        return result, entry, history

    async def _generate_planner_turn(
        self, session, forwarded, task_name, chat_messages
    ):
        images = [
            part["image_url"]["url"]
            for message in chat_messages or ()
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part.get("type") == "image_url"
        ]
        if not images:
            return await super()._sample_turn(
                session, forwarded, task_name, chat_messages
            )
        if any(
            not isinstance(url, str) or not url.startswith("data:image/")
            for url in images
        ):
            raise ValueError(
                "RPent image observations must be embedded image data URLs"
            )

        prompt_ids = forwarded["input_ids"]
        spans = []
        for token_id, positions in groupby(
            enumerate(prompt_ids), key=lambda item: item[1]
        ):
            if token_id == self._image_token_id:
                indices = [position for position, _ in positions]
                spans.append((indices[0], indices[-1] + 1))
        if len(spans) != len(images):
            raise ValueError(
                f"Rendered {len(spans)} image spans for {len(images)} images"
            )

        previous = session.image_history
        if previous is not None:
            prefix = session.last_input + session.last_output
            stored = session.chat.messages
            if (
                prompt_ids[: len(prefix)] != prefix
                or chat_messages[: len(stored)] != stored
            ):
                previous = None

        async def forward(**request):
            return await self._forward_generate(
                request, routing_key_for(session.group_id), task_name
            )

        result, entry, history = await generate_with_capture(
            forward,
            prompt_ids,
            images,
            forwarded["sampling_params"],
            {
                "image_spans": spans,
                "history": previous,
                "token_budget": (
                    self._max_seq_length,
                    self._cfg.agentloop.minimum_output_tokens,
                ),
                "store_dir": str(
                    Path(self._cfg.tools.rpent.output_dir) / "training_images"
                ),
            },
            forwarded["return_logprob"],
        )
        entry["extends_previous_turn"] = previous is not None
        return result, entry, history
