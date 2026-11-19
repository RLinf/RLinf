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

"""Qwen VLM turns over a patched SGLang text server.

The adapter keeps the chat, the sticky session, and the token packing the
actor replays. Weights and GPU memory stay on the SGLang scheduler's CUDA-IPC
path; this module only talks to ``/generate`` and the session routes.
"""

from __future__ import annotations

import base64
import fcntl
import io
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from rlinf.models.embodiment.qwen_vlm.conversation import (
    IncrementalEpisodeTokens,
    render_incremental_user_delta,
)
from rlinf.models.embodiment.qwen_vlm.policy import (
    QwenVLMPolicy,
    _state_from_observation,
    _turn_images_from_observation,
    extend_mm_token_type_ids,
)
from rlinf.models.embodiment.qwen_vlm.protocol import (
    DEFAULT_REASONING_EFFORT,
    LIBERO_SCHEMA,
    ROBODOJO_SCHEMA,
    InvalidAction,
    action_to_vector,
    build_user_content,
    cameras_for_schema,
    chat_template_kwargs,
    parse_action,
    system_prompt,
)


def extract_sglang_topk_logprobs(
    response: dict[str, Any],
    output_ids: list[int],
    topk: int,
) -> tuple[list[float] | None, list[list[int]] | None, list[list[float]] | None]:
    """Extract per-token logprobs and top-K heads from an SGLang ``/generate`` response.

    Supports SGLang 0.5.x ``meta_info.output_top_logprobs`` where each token
    entry is a list of ``(logprob, token_id[, text])`` tuples or dicts with
    ``logprob``/``token_id`` keys.
    """
    if topk <= 0:
        return None, None, None
    meta = response.get("meta_info") or {}
    raw_token_lps = meta.get("output_token_logprobs")
    token_logprobs: list[float] | None = None
    if isinstance(raw_token_lps, list) and len(raw_token_lps) == len(output_ids):
        token_logprobs = []
        for entry in raw_token_lps:
            if isinstance(entry, (list, tuple)) and entry:
                token_logprobs.append(float(entry[0]))
            elif isinstance(entry, dict) and "logprob" in entry:
                token_logprobs.append(float(entry["logprob"]))
            else:
                token_logprobs.append(float(entry))

    raw_topk = meta.get("output_top_logprobs")
    if not isinstance(raw_topk, list) or len(raw_topk) != len(output_ids):
        return token_logprobs, None, None

    placeholder_lp = -math.log(max(topk, 1))
    topk_ids: list[list[int]] = []
    topk_lps: list[list[float]] = []
    for step_entries in raw_topk:
        step_ids: list[int] = []
        step_lps: list[float] = []
        if isinstance(step_entries, list):
            for entry in step_entries[:topk]:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    lp, tid = float(entry[0]), int(entry[1])
                elif isinstance(entry, dict):
                    lp = float(entry["logprob"])
                    tid = int(entry["token_id"])
                else:
                    continue
                step_ids.append(tid)
                step_lps.append(lp)
        while len(step_ids) < topk:
            step_ids.append(len(step_ids))
            step_lps.append(placeholder_lp)
        topk_ids.append(step_ids)
        topk_lps.append(step_lps)
    return token_logprobs, topk_ids, topk_lps


def session_turn(committed: str, full_text: str, has_session: bool) -> tuple[str, bool]:
    """Return the text to send and whether it continues an open session.

    A later turn sends only the suffix when the rendered prompt still starts
    with the text already committed to that session. Any rewrite of the prefix
    closes the session and resends the full prompt.
    """
    continuation = (
        has_session
        and bool(committed)
        and full_text.startswith(committed)
        and len(full_text) > len(committed)
    )
    if continuation:
        return full_text[len(committed) :], True
    return full_text, False


def _session_id_from_open(opened: Any) -> str:
    """SGLang 0.5.19 returns the id as a JSON string. Older builds return an object."""
    if isinstance(opened, str) and opened:
        return opened
    if isinstance(opened, dict):
        session_id = opened.get("session_id") or opened.get("id")
        if session_id:
            return str(session_id)
    raise RuntimeError("SGLang /open_session did not return a session_id")


def append_decode_log(path: Path, record: str) -> None:
    """Append one decoded turn. A lock keeps the eight rollout ranks intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.write(record)
            if not record.endswith("\n"):
                handle.write("\n")
            handle.flush()
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _png_b64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class QwenVLMSGLangAdapter:
    """One LIBERO environment, one SGLang session, one JSON macro action."""

    action_path = "/generate"
    # Required by SGLang 0.5.19 OpenSessionReqInput. The scheduler stores it
    # and does not enforce it; the value only has to cover one episode's text.
    _session_str_capacity = 1_000_000
    use_msgpack = False
    per_env = True

    def __init__(self, cfg: Any, rank: int):
        # Eight rollout workers share one SGLang server. A request id of only
        # the local env index collides (every rank sends qwen-env-0).
        self._rank = int(rank)
        model_cfg = cfg.rollout.model
        sglang_cfg = cfg.rollout.get("sglang", {})
        self.model_path = str(model_cfg.model_path)
        # The response budget is the episode budget. A turn may use
        # whatever remains; there is no separate per-turn cap.
        self._response_budget = int(
            sglang_cfg.get(
                "response_budget",
                getattr(model_cfg, "max_new_tokens", None)
                or cfg.rollout.sampling_params.max_new_tokens,
            )
        )
        self.max_new_tokens = self._response_budget
        self.reasoning_effort = str(
            sglang_cfg.get(
                "reasoning_effort",
                getattr(model_cfg, "reasoning_effort", DEFAULT_REASONING_EFFORT),
            )
        )
        chat_template_kwargs(self.reasoning_effort)
        embodiment = str(getattr(model_cfg, "embodiment", "libero"))
        self.schema = ROBODOJO_SCHEMA if embodiment == "robodojo" else LIBERO_SCHEMA
        self.cameras = cameras_for_schema(self.schema)
        self.action_width = 8 if embodiment == "libero" else 23
        self._timeout_s = float(sglang_cfg.get("http_timeout_s", 3600.0))
        self._log_decode_text = bool(sglang_cfg.get("log_decode_text", False))
        self._decode_log_path = self._decode_log_file(cfg)
        algorithm_cfg = getattr(cfg, "algorithm", None)
        pear_cfg = (
            algorithm_cfg.get("pear", {})
            if algorithm_cfg is not None and hasattr(algorithm_cfg, "get")
            else {}
        )
        score_centering = bool(pear_cfg.get("score_centering", False))
        default_topk = 32 if score_centering else 0
        self._topk_log_probs = int(sglang_cfg.get("topk_log_probs", default_topk))
        self._return_logprob = bool(
            sglang_cfg.get("return_logprob", self._topk_log_probs > 0)
        )
        self._keep_sessions_on_weight_sync = bool(
            sglang_cfg.get("keep_sessions_on_weight_sync", False)
        )
        env_cfg = getattr(cfg, "env", None)
        train_env_cfg = (
            env_cfg.get("train", None)
            if env_cfg is not None and hasattr(env_cfg, "get")
            else getattr(env_cfg, "train", None)
        )
        self._require_vlm_state = bool(
            sglang_cfg.get(
                "require_vlm_state",
                bool(
                    train_env_cfg.get("vlm_action_mode", False)
                    if train_env_cfg is not None and hasattr(train_env_cfg, "get")
                    else getattr(train_env_cfg, "vlm_action_mode", False)
                ),
            )
        )
        self._http = None
        self._processor = None
        self._sessions: dict[int, str] = {}
        self._committed: dict[int, str] = {}
        self._histories: dict[int, list[dict[str, Any]]] = {}
        self._response_ids: dict[int, list[list[int]]] = {}
        self._episodes: dict[int, IncrementalEpisodeTokens] = {}
        self._all_images: dict[int, list[Image.Image]] = {}
        self._idle_samples: dict[int, dict[str, torch.Tensor]] = {}
        self._weight_version = 0

    def _remaining_response_tokens(self, env_index: int) -> int:
        """Response tokens this episode may still use (generated + injected obs)."""
        episodes = getattr(self, "_episodes", None)
        if episodes and env_index in episodes:
            return episodes[env_index].remaining_budget()
        used = sum(len(turn) for turn in self._response_ids.get(env_index, []))
        return max(self._response_budget - used, 0)

    @staticmethod
    def _decode_log_file(cfg: Any) -> Path | None:
        runner = getattr(cfg, "runner", None)
        logger_cfg = getattr(runner, "logger", None) if runner is not None else None
        log_path = None
        if logger_cfg is not None:
            getter = getattr(logger_cfg, "get", None)
            log_path = (
                getter("log_path", None)
                if getter
                else getattr(logger_cfg, "log_path", None)
            )
        if not log_path:
            return None
        return Path(str(log_path)) / "sglang_decode.log"

    def bind_http_client(self, http_client: Any) -> None:
        self._http = http_client

    def _processor_or_load(self):
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self.model_path, trust_remote_code=True
            )
        return self._processor

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        if self._http is None:
            raise RuntimeError("Qwen VLM SGLang adapter has no HTTP client")
        return self._http.post(path, body, msgpack=False, timeout_s=self._timeout_s)

    def close_session(self, env_index: int, *, reset_history: bool = True) -> None:
        self._committed.pop(env_index, None)
        self._idle_samples.pop(env_index, None)
        if reset_history:
            self._histories.pop(env_index, None)
            self._response_ids.pop(env_index, None)
            getattr(self, "_episodes", {}).pop(env_index, None)
            getattr(self, "_all_images", {}).pop(env_index, None)
        session_id = self._sessions.pop(env_index, None)
        if session_id is not None and self._http is not None:
            self._post("/close_session", {"session_id": session_id})

    def _open_session(self) -> str:
        opened = self._post(
            "/open_session", {"capacity_of_str_len": self._session_str_capacity}
        )
        return _session_id_from_open(opened)

    def close_sessions(self) -> None:
        for env_index in list(self._sessions):
            self.close_session(env_index)

    def on_weights_synced(self) -> None:
        """Bump version and drop sticky sessions unless async KV retention is enabled."""
        self._weight_version += 1
        self._idle_samples.clear()
        if not getattr(self, "_keep_sessions_on_weight_sync", False):
            self.close_sessions()

    def env_is_finished(self, env_obs: dict[str, Any]) -> bool:
        """Return whether this environment's episode has ended or exhausted budget."""
        flag = env_obs.get("vlm_finished")
        if flag is not None:
            value = flag[0]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().reshape(-1)[0].item()
            if bool(value):
                return True
        env_index = int(env_obs.get("_env_index", 0))
        reset_mask = env_obs.get("reset_mask")
        if reset_mask is not None:
            reset_flag = reset_mask[0]
            if isinstance(reset_flag, torch.Tensor):
                reset_flag = reset_flag.detach().cpu().item()
            if bool(reset_flag):
                return False
        episodes = getattr(self, "_episodes", None)
        if episodes and env_index in episodes and episodes[env_index].budget_exhausted:
            return True
        return False

    def _minimal_idle_sample(self, env_index: int) -> dict[str, torch.Tensor] | None:
        """Return a tiny masked-out sample reusing known pixel/grid feature dims."""
        episodes = getattr(self, "_episodes", None)
        episode = episodes.get(env_index) if episodes else None
        if episode is None or not episode.initialized or not episode.pixel_chunks:
            return None
        pixel_shape = episode.pixel_chunks[0].shape[1:]
        pixel_dtype = episode.pixel_chunks[0].dtype
        grid_shape = episode.grid_chunks[0].shape[1:]
        sample: dict[str, torch.Tensor] = {
            "input_ids": torch.zeros(2, dtype=torch.long),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "pixel_values": torch.zeros((1, *pixel_shape), dtype=pixel_dtype),
            "image_grid_thw": torch.ones((1, *grid_shape), dtype=torch.long),
            "response_logprobs": torch.zeros(1, dtype=torch.float32),
            "response_index": torch.tensor([1], dtype=torch.long),
        }
        if episode.has_mm_token_types:
            sample["mm_token_type_ids"] = torch.zeros(2, dtype=torch.long)
        topk = int(getattr(self, "_topk_log_probs", 0))
        if topk > 0:
            sample["rollout_topk_ids"] = torch.arange(topk, dtype=torch.long).unsqueeze(
                0
            )
            sample["sampler_head_log_probs"] = torch.full(
                (1, topk), -math.log(max(topk, 1)), dtype=torch.float32
            )
        return sample

    def finished_result(
        self, env_obs: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return an invalid action without calling SGLang.

        The episode has already ended (or exhausted its response budget), so no
        server request is made. When an active episode exhausted its budget and
        the env has not yet acknowledged ``vlm_finished``, its accumulated
        sample is preserved once so the actor still trains on earlier valid
        turns; post-termination idle turns reuse a tiny placeholder sample that
        is masked out of the loss and dropped during episode compaction.
        """
        env_index = int(env_obs.get("_env_index", 0))
        env_already_done = False
        flag = env_obs.get("vlm_finished")
        if flag is not None:
            value = flag[0]
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().reshape(-1)[0].item()
            env_already_done = bool(value)
        episodes = getattr(self, "_episodes", None)
        episode = episodes.get(env_index) if episodes else None
        if (
            not env_already_done
            and episode is not None
            and episode.initialized
            and episode.budget_exhausted
        ):
            sample = episode.build_sample()
        else:
            sample = self._idle_samples.get(env_index)
            if sample is None:
                sample = self._minimal_idle_sample(env_index)
                if sample is None:
                    sample = self._idle_sample(env_obs)
                self._idle_samples[env_index] = sample
        action = torch.zeros(1, 1, self.action_width, dtype=torch.float32)
        return action, {"sample": sample}

    def _idle_sample(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        instruction = env_obs["task_descriptions"][0]
        state = _state_from_observation(
            env_obs,
            0,
            require_vlm_state=getattr(self, "_require_vlm_state", False),
        )
        cameras = getattr(self, "cameras", cameras_for_schema(self.schema))
        images = _turn_images_from_observation(env_obs, 0, cameras)
        messages = [
            {"role": "system", "content": system_prompt(instruction, self.schema)},
            {
                "role": "user",
                "content": build_user_content(state, images, cameras),
            },
        ]
        processor = self._processor_or_load()
        prompt = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **chat_template_kwargs(self.reasoning_effort),
        )
        inputs = processor(
            text=[prompt], images=images, return_tensors="pt", padding=True
        )
        sequence = inputs["input_ids"][0].detach().cpu()
        pixels = inputs["pixel_values"].detach().cpu()
        grids = inputs["image_grid_thw"].detach().cpu()
        if pixels.ndim == 3 and pixels.shape[0] == 1:
            pixels = pixels[0]
        if grids.ndim == 3 and grids.shape[0] == 1:
            grids = grids[0]
        position = max(int(sequence.shape[0]) - 1, 1)
        sample = {
            "input_ids": sequence,
            "attention_mask": torch.ones_like(sequence, dtype=torch.long),
            "pixel_values": pixels,
            "image_grid_thw": grids,
            "response_logprobs": torch.zeros(1, dtype=torch.float32),
            "response_index": torch.tensor([position], dtype=torch.long),
        }
        token_types = extend_mm_token_type_ids(inputs, int(sequence.shape[0]))
        if token_types is not None:
            sample["mm_token_type_ids"] = token_types
        topk = int(getattr(self, "_topk_log_probs", 0))
        if topk > 0:
            sample["rollout_topk_ids"] = torch.arange(topk, dtype=torch.long).unsqueeze(
                0
            )
            sample["sampler_head_log_probs"] = torch.full(
                (1, topk), -math.log(max(topk, 1)), dtype=torch.float32
            )
        return sample

    def build_request(
        self, env_obs: dict[str, Any], mode: str = "train"
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del mode
        env_index = int(env_obs.get("_env_index", 0))
        reset_mask = env_obs.get("reset_mask")
        if reset_mask is not None:
            flag = reset_mask[0]
            if isinstance(flag, torch.Tensor):
                flag = flag.detach().cpu().item()
            if bool(flag):
                self.close_session(env_index)
        instruction = env_obs["task_descriptions"][0]
        state = _state_from_observation(
            env_obs,
            0,
            require_vlm_state=getattr(self, "_require_vlm_state", False),
        )
        cameras = getattr(self, "cameras", cameras_for_schema(self.schema))
        turn_images = _turn_images_from_observation(env_obs, 0, cameras)
        user = {
            "role": "user",
            "content": build_user_content(state, turn_images, cameras),
        }
        processor = self._processor_or_load()
        template_kwargs = chat_template_kwargs(self.reasoning_effort)
        episodes = self.__dict__.setdefault("_episodes", {})
        all_images_map = self.__dict__.setdefault("_all_images", {})
        episode = episodes.get(env_index)

        if episode is None or not episode.initialized:
            episode = IncrementalEpisodeTokens(response_budget=self._response_budget)
            messages = [
                {"role": "system", "content": system_prompt(instruction, self.schema)},
                user,
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            inputs = processor(
                text=[prompt], images=turn_images, return_tensors="pt", padding=True
            )
            episode.start_episode(prompt_text=prompt, inputs=inputs)
            episodes[env_index] = episode
            all_images_map[env_index] = list(turn_images)
            delta_text = prompt
        else:
            delta_full, delta_no_eos = render_incremental_user_delta(
                processor, user, template_kwargs
            )
            tokenizer = getattr(processor, "tokenizer", processor)
            eos_id = getattr(tokenizer, "eos_token_id", None)
            use_no_eos = (
                eos_id is not None
                and bool(episode.input_ids)
                and episode.input_ids[-1] == eos_id
            )
            delta_text = delta_no_eos if use_no_eos else delta_full
            delta_inputs = processor(
                text=[delta_text],
                images=turn_images,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
            fits = episode.append_user_observation(
                delta_text=delta_text, delta_inputs=delta_inputs
            )
            if not fits:
                # Signal caller not to hit SGLang: episode ended cleanly on BUDGET.
                return {}, {
                    "env_index": env_index,
                    "budget_exhausted": True,
                }
            all_images_map.setdefault(env_index, []).extend(turn_images)

        remaining = episode.remaining_budget()
        if remaining <= 0:
            episode.budget_exhausted = True
            return {}, {
                "env_index": env_index,
                "budget_exhausted": True,
            }

        committed = self._committed.get(env_index, "")
        full_prompt = episode.rendered_text
        text, continuation = session_turn(
            committed, full_prompt, env_index in self._sessions
        )
        if not continuation:
            if env_index in self._sessions:
                self.close_session(env_index, reset_history=False)
            self._sessions[env_index] = self._open_session()
            image_payload = all_images_map.get(env_index, turn_images)
        else:
            image_payload = turn_images

        topk = int(getattr(self, "_topk_log_probs", 0))
        return_logprob = bool(getattr(self, "_return_logprob", False)) or topk > 0
        payload: dict[str, Any] = {
            "text": text,
            # 0.5.19 SessionParams.id. The prompt is already rendered locally,
            # and GenerateReqInput has no chat_template_kwargs field.
            "session_params": {"id": self._sessions[env_index]},
            "sampling_params": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "max_new_tokens": remaining,
            },
            "rid": f"qwen-r{self._rank}-env-{env_index}-v{self._weight_version}",
            "image_data": [_png_b64(image) for image in image_payload],
            "return_logprob": return_logprob,
        }
        if topk > 0:
            payload["top_logprobs_num"] = topk
        return payload, {
            "env_index": env_index,
            "prompt": full_prompt,
            "user": user,
            "continuation": continuation,
            "budget_exhausted": False,
        }

    def _log_turn(self, env_index: int, token_ids: list[int], server_text: str) -> None:
        """Decode this turn's output tokens and append them to the decode log."""
        if not self._log_decode_text or self._decode_log_path is None:
            return
        turn = len(self._response_ids.get(env_index, [])) + 1
        try:
            processor = self._processor_or_load()
            tokenizer = getattr(processor, "tokenizer", processor)
            decoded = tokenizer.decode(token_ids, skip_special_tokens=False)
        except Exception as exc:
            decoded = f"<decode failed: {exc}>\n{server_text}"
        record = (
            f"===== rank={self._rank} env={env_index} turn={turn} "
            f"tokens={len(token_ids)} weight_v={self._weight_version} =====\n"
            f"{decoded}\n\n"
        )
        append_decode_log(self._decode_log_path, record)

    def parse_response(
        self, response: dict[str, Any], state: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        env_index = int(state["env_index"])
        episodes = self.__dict__.setdefault("_episodes", {})
        episode = episodes.get(env_index)
        if state.get("budget_exhausted", False):
            sample = (
                episode.build_sample()
                if episode is not None and episode.initialized
                else self._idle_samples[env_index]
            )
            action_tensor = torch.zeros(1, 1, self.action_width, dtype=torch.float32)
            return action_tensor, {"sample": sample}

        if "text" not in response or "output_ids" not in response:
            raise RuntimeError("SGLang /generate did not return text and output_ids")
        text = str(response["text"])
        output_ids = [int(t) for t in response["output_ids"]]
        self._log_turn(env_index, output_ids, text)
        self._committed[env_index] = state["prompt"] + text

        topk = int(getattr(self, "_topk_log_probs", 0))
        token_lps, topk_ids, topk_lps = extract_sglang_topk_logprobs(
            response, output_ids, topk
        )
        if episode is not None and episode.initialized:
            episode.append_assistant_response(
                response_ids=output_ids,
                response_text=text,
                token_logprobs=token_lps,
                topk_ids=topk_ids,
                topk_logprobs=topk_lps,
            )
            sample = episode.build_sample()
            exhausted = episode.budget_exhausted
        else:
            # Fallback for callers bypassing build_request.
            inputs = state["inputs"]
            prompt_length = int(inputs["input_ids"].shape[-1])
            response_tensor = torch.tensor(output_ids, dtype=torch.long)
            sequence = torch.cat(
                [inputs["input_ids"][0].detach().cpu(), response_tensor]
            )
            positions = list(range(prompt_length, int(sequence.shape[0])))
            pixels = inputs["pixel_values"].detach().cpu()
            grids = inputs["image_grid_thw"].detach().cpu()
            if pixels.ndim == 3 and pixels.shape[0] == 1:
                pixels = pixels[0]
            if grids.ndim == 3 and grids.shape[0] == 1:
                grids = grids[0]
            sample = {
                "input_ids": sequence,
                "attention_mask": torch.ones_like(sequence, dtype=torch.long),
                "pixel_values": pixels,
                "image_grid_thw": grids,
                "response_logprobs": (
                    torch.tensor(token_lps, dtype=torch.float32)
                    if token_lps is not None
                    else torch.zeros(len(positions), dtype=torch.float32)
                ),
                "response_index": torch.tensor(positions, dtype=torch.long),
            }
            token_types = extend_mm_token_type_ids(inputs, int(sequence.shape[0]))
            if token_types is not None:
                sample["mm_token_type_ids"] = token_types
            exhausted = False

        if exhausted:
            vector = action_to_vector(None, self.schema)
        else:
            try:
                action = parse_action(text, self.schema)
                vector = action_to_vector(action, self.schema)
            except InvalidAction:
                vector = action_to_vector(None, self.schema)
        history = self._histories.setdefault(env_index, [])
        history.extend([state["user"], {"role": "assistant", "content": text}])
        turns = self._response_ids.setdefault(env_index, [])
        turns.append(output_ids)
        action_tensor = torch.tensor(vector, dtype=torch.float32).reshape(
            1, 1, self.action_width
        )
        return action_tensor, {"sample": sample}

    def merge_env_results(
        self, outputs: list[tuple[torch.Tensor, dict[str, Any]]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        actions = torch.cat([action for action, _ in outputs], dim=0)
        forward_inputs = QwenVLMPolicy._collate(
            self, [info["sample"] for _, info in outputs]
        )
        return actions, {
            "prev_logprobs": forward_inputs["response_logprobs"],
            "prev_values": None,
            "forward_inputs": forward_inputs,
        }
