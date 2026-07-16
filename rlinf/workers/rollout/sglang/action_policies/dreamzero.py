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

"""DreamZero embodied action policy backed by an SGLang action server.

This module is the SGLang counterpart of
``rlinf.models.embodiment.dreamzero.dreamzero_policy.DreamZeroPolicy`` for eval
rollout.  It intentionally does not import or construct the HF policy/model:
the large DreamZero network lives in a separately spawned ``sglang serve``
process.  The local policy keeps only the lightweight parts required on the
rollout worker:

1. convert RLinf env observations to DreamZero modality keys;
2. run DreamZero dataset transforms and metadata-based normalization;
3. call the SGLang ``/v1/actions/generations`` action endpoint;
4. invert normalized actions back to environment-scale action chunks.

For a Libero batch of size ``B`` and action horizon ``H``, the main shape flow is
roughly:

``env_obs`` -> converted obs -> normalized input -> server action
``[B, H, max_action_dim]`` -> unnormalized env action ``[B, H, action_dim]``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib import error as urllib_error
from urllib import request as urllib_request

import numpy as np
import torch
from omegaconf import DictConfig, open_dict
from tianshou.data import Batch

from rlinf.data.datasets.dreamzero.data_transforms import (
    build_dreamzero_composed_transform,
    collect_dreamzero_dataset_keys,
    convert_rollout_env_obs,
    format_training_prompt,
    load_dreamzero_dataset_metadata,
    normalize_instruction_text,
    rollout_obs_layout_for_embodiment,
)
from rlinf.workers.rollout.sglang.action_policies.base import EmbodiedActionPolicy
from rlinf.workers.rollout.sglang.action_policies.registry import (
    register_action_policy,
)

_RLINF_POLICY_CONTEXT_KEYS = ("_rlinf_stage_id",)

_NO_PROXY_HTTP_OPENER = urllib_request.build_opener(urllib_request.ProxyHandler({}))


def _urlopen_no_proxy(request_or_url, *, timeout: float):
    return _NO_PROXY_HTTP_OPENER.open(request_or_url, timeout=timeout)


@dataclass
class DreamZeroActionRequest:
    """Request contract used by the DreamZero SGLang action endpoint.

    ``normalized_input`` is the output of the local DreamZero data transform and
    is passed to the server as ``parameters.action_input``. ``session_ids`` name
    the logical env slots for the server-side video/text cache. ``reset_mask`` is
    the client-side cache invalidation handle for those logical sessions: setting
    a row to ``True`` asks the server to drop cached state for that session before
    processing the request.  Model/cache lifecycle resets caused by window
    limits, language changes, etc. are still decided by the server.
    """

    normalized_input: dict[str, Any]
    session_ids: list[str]
    reset_mask: list[bool]
    prompt_cache_keys: list[str]
    negative_prompt_cache_keys: list[str]
    seed: int


@dataclass
class DreamZeroActionResult:
    """Normalized action tensor returned by the SGLang action endpoint."""

    normalized_action: Any


class HttpDreamZeroActionClient:
    """Small synchronous client for the SGLang action API."""

    def __init__(
        self,
        server_url: str,
        *,
        timeout_s: float,
        max_retries: int,
        retry_backoff_s: float,
        payload_format: str,
        model: str | None = None,
    ):
        self._server_url = server_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max(0, int(max_retries))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._payload_format = str(payload_format).lower()
        if self._payload_format not in ("json", "msgpack"):
            raise ValueError(
                "DreamZero action HTTP payload_format must be 'json' or 'msgpack', "
                f"got {payload_format!r}."
            )
        self._model = model

    def infer(self, request: DreamZeroActionRequest) -> DreamZeroActionResult:
        """Send one batched action request and parse the normalized action.

        The server response is an action-generation envelope whose values are
        still in DreamZero normalized/padded action space.  Environment scaling
        is handled locally by ``_DreamZeroActionAdapter.unapply``.
        """

        payload = {
            "model": self._model,
            "parameters": {
                "action_input": request.normalized_input,
                "session_ids": request.session_ids,
                "reset_mask": request.reset_mask,
                "prompts": request.prompt_cache_keys,
                "negative_prompts": request.negative_prompt_cache_keys,
                "seed": request.seed,
            },
            "runtime": {
                "response_format": "envelope",
                "output_format": "numpy",
            },
        }
        body, content_type = self._encode_payload(payload)
        response_body = self._post_action_request(body, content_type)
        try:
            action_data = response_body["data"][0]["action"]["values"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                "DreamZero SGLang action HTTP response missing "
                f"data[0].action.values: {response_body}"
            ) from exc
        return DreamZeroActionResult(
            normalized_action=np.asarray(action_data, dtype=np.float32)
        )

    def _encode_payload(self, payload: dict[str, Any]) -> tuple[bytes, str]:
        if self._payload_format == "msgpack":
            from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
                pack_msgpack,
            )

            return (
                pack_msgpack(self._to_msgpackable(payload)),
                "application/msgpack",
            )
        return json.dumps(self._to_jsonable(payload)).encode(
            "utf-8"
        ), "application/json"

    def _post_action_request(self, body: bytes, content_type: str) -> dict[str, Any]:
        retry_statuses = {500, 502, 503, 504}
        last_error: Exception | None = None
        url = f"{self._server_url}/v1/actions/generations"
        for attempt in range(self._max_retries + 1):
            http_request = urllib_request.Request(
                url,
                data=body,
                headers={"Content-Type": content_type, "Accept": content_type},
                method="POST",
            )
            try:
                with _urlopen_no_proxy(
                    http_request, timeout=self._timeout_s
                ) as response:
                    response_bytes = response.read()
                    if "msgpack" in response.headers.get("content-type", "").lower():
                        from sglang.multimodal_gen.runtime.entrypoints.vla.protocol import (
                            unpack_msgpack,
                        )

                        return unpack_msgpack(response_bytes)
                    return json.loads(response_bytes.decode("utf-8"))
            except urllib_error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                if exc.code in retry_statuses and attempt < self._max_retries:
                    last_error = RuntimeError(
                        f"status={exc.code}, body={detail[:1000]}"
                    )
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(
                    "DreamZero SGLang action HTTP request failed: "
                    f"status={exc.code}, body_bytes={len(body)}, body={detail}"
                ) from exc
            except (
                ConnectionResetError,
                TimeoutError,
                OSError,
                urllib_error.URLError,
            ) as exc:
                if attempt < self._max_retries:
                    last_error = exc
                    self._sleep_before_retry(attempt)
                    continue
                raise RuntimeError(
                    "DreamZero SGLang action HTTP request failed after retries: "
                    f"attempts={self._max_retries + 1}, body_bytes={len(body)}, "
                    f"last_error={last_error or exc}"
                ) from exc
        raise RuntimeError(
            "DreamZero SGLang action HTTP request failed after retries: "
            f"attempts={self._max_retries + 1}, body_bytes={len(body)}, "
            f"last_error={last_error}"
        )

    def _sleep_before_retry(self, attempt: int) -> None:
        if self._retry_backoff_s > 0:
            time.sleep(self._retry_backoff_s * float(attempt + 1))

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if isinstance(value, Batch):
            value = value.__getstate__()
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, Mapping):
            return {
                str(key): HttpDreamZeroActionClient._to_jsonable(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [HttpDreamZeroActionClient._to_jsonable(item) for item in value]
        return value

    @staticmethod
    def _to_msgpackable(value: Any) -> Any:
        if isinstance(value, Batch):
            value = value.__getstate__()
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        if isinstance(value, Mapping):
            return {
                str(key): HttpDreamZeroActionClient._to_msgpackable(item)
                for key, item in value.items()
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [HttpDreamZeroActionClient._to_msgpackable(item) for item in value]
        return value


class _DreamZeroActionAdapter:
    """Reuse DreamZero observation/action transforms without the HF model.

    This mirrors the preprocessing and postprocessing pieces of
    ``DreamZeroPolicy``:

    - ``observation_convert`` maps RLinf rollout observations to DreamZero
      modality keys such as ``video.*``, ``state.*`` and ``annotation.task``.
    - ``normalize_obs`` applies the composed dataset transform and metadata
      normalization used by the HF policy.
    - ``unapply`` inverts normalized server actions from padded model space back
      to per-modality environment-scale action tensors.
    - ``actions_from_unapply`` concatenates those tensors into the action array
      expected by the env worker.
    """

    def __init__(self, cfg: DictConfig):
        tokenizer_path = cfg.get("tokenizer_path", "google/umt5-xxl")
        self.embodiment_tag = str(cfg.embodiment_tag)
        self._rollout_obs_layout = rollout_obs_layout_for_embodiment(
            self.embodiment_tag
        )
        self.data_transforms = build_dreamzero_composed_transform(cfg, tokenizer_path)
        self.data_transforms.set_metadata(load_dreamzero_dataset_metadata(cfg))
        self.data_transforms.eval()
        _, _, action_keys, _ = collect_dreamzero_dataset_keys(
            self.data_transforms, self.embodiment_tag
        )
        self._action_keys = tuple(action_keys)
        self._dream_transform = self.data_transforms.transforms[-1]
        self._relative_action = bool(cfg.get("relative_action", False))
        self._relative_action_per_horizon = bool(
            cfg.get("relative_action_per_horizon", False)
        )
        self._relative_action_keys = list(cfg.get("relative_action_keys") or [])

    def observation_convert(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Map RLinf env observation keys to DreamZero dataset modality keys.

        Example input keys are ``main_images``, ``wrist_images``, ``states`` and
        ``task_descriptions``.  The converted observation is the same structure
        consumed by the HF policy's DreamZero data transforms.
        """

        converted = convert_rollout_env_obs(self.embodiment_tag, env_obs)
        tasks = converted.get("annotation.task")
        if isinstance(tasks, list) and all(isinstance(item, str) for item in tasks):
            converted["annotation.task"] = np.asarray(tasks, dtype=object)
        return converted

    def normalize_obs(self, obs: dict[str, Any]) -> dict[str, Any]:
        """Apply DreamZero dataset transforms to obtain server model inputs.

        The output contains normalized tensors such as images/video, state,
        tokenized text, attention masks and ``embodiment_id``.  Prompt tokens are
        aligned after the transform so the rollout path uses the training prompt
        format expected by the server-side text cache.
        """

        normalized_input = self.data_transforms(obs)
        if isinstance(normalized_input, Batch):
            normalized_input = normalized_input.__getstate__()
        normalized_input = dict(normalized_input)
        self._align_rollout_prompt_tokens(obs, normalized_input)
        return normalized_input

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if value.ndim == 0:
                return [value.item()]
            return value.flatten().tolist()
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return [value.item()]
            return value.reshape(-1).tolist()
        if isinstance(value, list):
            return value
        if isinstance(value, tuple):
            return list(value)
        return [value]

    def _align_rollout_prompt_tokens(
        self, obs: dict[str, Any], normalized_input: dict[str, Any]
    ) -> None:
        """Replace rollout text tokens with DreamZero training-format tokens."""

        tasks = obs.get("annotation.task")
        if tasks is None or "text" not in normalized_input:
            return
        embodiment_ids = self._as_list(normalized_input["embodiment_id"])
        task_list = self._as_list(tasks)
        if len(task_list) == 1 and len(embodiment_ids) > 1:
            task_list = task_list * len(embodiment_ids)
        texts = [
            format_training_prompt(
                normalize_instruction_text(task),
                int(embodiment_id),
                self._dream_transform.embodiment_tag_mapping,
            )
            for task, embodiment_id in zip(task_list, embodiment_ids, strict=False)
        ]
        ids, mask = self._dream_transform.tokenizer(
            texts,
            return_mask=True,
            add_special_tokens=True,
        )
        normalized_input["text"] = ids
        normalized_input["text_attention_mask"] = mask
        normalized_input["_dreamzero_prompt_texts"] = texts

    def unapply(self, batch: Batch, obs: dict[str, Any] | None = None, **kwargs):
        """Invert normalized actions back to environment-scale action tensors.

        ``batch.normalized_action`` comes from the SGLang server and has the
        padded DreamZero model width, e.g. ``[B, H, max_action_dim]``.  The
        reverse transform slices and unnormalizes it according to metadata,
        producing keys such as ``action.actions`` with env width, e.g.
        ``[B, H, 7]`` for Libero.
        """

        unnormalized_action = self.data_transforms.unapply(
            {"action": batch.normalized_action.cpu()}
        )
        if (
            (self._relative_action or self._relative_action_per_horizon)
            and self._relative_action_keys
            and obs is not None
        ):
            for key in self._relative_action_keys:
                action_key = f"action.{key}"
                state_key = f"state.{key}"
                if action_key not in unnormalized_action:
                    continue
                last_state = obs.get(state_key)
                if last_state is None:
                    for obs_key, obs_value in obs.items():
                        if "state" in obs_key and key in obs_key:
                            last_state = obs_value
                            break
                if last_state is None and "state" in obs:
                    state_data = obs["state"]
                    action_dim = unnormalized_action[action_key].shape[-1]
                    state_dim = (
                        state_data.shape[-1] if hasattr(state_data, "shape") else None
                    )
                    if state_dim == action_dim:
                        last_state = state_data
                if last_state is None:
                    continue
                if torch.is_tensor(last_state):
                    last_state = last_state.cpu().numpy()
                if len(last_state.shape) >= 2:
                    last_state = last_state[..., -1, :]
                if len(unnormalized_action[action_key].shape) > len(last_state.shape):
                    last_state = np.expand_dims(last_state, axis=-2)
                unnormalized_action[action_key] = (
                    unnormalized_action[action_key] + last_state
                )
        batch.act = unnormalized_action
        return batch

    def actions_from_unapply(self, act_dict: dict[str, Any]) -> np.ndarray:
        """Concatenate unnormalized action modalities in dataset action order."""

        parts: list[np.ndarray] = []
        for key in self._action_keys:
            if key not in act_dict:
                raise KeyError(
                    f"Unnormalized action missing {key!r}; "
                    f"available keys: {sorted(act_dict)}."
                )
            value = act_dict[key]
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            parts.append(np.asarray(value))
        actions = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=-1)
        if self._rollout_obs_layout.binarize_gripper:
            actions[..., -1] = np.where(actions[..., -1] > 0, 1.0, -1.0).astype(
                actions.dtype
            )
        return actions


@register_action_policy("dreamzero")
class DreamZeroActionPolicy(EmbodiedActionPolicy):
    """DreamZero eval policy for the SGLang embodied rollout worker.

    ``SGLangEmbodiedWorker`` owns the server subprocess and channel loop.  This
    policy owns only DreamZero-specific behavior: server launch arguments,
    observation normalization, action HTTP requests and action unnormalization.
    """

    @staticmethod
    def build_sglang_serve_args(
        *,
        model_path: str,
        sglang_cfg: Any,
        model_cfg: Any,
        tmpdir: str | None,
        rank: int,
        eval_batch_size: int,
    ) -> list[str]:
        """Build DreamZero-specific ``sglang serve`` command-line arguments.

        The returned flags are consumed by the server subprocess, not by the
        RLinf worker.  They select the native backend, force the registered
        ``DreamZeroPipeline`` and pass a JSON pipeline config generated from the
        RLinf rollout/model config.
        """

        sp_degree = int(
            getattr(sglang_cfg, "sp_degree", getattr(sglang_cfg, "sp_size", 1)) or 1
        )
        cfg_parallel_degree = int(getattr(sglang_cfg, "cfg_parallel_degree", 1) or 1)
        pipeline_config_path = DreamZeroActionPolicy._write_pipeline_config(
            sglang_cfg=sglang_cfg,
            model_cfg=model_cfg,
            tmpdir=tmpdir,
            rank=rank,
            eval_batch_size=eval_batch_size,
        )
        args = [
            "--backend",
            "sglang",
            "--pipeline",
            str(getattr(sglang_cfg, "pipeline_class_name", "DreamZeroPipeline")),
            "--pipeline-config-path",
            pipeline_config_path,
            "--sp-degree",
            str(sp_degree),
            "--cfg-parallel-size",
            str(cfg_parallel_degree),
        ]
        if v := getattr(sglang_cfg, "attention_backend", None):
            args += ["--attention-backend", str(v)]
        if (v := getattr(sglang_cfg, "scheduler_port", None)) is not None:
            args += [
                "--scheduler-port",
                str(int(v) + rank * int(getattr(sglang_cfg, "port_stride", 100))),
            ]
        if isinstance(getattr(sglang_cfg, "dit_cpu_offload", None), bool):
            args += [
                "--dit-cpu-offload",
                "true" if getattr(sglang_cfg, "dit_cpu_offload") else "false",
            ]
        for component in (
            "dreamzero_dit",
            "dreamzero_vae",
            "dreamzero_text_encoder",
            "dreamzero_image_encoder",
        ):
            args += [f"--{component.replace('_', '-')}-path", model_path]
        return args

    @staticmethod
    def _write_pipeline_config(
        *,
        sglang_cfg: Any,
        model_cfg: Any,
        tmpdir: str | None,
        rank: int,
        eval_batch_size: int,
    ) -> str:
        """Write the SGLang ``DreamZeroPipelineConfig`` override file.

        ``PipelineConfig.from_kwargs`` loads this file from
        ``--pipeline-config-path`` inside the server process.  It is the bridge
        from RLinf/Hydra fields (action horizon, image size, CFG scale,
        sequence-parallel size, compile flag) to ``server_args.pipeline_config``
        used by the DreamZero server stages.
        """

        if tmpdir is None:
            raise ValueError("DreamZero sglang serve args require a temp directory")
        action_head_cfg = getattr(
            getattr(model_cfg, "action_head_cfg", None), "config", {}
        )
        if action_head_cfg is None:
            action_head_cfg = {}
        cfg = {
            "dreamzero_compile_components": bool(
                getattr(sglang_cfg, "compile_components", True)
            ),
            "dreamzero_sequence_parallel_size": int(
                getattr(sglang_cfg, "sp_degree", getattr(sglang_cfg, "sp_size", 1)) or 1
            ),
            "dreamzero_max_sessions": int(
                getattr(sglang_cfg, "max_sessions", eval_batch_size) or eval_batch_size
            ),
            "cfg_scale": float(getattr(sglang_cfg, "cfg_scale", 5.0)),
            "action_horizon": int(getattr(model_cfg, "action_horizon")),
            "num_inference_steps": int(getattr(sglang_cfg, "num_inference_steps", 16)),
            "dynamic_cache_schedule": bool(
                getattr(sglang_cfg, "dynamic_cache_schedule", False)
            ),
            "num_frames": int(getattr(action_head_cfg, "num_frames", 33)),
            "synthetic_height": int(getattr(model_cfg, "target_video_height")),
            "synthetic_width": int(getattr(model_cfg, "target_video_width")),
            "tile_size_height": int(getattr(action_head_cfg, "tile_size_height", 34)),
            "tile_size_width": int(getattr(action_head_cfg, "tile_size_width", 34)),
            "tile_stride_height": int(
                getattr(action_head_cfg, "tile_stride_height", 18)
            ),
            "tile_stride_width": int(getattr(action_head_cfg, "tile_stride_width", 16)),
            "tiled": bool(getattr(action_head_cfg, "tiled", False)),
        }
        if "dit_step_mask" in sglang_cfg:
            mask = sglang_cfg.get("dit_step_mask")
            cfg["dit_step_mask"] = None if mask is None else list(mask)
        path = os.path.join(tmpdir, f"dreamzero_pipeline_rank{rank}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
        return path

    def __init__(self, cfg: Any, server_url: str | None, rank: int):
        super().__init__(cfg, server_url, rank)
        if server_url is None:
            raise ValueError("DreamZeroActionPolicy requires a local sglang server URL")
        rollout_model_config = self._rollout_model_config()
        self.action_adapter = _DreamZeroActionAdapter(rollout_model_config)
        sglang_cfg = self.cfg_rollout.get("sglang", {})
        self.action_client = HttpDreamZeroActionClient(
            server_url,
            timeout_s=float(
                sglang_cfg.get("http_timeout_s", sglang_cfg.get("timeout_s", 120.0))
            ),
            max_retries=int(sglang_cfg.get("http_max_retries", 5)),
            retry_backoff_s=float(sglang_cfg.get("http_retry_backoff_s", 1.0)),
            payload_format=sglang_cfg.get("http_payload_format", "json"),
            model=sglang_cfg.get("model", str(rollout_model_config.model_path)),
        )
        self._eval_predict_calls = 0
        self._debug_sessions = bool(sglang_cfg.get("debug_sessions", False))
        self._debug_batch_print = bool(sglang_cfg.get("debug_batch", False)) or bool(
            int(os.environ.get("DREAMZERO_SGLANG_WORKER_DEBUG_BATCH", "0") or 0)
        )
        self._seed = int(sglang_cfg.get("seed", 1140))

    def _rollout_model_config(self) -> DictConfig:
        model_cfg = self.model_cfg.copy()
        with open_dict(model_cfg):
            model_cfg.precision = self.cfg_rollout.model.precision
            model_cfg.model_path = self.cfg_rollout.model.model_path
        return model_cfg

    def infer(
        self, env_obs: dict, mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict]:
        """Run one eval rollout action request.

        The returned action tensor has shape ``[B, num_action_chunks,
        action_dim]``.  The info dict mirrors the HF embodied policy interface
        with placeholder logprobs/values, because this SGLang eval path does not
        compute policy likelihoods.
        """

        if mode != "eval":
            raise NotImplementedError(
                "DreamZero sglang action policy currently supports eval only."
            )
        rollout_env_obs, policy_context = self._split_policy_context(env_obs)
        converted_obs = self.action_adapter.observation_convert(rollout_env_obs)
        normalized_input = self.action_adapter.normalize_obs(converted_obs)
        actions = self._run_batch_request(
            normalized_input, converted_obs, policy_context
        )
        if self._debug_sessions:
            self.logger.info(
                "DreamZero sglang session rank=%s calls=%s batch_size=%s",
                self.rank,
                self._eval_predict_calls,
                self._infer_batch_size(normalized_input),
            )
        actions_t = torch.as_tensor(actions, dtype=torch.float32)
        flat = actions_t.reshape(actions_t.shape[0], -1)
        result = {
            "prev_logprobs": torch.zeros_like(flat, dtype=torch.float32),
            "prev_values": torch.zeros((flat.shape[0], 1), dtype=torch.float32),
            "forward_inputs": {"action": flat.cpu()},
        }
        return actions_t, result

    def _split_policy_context(self, env_obs: Any) -> tuple[Any, dict[str, Any]]:
        """Separate RLinf routing metadata from the env observation payload."""

        if not isinstance(env_obs, Mapping):
            return env_obs, {}
        context = {
            key: env_obs[key] for key in _RLINF_POLICY_CONTEXT_KEYS if key in env_obs
        }
        if not context:
            return env_obs, {}
        cleaned = {
            key: value
            for key, value in env_obs.items()
            if key not in _RLINF_POLICY_CONTEXT_KEYS
        }
        return cleaned, context

    def _session_ids_for_batch(
        self, batch_size: int, context: dict[str, Any]
    ) -> list[str]:
        """Create stable logical session ids for server-side DreamZero caches."""

        stage_id = self._stage_id_from_context(context)
        session_ids = [
            f"rlinf-eval-r{self.rank}-stage{stage_id}-slot{slot_id}"
            for slot_id in range(batch_size)
        ]
        return session_ids

    @staticmethod
    def _stage_id_from_context(context: dict[str, Any]) -> int:
        stage_id = context.get("_rlinf_stage_id", 0)
        if torch.is_tensor(stage_id):
            stage_id = stage_id.item()
        if isinstance(stage_id, np.ndarray):
            stage_id = stage_id.item()
        return int(stage_id)

    def _build_action_request(
        self,
        normalized_input: dict[str, Any],
        *,
        session_ids: list[str],
        reset_mask: list[bool],
    ) -> DreamZeroActionRequest:
        """Assemble the HTTP action request from normalized model inputs."""

        request_input = dict(normalized_input)
        prompt_texts = request_input.pop("_dreamzero_prompt_texts", None)
        batch_size = len(session_ids)
        return DreamZeroActionRequest(
            normalized_input=request_input,
            session_ids=session_ids,
            reset_mask=reset_mask,
            prompt_cache_keys=self._prompt_cache_keys(
                request_input,
                "text",
                batch_size,
                prompt_texts=prompt_texts,
            ),
            negative_prompt_cache_keys=self._prompt_cache_keys(
                request_input,
                "text_negative",
                batch_size,
            ),
            seed=self._seed,
        )

    @staticmethod
    def _prompt_cache_keys(
        normalized_input: dict[str, Any],
        key: str,
        batch_size: int,
        *,
        prompt_texts: Any = None,
    ) -> list[str]:
        """Build per-row prompt cache keys for server-side text embeddings.

        Training-format prompt text is preferred because it is stable and human
        readable.  If only token ids are available, a SHA1 over each token row is
        used.  Missing prompt branches are explicitly marked so the server sees a
        deterministic cache key for every batch row.
        """

        if prompt_texts is not None:
            values = (
                list(prompt_texts)
                if isinstance(prompt_texts, Sequence)
                and not isinstance(prompt_texts, (str, bytes))
                else [str(prompt_texts)]
            )
            if len(values) == 1 and batch_size > 1:
                values = values * batch_size
            if len(values) == batch_size:
                return [str(value) for value in values]
        value = normalized_input.get(key)
        if torch.is_tensor(value):
            rows = value.detach().cpu()
            if rows.ndim == 0:
                rows = rows.reshape(1)
            if rows.shape[0] == 1 and batch_size > 1:
                rows = rows.repeat(batch_size, *([1] * (rows.ndim - 1)))
            if rows.shape[0] == batch_size:
                return [
                    f"{key}:tokens:{hashlib.sha1(row.contiguous().numpy().tobytes()).hexdigest()}"
                    for row in rows
                ]
        return [f"{key}:missing"] * batch_size

    def _extract_action_output(
        self,
        normalized_action: np.ndarray | torch.Tensor,
        converted_obs: dict[str, Any],
    ) -> np.ndarray:
        """Convert server normalized action output into env action chunks."""

        normalized_action_t = torch.as_tensor(normalized_action, dtype=torch.float32)
        action_batch = self.action_adapter.unapply(
            Batch(normalized_action=normalized_action_t.detach().cpu().float()),
            obs=converted_obs,
        )
        actions = self.action_adapter.actions_from_unapply(action_batch.act)
        return actions.astype(np.float32, copy=False)

    @staticmethod
    def _infer_batch_size(value: Any) -> int:
        """Infer batch size from the first tensor/array/list in a nested input."""

        if torch.is_tensor(value) or isinstance(value, np.ndarray):
            return int(value.shape[0])
        if isinstance(value, Mapping):
            for item in value.values():
                try:
                    return DreamZeroActionPolicy._infer_batch_size(item)
                except (TypeError, IndexError):
                    continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return len(value)
        raise TypeError("Unable to infer DreamZero rollout batch size.")

    def _run_batch_request(
        self,
        normalized_input: dict[str, Any],
        converted_obs: dict[str, Any],
        policy_context: dict[str, Any],
    ) -> np.ndarray:
        """Build session metadata, call the server and postprocess the response.

        RLinf supplies stable logical session ids and keeps a client-side reset
        mask as the cache invalidation hook.  Normal eval requests leave it all
        ``False``; if RLinf later needs to explicitly clear a server session, the
        corresponding row can be set to ``True`` while model/cache lifecycle
        resets remain server-owned.
        """

        batch_size = self._infer_batch_size(normalized_input)
        session_ids = self._session_ids_for_batch(batch_size, policy_context)
        reset_mask = [False] * batch_size
        if self._debug_batch_print:
            print(
                f"[DreamZeroActionPolicy rank={self.rank} "
                f"call={self._eval_predict_calls}] batch_size={batch_size} "
                f"reset_count={sum(reset_mask)} session_ids={session_ids[:4]}",
                flush=True,
            )
        action_request = self._build_action_request(
            normalized_input,
            session_ids=session_ids,
            reset_mask=reset_mask,
        )
        action_result = self.action_client.infer(action_request)
        self._eval_predict_calls += 1
        return self._extract_action_output(
            action_result.normalized_action, converted_obs
        )
