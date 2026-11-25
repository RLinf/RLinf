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

"""Shared SGLang image preprocessing for rollout preflight and learner replay."""

import copy
import dataclasses
import hashlib
import inspect
import json
import weakref
from functools import lru_cache
from importlib.metadata import distribution, version
from pathlib import Path

import torch

SGLANG_TOKEN_SPACE_COMMIT = "18a44326f6de5134fe921927d760d81f8ee48fc9"
TOKEN_SPACE_FORMAT = "sglang-token-space-v1"
CACHE_SETTINGS = {
    "enable_token_space_preprocess_cache": False,
    "token_space_preprocess_cache_dir": None,
    "token_space_preprocess_cache_trust_url": False,
    "token_space_preprocess_cache_check_rate": 0.0,
}


def is_token_space_build() -> bool:
    """Identify the pinned experimental SGLang build outside the release range."""
    direct_url = distribution("sglang").read_text("direct_url.json")
    return (
        bool(direct_url)
        and json.loads(direct_url).get("vcs_info", {}).get("commit_id")
        == SGLANG_TOKEN_SPACE_COMMIT
    )


def require_token_space_support() -> None:
    """Reject missing request and preprocessing APIs before spawning an engine."""
    import msgspec
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.server_args import ServerArgs

    fields = (
        dataclasses.fields(ServerArgs)
        if dataclasses.is_dataclass(ServerArgs)
        else msgspec.structs.fields(ServerArgs)
    )
    required = {"mm_token_expansion_start_len", "return_prompt_token_ids"}
    if not required.issubset(GenerateReqInput.__dataclass_fields__) or (
        "enable_token_space_processor" not in {field.name for field in fields}
    ):
        raise RuntimeError(
            "Token-space rollout needs SGLang prompt-ID return and incremental "
            f"media expansion APIs; use the pinned build {SGLANG_TOKEN_SPACE_COMMIT}. "
            "The default agentic environment does not provide these APIs."
        )
    from sglang.srt.multimodal.token_space_mm_process.processor import (
        TokenSpaceProcessor,
    )

    if (
        "mm_token_expansion_start_len"
        not in inspect.signature(TokenSpaceProcessor.process).parameters
    ):
        raise RuntimeError("SGLang TokenSpaceProcessor lacks incremental expansion")


def processor_settings(server_args: dict) -> dict:
    """Resolve an independent copy for preprocessing, preserving launch inputs."""
    from sglang.srt.server_args import ServerArgs

    resolved = ServerArgs(**copy.deepcopy(server_args))
    resolved.resolve_once()
    settings = resolved.resolved_dict()
    names = (
        "model_path",
        "tokenizer_path",
        "tokenizer_mode",
        "tokenizer_backend",
        "trust_remote_code",
        "revision",
        "model_config_parser",
        "json_model_override_args",
        "image_processor_backend",
        "disable_fast_image_processor",
        "mm_process_config",
    )
    processor = {name: settings[name] for name in names}
    if settings.get("enable_token_space_preprocess_cache", False):
        processor.update(
            {
                name: settings.get(name, default)
                for name, default in CACHE_SETTINGS.items()
            }
        )
    return processor


def tensor_digest(tensor: torch.Tensor) -> str:
    """Hash a tensor's dtype, shape and exact CPU bytes."""
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(str((tensor.dtype, tuple(tensor.shape))).encode())
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def image_spans(output: dict) -> list[tuple[int, int]]:
    """Read one contiguous placeholder span per image from SGLang's output."""
    spans = output["mm_spans"]
    if any(len(item) != 1 for item in spans):
        raise ValueError("Image token-space processing expects one span per image")
    return [tuple(item[0]) for item in spans]


def collapse_image_spans(ids: list[int], spans: list, image_id: int) -> list[int]:
    """Validate explicit media placement and collapse each span to one marker."""
    collapsed, cursor = [], 0
    for start, end in spans:
        if not cursor <= start < end <= len(ids):
            raise ValueError(
                "Image spans must be ordered, non-overlapping token ranges"
            )
        if ids[start:end] != [image_id] * (end - start):
            raise ValueError("Image span contains a non-image token")
        if image_id in ids[cursor:start]:
            raise ValueError("Image token outside structured media spans")
        collapsed.extend(ids[cursor:start])
        collapsed.append(image_id)
        cursor = end
    if image_id in ids[cursor:]:
        raise ValueError("Image token outside structured media spans")
    collapsed.extend(ids[cursor:])
    return collapsed


class SGLangTokenSpace:
    """Own one CPU image processor with the serving model's exact settings.

    The initial integration uses PIL image preprocessing. Vision encoding and
    its weights stay with the existing serving and training model providers.
    """

    def __init__(self, settings: dict):
        from sglang.srt.multimodal.processors.processor_config import (
            MultimodalProcessorConfig,
        )
        from sglang.srt.multimodal.token_space_mm_process.processor import (
            TokenSpaceProcessor,
        )
        from sglang.srt.multimodal.token_space_mm_process.registry import (
            register_token_space_models,
            strategy_for,
        )
        from sglang.srt.utils.hf_transformers import get_config, get_processor

        self.settings = settings
        if not settings["disable_fast_image_processor"]:
            raise ValueError(
                "Token-space image replay currently requires the PIL processor"
            )
        config = get_config(
            settings["model_path"],
            trust_remote_code=settings["trust_remote_code"],
            revision=settings["revision"],
            model_config_parser=settings["model_config_parser"],
            model_override_args=json.loads(
                settings["json_model_override_args"] or "{}"
            ),
        )
        register_token_space_models()
        strategy = strategy_for(config)
        if strategy is None:
            raise ValueError(f"No token-space strategy for {config.architectures}")
        processor = get_processor(
            settings["tokenizer_path"] or settings["model_path"],
            tokenizer_mode=settings["tokenizer_mode"],
            tokenizer_backend=settings["tokenizer_backend"],
            trust_remote_code=settings["trust_remote_code"],
            revision=settings["revision"],
            image_processor_backend="pil",
            model_name=settings["model_path"],
        )
        self.fingerprint = self._fingerprint(config, processor, TokenSpaceProcessor)
        self.cache_directory = None
        cache = None
        if settings.get("enable_token_space_preprocess_cache", False):
            from sglang.srt.multimodal.cache import DiskPreprocessCache

            self.cache_directory = str(
                Path(settings["token_space_preprocess_cache_dir"]) / self.fingerprint
            )
            Path(self.cache_directory).mkdir(mode=0o700, parents=True, exist_ok=True)
            cache = DiskPreprocessCache(self.cache_directory)
        self.core = TokenSpaceProcessor.for_strategy(strategy)(
            config,
            processor,
            processor_config=MultimodalProcessorConfig(
                image_processor_backend=settings["image_processor_backend"],
                disable_fast_image_processor=True,
                mm_process_config=settings["mm_process_config"] or {},
                mm_processor_worker_num=0,
                mm_io_worker_num=1,
            ),
            item_feature_cache=cache,
            item_feature_cache_check_rate=settings.get(
                "token_space_preprocess_cache_check_rate", 0.0
            ),
        )
        self._close = weakref.finalize(self, self.core.shutdown)
        self.image_id = config.image_token_id

    def shutdown(self) -> None:
        """Release the native processor's IO and preprocessing executors."""
        self._close()

    def _fingerprint(self, config, processor, core_class) -> str:
        source = hashlib.sha256()
        root = Path(inspect.getfile(core_class)).parent.parent
        for path in sorted(root.rglob("*.py")):
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
        description = {
            "format": TOKEN_SPACE_FORMAT,
            # Cache location and checking frequency do not change model inputs.
            "settings": {
                key: value
                for key, value in self.settings.items()
                if key not in CACHE_SETTINGS
            },
            "config": config.to_dict(),
            "processor": processor.to_dict(),
            "image_processor": processor.image_processor.to_dict(),
            "tokenizer_vocab": processor.tokenizer.get_vocab(),
            "chat_template": processor.chat_template,
            "sglang_source": source.hexdigest(),
            "versions": {
                name: version(name)
                for name in ("sglang", "transformers", "torch", "Pillow")
            },
        }
        return hashlib.sha256(
            json.dumps(description, sort_keys=True).encode()
        ).hexdigest()

    async def process(self, ids: list[int], images: list, prefix_length: int) -> dict:
        """Expand only the new suffix, using SGLang's native processing core."""
        return await self.core.process(
            input_ids=ids,
            image_data=images,
            mm_token_expansion_start_len=prefix_length,
            compute_mrope_positions=True,
        )

    async def prepare(self, ids: list[int], images: list, request: dict) -> dict:
        """Return canonical tokens and a compact, tensor-free replay contract."""
        prefix = request["prefix_length"]
        spans = request["image_spans"]
        if not 0 <= prefix <= len(ids) or len(spans) != len(images):
            raise ValueError("Invalid expanded prefix length or media occurrence count")
        if any(
            start < prefix < end or (start >= prefix and end - start != 1)
            for start, end in spans
        ):
            raise ValueError(
                "Only single new image markers may follow the expanded prefix"
            )
        compact = collapse_image_spans(ids, spans, self.image_id)
        output = await self.process(ids, images, prefix)
        expanded = torch.as_tensor(output["input_ids"]).flatten().tolist()
        actual_spans = image_spans(output)
        if (
            expanded[:prefix] != ids[:prefix]
            or collapse_image_spans(expanded, actual_spans, self.image_id) != compact
        ):
            raise ValueError(
                "SGLang expansion changed text or previously sampled tokens"
            )
        return {
            "format": TOKEN_SPACE_FORMAT,
            "processor_settings": self.settings,
            "processor_fingerprint": self.fingerprint,
            "prompt_token_ids": expanded,
            "image_spans": actual_spans,
            "image_grid_thw": output["image_grid_thw"].cpu(),
            "pixel_values_sha256": tensor_digest(output["pixel_values"]),
            # Learners replay pixels in BF16, independently of processor dtype.
            "pixel_values_bytes": output["pixel_values"].numel() * 2,
            "mrope_sha256": tensor_digest(output["mrope_positions"]),
        }

    def positions(self, entry: dict, ids: list[int]) -> torch.Tensor:
        """Compute the complete training row's M-RoPE, including response tokens."""
        collapse_image_spans(ids, entry["image_spans"], self.image_id)
        positions, _ = self.core.strategy.compute_mrope_positions(
            {
                "input_ids": ids,
                "image_grid_thw": entry["image_grid_thw"],
                "mm_spans": [[tuple(span)] for span in entry["image_spans"]],
            }
        )
        return positions.cpu()

    async def replay(
        self, entry: dict, images: list
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rebuild and verify recorded pixels, placement and prompt M-RoPE."""
        if (
            entry["format"] != TOKEN_SPACE_FORMAT
            or entry["processor_fingerprint"] != self.fingerprint
        ):
            raise ValueError(
                "Token-space processor or trajectory format changed; start a fresh rollout"
            )
        ids = entry["prompt_token_ids"]
        output = await self.process(ids, images, len(ids))
        if torch.as_tensor(
            output["input_ids"]
        ).flatten().tolist() != ids or image_spans(output) != [
            tuple(s) for s in entry["image_spans"]
        ]:
            raise ValueError("Learner token IDs or image spans differ from rollout")
        if not torch.equal(output["image_grid_thw"].cpu(), entry["image_grid_thw"]):
            raise ValueError("Learner image grid differs from rollout")
        for key, digest in (
            ("pixel_values", "pixel_values_sha256"),
            ("mrope_positions", "mrope_sha256"),
        ):
            if tensor_digest(output[key]) != entry[digest]:
                raise ValueError(f"Learner {key} differs from rollout")
        return output["pixel_values"].to(torch.bfloat16).cpu(), output[
            "image_grid_thw"
        ].cpu()


@lru_cache(maxsize=2)
def token_space_processor(settings_json: str) -> SGLangTokenSpace:
    """Reuse the processor across learner microbatches with identical settings."""
    return SGLangTokenSpace(json.loads(settings_json))
