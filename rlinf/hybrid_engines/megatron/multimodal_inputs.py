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

"""Keep literal visual special tokens in the text domain for Megatron-Bridge VLMs."""

import inspect
from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from types import FunctionType, MethodType
from typing import Any

import torch


def build_visual_lookup_ids(
    input_ids: torch.Tensor, mm_token_type_ids: torch.Tensor, model: Any
) -> torch.Tensor:
    """Mask literal media and vision-start tokens to ``-1`` for visual scans only."""
    if input_ids.shape != mm_token_type_ids.shape:
        raise ValueError(
            "mm_token_type_ids shape must match input_ids: "
            f"expected {tuple(input_ids.shape)}, got {tuple(mm_token_type_ids.shape)}."
        )
    token_types = mm_token_type_ids.to(device=input_ids.device)
    visual = input_ids == model.image_token_id
    video_token_id = getattr(model, "video_token_id", None)
    if video_token_id is not None:
        visual = visual | (input_ids == video_token_id)
    start = input_ids == model.vision_start_token_id
    actual_start = torch.zeros_like(start)
    actual_start[..., :-1] = token_types[..., 1:] != 0
    return input_ids.masked_fill(
        (visual & (token_types == 0)) | (start & ~actual_start), -1
    )


def install_qwen_vl_token_mask(
    model_chunks: list[torch.nn.Module],
) -> list[torch.nn.Module]:
    """Attach a ``mm_token_type_ids`` mask hook on Bridge Qwen-VL forward calls.

    When ``mm_token_type_ids`` is supplied, only tokens belonging to real image or
    video spans enter ``reorganize_inputs`` and ``get_rope_index``. Language
    embeddings still see the unmodified ``input_ids``, and native PP, VPP, CP,
    and deepstack handling remain intact.
    """
    for model in model_chunks:
        if getattr(model, "_rlinf_qwen_vl_token_mask", False):
            continue
        if not (
            hasattr(model, "image_token_id") and hasattr(model, "vision_start_token_id")
        ):
            continue
        forward_fn = getattr(model, "forward", None)
        original = getattr(forward_fn, "__func__", None)
        required = ("reorganize_inputs", "get_rope_index", "preprocess_packed_seqs")
        if original is None:
            raise ValueError("Unsupported Megatron-Bridge Qwen-VL forward contract.")
        namespace = original.__globals__
        if not all(callable(namespace.get(name)) for name in required):
            raise ValueError("Unsupported Megatron-Bridge Qwen-VL forward contract.")
        signature = inspect.signature(original)
        reorganize = namespace["reorganize_inputs"]
        rope = namespace["get_rope_index"]
        pack = namespace["preprocess_packed_seqs"]
        rope_signature = inspect.signature(rope)

        def make_forward(
            original, signature, namespace, reorganize, rope, pack, rope_signature
        ):
            @wraps(original)
            def forward(self, *args, **kwargs):
                bound = signature.bind(self, *args, **kwargs)
                bound.apply_defaults()
                values = bound.arguments
                input_ids = values["input_ids"]
                types = values.get("mm_token_type_ids")
                if types is None:
                    return original(self, *args, **kwargs)
                types = types.to(device=input_ids.device)
                lookup_ids = build_visual_lookup_ids(input_ids, types, self)
                packed_types = None
                has_visual_grid = False

                def masked_reorganize(*call_args, **call_kwargs):
                    nonlocal packed_types, has_visual_grid
                    ids = call_kwargs.get(
                        "input_ids", call_args[0] if call_args else None
                    )
                    if ids is input_ids:
                        vision_data, vision_grid, vision_mask = reorganize(
                            **{
                                **call_kwargs,
                                "input_ids": lookup_ids,
                                "image_input_mask": lookup_ids == self.image_token_id,
                                "video_input_mask": (
                                    lookup_ids == getattr(self, "video_token_id", -2)
                                ),
                            }
                        )
                        has_visual_grid = (
                            vision_grid is not None and vision_grid.shape[0] > 0
                        )
                        return (
                            vision_data,
                            vision_grid,
                            vision_mask if has_visual_grid else None,
                        )
                    # Second call after packing / CP splitting: project mm_token_type_ids
                    # with the exact same packed sequence layout.
                    if values.get("packed_seq_params") is None:
                        raise ValueError("Unexpected Qwen-VL visual ID layout.")
                    if not has_visual_grid:
                        return None, None, None
                    if packed_types is None:
                        mask = values.get("attention_mask")
                        if mask is None:
                            mask = torch.ones_like(input_ids, dtype=torch.bool)
                        packed_types = pack(
                            types,
                            mask,
                            pre_process=True,
                            pg_collection=self.pg_collection,
                        )[0]
                    if packed_types.shape != ids.shape:
                        raise ValueError(
                            "Packed mm_token_type_ids shape must match CP-local input_ids."
                        )
                    visual = ids == self.image_token_id
                    video_token_id = getattr(self, "video_token_id", None)
                    if video_token_id is not None:
                        visual = visual | (ids == video_token_id)
                    return None, None, visual & (packed_types != 0)

                def masked_rope(*call_args, **call_kwargs):
                    rope_bound = rope_signature.bind(*call_args, **call_kwargs)
                    rope_bound.arguments["input_ids"] = lookup_ids
                    return rope(*rope_bound.args, **rope_bound.kwargs)

                scoped = FunctionType(
                    original.__code__,
                    {
                        **namespace,
                        "reorganize_inputs": masked_reorganize,
                        "get_rope_index": masked_rope,
                    },
                    original.__name__,
                    original.__defaults__,
                    original.__closure__,
                )
                scoped.__kwdefaults__ = original.__kwdefaults__
                return scoped(self, *args, **kwargs)

            return forward

        model.forward = MethodType(
            make_forward(
                original, signature, namespace, reorganize, rope, pack, rope_signature
            ),
            model,
        )
        model._rlinf_qwen_vl_token_mask = True
    return model_chunks


@contextmanager
def ragged_multimodal_inputs(
    batch: dict,
    model: torch.nn.Module,
    rows: list[torch.Tensor],
    *,
    aligned_mode: bool = True,
) -> Iterator[dict]:
    """Build TP/CP-aligned ragged multimodal forward inputs for a Bridge VLM."""
    from megatron.training.training import unwrap_model

    from rlinf.hybrid_engines.megatron import utils as megatron_utils
    from rlinf.utils.multimodal import dense_token_inputs, materialize_multimodal_inputs

    model = unwrap_model(model)
    if not megatron_utils.model_packs_inputs(model):
        raise ValueError(
            "Ragged multimodal training requires a Megatron-Bridge Qwen3VLModel, "
            f"got {type(model).__name__}."
        )
    device = rows[0].device
    dense = dense_token_inputs(batch)
    media = materialize_multimodal_inputs(
        dense.get("multi_modal_inputs") or [],
        device,
        input_ids=dense["input_ids"],
        attention_mask=dense["attention_mask"],
        aligned_mode=aligned_mode,
    )
    if "mm_token_type_ids" in media and not getattr(
        model, "_rlinf_qwen_vl_token_mask", False
    ):
        raise ValueError(
            "Qwen-VL token mask hook is not installed on this Megatron-Bridge model."
        )
    raw_ids = dense["input_ids"].to(device)
    raw_mask = dense["attention_mask"].to(device)
    ids, aligned_mask, packed_params = megatron_utils.prepare_full_model_inputs(
        raw_ids, raw_mask
    )
    if "mm_token_type_ids" in media:
        token_types = media["mm_token_type_ids"]
        order = torch.sort(
            (~raw_mask.bool()).to(torch.int8), dim=1, stable=True
        ).indices
        token_types = token_types.gather(1, order)
        width = ids.shape[1]
        if token_types.shape[1] < width:
            token_types = torch.nn.functional.pad(
                token_types, (0, width - token_types.shape[1]), value=0
            )
        else:
            token_types = token_types[:, :width]
        media["mm_token_type_ids"] = token_types
    yield dict(
        input_ids=ids,
        attention_mask=aligned_mask,
        position_ids=None,
        packed_seq_params=packed_params,
        **media,
    )
