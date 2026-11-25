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

"""Verify canonical M-RoPE at the existing Bridge language-model boundary."""

from contextlib import contextmanager
from typing import Iterator

import torch


@contextmanager
def verify_qwen_vl_positions(
    model: torch.nn.Module, expected: torch.Tensor, attention_mask: torch.Tensor
) -> Iterator[None]:
    """Check positions after Bridge's native THD packing, without overriding it."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs,
    )

    if not any(
        cls.__module__.startswith("megatron.bridge.models.qwen_vl.")
        for cls in type(model).__mro__
    ):
        raise ValueError(
            "Token-space position verification requires a Bridge Qwen-VL model"
        )
    checked = False

    def check_positions(module, args, kwargs):
        nonlocal checked
        actual = kwargs["position_ids"]
        canonical = expected
        if kwargs.get("packed_seq_params") is not None:
            canonical = (
                preprocess_packed_seqs(
                    canonical.permute(1, 2, 0),
                    attention_mask,
                    pre_process=True,
                    pg_collection=model.pg_collection,
                )[0]
                .permute(2, 0, 1)
                .contiguous()
            )
        if actual.shape != canonical.shape or not torch.equal(actual, canonical):
            raise ValueError(
                "Megatron-Bridge consumed M-RoPE positions different from SGLang"
            )
        checked = True

    handle = model.language_model.register_forward_pre_hook(
        check_positions, with_kwargs=True
    )
    try:
        yield
        if not checked:
            raise RuntimeError(
                "Bridge language forward did not expose M-RoPE positions"
            )
    finally:
        handle.remove()


@contextmanager
def qwen_vl_replay_inputs(
    model, expected: torch.Tensor, inputs: dict
) -> Iterator[dict]:
    """Use real image spans for Bridge geometry and sampled IDs for language."""
    from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.utils import (
        preprocess_packed_seqs,
    )

    attention_mask = inputs["attention_mask"]
    with verify_qwen_vl_positions(model, expected, attention_mask):
        image_mask = inputs.get("image_input_mask")
        if image_mask is None:
            yield inputs
            return

        # Bridge 0.5 scans IDs for M-RoPE and again after packing for the
        # visual mask. Only a recorded span may participate in either scan.
        original_ids = inputs["input_ids"]
        structure = torch.zeros_like(original_ids)
        structure[:, :-1].masked_fill_(
            image_mask[:, 1:] & ~image_mask[:, :-1], model.vision_start_token_id
        )
        structure.masked_fill_(image_mask, model.image_token_id)
        geometry_inputs = dict(inputs, input_ids=structure)
        del geometry_inputs["image_input_mask"]
        embedded = False

        def embed_tokens(module, args, kwargs):
            nonlocal embedded
            if kwargs.get("input_ids") is structure:
                kwargs["input_ids"] = original_ids
                embedded = True

        def restore_tokens(module, args, kwargs):
            if not embedded:
                raise RuntimeError("Bridge did not embed the replay token IDs")
            token_ids = original_ids
            if kwargs.get("packed_seq_params") is not None:
                token_ids, _ = preprocess_packed_seqs(
                    token_ids,
                    attention_mask,
                    pre_process=True,
                    pg_collection=model.pg_collection,
                )
            kwargs["input_ids"] = token_ids

        language_model = model.language_model
        with (
            language_model.embedding.register_forward_pre_hook(
                embed_tokens, with_kwargs=True, prepend=True
            ),
            language_model.register_forward_pre_hook(
                restore_tokens, with_kwargs=True, prepend=True
            ),
        ):
            yield geometry_inputs


@contextmanager
def ragged_qwen_vl_inputs(batch, preparer, model, packed_params, device):
    """Embed images in row order before Bridge packs the decoder sequence."""
    dense = preparer.dense_token_inputs(batch)
    media = preparer.for_forward(dense, device)
    ids = dense["input_ids"].to(device)
    expected = media.pop("multimodal_position_ids", None)
    inputs = dict(
        input_ids=ids,
        attention_mask=dense["attention_mask"].to(device),
        position_ids=torch.arange(ids.shape[1], device=device).expand_as(ids),
        packed_seq_params=packed_params,
        **media,
    )
    if expected is None:
        yield inputs
    else:
        with qwen_vl_replay_inputs(model, expected, inputs) as replay:
            yield replay
