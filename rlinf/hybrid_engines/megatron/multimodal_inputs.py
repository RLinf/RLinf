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

"""Build and verify TP/CP-aligned ragged multimodal inputs for Megatron-Bridge VLMs."""

from collections.abc import Iterator
from contextlib import contextmanager

import torch

from rlinf.hybrid_engines.megatron.utils import (
    model_packs_inputs,
    prepare_full_model_inputs,
)


def _extend_expected_positions(
    expected: torch.Tensor,
    orig_mask: torch.Tensor,
    actual_mask: torch.Tensor,
) -> torch.Tensor:
    """Extend canonical M-RoPE positions across trailing TP-alignment padding."""
    batch_size, width = actual_mask.shape
    if expected.shape == (3, batch_size, width) and torch.equal(
        orig_mask.bool(), actual_mask.bool()
    ):
        return expected
    padded = torch.zeros(
        (3, batch_size, width), dtype=expected.dtype, device=expected.device
    )
    orig_bool = orig_mask.bool()
    actual_bool = actual_mask.bool()
    for row in range(batch_size):
        row_orig = orig_bool[row]
        valid = int(row_orig.sum().item())
        aligned = int(actual_bool[row].sum().item())
        if valid > 0:
            padded[:, row, :valid] = expected[:, row, row_orig]
        if aligned > valid:
            base = int(padded[:, row, valid - 1].max().item()) + 1 if valid > 0 else 0
            extra = torch.arange(
                base,
                base + (aligned - valid),
                dtype=expected.dtype,
                device=expected.device,
            )
            padded[:, row, valid:aligned] = extra
    return padded


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
            "Token-space position verification requires a Bridge Qwen-VL model, "
            f"got {type(model).__name__}."
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
                "Megatron-Bridge consumed M-RoPE positions different from SGLang: "
                f"expected shape {tuple(canonical.shape)}, got {tuple(actual.shape)}."
            )
        checked = True

    handle = model.language_model.register_forward_pre_hook(
        check_positions, with_kwargs=True
    )
    try:
        yield
        if not checked:
            raise RuntimeError(
                "Bridge language forward did not expose M-RoPE positions."
            )
    finally:
        handle.remove()


@contextmanager
def qwen_vl_replay_inputs(
    model: torch.nn.Module, expected: torch.Tensor, inputs: dict
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

        # Bridge 0.5 scans input_ids for M-RoPE and again after packing for the
        # visual mask. Only recorded image spans may participate in either scan.
        original_ids = inputs["input_ids"]
        structure = torch.zeros(
            original_ids.shape, dtype=original_ids.dtype, device=original_ids.device
        )
        structure[:, :-1].masked_fill_(
            image_mask[:, 1:] & ~image_mask[:, :-1], model.vision_start_token_id
        )
        structure.masked_fill_(image_mask, model.image_token_id)
        geometry_inputs = dict(inputs, input_ids=structure)
        del geometry_inputs["image_input_mask"]
        embedded = False

        def embed_tokens(module, args, kwargs):
            nonlocal embedded
            if args and args[0] is structure:
                embedded = True
                return (original_ids, *args[1:]), kwargs
            if kwargs.get("input_ids") is structure:
                kwargs["input_ids"] = original_ids
                embedded = True
                return args, kwargs
            return None

        def restore_tokens(module, args, kwargs):
            if not embedded:
                raise RuntimeError("Bridge did not embed the replay token IDs.")
            token_ids = original_ids
            if kwargs.get("packed_seq_params") is not None:
                token_ids, _ = preprocess_packed_seqs(
                    token_ids,
                    attention_mask,
                    pre_process=True,
                    pg_collection=model.pg_collection,
                )
            if "input_ids" in kwargs:
                kwargs["input_ids"] = token_ids
                return args, kwargs
            if args:
                return (token_ids, *args[1:]), kwargs
            kwargs["input_ids"] = token_ids
            return args, kwargs

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
def ragged_multimodal_inputs(
    batch: dict, preparer, model: torch.nn.Module, rows: list[torch.Tensor]
) -> Iterator[dict]:
    """Build TP/CP-aligned ragged multimodal forward inputs for a Bridge VLM."""
    from megatron.training.training import unwrap_model

    model = unwrap_model(model)
    if not model_packs_inputs(model):
        raise ValueError(
            "Ragged multimodal training requires a Megatron-Bridge Qwen3VLModel, "
            f"got {type(model).__name__}."
        )
    device = rows[0].device
    if preparer.prefetch_batches:
        # Pop prepared CPU tensors directly from the caller's batch so they are
        # released on handoff instead of surviving on the ragged batch dict.
        media = preparer.for_forward(batch, device)
        dense = preparer.dense_token_inputs(batch)
    else:
        dense = preparer.dense_token_inputs(batch)
        media = preparer.for_forward(dense, device)
    raw_ids = dense["input_ids"].to(device)
    raw_mask = dense["attention_mask"].to(device)
    ids, aligned_mask, packed_params = prepare_full_model_inputs(raw_ids, raw_mask)
    expected = media.pop("multimodal_position_ids", None)
    if expected is not None:
        expected = _extend_expected_positions(expected, raw_mask, aligned_mask)
    if "image_input_mask" in media:
        image_mask = media["image_input_mask"]
        width = ids.shape[1]
        if image_mask.shape[1] < width:
            image_mask = torch.nn.functional.pad(
                image_mask, (0, width - image_mask.shape[1]), value=False
            )
        else:
            image_mask = image_mask[:, :width]
        media["image_input_mask"] = image_mask
    inputs = dict(
        input_ids=ids,
        attention_mask=aligned_mask,
        position_ids=None,
        packed_seq_params=packed_params,
        **media,
    )
    if expected is None:
        yield inputs
    else:
        with qwen_vl_replay_inputs(model, expected, inputs) as replay:
            yield replay
