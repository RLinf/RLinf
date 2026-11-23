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

"""Megatron Actor output projection during packed-token scoring."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from megatron.core import tensor_parallel
from megatron.core.tensor_parallel.layers import ColumnParallelLinear

from .lm_head import chunked_vocab_parallel_log_probs
from .utils import _resolve_output_layer

# GPT still transposes the returned tensor. Use its normal forward path with one
# score channel per token (two when entropy is requested) instead of a vocab axis.
_scoring = ContextVar("megatron_actor_output_scoring", default=None)


@contextmanager
def chunked_scoring(model, **score_args):
    layer = _resolve_output_layer(model)
    token = _scoring.set((layer, score_args))
    try:
        yield
    finally:
        _scoring.reset(token)


class _ChunkedProjection:
    def forward(self, input_, *args, **kwargs):
        scoring = _scoring.get()
        if scoring is None or scoring[0] is not self:
            return super().forward(input_, *args, **kwargs)

        weight = kwargs.get("weight", args[0] if args else None)
        if weight is None:
            weight = self.weight
        hidden = input_.to(weight.dtype)
        group = self.tp_group
        if group.size() > 1:
            if self.sequence_parallel:
                hidden = tensor_parallel.gather_from_sequence_parallel_region(
                    hidden, tensor_parallel_output_grad=True, group=group
                )
            elif not self.disable_grad_reduce:
                hidden = tensor_parallel.copy_to_tensor_model_parallel_region(
                    hidden, group=group
                )
        scores = chunked_vocab_parallel_log_probs(
            hidden.transpose(0, 1),
            weight,
            bias=None if self.skip_bias_add else self.bias,
            tp_group=group,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            **scoring[1],
        )
        return torch.stack(list(scores.values()), dim=-1).transpose(0, 1), None


class ChunkedColumnParallelLinear(_ChunkedProjection, ColumnParallelLinear):
    """Column-parallel linear with optional packed-token scoring."""


def install_chunked_projection(model, *, bridge=False):
    if not model.post_process:
        return
    layer = _resolve_output_layer(model)
    cls = type(layer)
    if (
        not isinstance(layer, ColumnParallelLinear)
        or cls.forward is not ColumnParallelLinear.forward
        or "forward" in vars(layer)
    ):
        raise ValueError(
            f"Chunked LM head requires the standard ColumnParallelLinear forward, got {cls.__name__}"
        )
    if layer.is_expert:
        raise ValueError("Chunked LM head requires a vocabulary-parallel output layer")
    config = layer.config
    for name in (
        "mtp_num_layers",
        "use_mup",
        "defer_embedding_wgrad_compute",
        "fp8",
        "enable_autocast",
    ):
        if getattr(config, name, False):
            raise ValueError(f"Chunked LM head does not support {name}")
    if cls is ColumnParallelLinear:
        projection = ChunkedColumnParallelLinear
    else:
        projection = type(f"Chunked{cls.__name__}", (_ChunkedProjection, cls), {})
    if bridge:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping

        AutoMapping.register_module_type(projection.__name__, "column")
    layer.__class__ = projection
