# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Batched apply of compose()'d openpi input/output transforms."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable

import numpy as np
import torch
from torch.utils._pytree import tree_map


def _to_numpy(x):
    """Convert tensors to numpy. Cast bf16/fp16 first — numpy has no bfloat16."""
    if not torch.is_tensor(x):
        return x
    x = x.detach().cpu()
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    return np.asarray(x)


def apply_input_transform(
    transform_fn: Callable,
    obs: dict,
    *,
    transpose: bool = False,
) -> dict:
    """Apply the openpi input pipeline per-sample then recombine."""
    inputs = tree_map(lambda x: x, obs)
    first_process = "prompt" in inputs.keys()
    if first_process:
        inputs.pop("prompt")
    else:
        inputs = {k: inputs[k] for k in inputs.keys() if "/" in k}

    inputs = tree_map(_to_numpy, inputs)
    batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

    batch_samples = []
    for i in range(batch_size):
        sample = tree_map(lambda x: x[i], inputs)
        if transpose:
            sample = tree_map(
                lambda x: (
                    x.transpose(1, 2, 0)
                    if isinstance(x, np.ndarray) and x.ndim == 3
                    else x
                ),
                sample,
            )
        if first_process:
            prompts = obs["prompt"]
            if isinstance(prompts, np.ndarray):
                prompts = prompts.tolist()
            sample["prompt"] = prompts[i]
        else:
            sample["prompt"] = "xxxx"
        batch_samples.append(sample)

    with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
        transformed = list(ex.map(transform_fn, batch_samples))

    recombined = tree_map(
        lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
        *transformed,
    )
    if not first_process:
        recombined["tokenized_prompt"] = obs["tokenized_prompt"]
        recombined["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
    return recombined


def apply_output_transform(
    transform_fn: Callable,
    outputs: dict,
    *,
    action_chunk: int | None = None,
) -> dict:
    """Apply the openpi output pipeline per-sample then recombine."""
    batch_size = outputs["actions"].shape[0]
    transformed = []
    for i in range(batch_size):
        sample = tree_map(
            lambda x: _to_numpy(x[i]) if torch.is_tensor(x) else x[i],
            outputs,
        )
        sample = transform_fn(sample)
        transformed.append(sample)
    recombined = tree_map(
        lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
        *transformed,
    )
    if action_chunk is not None:
        recombined["actions"] = recombined["actions"][:, :action_chunk]
    return recombined
