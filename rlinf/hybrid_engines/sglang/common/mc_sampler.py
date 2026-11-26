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

"""SGLang Student sampling for multi-sample distillation."""

import torch
from sglang.srt.layers.sampler import Sampler


class MCSampler(Sampler):
    """Return independent draws from the same decode row as the emitted token."""

    def __init__(self, mc_k: int):
        super().__init__()
        self.mc_k = mc_k

    def _attach_logprobs_to_output(
        self,
        logits_output,
        logprobs,
        top_logprobs_nums,
        token_ids_logprobs,
        sampling_info,
        batch_next_token_ids,
    ):
        super()._attach_logprobs_to_output(
            logits_output,
            logprobs,
            top_logprobs_nums,
            token_ids_logprobs,
            sampling_info,
            batch_next_token_ids,
        )
        if (
            sampling_info.is_all_greedy
            or sampling_info.need_top_p_sampling
            or sampling_info.need_top_k_sampling
            or sampling_info.need_min_p_sampling
        ):
            raise ValueError("MC-K requires untruncated categorical sampling")
        ids = torch.multinomial(logprobs.exp(), self.mc_k - 1, replacement=True)
        scores = logprobs.gather(-1, ids)
        if logits_output.customized_info is None:
            logits_output.customized_info = {}
        # SGLang carries one entry per emitted token through retraction and HTTP.
        logits_output.customized_info.update(
            mc_extra_ids=ids.tolist(), mc_extra_logprobs=scores.tolist()
        )
