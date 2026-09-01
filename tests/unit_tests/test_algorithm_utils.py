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

import torch

from rlinf.algorithms.utils import preprocess_loss_inputs


def test_sequence_token_level_broadcasts_advantage_and_combines_masks():
    logprobs = torch.zeros(2, 4)
    old_logprobs = torch.zeros_like(logprobs)
    advantages = torch.tensor([[[1.5]], [[-0.5]]])
    env_loss_mask = torch.tensor([[[True]], [[False]]])
    token_mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    out = preprocess_loss_inputs(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=advantages,
        logprob_type="sequence_token_level",
        reward_type="chunk_level",
        loss_mask=env_loss_mask,
        logprob_mask=token_mask,
    )

    assert out["logprobs"].shape == (2, 4)
    assert out["advantages"].shape == (2, 1)
    assert torch.equal(out["advantages"], torch.tensor([[1.5], [-0.5]]))
    assert torch.equal(
        out["loss_mask"],
        torch.tensor([[True, True, False, False], [False, False, False, False]]),
    )
