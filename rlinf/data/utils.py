# Copyright 2025 The RLinf Authors.
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


def batch_pad_to_fixed_len(
    batch: list[torch.Tensor],
    max_batch_len: int,
    pad_token: int,
    left_pad: bool = False,
) -> torch.Tensor:
    if left_pad:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),  # pad on the left
                        seq,
                    ]
                )
                for seq in batch
            ]
        )
    else:
        batch_pad = torch.stack(
            [
                torch.cat(
                    [
                        seq,
                        torch.full(
                            (max_batch_len - len(seq),), pad_token, dtype=seq.dtype
                        ),
                    ]
                )
                for seq in batch
            ]
        )
    return batch_pad
