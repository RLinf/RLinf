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

"""Measure peak CUDA memory for dense and chunked BF16 LM head forward/backward."""

import argparse

import torch
import torch.nn.functional as F

from rlinf.hybrid_engines.megatron.lm_head import chunked_vocab_parallel_log_probs


def measure(chunk_size, args):
    torch.manual_seed(0)
    hidden = torch.randn(
        args.tokens, args.hidden_size, device="cuda", dtype=torch.bfloat16
    )
    weight = torch.randn(
        args.vocab_size, args.hidden_size, device="cuda", dtype=hidden.dtype
    )
    weight.mul_(args.hidden_size**-0.5)
    hidden.requires_grad_()
    weight.requires_grad_()
    labels = torch.randint(args.vocab_size, (args.tokens,), device="cuda")
    mask = torch.ones_like(labels, dtype=torch.bool)
    for _ in range(2):  # Warm up once before recording the second pass.
        weight.grad = hidden.grad = None
        torch.cuda.reset_peak_memory_stats()
        if chunk_size:
            scores = chunked_vocab_parallel_log_probs(
                hidden, weight, labels, mask, chunk_size=chunk_size
            )["log_probs"]
        else:
            scores = (
                F.linear(hidden, weight)
                .float()
                .log_softmax(-1)
                .gather(1, labels[:, None])
                .squeeze(1)
            )
        loss = -scores.mean()
        loss.backward()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        del loss, scores
    return peak / 2**30


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()
    if min(args.tokens, args.hidden_size, args.vocab_size, args.chunk_size) <= 0:
        parser.error("all sizes must be positive")
    for name, chunk_size in (("dense", 0), ("chunked", args.chunk_size)):
        print(f"{name}: {measure(chunk_size, args):.3f} GiB")
