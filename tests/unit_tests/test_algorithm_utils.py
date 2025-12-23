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

import pytest
import torch
import torch.nn.functional as F

from rlinf.utils.utils import (
    compute_entropy_from_logits as compute_entropy_from_logits_new,
)
from rlinf.utils.utils import (
    compute_logprobs_from_logits as compute_logprobs_from_logits_new,
)


def compute_logprobs_from_logits_old(logits, target):
    logprobs = -F.cross_entropy(
        logits, target=target, reduction="none"
    )  # [B, action-dim]
    return logprobs


def compute_entropy_from_logits_old(logits, epsilon=1e-10):
    """
    Compute entropy by logits.

    Args:
        logits: [B, vocab-size, seq-len]
    Returns:
        entropy: [B, seq-len]
    """
    all_probs = F.softmax(logits, dim=1)  # [B, vocab-size, seq-len]
    all_log_probs = torch.log(all_probs + epsilon)
    entropy = -torch.sum(all_probs * all_log_probs, dim=1)  # [B, seq-len]
    return entropy


# --- Tests ---


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [5])
@pytest.mark.parametrize("vocab_size", [10])
def test_compute_logprobs_correctness(batch_size, seq_len, vocab_size):
    torch.manual_seed(42)

    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    target = torch.randint(0, vocab_size, (batch_size, seq_len), device=logits.device)

    try:
        logprobs_new = compute_logprobs_from_logits_new(logits, target)
    except ImportError:
        pytest.skip("flash_attn not installed, skipping new implementation test")
    except Exception as e:
        pytest.fail(f"New implementation failed: {e}")

    logits_old = logits.permute(0, 2, 1)

    logprobs_old = compute_logprobs_from_logits_old(logits_old, target)

    # Check shapes
    assert logprobs_new.shape == (batch_size, seq_len)
    assert logprobs_old.shape == (batch_size, seq_len)

    # Check values
    # Note: flash_attn might have slight numerical differences, but should be close
    torch.testing.assert_close(logprobs_new, logprobs_old, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("seq_len", [5])
@pytest.mark.parametrize("vocab_size", [10])
def test_compute_entropy_correctness(batch_size, seq_len, vocab_size):
    torch.manual_seed(42)

    # Generate inputs in [B, S, V] format
    logits = torch.randn(
        batch_size,
        seq_len,
        vocab_size,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    entropy_new = compute_entropy_from_logits_new(logits, dim=-1)

    logits_old = logits.permute(0, 2, 1)
    entropy_old = compute_entropy_from_logits_old(logits_old)

    # Check shapes
    assert entropy_new.shape == (batch_size, seq_len)
    assert entropy_old.shape == (batch_size, seq_len)

    # Check values
    # The old implementation uses epsilon=1e-10 and manual log(softmax + eps)
    # The new implementation uses log_softmax which is more stable.
    # There might be small differences due to epsilon and numerical stability.
    torch.testing.assert_close(entropy_new, entropy_old, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__])
