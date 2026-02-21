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

"""
Value bin special tokens for discretized return prediction.

Provides special tokens <v0>, <v1>, ..., <v{num_bins-1}> for value prediction.
Using single special tokens instead of multi-digit strings ensures:
1. Equal prediction complexity for all bins
2. Proper softmax over bins for probability-weighted value computation
3. Clean top-1 accuracy calculation
"""

# Default number of bins (matching paper: 201 bins for [-1, 0] range)
DEFAULT_NUM_VALUE_BINS = 201

# Token format: <v0>, <v1>, ..., <v200>
VALUE_TOKEN_PREFIX = "<v"
VALUE_TOKEN_SUFFIX = ">"


def get_value_token(bin_id: int) -> str:
    """Get the special token string for a bin ID.

    Args:
        bin_id: The bin index (0 to num_bins-1)

    Returns:
        Token string like "<v0>", "<v100>", "<v200>"
    """
    return f"{VALUE_TOKEN_PREFIX}{bin_id}{VALUE_TOKEN_SUFFIX}"


def get_all_value_tokens(num_bins: int = DEFAULT_NUM_VALUE_BINS) -> list[str]:
    """Get list of all value bin tokens.

    Args:
        num_bins: Number of bins (default: 201)

    Returns:
        List of token strings ["<v0>", "<v1>", ..., "<v{num_bins-1}>"]
    """
    return [get_value_token(i) for i in range(num_bins)]


def parse_value_token(token: str) -> int:
    """Parse a value token string to get the bin ID.

    Args:
        token: Token string like "<v100>"

    Returns:
        Bin ID as integer

    Raises:
        ValueError: If token format is invalid
    """
    if not token.startswith(VALUE_TOKEN_PREFIX) or not token.endswith(
        VALUE_TOKEN_SUFFIX
    ):
        raise ValueError(f"Invalid value token format: {token}")

    bin_str = token[len(VALUE_TOKEN_PREFIX) : -len(VALUE_TOKEN_SUFFIX)]
    return int(bin_str)


def add_value_tokens_to_tokenizer(
    tokenizer, num_bins: int = DEFAULT_NUM_VALUE_BINS
) -> dict[str, int]:
    """Add value bin tokens to a tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer
        num_bins: Number of bins to add

    Returns:
        Dict mapping token strings to their IDs
    """
    value_tokens = get_all_value_tokens(num_bins)
    tokenizer.add_special_tokens({"additional_special_tokens": value_tokens})

    # Build mapping from token to ID
    token_to_id = {
        token: tokenizer.convert_tokens_to_ids(token) for token in value_tokens
    }

    return token_to_id
