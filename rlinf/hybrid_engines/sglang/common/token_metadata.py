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

"""Per-token metadata exchanged across SGLang output packets."""


def accumulate_token_metadata(recv_obj, request_states: dict) -> None:
    """Join packet-local metadata before SGLang emits cumulative output IDs."""
    metadata = getattr(recv_obj, "customized_info", None) or {}
    for name in ("rollout_weight_versions", "forced_token_mask"):
        values = metadata.get(name)
        if values is None:
            continue
        for index, rid in enumerate(recv_obj.rids):
            state = request_states.get(rid)
            if state is None:
                continue
            state_key = f"rlinf_{name}"
            accumulated = getattr(state, state_key, [])
            accumulated.extend(values[index])
            setattr(state, state_key, accumulated)
            values[index] = list(accumulated)


def decode_generation_result(result: dict, *, return_logprobs: bool) -> dict:
    """Validate engine metadata before completing the requesting channel call."""
    # Text can omit matched stop tokens; preserve the exact generated IDs.
    result_dict = {
        "output_ids": result["output_ids"],
        "finish_reason": result["meta_info"]["finish_reason"]["type"],
        "prompt_tokens": result["meta_info"]["prompt_tokens"],
    }
    if "prompt_token_ids" in result:
        result_dict["prompt_token_ids"] = result["prompt_token_ids"]
    if "forced_token_mask" in result["meta_info"]:
        flags = result["meta_info"]["forced_token_mask"]
        if len(flags) != len(result_dict["output_ids"]) or any(
            type(flag) is not bool for flag in flags
        ):
            raise ValueError("Engine forced-token metadata must align with output IDs")
        result_dict["forced_token_positions"] = [
            index for index, forced in enumerate(flags) if forced
        ]
    if return_logprobs:
        result_dict["logprobs"] = [
            item[0] for item in result["meta_info"]["output_token_logprobs"]
        ]
    if "rollout_weight_versions" in result["meta_info"]:
        result_dict["weight_versions"] = result["meta_info"]["rollout_weight_versions"]
    return result_dict
