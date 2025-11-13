from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from rlinf.data.io_struct.utils import (
    put_tensor_device, 
    stack_list_of_dict_tensor, 
    split_dict_to_chunk
)


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    # required
    prev_logprobs: List[torch.Tensor] = field(default_factory=list)
    prev_values: List[torch.Tensor] = field(default_factory=list)
    dones: List[torch.Tensor] = field(default_factory=list)
    rewards: List[torch.Tensor] = field(default_factory=list)
    transitions: List[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)

    forward_inputs: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.prev_logprobs = (
            [prev_logprob.cpu().contiguous() for prev_logprob in self.prev_logprobs]
            if self.prev_logprobs is not None
            else []
        )
        self.prev_values = (
            [prev_value.cpu().contiguous() for prev_value in self.prev_values]
            if self.prev_values is not None
            else []
        )
        self.dones = (
            [done.cpu().contiguous() for done in self.dones]
            if self.dones is not None
            else []
        )
        self.rewards = (
            [reward.cpu().contiguous() for reward in self.rewards]
            if self.rewards is not None
            else []
        )

        self.transitions = [
            put_tensor_device(t, "cpu") for t in self.transitions
        ] if self.transitions is not None else []

        self.forward_inputs = [
            put_tensor_device(forward_inputs, "cpu") for forward_inputs in self.forward_inputs
        ]

    def append_result(self, result: Dict[str, Any]):
        self.prev_logprobs.append(
            result["prev_logprobs"].cpu().contiguous()
        ) if "prev_logprobs" in result else []
        self.prev_values.append(
            result["prev_values"].cpu().contiguous()
        ) if "prev_values" in result else []
        self.dones.append(
            result["dones"].cpu().contiguous()
        ) if "dones" in result else []
        self.rewards.append(
            result["rewards"].cpu().contiguous()
        ) if "rewards" in result else []

        self.forward_inputs.append(put_tensor_device(result["forward_inputs"], "cpu"))

    def add_transition(self, obs, next_obs):
        self.transitions.append(
            {
                "obs": put_tensor_device(obs, "cpu"), 
                "next_obs": put_tensor_device(next_obs, "cpu")
            }
        )

    def to_dict(self):
        rollout_result_dict = {}
        rollout_result_dict["prev_logprobs"] = (
            torch.stack(self.prev_logprobs, dim=0).cpu().contiguous()
            if len(self.prev_logprobs) > 0
            else None
        )
        rollout_result_dict["prev_values"] = (
            torch.stack(self.prev_values, dim=0).cpu().contiguous()
            if len(self.prev_values) > 0
            else None
        )
        rollout_result_dict["dones"] = (
            torch.stack(self.dones, dim=0).cpu().contiguous()
            if len(self.dones) > 0
            else None
        )
        rollout_result_dict["rewards"] = (
            torch.stack(self.rewards, dim=0).cpu().contiguous()
            if len(self.rewards) > 0
            else None
        )

        merged_forward_inputs = stack_list_of_dict_tensor(self.forward_inputs)
        for k in merged_forward_inputs.keys():
            assert k not in ["dones", "rewards", "prev_logprobs", "prev_values"]
            rollout_result_dict[k] = merged_forward_inputs[k]

        transition_dict = stack_list_of_dict_tensor(self.transitions)
        rollout_result_dict["transitions"] = transition_dict

        return rollout_result_dict

    def to_splited_dict(self, split_size) -> List[Dict[str, Any]]:
        return split_dict_to_chunk(self.to_dict(), split_size, dim=1)
