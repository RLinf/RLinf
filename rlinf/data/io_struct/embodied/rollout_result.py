from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch
from rlinf.data.io_struct.utils import put_tensor_cpu


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

        self.transitions = (
            [
                (put_tensor_cpu(obs), put_tensor_cpu(next_obs)) 
                for obs, next_obs in self.transitions
            ] if self.transitions is not None else []
        )

        self.forward_inputs = [
            put_tensor_cpu(forward_inputs) for forward_inputs in self.forward_inputs
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

        self.forward_inputs.append(put_tensor_cpu(result["forward_inputs"]))

    def add_transition(self, obs, next_obs):
        self.transitions.append(
            (put_tensor_cpu(obs), put_tensor_cpu(next_obs))
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

        merged_forward_inputs = {}
        
        for obs, next_obs in self.transitions:
            # only consider the case that obs is dict
            for orig_key in obs.keys():
                obs_key = f"transitions/obs/{orig_key}"
                if obs_key in merged_forward_inputs:
                    merged_forward_inputs[obs_key].append(obs[orig_key])
                else:
                    merged_forward_inputs[obs_key] = [obs[orig_key]]
                
                next_obs_key = f"transitions/next_obs/{orig_key}"
                if next_obs_key in merged_forward_inputs:
                    merged_forward_inputs[next_obs_key].append(next_obs[orig_key])
                else:
                    merged_forward_inputs[next_obs_key] = [next_obs[orig_key]]
                

        for data in self.forward_inputs:
            for k, v in data.items():
                if k in merged_forward_inputs:
                    merged_forward_inputs[k].append(v)
                else:
                    merged_forward_inputs[k] = [v]
 
        for k in merged_forward_inputs.keys():
            assert k not in ["dones", "rewards", "prev_logprobs", "prev_values"]
            rollout_result_dict[k] = (
                torch.stack(merged_forward_inputs[k], dim=0).cpu().contiguous()
            )

        return rollout_result_dict

    def to_splited_dict(self, split_size) -> List[Dict[str, Any]]:
        rollout_result_list = []
        for i in range(split_size):
            rollout_result_list.append(self.to_dict())

            for key, value in rollout_result_list[i].items():
                if isinstance(value, torch.Tensor):
                    rollout_result_list[i][key] = torch.chunk(value, split_size, dim=1)[
                        i
                    ].contiguous()
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")

        return rollout_result_list
