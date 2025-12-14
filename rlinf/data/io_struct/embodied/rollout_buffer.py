from dataclasses import dataclass, field
import torch
from typing import Tuple, Dict, Any
from rlinf.scheduler import Channel
from rlinf.data.io_struct.utils import (
    put_tensor_device, 
    stack_list_of_dict_tensor, 
    split_dict_to_chunk, 
    cat_list_of_dict_tensor
)
import asyncio


@dataclass(kw_only=True)
class AsyncEmbodiedRolloutBuffer:
    prev_logprobs: asyncio.Queue[torch.Tensor] = field(default_factory=asyncio.Queue)
    prev_values: asyncio.Queue[torch.Tensor] = field(default_factory=asyncio.Queue)
    dones: asyncio.Queue[torch.Tensor] = field(default_factory=asyncio.Queue)
    terminations: asyncio.Queue[torch.Tensor] = field(default_factory=asyncio.Queue)
    truncations: asyncio.Queue[torch.Tensor] = field(default_factory=asyncio.Queue)
    rewards: asyncio.Queue[torch.Tensor] = field(default_factory=asyncio.Queue)
    transitions: asyncio.Queue[Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=asyncio.Queue)

    forward_inputs: asyncio.Queue[Dict[str, Any]] = field(default_factory=asyncio.Queue) 

    batches_per_send = 1

    @staticmethod
    def create_from_dict(data_dict_list):
        data_dict_keys = data_dict_list[0].keys()

        prev_logprobs = asyncio.Queue()
        prev_values = asyncio.Queue()
        dones = asyncio.Queue()
        truncations = asyncio.Queue()
        terminations = asyncio.Queue()
        rewards = asyncio.Queue()
        transitions = asyncio.Queue()
        forward_inputs = asyncio.Queue()


        for data_dict in data_dict_list:
            if "prev_logprobs" in data_dict_keys:
                prev_logprobs.put(data_dict["prev_logprobs"])
            if "prev_values" in data_dict_keys:
                prev_values.put(data_dict["prev_values"])
            if "dones" in data_dict_keys:
                dones.put(data_dict["dones"])
            if "truncations" in data_dict_keys:
                truncations.put(data_dict["truncations"])
            if "terminations" in data_dict_keys:
                terminations.put(data_dict["terminations"])
            if "rewards" in data_dict_keys:
                rewards.put(data_dict["rewards"])
            if "transitions" in data_dict_keys:
                transitions.put(data_dict["transitions"])
            if "forward_inputs" in data_dict_keys:
                forward_inputs.put(data_dict["forward_inputs"])

        return AsyncEmbodiedRolloutBuffer(
            prev_logprobs=prev_logprobs, 
            prev_values=prev_values, 
            dones=dones, 
            terminations=terminations, 
            truncations=truncations, 
            rewards=rewards, 
            forward_inputs=forward_inputs, 
            transitions=transitions
        )


    async def add_result(self, result: Dict[str, Any]):
        assert "prev_logprobs" in result
        await self.prev_logprobs.put(
            result["prev_logprobs"].cpu().contiguous()
        ) # if "prev_logprobs" in result else None
        await self.prev_values.put(
            result["prev_values"].cpu().contiguous()
        ) # if "prev_values" in result else None

        await self.forward_inputs.put(put_tensor_device(result["forward_inputs"], "cpu"))

    async def add_transition(self, obs, next_obs):
        await self.transitions.put(
            {
                "obs": put_tensor_device(obs, "cpu"), 
                "next_obs": put_tensor_device(next_obs, "cpu")
            }
        )
    
    async def add(self, key, items):
        if key == "rewards":
            await self.rewards.put(items)
        elif key == "dones":
            await self.dones.put(items)
        elif key == "terminations":
            await self.terminations.put(items)
        elif key == "truncations":
            await self.truncations.put(items)
        elif key == "prev_values":
            await self.prev_values.put(items)
        else:
            raise NotImplementedError
    
    async def send_data(self, data_channel: Channel, split_num):
        # Collect data
        prev_logprobs = []
        prev_values = []
        dones = []
        truncations = []
        terminations = []
        rewards = []
        transitions = []
        forward_inputs = []
        for _ in range(self.batches_per_send):
            prev_logprobs.append(await self.prev_logprobs.get())
            # prev_values.append(self.prev_values.get())
            dones.append(await self.dones.get())
            truncations.append(await self.truncations.get())
            terminations.append(await self.terminations.get())
            rewards.append(await self.rewards.get())
            transitions.append(await self.transitions.get())
            forward_inputs.append(await self.forward_inputs.get())

        data = {
            "prev_logprobs": torch.cat(prev_logprobs, dim=0).cpu().contiguous(), 
            # "prev_values": torch.cat(prev_values, dim=0).cpu().contiguous(), 
            "dones": torch.cat(dones, dim=0).cpu().contiguous(),
            "truncations": torch.cat(truncations, dim=0).cpu().contiguous(), 
            "terminations": torch.cat(terminations, dim=0).cpu().contiguous(), 
            "rewards": torch.cat(rewards, dim=0).cpu().contiguous(), 
            "transitions": cat_list_of_dict_tensor(transitions)
        }
        data.update(cat_list_of_dict_tensor(forward_inputs))
        splited_data = split_dict_to_chunk(data, split_size=split_num, dim=0)

        # Organize data
        for i in range(split_num):
            data_channel.put(splited_data[i])

    async def run(self, data_channel, split_num):
        cnt = 0
        while True:
            cnt += 1
            await self.send_data(data_channel, split_num)
            
