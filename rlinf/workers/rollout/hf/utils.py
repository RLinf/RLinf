import torch
from typing import Dict, Union
from rlinf.algorithms.utils import expand_to_target_dim

def init_real_next_obs(next_extracted_obs: Union[torch.Tensor, Dict]):
    # Copy the next-extracted-obs
    if isinstance(next_extracted_obs, torch.Tensor):
        real_next_extracted_obs = next_extracted_obs.clone()
    elif isinstance(next_extracted_obs, Dict):
        real_next_extracted_obs = copy_dict_tensor(
            next_extracted_obs
        )
    else:
        raise NotImplementedError
    return real_next_extracted_obs


def copy_dict_tensor(next_extracted_obs: Dict):
    ret = dict()
    for key, value in next_extracted_obs.items():
        if isinstance(value, torch.Tensor):
            ret[key] = value
        elif isinstance(value, Dict):
            ret[key] = init_real_next_obs(value)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
    return ret

def update_real_next_obs(
        real_next_extracted_obs: Union[torch.Tensor, Dict], 
        final_extracted_obs: Union[torch.Tensor, Dict], 
        last_step_dones: torch.BoolTensor
    ):
    # Update the next-extracted-obs according to the final doness
    if isinstance(real_next_extracted_obs, torch.Tensor):
        dones_mask = expand_to_target_dim(last_step_dones, final_extracted_obs.shape)
        dones_mask = dones_mask.expand_as(final_extracted_obs)
        real_next_extracted_obs[dones_mask] = final_extracted_obs[dones_mask]
    elif isinstance(real_next_extracted_obs, Dict):
        real_next_extracted_obs = update_dict_real_next_obs(
            real_next_extracted_obs, final_extracted_obs, last_step_dones
        )
    return real_next_extracted_obs

def update_dict_real_next_obs(
        real_next_extracted_obs: Dict, 
        final_extracted_obs: Dict, 
        last_step_dones: torch.BoolTensor
    ):
    # Update the next-extracted-obs according to the final dones
    for key, value in real_next_extracted_obs.items():
        if isinstance(value, torch.Tensor): 
            dones_mask = expand_to_target_dim(last_step_dones, final_extracted_obs[key].shape)
            dones_mask = dones_mask.expand_as(final_extracted_obs[key])
            real_next_extracted_obs[key][dones_mask] = final_extracted_obs[key][dones_mask]
        elif isinstance(value, Dict):
            real_next_extracted_obs[key] = update_dict_real_next_obs(
                value, final_extracted_obs[key], last_step_dones
            )
    return real_next_extracted_obs