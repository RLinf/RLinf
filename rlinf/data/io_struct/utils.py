import torch
from typing import Dict, List

def put_tensor_device(data_dict, device):
    if data_dict is None:
        return None

    if isinstance(data_dict, torch.Tensor):
        return data_dict.value.to(device=device).contiguous()
    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_device(value, device)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device=device).contiguous()
    return data_dict


def stack_list_of_dict_tensor(list_of_dict: List, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = dict()
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.stack(v_list, dim=dim)
        elif isinstance(_v0, Dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = stack_list_of_dict_tensor(v_list)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret

def cat_list_of_dict_tensor(list_of_dict: List, dim=0):
    if len(list_of_dict) == 0:
        return {}
    keys = list_of_dict[0].keys()

    ret = dict()
    for key in keys:
        _v0 = list_of_dict[0][key]
        if isinstance(_v0, torch.Tensor):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = torch.cat(v_list, dim=dim)
        elif isinstance(_v0, Dict):
            v_list = [d[key] for d in list_of_dict]
            ret[key] = cat_list_of_dict_tensor(v_list)
        else:
            raise ValueError(f"{key=}, {type(_v0)} is not supported!")
    return ret

def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = dict()
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(
                new_value.shape[0], -1, *new_value.shape[3:]
            )
            ret_dict[key] = new_value
        elif isinstance(value, Dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict

def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = dict()
    for key, value in nested_dict.items():
        if key in ["dones", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, Dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict

def process_nested_dict_for_replay_buffer(nested_dict):
    ret_dict = dict()
    num_data = None
    for key, value in nested_dict.items():
        if key in ["dones", "prev_values"]:
            value = value[:-1]
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:]).cpu()
            if num_data is not None:
                assert num_data == ret_dict[key].shape[0]
            num_data = ret_dict[key].shape[0]
        elif isinstance(value, Dict):
            ret_dict[key], num_data = process_nested_dict_for_replay_buffer(value)
    if len(ret_dict) > 0:
        assert num_data is not None
    return ret_dict, num_data
        
def split_dict_to_chunk(data: Dict, split_size, dim=0):
    splited_list = [{} for _ in range(split_size)]
    for key, value in data.items():
        if isinstance(value, torch.Tensor):
            split_vs = torch.chunk(value, split_size, dim=dim)
        elif value is None:
            split_vs = [None for _ in range(split_size)]
        elif isinstance(value, Dict):
            split_vs = split_dict_to_chunk(value, split_size, dim)
        else:
            raise ValueError(f"{key=}, {type(value)} is not supported.")
        for split_id in range(split_size):
            splited_list[split_id][key] = split_vs[split_id]
    return splited_list


def get_batch_size(
    batch: Dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    """Get the batch size from the batch dictionary."""
    return batch[batch_tensor_key].size(0)


def get_seq_length(
    batch: Dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    """Get the sequence length from the batch dictionary."""
    return batch[batch_tensor_key].size(1)

