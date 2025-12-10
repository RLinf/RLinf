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

import copy
import heapq
import itertools
import logging
from collections import UserDict
from typing import Any, Iterator, Optional, Union

import numpy as np
import torch
from torch import distributed as dist


def concat_dict_list(list_of_dicts: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Concatenates torch.Tensor or np.ndarray objects corresponding to the same keys in a list of dictionaries.
    Values of other types are collected into lists.

    Args:
        list_of_dicts: Input list of dictionaries, where each dictionary contains the same set of keys

    Returns:
        Processed dictionary where tensors/arrays are concatenated, and other types are stored as lists
    """
    if not list_of_dicts:
        return {}

    # Get all keys (based on the first dictionary) and sort them for consistency
    keys = sorted(list_of_dicts[0].keys())
    result = {key: [] for key in keys}

    for d in list_of_dicts:
        for key in keys:
            assert key in d, f"Missing key in dictionary: {key}"
            result[key].append(d[key])

    for key in result:
        values = result[key]
        first_val = values[0]

        if isinstance(first_val, torch.Tensor):
            result[key] = torch.cat(values)
        elif isinstance(first_val, np.ndarray):
            result[key] = np.concatenate(values)
        # Keep non-tensor/non-array types as lists

    return result


def split_list(
    inputs: list, num_chunks: int, enforce_divisible_batch: Optional[bool] = True
):
    """
    Split a list into equal sized chunks
    """
    if enforce_divisible_batch:
        chunk_size = len(inputs) // num_chunks
        assert len(inputs) % chunk_size == 0, (
            f"Issue with batch size configuration! inputs len:{len(inputs)} num_chunks:{num_chunks}"
        )
        return [inputs[i : i + chunk_size] for i in range(0, len(inputs), chunk_size)]
    else:
        k, m = divmod(len(inputs), num_chunks)
        return [
            inputs[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)]
            for i in range(num_chunks)
        ]


def get_iterator_k_split(
    batch: Union[dict, list[torch.Tensor]],
    num_splits: int,
    enforce_divisible_batch: Optional[bool] = True,
    shuffle: bool = False,
    shuffle_seed: Optional[int] = None,
) -> Iterator:
    """
    Split a batch into k microbatches, where the batch size is divisible by k. Batch could be
    a dictionary of tensors or a list of tensors. A dictionary batch could also have items of List type,
    as long as the length of that list is the same as the batch size.

    Args:
        batch: Input batch data (dict or list of tensors).
        num_splits: Number of microbatches to split into.
        enforce_divisible_batch: Whether to enforce batch size being divisible by k.
        shuffle: Whether to shuffle the batch before splitting.
        shuffle_seed: Seed for reproducible shuffling.
    """
    if shuffle:
        g = torch.Generator()
        g.manual_seed(shuffle_seed)

        if isinstance(batch, (dict, UserDict)):
            tensor_items = {
                k: v for k, v in batch.items() if isinstance(v, torch.Tensor)
            }
            if tensor_items:
                batch_size = next(iter(tensor_items.values())).shape[0]
            else:
                list_items = {k: v for k, v in batch.items() if isinstance(v, list)}
                if not list_items:
                    raise ValueError(
                        "Batch contains no tensors or lists to determine batch size."
                    )
                batch_size = len(list_items[next(iter(list_items))])
        else:
            if not batch:
                raise ValueError("Batch is empty.")
            if torch.is_tensor(batch[0]):
                batch_size = batch[0].shape[0]
            elif isinstance(batch[0], list) and torch.is_tensor(batch[0][0]):
                batch_size = batch[0][0].shape[0]
            else:
                batch_size = len(batch[0])

        indices = torch.randperm(batch_size, generator=g).tolist()

        if isinstance(batch, (dict, UserDict)):
            for k, v in tensor_items.items():
                batch[k] = v[indices]

            list_items = {k: v for k, v in batch.items() if isinstance(v, list)}
            for k, v in list_items.items():
                batch[k] = [v[i] for i in indices]
        else:
            for i, item in enumerate(batch):
                if torch.is_tensor(item):
                    batch[i] = item[indices]
                elif isinstance(item, list):
                    if torch.is_tensor(item[0]):
                        batch[i] = [t[indices] for t in item]
                    else:
                        batch[i] = [item[idx] for idx in indices]
                elif item is None:
                    continue
                else:
                    raise ValueError(
                        f"Unsupported item type during shuffling: {type(item)}"
                    )

    if isinstance(batch, (dict, UserDict)):
        discard_items = [
            k for k, v in batch.items() if not isinstance(v, (torch.Tensor, list))
        ]
        if len(discard_items) > 0:
            logging.warning(
                f"Only support splitting torch.Tensor and List[torch.Tensor]. Discarding the following keys from the batch: {discard_items}",
            )

        batch = {k: v for k, v in batch.items() if isinstance(v, (torch.Tensor, list))}
        tensor_items = {k: v for k, v in batch.items() if isinstance(v, torch.Tensor)}
        list_items = {k: v for k, v in batch.items() if isinstance(v, list)}

        # Split tensor items
        items = list(tensor_items.items())
        if enforce_divisible_batch:
            assert items[0][1].shape[0] % num_splits == 0, (
                "Issue with batch size configuration!"
            )
        split_batch = [torch.tensor_split(item[1], num_splits, dim=0) for item in items]
        # handle the case where the batch size from dynamic bucketting is not divisible
        if items[0][1].shape[0] % num_splits != 0:
            chunk_size = split_batch[0][-1].shape[0]
            split_batch = [[j[:chunk_size] for j in i] for i in split_batch]

        if len(list_items) == 0:
            # Only have tensor items
            microbatches = [
                [(items[i][0], split_batch[i][j]) for i in range(len(items))]
                for j in range(num_splits)
            ]
        else:
            # Split list items
            list_items = list(list_items.items())
            split_list_batch = [
                split_list(
                    item[1],
                    num_splits,
                    enforce_divisible_batch=enforce_divisible_batch,
                )
                for item in list_items
            ]
            # Merge tensor and list items
            all_keys = [item[0] for item in items] + [item[0] for item in list_items]
            all_split_batch = split_batch + split_list_batch
            microbatches = [
                [(all_keys[i], all_split_batch[i][j]) for i in range(len(all_keys))]
                for j in range(num_splits)
            ]
        microbatches = [dict(elem) for elem in microbatches]
    else:
        # Split a list of torch tensors
        assert batch[0].shape[0] % num_splits == 0, (
            "Issue with batch size configuration!"
        )
        split_batch = []
        for item in batch:
            if torch.is_tensor(item):
                split_batch.append(torch.tensor_split(item, num_splits, dim=0))
            elif isinstance(item, list):
                if isinstance(item[0], torch.Tensor):
                    split_tensors = [
                        torch.tensor_split(elem, num_splits, dim=0) for elem in item
                    ]
                    split_tuple = []
                    for mbi in range(num_splits):
                        split_tuple.append(
                            [split_tensors[i][mbi] for i in range(len(split_tensors))]
                        )
                    split_tuple = tuple(split_tuple)
                    split_batch.append(split_tuple)
                else:
                    split_batch.append(split_list(item, num_splits))
            elif item is None:
                split_batch.append(item)
            else:
                raise ValueError(f"Unsupported item type: {type(item)}")

        microbatches = [
            [elem[i] if elem is not None else elem for elem in split_batch]
            for i in range(num_splits)
        ]

    return itertools.chain(microbatches)


def get_last_rank():
    return torch.distributed.get_world_size() - 1


def ceildiv(a, b):
    return -(a // -b)


def roundup_divisible(a, b):
    return ((a + b - 1) // b) * b


def karmarkar_karp(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    # see: https://en.wikipedia.org/wiki/Largest_differencing_method
    class Set:
        def __init__(self) -> None:
            self.sum = 0
            self.items = []

        def add(self, idx: int, val: int):
            self.items.append((idx, val))
            self.sum += val

        def merge(self, other):
            for idx, val in other.items:
                self.items.append((idx, val))
                self.sum += val

        def __lt__(self, other):
            if self.sum != other.sum:
                return self.sum < other.sum
            if len(self.items) != len(other.items):
                return len(self.items) < len(other.items)
            return self.items < other.items

    class State:
        def __init__(self, items: list[tuple[int, int]], k: int) -> None:
            self.k = k
            # sets should always be decreasing order
            self.sets = [Set() for _ in range(k)]
            assert len(items) in [1, k], f"{len(items)} not in [1, {k}]"
            for i, (idx, seqlen) in enumerate(items):
                self.sets[i].add(idx=idx, val=seqlen)
            self.sets = sorted(self.sets, reverse=True)

        def get_partitions(self):
            partitions = []
            for i in range(len(self.sets)):
                cur_partition = []
                for idx, _ in self.sets[i].items:
                    cur_partition.append(idx)
                partitions.append(cur_partition)
            return partitions

        def merge(self, other):
            for i in range(self.k):
                self.sets[i].merge(other.sets[self.k - 1 - i])
            self.sets = sorted(self.sets, reverse=True)

        @property
        def spread(self) -> int:
            return self.sets[0].sum - self.sets[-1].sum

        def __lt__(self, other):
            # least heap, let the state with largest spread to be popped first,
            # if the spread is the same, let the state who has the largest set
            # to be popped first.
            if self.spread != other.spread:
                return self.spread > other.spread
            return self.sets[0] > other.sets[0]

        def __repr__(self) -> str:
            repr_str = "["
            for i in range(self.k):
                if i > 0:
                    repr_str += ","
                repr_str += "{"
                for j, (_, seqlen) in enumerate(self.sets[i].items):
                    if j > 0:
                        repr_str += ","
                    repr_str += str(seqlen)
                repr_str += "}"
            repr_str += "]"
            return repr_str

    sorted_seqlen_list = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)])
    states_pq = []
    if equal_size:
        assert len(seqlen_list) % k_partitions == 0, (
            f"{len(seqlen_list)} % {k_partitions} != 0"
        )
        for offset in range(0, len(sorted_seqlen_list), k_partitions):
            items = []
            for i in range(k_partitions):
                seqlen, idx = sorted_seqlen_list[offset + i]
                items.append((idx, seqlen))
            heapq.heappush(states_pq, State(items=items, k=k_partitions))
    else:
        for seqlen, idx in sorted_seqlen_list:
            heapq.heappush(states_pq, State(items=[(idx, seqlen)], k=k_partitions))

    while len(states_pq) > 1:
        state0 = heapq.heappop(states_pq)
        state1 = heapq.heappop(states_pq)
        # merge states
        state0.merge(state1)
        heapq.heappush(states_pq, state0)

    final_state = states_pq[0]
    partitions = final_state.get_partitions()
    if equal_size:
        for i, partition in enumerate(partitions):
            assert len(partition) * k_partitions == len(seqlen_list), (
                f"{len(partition)} * {k_partitions} != {len(seqlen_list)}"
            )
    return partitions


def get_seqlen_balanced_partitions(
    seqlen_list: list[int], k_partitions: int, equal_size: bool
):
    """get order of seq lengths to make partitions balanced, this is
        used in balacing sum of seqlength across dp ranks and microbatches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    assert len(seqlen_list) >= k_partitions, (
        f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]"
    )

    def _check_and_sort_partitions(partitions):
        assert len(partitions) == k_partitions, f"{len(partitions)} != {k_partitions}"
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            assert len(partition) > 0, f"the {i}-th partition is empty"
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        assert seen_idx == set(range(len(seqlen_list)))
        return sorted_partitions

    partitions = karmarkar_karp(
        seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size
    )
    return _check_and_sort_partitions(partitions)


def get_seqlen_BFD_partitions(seq_len_list, max_tokens_per_mbs):
    import numpy as np

    seq_lengths = np.array(seq_len_list)

    n = len(seq_lengths)

    if np.any(seq_lengths > max_tokens_per_mbs):
        raise ValueError(
            f"Sequence length {np.max(seq_lengths)} exceeds the threshold {max_tokens_per_mbs}"
        )

    indexed_lengths = [(seq_lengths[i], i) for i in range(n)]
    indexed_lengths.sort(key=lambda x: x[0], reverse=True)

    partitions = []
    group_remaining_capacity = []

    # Best Fit Decreasing
    for seq_len, original_idx in indexed_lengths:
        best_group_idx = -1
        min_remaining_after_fit = float("inf")

        for group_idx in range(len(partitions)):
            remaining = group_remaining_capacity[group_idx]
            if remaining >= seq_len:
                remaining_after_fit = remaining - seq_len
                if remaining_after_fit < min_remaining_after_fit:
                    min_remaining_after_fit = remaining_after_fit
                    best_group_idx = group_idx

        if best_group_idx != -1:
            partitions[best_group_idx].append(original_idx)
            group_remaining_capacity[best_group_idx] -= seq_len
        else:
            partitions.append([original_idx])
            group_remaining_capacity.append(max_tokens_per_mbs - seq_len)

    for i, group in enumerate(partitions):
        group_total = sum(seq_lengths[idx] for idx in group)
        assert group_total <= max_tokens_per_mbs, (
            f"Group {i} total length {group_total} exceeds the threshold {max_tokens_per_mbs}"
        )

    all_assigned_indices = set()
    for group in partitions:
        for idx in group:
            assert idx not in all_assigned_indices, (
                f"Index {idx} is assigned repeatedly"
            )
            all_assigned_indices.add(idx)
    assert len(all_assigned_indices) == n, (
        f"Number of assigned indices {len(all_assigned_indices)} does not equal number of original sequences {n}"
    )

    return partitions


def get_iterator_dynamic(
    batch: Union[dict, list[torch.Tensor]],
    max_tokens_per_mbs: Optional[int] = None,
    dp_group=None,
    num_batches_divided_by=None,
    same_micro_num_in_dp=True,
    min_num_micro_batch=None,
) -> Iterator:
    """
    Split a batch into microbatches based on max token length or fixed number

    Args:
        batch: Input batch as dict or list of tensors
        max_tokens_per_mbs: Max sum of attention_mask per micro-batch
        dp_group: Data parallel group for synchronizing micro-batch numbers
        num_batches_divided_by: Ensure number of micro-batches is divisible by this number
        same_micro_num_in_dp: Whether to synchronize micro-batch numbers across data parallel ranks
        min_num_micro_batch: Minimum number of micro-batches to create
    """
    if isinstance(batch, (dict, UserDict)):
        # Get effective sequence length of each sample
        seq_len_effective = batch["attention_mask"].sum(dim=1)
        max_seq_len = batch["attention_mask"].shape[-1]

        # Validate max_tokens_per_mbs
        assert max_tokens_per_mbs >= max_seq_len, (
            f"max_tokens_per_mbs must be greater than sequence length. Got {max_tokens_per_mbs=} and {max_seq_len=}"
        )

        # Compute total token count and the actual number of microbatches needed
        seq_len_list = seq_len_effective.tolist()
        num_micro_batches = len(
            get_seqlen_BFD_partitions(seq_len_list, max_tokens_per_mbs)
        )

        if min_num_micro_batch is not None:
            # used to support pp
            num_micro_batches = max(min_num_micro_batch, num_micro_batches)
        if dist.is_initialized() and same_micro_num_in_dp:
            num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
            dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
            num_micro_batches = num_micro_batches.cpu().item()
        if num_batches_divided_by is not None:
            num_micro_batches = roundup_divisible(
                num_micro_batches, num_batches_divided_by
            )

        # print(f"num_microbatches: {num_micro_batches}")
        # Use get_seqlen_balanced_partitions for partitioning
        partitions = get_seqlen_balanced_partitions(
            seqlen_list=seq_len_list, k_partitions=num_micro_batches, equal_size=False
        )

        # Create microbatches
        microbatches = []
        for partition in partitions:
            curr_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    curr_batch[key] = torch.stack([value[idx] for idx in partition])
                elif isinstance(value, list):
                    curr_batch[key] = [value[idx] for idx in partition]
                else:
                    continue
            microbatches.append(curr_batch)

    else:
        # Handle list of tensors
        if not batch:
            return itertools.chain([])

        # Check if attention_mask exists and calculate sequence lengths
        attention_mask = batch[1]
        seq_len_effective = attention_mask.sum(dim=1)
        max_seq_len = attention_mask.shape[-1]

        assert max_tokens_per_mbs >= max_seq_len, (
            f"max_tokens_per_mbs must be greater than sequence length. Got {max_tokens_per_mbs=} and {max_seq_len=}"
        )

        seq_len_list = seq_len_effective.tolist()
        num_micro_batches = len(
            get_seqlen_BFD_partitions(seq_len_list, max_tokens_per_mbs)
        )
        # num_micro_batches = min(len(seq_len_effective), ceildiv(total_seqlen, max_tokens_per_mbs))
        if min_num_micro_batch is not None:
            # used to support pp
            num_micro_batches = max(min_num_micro_batch, num_micro_batches)
        if dist.is_initialized() and same_micro_num_in_dp:
            num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
            dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
            num_micro_batches = num_micro_batches.cpu().item()
        if num_batches_divided_by is not None:
            num_micro_batches = roundup_divisible(
                num_micro_batches, num_batches_divided_by
            )

        seq_len_list = seq_len_effective.tolist()

        partitions = get_seqlen_balanced_partitions(
            seqlen_list=seq_len_list, k_partitions=num_micro_batches, equal_size=False
        )

        microbatches = []
        for partition in partitions:
            curr_batch = []
            for item in batch:
                if torch.is_tensor(item):
                    curr_batch.append(torch.stack([item[idx] for idx in partition]))
                elif isinstance(item, list):
                    if isinstance(item[0], torch.Tensor):
                        curr_batch.append([item[idx] for idx in partition])
                    else:
                        curr_batch.append([item[idx] for idx in partition])
                elif item is None:
                    curr_batch.append(None)
                else:
                    raise ValueError(f"Unsupported item type: {type(item)}")
            microbatches.append(curr_batch)
    n_micro_batch = len(microbatches)
    return itertools.chain(microbatches), partitions, n_micro_batch


def split_dynamic_batch_size(
    batch: dict[str, torch.Tensor],
    cp_world_size: int,
    vpp_world_size: int,
    max_tokens_per_mbs: int,
    microbatch_group_size_per_vp_stage: int,
):
    """Split a global batch using dynamic batch sizing."""
    max_tokens_per_mbs = max_tokens_per_mbs * cp_world_size
    vpp_size = vpp_world_size
    if vpp_size is not None and vpp_size > 1:
        microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage
        data_iter, indices, n_micro_batch = get_iterator_dynamic(
            batch,
            num_batches_divided_by=microbatch_group_size_per_vp_stage,
            max_tokens_per_mbs=max_tokens_per_mbs,
        )
        assert n_micro_batch % microbatch_group_size_per_vp_stage == 0, (
            f"micro_batches {data_iter} must be divisible by microbatch_group_size_per_vp_stage {microbatch_group_size_per_vp_stage} for megatron backend"
        )
    else:
        data_iter, indices, n_micro_batch = get_iterator_dynamic(
            batch, max_tokens_per_mbs=max_tokens_per_mbs
        )
    total_seqlen = max_tokens_per_mbs
    return data_iter, total_seqlen, n_micro_batch, indices


def get_reverse_idx(idx_map):
    """
    Build the inverse of an index mapping.

    Args:
        idx_map (Sequence[int]): Sequence where idx_map[i] = j.

    Returns:
        List[int]: Inverse mapping list such that output[j] = i for each i.
    """
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
