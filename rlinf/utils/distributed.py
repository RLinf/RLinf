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

from collections import UserDict
from contextlib import contextmanager
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np
import torch
import torch.distributed

try:
    from megatron.core import parallel_state
except ImportError:
    parallel_state = None  # type: ignore
from torch.distributed import ProcessGroup, ReduceOp
from typing_extensions import Self

from rlinf.utils.timers import NamedTimer


def compute_rollout_metrics(
    rollout_batch: dict[str, torch.Tensor],
    max_prompt_len: int,
    response_len: int,
    data_parallel_group: Optional[ProcessGroup] = None,
    use_critic: bool = False,
):
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    advantages = rollout_batch["advantages"].to(device=device)
    mask = rollout_batch["response_mask"][:, -response_len:].to(device=device)
    prompt_lengths = rollout_batch["prompt_lengths"].clone().to(device=device)
    response_lengths = rollout_batch["response_lengths"].clone().to(device=device)
    reward_scores = rollout_batch["rewards"].clone().to(device=device)
    is_end = rollout_batch["is_end"].clone().float().to(device=device)

    dp_world_size = torch.distributed.get_world_size(data_parallel_group)

    prompt_lengths_list: list[list[int]] = [None for _ in range(dp_world_size)]
    decode_lengths_list: list[list[int]] = [None for _ in range(dp_world_size)]
    torch.distributed.all_gather_object(
        prompt_lengths_list,
        prompt_lengths.tolist(),
        group=data_parallel_group,
    )
    torch.distributed.all_gather_object(
        decode_lengths_list,
        response_lengths.tolist(),
        group=data_parallel_group,
    )

    total_prompt_lengths = torch.tensor(sum(prompt_lengths_list, []), device=device)
    total_decode_lengths = torch.tensor(sum(decode_lengths_list, []), device=device)

    sum_plen = prompt_lengths.sum().detach().item()
    sum_rlen = response_lengths.sum().detach().item()
    sum_rewards = reward_scores.sum().detach().item()
    sum_end = is_end.sum().detach().item()

    valid_adv = torch.masked_select(advantages, mask)
    n_valid_token = mask.sum().detach().item()
    sum_adv = valid_adv.to(torch.float64).sum().detach().item()

    num_seq = prompt_lengths.numel()
    reduce_metrics = torch.as_tensor(
        [sum_plen, sum_rlen, sum_rewards, sum_end, sum_adv, num_seq, n_valid_token],
        device=device,
        dtype=torch.float32,
    )

    torch.distributed.all_reduce(
        reduce_metrics,
        torch.distributed.ReduceOp.SUM,
        group=data_parallel_group,
    )

    sum_plen, sum_rlen, sum_rewards, sum_end, sum_adv, num_seq, n_valid_token = (
        reduce_metrics.tolist()
    )

    adv_max = torch.max(valid_adv).detach().item()
    adv_min = torch.min(valid_adv).detach().item()
    reduce_tensor = torch.as_tensor(
        [-adv_min, adv_max], device=torch.cuda.current_device(), dtype=torch.float32
    )
    torch.distributed.all_reduce(
        reduce_tensor,
        torch.distributed.ReduceOp.MAX,
        group=data_parallel_group,
    )
    adv_min, adv_max = reduce_tensor.tolist()

    rollout_metrics = {
        "total_num_sequence": num_seq,
        "prompt_length": sum_plen / num_seq,
        "response_length": sum_rlen / num_seq,
        "total_length": (sum_plen + sum_rlen) / num_seq,
        "reward_scores": sum_rewards / num_seq,
        "fraction_of_samples_properly_ended": sum_end / num_seq,
        "advantages_mean": sum_adv / n_valid_token,
        "advantages_max": adv_max,
        "advantages_min": -adv_min,
    }
    return rollout_metrics, total_prompt_lengths, total_decode_lengths


class RolloutDataBalance(UserDict):
    def __init__(
        self,
        dictionary_data: Optional[dict[str, torch.Tensor]] = None,
        ordered_keys_hint: Optional[list[str]] = None,
    ):
        super().__init__(dictionary_data if dictionary_data is not None else {})

        if ordered_keys_hint and self.data:
            self._ordered_keys = [k for k in ordered_keys_hint if k in self.data]
            if len(self._ordered_keys) != len(self.data) or not all(
                k in self.data for k in self._ordered_keys
            ):
                self._ordered_keys = sorted(self.data.keys())
        elif self.data:
            self._ordered_keys = sorted(self.data.keys())
        else:
            self._ordered_keys = []

    def __getitem__(self, key: Any) -> torch.Tensor:
        if isinstance(key, int):
            if not self._ordered_keys:
                raise IndexError(
                    f"RolloutDataBalance is empty or has no ordered keys for integer indexing. Data keys: {list(self.data.keys())}"
                )
            if 0 <= key < len(self._ordered_keys):
                actual_key = self._ordered_keys[key]
                if actual_key not in self.data:
                    raise KeyError(
                        f"Internal error: Key '{actual_key}' (from index {key}) not in data. Ordered: {self._ordered_keys}. Data: {list(self.data.keys())}"
                    )
                return self.data[actual_key]
            else:
                raise IndexError(
                    f"Integer index {key} out of range for {len(self._ordered_keys)} ordered keys. Ordered: {self._ordered_keys}"
                )
        return super().__getitem__(key)

    @classmethod
    def from_rollout_batches(
        cls: Self,
        rollout_batches: dict[str, torch.Tensor],
        dp_world_size: int,
        dp_rank: int,
        dp_group: Optional[ProcessGroup],
        partitioning_tool: Callable,
    ) -> Self:
        current_device = torch.cuda.current_device()

        attn_mask = rollout_batches.get("attention_mask")
        current_num_samples = attn_mask.size(0)

        # 2. Calculate local sample token counts
        local_token_counts = torch.zeros(
            current_num_samples, dtype=torch.int, device=current_device
        )
        if current_num_samples > 0 and "attention_mask" in rollout_batches:
            attn_mask = rollout_batches["attention_mask"]
            if (
                isinstance(attn_mask, torch.Tensor)
                and attn_mask.size(0) == current_num_samples
            ):
                local_token_counts = attn_mask.sum(dim=1).int()

        # 3. Gather global information: sample counts from each rank
        num_samples_tensor = torch.tensor(
            current_num_samples, device=current_device, dtype=torch.long
        )
        all_num_samples_t = [
            torch.empty_like(num_samples_tensor) for _ in range(dp_world_size)
        ]
        if dp_group and dp_world_size > 1:
            torch.distributed.all_gather(
                all_num_samples_t, num_samples_tensor, group=dp_group
            )
        else:
            all_num_samples_t = [num_samples_tensor]
        all_num_samples = [s.item() for s in all_num_samples_t]
        global_total_samples = sum(all_num_samples)
        max_samples_rank = max(all_num_samples) if global_total_samples > 0 else 0

        # 4. Gather global token counts for all samples
        global_token_counts_list: list[int] = []
        all_ranks_local_token_counts_list: list[list[int]] = [
            [] for _ in range(dp_world_size)
        ]

        if global_total_samples > 0:
            padded_local_tokens = torch.zeros(
                max_samples_rank, dtype=torch.int, device=current_device
            )
            if local_token_counts.numel() > 0:
                padded_local_tokens[: local_token_counts.size(0)] = local_token_counts

            all_padded_tokens_t = [
                torch.empty_like(padded_local_tokens) for _ in range(dp_world_size)
            ]
            if dp_group and dp_world_size > 1:
                torch.distributed.all_gather(
                    all_padded_tokens_t, padded_local_tokens, group=dp_group
                )
            else:
                all_padded_tokens_t = [padded_local_tokens]

            for i_rank in range(dp_world_size):
                num_s_rank = all_num_samples[i_rank]
                if num_s_rank > 0:
                    rank_tokens = all_padded_tokens_t[i_rank][:num_s_rank].tolist()
                    global_token_counts_list.extend(rank_tokens)
                    all_ranks_local_token_counts_list[i_rank] = rank_tokens

        # 5. Calculate global sample indices assigned to current rank
        my_assigned_global_indices: list[int] = []
        all_ranks_assigned_tokens_after_balance: list[int] = [
            0
        ] * dp_world_size  # For rank 0 to print summary

        if global_total_samples > 0:
            if not global_token_counts_list:
                global_token_counts_list = [1] * global_total_samples

            k_partitions = min(global_total_samples, dp_world_size)
            if k_partitions > 0 and len(global_token_counts_list) >= k_partitions:
                partitions_indices_all_ranks = partitioning_tool(
                    seqlen_list=global_token_counts_list,
                    k_partitions=k_partitions,
                    equal_size=True,
                )

                if dp_rank < k_partitions and dp_rank < len(
                    partitions_indices_all_ranks
                ):
                    my_assigned_global_indices = partitions_indices_all_ranks[dp_rank]

                if dp_group and dp_world_size > 1:
                    if dp_rank == 0:
                        for r_idx in range(k_partitions):
                            if r_idx < len(partitions_indices_all_ranks):
                                rank_indices = partitions_indices_all_ranks[r_idx]
                                all_ranks_assigned_tokens_after_balance[r_idx] = sum(
                                    global_token_counts_list[g_idx]
                                    for g_idx in rank_indices
                                )

        # 6. Get superset of all keys that appear on all DP ranks and sort them
        local_keys = set(rollout_batches.keys())
        all_keys_sets: list[Optional[set[str]]] = [None] * dp_world_size
        if dp_group and dp_world_size > 1:
            torch.distributed.all_gather_object(
                all_keys_sets, local_keys, group=dp_group
            )
        else:
            all_keys_sets = [local_keys]

        superset_keys = set().union(*(s for s in all_keys_sets if s is not None))
        final_ordered_keys = sorted(superset_keys)

        # 7. Gather all data from all ranks (CPU)
        payload_cpu = {
            k: v.cpu()
            for k, v in rollout_batches.items()
            if k in final_ordered_keys and isinstance(v, torch.Tensor)
        }
        all_payloads_cpu: list[Optional[dict[str, torch.Tensor]]] = [
            None
        ] * dp_world_size
        if dp_group and dp_world_size > 1:
            torch.distributed.all_gather_object(
                all_payloads_cpu, payload_cpu, group=dp_group
            )
        else:
            all_payloads_cpu = [payload_cpu]

        # 8. Rebuild global batch on CPU and record template specifications
        global_batch_cpu: dict[str, torch.Tensor] = {}
        template_specs: dict[str, dict[str, Any]] = {}
        if global_total_samples > 0:
            for key in final_ordered_keys:
                tensors_for_key = []
                for i_rank, rank_payload in enumerate(all_payloads_cpu):
                    if isinstance(rank_payload, dict) and all_num_samples[i_rank] > 0:
                        tensor = rank_payload.get(key)
                        if (
                            isinstance(tensor, torch.Tensor)
                            and tensor.numel() > 0
                            and tensor.size(0) == all_num_samples[i_rank]
                        ):
                            tensors_for_key.append(tensor)
                            if (
                                key not in template_specs
                            ):  # Store spec from first valid tensor
                                template_specs[key] = {
                                    "dtype": tensor.dtype,
                                    "shape_suffix": list(tensor.shape[1:]),
                                }

                if tensors_for_key:
                    try:
                        cat_tensor = torch.cat(tensors_for_key, dim=0)
                        global_batch_cpu[key] = cat_tensor
                        if (
                            key not in template_specs and cat_tensor.numel() > 0
                        ):  # Update spec if first was empty
                            template_specs[key] = {
                                "dtype": cat_tensor.dtype,
                                "shape_suffix": list(cat_tensor.shape[1:]),
                            }
                    except Exception:
                        pass

        # 9. Select data for current rank
        final_rank_data: dict[str, torch.Tensor] = {}

        def _create_empty_tensor_for_key(
            k: str, specs: dict[str, dict[str, Any]], dev: torch.device
        ) -> torch.Tensor:
            spec = specs.get(k)
            if spec:
                return torch.empty(
                    [0] + spec["shape_suffix"], dtype=spec["dtype"], device=dev
                )
            return torch.empty(0, dtype=torch.float32, device=dev)

        if my_assigned_global_indices:
            indices_cpu = torch.tensor(my_assigned_global_indices, dtype=torch.long)
            for key in final_ordered_keys:
                full_tensor = global_batch_cpu.get(key)
                if (
                    isinstance(full_tensor, torch.Tensor)
                    and full_tensor.numel() > 0
                    and full_tensor.size(0) == global_total_samples
                ):
                    try:
                        final_rank_data[key] = full_tensor.index_select(
                            0, indices_cpu
                        ).to(current_device)
                    except IndexError:
                        final_rank_data[key] = _create_empty_tensor_for_key(
                            key, template_specs, current_device
                        )

        return cls(final_rank_data, ordered_keys_hint=final_ordered_keys)

    def gather_and_balance_globally(self):
        global_rollout_batch = type(self)()

        for k, tensor in self.data.items():
            tensor = rebalance_nd_tensor(
                tensor, group=parallel_state.get_data_parallel_group()
            )
            global_rollout_batch[k] = tensor

        return global_rollout_batch

    def chunk(self, rank, split_size):
        chunked_rollout_batch = type(self)()

        batch_set = {tensor.size(0) for tensor in self.data.values()}
        assert len(batch_set) == 1, (
            "batch sizes are not the same across the rollout batch"
        )
        B = batch_set.pop()

        indices = torch.arange(B).tensor_split(split_size)[rank]

        for k in self.data:
            chunked_rollout_batch[k] = self.data[k][indices].clone()

        return chunked_rollout_batch


def rebalance_nd_tensor(tensor, group):
    """
    Takes tensors with variable leading sizes (at dim=0) and then stack them into a single tensor.

    NOTE: assumes all other (i.e., non-zero) dimensions are equal.
    """
    num_samples = torch.as_tensor(
        tensor.size(0), dtype=torch.int64, device=torch.cuda.current_device()
    )
    batch_num_per_rank = torch.zeros(
        torch.distributed.get_world_size(group),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_gather_into_tensor(
        batch_num_per_rank, num_samples, group=group
    )

    B = batch_num_per_rank.sum()
    other_dims = tensor.shape[1:]

    indices = batch_num_per_rank.cumsum(dim=0)
    output_tensor = torch.zeros(
        B, *other_dims, dtype=tensor.dtype, device=torch.cuda.current_device()
    )

    # tensor_split is a view we can copy into
    output_tensor.tensor_split(indices[0:-1].cpu())[
        torch.distributed.get_rank(group=group)
    ].copy_(tensor)
    torch.distributed.all_reduce(output_tensor, group=group)
    return output_tensor


def broadcast_tensor(
    tensor: Union[torch.Tensor, None],
    src,
    group,
    dtype: Union[torch.dtype, None] = None,
):
    """
    Broadcast a tensor from the source rank to every other rank in the given group.
    All the ranks that send or receive data must call this function.

    Parameters:
    - tensor: The tensor to be broadcasted (or None for non source ranks).
    - src: The rank of the source tensor.
    - group: The process group to use for the broadcast.
    - dtype: (Optional) The desired data type to cast the tensor before broadcasting.

    Returns:
    - The broadcasted tensor.
    """

    if torch.distributed.get_rank() == src:
        tensor = tensor.cuda()
        if dtype:
            tensor = tensor.to(dtype)

        metadata = [tensor.dtype, tensor.shape]

        torch.distributed.broadcast_object_list(metadata, src, group)
        torch.distributed.broadcast(tensor, src, group)
    else:
        metadata = [None, None]
        torch.distributed.broadcast_object_list(metadata, src, group)

        dtype, input_shape = metadata
        tensor = torch.empty(input_shape, dtype=dtype, device="cuda")
        torch.distributed.broadcast(tensor, src, group)
    return tensor


def broadcast_tensor_within_mp(tensor, dtype=torch.float32):
    """helper function to broadcast within the model parallel group"""
    group = parallel_state.get_model_parallel_group()

    if torch.distributed.get_world_size(group) > 1:
        return broadcast_tensor(
            tensor, parallel_state.get_model_parallel_src_rank(), group, dtype=dtype
        )

    return tensor


def broadcast_tensor_within_pp(
    tensor: Union[torch.Tensor, None], dtype: torch.dtype = None, from_last: bool = True
):
    """
    tensor: Should be a valid tensor on src rank and None elsewhere
    dtype: no dtype means that the dtype is inferred
    from_last: True=broadcast from the last PP rank and False=broadcast from first PP rank (default=True)
    """
    if parallel_state.get_pipeline_model_parallel_world_size() > 1:
        return broadcast_tensor(
            tensor,
            parallel_state.get_pipeline_model_parallel_last_rank()
            if from_last
            else parallel_state.get_pipeline_model_parallel_first_rank(),
            parallel_state.get_pipeline_model_parallel_group(),
            dtype=dtype,
        )

    return tensor


def broadcast_tensor_within_dp(tensor: torch.Tensor, dtype: torch.dtype):
    if parallel_state.get_tensor_model_parallel_world_size() > 1:
        tensor = broadcast_tensor(
            tensor,
            parallel_state.get_tensor_model_parallel_src_rank(),
            parallel_state.get_tensor_model_parallel_group(),
            dtype,
        )
    return tensor


def gather_tensor(tensor, dst, group, dtype=None):
    """Gather any tensor to the dst rank from every other rank in the given group.
    All the ranks that send or receive data must call this function."""
    tensor = tensor.to(device=torch.cuda.current_device(), dtype=dtype)
    if torch.distributed.get_rank() == dst:
        gather_list = [
            torch.empty_like(tensor)
            for _ in range(torch.distributed.get_world_size(group))
        ]
    else:
        gather_list = None

    torch.distributed.gather(tensor, gather_list=gather_list, dst=dst, group=group)
    return gather_list


def all_reduce_int(
    obj: int,
    op: ReduceOp = ReduceOp.MIN,
    group: ProcessGroup = None,
):
    obj_tensor = torch.tensor(
        [obj], dtype=torch.long, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(
        obj_tensor,
        op,
        group=group,
    )
    return obj_tensor.item()


def run_if_model_parallel_src(fn, *fn_args, **fn_kwargs):
    """This function is meant to wrap an arbitary function to only call the function
    if it's the model parallel src. So if we have DP=2, this function will be called
    only twice."""
    src_rank = parallel_state.get_model_parallel_src_rank()

    output = None
    if torch.distributed.get_rank() == src_rank:
        output = fn(*fn_args, **fn_kwargs)

    return output


def normalize_tensor(tensor, mask, group=None):
    """normalizes a tensor using global mean and std"""
    dtype = torch.float64
    tensor = tensor.to(dtype)
    tensor = tensor.to(device=torch.cuda.current_device())
    mask = mask.to(device=torch.cuda.current_device())

    tensor_global_mean, tensor_global_var = masked_global_mean_var(
        tensor, mask, group=group
    )
    tensor = (tensor - tensor_global_mean) * torch.rsqrt(tensor_global_var + 1e-5)
    return tensor.float()


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    dim: Optional[int | tuple[int, ...]] = None,
    inplace: Optional[bool] = False,
    unbiased: Optional[bool] = False,
    eps: Optional[float] = 1e-5,
    high_precision: Optional[bool] = True,
    all_reduce: Optional[bool] = True,
    group: Optional[ProcessGroup] = None,
):
    """Normalize x with a mask. Typically used in advantage normalization.

    Args:
        x (torch.Tensor):
            Tensor to be normalized.
        mask (torch.Tensor, optional):
            A mask with the same shape as x. Defaults to None.
        dim (int or tuple of ints, optional):
            Dimensions to be normalized. Defaults to None.
        inplace (bool, optional):
            Whether to perform in-place operation. Defaults to False.
        eps (torch.Tensor, optional):
            Minimal denominator. Defaults to 1e-5.

    Returns:
        torch.Tensor:
            Normalized x, with the same shape as x.
    """
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype=dtype).cuda()
    if not inplace:
        x = x.clone()
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype=dtype).cuda()
        assert len(mask.shape) == len(x.shape), (mask.shape, x.shape, dim)
        for i in range(len(x.shape)):
            if i in dim:
                assert mask.shape[i] == x.shape[i], (mask.shape, x.shape, dim)
            else:
                assert mask.shape[i] == 1, (mask.shape, x.shape, dim)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)

    if torch.distributed.is_initialized() and all_reduce:
        torch.distributed.all_reduce(
            factor,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        torch.distributed.all_reduce(
            x_sum,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
        torch.distributed.all_reduce(
            x_sum_sq,
            op=torch.distributed.ReduceOp.SUM,
            group=group,
        )
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def masked_global_mean_var(values, mask, group=None):
    """computes the global mean and var when there is a mask

    NOTE: the variance here is uncorrected

    mask and values must have same shape, with mask being {0,1} with 1 being the values we want to keep
    """
    assert values.shape == mask.shape, (values.shape, mask.shape)
    values = values.to(device=torch.cuda.current_device())
    mask = mask.to(device=torch.cuda.current_device())

    values = values * mask

    # Get global sum and count and calculate the global mean and variance
    sum_and_count = torch.tensor(
        [values.sum(), mask.sum()],
        dtype=torch.float64,
        device=torch.cuda.current_device(),
    )
    torch.distributed.all_reduce(sum_and_count, group=group)
    global_sum, global_count = sum_and_count
    global_mean = global_sum / global_count
    variance_summed = (
        (((values - global_mean) ** 2) * mask)
        .sum()
        .to(device=torch.cuda.current_device(), dtype=torch.float64)
    )

    torch.distributed.all_reduce(variance_summed, group=group)

    return global_mean, variance_summed / global_count


def report_device_info(info_str):
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    free_gpu_memory /= 2**30
    total_gpu_memory /= 2**30

    memory_allocated = torch.cuda.memory_allocated() / 2**30
    memory_reserved = torch.cuda.memory_reserved() / 2**30

    print(
        f"[Rank {torch.distributed.get_rank()}] {info_str}, {free_gpu_memory=:.2f} GiB, {total_gpu_memory=:.2f} GiB, {memory_allocated=:.2f} GiB, {memory_reserved=:.2f} GiB"
    )


def ensure_divisibility(numerator, denominator):
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(
        numerator, denominator
    )


def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return the
    division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


class VocabUtility:
    """Split the vocabulary into `world_size` chunks and return the first
    and last index of the vocabulary belonging to the `rank`
    partition: Note that indices in [fist, last)

    """

    @staticmethod
    def vocab_range_from_per_partition_vocab_size(
        per_partition_vocab_size: int, rank, world_size: int
    ) -> Sequence[int]:
        index_f = rank * per_partition_vocab_size
        index_l = index_f + per_partition_vocab_size
        return index_f, index_l

    @staticmethod
    def vocab_range_from_global_vocab_size(
        global_vocab_size: int, rank: int, world_size: int
    ) -> Sequence[int]:
        per_partition_vocab_size = divide(global_vocab_size, world_size)
        return VocabUtility.vocab_range_from_per_partition_vocab_size(
            per_partition_vocab_size, rank, world_size
        )


def all_reduce_dict(
    dictionary, dtype=torch.float32, group=None, op=torch.distributed.ReduceOp.SUM
):
    keys = sorted(dictionary)
    tensor = torch.as_tensor(
        [dictionary[k] for k in keys], dtype=dtype, device=torch.cuda.current_device()
    )
    torch.distributed.all_reduce(tensor, op=op, group=group)
    return dict(zip(keys, tensor.tolist()))


class _VocabParallelEntropyAndCrossEntropy(torch.autograd.Function):
    """Compute entropy and cross-entropy (corresponding to log-probs) in a single forward pass. Returns (entropy, ce_loss)."""

    @staticmethod
    def forward(
        ctx,
        vocab_parallel_logits: torch.Tensor,
        target: torch.Tensor,
        label_smoothing: float = 0.0,
        calculate_entropy_loss: bool = True,
    ):
        """Forward pass: returns two tensors — entropy and cross-entropy loss"""
        vocab_parallel_logits = vocab_parallel_logits.float()
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        norm_logits = vocab_parallel_logits - logits_max
        exp_logits = norm_logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1, keepdim=True)

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_tensor_model_parallel_group(),
        )

        softmax = exp_logits.div_(sum_exp_logits)
        sum_exp_logits_log = sum_exp_logits.log()

        entropy = torch.zeros_like(logits_max.squeeze(-1))

        sum_softmax_times_logits = (softmax * vocab_parallel_logits).sum(
            dim=-1, keepdim=True
        )
        torch.distributed.all_reduce(
            sum_softmax_times_logits,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        entropy = (
            logits_max.squeeze(-1)
            + sum_exp_logits_log.squeeze(-1)
            - sum_softmax_times_logits.squeeze(-1)
        )

        partition_vocab_size = norm_logits.size(-1)
        rank = parallel_state.get_tensor_model_parallel_rank()
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        vocab_start, vocab_end = VocabUtility.vocab_range_from_per_partition_vocab_size(
            partition_vocab_size, rank, world_size
        )

        target_mask = (target < vocab_start) | (target >= vocab_end)
        masked_target = target.clone() - vocab_start
        masked_target[target_mask] = 0

        logits_2d = norm_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(logits_2d.size(0), device=logits_2d.device)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits = predicted_logits_1d.view_as(target)
        predicted_logits[target_mask] = 0.0

        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_tensor_model_parallel_group(),
        )
        ce_loss = sum_exp_logits_log.squeeze(dim=-1) - predicted_logits

        if label_smoothing > 0:
            smoothing = (
                label_smoothing * partition_vocab_size / (partition_vocab_size - 1)
            )
            mean_log_probs = softmax.log().mean(dim=-1)
            ce_loss = (1.0 - smoothing) * ce_loss - smoothing * mean_log_probs

        calculate_entropy_tensor = torch.tensor(
            calculate_entropy_loss,
            dtype=torch.bool,
            device=vocab_parallel_logits.device,
        )
        ctx.label_smoothing = label_smoothing
        ctx.vocab_size = partition_vocab_size

        if calculate_entropy_loss:
            ctx.save_for_backward(
                softmax,
                norm_logits,
                sum_softmax_times_logits,
                target_mask,
                masked_target_1d,
                calculate_entropy_tensor,
            )
        else:
            ctx.save_for_backward(
                softmax,
                None,
                None,
                target_mask,
                masked_target_1d,
                calculate_entropy_tensor,
            )

        return entropy, ce_loss

    @staticmethod
    def backward(ctx, grad_entropy, grad_ce):
        saved_tensors = ctx.saved_tensors
        (
            softmax,
            norm_logits,
            sum_softmax_times_logits,
            target_mask,
            masked_target_1d,
            calculate_entropy_tensor,
        ) = saved_tensors

        label_smoothing = ctx.label_smoothing
        vocab_size = ctx.vocab_size
        calculate_entropy_loss = calculate_entropy_tensor.item()

        grad_input = torch.zeros_like(softmax)
        if grad_ce is not None:
            from megatron.core.tensor_parallel.cross_entropy import (
                VocabParallelCrossEntropy as _CEUtils,
            )

            grad_2d, arange_1d, softmax_update, ce_grad_input = (
                _CEUtils.prepare_gradient_calculation_operands(softmax, target_mask)
            )

            if label_smoothing > 0:
                smoothing = label_smoothing * vocab_size / (vocab_size - 1)
                grad_2d[arange_1d, masked_target_1d] -= (
                    1.0 - smoothing
                ) * softmax_update
                grad_2d[arange_1d, :] -= smoothing / vocab_size
                grad_input += ce_grad_input * grad_ce.unsqueeze(dim=-1)
            else:
                ce_grad_result = _CEUtils.calculate_gradients(
                    grad_2d,
                    arange_1d,
                    masked_target_1d,
                    softmax_update,
                    ce_grad_input,
                    grad_ce,
                )
                grad_input += ce_grad_result

        # Memory optimized entropy gradient with chunking
        if (
            calculate_entropy_loss
            and grad_entropy is not None
            and norm_logits is not None
            and sum_softmax_times_logits is not None
        ):
            batch_size, seq_len, vocab_size = softmax.shape
            chunk_size = min(128, seq_len)

            for start_idx in range(0, seq_len, chunk_size):
                end_idx = min(start_idx + chunk_size, seq_len)

                softmax_chunk = softmax[:, start_idx:end_idx, :]
                norm_logits_chunk = norm_logits[:, start_idx:end_idx, :]
                sum_softmax_times_logits_chunk = sum_softmax_times_logits[
                    :, start_idx:end_idx, :
                ]
                grad_entropy_chunk = grad_entropy[:, start_idx:end_idx]

                grad_entropy_expanded = grad_entropy_chunk.unsqueeze(dim=-1)
                sum_expanded = sum_softmax_times_logits_chunk
                grad_entropy_input_chunk = (
                    grad_entropy_expanded
                    * softmax_chunk
                    * (sum_expanded - norm_logits_chunk)
                )
                grad_input[:, start_idx:end_idx, :] += grad_entropy_input_chunk

        return grad_input, None, None, None


def vocab_parallel_entropy_and_log_probs(
    vocab_parallel_logits: torch.Tensor,
    target: torch.Tensor,
    label_smoothing: float = 0.0,
    calculate_entropy_loss: bool = True,
):
    """Perform a single forward pass to obtain entropy and log_probs (-cross_entropy)"""
    entropy, ce_loss = _VocabParallelEntropyAndCrossEntropy.apply(
        vocab_parallel_logits, target, label_smoothing, calculate_entropy_loss
    )
    log_probs = -ce_loss
    return entropy, log_probs


def vocab_parallel_log_probs_from_logits(logits, labels):
    from megatron.core import tensor_parallel

    return -tensor_parallel.vocab_parallel_cross_entropy(
        vocab_parallel_logits=logits, target=labels
    )


class ScopedTimer:
    """
    A thin adapter over the NamedTimer class to help time sections of code
    using a context manager.

    This class is useful for tracking timings automatically so you don't need
    to manually collect them. You only need to pass the timer around and can
    collect the durations in one place, instead of returning and mutating
    dictionaries throughout your code.

    The ScopedTimer ensures that durations are logged and consumed properly,
    preventing accidental overwriting of previous measurements.

    Usage:
        timer = ScopedTimer()

        # All durations are logged in the timer
        with timer("step_time"):
            with timer("fwd"):
                model.fwd()
            with timer("bwd"):
                model.bwd()

        # Consume all durations and reset internal store
        durations = timer.consume_durations()

        # Durations that are not consumed will raise a ValueError
        with timer("fwd"):
            model.fwd()
        with timer("fwd"):
            model.fwd()  # <-- This will raise an error as timer.consume_durations()
                         # is not called, meaning the previous measurement is
                         # still stored.

    Methods:
        consume_durations() -> dict[str, float]:
            Returns a dictionary of all logged durations and resets the internal log.

        __call__(name: str):
            Context manager for timing a section of code. Raises a ValueError if
            durations are not consumed before starting a new measurement for the
            same name.

    Raises:
        ValueError: If attempting to start a new timing section for a name that
                    already has a recorded duration without consuming the previous
                    measurement using consume_durations().
    """

    def __init__(self, *args, **kwargs):
        self._timer = NamedTimer(*args, **kwargs)
        self._duration_log = {}

    def consume_durations(self) -> dict[str, float]:
        durations = self._duration_log
        self._duration_log = {}
        self._timer.reset()
        return durations

    @contextmanager
    def __call__(self, name: str):
        try:
            self._timer.start(name=name)
            yield
        finally:
            self._timer.stop(name=name)
            if name in self._duration_log:
                raise ValueError(
                    f"Attempted to store new duration for {name=} before consuming last measurement. Call consume_durations() to consume the last set of measurements."
                )
            self._duration_log[name] = self._timer.get(name=name)
