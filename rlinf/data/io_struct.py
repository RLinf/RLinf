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

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import torch
from megatron.core.num_microbatches_calculator import get_num_microbatches
from omegaconf import DictConfig

from rlinf.data.datasets import batch_pad_to_fixed_len
from rlinf.utils.data_iter_utils import (
    get_iterator_dynamic,
    get_iterator_k_split,
    split_list,
)


@dataclass
class RolloutRequest:
    """
    Attr
    input_ids: List of input token IDs for rollout
    n: Number of completions to generate for each input
    idx: List of unique identifiers for the requests, used for tracking
    input_lengths: List of lengths of the input sequences, corresponding to input_ids
    answers: Optional list of answers for the requests, if available
    """

    n: int
    input_ids: List[List[int]]
    answers: List[str]

    def repeat_and_split(
        self, rollout_batch_size: Optional[int] = None
    ) -> List["RolloutRequest"]:
        input_ids, answers = zip(
            *[
                (input_id, answer)
                for input_id, answer in zip(self.input_ids, self.answers)
                for _ in range(self.n)
            ]
        )
        input_ids, answers = (list(input_ids), list(answers))

        # Split input ids based on rollout_batch_size_per_gpu
        if rollout_batch_size is None:
            num_batches = 1
        else:
            assert len(input_ids) % rollout_batch_size == 0, (
                f"Input IDs length {len(input_ids)} is not divisible by rollout batch size {rollout_batch_size}"
            )
            num_batches = len(input_ids) // rollout_batch_size

        splitted_requests = []
        input_ids_split_list = split_list(input_ids, num_batches)
        answers_split_list = split_list(answers, num_batches)

        for input_ids_batch, answers_batch in zip(
            input_ids_split_list, answers_split_list
        ):
            request = RolloutRequest(
                n=self.n,
                input_ids=input_ids_batch,
                answers=answers_batch,
            )
            splitted_requests.append(request)

        return splitted_requests


class CompletionInfo:
    def __init__(self, logger=None):
        self.input_ids: Dict[int, List[int]] = {}  # hash -> input token IDs
        self.complete_num: Dict[int, int] = {}  # hash -> completion count
        self.results: Dict[int, List[Dict]] = {}  # hash -> list of results

        self.num_requests: int = 0
        self.num_completed: int = 0
        self._num_returned: int = 0  # Number of results returned

        self.n_result_each_request: int = 0

        self.logger = logger

    def hash(self, token_ids: List[int]) -> int:
        """Generate a hash for the token IDs."""
        return hash(tuple(token_ids))

    def clear(self):
        self.complete_num.clear()
        self.input_ids.clear()
        self.results.clear()
        self.num_requests = 0
        self.num_completed = 0
        self._num_returned = 0

    def add_request(self, req: RolloutRequest):
        """Add a new request to the completion info."""
        if self.n_result_each_request != 0:
            assert self.n_result_each_request == req.n
        else:
            self.n_result_each_request = req.n

        self.num_requests += len(req.input_ids)

        for ids in req.input_ids:
            hash_id = self.hash(ids)
            if hash_id not in self.input_ids:
                self.input_ids[hash_id] = ids
                self.complete_num[hash_id] = 0
                self.results[hash_id] = []
            else:
                assert self.input_ids[hash_id] == ids, (
                    "Input IDs mismatch for existing hash ID"
                )

    def clear_and_set(self, req: RolloutRequest):
        self.clear()
        self.add_request(req)

    def is_empty(self) -> bool:
        return len(self.complete_num) == 0 and len(self.results) == 0

    def record_result(self, token_ids: List[int], result: Dict) -> int:
        hash_id = self.hash(token_ids)

        self.complete_num[hash_id] += 1
        self.results[hash_id].append(result)

        if self.complete_num[hash_id] == self.n_result_each_request:
            self.num_completed += 1
            if self.logger is not None:
                self.logger.debug(f"Completed all rollouts for hash: {hash_id}")

        return self.complete_num[hash_id]

    def is_completed(self, hash_id: int) -> bool:
        return self.complete_num[hash_id] == self.n_result_each_request

    def get_results(self, hash_id: int) -> List[Dict]:
        """Get the results for the given token IDs."""
        assert hash_id in self.results, "Hash ID not found in results"
        assert self.complete_num[hash_id] == self.n_result_each_request, (
            "Not all results for this hash ID are completed"
        )
        value = self.results.pop(hash_id)
        return value

    def record_returned(self):
        """Record that a result has been returned."""
        self._num_returned += 1
        if self.logger is not None:
            self.logger.debug(
                f"Returned / Completed: {self._num_returned} / {self.num_completed}"
            )

    def all_returned(self) -> bool:
        """Check if all results have been returned."""
        return self._num_returned == self.num_requests


@dataclass(kw_only=True)
class RolloutResult:
    """
    Rollout Result
    """

    num_sequence: int
    group_size: int
    prompt_lengths: List[int]
    prompt_ids: List[List[int]]
    response_lengths: List[int]
    response_ids: List[List[int]]
    is_end: List[bool]
    rewards: Optional[List[float] | torch.Tensor] = None
    advantages: Optional[List[float] | torch.Tensor] = None
    prompt_texts: Optional[List[str]] = None
    response_texts: Optional[List[str]] = None
    answers: Optional[List[str]] = None

    # Inference
    # Only set when recompute_logprobs is False
    rollout_logprobs: Optional[List[List[float]]] = None
    prev_logprobs: Optional[torch.Tensor] = None
    ref_logprobs: Optional[torch.Tensor] = None

    @property
    def batch_size(self):
        return self.num_sequence // self.group_size

    @staticmethod
    def _get_attention_masks_and_position_ids(
        prompt_lengths: torch.Tensor,
        response_lengths: torch.Tensor,
        max_prompt_len: int,
        total_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = prompt_lengths.size(0)

        # =========================
        # Attention Mask
        # =========================
        arange_ids = (
            torch.arange(total_len).unsqueeze(0).expand(B, -1)
        )  # [B, total_len]

        # Compute the start and end positions of the prompt and response tokens
        prompt_start = max_prompt_len - prompt_lengths  # [B]
        response_end = max_prompt_len + response_lengths  # [B]

        # Broadcast [B, total_len]
        prompt_start = prompt_start.unsqueeze(1)
        response_end = response_end.unsqueeze(1)

        attention_mask = (arange_ids >= prompt_start) & (arange_ids < response_end)

        # =========================
        # Position IDs
        # =========================
        position_ids = torch.zeros_like(arange_ids)

        for i in range(B):
            ps = prompt_start[i].item()
            position_ids[i, ps:] = torch.arange(total_len - ps)

        return attention_mask, position_ids

    @staticmethod
    def from_engine_results(
        results: List[Dict],
        group_size: int,
        input_ids: List[List[int]],
        answers: Optional[List[List[int]]] = None,
        return_logprobs: bool = False,
    ) -> "RolloutResult":
        """Create a MathRolloutResult from the given results and input IDs.

        Args:
            results (List[Dict]): The rollout results from the model.
            input_ids (List[List[int]]): The input IDs for the prompts.
            return_logprobs (bool): Whether to return log probabilities.
        """
        assert len(results) == len(input_ids), (
            f"Results length {len(results)} does not match input_ids length {len(input_ids)}"
        )
        assert isinstance(results, list) and all(
            isinstance(res, dict) for res in results
        ), "Results should be a list of dictionaries."
        assert isinstance(input_ids, list) and all(
            isinstance(id_list, list) for id_list in input_ids
        ), "Input IDs should be a list of lists."
        result = RolloutResult(
            num_sequence=len(results),
            group_size=group_size,
            prompt_lengths=[len(input_id) for input_id in input_ids],
            prompt_ids=input_ids,
            response_lengths=[len(res["output_ids"]) for res in results],
            response_ids=[res["output_ids"] for res in results],
            answers=answers,
            is_end=[
                res["meta_info"]["finish_reason"]["type"] == "stop" for res in results
            ],
        )
        if return_logprobs:
            logprobs = [
                [item[0] for item in res["meta_info"]["output_token_logprobs"]]
                for res in results
            ]
            result.rollout_logprobs = logprobs
        return result

    @staticmethod
    def merge_result_list(
        rollout_results: List["RolloutResult"],
    ) -> "RolloutResult":
        assert len(rollout_results) > 0, "No rollout results to merge."
        if len(rollout_results) == 1:
            return rollout_results[0]
        merged_result = RolloutResult(
            num_sequence=sum(res.num_sequence for res in rollout_results),
            group_size=rollout_results[0].group_size,
            prompt_lengths=[],
            prompt_ids=[],
            response_lengths=[],
            response_ids=[],
            is_end=[],
        )
        for res in rollout_results:
            merged_result.prompt_lengths.extend(res.prompt_lengths)
            merged_result.prompt_ids.extend(res.prompt_ids)
            merged_result.response_lengths.extend(res.response_lengths)
            merged_result.response_ids.extend(res.response_ids)
            merged_result.is_end.extend(res.is_end)
            if res.answers is not None:
                if merged_result.answers is None:
                    merged_result.answers = []
                merged_result.answers.extend(res.answers)
            if res.advantages is not None:
                if merged_result.advantages is None:
                    merged_result.advantages = []
                merged_result.advantages.extend(res.advantages)
            if res.rewards is not None:
                if merged_result.rewards is None:
                    merged_result.rewards = []
                merged_result.rewards.extend(res.rewards)
            if res.rollout_logprobs is not None:
                if merged_result.rollout_logprobs is None:
                    merged_result.rollout_logprobs = []
                merged_result.rollout_logprobs.extend(res.rollout_logprobs)

        # TODO Align advantages and rewards format
        if (
            isinstance(merged_result.advantages, list)
            and len(merged_result.advantages) > 0
            and isinstance(merged_result.advantages[0], torch.Tensor)
        ):
            merged_result.advantages = torch.cat(merged_result.advantages, dim=0)
        if (
            isinstance(merged_result.rewards, list)
            and len(merged_result.rewards) > 0
            and isinstance(merged_result.rewards[0], torch.Tensor)
        ):
            merged_result.rewards = torch.cat(merged_result.rewards, dim=0)

        return merged_result

    def to_actor_batch(
        self,
        data_seq_length: int,
        training_seq_length: int,
        pad_token: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Transform the rollout result into a format suitable for the actor.

        Args:
            data_seq_length (int): Maximum prompt length, e.g., 1024.
            training_seq_length (int): Total sequence length for training, e.g., 8192.
                The maximum response length is calculated as `training_seq_length - data_seq_length`.
            pad_token (int): Token used for padding, e.g., `tokenizer.pad_token_id`.

        Returns:
            Dict[str, torch.Tensor]: A dictionary with keys:

            input_ids (torch.Tensor):
                Concatenated prompt and response token IDs,
                shape ``[batch_size, training_seq_length]``.

            attention_mask (torch.Tensor):
                Attention mask for the input sequence,
                shape ``[batch_size, training_seq_length]``.

            is_end (torch.Tensor):
                Boolean tensor indicating whether the sequence ends,
                shape ``[batch_size]``.

            position_ids (torch.Tensor):
                Position IDs for the input sequence,
                shape ``[batch_size, training_seq_length]``.

            prompt_lengths (torch.Tensor):
                Lengths of the prompt sequences,
                shape ``[batch_size]``.

            response_lengths (torch.Tensor):
                Lengths of the response sequences,
                shape ``[batch_size]``.

            advantages (torch.Tensor), optional:
                Advantage values for the responses,
                shape ``[batch_size, training_seq_length - data_seq_length]``.
        """

        # len = training_seq_length: input_ids, attention_mask, position_ids
        #           [prompt_padding, prompt_ids,    response_ids, ... ,response_padding]
        #           |<-- padding -->|<-- pmp len -->|<-- resp len --->|<-- padding --->|
        #           |<---- cfg.data.seq_length ---->|
        #           |<------------------ cfg.runner.seq_length --------------------->|

        # len = training_seq_length - data_seq_length: advantage, prev_logprobs, ref_logprobs
        # each row: [response_ids, ...,                , response_padding]
        #           |<----- true response length ----->|<--- padding --->|
        #           |<-- cfg.runner.seq_length - cfg.data.seq_length ->|

        max_response_len = training_seq_length - data_seq_length

        prompt_lengths = torch.tensor(self.prompt_lengths)
        response_lengths = torch.tensor(self.response_lengths)
        is_end = torch.tensor(self.is_end, dtype=torch.bool)

        attention_mask, position_ids = self._get_attention_masks_and_position_ids(
            prompt_lengths=prompt_lengths,
            response_lengths=response_lengths,
            max_prompt_len=data_seq_length,
            total_len=training_seq_length,
        )

        prompt_ids = batch_pad_to_fixed_len(
            [torch.as_tensor(ids, dtype=torch.long) for ids in self.prompt_ids],
            max_batch_len=data_seq_length,
            pad_token=pad_token,
            left_pad=True,
        )

        response_ids = batch_pad_to_fixed_len(
            [torch.as_tensor(ids, dtype=torch.long) for ids in self.response_ids],
            max_batch_len=max_response_len,
            pad_token=pad_token,
        )
        input_ids = torch.cat(
            [prompt_ids, response_ids], dim=1
        )  # [B, training_seq_length]

        batch = {
            "input_ids": input_ids.cuda(),
            "attention_mask": attention_mask.cuda(),
            "is_end": is_end.cuda(),
            "position_ids": position_ids.cuda(),
            "prompt_lengths": prompt_lengths.cuda(),
            "response_lengths": response_lengths.cuda(),
        }

        if self.advantages is not None:
            if isinstance(self.advantages, torch.Tensor):
                batch["advantages"] = self.advantages.cuda()
            else:
                response_attention_mask = attention_mask[
                    :, -max_response_len:
                ]  # [B, max_response_len]
                advantages = torch.tensor(self.advantages, dtype=torch.float32).reshape(
                    -1, 1
                )  # [B, 1]
                advantages = response_attention_mask.float().cuda() * advantages.cuda()
                batch["advantages"] = advantages.cuda()

        if self.prev_logprobs is not None:
            batch["prev_logprobs"] = self.prev_logprobs.cuda()

        if self.ref_logprobs is not None:
            batch["ref_logprobs"] = self.ref_logprobs.cuda()

        if self.rollout_logprobs is not None:
            logprobs = batch_pad_to_fixed_len(
                [
                    torch.as_tensor(logprobs, dtype=torch.float)
                    for logprobs in self.rollout_logprobs
                ],
                max_batch_len=max_response_len,
                pad_token=pad_token,
            )
            batch["prev_logprobs"] = logprobs.cuda()

        return batch

    @staticmethod
    def batch_to_cpu(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cpu()
        return batch

    @staticmethod
    def batch_to_cuda(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda()
        return batch


class BatchResizingIterator:
    """The iterator for handling getting a batch and split it as a batch iterator with optional dynamic batch size."""

    def __init__(
        self,
        cfg: DictConfig,
        get_batch_fn: Callable,
        micro_batch_size: int,
        total_batch_size: int,
        num_global_batches: int,
        forward_only: bool,
        batch_tensor_key: str = "input_ids",
    ):
        """Initialize the BatchResizingIterator.

        Args:
            cfg (DictConfig): The configuration object.
            get_batch_fn (Callable): The function to get the batch.
            micro_batch_size (int): The size of the micro batch.
            global_batch_size_per_dp (int): The global batch size per data parallel. Here a global batch means the data required for running a single step of inference/training.
            batch_tensor_key (str): The key for retrieving a sample batch tensor, which will be used to measure the batch size and sequence length. By default, this is "input_ids", which means the input_ids tensor's shape will be used to determine batch size and sequence length.
        """
        self.cfg = cfg
        self.get_batch_fn = get_batch_fn
        self.micro_batch_size = micro_batch_size
        self.num_global_batches = num_global_batches
        self.total_batch_size = total_batch_size
        self.global_batch_size = total_batch_size // num_global_batches
        self.forward_only = forward_only
        self.batch_tensor_key = batch_tensor_key

        # Dynamic batch size
        self.cp_world_size = None
        self.vpp_world_size = None
        self.max_tokens_per_mbs = None
        self.microbatch_group_size_per_vp_stage = None
        self.dbs_total_seqlen = None
        self.dbs_indices = None
        self.dbs_n_microbatches = None
        self.has_enabled_dynamic_batch_size = False

        # Iterator states
        self.consumed_batch_size = 0
        self.micro_batch_iter = iter([])
        self.global_batch_iter = iter([])
        self.prefetch_micro_batch = None  # Used for computing batch info
        self.global_batch_done = False
        self.current_global_batch = []
        self.previous_global_batch = []
        self.global_batch_handler = None

    def check_finished_global_batch(self):
        assert self.global_batch_done, (
            f"Batch iterator has not finished for this global batch, only consumed {self.consumed_batch_size} sequences, expected {self.global_batch_size}"
        )

    @property
    def require_full_global_batch(self):
        return (
            self.has_enabled_dynamic_batch_size or self.global_batch_handler is not None
        )

    def enable_dynamic_batch_size(
        self,
        cp_world_size: int,
        vpp_world_size: int,
        max_tokens_per_mbs: int,
        microbatch_group_size_per_vp_stage: int,
    ):
        """Configure and enable dynamic batch sizing.

        Args:
            cp_world_size: The context parallel world size for the model.
            vpp_world_size: The virtual pipeline parallel world size.
            max_tokens_per_mbs: The maximum tokens per micro batch.
            microbatch_group_size_per_vp_stage: The microbatch group size per virtual pipeline stage.
        """
        self.has_enabled_dynamic_batch_size = True
        self.cp_world_size = cp_world_size
        self.vpp_world_size = vpp_world_size
        self.max_tokens_per_mbs = max_tokens_per_mbs
        self.microbatch_group_size_per_vp_stage = microbatch_group_size_per_vp_stage

    def get_batch_info(self, forward_micro_batch_size: int):
        """Get the total sequence length, number of microbatches, and indices based on the batch information and dynamic batch sizing.

        Args:
            forward_micro_batch_size: The size of the forward micro batch.
            forward_only: Whether to only consider the forward pass.
        """
        if self.prefetch_micro_batch is None:
            self.prefetch_micro_batch = next(self)
        if self.has_enabled_dynamic_batch_size:
            assert self.dbs_total_seqlen is not None, (
                "Dynamic batch size is not enabled"
            )
            assert self.dbs_indices is not None, "Dynamic batch size is not enabled"
            assert self.dbs_n_microbatches is not None, (
                "Dynamic batch size is not enabled"
            )
            return self.dbs_total_seqlen, self.dbs_n_microbatches, self.dbs_indices
        else:
            n_microbatches = (
                max(1, self.global_batch_size // forward_micro_batch_size)
                if self.forward_only
                else get_num_microbatches()
            )
            seqlen = self.prefetch_micro_batch[self.batch_tensor_key].shape[1]
            total_seqlen = seqlen * forward_micro_batch_size
            return total_seqlen, n_microbatches, None

    def replay_last_global_batch(self):
        """Replay the last global batch step from the start. Useful when you wish to run different computations with the last global batch data."""
        assert self.global_batch_done, (
            f"The last global batch has not been completed yet. Only {self.consumed_batch_size} sequences have been processed, expected {self.global_batch_size}."
        )
        self.micro_batch_iter = iter(self.previous_global_batch)

    def register_global_batch_handler(self, handler: Callable):
        """This enables processing a global batch before it's splitting into microbatches and consumed.

        Args:
            handler (Callable): The handler function to process the global batch. This function will receive a single argument, which is the global batch to process, and returns the processed global batch.
        """
        self.global_batch_handler = handler

    def _get_next_micro_batch(self):
        """Retrieve the next micro batch from the current microbatch iterator."""
        if self.prefetch_micro_batch is not None:
            # If a microbatch has already been prefetched for batch info computation
            # Return the prefetched microbatch
            micro_batch = self.prefetch_micro_batch
            self.prefetch_micro_batch = None
        else:
            micro_batch: Dict[str, torch.Tensor] = next(self.micro_batch_iter)
            self.global_batch_done = False
            self.consumed_batch_size += micro_batch[self.batch_tensor_key].shape[0]
            self.current_global_batch.append(micro_batch)
            if self.consumed_batch_size == self.global_batch_size:
                # A global batch has been consumed, store the global batch step history
                self.previous_global_batch = self.current_global_batch
                self.current_global_batch = []
                self.consumed_batch_size = 0
                self.global_batch_done = True
            else:
                assert self.consumed_batch_size < self.global_batch_size, (
                    f"Recevied batches with a total size of {self.consumed_batch_size}, which exceeds the global batch size per dp {self.global_batch_size}. This suggests that the configured global batch size cannot be divided by the actual batch size."
                )
        return micro_batch

    def _dynamic_batch_sizing(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Split a batch using dynamic batch sizing."""
        max_tokens_per_mbs = self.max_tokens_per_mbs * self.cp_world_size
        vpp_size = self.vpp_world_size
        if vpp_size is not None and vpp_size > 1:
            microbatch_group_size_per_vp_stage = self.microbatch_group_size_per_vp_stage
            data_iter, indices, n_micro_batch = get_iterator_dynamic(
                batch,
                num_batches_divided_by=microbatch_group_size_per_vp_stage,
                max_tokens_per_mbs=max_tokens_per_mbs,
            )
            assert n_micro_batch % self.microbatch_group_size_per_vp_stage == 0, (
                f"micro_batches {data_iter} must be divisible by microbatch_group_size_per_vp_stage {microbatch_group_size_per_vp_stage} for megatron backend"
            )
        else:
            data_iter, indices, n_micro_batch = get_iterator_dynamic(
                batch, max_tokens_per_mbs=max_tokens_per_mbs
            )
        total_seqlen = max_tokens_per_mbs
        self.dbs_total_seqlen = total_seqlen
        self.dbs_indices = indices
        self.dbs_n_microbatches = n_micro_batch
        return data_iter

    def _merge_batches(
        self, batch1: Dict[str, torch.Tensor], batch2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Merge two batches into one."""
        merged_batch = {}
        for key in batch1.keys():
            assert torch.is_tensor(batch1[key]), (
                f"Expected tensor for key {key} in batch1, got {type(batch1[key])}"
            )
            assert torch.is_tensor(batch2[key]), (
                f"Expected tensor for key {key} in batch2, got {type(batch2[key])}"
            )
            merged_batch[key] = torch.cat([batch1[key], batch2[key]], dim=0)
        return merged_batch

    def _fill_global_batches(self, current_batch: Dict[str, torch.Tensor]):
        """Keep getting batches until the batch size is multiple of a global batch if requires_full_global_batch."""
        current_batch_size = current_batch[self.batch_tensor_key].shape[0]
        while (
            current_batch_size < self.global_batch_size
            and current_batch_size % self.global_batch_size != 0
        ):
            new_batch = self.get_batch_fn(self.forward_only)
            current_batch = self._merge_batches(current_batch, new_batch)
            current_batch_size = current_batch[self.batch_tensor_key].shape[0]
        return current_batch, current_batch_size

    def _get_global_batches(self):
        """Split a batch into multiple global batches, each of which will be used for one step of inference/training."""
        batch = self.get_batch_fn(self.forward_only)
        batch_size = batch[self.batch_tensor_key].shape[0]
        if batch_size % self.global_batch_size != 0:
            # If the batch size is smaller than the global batch size per data parallel group,
            # we can return the batch as is if requires_full_global_batch is False. This usually occurs in pipelining mode.
            if self.require_full_global_batch:
                batch, batch_size = self._fill_global_batches(batch)
            else:
                return iter([batch])
        num_splits = batch_size // self.global_batch_size
        return get_iterator_k_split(
            batch,
            num_splits=num_splits,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

    def __iter__(self):
        """Return the iterator object itself."""
        return self

    def __next__(self):
        """Retrieve the next micro batch from the current microbatch iterator."""
        try:
            return self._get_next_micro_batch()
        except StopIteration:
            try:
                global_batch = next(self.global_batch_iter)
            except StopIteration:
                # If both the current micro and global batch iterators are exhausted, fetch a new batch
                self.global_batch_iter = self._get_global_batches()
                global_batch = next(self.global_batch_iter)

            if self.global_batch_handler is not None:
                global_batch = self.global_batch_handler(global_batch)
                assert global_batch is not None, (
                    f"global batch handler {self.global_batch_handler} must not return None."
                )

            if self.has_enabled_dynamic_batch_size:
                self.micro_batch_iter = self._dynamic_batch_sizing(global_batch)
            else:
                global_batch_size = global_batch[self.batch_tensor_key].shape[0]
                self.micro_batch_iter = get_iterator_k_split(
                    global_batch,
                    num_splits=global_batch_size // self.micro_batch_size,
                    shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
                    shuffle_seed=self.cfg.actor.seed,
                )

            return self._get_next_micro_batch()
