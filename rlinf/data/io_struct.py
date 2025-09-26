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

from numpy import isin
from typing import Any

import torch
from omegaconf import DictConfig
from vllm.outputs import CompletionOutput
from vllm.outputs import RequestOutput as VllmRequestOutput

from rlinf.data.datasets import batch_pad_to_fixed_len
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    split_list,
)


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

    def repeat(self) -> "RolloutRequest":
        """Repeat each input in the RolloutRequest a specified number of times.

        Args:
            times (int): The number of times to repeat each input.

        Returns:
            RolloutRequest: A new RolloutRequest with repeated inputs.
        """
        assert self.n > 0, "n must be greater than 0"

        input_ids, answers = zip(
            *[
                (input_id, answer)
                for input_id, answer in zip(self.input_ids, self.answers)
                for _ in range(self.n)
            ]
        )
        return RolloutRequest(
            n=self.n,
            input_ids=list(input_ids),
            answers=list(answers),
        )

    def split(self, num_splits: int) -> List["RolloutRequest"]:
        """Split the RolloutRequest into multiple smaller requests.

        Args:
            num_splits (int): The number of splits to create.

        Returns:
            List[RolloutRequest]: A list of smaller RolloutRequest instances.
        """
        assert num_splits > 0, "num_splits must be greater than 0"
        assert len(self.input_ids) % num_splits == 0, (
            f"Input IDs length {len(self.input_ids)} is not divisible by num_splits {num_splits}"
        )

        input_ids_split_list = split_list(self.input_ids, num_splits)
        answers_split_list = split_list(self.answers, num_splits)

        splitted_requests = []
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
    def from_vllm_results(
        group_size: int,
        results: List[VllmRequestOutput],
        answers: Optional[List[str]] = None,
        return_logprobs: bool = False,
    ) -> "RolloutResult":
        def get_logprobs(
            response_ids: List[int], output: CompletionOutput
        ) -> List[float]:
            logprobs = []
            returned_logprobs = output.logprobs
            assert logprobs is not None, (
                "vllm returned None logprobs, while return_logprobs is set."
            )
            for i, logprob in enumerate(returned_logprobs):
                logprobs.append(logprob[response_ids[i]].logprob)
            return logprobs

        num_sequences = len(results) * group_size

        prompt_lengths = []
        prompt_ids = []
        response_lengths = []
        response_ids = []
        logprobs = []
        is_end = []
        response_texts = []
        rollout_answers = (
            [answer for answer in answers for _ in range(group_size)]
            if answers
            else None
        )
        for vllm_result in results:
            if vllm_result.prompt_token_ids is not None:
                prompt_ids.extend([vllm_result.prompt_token_ids] * group_size)
                prompt_lengths.extend([len(vllm_result.prompt_token_ids)] * group_size)
            else:
                raise NotImplementedError("vllm should return tokenized prompt.")
            response_ids.extend(
                [list(output.token_ids) for output in vllm_result.outputs]
            )
            response_texts.extend([output.text for output in vllm_result.outputs])
            response_lengths.extend(
                [len(output.token_ids) for output in vllm_result.outputs]
            )
            is_end.extend([vllm_result.finished] * group_size)
            if return_logprobs:
                logprobs.extend(
                    [
                        get_logprobs(list(output.token_ids), output)
                        for output in vllm_result.outputs
                    ]
                )
        result: RolloutResult = RolloutResult(
            group_size=group_size,
            num_sequence=num_sequences,
            answers=rollout_answers,
            prompt_ids=prompt_ids,
            prompt_lengths=prompt_lengths,
            response_ids=response_ids,
            response_lengths=response_lengths,
            response_texts=response_texts,
            is_end=is_end,
        )
        if return_logprobs:
            result.rollout_logprobs = logprobs
        return result

    @staticmethod
    def from_sglang_results(
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

        def merge_tensor(dst_tensor: torch.Tensor, src_tensor: torch.Tensor):
            assert dst_tensor is None or torch.is_tensor(dst_tensor), (
                f"Expected tensor, got {type(dst_tensor)}"
            )
            assert torch.is_tensor(src_tensor), (
                f"Expected tensor, got {type(src_tensor)}"
            )
            if dst_tensor is None:
                return src_tensor
            else:
                return torch.cat([dst_tensor, src_tensor], dim=0)

        def merge_list(dst_list: List, src_list: List):
            assert dst_list is None or isinstance(dst_list, list), (
                f"Expected list, got {type(dst_list)}"
            )
            assert isinstance(src_list, list), f"Expected list, got {type(src_list)}"
            if dst_list is None:
                return src_list
            else:
                dst_list.extend(src_list)
                return dst_list

        for res in rollout_results:
            merged_result.prompt_lengths.extend(res.prompt_lengths)
            merged_result.prompt_ids.extend(res.prompt_ids)
            merged_result.response_lengths.extend(res.response_lengths)
            merged_result.response_ids.extend(res.response_ids)
            merged_result.is_end.extend(res.is_end)
            if res.answers is not None:
                merged_result.answers = merge_list(merged_result.answers, res.answers)
            if res.advantages is not None:
                if isinstance(res.advantages, list):
                    merged_result.advantages = merge_list(
                        merged_result.advantages, res.advantages
                    )
                elif isinstance(res.advantages, torch.Tensor):
                    merged_result.advantages = merge_tensor(
                        merged_result.advantages, res.advantages
                    )
                else:
                    raise ValueError(
                        f"Wrong type of advantages {type(merged_result.advantages)}"
                    )
            if res.rewards is not None:
                if isinstance(res.rewards, list):
                    merged_result.rewards = merge_list(
                        merged_result.rewards, res.rewards
                    )
                elif isinstance(res.rewards, torch.Tensor):
                    merged_result.rewards = merge_tensor(
                        merged_result.rewards, res.rewards
                    )
                else:
                    raise ValueError(
                        f"Wrong type of rewards {type(merged_result.rewards)}"
                    )
            if res.rollout_logprobs is not None:
                merged_result.rollout_logprobs = merge_list(
                    merged_result.rollout_logprobs, res.rollout_logprobs
                )
            if res.prev_logprobs is not None:
                merged_result.prev_logprobs = merge_tensor(
                    merged_result.prev_logprobs, res.prev_logprobs
                )
            if res.ref_logprobs is not None:
                merged_result.ref_logprobs = merge_tensor(
                    merged_result.ref_logprobs, res.ref_logprobs
                )

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

        if self.rewards is not None:
            batch["rewards"] = self.rewards.cuda()

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
    def merge_batches(
        batches: List[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        """Merge two batches into one."""
        merged_batch = {}
        if len(batches) == 0:
            return merged_batch
        if len(batches) == 1:
            return batches[0]
        for key in batches[0].keys():
            assert torch.is_tensor(batches[0][key]), (
                f"Expected tensor for key {key} in batches, got {type(batches[0][key])}"
            )
            assert torch.is_tensor(batches[0][key]), (
                f"Expected tensor for key {key} in batches, got {type(batches[0][key])}"
            )
            merged_batch[key] = torch.cat([batch[key] for batch in batches], dim=0)
        return merged_batch


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

        # Iterator states
        self.consumed_batch_size = 0
        self.micro_batch_iter = iter([])
        self.global_batch_iter = iter([])
        self.prefetch_micro_batch = None  # Used for computing batch info
        self.global_batch_done = False
        self.batches = []

    def check_finished_global_batch(self):
        assert self.global_batch_done, (
            f"Batch iterator has not finished for this global batch, only consumed {self.consumed_batch_size} sequences, expected {self.global_batch_size}"
        )

    def get_all_batches(self):
        """Retrieve all the batches (merged) iterated after the last call to get_all_batches."""
        batch = RolloutResult.merge_batches(self.batches)
        self.batches = []
        return batch

    def prefetch_one_batch(self):
        """Get the total sequence length, number of microbatches, and indices based on the batch information and dynamic batch sizing.

        Args:
            forward_micro_batch_size: The size of the forward micro batch.
            forward_only: Whether to only consider the forward pass.
        """
        if self.prefetch_micro_batch is None:
            self.prefetch_micro_batch = next(self)

        return self.prefetch_micro_batch

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
            self.batches.append(micro_batch)
            if self.consumed_batch_size == self.global_batch_size:
                # A global batch has been consumed, store the global batch step history
                self.consumed_batch_size = 0
                self.global_batch_done = True
            else:
                assert self.consumed_batch_size < self.global_batch_size, (
                    f"Recevied batches with a total size of {self.consumed_batch_size}, which exceeds the global batch size per dp {self.global_batch_size}. This suggests that the configured global batch size cannot be divided by the actual batch size."
                )
        return micro_batch

    def _get_global_batches(self):
        """Split a batch into multiple global batches, each of which will be used for one step of inference/training."""
        batch, result = self.get_batch_fn()
        batch_size = result.num_sequence
        if batch_size % self.global_batch_size != 0:
            # If the batch size is smaller than the global batch size per data parallel group,
            # we can return the batch as is
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

            global_batch_size = get_batch_size(global_batch, self.batch_tensor_key)
            self.micro_batch_iter = get_iterator_k_split(
                global_batch,
                num_splits=global_batch_size // self.micro_batch_size,
                shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
                shuffle_seed=self.cfg.actor.seed,
            )

            return self._get_next_micro_batch()

def put_tensor_cpu(data_dict):
    if data_dict is None:
        return None

    for key, value in data_dict.items():
        if isinstance(value, dict):
            data_dict[key] = put_tensor_cpu(value)
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.cpu().contiguous()
    return data_dict

@dataclass(kw_only=True)
class EnvOutput:
    simulator_type: str
    obs: Dict[str, Any]
    final_obs: Optional[Dict[str, Any]]=None
    dones: torch.Tensor # [B]
    rewards: Optional[torch.Tensor]=None # [B]

    def __post_init__(self):
        self.obs = put_tensor_cpu(self.obs)
        self.final_obs = put_tensor_cpu(self.final_obs) if self.final_obs is not None else None
        self.dones = self.dones.cpu().contiguous()
        self.rewards = self.rewards.cpu().contiguous() if self.rewards is not None else None

    def prepare_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:

        wrist_image_tensor = None
        if self.simulator_type == "libero":
            image_tensor = torch.stack(
                [
                    value.clone().permute(2, 0, 1)
                    for value in obs["images_and_states"]["full_image"]
                ]
            )
            if "wrist_image" in obs["images_and_states"]:
                wrist_image_tensor = torch.stack(
                    [
                        value.clone().permute(2, 0, 1)
                        for value in obs["images_and_states"]["wrist_image"]
                    ]
                )
        elif self.simulator_type == "maniskill":
            image_tensor = obs["images"]
        elif self.simulator_type == "robotwin":
            image_tensor = obs["images"]
        else:
            raise NotImplementedError
        # Add num_images dimension
        if image_tensor.ndim == 4:
            image_tensor = image_tensor.unsqueeze(1)
        assert image_tensor.ndim == 5

        states = obs["images_and_states"]["state"] if "state" in obs["images_and_states"] else None

        task_descriptions = list(obs["task_descriptions"]) if "task_descriptions" in obs else None

        return {
            "images": image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def to_dict(self):
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = self.prepare_observations(self.final_obs) if self.final_obs is not None else None
        env_output_dict["dones"] = self.dones
        env_output_dict["rewards"] = self.rewards

        return env_output_dict

@dataclass(kw_only=True)
class EmbodiedRolloutResult:

    # required
    actions: List[torch.Tensor] = []
    prev_logprobs: List[torch.Tensor] = []
    prev_values: List[torch.Tensor] = []
    dones: List[torch.Tensor] = []
    rewards: List[torch.Tensor] = []

    processed_obs: List[Dict[str, Any]] = []

    # vla
    action_tokens: Optional[List[torch.Tensor]] = None

    # openpi
    chains: Optional[List[torch.Tensor]] = None
    denoise_inds: Optional[List[torch.Tensor]] = None
    

    def __post_init__(self):
        self.actions = [action.cpu().contiguous() for action in self.actions]
        self.prev_logprobs = [prev_logprob.cpu().contiguous() for prev_logprob in self.prev_logprobs] if self.prev_logprobs is not None else []
        self.prev_values = [prev_value.cpu().contiguous() for prev_value in self.prev_values] if self.prev_values is not None else []

        self.dones = [done.cpu().contiguous() for done in self.dones] if self.dones is not None else []
        self.rewards = [reward.cpu().contiguous() for reward in self.rewards] if self.rewards is not None else []

        self.action_tokens = [action_token.cpu().contiguous() for action_token in self.action_tokens] if self.action_tokens is not None else None

        self.chains = [chain.cpu().contiguous() for chain in self.chains] if self.chains is not None else None
        self.denoise_inds = [denoise_ind.cpu().contiguous() for denoise_ind in self.denoise_inds] if self.denoise_inds is not None else None

        self.processed_obs = [put_tensor_cpu(processed_obs) for processed_obs in self.processed_obs]

    def append_result(self, result: Dict[str, Any]):
        self.actions.append(result["actions"].cpu().contiguous())
        self.prev_logprobs.append(result["prev_logprobs"].cpu().contiguous()) if "prev_logprobs" in result else []
        self.prev_values.append(result["prev_values"].cpu().contiguous()) if "prev_values" in result else []
        
        self.dones.append(result["dones"].cpu().contiguous()) if "dones" in result else []
        self.rewards.append(result["rewards"].cpu().contiguous()) if "rewards" in result else []

        if "action_tokens" in result:
            self.action_tokens.append(result["action_tokens"].cpu().contiguous()) 

        if "chains" in result:
            self.chains.append(result["chains"].cpu().contiguous()) 

        if "denoise_inds" in result:
            self.denoise_inds.append(result["denoise_inds"].cpu().contiguous()) 

        self.processed_obs.append(put_tensor_cpu(result["processed_obs"]))

    def to_dict(self):

        merged_processed_obs = {}
        for processed_obs in self.processed_obs:
            for k, v in processed_obs.items():
                merged_processed_obs[k].append(v) if k in merged_processed_obs else merged_processed_obs[k] = [v]
        for k in merged_processed_obs:
            merged_processed_obs[k] = torch.cat(merged_processed_obs[k], dim=0).contiguous().cpu()

        rollout_result_dict = {}
        rollout_result_dict["actions"] = torch.cat(self.actions, dim=0).contiguous().cpu()
        rollout_result_dict["prev_logprobs"] = torch.cat(self.prev_logprobs, dim=0).contiguous().cpu()
        rollout_result_dict["prev_values"] = torch.cat(self.prev_values, dim=0).contiguous().cpu()

        rollout_result_dict["dones"] = torch.cat(self.dones, dim=0).contiguous().cpu()
        rollout_result_dict["rewards"] = torch.cat(self.rewards, dim=0).contiguous().cpu()
        
        if self.action_tokens is not None:
            rollout_result_dict["action_tokens"] = torch.cat(self.action_tokens, dim=0).contiguous().cpu()
        
        if self.chains is not None:
            rollout_result_dict["chains"] = torch.cat(self.chains, dim=0).contiguous().cpu()
        if self.denoise_inds is not None:
            rollout_result_dict["denoise_inds"] = torch.cat(self.denoise_inds, dim=0).contiguous().cpu()

        rollout_result_dict["processed_obs"] = merged_processed_obs
        return rollout_result_dict


    def to_splited_dict(self, split_size)->List[Dict[str, Any]]:

        rollout_result_list = []
        for i in range(split_size):
            rollout_result_list.append(self.to_dict())
        
            for key, value in rollout_result_list[i].items():
                if isinstance(value, torch.Tensor):
                    rollout_result_list[i][key] = torch.chunk(value, split_size, dim=0)[i]
                    
                elif isinstance(value, dict):
                    for k, v in value.items():
                        rollout_result_list[i][key][k] = torch.chunk(v, split_size, dim=0)[i]
                else:
                    raise ValueError(f"Unsupported type: {type(value)}")

        return rollout_result_list
    