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
from typing import Dict, List, Optional, Tuple

import torch

from rlinf.data.datasets import batch_pad_to_fixed_len
from rlinf.utils.data_iter_utils import get_iterator_k_split, split_list


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
    prompt_lengths: List[int]
    prompt_ids: List[List[int]]
    response_lengths: List[int]
    response_ids: List[List[int]]
    is_end: List[bool]
    rewards: Optional[List[float]] = None
    advantages: Optional[List[float]] = None
    prompt_texts: Optional[List[str]] = None
    response_texts: Optional[List[str]] = None
    answers: Optional[List[str]] = None

    # Inference
    # Only set when recompute_logprobs is False
    rollout_logprobs: Optional[List[List[float]]] = None
    prev_logprobs: Optional[torch.Tensor] = None
    ref_logprobs: Optional[torch.Tensor] = None

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
        merged_result = RolloutResult(
            num_sequence=sum(res.num_sequence for res in rollout_results),
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
        return merged_result

    def to_actor_batch(
        self,
        data_seq_length: int,
        training_seq_length: int,
        pad_token: int,
    ) -> Dict[str, torch.Tensor]:
        """
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
            response_attention_mask = attention_mask[
                :, -max_response_len:
            ]  # [B, max_response_len]
            advantages = torch.tensor(self.advantages, dtype=torch.float32).reshape(
                -1, 1
            )  # [B, 1]
            advantages = response_attention_mask.float() * advantages
            batch["advantages"] = advantages.cuda()

        if self.prev_logprobs is not None:
            batch["prev_logprobs"] = self.prev_logprobs

        if self.ref_logprobs is not None:
            batch["ref_logprobs"] = self.ref_logprobs

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
    def to_metrics(results: List["RolloutResult"]) -> Dict[str, torch.Tensor]:
        batch_size_per_dp = torch.tensor(
            sum(r.num_sequence for r in results), dtype=torch.int64
        ).reshape(1, 1)

        prompt_lengths = torch.cat(
            [
                torch.tensor(r.prompt_lengths, dtype=torch.int32).reshape(1, -1)
                for r in results
            ],
            dim=0,
        )
        response_lengths = torch.cat(
            [
                torch.tensor(r.response_lengths, dtype=torch.int32).reshape(1, -1)
                for r in results
            ],
            dim=0,
        )
        rewards = torch.cat(
            [
                torch.tensor(r.rewards, dtype=torch.float32).reshape(1, -1)
                for r in results
            ],
            dim=0,
        )
        is_end = torch.cat(
            [
                torch.tensor(r.is_end, dtype=torch.float32).reshape(1, -1)
                for r in results
            ],
            dim=0,
        )
        adv = torch.cat(
            [
                torch.tensor(r.advantages, dtype=torch.float32).reshape(1, -1)
                for r in results
            ],
            dim=0,
        )

        return {
            "batch_size_per_dp": batch_size_per_dp,
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "total_lengths": prompt_lengths + response_lengths,
            "rewards": rewards,
            "is_end": is_end,
            "advantages": adv,
        }


class BatchResizingIterator:
    def __init__(
        self,
        fetch_batch_fn,
        micro_batch_size: int,
        global_batch_size_per_dp: int,
    ):
        self.fetch_batch_fn = fetch_batch_fn
        self.micro_batch_size = micro_batch_size
        self.global_batch_size_per_dp = global_batch_size_per_dp

        self.micro_batch_counter = 0
        self.current_batch: RolloutResult = None
        self.current_iter = iter([])

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = next(self.current_iter)
            self.micro_batch_counter += 1
            return result
        except StopIteration:
            # If the current iterator is exhausted, fetch a new batch
            self.current_batch = self.fetch_batch_fn()
            fetch_batch_size = self.current_batch[
                list(self.current_batch.keys())[0]
            ].shape[0]
            self.current_iter = get_iterator_k_split(
                self.current_batch,
                num_microbatches=fetch_batch_size // self.micro_batch_size,
            )

        self.micro_batch_counter += 1
        return next(self.current_iter)

    def finalize(self):
        assert (
            self.micro_batch_counter * self.micro_batch_size
            == self.global_batch_size_per_dp
        ), (
            f"{self.__class__.__name__}: micro_batch_counter * micro_batch_size = "
            f"{self.micro_batch_counter} * {self.micro_batch_size} does not match"
            f" global batch size ({self.global_batch_size_per_dp})."
        )
        self.micro_batch_counter = 0
        # Here, a global batch is completed.
        # Should not reset current_iter and current_batch
        # because global_batch_size_per_dp may not be a multiple of fetch_batch_size
