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

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import torch
from omegaconf import DictConfig

if TYPE_CHECKING:
    from vllm.outputs import CompletionOutput
    from vllm.outputs import RequestOutput as VllmRequestOutput

from rlinf.data.datasets.utils import batch_pad_to_fixed_len
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    split_list,
)


def get_batch_size(
    batch: dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    """Get the batch size from the batch dictionary."""
    return batch[batch_tensor_key].size(0)


def get_seq_length(
    batch: dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    """Get the sequence length from the batch dictionary."""
    return batch[batch_tensor_key].size(1)


@dataclass
class RolloutRequest:
    """
    Attr
    input_ids: list of input token IDs for rollout
    n: Number of completions to generate for each input
    image_data: list of image data (bytes or URLs) for multimodal inputs
    answers: list of answers for the requests, where each answer can be either a list of strings (for typical tasks) or a dict (for VQA tasks), if available.
    multi_modal_inputs: list of multi-modal inputs for the requests
    """

    n: int
    input_ids: list[list[int]]
    image_data: Union[list[list[bytes]], list[list[str]]]
    answers: list[Union[list[str], dict]]
    multi_modal_inputs: list[Optional[dict]]

    def to_seq_group_infos(self) -> list["SeqGroupInfo"]:
        """Convert the RolloutRequest into a list of SeqGroupInfo objects.

        Returns:
            list[SeqGroupInfo]: A list of SeqGroupInfo objects.
        """
        return [
            SeqGroupInfo(
                id=uuid.uuid4().int,
                input_ids=input_ids,
                answer=answer,
                group_size=self.n,
                image_data=image_data,
                multi_modal_inputs=multi_modal_inputs,
            )
            for input_ids, answer, image_data, multi_modal_inputs in zip(
                self.input_ids,
                self.answers,
                self.image_data,
                self.multi_modal_inputs,
                strict=True,
            )
        ]


class FinishReasonEnum(str, Enum):
    ABORT = "abort"
    STOP = "stop"
    LENGTH = "length"


@dataclass
class SeqGroupInfo:
    """
    SeqGroupInfo represents a group of sequences and tracks their processing status and results.

    Each SeqGroupInfo instance corresponds to a single input sequence that is expanded into `group_size` sequences

    Attributes:
        id (int): Unique identifier for the sequence group.
        input_ids (list[int]): list of input IDs of the original sequence.
        answer (Union[list[str], dict]): list of answers of the original sequence.(One sequence can have multiple equivalent answers), or a dict in case of vqa task.
        group_size (int): Number of sequences in the group.
        idx_completed (set[int]): Set of indices for sequences that have completed rollout and are ready for evaluation.
        idx_aborted (set[int]): Set of indices for sequences that have been aborted. These sequences need to be re-rolled out before they can be evaluated.
        results (list[Optional[dict]]): list storing result for each sequence, or None if not yet available.
    """

    id: int
    input_ids: list[int]
    answer: Union[list[str], dict]
    group_size: int
    idx_completed: set[int] = field(init=False, compare=False)
    idx_aborted: set[int] = field(init=False, compare=False)
    results: list[Optional[Union[dict, "VllmRequestOutput"]]] = field(
        init=False, compare=False
    )
    image_data: Optional[list] = None
    multi_modal_inputs: Optional[dict] = None

    def __post_init__(self):
        assert self.group_size > 0, "group_size must be greater than 0"
        self.idx_completed = set()
        self.idx_aborted = set()
        self.results = [None for _ in range(self.group_size)]

    def record_vllm_result(self, idx: int, result: "VllmRequestOutput", logger=None):
        finish_reason = result.outputs[0].finish_reason
        if finish_reason is None or finish_reason == "abort":
            self.idx_aborted.add(idx)
        else:
            self.idx_completed.add(idx)

        if self.results[idx] is None:
            self.results[idx] = result
        else:
            self.results[idx].add(next_output=result, aggregate=True)

    def record_sglang_result(self, idx: int, result: dict, logger=None):
        """Record a single sglang execution result and update internal tracking.

        This method is responsible for updating the internal state of the SeqGroupInfo
        instance based on the result of a single sglang execution. It accepts the index of the
        sequence within the group and the result dictionary returned by sglang. Then it updates
        the sets of completed and aborted indices based on the finish reason provided in the result.
        If the result for the given index already exists, indicating that this is a re-rollout
        sequence, this method will merge the new result with the previous one by concatenating the output IDs.

        Args:
            idx: int
                The index of the sequence within the group (0 <= idx < group_size).
            result: dict
                Result of SGLang. Expected to contain at least:
                - "meta_info": {"finish_reason": {"type": FinishReasonEnum}}
                - "output_ids": a list (or list-like) of output identifier elements
            logger: optional
                Optional logger for diagnostic messages.
        """

        finished_reason = result["meta_info"]["finish_reason"]["type"]
        match finished_reason:
            case FinishReasonEnum.ABORT:
                self.idx_aborted.add(idx)
            case FinishReasonEnum.STOP | FinishReasonEnum.LENGTH:
                self.idx_completed.add(idx)
            case _:
                raise ValueError(f"Unknown finish reason: {finished_reason}")
        if self.results[idx] is None:
            self.results[idx] = result
        else:
            prev_output_ids = self.results[idx]["output_ids"]
            self.results[idx] = result
            self.results[idx]["output_ids"] = prev_output_ids + result["output_ids"]

    def __hash__(self):
        return self.id

    @property
    def num_completed(self) -> int:
        """Returns the number of completed sequences."""
        return len(self.idx_completed)

    @property
    def num_aborted(self) -> int:
        """Returns the number of aborted sequences."""
        return len(self.idx_aborted)

    @property
    def num_returned(self) -> int:
        """Returns the total number of sequences that have either completed or aborted."""
        return self.num_completed + self.num_aborted

    @property
    def num_running(self) -> int:
        """Returns the number of sequences still running."""
        return self.group_size - self.num_returned

    @property
    def all_returned(self) -> bool:
        """Returns True if all sequences have either completed or aborted."""
        return self.num_returned == self.group_size

    @property
    def all_completed(self) -> bool:
        """Returns True if all sequences have completed."""
        return self.num_completed == self.group_size


@dataclass(kw_only=True)
class RolloutResult:
    """
    Rollout Result
    """

    num_sequence: int
    group_size: int
    prompt_lengths: list[int]
    prompt_ids: list[list[int]]
    response_lengths: list[int]
    response_ids: list[list[int]]
    is_end: list[bool]
    rewards: Optional[list[float] | torch.Tensor] = None
    advantages: Optional[list[float] | torch.Tensor] = None
    prompt_texts: Optional[list[str]] = None
    response_texts: Optional[list[str]] = None
    answers: Optional[list[str | dict]] = None
    image_data: Optional[Union[list[list[bytes]], list[list[str]]]] = None
    multi_modal_inputs: Optional[list[dict]] = None
    response_mask: Optional[list[list[int]]] = None
    # Inference
    # Logprobs returned by rollout engines
    rollout_logprobs: Optional[list[list[float]]] = None
    # Logprobs recomputed by inference when rollout has returned logprobs
    recompute_prev_logprobs: Optional[torch.Tensor] = None
    # The final prev_logprobs used for training
    # When rollout has returned logprobs, this is rollout_logprobs
    # Otherwise, this is the logprobs recomputed by inference
    prev_logprobs: Optional[torch.Tensor] = None
    # Reference logprobs for comparison
    ref_logprobs: Optional[torch.Tensor] = None

    @property
    def batch_size(self):
        return self.num_sequence // self.group_size

    @staticmethod
    def _get_attention_masks_and_position_ids(
        prompt_lengths: torch.Tensor,
        response_lengths: torch.Tensor,
        response_mask: torch.Tensor | None,
        max_prompt_len: int,
        total_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        if response_mask is not None:
            max_response_len = total_len - max_prompt_len
            response_mask = batch_pad_to_fixed_len(
                [torch.as_tensor(ids, dtype=torch.long) for ids in response_mask],
                max_batch_len=max_response_len,
                pad_token=0,
            )

            attention_mask = torch.cat(
                [
                    (
                        torch.arange(max_prompt_len)
                        .unsqueeze(0)
                        .expand(response_mask.size(0), -1)
                        >= prompt_start
                    ),
                    response_mask,
                ],
                dim=1,
            ).bool()
        else:
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
        results: list["VllmRequestOutput"],
        answers: Optional[Union[list[str], dict]] = None,
        multi_modal_inputs: Optional[list[dict]] = None,
        return_logprobs: bool = False,
    ) -> "RolloutResult":
        """
        Create a RolloutResult from the given vLLM results. every result is generated with n=1,
        so its outputs len is 1

        Args:
            group_size (int): The group size used during rollout.
            results (list[VllmRequestOutput]): The rollout results from vLLM.
            answers (Optional[Union[list[str], dict]]): The answers corresponding to the inputs, notably, if task type is vqa, answers is a dict.
            multi_modal_inputs (Optional[list[dict]]): The multi-modal inputs corresponding to the inputs.
            return_logprobs (bool): Whether to return log probabilities.

        Returns:
            RolloutResult: The constructed RolloutResult object.
        """

        def get_logprobs(
            response_ids: list[int], output: "CompletionOutput"
        ) -> list[float]:
            logprobs = []
            returned_logprobs = output.logprobs
            assert logprobs is not None, (
                "vllm returned None logprobs, while return_logprobs is set."
            )
            for i, logprob in enumerate(returned_logprobs):
                logprobs.append(logprob[response_ids[i]].logprob)
            return logprobs

        # for VQA task, answers is a dict
        if isinstance(answers, dict):
            answers = [answers]

        # here vllm must return prompt ids because we pass input_ids as input
        prompt_ids = [vllm_result.prompt_token_ids for vllm_result in results]
        prompt_lengths = [len(vllm_result.prompt_token_ids) for vllm_result in results]
        response_ids = [vllm_result.outputs[0].token_ids for vllm_result in results]
        response_texts = [vllm_result.outputs[0].text for vllm_result in results]
        response_lengths = [
            len(vllm_result.outputs[0].token_ids) for vllm_result in results
        ]
        is_end = [vllm_result.finished for vllm_result in results]
        if return_logprobs:
            logprobs = [
                get_logprobs(
                    list(vllm_result.outputs[0].token_ids), vllm_result.outputs[0]
                )
                for vllm_result in results
            ]
        result: RolloutResult = RolloutResult(
            group_size=group_size,
            num_sequence=len(results),
            answers=answers,
            prompt_ids=prompt_ids,
            prompt_lengths=prompt_lengths,
            response_ids=response_ids,
            response_lengths=response_lengths,
            response_texts=response_texts,
            multi_modal_inputs=multi_modal_inputs,
            is_end=is_end,
        )
        if return_logprobs:
            result.rollout_logprobs = logprobs
        return result

    @staticmethod
    def from_sglang_results(
        results: list[dict],
        group_size: int,
        input_ids: list[list[int]],
        answers: Optional[list[list[int]]] = None,
        image_data: Optional[Union[list[list[bytes]], list[list[str]]]] = None,
        multi_modal_inputs: Optional[list[dict]] = None,
        return_logprobs: bool = False,
    ) -> "RolloutResult":
        """Create a MathRolloutResult from the given results and input IDs.

        Args:
            results (list[dict]): The rollout results from the model.
            input_ids (list[list[int]]): The input IDs for the prompts.
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
            image_data=image_data,
            multi_modal_inputs=multi_modal_inputs,
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

    @classmethod
    def from_sglang_seq_group(cls, seq_group: SeqGroupInfo, return_logprobs: bool):
        return cls.from_sglang_results(
            seq_group.results,
            seq_group.group_size,
            [seq_group.input_ids] * seq_group.group_size,
            [seq_group.answer] * seq_group.group_size,
            image_data=[seq_group.image_data] * seq_group.group_size,
            multi_modal_inputs=[seq_group.multi_modal_inputs] * seq_group.group_size,
            return_logprobs=return_logprobs,
        )

    @classmethod
    def from_vllm_seq_group(cls, seq_group: SeqGroupInfo, return_logprobs: bool):
        return cls.from_vllm_results(
            seq_group.group_size,
            seq_group.results,
            answers=[seq_group.answer] * seq_group.group_size,
            multi_modal_inputs=[seq_group.multi_modal_inputs] * seq_group.group_size,
            return_logprobs=return_logprobs,
        )

    @staticmethod
    def merge_result_list(
        rollout_results: list["RolloutResult"],
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

        def merge_list(dst_list: list, src_list: list):
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
            if res.recompute_prev_logprobs is not None:
                merged_result.recompute_prev_logprobs = merge_tensor(
                    merged_result.recompute_prev_logprobs, res.recompute_prev_logprobs
                )
        return merged_result

    @staticmethod
    def split_result_list_by_group(
        rollout_results: list["RolloutResult"],
    ) -> list["RolloutResult"]:
        """
        Split RolloutResult objects by group_size.

        If input has only one RolloutResult, split it into multiple RolloutResult objects by group_size.
        If input has multiple RolloutResult objects, split each one and merge the results.

        Args:
            rollout_results: list of input RolloutResult objects

        Returns:
            list of RolloutResult objects grouped by group_size
        """
        assert len(rollout_results) > 0, "No rollout results to split."

        all_split_results = []

        for rollout_result in rollout_results:
            split_results = RolloutResult._split_single_result_by_group(rollout_result)
            all_split_results.extend(split_results)

        return all_split_results

    @staticmethod
    def _split_single_result_by_group(
        rollout_result: "RolloutResult",
    ) -> list["RolloutResult"]:
        """
        Split a single RolloutResult into multiple RolloutResult objects by group_size.

        Args:
            rollout_result: The RolloutResult to be split

        Returns:
            list of split RolloutResult objects
        """
        group_size = rollout_result.group_size
        num_sequence = rollout_result.num_sequence

        assert num_sequence % group_size == 0, (
            f"num_sequence ({num_sequence}) must be divisible by group_size ({group_size})"
        )

        num_groups = num_sequence // group_size
        split_results = []

        # Split list fields
        prompt_lengths_split = split_list(rollout_result.prompt_lengths, num_groups)
        prompt_ids_split = split_list(rollout_result.prompt_ids, num_groups)
        response_lengths_split = split_list(rollout_result.response_lengths, num_groups)
        response_ids_split = split_list(rollout_result.response_ids, num_groups)
        is_end_split = split_list(rollout_result.is_end, num_groups)

        # Handle optional fields
        answers_split = None
        if rollout_result.answers is not None:
            answers_split = split_list(rollout_result.answers, num_groups)

        image_data_split = None
        if rollout_result.image_data is not None:
            image_data_split = split_list(rollout_result.image_data, num_groups)

        multi_modal_inputs_split = None
        if rollout_result.multi_modal_inputs is not None:
            multi_modal_inputs_split = split_list(
                rollout_result.multi_modal_inputs, num_groups
            )

        prompt_texts_split = None
        if rollout_result.prompt_texts is not None:
            prompt_texts_split = split_list(rollout_result.prompt_texts, num_groups)

        response_texts_split = None
        if rollout_result.response_texts is not None:
            response_texts_split = split_list(rollout_result.response_texts, num_groups)

        rollout_logprobs_split = None
        if rollout_result.rollout_logprobs is not None:
            rollout_logprobs_split = split_list(
                rollout_result.rollout_logprobs, num_groups
            )

        # Handle tensor fields
        rewards_split = None
        if rollout_result.rewards is not None:
            if isinstance(rollout_result.rewards, torch.Tensor):
                rewards_split = torch.chunk(rollout_result.rewards, num_groups, dim=0)
            else:
                rewards_split = split_list(rollout_result.rewards, num_groups)

        advantages_split = None
        if rollout_result.advantages is not None:
            if isinstance(rollout_result.advantages, torch.Tensor):
                advantages_split = torch.chunk(
                    rollout_result.advantages, num_groups, dim=0
                )
            else:
                advantages_split = split_list(rollout_result.advantages, num_groups)

        prev_logprobs_split = None
        if rollout_result.prev_logprobs is not None:
            prev_logprobs_split = torch.chunk(
                rollout_result.prev_logprobs, num_groups, dim=0
            )

        ref_logprobs_split = None
        if rollout_result.ref_logprobs is not None:
            ref_logprobs_split = torch.chunk(
                rollout_result.ref_logprobs, num_groups, dim=0
            )

        # Create split RolloutResult objects
        for i in range(num_groups):
            split_result = RolloutResult(
                num_sequence=group_size,
                group_size=group_size,
                prompt_lengths=prompt_lengths_split[i],
                prompt_ids=prompt_ids_split[i],
                response_lengths=response_lengths_split[i],
                response_ids=response_ids_split[i],
                is_end=is_end_split[i],
                answers=answers_split[i] if answers_split is not None else None,
                image_data=image_data_split[i]
                if image_data_split is not None
                else None,
                multi_modal_inputs=multi_modal_inputs_split[i]
                if multi_modal_inputs_split is not None
                else None,
                prompt_texts=prompt_texts_split[i]
                if prompt_texts_split is not None
                else None,
                response_texts=response_texts_split[i]
                if response_texts_split is not None
                else None,
                rollout_logprobs=rollout_logprobs_split[i]
                if rollout_logprobs_split is not None
                else None,
                rewards=rewards_split[i] if rewards_split is not None else None,
                advantages=advantages_split[i]
                if advantages_split is not None
                else None,
                prev_logprobs=prev_logprobs_split[i]
                if prev_logprobs_split is not None
                else None,
                ref_logprobs=ref_logprobs_split[i]
                if ref_logprobs_split is not None
                else None,
            )
            split_results.append(split_result)

        return split_results

    def to_actor_batch(
        self,
        data_seq_length: int,
        training_seq_length: int,
        pad_token: int,
    ) -> dict[str, torch.Tensor]:
        """
        Transform the rollout result into a format suitable for the actor.

        Args:
            data_seq_length (int): Maximum prompt length, e.g., 1024.
            training_seq_length (int): Total sequence length for training, e.g., 8192.
                The maximum response length is calculated as `training_seq_length - data_seq_length`.
            pad_token (int): Token used for padding, e.g., `tokenizer.pad_token_id`.

        Returns:
            dict[str, torch.Tensor]: A dictionary with keys:

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
            response_mask=self.response_mask,
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

        if (
            self.multi_modal_inputs is not None
            and self.multi_modal_inputs[0] is not None
        ):
            batch["multi_modal_inputs"] = self.multi_modal_inputs

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

        if self.recompute_prev_logprobs is not None:
            batch["recompute_prev_logprobs"] = self.recompute_prev_logprobs.cuda()

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
        batches: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Merge two batches into one."""
        merged_batch = {}
        if len(batches) == 0:
            return merged_batch
        if len(batches) == 1:
            return batches[0]

        for key in batches[0].keys():
            if torch.is_tensor(batches[0][key]):
                merged_batch[key] = torch.cat([batch[key] for batch in batches], dim=0)
            elif isinstance(batches[0][key], list):
                merged_batch[key] = []
                for batch in batches:
                    merged_batch[key].extend(batch[key])
            else:
                raise ValueError(f"Unsupported batch key type: {type(batches[0][key])}")
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
            total_batch_size (int): The total batch size.
            num_global_batches (int): The number of global batches.
            forward_only (bool): Whether to run in forward-only mode.
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
        self.get_batch_fn_handler = None
        self.global_batch_handler: Optional[Callable] = None

    def register_get_batch_handler(self, handler: Callable):
        """This enables processing a batch after calling self.get_batch_fn.

        Args:
            handler (Callable): The handler function to process the return value of self.get_batch_fn.
        """
        self.get_batch_fn_handler = handler

    def reset_total_batch_size(self, total_batch_size: int):
        self.total_batch_size = total_batch_size
        self.global_batch_size = total_batch_size // self.num_global_batches

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
        """Get the total sequence length, number of microbatches, and indices based on the batch information and dynamic batch sizing."""
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
            micro_batch: dict[str, torch.Tensor] = next(self.micro_batch_iter)
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

    def register_global_batch_handler(self, handler: Callable):
        """This enables processing a global batch before it's splitting into microbatches and consumed.

        Args:
            handler (Callable): The handler function to process the global batch. This function will receive a single argument, which is the global batch to process, and returns the processed global batch.
        """
        self.global_batch_handler = handler

    def _fill_global_batches(self, current_batch: dict[str, torch.Tensor]):
        """Keep getting batches until the batch size is multiple of a global batch if requires_global_batch."""
        current_batch_size = current_batch[self.batch_tensor_key].shape[0]
        while (
            current_batch_size < self.global_batch_size
            and current_batch_size % self.global_batch_size != 0
        ):
            new_batch, result = self.get_batch_fn()
            if self.get_batch_fn_handler is not None:
                new_batch = self.get_batch_fn_handler(new_batch)
            current_batch = RolloutResult.merge_batches([current_batch, new_batch])
            current_batch_size = current_batch[self.batch_tensor_key].shape[0]
        return current_batch

    def _get_global_batches(self):
        """Split a batch into multiple global batches, each of which will be used for one step of inference/training."""
        batch, result = self.get_batch_fn()
        if self.get_batch_fn_handler is not None:
            batch = self.get_batch_fn_handler(batch)
        batch_size = result.num_sequence
        if batch_size % self.global_batch_size != 0:
            if self.global_batch_handler is not None:
                # No need to fill global batches if it's already filled by full batches
                batch = self._fill_global_batches(batch)
                batch_size = get_batch_size(batch, self.batch_tensor_key)
            else:
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

            if self.global_batch_handler is not None:
                global_batch = self.global_batch_handler(global_batch)
                assert global_batch is not None, (
                    f"global batch handler {self.global_batch_handler} must not return None."
                )

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
    obs: dict[str, Any]
    final_obs: Optional[dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    rewards: Optional[torch.Tensor] = None  # [B]

    def __post_init__(self):
        self.obs = put_tensor_cpu(self.obs)
        self.final_obs = (
            put_tensor_cpu(self.final_obs) if self.final_obs is not None else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )

    def prepare_observations(self, obs: dict[str, Any]) -> dict[str, Any]:
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
        elif self.simulator_type == "behavior":
            image_tensor = obs["images"]
            wrist_image_tensor = obs["wrist_images"]
        elif self.simulator_type == "metaworld":
            image_tensor = torch.stack(
                [
                    value.clone().permute(2, 0, 1)
                    for value in obs["images_and_states"]["full_image"]
                ]
            )
        else:
            raise NotImplementedError

        states = None
        if "images_and_states" in obs and "state" in obs["images_and_states"]:
            states = obs["images_and_states"]["state"]
        if "state" in obs:
            states = obs["state"]

        task_descriptions = (
            list(obs["task_descriptions"]) if "task_descriptions" in obs else None
        )

        return {
            "images": image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def to_dict(self):
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["rewards"] = self.rewards

        return env_output_dict


@dataclass(kw_only=True)
class EmbodiedRolloutResult:
    # required
    prev_logprobs: list[torch.Tensor] = field(default_factory=list)
    prev_values: list[torch.Tensor] = field(default_factory=list)
    dones: list[torch.Tensor] = field(default_factory=list)
    rewards: list[torch.Tensor] = field(default_factory=list)

    forward_inputs: list[dict[str, Any]] = field(default_factory=list)

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

        self.forward_inputs = [
            put_tensor_cpu(forward_inputs) for forward_inputs in self.forward_inputs
        ]

    def append_result(self, result: dict[str, Any]):
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

    def to_splited_dict(self, split_size) -> list[dict[str, Any]]:
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
