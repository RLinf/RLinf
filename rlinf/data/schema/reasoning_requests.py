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

"""Reasoning request and sequence-group structures."""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional, Union

import torch

from rlinf.utils.data_iter_utils import split_list

if TYPE_CHECKING:
    from vllm.outputs import RequestOutput as VllmRequestOutput


def get_batch_size(
    batch: dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
    return batch[batch_tensor_key].size(0)


def get_seq_length(
    batch: dict[str, torch.Tensor], batch_tensor_key: str = "input_ids"
) -> int:
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


def build_rollout_requests_from_batch(
    batch: dict[str, Any],
    *,
    group_size: int,
    split_size: int,
    enforce_divisible_batch: bool = False,
) -> list[RolloutRequest]:
    """Convert one collated batch into split rollout requests."""
    prompt_ids = batch["prompt"].tolist()
    lengths = batch["length"].tolist()
    answers = batch["answer"]
    image_data = batch["image_data"]
    multi_modal_inputs = batch["multi_modal_inputs"]
    prompt_ids = [ids[-pmp_len:] for ids, pmp_len in zip(prompt_ids, lengths)]

    requests = []
    for (
        split_prompt_ids,
        split_answers,
        split_image_data,
        split_multi_modal_inputs,
    ) in zip(
        split_list(
            prompt_ids, split_size, enforce_divisible_batch=enforce_divisible_batch
        ),
        split_list(
            answers, split_size, enforce_divisible_batch=enforce_divisible_batch
        ),
        split_list(
            image_data, split_size, enforce_divisible_batch=enforce_divisible_batch
        ),
        split_list(
            multi_modal_inputs,
            split_size,
            enforce_divisible_batch=enforce_divisible_batch,
        ),
    ):
        requests.append(
            RolloutRequest(
                n=group_size,
                input_ids=split_prompt_ids,
                answers=split_answers,
                image_data=split_image_data,
                multi_modal_inputs=split_multi_modal_inputs,
            )
        )
    return requests


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


__all__ = [
    "FinishReasonEnum",
    "RolloutRequest",
    "build_rollout_requests_from_batch",
    "SeqGroupInfo",
    "get_batch_size",
    "get_seq_length",
]
