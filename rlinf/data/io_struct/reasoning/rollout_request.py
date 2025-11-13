from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from rlinf.utils.data_iter_utils import split_list

@dataclass
class RolloutRequest:
    """
    Attr
    input_ids: List of input token IDs for rollout
    n: Number of completions to generate for each input
    image_data: list of image data (bytes or URLs) for multimodal inputs
    answers: Optional list of answers for the requests, if available
    multi_modal_inputs: list of multi-modal inputs for the requests
    """

    n: int
    input_ids: List[List[int]]
    image_data: Union[List[List[bytes]], List[List[str]]]
    answers: List[str]
    multi_modal_inputs: List[Dict]

    def repeat(self) -> "RolloutRequest":
        """Repeat each input in the RolloutRequest a specified number of times.

        Args:
            times (int): The number of times to repeat each input.

        Returns:
            RolloutRequest: A new RolloutRequest with repeated inputs.
        """
        assert self.n > 0, "n must be greater than 0"

        input_ids, answers, image_data, multi_modal_inputs = zip(
            *[
                (input_id, answer, image_data, multi_modal_inputs)
                for input_id, answer, image_data, multi_modal_inputs in zip(
                    self.input_ids,
                    self.answers,
                    self.image_data,
                    self.multi_modal_inputs,
                )
                for _ in range(self.n)
            ]
        )
        return RolloutRequest(
            n=self.n,
            input_ids=list(input_ids),
            answers=list(answers),
            image_data=list(image_data),
            multi_modal_inputs=list(multi_modal_inputs),
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
        image_data_split_list = split_list(self.image_data, num_splits)
        multi_modal_inputs_split_list = split_list(self.multi_modal_inputs, num_splits)

        splitted_requests = []
        for (
            input_ids_batch,
            answers_batch,
            image_data_batch,
            multi_modal_inputs_batch,
        ) in zip(
            input_ids_split_list,
            answers_split_list,
            image_data_split_list,
            multi_modal_inputs_split_list,
        ):
            request = RolloutRequest(
                n=self.n,
                input_ids=input_ids_batch,
                answers=answers_batch,
                image_data=image_data_batch,
                multi_modal_inputs=multi_modal_inputs_batch,
            )
            splitted_requests.append(request)

        return splitted_requests

    def repeat_and_split(
        self, rollout_batch_size: Optional[int] = None
    ) -> List["RolloutRequest"]:
        input_ids, answers, image_data, multi_modal_inputs = zip(
            *[
                (input_id, answer, image_data, multi_modal_inputs)
                for input_id, answer, image_data, multi_modal_inputs in zip(
                    self.input_ids,
                    self.answers,
                    self.image_data,
                    self.multi_modal_inputs,
                )
                for _ in range(self.n)
            ]
        )
        input_ids, answers, image_data, multi_modal_inputs = (
            list(input_ids),
            list(answers),
            list(image_data),
            list(multi_modal_inputs),
        )

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
        image_data_split_list = split_list(image_data, num_batches)
        multi_modal_inputs_split_list = split_list(multi_modal_inputs, num_batches)

        for (
            input_ids_batch,
            answers_batch,
            image_data_batch,
            multi_modal_inputs_batch,
        ) in zip(
            input_ids_split_list,
            answers_split_list,
            image_data_split_list,
            multi_modal_inputs_split_list,
        ):
            request = RolloutRequest(
                n=self.n,
                input_ids=input_ids_batch,
                answers=answers_batch,
                image_data=image_data_batch,
                multi_modal_inputs=multi_modal_inputs_batch,
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