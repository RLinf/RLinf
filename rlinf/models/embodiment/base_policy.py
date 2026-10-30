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

from abc import ABC, abstractmethod
from enum import Enum


def load_state_dict(model, state_dict, allowed_missing_prefixes=()) -> None:
    """Load a checkpoint while allowing selected new model components."""
    if not allowed_missing_prefixes:
        model.load_state_dict(state_dict)
        return
    incompatible = model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(tuple(allowed_missing_prefixes))
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )


class ForwardType(Enum):
    DEFAULT = "default"
    SFT = "sft"
    SAC = "sac"
    SAC_Q = "sac_q"
    CROSSQ = "crossq"
    CROSSQ_Q = "crossq_q"
    IQL_ACTOR = "iql_actor"
    IQL_CRITIC = "iql_critic"
    IQL_VALUE = "iql_value"
    NFT = "nft"
    OGPO_SAMPLE = "ogpo_sample"
    OGPO_LOG_PROB = "ogpo_log_prob"
    OGPO_BC = "ogpo_bc"


class BasePolicy(ABC):
    """
    Base interface for all policies.

    Subclasses must implement:
        - forward
        - default_forward
        - predict_action_batch

    Optional overrides:
        - sft_forward
        - sac_forward
        - sac_q_forward
        - crossq_forward
        - crossq_q_forward
        - iql_forward
        - prepare_dagger_sft_batch
    """

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def sac_forward(self, **kwargs):
        raise NotImplementedError

    def sac_q_forward(self, **kwargs):
        raise NotImplementedError

    def crossq_forward(self, **kwargs):
        raise NotImplementedError

    def crossq_q_forward(self, **kwargs):
        raise NotImplementedError

    def iql_forward(self, **kwargs):
        raise NotImplementedError

    def prepare_dagger_sft_batch(self, batch):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support DAgger SFT training."
        )

    @abstractmethod
    def default_forward(self, **kwargs): ...

    @abstractmethod
    def predict_action_batch(self, **kwargs): ...

    def enable_torch_compile(
        self,
        mode: str = "max-autotune-no-cudagraphs",
    ):
        raise NotImplementedError(
            "torch compile is not supported for current policy, please set `enable_torch_compile=False` for now"
        )

    def capture_cuda_graph(self, train_batch_size: int, eval_batch_size: int):
        raise NotImplementedError(
            "cuda graph is not supported for current policy, please set `enable_cuda_graph=False` for now"
        )

    def release_cuda_graph(self):
        from rlinf.utils.cuda_graph import CUDAGraphManager

        if self.is_cuda_graph_enabled():
            self.cuda_graph_manager: CUDAGraphManager
            self.cuda_graph_manager.destroy()
            self.cuda_graph_manager = None

    def is_cuda_graph_enabled(self) -> bool:
        return (
            hasattr(self, "cuda_graph_manager") and self.cuda_graph_manager is not None
        )
