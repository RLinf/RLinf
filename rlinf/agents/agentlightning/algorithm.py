# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import typing
from typing import Any, Optional

if typing.TYPE_CHECKING:
    from agentlightning.types import Dataset

from omegaconf import DictConfig, OmegaConf

_RLinf: type | None = None


def _make_rlinf_class() -> type:
    from agentlightning.algorithm.base import Algorithm

    from .entrypoint import run_rlinf_training

    class RLinf(Algorithm):
        """Agent Lightning ``Algorithm`` that runs RL training on the RLinf stack.

        In Agent Lightning, an ``Algorithm`` is the training strategy wired into
        ``Trainer``: it uses the shared ``LightningStore`` and trace adapter from the
        trainer and implements ``run`` to drive rollouts and learning.

        ``RLinf`` is RLinf's implementation of that hook. It calls
        ``run_rlinf_training``, which constructs RLinf cluster placement, distributed
        workers (rollout, inference, actor, …), and the AgentLightning training or eval
        runner—RLinf's RL training and resource-management path, configured by a Hydra
        ``DictConfig``.

        Args:
            config: RLinf config (``DictConfig`` or dict), typically from ``@hydra.main``.
            eval: If True, use evaluation routing instead of training.
        """

        def __init__(
            self,
            config: dict[str, Any] | DictConfig,
            eval: bool = False,
        ):
            super().__init__()

            if isinstance(config, dict):
                self.config = OmegaConf.create(config)
            else:
                self.config = config
            self.eval = eval

        def run(
            self,
            train_dataset: Optional[Dataset[Any]] = None,
            val_dataset: Optional[Dataset[Any]] = None,
        ) -> None:
            store = self.get_store()
            adapter = self.get_adapter()
            run_rlinf_training(
                config=self.config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                store=store,
                adapter=adapter,
                eval=self.eval,
            )

    return RLinf


def __getattr__(name: str):
    global _RLinf
    if name == "RLinf":
        if _RLinf is None:
            _RLinf = _make_rlinf_class()
        return _RLinf
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
