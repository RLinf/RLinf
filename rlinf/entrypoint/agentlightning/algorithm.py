from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from agentlightning.algorithm.base import Algorithm
from agentlightning.types import Dataset

from .entrypoint import run_rlinf_training

logger = logging.getLogger(__name__)


class RlinfAlgorithm(Algorithm):
    """Add rlinf tag to agentlightning algorithm."""

    def __init__(
        self,
        config: dict[str, Any] | DictConfig,
        eval: bool = False,
        eval_checkpoint_dir: Optional[str] = None,
    ):
        super().__init__()

        if isinstance(config, dict):
            self.config = OmegaConf.create(config)
        else:
            self.config = config
        self.eval = eval
        self.eval_checkpoint_dir = eval_checkpoint_dir

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
            eval_checkpoint_dir=self.eval_checkpoint_dir,
        )

