"""RLT Stage 1 model package."""

from __future__ import annotations

from omegaconf import DictConfig

from .rlt_stage1_policy import RLTStage1Policy


def get_model(cfg: DictConfig, torch_dtype=None):
    del torch_dtype
    return RLTStage1Policy(cfg)
