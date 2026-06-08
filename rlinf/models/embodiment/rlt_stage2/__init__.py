"""RLT Stage 2 model package."""

from __future__ import annotations

from omegaconf import DictConfig

from .rlt_stage2_policy import RLTStage2Policy


def get_model(cfg: DictConfig, torch_dtype=None):
    del torch_dtype
    return RLTStage2Policy(cfg)
