#!/usr/bin/env python3
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

"""Standalone script for training ResNet reward model.

Usage:
    python train_reward_model.py --config-name maniskill_train_reward_model
"""

import json
import logging
import os
from pathlib import Path

# Auto-set EMBODIED_PATH if not set
if "EMBODIED_PATH" not in os.environ:
    os.environ["EMBODIED_PATH"] = str(Path(__file__).parent.resolve())

import hydra
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_reward_model_train(cfg: DictConfig) -> None:
    """Train reward model from collected data.

    This is a standalone supervised learning task, separate from RL training.
    """
    from rlinf.algorithms.rewards.embodiment.reward_model_trainer import (
        RewardModelTrainer,
    )
    from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel

    logger.info("Starting reward model training...")

    train_cfg = cfg.get("reward_model_training", {})

    if not train_cfg.get("data_path"):
        raise ValueError("reward_model_training.data_path must be specified")

    # Create model
    model_cfg = cfg.get("reward", {}).get("resnet", {})
    model = ResNetRewardModel(DictConfig(model_cfg))

    # Create trainer with config and model
    trainer = RewardModelTrainer(train_cfg, model)

    # Train
    data_path = train_cfg.get("data_path")
    results = trainer.train(data_path)

    logger.info(
        f"Training completed! Best val accuracy: {results.get('best_val_acc', 'N/A')}"
    )


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_train_reward_model"
)
def main(cfg: DictConfig) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    run_reward_model_train(cfg)


if __name__ == "__main__":
    main()
