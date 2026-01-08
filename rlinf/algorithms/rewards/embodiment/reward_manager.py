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

"""Reward Manager for unified reward computation across RL algorithms.

This module provides a RewardManager class that serves as the unified interface
for reward computation in embodied RL. It supports multiple reward model types
through a registry pattern, making it easy to switch between different models
and add new ones.
"""

import logging
from typing import Any, Dict, List, Optional, Type

import torch
from omegaconf import DictConfig

from rlinf.algorithms.rewards.embodiment.base_reward_model import BaseRewardModel

logger = logging.getLogger(__name__)


class RewardManager:
    """Unified reward computation interface for all RL algorithms.

    The RewardManager provides a single entry point for reward computation,
    automatically selecting and configuring the appropriate reward model
    based on configuration. It is designed to be framework-agnostic,
    working seamlessly with PPO, SAC, GRPO, and other RL algorithms.

    The manager uses a registry pattern to support different reward model
    types, making it easy to add new models without modifying existing code.

    Example usage:
        ```python
        # Initialize manager from config
        reward_manager = RewardManager(cfg.reward)

        # Compute rewards
        rewards = reward_manager.compute_rewards(observations, task_descriptions)
        ```

    Attributes:
        cfg: Configuration dictionary for reward computation.
        model: The instantiated reward model.
        model_type: String identifier for the active model type.
    """

    # Registry of available reward model types
    # Maps model type string to model class
    REGISTRY: Dict[str, Type[BaseRewardModel]] = {}

    def __init__(self, cfg: DictConfig):
        """Initialize the RewardManager.

        Args:
            cfg: Configuration dictionary containing:
                - use_reward_model: Whether to use model-based reward (default: True).
                - reward_model_type: Type of reward model ("resnet", "qwen3_vl", etc.).
                - Model-specific configuration under the model type key.

        Raises:
            ValueError: If the specified reward model type is not registered.
        """
        self.cfg = cfg
        use_model = cfg.get("use_reward_model", True)
        self.use_reward_model = bool(use_model) if hasattr(use_model, '__bool__') else use_model
        self.model_type = cfg.get("reward_model_type", "resnet")
        self.model: Optional[BaseRewardModel] = None

        if self.use_reward_model:
            self._build_model()

    @classmethod
    def register(cls, name: str) -> callable:
        """Decorator to register a reward model class.

        Args:
            name: String identifier for the model type.

        Returns:
            Decorator function that registers the class.

        Example:
            ```python
            @RewardManager.register("custom")
            class CustomRewardModel(BaseRewardModel):
                pass
            ```
        """
        def decorator(model_cls: Type[BaseRewardModel]) -> Type[BaseRewardModel]:
            cls.REGISTRY[name] = model_cls
            return model_cls
        return decorator

    @classmethod
    def register_model(cls, name: str, model_cls: Type[BaseRewardModel]) -> None:
        """Register a reward model class programmatically.

        Args:
            name: String identifier for the model type.
            model_cls: The reward model class to register.
        """
        cls.REGISTRY[name] = model_cls

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of registered model types.

        Returns:
            List of registered model type names.
        """
        return list(cls.REGISTRY.keys())

    def _build_model(self) -> None:
        """Build the reward model based on configuration.

        Raises:
            ValueError: If the model type is not registered.
        """
        if self.model_type not in self.REGISTRY:
            available = ", ".join(self.REGISTRY.keys())
            raise ValueError(
                f"Unknown reward model type: '{self.model_type}'. "
                f"Available types: {available}"
            )

        # Get model-specific configuration
        model_cfg = self.cfg.get(self.model_type, {})

        # Merge with common configuration
        merged_cfg = DictConfig({
            **self.cfg,
            **model_cfg,
        })

        # Instantiate model
        model_cls = self.REGISTRY[self.model_type]
        self.model = model_cls(merged_cfg)

        logger.info(f"Initialized reward model: {self.model_type}")

    def compute_rewards(
        self,
        observations: Dict[str, Any],
        task_descriptions: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """Compute rewards from observations.

        This is the main interface method for reward computation. It handles
        both model-based and rule-based rewards based on configuration.

        Args:
            observations: Dictionary containing observation data:
                - For image models: 'images' or 'main_images' key with tensor
                  of shape [B, C, H, W] or [B, H, W, C].
                - For video models: tensor of shape [B, T, C, H, W].
                - Optional 'states' key with state tensor.
            task_descriptions: Optional list of task descriptions for VLM models.

        Returns:
            torch.Tensor: Reward tensor of shape [B].

        Raises:
            RuntimeError: If use_reward_model is True but model is not initialized.
        """
        if not self.use_reward_model:
            # Return zeros if reward model is disabled
            batch_size = self._get_batch_size(observations)
            return torch.zeros(batch_size)

        if self.model is None:
            raise RuntimeError(
                "Reward model is not initialized. "
                "Set use_reward_model=True and provide valid configuration."
            )

        return self.model.compute_reward(observations, task_descriptions)

    def _get_batch_size(self, observations: Dict[str, Any]) -> int:
        """Extract batch size from observations.

        Args:
            observations: Observation dictionary.

        Returns:
            int: Batch size.
        """
        images = observations.get("images")
        if images is None:
            images = observations.get("main_images")
        if images is not None:
            return images.shape[0]

        states = observations.get("states")
        if states is not None:
            return states.shape[0]

        return 1

    def to_device(self, device: torch.device) -> "RewardManager":
        """Move reward model to specified device.

        Args:
            device: Target device.

        Returns:
            Self for method chaining.
        """
        if self.model is not None:
            self.model.to_device(device)
        return self

    @property
    def is_enabled(self) -> bool:
        """Check if reward model is enabled and initialized."""
        return self.use_reward_model and self.model is not None


# Import and register default models
# This is done at module load time to populate the registry
def _register_default_models():
    """Register default reward models in the registry."""
    try:
        from rlinf.algorithms.rewards.embodiment.resnet_reward_model import (
            ResNetRewardModel,
        )
        RewardManager.register_model("resnet", ResNetRewardModel)
    except ImportError as e:
        logger.warning(f"Failed to register ResNetRewardModel: {e}")

    try:
        from rlinf.algorithms.rewards.embodiment.qwen3_vl_reward_model import (
            Qwen3VLRewardModel,
        )
        RewardManager.register_model("qwen3_vl", Qwen3VLRewardModel)
    except ImportError as e:
        logger.warning(f"Failed to register Qwen3VLRewardModel: {e}")


# Register default models when module is imported
_register_default_models()

