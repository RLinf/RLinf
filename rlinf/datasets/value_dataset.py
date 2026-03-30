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

"""
Value Dataset for return prediction.

This module provides a dataset that extends LeRobotRLDataset to format samples
for training a value model to predict normalized returns.

Inheritance chain:
    ValueDataset -> LeRobotRLDataset -> LeRobotPyTorchDataset -> Dataset

Sample output format (compatible with ValueDataCollator):
{
    'images': Dict[str, Tensor],      # Camera images
    'image_masks': Dict[str, Tensor], # Image validity masks
    'prompt': str,                    # Task instruction
    'target_values': float,           # Normalized return value
    'actions': None,                  # Explicitly None to trigger VLM mode
}
"""

import logging
from typing import Any, Optional

import torch

from rlinf.datasets.lerobot.config import DataConfigFactory

from .config import RLDataConfig, create_rl_config
from .rl_dataset import LeRobotRLDataset

logger = logging.getLogger(__name__)


class ValueDataset(LeRobotRLDataset):
    """Dataset for value prediction training.

    Extends LeRobotRLDataset with VLM-mode sample formatting for training
    models to predict normalized return values.

    Inheritance chain:
        ValueDataset -> LeRobotRLDataset -> LeRobotPyTorchDataset -> Dataset

    The sample structure matches ValueDataCollator expectations:
    - 'images': camera images
    - 'prompt': task instruction
    - 'target_values': normalized return value
    - 'actions': None (triggers VLM-only forward)
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        repo_id: str | None = None,
        # RL configuration (either provide this OR the individual params below)
        rl_config: Optional[RLDataConfig] = None,
        # Individual RL params (used only if rl_config is None)
        history_length: int = 0,
        history_keys: Optional[list[str]] = None,
        action_horizon: int = 10,
        gamma: float = 0.99,
        include_next_obs: bool = False,  # Set True for distributional RL
        return_norm_stats_path: Optional[str] = None,
        return_min: Optional[float] = None,
        return_max: Optional[float] = None,
        normalize_to_minus_one_zero: bool = True,
        # VLA dataset configuration (inherited)
        split: str = "train",
        data_config_factory: Optional[DataConfigFactory] = None,
        action_dim: Optional[int] = None,
        robot_type: Optional[str] = None,
        model_type: Optional[str] = None,
        default_prompt: Optional[str] = None,
        extra_delta_transform: bool = False,
        norm_stats_dir: Optional[str] = None,
        asset_id: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        max_samples: Optional[int] = None,
        action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
        # Episode filtering
        episode_percentage: Optional[float] = None,
        shuffle_episodes: bool = False,
        episode_seed: int = 42,
        # Sidecar tag (e.g. returns_{tag}.parquet)
        tag: Optional[str] = None,
    ):
        """Initialize value dataset.

        Args:
            dataset_path: LeRobot dataset path or repo ID
            repo_id: Alias for dataset_path
            rl_config: Complete RL config. If provided, individual RL params are ignored.
            history_length: Number of past observations
            history_keys: Keys to include in history
            action_horizon: Number of future actions/rewards
            gamma: Discount factor
            return_norm_stats_path: Path to norm_stats.json for min/max
            return_min: Override minimum return value
            return_max: Override maximum return value
            split: Dataset split
            data_config_factory: Factory for transforms
            action_dim: Action dimension
            robot_type: Robot type for auto-config
            model_type: Model type (pi0, pi05)
            default_prompt: Default prompt
            extra_delta_transform: Apply extra delta transform
            norm_stats_dir: Normalization stats directory
            asset_id: Asset ID
            config: Full config dict from YAML
            max_samples: Limit dataset size
            episode_percentage: Percentage of episodes to use
            shuffle_episodes: Random episode selection
            episode_seed: Seed for reproducibility
        """
        if rl_config is None:
            rl_config = create_rl_config(
                history_length=history_length,
                history_keys=history_keys,
                action_horizon=action_horizon,
                include_next_obs=include_next_obs,  # True for distributional RL
                include_return=True,  # Required for value training
                include_done=False,  # Not needed for offline value training
                gamma=gamma,
                normalize_return=True,
                return_norm_stats_path=return_norm_stats_path,
                return_min=return_min,
                return_max=return_max,
                normalize_to_minus_one_zero=normalize_to_minus_one_zero,
            )
        elif not rl_config.normalize_return or not rl_config.include_return:
            raise ValueError(
                "ValueDataset requires normalize_return=True and include_return=True."
            )
        super().__init__(
            dataset_path=dataset_path,
            repo_id=repo_id,
            rl_config=rl_config,
            split=split,
            data_config_factory=data_config_factory,
            action_dim=action_dim,
            robot_type=robot_type,
            model_type=model_type,
            default_prompt=default_prompt,
            extra_delta_transform=extra_delta_transform,
            norm_stats_dir=norm_stats_dir,
            asset_id=asset_id,
            config=config,
            max_samples=max_samples,
            action_norm_skip_dims=action_norm_skip_dims,
            episode_percentage=episode_percentage,
            shuffle_episodes=shuffle_episodes,
            episode_seed=episode_seed,
            tag=tag,
        )

        logger.info("ValueDataset initialized")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample formatted for value prediction training.

        Extends parent's __getitem__ to format samples for value training.

        Returns:
            Dict with:
                - images: Dict[str, Tensor] camera images
                - image_masks: Dict[str, Tensor] (optional)
                - prompt: str task instruction
                - target_values: float continuous return
                - actions: None (explicitly for VLM mode)
        """
        # Get RL sample from parent
        rl_sample = super().__getitem__(idx)

        if "return_normalized" in rl_sample:
            ret_norm = rl_sample["return_normalized"]
            target_value = (
                ret_norm.item() if hasattr(ret_norm, "item") else float(ret_norm)
            )
        elif "return" in rl_sample:
            ret_val = rl_sample["return"]
            target_value = (
                ret_val.item() if hasattr(ret_val, "item") else float(ret_val)
            )
        else:
            target_value = 0.0

        sample = {
            "target_values": target_value,
            "actions": None,  # None triggers VLM-only forward pass
        }

        if "prompt" in rl_sample:
            sample["prompt"] = rl_sample["prompt"]
        elif "task" in rl_sample:
            sample["prompt"] = rl_sample["task"]
        else:
            sample["prompt"] = "perform the task"

        images = {}
        image_masks = {}

        if "image" in rl_sample and isinstance(rl_sample["image"], dict):
            images = rl_sample["image"]
        elif "images" in rl_sample and isinstance(rl_sample["images"], dict):
            images = rl_sample["images"]
        else:
            for key in rl_sample:
                if "image" in key.lower() and isinstance(rl_sample[key], torch.Tensor):
                    if rl_sample[key].dim() >= 3:
                        cam_name = (
                            key.replace("observation.images.", "")
                            .replace("images.", "")
                            .replace("images_", "")
                        )
                        images[cam_name] = rl_sample[key]

        if "image_mask" in rl_sample:
            image_masks = rl_sample["image_mask"]
        elif "image_masks" in rl_sample:
            image_masks = rl_sample["image_masks"]

        sample["images"] = images
        if image_masks:
            sample["image_masks"] = image_masks

        if "return" in rl_sample:
            ret_val = rl_sample["return"]
            sample["return_raw"] = (
                ret_val.item() if isinstance(ret_val, torch.Tensor) else float(ret_val)
            )
        if "return_normalized" in rl_sample:
            sample["return_normalized"] = rl_sample["return_normalized"]

        next_obs = rl_sample.get("next_observation", {})
        if next_obs:
            if not getattr(self, "_logged_next_keys", False):
                logger.info(f"Next observation keys: {list(next_obs.keys())}")
                self._logged_next_keys = True

            if next_obs.get("images"):
                sample["next_images"] = next_obs["images"]
            if next_obs.get("state") is not None:
                sample["next_state"] = next_obs["state"]
            sample["next_state_is_pad"] = next_obs.get("is_pad", False)

        reward_key = "reward"
        if reward_key in rl_sample:
            reward_chunk = rl_sample[reward_key]
            sample["rewards"] = reward_chunk
            reward_is_pad = rl_sample.get(f"{reward_key}_is_pad")

            # Compute discounted n-step reward sum
            gamma = (
                getattr(self.rl_config, "gamma", 0.99)
                if hasattr(self, "rl_config")
                else 0.99
            )

            if isinstance(reward_chunk, torch.Tensor):
                n = reward_chunk.shape[0]
                gamma_powers = torch.tensor(
                    [gamma**i for i in range(n)], dtype=reward_chunk.dtype
                )

                if reward_is_pad is not None:
                    valid_mask = ~reward_is_pad.bool()
                    masked_rewards = reward_chunk * valid_mask.float()
                    reward_sum_raw = (masked_rewards * gamma_powers).sum().item()
                    sample["num_valid_rewards"] = valid_mask.sum().item()
                else:
                    reward_sum_raw = (reward_chunk * gamma_powers).sum().item()
                    sample["num_valid_rewards"] = n

                # Normalize reward_sum using same scale as returns
                if self.return_normalizer is not None:
                    sample["reward_sum"] = self.return_normalizer.normalize_value(
                        reward_sum_raw
                    )
                else:
                    sample["reward_sum"] = reward_sum_raw

        # Done flag (terminal within action horizon)
        # Use combination of explicit done flag and padding information
        if "done" in rl_sample:
            done = rl_sample["done"]
            sample["dones"] = done.item() if hasattr(done, "item") else bool(done)
        elif sample.get("next_state_is_pad", False):
            # If next_state is padded, episode ended before t+H
            sample["dones"] = True
        else:
            sample["dones"] = False

        return sample
