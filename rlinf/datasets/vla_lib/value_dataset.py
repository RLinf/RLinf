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

"""
Value Dataset for VLM-based return prediction.

This module provides a dataset that extends LeRobotRLDataset to format samples
for training a PI0.5 model in VLM mode to predict discretized returns.

Inheritance chain:
    ValueDataset -> LeRobotRLDataset -> LeRobotPyTorchDataset -> UnifiedDatasetInterface

The key difference from VLA training:
- VLA: predicts actions via flow matching
- Value: predicts return token via cross-entropy loss (VLM mode)

Sample output format (compatible with PI05DataCollator in VLM mode):
{
    'images': Dict[str, Tensor],      # Camera images
    'image_masks': Dict[str, Tensor], # Image validity masks
    'prompt': str,                    # Task instruction
    'state': Tensor,                  # Robot state (for optional discretization)
    'prefix': str,                    # "Value: " (configurable)
    'response': str,                  # The return_token, e.g., "42"
    'actions': None,                  # Explicitly None to trigger VLM mode
}
"""

import logging
from typing import Any, Optional

import torch

from rlinf.datasets.factory import register_dataset
from rlinf.datasets.vla_lib.lerobot_datasets.config import DataConfigFactory

from .config import RLDataConfig, create_rl_config, load_return_range_from_norm_stats
from .rl_dataset import LeRobotRLDataset

logger = logging.getLogger(__name__)


@register_dataset("rl/value", [r"value/", r"value_"])
class ValueDataset(LeRobotRLDataset):
    """Dataset for value prediction training.

    Extends LeRobotRLDataset with VLM-mode sample formatting for training
    models to predict discretized return tokens.

    Inheritance chain:
        ValueDataset -> LeRobotRLDataset -> LeRobotPyTorchDataset -> UnifiedDatasetInterface

    The sample structure matches PI05DataCollator expectations for VLM mode:
    - 'images': camera images
    - 'prompt': task instruction
    - 'state': robot state (used for discretization in PI0.5)
    - 'prefix': text preceding the value token (e.g., "Value: ")
    - 'response': the return_token to predict (e.g., "42")
    - 'actions': None (triggers VLM-only forward)
    """

    def __init__(
        self,
        dataset_path: str | None = None,
        repo_id: str | None = None,
        # Value-specific configuration
        value_prefix: str = "Value: ",
        return_token_key: str = "return_token",
        include_state: bool = True,
        skip_vlm_response: bool = False,  # Skip VLM response for expert-only mode
        # RL configuration (either provide this OR the individual params below)
        rl_config: Optional[RLDataConfig] = None,
        # Individual RL params (used only if rl_config is None)
        history_length: int = 0,
        history_keys: Optional[list[str]] = None,
        action_horizon: int = 10,
        gamma: float = 0.99,
        include_next_obs: bool = False,  # Set True for distributional RL
        num_return_bins: int = 201,
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
    ):
        """Initialize value dataset.

        Args:
            dataset_path: LeRobot dataset path or repo ID
            repo_id: Alias for dataset_path
            value_prefix: Prefix text before value prediction (e.g., "Value: ")
            return_token_key: Key for discretized return token
            include_state: Whether to include state in output (for discretization)
            rl_config: Complete RL config. If provided, individual RL params are ignored.
            history_length: Number of past observations
            history_keys: Keys to include in history
            action_horizon: Number of future actions/rewards
            gamma: Discount factor
            num_return_bins: Number of bins for discretization (paper uses 201)
            return_norm_stats_path: Path to norm_stats.json for min/max
            return_min: Override minimum return value
            return_max: Override maximum return value
            normalize_to_minus_one_zero: Normalize returns to (-1, 0) range
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
        # Store value-specific config before calling parent
        self.value_prefix = value_prefix
        self.include_state = include_state
        self.skip_vlm_response = skip_vlm_response

        # Build rl_config from individual params if not provided
        if rl_config is None:
            rl_config = create_rl_config(
                history_length=history_length,
                history_keys=history_keys,
                action_horizon=action_horizon,
                include_next_obs=include_next_obs,  # True for distributional RL
                include_return=True,  # Required for value training
                include_done=False,  # Not needed for offline value training
                gamma=gamma,
                discretize_return=True,  # Required for value training
                num_return_bins=num_return_bins,
                return_norm_stats_path=return_norm_stats_path,
                return_min=return_min,
                return_max=return_max,
                normalize_to_minus_one_zero=normalize_to_minus_one_zero,
            )
        elif not rl_config.discretize_return:
            raise ValueError(
                "ValueDataset requires discretize_return=True in rl_config. "
                "Value training predicts discretized return tokens."
            )

        # Use return_token_key from rl_config
        self.return_token_key = (
            rl_config.return_token_key
            if return_token_key == "return_token"
            else return_token_key
        )

        # Initialize parent LeRobotRLDataset with only rl_config
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
        )

        logger.info("ValueDataset initialized:")
        logger.info(f"  Value prefix: '{self.value_prefix}'")
        logger.info(f"  Return token key: '{self.return_token_key}'")
        logger.info(f"  Include state: {self.include_state}")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample formatted for value prediction training.

        Extends parent's __getitem__ to format samples for value training.
        Supports both VLM mode (token prediction) and expert mode (continuous value).

        Returns:
            Dict with:
                - images: Dict[str, Tensor] camera images
                - image_masks: Dict[str, Tensor] (optional)
                - prompt: str task instruction
                - state: Tensor robot state (if include_state=True)
                - prefix: str value prefix (for VLM/dual modes)
                - response: str return token (for VLM/dual modes)
                - actions: None (explicitly for VLM mode)
                - target_values: float continuous return (for expert/dual modes)
        """
        # Get RL sample from parent
        rl_sample = super().__getitem__(idx)

        # Get target value (normalized return)
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

        # Format for value training
        # For expert-only mode, skip VLM response entirely (set to None, not empty)
        sample = {
            "target_values": target_value,
        }

        if self.skip_vlm_response:
            # Expert-only mode: no VLM tokens needed at all
            # Setting to None (not empty string) ensures no EOS/response tokens are added
            sample["prefix"] = None
            sample["response"] = None
        else:
            # VLM or dual mode: include text response for VLM loss
            sample["prefix"] = self.value_prefix
            sample["response"] = f"{target_value:.2f}"

        sample["actions"] = None  # Explicitly None to trigger VLM mode

        # Copy prompt
        if "prompt" in rl_sample:
            sample["prompt"] = rl_sample["prompt"]
        elif "task" in rl_sample:
            sample["prompt"] = rl_sample["task"]
        else:
            sample["prompt"] = "perform the task"

        # Extract images
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

        # Copy state if requested
        if self.include_state:
            if "state" in rl_sample:
                sample["state"] = rl_sample["state"]
            elif "state_tcp_pose" in rl_sample:
                state_parts = [rl_sample["state_tcp_pose"]]
                if "state_gripper_pose" in rl_sample:
                    state_parts.append(rl_sample["state_gripper_pose"])
                sample["state"] = (
                    torch.cat(state_parts, dim=-1)
                    if len(state_parts) > 1
                    else state_parts[0]
                )

        # Pass through raw return values for debugging and metrics
        if "return" in rl_sample:
            ret_val = rl_sample["return"]
            sample["return_raw"] = (
                ret_val.item() if isinstance(ret_val, torch.Tensor) else float(ret_val)
            )
        if "return_normalized" in rl_sample:
            sample["return_normalized"] = rl_sample["return_normalized"]
        if "return_bin_id" in rl_sample:
            sample["return_bin_id"] = rl_sample["return_bin_id"]

        # =====================================================================
        # Distributional RL fields (for n-step TD target)
        # These are used when expert_loss_type="distributional"
        # =====================================================================

        # Next observation (for computing V(s_{t+H}))
        # RL dataset applies the same VLA transforms to next obs, storing result
        # in 'next_observation' with same structure as current obs
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

        # Reward chunk (for n-step TD target)
        # RL dataset preserves original key name (e.g., 'reward' not 'reward_chunk')
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

                # Compute raw discounted reward sum
                if reward_is_pad is not None:
                    valid_mask = ~reward_is_pad.bool()
                    masked_rewards = reward_chunk * valid_mask.float()
                    reward_sum_raw = (masked_rewards * gamma_powers).sum().item()
                    sample["num_valid_rewards"] = valid_mask.sum().item()
                else:
                    reward_sum_raw = (reward_chunk * gamma_powers).sum().item()
                    sample["num_valid_rewards"] = n

                # Normalize reward_sum to match value range [-1, 0]
                # Use same normalization as returns: normalized = raw / |raw_return_min|
                if self.return_discretizer is not None:
                    sample["reward_sum"] = self.return_discretizer.normalize_value(
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

    def get_source_name(self) -> str:
        """Get a readable source name for this dataset."""
        base_name = self.repo_id.replace("/", "_").replace("-", "_").lower()
        return f"value_{base_name}"


def create_value_dataset(
    dataset_path: str | None = None,
    repo_id: str | None = None,
    # Value-specific config
    value_prefix: str = "Value: ",
    include_state: bool = True,
    skip_vlm_response: bool = False,  # Skip VLM response for expert-only mode
    # RL config (either provide this OR the individual params below)
    rl_config: Optional[RLDataConfig] = None,
    # Individual RL params (used only if rl_config is None)
    history_length: int = 0,
    history_keys: Optional[list[str]] = None,
    action_horizon: int = 10,
    gamma: float = 0.99,
    include_next_obs: bool = False,  # Set True for distributional RL
    num_return_bins: int = 201,
    return_norm_stats_path: Optional[str] = None,
    return_min: Optional[float] = None,
    return_max: Optional[float] = None,
    normalize_to_minus_one_zero: bool = True,
    # VLA config
    split: str = "train",
    robot_type: Optional[str] = None,
    model_type: Optional[str] = None,
    default_prompt: Optional[str] = None,
    norm_stats_dir: Optional[str] = None,
    asset_id: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
    action_dim: Optional[int] = None,
    max_samples: Optional[int] = None,
    action_norm_skip_dims: Optional[dict[str, list[int]]] = None,
    # Episode filtering
    episode_percentage: Optional[float] = None,
    shuffle_episodes: bool = False,
    episode_seed: int = 42,
) -> "ValueDataset":
    """Factory function to create a ValueDataset.

    This is a convenience wrapper around ValueDataset constructor.

    Args:
        dataset_path: Path to LeRobot dataset
        repo_id: Alias for dataset_path
        value_prefix: Prefix text for value prediction (e.g., "Value: ")
        include_state: Whether to include state for discretization
        rl_config: Complete RL config. If provided, individual RL params are ignored.
        history_length: Number of past observations (0 = current only)
        history_keys: Keys to include in history (auto-detected if None)
        action_horizon: Number of future steps for return calculation
        gamma: Discount factor
        num_return_bins: Number of bins for discretization (paper uses 201)
        return_norm_stats_path: Path to norm_stats.json for min/max
        return_min: Override minimum return value
        return_max: Override maximum return value
        normalize_to_minus_one_zero: Normalize returns to (-1, 0) range
        split: Dataset split
        robot_type: Robot type (franka, libero, etc.)
        model_type: Model type (pi0, pi05)
        default_prompt: Default prompt
        norm_stats_dir: Normalization stats directory
        asset_id: Asset ID
        config: Full config dict from YAML
        action_dim: Action dimension
        max_samples: Limit dataset size
        episode_percentage: Use subset of episodes
        shuffle_episodes: Random episode selection
        episode_seed: Seed for reproducibility

    Returns:
        ValueDataset instance
    """
    # Auto-load return_min/return_max from norm_stats if not specified
    if (return_min is None or return_max is None) and norm_stats_dir:
        loaded_min, loaded_max = load_return_range_from_norm_stats(
            norm_stats_dir, asset_id
        )
        if loaded_min is not None and loaded_max is not None:
            return_min = return_min if return_min is not None else loaded_min
            return_max = return_max if return_max is not None else loaded_max
            logger.info(
                f"Loaded return range from norm_stats: [{return_min}, {return_max}]"
            )

    if return_min is None or return_max is None:
        raise ValueError(
            "return_min and return_max must be specified. "
            "Either set them explicitly or ensure norm_stats_dir contains 'return' key."
        )

    # Build rl_config from individual params if not provided
    if rl_config is None:
        rl_config = create_rl_config(
            history_length=history_length,
            history_keys=history_keys,
            action_horizon=action_horizon,
            include_next_obs=include_next_obs,  # True for distributional RL
            include_return=True,
            gamma=gamma,
            discretize_return=True,
            num_return_bins=num_return_bins,
            return_norm_stats_path=return_norm_stats_path,
            return_min=return_min,
            return_max=return_max,
            normalize_to_minus_one_zero=normalize_to_minus_one_zero,
        )

    return ValueDataset(
        dataset_path=dataset_path or repo_id,
        value_prefix=value_prefix,
        include_state=include_state,
        skip_vlm_response=skip_vlm_response,
        rl_config=rl_config,
        split=split,
        robot_type=robot_type,
        model_type=model_type,
        default_prompt=default_prompt,
        norm_stats_dir=norm_stats_dir,
        asset_id=asset_id,
        config=config,
        action_dim=action_dim,
        max_samples=max_samples,
        action_norm_skip_dims=action_norm_skip_dims,
        episode_percentage=episode_percentage,
        shuffle_episodes=shuffle_episodes,
        episode_seed=episode_seed,
    )
