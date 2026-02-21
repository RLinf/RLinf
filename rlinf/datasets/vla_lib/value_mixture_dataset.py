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
Value Mixture Dataset for training on multiple RL datasets.

This module provides a mixture dataset for value training that combines
multiple ValueDataset instances with weighted sampling.

Inheritance chain:
    ValueMixtureDataset uses ValueDataset instances (via composition)
"""

import hashlib
from pathlib import Path
from typing import (
    Any,
    Optional,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

import numpy as np
import yaml
from vla_lib import get_dataset_config_dir
from vla_lib.datasets.base_interface import UnifiedDatasetInterface
from vla_lib.datasets.factory import register_dataset
from vla_lib.utils.dist_utils import get_logger

from .config import load_return_range_from_norm_stats
from .value_dataset import ValueDataset, create_value_dataset


@runtime_checkable
class SizedDataset(Protocol):
    """Protocol for datasets that have both __len__ and __getitem__."""

    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> Any: ...


logger = get_logger(__name__)


def _safe_hash(input_tuple) -> int:
    """Create a deterministic hash for seeding RNG."""
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)
    seed = int(sha256.hexdigest(), 16)
    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


def load_dataset_config(
    config_name: str, config_dir: Optional[Path] = None
) -> dict[str, Any]:
    """Load a dataset configuration from YAML file.

    Args:
        config_name: Name of the dataset config (e.g., 'franka_peg_fmb')
        config_dir: Directory containing dataset configs. Defaults to configs/dataset/

    Returns:
        Dictionary with dataset configuration
    """
    if config_dir is None:
        config_dir = get_dataset_config_dir()

    config_path = config_dir / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class ValueMixtureDataset(UnifiedDatasetInterface):
    """
    A mixture of multiple ValueDatasets with weighted sampling.

    This class combines multiple value datasets with configurable weights and sampling
    strategies for training value models on heterogeneous data.
    """

    def __init__(
        self,
        datasets: Sequence[tuple[SizedDataset, float]],
        mode: str = "train",
        balance_dataset_weights: bool = True,
        seed: int = 42,
    ):
        # Filter out empty datasets
        valid_datasets = []
        for ds, weight in datasets:
            if len(ds) == 0:
                logger.warning("Skipping empty dataset")
                continue
            valid_datasets.append((ds, weight))

        if not valid_datasets:
            raise ValueError("No valid (non-empty) datasets provided")

        self.datasets = [ds for ds, _ in valid_datasets]
        self._raw_weights = np.array([w for _, w in valid_datasets], dtype=np.float32)
        self.mode = mode
        self.balance_dataset_weights = balance_dataset_weights
        self.seed = seed

        self._dataset_lengths = np.array([len(ds) for ds in self.datasets])

        self._dataset_sampling_weights = self._raw_weights.copy()
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths

        weight_sum = self._dataset_sampling_weights.sum()
        if weight_sum <= 0 or np.isnan(weight_sum):
            logger.warning(f"Invalid weight sum {weight_sum}, using uniform weights")
            self._dataset_sampling_weights = np.ones(len(self.datasets)) / len(
                self.datasets
            )
        else:
            self._dataset_sampling_weights /= weight_sum

        self._primary_indices = self._raw_weights == 1.0
        if not np.any(self._primary_indices):
            max_weight = self._raw_weights.max()
            self._primary_indices = self._raw_weights == max_weight

        self._epoch = 0

        logger.info("ValueMixtureDataset initialized:")
        logger.info(f"  Datasets: {len(self.datasets)}")
        logger.info(f"  Total samples: {sum(self._dataset_lengths)}")
        logger.info(f"  Dataset lengths: {self._dataset_lengths.tolist()}")
        logger.info(f"  Raw weights: {self._raw_weights.tolist()}")
        logger.info(f"  Sampling weights: {self._dataset_sampling_weights.tolist()}")
        logger.info(f"  Mode: {mode}")

    @property
    def dataset_lengths(self) -> np.ndarray:
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        return self._dataset_sampling_weights

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        primary_lengths = self._dataset_lengths[self._primary_indices]
        primary_weights = self._dataset_sampling_weights[self._primary_indices]

        valid_mask = primary_weights > 0
        if not np.any(valid_mask):
            return int(self._dataset_lengths.sum())

        ratios = primary_lengths[valid_mask] / primary_weights[valid_mask]
        return int(ratios.max())

    def _sample_step(self, index: int) -> tuple[SizedDataset, int]:
        # breakpoint()
        if self.mode != "train":
            seed = index
        else:
            seed = _safe_hash((self._epoch, index, self.seed))

        rng = np.random.default_rng(seed)
        ds_idx = rng.choice(len(self.datasets), p=self._dataset_sampling_weights)
        dataset = self.datasets[ds_idx]
        sample_idx = int(rng.integers(0, len(dataset)))

        return dataset, sample_idx

    def __getitem__(self, index: int) -> dict[str, Any]:
        dataset, sample_idx = self._sample_step(index)
        return dataset[sample_idx]

    def get_train_val_split(
        self, validation_split: Optional[float] = None
    ) -> tuple["ValueMixtureDataset", "ValueMixtureDataset"]:
        if validation_split is None:
            validation_split = 0.1

        train_datasets = []
        val_datasets = []

        for ds, weight in zip(self.datasets, self._raw_weights):
            if hasattr(ds, "get_train_val_split"):
                train_ds, val_ds = ds.get_train_val_split(validation_split)
            else:
                total = len(ds)
                val_size = int(total * validation_split)
                indices = np.arange(total)
                np.random.shuffle(indices)
                from torch.utils.data import Subset

                val_ds = Subset(ds, indices[:val_size].tolist())
                train_ds = Subset(ds, indices[val_size:].tolist())

            train_datasets.append((train_ds, weight))
            val_datasets.append((val_ds, weight))

        train_mixture = ValueMixtureDataset(
            datasets=train_datasets,
            mode="train",
            balance_dataset_weights=self.balance_dataset_weights,
            seed=self.seed,
        )
        val_mixture = ValueMixtureDataset(
            datasets=val_datasets,
            mode="val",
            balance_dataset_weights=self.balance_dataset_weights,
            seed=self.seed,
        )

        return train_mixture, val_mixture

    def get_custom_tokens(self) -> Optional[list[str]]:
        tokens = set()
        for ds in self.datasets:
            if hasattr(ds, "get_custom_tokens"):
                ds_tokens = ds.get_custom_tokens()
                if ds_tokens:
                    tokens.update(ds_tokens)
        return list(tokens) if tokens else None

    def get_source_name(self) -> str:
        names = []
        for ds in self.datasets:
            if hasattr(ds, "get_source_name"):
                names.append(ds.get_source_name())
            elif hasattr(ds, "repo_id"):
                names.append(ds.repo_id)
        return "_".join(names[:3]) if names else "value_mixture"


def create_single_value_dataset_from_config(
    dataset_config: dict[str, Any],
    data_root: Union[str, Path],
    norm_stats_dir: Optional[str] = None,
    model_type: Optional[str] = None,
    action_horizon: int = 10,
    action_dim: Optional[int] = None,
    max_samples: Optional[int] = None,
    # Value training settings
    value_prefix: str = "Value: ",
    include_state: bool = True,
    skip_vlm_response: bool = False,  # Skip VLM response for expert-only mode
    history_length: int = 0,
    gamma: float = 0.99,
    include_next_obs: bool = False,  # Set True for distributional RL
    num_return_bins: int = 201,
    return_min: Optional[float] = None,
    return_max: Optional[float] = None,
    normalize_to_minus_one_zero: bool = True,
    # Episode filtering
    episode_percentage: Optional[float] = None,
    shuffle_episodes: bool = False,
    episode_seed: int = 42,
) -> ValueDataset:
    """Create a single value dataset from its config dictionary.

    Args:
        dataset_config: Config dict from dataset YAML
        data_root: Root directory containing datasets
        norm_stats_dir: Base norm_stats directory
        model_type: Override model_type from config
        action_horizon: Action horizon
        action_dim: Action dimension
        max_samples: Limit samples (for testing)
        value_prefix: Prefix text for value prediction
        include_state: Whether to include state
        history_length: Number of past observations
        gamma: Discount factor
        num_return_bins: Number of bins for discretization
        return_min: Minimum return value
        return_max: Maximum return value
        normalize_to_minus_one_zero: Normalize returns to (-1, 0) range
        episode_percentage: Percentage of episodes to use
        shuffle_episodes: Random episode selection
        episode_seed: Seed for reproducibility

    Returns:
        ValueDataset instance
    """
    dataset_name = dataset_config.get("dataset_name") or dataset_config.get("name")
    if not dataset_name:
        raise ValueError("Dataset config must have 'dataset_name' or 'name'")

    dataset_path = Path(data_root) / dataset_name

    robot_type = dataset_config.get("robot_type")
    effective_model_type = model_type or dataset_config.get("model_type")
    default_prompt = dataset_config.get("default_prompt")
    asset_id = dataset_config.get("asset_id") or dataset_name
    config_norm_stats_dir = dataset_config.get("norm_stats_dir") or norm_stats_dir
    action_norm_skip_dims = dataset_config.get("action_norm_skip_dims")

    # Get return range - dataset-specific overrides take priority
    ds_return_min = dataset_config.get("return_min", return_min)
    ds_return_max = dataset_config.get("return_max", return_max)

    # Try loading from norm_stats if not specified
    if (ds_return_min is None or ds_return_max is None) and config_norm_stats_dir:
        loaded_min, loaded_max = load_return_range_from_norm_stats(
            config_norm_stats_dir, asset_id
        )
        if loaded_min is not None and loaded_max is not None:
            ds_return_min = ds_return_min if ds_return_min is not None else loaded_min
            ds_return_max = ds_return_max if ds_return_max is not None else loaded_max
            logger.info(
                f"Loaded return range from norm_stats: [{ds_return_min}, {ds_return_max}]"
            )

    if ds_return_min is None or ds_return_max is None:
        raise ValueError(
            f"return_min and return_max must be specified for dataset {dataset_name}. "
            "Either set them in the config or ensure norm_stats_dir contains 'return' key."
        )

    logger.info(f"Creating value dataset: {dataset_name}")
    logger.info(f"  robot_type={robot_type}, model_type={effective_model_type}")
    logger.info(f"  return_range=[{ds_return_min}, {ds_return_max}]")
    if action_norm_skip_dims:
        logger.info(f"  action_norm_skip_dims={action_norm_skip_dims}")
    if episode_percentage is not None:
        logger.info(
            f"  episode_percentage={episode_percentage}%, shuffle={shuffle_episodes}"
        )

    return create_value_dataset(
        dataset_path=str(dataset_path),
        value_prefix=value_prefix,
        include_state=include_state,
        skip_vlm_response=skip_vlm_response,
        history_length=history_length,
        action_horizon=action_horizon,
        gamma=gamma,
        include_next_obs=include_next_obs,
        num_return_bins=num_return_bins,
        return_min=ds_return_min,
        return_max=ds_return_max,
        normalize_to_minus_one_zero=normalize_to_minus_one_zero,
        split="train",
        robot_type=robot_type,
        model_type=effective_model_type,
        default_prompt=default_prompt,
        norm_stats_dir=config_norm_stats_dir,
        asset_id=asset_id,
        action_dim=action_dim,
        max_samples=max_samples,
        action_norm_skip_dims=action_norm_skip_dims,
        episode_percentage=episode_percentage,
        shuffle_episodes=shuffle_episodes,
        episode_seed=episode_seed,
    )


def create_value_mixture_dataset(
    data_root: Union[str, Path],
    datasets_config: list[dict[str, Any]],
    mode: str = "train",
    action_horizon: int = 10,
    action_dim: Optional[int] = None,
    balance_dataset_weights: bool = True,
    seed: int = 42,
    model_type: Optional[str] = None,
    norm_stats_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    config_dir: Optional[Path] = None,
    # Value training settings
    value_prefix: str = "Value: ",
    include_state: bool = True,
    skip_vlm_response: bool = False,  # Skip VLM response for expert-only mode
    history_length: int = 0,
    gamma: float = 0.99,
    include_next_obs: bool = False,  # Set True for distributional RL
    num_return_bins: int = 201,
    return_min: Optional[float] = None,
    return_max: Optional[float] = None,
    normalize_to_minus_one_zero: bool = True,
    # Episode filtering
    episode_percentage: Optional[float] = None,
    shuffle_episodes: bool = False,
    episode_seed: int = 42,
) -> ValueMixtureDataset:
    """
    Create a value mixture dataset from a list of dataset configs.

    Args:
        data_root: Root directory containing all datasets
        datasets_config: List of dataset entries with 'name' and 'weight'
        mode: "train", "val", or "test"
        action_horizon: Number of future actions
        action_dim: Action dimensionality
        balance_dataset_weights: Whether to balance by dataset length
        seed: Random seed
        model_type: Override model_type for all datasets
        norm_stats_dir: Base norm_stats directory
        max_samples: Limit samples per dataset
        config_dir: Directory containing dataset configs
        value_prefix: Prefix text for value prediction
        include_state: Whether to include state
        history_length: Number of past observations
        gamma: Discount factor
        include_next_obs: Whether to include next obs (for distributional RL)
        num_return_bins: Number of bins for discretization
        return_min: Default minimum return value
        return_max: Default maximum return value
        normalize_to_minus_one_zero: Normalize returns to (-1, 0) range
        episode_percentage: Percentage of episodes to use
        shuffle_episodes: Random episode selection
        episode_seed: Seed for reproducibility

    Returns:
        ValueMixtureDataset instance
    """
    data_root = Path(data_root)

    datasets_with_weights = []
    seen = set()

    for entry in datasets_config:
        config_name = entry.get("name") or entry.get("config")
        weight = entry.get("weight", 1.0)

        if not config_name:
            raise ValueError(f"Dataset entry must have 'name' or 'config': {entry}")

        if config_name in seen:
            logger.warning(f"Skipping duplicate: {config_name}")
            continue
        seen.add(config_name)

        # Load full config from YAML file
        dataset_config = load_dataset_config(config_name, config_dir)

        # Apply overrides from the mixture entry
        if entry.get("robot_type"):
            dataset_config["robot_type"] = entry["robot_type"]
        if entry.get("model_type"):
            dataset_config["model_type"] = entry["model_type"]
        if entry.get("asset_id"):
            dataset_config["asset_id"] = entry["asset_id"]
        if entry.get("return_min") is not None:
            dataset_config["return_min"] = entry["return_min"]
        if entry.get("return_max") is not None:
            dataset_config["return_max"] = entry["return_max"]

        try:
            ds = create_single_value_dataset_from_config(
                dataset_config=dataset_config,
                data_root=data_root,
                norm_stats_dir=norm_stats_dir,
                model_type=model_type,
                action_horizon=action_horizon,
                action_dim=action_dim,
                max_samples=max_samples,
                value_prefix=value_prefix,
                include_state=include_state,
                skip_vlm_response=skip_vlm_response,
                history_length=history_length,
                gamma=gamma,
                include_next_obs=include_next_obs,
                num_return_bins=num_return_bins,
                return_min=return_min,
                return_max=return_max,
                normalize_to_minus_one_zero=normalize_to_minus_one_zero,
                episode_percentage=episode_percentage,
                shuffle_episodes=shuffle_episodes,
                episode_seed=episode_seed,
            )
            datasets_with_weights.append((ds, weight))
            logger.info(
                f"Loaded value dataset: {config_name} ({len(ds)} samples, weight={weight})"
            )
        except Exception as e:
            logger.error(f"Failed to load {config_name}: {e}")
            raise

    if not datasets_with_weights:
        raise RuntimeError("No datasets could be loaded")

    return ValueMixtureDataset(
        datasets=datasets_with_weights,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        seed=seed,
    )


def create_value_mixture_dataset_from_config(
    config: dict[str, Any],
    data_root: Optional[Union[str, Path]] = None,
    config_dir: Optional[Path] = None,
) -> ValueMixtureDataset:
    """
    Create a value mixture dataset from a Hydra configuration dictionary.

    Args:
        config: Full Hydra config dictionary with:
            - data_root: Root directory for datasets
            - norm_stats_dir: Norm stats directory
            - value_mixture.datasets: List of {name, weight} entries
            - value_mixture settings (value_prefix, num_return_bins, etc.)
        data_root: Override for data_root
        config_dir: Directory containing dataset config files

    Returns:
        ValueMixtureDataset instance

    Example config:
        ```yaml
        data_root: "data/"
        norm_stats_dir: "norm_stats/pi05_franka_peg_value"

        value_mixture:
          datasets:
            - name: franka_peg_fmb
              weight: 1.0
            - name: franka_peg_fmb_iter1
              weight: 1.0
          model_type: pi05
          balance_weights: true
          seed: 42

        value:
          value_prefix: "Value: "
          num_return_bins: 201
          normalize_to_minus_one_zero: true
        ```
    """
    root = data_root or config.get("data_root") or config.get("data_path")
    if root is None:
        raise ValueError("data_root must be provided in config or as argument")

    value_mixture = config.get("value_mixture", {})
    value_config = config.get("value", {})
    rl_config = config.get("rl_config", {})
    model_config = config.get("model", {})

    datasets_list = value_mixture.get("datasets", [])
    if not datasets_list:
        raise ValueError("No datasets found in value_mixture.datasets")

    # Distributional RL requires next observations for TD target computation
    expert_loss_type = model_config.get("expert_loss_type", "mse")
    include_next_obs = expert_loss_type == "distributional"

    # Skip VLM response for expert-only mode (no CE loss on text tokens)
    critic_forward_mode = model_config.get("critic_forward_mode", "vlm")
    skip_vlm_response = critic_forward_mode == "expert"

    logger.info(
        f"Value dataset config: forward_mode={critic_forward_mode}, "
        f"loss_type={expert_loss_type}, include_next_obs={include_next_obs}, "
        f"skip_vlm_response={skip_vlm_response}"
    )

    return create_value_mixture_dataset(
        data_root=root,
        datasets_config=datasets_list,
        mode=value_mixture.get("mode", "train"),
        action_horizon=rl_config.get(
            "action_horizon", model_config.get("action_horizon", 10)
        ),
        action_dim=model_config.get("action_dim"),
        balance_dataset_weights=value_mixture.get("balance_weights", True),
        seed=value_mixture.get("seed", 42),
        model_type=value_mixture.get("model_type"),
        norm_stats_dir=config.get("norm_stats_dir"),
        max_samples=value_mixture.get("max_samples"),
        config_dir=config_dir,
        # Value training settings
        value_prefix=value_config.get("value_prefix", "Value: "),
        include_state=value_config.get("include_state", True),
        skip_vlm_response=skip_vlm_response,
        history_length=rl_config.get("history_length", 0),
        gamma=rl_config.get("gamma", 0.99),
        include_next_obs=include_next_obs,
        num_return_bins=value_config.get("num_return_bins", 201),
        return_min=value_config.get("return_min"),
        return_max=value_config.get("return_max"),
        normalize_to_minus_one_zero=value_config.get(
            "normalize_to_minus_one_zero", True
        ),
        # Episode filtering
        episode_percentage=value_mixture.get("episode_percentage"),
        shuffle_episodes=value_mixture.get("shuffle_episodes", False),
        episode_seed=value_mixture.get("episode_seed", 42),
    )


@register_dataset("rl/value_mixture", [r"value_mixture/"])
def create_value_mixture_dataset_factory(
    config: Optional[dict[str, Any]] = None,
    config_dir: Optional[Path] = None,
    **kwargs,
) -> ValueMixtureDataset:
    """
    Factory function for value mixture datasets.

    Creates a mixture dataset from config.value_mixture.datasets.

    Args:
        config: Hydra config dict with value_mixture.datasets
        config_dir: Directory containing dataset config YAML files
        **kwargs: Additional overrides

    Returns:
        ValueMixtureDataset instance
    """
    if config is None:
        raise ValueError("config must be provided")

    if "value_mixture" not in config:
        raise ValueError("Config must have 'value_mixture' section")

    return create_value_mixture_dataset_from_config(config, config_dir=config_dir)
