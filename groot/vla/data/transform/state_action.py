import functools
import random
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, PrivateAttr, field_validator, model_validator
import torch

from groot.vla.data.schema import DatasetMetadata, RotationType, StateActionMetadata
from groot.vla.data.transform.base import InvertibleModalityTransform, ModalityTransform


class RotationTransform:
    """Adapted from https://github.com/real-stanford/diffusion_policy/blob/548a52bbb105518058e27bf34dcf90bf6f73681a/diffusion_policy/model/common/rotation_transformer.py"""

    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(self, from_rep="axis_angle", to_rep="rotation_6d"):
        """
        Valid representations

        Always use matrix as intermediate representation.
        """
        if from_rep.startswith("euler_angles"):
            from_convention = from_rep.split("_")[-1]
            from_rep = "euler_angles"
            from_convention = from_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            from_convention = None
        if to_rep.startswith("euler_angles"):
            to_convention = to_rep.split("_")[-1]
            to_rep = "euler_angles"
            to_convention = to_convention.replace("r", "X").replace("p", "Y").replace("y", "Z")
        else:
            to_convention = None
        assert from_rep != to_rep, f"from_rep and to_rep cannot be the same: {from_rep}"
        assert from_rep in self.valid_reps, f"Invalid from_rep: {from_rep}"
        assert to_rep in self.valid_reps, f"Invalid to_rep: {to_rep}"

        forward_funcs = list()
        inverse_funcs = list()

        if from_rep != "matrix":
            import pytorch3d.transforms as pt  # noqa: F401

            funcs = [getattr(pt, f"{from_rep}_to_matrix"), getattr(pt, f"matrix_to_{from_rep}")]
            if from_convention is not None:
                funcs = [functools.partial(func, convention=from_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [getattr(pt, f"matrix_to_{to_rep}"), getattr(pt, f"{to_rep}_to_matrix")]
            if to_convention is not None:
                funcs = [functools.partial(func, convention=to_convention) for func in funcs]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(x: torch.Tensor, funcs: list) -> torch.Tensor:
        assert isinstance(x, torch.Tensor)
        for func in funcs:
            x = func(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        return self._apply_funcs(x, self.inverse_funcs)


class Normalizer:
    valid_modes = ["q99", "mean_std", "min_max", "binary", "scale"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = statistics
        for key, value in self.statistics.items():
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Normalize the tensor
        if self.mode == "q99":

            # Range of q99 is [-1, 1]
            q01 = self.statistics["q01"].to(x.dtype)
            q99 = self.statistics["q99"].to(x.dtype)

            # In the case of q01 == q99, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = q01 != q99
            normalized = torch.zeros_like(x)

            # Normalize the values where q01 != q99
            # Formula: 2 * (x - q01) / (q99 - q01) - 1
            normalized[..., mask] = (x[..., mask] - q01[..., mask]) / (
                q99[..., mask] - q01[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where q01 == q99
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            # Range of mean_std is not fixed, but can be positive or negative
            mean = self.statistics["mean"].to(x.dtype)
            std = self.statistics["std"].to(x.dtype)

            # In the case of std == 0, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = std != 0
            normalized = torch.zeros_like(x)

            # Normalize the values where std != 0
            # Formula: (x - mean) / std
            normalized[..., mask] = (x[..., mask] - mean[..., mask]) / std[..., mask]

            # Set the normalized values to the original values where std == 0
            normalized[..., ~mask] = x[..., ~mask].to(x.dtype)

        elif self.mode == "min_max":
            # Range of min_max is [-1, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)

            # In the case of min == max, the normalization will be undefined
            # So we set the normalized values to the original values
            mask = min != max
            normalized = torch.zeros_like(x)

            # Normalize the values where min != max
            # Formula: 2 * (x - min) / (max - min) - 1
            normalized[..., mask] = (x[..., mask] - min[..., mask]) / (
                max[..., mask] - min[..., mask]
            )
            normalized[..., mask] = 2 * normalized[..., mask] - 1

            # Set the normalized values to the original values where min == max
            # normalized[..., ~mask] = x[..., ~mask].to(x.dtype)
            # Set the normalized values to 0 where min == max
            normalized[..., ~mask] = 0

            if not (torch.all(normalized <= 1 + 1e-5) and torch.all(normalized >= -1 - 1e-5)):
                print(
                    f"WARNING: normalization out of range: {x.shape=}, max: {torch.max(normalized)}, min: {torch.min(normalized)}"
                )

            # Clip the normalized values to be between -1 and 1
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "scale":
            # Range of scale is [0, 1]
            min = self.statistics["min"].to(x.dtype)
            max = self.statistics["max"].to(x.dtype)
            abs_max = torch.max(torch.abs(min), torch.abs(max))
            mask = abs_max != 0
            normalized = torch.zeros_like(x)
            normalized[..., mask] = x[..., mask] / abs_max[..., mask]
            normalized[..., ~mask] = 0

        elif self.mode == "binary":
            # Range of binary is [0, 1]
            normalized = (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(dtype=x.dtype, device=x.device)
            q99 = self.statistics["q99"].to(dtype=x.dtype, device=x.device)
            return (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(dtype=x.dtype, device=x.device)
            std = self.statistics["std"].to(dtype=x.dtype, device=x.device)
            return x * std + mean
        elif self.mode == "min_max":
            min = self.statistics["min"].to(dtype=x.dtype, device=x.device)
            max = self.statistics["max"].to(dtype=x.dtype, device=x.device)
            return (x + 1) / 2 * (max - min) + min
        elif self.mode == "binary":
            return (x > 0.5).to(dtype=x.dtype, device=x.device)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")


class PerHorizonNormalizer:
    """Normalizer that applies different statistics for each horizon index.
    
    Unlike the standard Normalizer which uses single stats for all timesteps,
    this normalizer uses per-horizon stats where each action horizon step has
    its own normalization parameters.
    
    Args:
        mode: Normalization mode ('q99', 'mean_std', 'min_max', 'binary', 'scale')
        statistics: Dict with stat names as keys and 2D tensors of shape (horizon_len, action_dim) as values.
            Format: {"q01": [[h0_vals], [h1_vals], ...], "q99": [[h0_vals], [h1_vals], ...], ...}
    """
    valid_modes = ["q99", "mean_std", "min_max", "binary", "scale"]

    def __init__(self, mode: str, statistics: dict):
        self.mode = mode
        self.statistics = {}
        for key, value in statistics.items():
            # Convert to tensor: shape (horizon_len, action_dim)
            self.statistics[key] = torch.tensor(value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize tensor with per-horizon statistics.
        
        Args:
            x: Input tensor of shape (horizon_len, action_dim) or (num_chunks * horizon_len, action_dim)
               or (..., horizon_len, action_dim)
            
        Returns:
            Normalized tensor of same shape
        """
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"

        # Get the stats horizon length
        stats_horizon_len = self.statistics["q01"].shape[0]
        action_dim = x.shape[-1]
        original_shape = x.shape
        
        # Handle case where x is flattened (num_chunks * horizon_len, action_dim)
        # We need to reshape to (..., horizon_len, action_dim) for proper broadcasting
        reshaped = False
        if len(x.shape) == 2:
            total_len = x.shape[0]
            if total_len > stats_horizon_len and total_len % stats_horizon_len == 0:
                # Flattened batch case: reshape to (num_chunks, horizon_len, action_dim)
                num_chunks = total_len // stats_horizon_len
                x = x.view(num_chunks, stats_horizon_len, action_dim)
                reshaped = True
            elif total_len == stats_horizon_len:
                # Single chunk, add batch dimension for consistent processing
                x = x.unsqueeze(0)
                reshaped = True
        
        # Now x should have shape (..., horizon_len, action_dim)
        horizon_len = x.shape[-2]
        
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(dtype=x.dtype, device=x.device)[:horizon_len]  # (horizon_len, action_dim)
            q99 = self.statistics["q99"].to(dtype=x.dtype, device=x.device)[:horizon_len]

            # Compute normalization with safe division
            range_vals = q99 - q01
            range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
            normalized = 2 * (x - q01) / range_vals - 1
            
            # For zero-range values, keep original
            mask = (q01 == q99).unsqueeze(0).expand_as(x) if len(x.shape) > 2 else (q01 == q99)
            normalized = torch.where(mask, x, normalized)
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            std = self.statistics["std"].to(dtype=x.dtype, device=x.device)[:horizon_len]

            std_safe = torch.where(std == 0, torch.ones_like(std), std)
            normalized = (x - mean) / std_safe
            
            mask = (std == 0).unsqueeze(0).expand_as(x) if len(x.shape) > 2 else (std == 0)
            normalized = torch.where(mask, x, normalized)

        elif self.mode == "min_max":
            min_val = self.statistics["min"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            max_val = self.statistics["max"].to(dtype=x.dtype, device=x.device)[:horizon_len]

            range_vals = max_val - min_val
            range_vals = torch.where(range_vals == 0, torch.ones_like(range_vals), range_vals)
            normalized = 2 * (x - min_val) / range_vals - 1
            
            mask = (min_val == max_val).unsqueeze(0).expand_as(x) if len(x.shape) > 2 else (min_val == max_val)
            normalized = torch.where(mask, torch.zeros_like(x), normalized)
            normalized = torch.clamp(normalized, -1, 1)

        elif self.mode == "scale":
            min_val = self.statistics["min"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            max_val = self.statistics["max"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
            
            abs_max_safe = torch.where(abs_max == 0, torch.ones_like(abs_max), abs_max)
            normalized = x / abs_max_safe
            
            mask = (abs_max == 0).unsqueeze(0).expand_as(x) if len(x.shape) > 2 else (abs_max == 0)
            normalized = torch.where(mask, torch.zeros_like(x), normalized)

        elif self.mode == "binary":
            normalized = (x > 0.5).to(x.dtype)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")

        # Reshape back to original shape
        if reshaped:
            normalized = normalized.view(original_shape)
            
        return normalized

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize tensor with per-horizon statistics.
        
        Args:
            x: Normalized tensor of shape (horizon_len, action_dim) or (num_chunks * horizon_len, action_dim)
               or (..., horizon_len, action_dim)
            
        Returns:
            Denormalized tensor of same shape
        """
        assert isinstance(
            x, torch.Tensor
        ), f"Unexpected input type: {type(x)}. Expected type: {torch.Tensor}"
        
        # Get the stats horizon length
        stats_horizon_len = self.statistics["q01"].shape[0]
        action_dim = x.shape[-1]
        original_shape = x.shape
        
        # Handle case where x is flattened (num_chunks * horizon_len, action_dim)
        reshaped = False
        if len(x.shape) == 2:
            total_len = x.shape[0]
            if total_len > stats_horizon_len and total_len % stats_horizon_len == 0:
                num_chunks = total_len // stats_horizon_len
                x = x.view(num_chunks, stats_horizon_len, action_dim)
                reshaped = True
            elif total_len == stats_horizon_len:
                x = x.unsqueeze(0)
                reshaped = True
        
        horizon_len = x.shape[-2]
        
        if self.mode == "q99":
            q01 = self.statistics["q01"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            q99 = self.statistics["q99"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            denormalized = (x + 1) / 2 * (q99 - q01) + q01
        elif self.mode == "mean_std":
            mean = self.statistics["mean"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            std = self.statistics["std"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            denormalized = x * std + mean
        elif self.mode == "min_max":
            min_val = self.statistics["min"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            max_val = self.statistics["max"].to(dtype=x.dtype, device=x.device)[:horizon_len]
            denormalized = (x + 1) / 2 * (max_val - min_val) + min_val
        elif self.mode == "binary":
            denormalized = (x > 0.5).to(dtype=x.dtype, device=x.device)
        else:
            raise ValueError(f"Invalid normalization mode: {self.mode}")
        
        # Reshape back to original shape
        if reshaped:
            denormalized = denormalized.view(original_shape)
            
        return denormalized


class PerHorizonActionTransform(InvertibleModalityTransform):
    """Transform for action normalization with per-horizon statistics.
    
    This transform applies different normalization parameters for each action
    horizon index, allowing for more precise normalization when the distribution
    of relative actions varies across the prediction horizon.
    
    Args:
        apply_to: List of action keys to transform (e.g., ['action.joint_position'])
        normalization_modes: Dict mapping action keys to normalization modes
        per_horizon_statistics: Dict mapping action keys to per-horizon stats.
            Format: {action_key: {stat_name: [[h0_vals], [h1_vals], ...]}}
    """

    apply_to: list[str] = Field(..., description="The action keys to transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each action key."
    )
    per_horizon_statistics: dict[str, dict[str, list]] = Field(
        default_factory=dict, description="Per-horizon statistics for each action key."
    )

    _normalizers: dict[str, PerHorizonNormalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes"}
        else:
            include = kwargs.pop("include", None)
        return super().model_dump(*args, include=include, **kwargs)

    def set_per_horizon_statistics(self, per_horizon_stats: dict[str, dict[str, list]]):
        """Set per-horizon statistics and initialize normalizers.
        
        Args:
            per_horizon_stats: Dict from dataset.lerobot_relative_horizon_stats_meta
                Format: {action_key: {stat_name: [[h0_vals], [h1_vals], ...]}}
        """
        self.per_horizon_statistics = per_horizon_stats
        self._initialize_normalizers()

    def _initialize_normalizers(self):
        """Initialize per-horizon normalizers from statistics."""
        for key in self.apply_to:
            if key not in self.normalization_modes:
                continue
            
            # Extract action subkey (e.g., 'action.joint_position' -> 'joint_position')
            subkey = key.replace("action.", "")
            
            if subkey not in self.per_horizon_statistics:
                print(f"Warning: No per-horizon statistics found for {subkey}, skipping")
                continue
            
            statistics = self.per_horizon_statistics[subkey]
            self._normalizers[key] = PerHorizonNormalizer(
                mode=self.normalization_modes[key],
                statistics=statistics
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply per-horizon normalization to action data.
        
        Args:
            data: Dict with action keys mapping to tensors of shape (horizon_len, action_dim)
            
        Returns:
            Dict with normalized action tensors
        """
        for key in self.apply_to:
            if key not in data:
                continue
            
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            
            action = data[key]
            if key in self._normalizers:
                action = self._normalizers[key].forward(action)
            data[key] = action
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply inverse per-horizon normalization (denormalization).
        
        Args:
            data: Dict with normalized action tensors
            
        Returns:
            Dict with denormalized action tensors
        """
        for key in self.apply_to:
            if key not in data:
                continue
            
            action = data[key]
            assert isinstance(
                action, torch.Tensor
            ), f"Unexpected action type: {type(action)}. Expected type: {torch.Tensor}"
            
            if key in self._normalizers:
                action = self._normalizers[key].inverse(action)
            
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    action = action.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    action = action.to(original_dtype)
            
            data[key] = action
        return data


class StateActionToTensor(InvertibleModalityTransform):
    """
    Transforms states and actions to tensors.
    """

    input_dtypes: dict[str, np.dtype] = Field(
        default_factory=dict, description="The input dtypes for each state key."
    )
    output_dtypes: dict[str, torch.dtype] = Field(
        default_factory=dict, description="The output dtypes for each state key."
    )

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("input_dtypes", "output_dtypes", mode="before")
    def validate_dtypes(cls, v):
        for key, dtype in v.items():
            if isinstance(dtype, str):
                if dtype.startswith("torch."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = getattr(torch, dtype_split)
                elif dtype.startswith("np.") or dtype.startswith("numpy."):
                    dtype_split = dtype.split(".")[-1]
                    v[key] = np.dtype(dtype_split)
                else:
                    raise ValueError(f"Invalid dtype: {dtype}")
        return v

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, np.ndarray
            ), f"Unexpected input type: {type(value)}. Expected type: {np.ndarray}"
            data[key] = torch.from_numpy(value)
            if key in self.output_dtypes:
                data[key] = data[key].to(self.output_dtypes[key])
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            value = data[key]
            assert isinstance(
                value, torch.Tensor
            ), f"Unexpected input type: {type(value)}. Expected type: {torch.Tensor}"
            data[key] = value.numpy()
            if key in self.input_dtypes:
                data[key] = data[key].astype(self.input_dtypes[key])
        return data


class StateActionTransform(InvertibleModalityTransform):
    """
    Class for state or action transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        normalization_modes (dict[str, str]): The normalization modes for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be normalized.
        target_rotations (dict[str, str]): The target representations for each state key.
            If a state key in apply_to is not present in the dictionary, it will not be rotated.
    """

    # Configurable attributes
    apply_to: list[str] = Field(..., description="The keys in the modality to load and transform.")
    normalization_modes: dict[str, str] = Field(
        default_factory=dict, description="The normalization modes for each state key."
    )
    target_rotations: dict[str, str] = Field(
        default_factory=dict, description="The target representations for each state key."
    )
    normalization_statistics: dict[str, dict] = Field(
        default_factory=dict, description="The statistics for each state key."
    )
    modality_metadata: dict[str, StateActionMetadata] = Field(
        default_factory=dict, description="The modality metadata for each state key."
    )

    # Model variables
    _rotation_transformers: dict[str, RotationTransform] = PrivateAttr(default_factory=dict)
    _normalizers: dict[str, Normalizer] = PrivateAttr(default_factory=dict)
    _input_dtypes: dict[str, np.dtype | torch.dtype] = PrivateAttr(default_factory=dict)

    # Model constants
    _DEFAULT_MIN_MAX_STATISTICS: ClassVar[dict] = {
        "rotation_6d": {
            "min": [-1, -1, -1, -1, -1, -1],
            "max": [1, 1, 1, 1, 1, 1],
        },
        "euler_angles": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
        "quaternion": {
            "min": [-1, -1, -1, -1],
            "max": [1, 1, 1, 1],
        },
        "axis_angle": {
            "min": [-np.pi, -np.pi, -np.pi],
            "max": [np.pi, np.pi, np.pi],
        },
    }

    def model_dump(self, *args, **kwargs):
        if kwargs.get("mode", "python") == "json":
            include = {"apply_to", "normalization_modes", "target_rotations"}
        else:
            include = kwargs.pop("include", None)

        return super().model_dump(*args, include=include, **kwargs)

    @field_validator("modality_metadata", mode="before")
    def validate_modality_metadata(cls, v):
        for modality_key, config in v.items():
            if isinstance(config, dict):
                config = StateActionMetadata.model_validate(config)
            else:
                assert isinstance(
                    config, StateActionMetadata
                ), f"Invalid source rotation config: {config}"
            v[modality_key] = config
        return v

    @model_validator(mode="after")
    def validate_normalization_statistics(self):
        for modality_key, normalization_statistics in self.normalization_statistics.items():
            if modality_key in self.normalization_modes:
                normalization_mode = self.normalization_modes[modality_key]
                if normalization_mode == "min_max":
                    assert (
                        "min" in normalization_statistics and "max" in normalization_statistics
                    ), f"Min and max statistics are required for min_max normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["min"]) == len(
                        normalization_statistics["max"]
                    ), f"Min and max statistics must have the same length, but got {normalization_statistics['min']} and {normalization_statistics['max']}"
                elif normalization_mode == "mean_std":
                    assert (
                        "mean" in normalization_statistics and "std" in normalization_statistics
                    ), f"Mean and std statistics are required for mean_std normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["mean"]) == len(
                        normalization_statistics["std"]
                    ), f"Mean and std statistics must have the same length, but got {normalization_statistics['mean']} and {normalization_statistics['std']}"
                elif normalization_mode == "q99":
                    assert (
                        "q01" in normalization_statistics and "q99" in normalization_statistics
                    ), f"q01 and q99 statistics are required for q99 normalization, but got {normalization_statistics}"
                    assert len(normalization_statistics["q01"]) == len(
                        normalization_statistics["q99"]
                    ), f"q01 and q99 statistics must have the same length, but got {normalization_statistics['q01']} and {normalization_statistics['q99']}"
                elif normalization_mode == "binary":
                    assert (
                        len(normalization_statistics) == 1
                    ), f"Binary normalization should only have one value, but got {normalization_statistics}"
                    assert normalization_statistics[0] in [
                        0,
                        1,
                    ], f"Binary normalization should only have 0 or 1, but got {normalization_statistics[0]}"
                else:
                    raise ValueError(f"Invalid normalization mode: {normalization_mode}")
        return self

    def set_metadata(self, dataset_metadata: DatasetMetadata):
        dataset_statistics = dataset_metadata.statistics
        modality_metadata = dataset_metadata.modalities

        # Check that all state keys specified in apply_to have their modality_metadata
        for key in self.apply_to:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            if key not in self.modality_metadata:
                modality, state_key = split_key
                assert hasattr(modality_metadata, modality), f"{modality} config not found"
                assert state_key in getattr(
                    modality_metadata, modality
                ), f"{state_key} config not found in {modality}"
                self.modality_metadata[key] = getattr(modality_metadata, modality)[state_key]

        # Check that all state keys specified in normalization_modes have their statistics in state_statistics
        for key in self.normalization_modes:
            split_key = key.split(".")
            assert len(split_key) == 2, "State keys should have two parts: 'modality.key'"
            modality, state_key = split_key
            assert hasattr(dataset_statistics, modality), f"{modality} statistics not found"
            assert state_key in getattr(
                dataset_statistics, modality
            ), f"{state_key} statistics not found"
            assert (
                len(getattr(modality_metadata, modality)[state_key].shape) == 1
            ), f"{getattr(modality_metadata, modality)[state_key].shape=}"
            self.normalization_statistics[key] = getattr(dataset_statistics, modality)[
                state_key
            ].model_dump()

        # Initialize the rotation transformers
        for key in self.target_rotations:
            # Get the original representation of the state
            from_rep = self.modality_metadata[key].rotation_type
            assert from_rep is not None, f"Source rotation type not found for {key}"

            # Get the target representation of the state, will raise an error if the target representation is not valid
            to_rep = RotationType(self.target_rotations[key])

            # If the original representation is not the same as the target representation, initialize the rotation transformer
            if from_rep != to_rep:
                self._rotation_transformers[key] = RotationTransform(
                    from_rep=from_rep.value, to_rep=to_rep.value
                )

        # Initialize the normalizers
        for key in self.normalization_modes:
            modality, state_key = key.split(".")
            # If the state has a nontrivial rotation, we need to handle it more carefully
            # For absolute rotations, we need to convert them to the target representation and normalize them using min_max mode,
            # since we can infer the bounds by the representation
            # For relative rotations, we cannot normalize them as we don't know the bounds
            if key in self._rotation_transformers:
                # Case 1: Absolute rotation
                if self.modality_metadata[key].absolute:
                    # Check that the normalization mode is valid
                    assert (
                        self.normalization_modes[key] == "min_max"
                    ), "Absolute rotations that are converted to other formats must be normalized using `min_max` mode"
                    rotation_type = RotationType(self.target_rotations[key]).value
                    # If the target representation is euler angles, we need to parse the convention
                    if rotation_type.startswith("euler_angles"):
                        rotation_type = "euler_angles"
                    # Get the statistics for the target representation
                    statistics = self._DEFAULT_MIN_MAX_STATISTICS[rotation_type]
                # Case 2: Relative rotation
                else:
                    raise ValueError(
                        f"Cannot normalize relative rotations: {key} that's converted to {self.target_rotations[key]}"
                    )
            # If the state is not continuous, we should not use normalization modes other than binary
            elif (
                not self.modality_metadata[key].continuous
                and self.normalization_modes[key] != "binary"
            ):
                raise ValueError(
                    f"{key} is not continuous, so it should be normalized using `binary` mode"
                )
            # Initialize the normalizer
            else:
                statistics = self.normalization_statistics[key]
            self._normalizers[key] = Normalizer(
                mode=self.normalization_modes[key], statistics=statistics
            )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                # We allow some keys to be missing in the data, and only process the keys that are present
                continue
            if key not in self._input_dtypes:
                input_dtype = data[key].dtype
                assert isinstance(
                    input_dtype, torch.dtype
                ), f"Unexpected input dtype: {input_dtype}. Expected type: {torch.dtype}"
                self._input_dtypes[key] = input_dtype
            else:
                assert (
                    data[key].dtype == self._input_dtypes[key]
                ), f"All states corresponding to the same key must be of the same dtype, input dtype: {data[key].dtype}, expected dtype: {self._input_dtypes[key]}"
            # Rotate the state
            state = data[key]
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].forward(state)
            # Normalize the state
            if key in self._normalizers:
                state = self._normalizers[key].forward(state)
            data[key] = state
        return data

    def unapply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            if key not in data:
                continue
            state = data[key]
            assert isinstance(
                state, torch.Tensor
            ), f"Unexpected state type: {type(state)}. Expected type: {torch.Tensor}"
            # Unnormalize the state
            if key in self._normalizers:
                state = self._normalizers[key].inverse(state)
            # Change the state back to its original representation
            if key in self._rotation_transformers:
                state = self._rotation_transformers[key].inverse(state)
            assert isinstance(
                state, torch.Tensor
            ), f"State should be tensor after unapplying transformations, but got {type(state)}"
            # Only convert back to the original dtype if it's known, i.e. `apply` was called before
            # If not, we don't know the original dtype, so we don't convert
            if key in self._input_dtypes:
                original_dtype = self._input_dtypes[key]
                if isinstance(original_dtype, np.dtype):
                    state = state.numpy().astype(original_dtype)
                elif isinstance(original_dtype, torch.dtype):
                    state = state.to(original_dtype)
                else:
                    raise ValueError(f"Invalid input dtype: {original_dtype}")
            data[key] = state
        return data


class StateActionPerturbation(ModalityTransform):
    """
    Class for state or action perturbation.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        std (float): Standard deviation of the noise to be added to the state or action.
    """

    # Configurable attributes
    std: float = Field(
        ..., description="Standard deviation of the noise to be added to the state or action."
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't perturb the data in eval mode
            return data
        if self.std < 0:
            # If the std is negative, we don't add any noise
            return data
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            transformed_data_min = torch.min(state)
            transformed_data_max = torch.max(state)
            noise = torch.randn_like(state) * self.std
            state += noise
            # Clip to the original range
            state = torch.clamp(state, transformed_data_min, transformed_data_max)
            data[key] = state
        return data


class StateActionDropout(ModalityTransform):
    """
    Class for state or action dropout.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
        dropout_prob (float): Probability of dropping out a state or action.
    """

    # Configurable attributes
    dropout_prob: float = Field(..., description="Probability of dropping out a state or action.")

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        if not self.training:
            # Don't drop out the data in eval mode
            return data
        if self.dropout_prob < 0:
            # If the dropout probability is negative, we don't drop out any states
            return data
        if self.dropout_prob > 1e-9 and random.random() < self.dropout_prob:
            for key in self.apply_to:
                state = data[key]
                assert isinstance(state, torch.Tensor)
                state = torch.zeros_like(state)
                data[key] = state
        return data


class StateActionSinCosTransform(ModalityTransform):
    """
    Class for state or action sin-cos transform.

    Args:
        apply_to (list[str]): The keys in the modality to load and transform.
    """

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in self.apply_to:
            state = data[key]
            assert isinstance(state, torch.Tensor)
            sin_state = torch.sin(state)
            cos_state = torch.cos(state)
            data[key] = torch.cat([sin_state, cos_state], dim=-1)
        return data
