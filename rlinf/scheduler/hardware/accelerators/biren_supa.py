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

"""Biren SUPA accelerator manager.

SUPA exposes the Biren runtime through the ``torch_supa`` package.
"""

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


def _ensure_torch_supa() -> bool:
    """Import the SUPA sidecar and report whether a device is available."""
    try:
        import torch

        if not hasattr(torch, "supa"):
            import torch_supa  # noqa: F401

        return hasattr(torch, "supa") and torch.supa.is_available()
    except Exception:
        return False


@AcceleratorManager.register_profiling_config(AcceleratorType.BIREN_GPU)
@dataclass
class BirenSUPAProfileConfig(ProfileConfig):
    """Biren SUPA profiling configuration."""


@AcceleratorManager.register_manager(AcceleratorType.BIREN_GPU)
class BirenSUPAManager(AcceleratorManager):
    """Utility class for Biren SUPA devices."""

    @staticmethod
    def get_num_devices():
        """Return the number of available SUPA devices."""
        if not _ensure_torch_supa():
            return 0
        import torch

        return torch.supa.device_count()

    @staticmethod
    def get_accelerator_type():
        """Return the SUPA accelerator type."""
        return AcceleratorType.BIREN_GPU

    @staticmethod
    def get_accelerator_model():
        """Return a model name suitable for hardware-resource serialization."""
        if not _ensure_torch_supa():
            return "UNKNOWN"
        try:
            import torch

            return torch.supa.get_device_name(0)
        except Exception:
            return "UNKNOWN"

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Return environment variables controlling SUPA visibility."""
        return {
            "SUPA_VISIBLE_DEVICES": ",".join(visible_accelerators),
            # Ray must not replace SUPA's device selection.
            "RAY_EXPERIMENTAL_NOSET_SUPA_VISIBLE_DEVICES": "1",
        }

    @staticmethod
    def get_visible_devices() -> list[int]:
        """Read visible device IDs from ``SUPA_VISIBLE_DEVICES``."""
        value = os.environ.get("SUPA_VISIBLE_DEVICES", "")
        if not value:
            return []
        try:
            return [int(device.strip()) for device in value.split(",")]
        except ValueError as exc:
            raise ValueError(
                f"Invalid visible device IDs: {value}. "
                "Please ensure they are integers separated by commas."
            ) from exc

    @staticmethod
    def get_ccl_backend():
        """Return the Biren collective communication backend."""
        return "bccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Return the BCCL socket-interface environment variable."""
        return "BCCL_SOCKET_IFNAME"

    @staticmethod
    def get_torch_platform():
        """Return the SUPA PyTorch platform module."""
        _ensure_torch_supa()
        import torch

        return torch.supa

    @staticmethod
    def get_device_type() -> str:
        """Return the SUPA device type.

        On this SUPA runtime ``torch_supa`` aliases the ``cuda`` string to
        ``supa``, so ``torch.device("cuda:0").type`` and
        ``tensor.device.type`` are both ``"supa"``. RLinf compares
        ``tensor.device.type`` against this value in the weight-syncer and
        ``utils`` paths, so returning ``"cuda"`` would make those comparisons
        fail. Report ``"supa"`` to keep those equality guards correct.
        """
        return "supa"

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Return no vendor-specific process-group options."""
        return None
