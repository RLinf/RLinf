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

# Override of Ray's NPUAcceleratorManager
# https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ray._private.accelerators.npu import NPUAcceleratorManager

from .accelerator import AcceleratorManager, AcceleratorType, ProfileConfig

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ...collective import CollectiveGroupOptions


@AcceleratorManager.register_profiling_config(AcceleratorType.NPU)
@dataclass
class AscendNPUProfileConfig(ProfileConfig):
    """Ascend NPU profiling configuration."""


def boolean_attention_mask(attn_mask, _cache={}):
    """The boolean form of a {0, -inf} float mask, or None if it carries biases.

    Verifying the mask forces a device sync, and HF models pass one mask tensor
    through every layer, so the verdict is cached per live tensor identity.
    """
    import weakref

    import torch

    if attn_mask is None or not attn_mask.dtype.is_floating_point:
        return None
    entry = _cache.get(id(attn_mask))
    if entry is not None:
        ref, version, result = entry
        if ref() is attn_mask and version == attn_mask._version:
            return result
    floor = torch.finfo(attn_mask.dtype).min * 0.5
    keep = attn_mask >= 0
    result = keep if bool((keep | (attn_mask <= floor)).all()) else None
    if len(_cache) >= 8:
        _cache.clear()
    _cache[id(attn_mask)] = (weakref.ref(attn_mask), attn_mask._version, result)
    return result


def install_npu_sdpa_mask_cast() -> bool:
    """Cast {0, -inf} float SDPA masks to boolean so the NPU keeps its fused kernel.

    torch-npu decomposes SDPA into matmul-softmax for floating masks (4-13x
    slower); the additive causal masks HF models build are boolean in content.
    Masks carrying real biases are left untouched. Idempotent; returns whether
    the wrapper is installed.
    """
    import torch

    functional = torch.nn.functional
    if getattr(functional.scaled_dot_product_attention, "_rlinf_npu_mask_cast", False):
        return True
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False

    original = functional.scaled_dot_product_attention

    def scaled_dot_product_attention(
        query, key, value, attn_mask=None, *args, **kwargs
    ):
        if attn_mask is not None and attn_mask.dtype.is_floating_point:
            floor = torch.finfo(attn_mask.dtype).min * 0.5
            keep = attn_mask >= 0
            if bool((keep | (attn_mask <= floor)).all()):
                attn_mask = keep
        return original(query, key, value, attn_mask, *args, **kwargs)

    scaled_dot_product_attention._rlinf_npu_mask_cast = True
    functional.scaled_dot_product_attention = scaled_dot_product_attention
    logger.info("NPU SDPA float-mask cast installed (FlashAttentionScore path).")
    return True


def install_npu_fused_rotary() -> bool:
    """Serve HF Llama's rotary embedding from ``npu_rotary_mul`` (3x faster).

    Module-level, so every Llama-family model built afterwards picks it up.
    Idempotent; returns whether the patch is installed.
    """
    import torch

    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        return False
    try:
        import torch_npu
        from transformers.models.llama import modeling_llama
    except ImportError:
        return False
    if getattr(modeling_llama.apply_rotary_pos_emb, "_rlinf_npu_fused", False):
        return True

    def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
        return (
            torch_npu.npu_rotary_mul(q, cos, sin),
            torch_npu.npu_rotary_mul(k, cos, sin),
        )

    apply_rotary_pos_emb._rlinf_npu_fused = True
    modeling_llama.apply_rotary_pos_emb = apply_rotary_pos_emb
    logger.info("NPU fused rotary embedding installed for Llama models.")
    return True


@AcceleratorManager.register_manager(AcceleratorType.NPU)
class AscendNPUManager(AcceleratorManager):
    """Utility Class for Ascend NPU."""

    @staticmethod
    def get_num_devices():
        """Get the number of Ascend NPU devices on the node."""
        return NPUAcceleratorManager.get_current_node_num_accelerators()

    @staticmethod
    def get_accelerator_type():
        """Get the type of the accelerator."""
        return AcceleratorType.NPU

    @staticmethod
    def get_accelerator_model():
        """Get the model of the Ascend NPU."""
        return NPUAcceleratorManager.get_current_node_accelerator_type()

    @staticmethod
    def get_accelerator_env_var(visible_accelerators: list[str]) -> dict[str, str]:
        """Get the environment variables related to the accelerator.

        Args:
            visible_accelerators (List[str]): A list of visible accelerator IDs.

        Returns:
            Dict[str, str]: A dictionary containing the accelerator environment variables.
        """
        env_vars = {}
        visible_accelerators_str = ",".join(visible_accelerators)

        env_vars["ASCEND_RT_VISIBLE_DEVICES"] = visible_accelerators_str
        env_vars["RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES"] = "1"
        # https://github.com/ray-project/ray/blob/161849364a784442cc659fb9780f1a6adee85fce/python/ray/_private/accelerators/npu.py#L91

        return env_vars

    @staticmethod
    def get_visible_devices():
        """Get the visible device IDs."""
        visible_devices = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", None)

        if visible_devices is None or visible_devices == "":
            return []
        else:
            try:
                visible_devices = [int(v.strip()) for v in visible_devices.split(",")]
            except ValueError:
                raise ValueError(
                    f"Invalid visible device IDs: {visible_devices}. "
                    "Please ensure they are integers separated by commas."
                )
            return visible_devices

    @staticmethod
    def get_ccl_backend():
        """Get the CCL backend."""
        return "hccl"

    @staticmethod
    def get_ccl_socket_ifname_env_var() -> str:
        """Get the network socket interface name environment variable.

        Returns:
            str: The network socket interface name environment variable.
        """
        return "HCCL_SOCKET_IFNAME"

    @staticmethod
    def get_torch_platform():
        """Get the PyTorch platform module."""
        import torch

        # torch.cuda.ipc_collect() exists for inter-process CUDA tensor cleanup;
        # torch_npu doesn't expose it (NPU has no equivalent IPC mechanism).
        # Scheduler call sites (collective_group, utils.utils) invoke
        # `Worker.torch_platform.ipc_collect()` unconditionally, so attach a
        # no-op when missing instead of guarding every call site.
        if not hasattr(torch.npu, "ipc_collect"):
            torch.npu.ipc_collect = lambda: None
        return torch.npu

    @staticmethod
    def get_device_type() -> str:
        """Get the device type."""
        return "npu"

    @staticmethod
    def setup_worker_torch():
        """Install the fused-kernel SDPA and rotary-embedding routes."""
        install_npu_sdpa_mask_cast()
        install_npu_fused_rotary()

    @staticmethod
    def get_accel_pg_options(options: Optional["CollectiveGroupOptions"]):
        """Get the accelerator CCL process group options."""
        return None
