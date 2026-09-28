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

import inspect

import torch


def get_radio_compatible_cuda_capability_on_musa(*_args, **_kwargs) -> tuple[int, int]:
    """RADIO's minimum accepted CUDA capability (Ampere 8.0), as a sentinel.

    Isaac-GR00T N1.5's radio_model calls ``torch.cuda.get_device_capability()``
    when swapping flash-attn into the ViT. MUSA exposes no CUDA device, so the
    call raises ``AssertionError: Invalid device id``. The value only has to
    clear RADIO's check — it does not describe the device.
    """
    return (8, 0)


def is_vendor_flash_attn_available_on_musa() -> bool:
    """Transformers' ``is_flash_attn_2_available`` for the MUSA vendor flash-attn."""
    return True


def bind_vendor_flash_attn_in_transformers() -> None:
    """Make Transformers use the MUSA vendor flash-attn.

    Transformers treats flash-attn as available only when
    ``torch.cuda.is_available()``, which is False on MUSA. It then imports
    ``modeling_flash_attention_utils`` without the flash-attn kernels, and the
    ``flash_attention_2`` layers that GR00T's Eagle backbone hard-codes fail
    with ``NameError: _flash_supports_window_size``. That module is already
    imported by the time GR00T loads, so the kernels are bound into it
    directly. The binding lasts for the process because forward passes run
    after model construction. The vendor package's top level exports only an
    inference interface; the standard kernels are in ``flash_attn_interface``.
    """
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_varlen_func
    from transformers import modeling_flash_attention_utils, modeling_utils

    fa_utils = modeling_flash_attention_utils
    fa_utils.is_flash_attn_2_available = is_vendor_flash_attn_available_on_musa
    modeling_utils.is_flash_attn_2_available = is_vendor_flash_attn_available_on_musa
    fa_utils.flash_attn_func = flash_attn_func
    fa_utils.flash_attn_varlen_func = flash_attn_varlen_func
    fa_utils.index_first_axis = index_first_axis
    fa_utils.pad_input = pad_input
    fa_utils.unpad_input = unpad_input
    fa_utils._flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters
    )


# Patcher references replacement objects by string path.
_MODULE = "rlinf.models.embodiment.gr00t.gr00t_n1d5.musa_patches"


def _is_musa() -> bool:
    """Whether this worker runs on a Moore Threads GPU, per the Worker device API."""
    from rlinf.scheduler import AcceleratorType, Worker

    return Worker.accelerator_type == AcceleratorType.MUSA_GPU


def apply_musa_patches(patcher) -> dict | None:
    """Register the MUSA patches for building GR00T N1.5; return restore state.

    No-op returning ``None`` off MUSA. Call before ``patcher.apply()``; pass the
    result to :func:`restore_musa_patches` after model construction.

    Unlike Ascend, MUSA has a working flash-attn (the vendor image ships one,
    and RADIO's v2 import path resolves to ``flash_attn_varlen_qkvpacked_func``),
    so the flash-attn import stub that Ascend installs would only disable a
    usable kernel here. Transformers is pointed at that kernel instead, and the
    ``get_device_capability`` sentinel is the only patch restored afterwards.
    """
    if not _is_musa():
        return None

    bind_vendor_flash_attn_in_transformers()

    # Capture before patcher.apply() replaces it, so restore can put it back.
    original_get_device_capability = torch.cuda.get_device_capability
    patcher.add_patch(
        "torch.cuda.get_device_capability",
        f"{_MODULE}.get_radio_compatible_cuda_capability_on_musa",
    )

    return {"get_device_capability": original_get_device_capability}


def restore_musa_patches(patcher, state: dict | None) -> None:
    """Undo the process-global patches from :func:`apply_musa_patches`.

    ``state`` is that call's return value; ``None`` (off MUSA) is a no-op.
    """
    if state is None:
        return

    torch.cuda.get_device_capability = state["get_device_capability"]
