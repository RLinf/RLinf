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

import torch


def get_radio_compatible_cuda_capability_on_musa(*_args, **_kwargs) -> tuple[int, int]:
    """RADIO's minimum accepted CUDA capability (Ampere 8.0), as a sentinel.

    Isaac-GR00T N1.5's radio_model calls ``torch.cuda.get_device_capability()``
    when swapping flash-attn into the ViT. MUSA exposes no CUDA device, so the
    call raises ``AssertionError: Invalid device id``. The value only has to
    clear RADIO's check — it does not describe the device.
    """
    return (8, 0)


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

    Only the ``get_device_capability`` sentinel is needed. Unlike Ascend, MUSA
    has a working flash-attn (the vendor image ships one, and RADIO's v2 import
    path resolves to ``flash_attn_varlen_qkvpacked_func``), so the flash-attn
    import stub that Ascend installs would only disable a usable kernel here.
    """
    if not _is_musa():
        return None

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
