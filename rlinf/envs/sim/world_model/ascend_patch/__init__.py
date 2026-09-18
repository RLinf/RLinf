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

"""Install Ascend operators before constructing a Wan pipeline."""

import torch


def install_ascend_patch(device: torch.device) -> None:
    """Patch Wan DiT on NPU when MindIE-SD is available; safe to call repeatedly."""
    if device.type != "npu":
        return

    from rlinf.utils.logging import get_logger

    from . import wan_video_dit

    logger = get_logger()
    if not wan_video_dit.MINDIESD_ENABLE:
        logger.warning(
            "Wan Ascend acceleration requires torch_npu and MindIE-SD; "
            "keeping the original diffsynth operators."
        )
        return

    import diffsynth.models.wan_video_dit as wan_dit

    operators = ("flash_attention", "rope_apply", "RMSNorm")
    missing = [name for name in operators if not hasattr(wan_dit, name)]
    if missing:
        logger.warning(
            "Wan Ascend patch not applied: diffsynth is missing %s.", missing
        )
        return
    if all(
        getattr(wan_dit, name) is getattr(wan_video_dit, name) for name in operators
    ):
        return

    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    for name in operators:
        Patcher.add_patch(
            f"diffsynth.models.wan_video_dit.{name}",
            f"{wan_video_dit.__name__}.{name}",
        )
    Patcher.apply()
    logger.info("Installed Wan Ascend attention, RoPE, and RMSNorm operators.")
