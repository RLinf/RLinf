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

"""Load RLinf full-state checkpoints for Cosmos3."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from rlinf.utils.logging import get_logger

if TYPE_CHECKING:
    from rlinf.models.embodiment.cosmos3.cosmos3_policy import Cosmos3Policy

logger = get_logger()


def load_inference_checkpoint(model: Cosmos3Policy, checkpoint_path: str) -> None:
    """Load RLinf weights from a memory-mapped CPU checkpoint."""
    state_dict = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    model.load_state_dict(state_dict, strict=True)
    logger.info("Cosmos3: loaded inference checkpoint %s", checkpoint_path)
