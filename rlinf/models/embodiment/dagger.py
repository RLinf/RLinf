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

"""Policy-specific behavior used by the shared DAgger actor."""

from collections.abc import Callable
from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.utils.nested_dict_process import put_tensor_device


class DAggerPolicyAdapter:
    """Defaults for policies trained by the shared DAgger actor."""

    # Policies using the runner's global step have no optimizer-step callback.
    set_optimizer_step: Callable[..., None] | None = None

    @staticmethod
    def lerobot_dataset_options(
        model_config: DictConfig,
        *,
        training_action_horizon: int,
        window_size: int | None,
        require_all_intervene: bool,
    ) -> dict[str, int | bool]:
        """Use the rolling dataset's default action-only windows."""
        return {}

    @staticmethod
    def stage_lerobot_batch(
        batch: dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Move the sampled batch to the actor device before input preparation."""
        return put_tensor_device(batch, device=device)


def get_dagger_policy_adapter(model_type: str) -> type[DAggerPolicyAdapter]:
    """Return the DAgger training adapter for the configured policy type."""
    if model_type == "cosmos3":
        from rlinf.models.embodiment.cosmos3.dagger import Cosmos3DAggerPolicyAdapter

        return Cosmos3DAggerPolicyAdapter
    return DAggerPolicyAdapter
