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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy


def get_dagger_actor_class(
    model_type: str, *, asynchronous: bool = False
) -> "type[EmbodiedDAGGERFSDPPolicy]":
    """Select the training actor used by an embodied DAgger launcher."""
    if model_type == "cosmos3":
        if asynchronous:
            from rlinf.models.embodiment.cosmos3.async_dagger_actor import (
                Cosmos3AsyncDAggerActor,
            )

            return Cosmos3AsyncDAggerActor
        from rlinf.models.embodiment.cosmos3.dagger_actor import Cosmos3DAggerActor

        return Cosmos3DAggerActor
    if asynchronous:
        from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
            AsyncEmbodiedDAGGERFSDPPolicy,
        )

        return AsyncEmbodiedDAGGERFSDPPolicy
    from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy

    return EmbodiedDAGGERFSDPPolicy
