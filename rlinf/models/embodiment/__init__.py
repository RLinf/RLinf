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

from omegaconf import DictConfig, OmegaConf


def get_dagger_component_class(default_cls: type, cfg: DictConfig | None) -> type:
    """Resolve a model-specific implementation for a built-in DAgger component."""
    if (
        not isinstance(cfg, DictConfig)
        or OmegaConf.select(cfg, "algorithm.loss_type") != "embodied_dagger"
        or OmegaConf.select(cfg, "actor.model.model_type") != "cosmos3"
    ):
        return default_cls

    from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner

    if default_cls is AsyncEmbodiedRunner:
        from rlinf.models.embodiment.cosmos3.tasks.async_dagger_runner import (
            Cosmos3AsyncDAggerRunner,
        )

        return Cosmos3AsyncDAggerRunner

    from rlinf.workers.actor.fsdp_dagger_policy_worker import EmbodiedDAGGERFSDPPolicy

    if default_cls is EmbodiedDAGGERFSDPPolicy:
        from rlinf.models.embodiment.cosmos3.tasks.dagger_actor import (
            Cosmos3DAggerActor,
        )

        return Cosmos3DAggerActor

    from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
        AsyncEmbodiedDAGGERFSDPPolicy,
    )

    if default_cls is AsyncEmbodiedDAGGERFSDPPolicy:
        from rlinf.models.embodiment.cosmos3.tasks.async_dagger_actor import (
            Cosmos3AsyncDAggerActor,
        )

        return Cosmos3AsyncDAggerActor

    return default_cls
