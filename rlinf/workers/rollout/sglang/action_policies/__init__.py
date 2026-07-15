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

"""Pluggable embodied action policies for the sglang-serve rollout worker.

An action policy knows how to turn an RLinf env observation batch into model
action chunks over a launched ``sglang serve`` HTTP server. The general worker
(``SGLangEmbodiedWorker``) launches the serve and drives the eval channel
loop; the model-specific action API (e.g. Cosmos3 ``/v1/videos`` action
generation) lives in a registered policy selected by ``rollout.model.model_type``.

A non-default model reuses the whole embodied design by:
  1. registering a policy here (``@register_action_policy("<model_type>")``),
  2. providing its model-specific config under ``rollout.model.<model_type>``,
  3. letting the simulator/embodiment facts (``env_type``, ``raw_action_dim``)
     come from the env config (``cfg.env.eval``) — no rollout-side embodiment
     block is needed.
"""

from rlinf.workers.rollout.sglang.action_policies.base import (
    EmbodiedActionPolicy,
)
from rlinf.workers.rollout.sglang.action_policies.registry import (
    get_action_policy_cls,
    register_action_policy,
)

# Register built-in policies (import for its side effect).
from rlinf.workers.rollout.sglang.action_policies import cosmos3  # noqa: F401,E401

__all__ = [
    "EmbodiedActionPolicy",
    "get_action_policy_cls",
    "register_action_policy",
]
