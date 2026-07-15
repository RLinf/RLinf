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

"""Base class for pluggable embodied action policies.

A policy turns an RLinf env observation batch into model action chunks by
calling a launched ``sglang serve`` HTTP server. The general rollout worker
(``SGLangEmbodiedWorker``) owns: the serve subprocess lifecycle, the
NO_PROXY/log-file/``/health`` plumbing, the eval channel loop, and the worker's
own HTTP server. A policy owns only the model-specific action API: how to build
the request body from ``env_obs``, which endpoint to hit, how to poll and parse
the returned action.

The contract is intentionally narrow so that a new model can plug in by
implementing :meth:`infer` (and optionally :meth:`evaluate_actions`):
"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import torch

from rlinf.utils.logging import get_logger


class EmbodiedActionPolicy(ABC):
    """Turn env observations into action chunks via a launched sglang serve.

    Args:
        cfg: the full RLinf ``DictConfig``. Policies read model-specific config
            from ``cfg.rollout.model.<model_type>`` and embodiment facts
            (``env_type``, ``raw_action_dim``) from ``cfg.env.eval`` — no
            rollout-side embodiment block is required.
        server_url: base URL of the already-launched ``sglang serve``
            (e.g. ``http://127.0.0.1:30010``). ``None`` means an external server
            URL was configured and the worker did not spawn one.
        rank: rollout rank of the worker holding this policy.
    """

    def __init__(self, cfg: Any, server_url: "str | None", rank: int):
        self.cfg = cfg
        self.cfg_rollout = cfg.rollout
        self.model_cfg = cfg.rollout.model
        self.server_url = server_url
        self.rank = rank
        self.logger = get_logger()

    @abstractmethod
    def infer(
        self, env_obs: dict, mode: Literal["train", "eval"] = "eval"
    ) -> "tuple[torch.Tensor, dict]":
        """Map an env observation batch to action chunks.

        Args:
            env_obs: the obs dict received from the env worker over the channel
                (uniform across simulators: ``main_images`` + ``task_descriptions``
                + optional proprioception).
            mode: ``"train"`` or ``"eval"`` (some policies have no distinction).

        Returns:
            (actions, info) where ``actions`` is a float tensor of shape
            ``[N, num_action_chunks, action_dim]`` and ``info`` is a dict.
        """
        raise NotImplementedError

    def evaluate_actions(self, observations: Any) -> dict:
        """Back the worker's HTTP ``/evaluate`` route.

        Default: normalize the request body to a list of obs dicts, run
        :meth:`infer`, and return a JSON payload ``{"actions": ..., "shape": ...}``.
        Override for a model-specific response shape.
        """
        if isinstance(observations, dict):
            obs_list = observations.get("observations") or [observations]
        elif isinstance(observations, list):
            obs_list = observations
        else:
            obs_list = []

        if not obs_list:
            raise ValueError("request body must be a non-empty object or list")

        # Treat a list of per-env obs dicts as an env batch (one "env" per item).
        env_obs: dict[str, Any] = {}
        keys = obs_list[0].keys() if isinstance(obs_list[0], dict) else []
        for key in keys:
            env_obs[key] = [o[key] for o in obs_list]
        actions, _ = self.infer(env_obs, mode="eval")
        actions_list = actions.detach().cpu().tolist()
        return {
            "actions": actions_list,
            "shape": list(actions.shape),
        }

    def shutdown(self) -> None:
        """Release model-specific resources. Default: no-op (serve lifecycle is
        owned by the worker)."""
        return None
