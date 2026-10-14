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

"""RLinf policy wrapper around the official LeRobot MolmoAct2 policy."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy

# RLinf LIBERO observation key -> MolmoAct2 batch key. MolmoAct2-LIBERO expects the
# camera order [agentview_rgb, wrist_rgb], and upstream collects the views by
# iterating the ``observation.images.*`` keys in insertion order, so this mapping is
# ordered on purpose. Separate per-view keys are used instead of a single ``images``
# entry, otherwise upstream infers the two views as ``batch_size=2``.
_IMAGE_KEY_MAP = (
    ("main_images", "observation.images.agentview"),
    ("wrist_images", "observation.images.wrist"),
)

_REQUIRED_OBS_KEYS = ("main_images", "wrist_images", "states", "task_descriptions")


class MolmoAct2ForRLActionPrediction(nn.Module, BasePolicy):
    """Wrap a LeRobot ``MolmoAct2Policy`` into RLinf's embodied policy interface.

    The upstream policy is inference-only (its ``forward`` and ``get_optim_params``
    raise ``NotImplementedError``), so this wrapper supports evaluation rollouts
    only: :meth:`predict_action_batch` is implemented and :meth:`default_forward`
    raises.

    Args:
        policy: An instantiated ``lerobot.policies.molmoact2.MolmoAct2Policy``.
        action_dim: Environment action dimension (LIBERO Franka: 7).
        num_action_chunks: Number of action steps RLinf executes per inference.
            ``MolmoAct2Policy.select_action`` pops a single step from its internal
            per-env queue per call, so this must be 1.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        action_dim: int = 7,
        num_action_chunks: int = 1,
    ):
        super().__init__()
        self.policy = policy
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks

    def default_forward(self, **kwargs):
        """Raise: the MolmoAct2 integration does not support training."""
        raise NotImplementedError(
            "The MolmoAct2 integration is evaluation-only; training forwards are "
            "not implemented. Use runner.only_eval=True."
        )

    def reset(self) -> None:
        """Clear the wrapped policy's per-env action queues and depth caches.

        ``MolmoAct2Policy`` refills a per-environment queue of ``n_action_steps``
        actions and pops one per call, so an episode whose length is not a
        multiple of ``n_action_steps`` leaves stale actions behind that the next
        episode would execute. The shipped eval configs run every episode to
        truncation at a multiple of it (240 / 320 / 520 against 10), so the queue
        is empty at each boundary and nothing calls this. Configs that terminate
        early or truncate off-cycle must call it themselves.
        """
        self.policy.reset()

    @staticmethod
    def _build_batch(env_obs: dict[str, Any]) -> dict[str, Any]:
        """Convert an RLinf LIBERO observation into a MolmoAct2 batch."""
        missing = [key for key in _REQUIRED_OBS_KEYS if env_obs.get(key) is None]
        if missing:
            raise KeyError(
                f"MolmoAct2 expects RLinf LIBERO observations with keys "
                f"{list(_REQUIRED_OBS_KEYS)}, but {missing} are missing or None. "
                f"Available keys: {sorted(env_obs.keys())}."
            )

        batch = {batch_key: env_obs[obs_key] for obs_key, batch_key in _IMAGE_KEY_MAP}

        task_descriptions = env_obs["task_descriptions"]
        if isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions]
        batch["language_instruction"] = task_descriptions

        # Raw (un-normalized) robot state; MolmoAct2 normalizes it with ``norm_tag``.
        batch["state"] = env_obs["states"]
        return batch

    def predict_action_batch(
        self,
        env_obs: dict[str, Any] | None = None,
        mode: str = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict one action chunk for a batch of environments.

        Args:
            env_obs: RLinf environment observations (see ``_REQUIRED_OBS_KEYS``).
            mode: Rollout mode. Only ``"eval"`` is supported.
            **kwargs: Ignored. MolmoAct2 runs continuous-action inference, so the
                token sampling parameters RLinf passes for autoregressive policies
                do not apply.

        Returns:
            A tuple of the action chunks with shape
            ``[batch_size, num_action_chunks, action_dim]`` and a result dict with
            placeholder ``prev_logprobs`` / ``prev_values`` and ``forward_inputs``.
        """
        del kwargs
        if env_obs is None:
            raise ValueError("MolmoAct2 predict_action_batch requires env_obs.")
        if mode != "eval":
            raise NotImplementedError(
                f"The MolmoAct2 integration is evaluation-only, got mode={mode!r}."
            )

        batch = self._build_batch(env_obs)
        with torch.no_grad():
            actions = self.policy.select_action(batch)

        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        actions = actions.detach().cpu()

        # RLinf expects action chunks: [B, num_action_chunks, action_dim].
        if actions.ndim == 2:
            chunk_actions = actions.unsqueeze(1)
        elif actions.ndim == 3:
            chunk_actions = actions
        else:
            raise ValueError(
                f"Unexpected MolmoAct2 action shape: {tuple(actions.shape)}."
            )

        if chunk_actions.shape[1] != self.num_action_chunks:
            raise ValueError(
                f"MolmoAct2 returned {chunk_actions.shape[1]} action step(s) per "
                f"inference but model.num_action_chunks is {self.num_action_chunks}. "
                "RLinf's rollout step budget assumes they match."
            )
        if chunk_actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"MolmoAct2 returned actions of dimension {chunk_actions.shape[-1]} "
                f"but model.action_dim is {self.action_dim}."
            )

        batch_size = chunk_actions.shape[0]
        placeholder = torch.zeros(
            batch_size, self.num_action_chunks, device=chunk_actions.device
        )
        result = {
            "prev_logprobs": placeholder,
            "prev_values": placeholder.clone(),
            "forward_inputs": {"action": chunk_actions},
        }
        return chunk_actions, result


__all__ = ["MolmoAct2ForRLActionPrediction"]
