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

from typing import Literal

import torch

from rlinf.models.embodiment.prefix.types import PrefixObs


class StateHistoryBuffer:
    """Maintain per-environment proprio history at the policy decision rate."""

    def __init__(
        self,
        *,
        enable: bool = False,
        steps: int = 4,
        proprio_dim: int = 0,
        pad: Literal["zero", "repeat"] = "zero",
    ) -> None:
        if steps < 1:
            raise ValueError(f"state_history.steps must be >= 1, got {steps}.")
        if pad not in ("zero", "repeat"):
            raise ValueError(
                f"state_history.pad must be 'zero' or 'repeat', got {pad!r}."
            )
        self.enabled = bool(enable)
        self.steps = int(steps)
        self.proprio_dim = int(proprio_dim)
        self.pad = pad
        self._buffer: torch.Tensor | None = None
        self._seen: torch.Tensor | None = None

    @property
    def extra_z_dim(self) -> int:
        """Return the feature width appended to ``z_rl``."""
        return self.steps * self.proprio_dim if self.enabled else 0

    def reset(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Clear all history or only environments selected by ``mask``."""
        if not self.enabled:
            return
        if mask is not None:
            if self._buffer is None or self._seen is None:
                return
            done = mask.to(device=self._buffer.device, dtype=torch.bool).reshape(-1)
            if done.numel() != self._buffer.shape[0]:
                raise ValueError(
                    f"history reset mask length {done.numel()} != buffer batch "
                    f"{self._buffer.shape[0]}."
                )
            self._buffer[done] = 0
            self._seen[done] = False
            return

        if batch_size is None or device is None or dtype is None:
            self._buffer = None
            self._seen = None
            return
        self._allocate(batch_size, device, dtype)

    def fuse(self, obs: PrefixObs, *, commit: bool = True) -> PrefixObs:
        """Append state history to ``z_rl`` and optionally advance the buffer."""
        if not self.enabled:
            return obs

        proprio = _flatten_proprio(obs["proprio"])
        z_rl = obs["z_rl"]
        if not torch.is_tensor(z_rl):
            z_rl = torch.as_tensor(z_rl)
        z_rl = z_rl.to(device=proprio.device, dtype=torch.float32)
        if z_rl.ndim == 1:
            z_rl = z_rl.unsqueeze(0)

        batch_size = proprio.shape[0]
        self._ensure_buffer(batch_size, proprio.device, proprio.dtype)
        history = self._next_history(proprio, commit=commit)
        fused = torch.cat([z_rl, history.reshape(batch_size, -1)], dim=-1)
        return {
            "z_rl": fused,
            "proprio": obs["proprio"],
            "ref_chunk": obs["ref_chunk"],
        }

    def _allocate(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self._buffer = torch.zeros(
            batch_size,
            self.steps,
            self.proprio_dim,
            device=device,
            dtype=dtype,
        )
        self._seen = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _ensure_buffer(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if (
            self._buffer is None
            or self._seen is None
            or self._buffer.shape[0] != batch_size
            or self._buffer.shape[-1] != self.proprio_dim
        ):
            self._allocate(batch_size, device, dtype)
            return
        if self._buffer.device != device or self._buffer.dtype != dtype:
            self._buffer = self._buffer.to(device=device, dtype=dtype)
            self._seen = self._seen.to(device=device)

    def _next_history(self, proprio: torch.Tensor, *, commit: bool) -> torch.Tensor:
        if self._buffer is None or self._seen is None:
            raise RuntimeError("State history must be allocated before it is updated.")
        first = ~self._seen
        rolled = torch.roll(self._buffer, shifts=-1, dims=1)
        rolled[:, -1] = proprio
        if self.pad == "repeat" and first.any():
            rolled[first] = proprio[first].unsqueeze(1).expand(-1, self.steps, -1)
        if commit:
            self._buffer = rolled
            self._seen[first] = True
            return self._buffer
        return rolled


def _flatten_proprio(proprio: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(proprio):
        proprio = torch.as_tensor(proprio)
    proprio = proprio.to(dtype=torch.float32)
    if proprio.ndim == 1:
        proprio = proprio.unsqueeze(0)
    elif proprio.ndim > 2:
        proprio = proprio.reshape(proprio.shape[0], -1)
    return proprio
