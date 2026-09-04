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

from rlinf.models.embodiment.prefix_ft.types import PrefixObs


class StateHistoryBuffer:
    """Per-env ring buffer of proprio at PrefixObs / decision rate.

    Disabled by default: ``fuse`` is identity and ``extra_z_dim`` is 0.
    When enabled:

        fused_z = cat(z_rl, s_{t-K+1}, ..., s_t)   # last slot is current
    """

    def __init__(
        self,
        *,
        enable: bool = False,
        steps: int = 4,
        proprio_dim: int = 0,
        pad: Literal["zero", "repeat"] = "zero",
    ):
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
        return 0 if not self.enabled else self.steps * self.proprio_dim

    def reset(
        self,
        batch_size: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        """Clear the ring (full batch or the done subset)."""
        if not self.enabled:
            return
        if mask is not None:
            if self._buffer is None:
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
        """If disabled, return ``obs`` unchanged.

        If enabled, append ``obs['proprio']`` and replace ``z_rl`` with the
        fused vector. ``proprio`` and ``ref_chunk`` are unchanged.

        ``commit=False`` peeks the fused vector without mutating the ring
        (used for ``final_obs`` / next_obs so the next decision does not
        double-push the same proprio).
        """
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
        hist = self._next_history(proprio, commit=commit)
        fused = torch.cat([z_rl, hist.reshape(batch_size, -1)], dim=-1)
        return {
            "z_rl": fused,
            "proprio": obs["proprio"],
            "ref_chunk": obs["ref_chunk"],
        }

    def _allocate(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        self._buffer = torch.zeros(
            batch_size, self.steps, self.proprio_dim, device=device, dtype=dtype
        )
        self._seen = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _ensure_buffer(
        self, batch_size: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if (
            self._buffer is None
            or self._buffer.shape[0] != batch_size
            or self._buffer.shape[-1] != self.proprio_dim
        ):
            self._allocate(batch_size, device, dtype)
            return
        if self._buffer.device != device or self._buffer.dtype != dtype:
            self._buffer = self._buffer.to(device=device, dtype=dtype)
            self._seen = self._seen.to(device=device)

    def _next_history(self, proprio: torch.Tensor, *, commit: bool) -> torch.Tensor:
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
