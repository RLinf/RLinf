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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class EntropyTemperature(nn.Module):
    """
    Temperature module for SAC automatic entropy tuning.
    """

    def __init__(
        self,
        initial_alpha: float,
        alpha_type: str = "softplus",
        device=None,
        dtype=torch.float32,
        num_temperatures: int = 1,
    ):
        """Create the temperatures.

        Args:
            initial_alpha: Starting temperature.
            alpha_type: ``"softplus"``, ``"exp"`` or ``"fixed_alpha"``.
            device: Device of the parameters.
            dtype: Dtype of the parameters.
            num_temperatures: Temperatures to tune, one per env configuration.
        """
        super().__init__()
        self.alpha_type = alpha_type

        if alpha_type == "exp":
            init = np.log(initial_alpha)
        elif alpha_type == "softplus":
            # softplus^{-1}(x) = log(exp(x) - 1)
            init = np.log(np.exp(initial_alpha) - 1.0)
        elif alpha_type == "fixed_alpha":
            self.register_buffer(
                "fixed_alpha",
                torch.full(
                    (num_temperatures,),
                    initial_alpha,
                    device=device,
                    dtype=dtype,
                ),
                persistent=True,
            )
            init = initial_alpha
        else:
            raise NotImplementedError(f"Unknown alpha_type: {alpha_type}")

        self.base_alpha = nn.Parameter(
            torch.full((num_temperatures,), float(init), device=device, dtype=dtype)
        )
        self._register_load_state_dict_pre_hook(self._load_scalar_base_alpha)

    @staticmethod
    def _load_scalar_base_alpha(state_dict, prefix, *args) -> None:
        # Checkpoints from before per-env temperatures hold a scalar.
        key = prefix + "base_alpha"
        if key in state_dict and state_dict[key].dim() == 0:
            state_dict[key] = state_dict[key].reshape(1)

    def compute_alpha(self) -> torch.Tensor:
        if self.alpha_type == "exp":
            return self.base_alpha.exp()
        elif self.alpha_type == "softplus":
            return F.softplus(self.base_alpha)
        elif self.alpha_type == "fixed_alpha":
            return self.fixed_alpha
        else:
            raise NotImplementedError

    @property
    def alpha(self) -> float:
        """Mean temperature, which is the temperature when there is one."""
        return self.compute_alpha().mean().item()

    def sample_alpha(self, env_index: torch.Tensor | None) -> torch.Tensor:
        """Return each sample's temperature, picked by the env it came from.

        Args:
            env_index: Env of each sample, shape ``[B]``; ``None`` uses the mean
                temperature.

        Returns:
            Temperatures of shape ``[B, 1]``, or a scalar without ``env_index``.
        """
        alpha = self.compute_alpha()
        if env_index is None:
            return alpha.mean()
        return alpha[env_index.to(alpha.device)].unsqueeze(-1)
