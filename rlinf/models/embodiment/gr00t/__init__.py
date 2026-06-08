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

"""GR00T embodiment models (N1.5 / N1.6) with shared I/O utilities."""

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    model_type = str(cfg.get("model_type", "gr00t"))
    if model_type == "gr00t_n1d6":
        from rlinf.models.embodiment.gr00t.gr00t_n1d6 import get_model as get_model_n1d6

        return get_model_n1d6(cfg, torch_dtype)

    if model_type in ("gr00t", "gr00t_n1d5"):
        from rlinf.models.embodiment.gr00t.gr00t_n1d5 import get_model as get_model_n1d5

        return get_model_n1d5(cfg, torch_dtype)

    raise ValueError(
        f"Unsupported GR00T model_type: {model_type!r}. "
        f"Supported values: ['gr00t', 'gr00t_n1d5', 'gr00t_n1d6']."
    )


__all__ = ["get_model"]
