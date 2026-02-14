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

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.cma.cma_action_model import CMAConfig, CMAPolicy


def get_model(cfg: DictConfig, torch_dtype=torch.float32):
    """Get CMA Policy model with component-wise weight loading.

    Args:
        cfg: Configuration dict with model parameters
        torch_dtype: Torch dtype for model weights

    Returns:
        CMAPolicy model instance
    """

    # Convert config to dict and create CMAConfig
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    model_config = CMAConfig()
    model_config.update_from_dict(config_dict)

    # Create model
    observation_space = None  # Can be provided if needed
    model = CMAPolicy(cfg=model_config, observation_space=observation_space)

    # Convert to specified dtype
    model = model.to(torch_dtype)

    model_dict = torch.load(cfg.model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(model_dict)

    return model
