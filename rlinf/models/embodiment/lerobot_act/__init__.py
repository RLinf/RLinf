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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype: torch.dtype | None = None):
    """Load a LeRobot ACT checkpoint and wrap it for RLinf.

    Args:
        cfg: Model config. model_path is a local directory or Hugging Face
            repo id of a LeRobot ACT checkpoint; num_action_chunks is the
            number of predicted actions executed per call.
        torch_dtype: Ignored; ACT runs in fp32.

    Returns:
        A LeRobotACTPolicy.
    """
    try:
        from lerobot.common.policies.act.modeling_act import ACTPolicy
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "LeRobot ACT requires the pinned LeRobot version. Install it with "
            "'bash requirements/install.sh embodied --model lerobot_act --env aloha'."
        ) from e

    from rlinf.models.embodiment.lerobot_act.lerobot_act_policy import (
        LeRobotACTPolicy,
    )

    act_policy = ACTPolicy.from_pretrained(cfg.model_path)
    return LeRobotACTPolicy(act_policy, num_action_chunks=cfg.num_action_chunks)
