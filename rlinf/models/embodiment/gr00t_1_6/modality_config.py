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



from typing import Optional, List, Dict, Any
from pathlib import Path

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy


class BaseN16DataConfig:
    def __init__(self):
        self.emb_tag: EmbodimentTag = None
        self._policy: Optional[Gr00tPolicy] = None
        self._modality_config = None
        self._transform = None

    def _init_policy(self):
        if self._policy is None:
            dummy_path = "/tmp/dummy_gr00t_checkpoint"
            self._policy = Gr00tPolicy(
                model_path=dummy_path,
                embodiment_tag=self.emb_tag,
                device="cpu",
                strict=False,
            )
            self._modality_config = self._policy.get_modality_config()

            for attr in ["_modality_transform", "modality_transform", "_transform", "transform"]:
                if hasattr(self._policy, attr):
                    self._transform = getattr(self._policy, attr)
                    print(f"✅ N1.6 loaded successfully transform: {attr}")
                    break
            else:
                print("⚠️ do not find transform attribute, but modality_config is loaded")

    def modality_config(self) -> Dict[str, Any]:
        self._init_policy()
        return self._modality_config

    def transform(self):
        self._init_policy()
        return self._transform

    def get_video_keys(self) -> List[str]:
        self._init_policy()
        return self._modality_config.get("video", {}).get("modality_keys", [])

    def get_state_keys(self) -> List[str]:
        self._init_policy()
        return self._modality_config.get("state", {}).get("modality_keys", [])

    def get_action_keys(self) -> List[str]:
        self._init_policy()
        return self._modality_config.get("action", {}).get("modality_keys", [])


class LiberoFrankaDataConfig(BaseN16DataConfig):
    def __init__(self):
        super().__init__()
        self.emb_tag = EmbodimentTag.LIBERO_PANDA
        self.video_keys = ["video.image", "video.wrist_image"]
        self.state_keys = ["state.proprio"]
        self.action_keys = ["action.franka_arm"]
        self.language_keys = ["language.task"]


class ManiskillWidowXDataConfig(BaseN16DataConfig):
    def __init__(self):
        super().__init__()
        self.emb_tag = EmbodimentTag.OXE_WIDOWX
        self.video_keys = ["video.ego_view"]
        self.state_keys = ["state.left_arm"]
        self.action_keys = ["action.left_arm"]
        self.language_keys = ["annotation.human.action.task_description"]


def load_data_config(config_str: str):
    if "LiberoFrankaDataConfig" in config_str:
        return LiberoFrankaDataConfig()
    elif "ManiskillWidowXDataConfig" in config_str:
        return ManiskillWidowXDataConfig()
    else:
        raise ValueError(f"Unknown config: {config_str}")