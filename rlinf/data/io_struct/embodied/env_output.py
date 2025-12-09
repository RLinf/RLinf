from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from rlinf.data.io_struct.utils import put_tensor_device

@dataclass(kw_only=True)
class EnvOutput:
    simulator_type: str
    obs: Dict[str, Any]
    final_obs: Optional[Dict[str, Any]] = None
    dones: Optional[torch.Tensor] = None  # [B]
    terminations: Optional[torch.Tensor] = None # [B]
    truncations: Optional[torch.Tensor] = None # [B]
    rewards: Optional[torch.Tensor] = None  # [B]

    def __post_init__(self):
        self.obs = put_tensor_device(self.obs, "cpu")
        self.final_obs = (
            put_tensor_device(self.final_obs, "cpu") if self.final_obs is not None else None
        )
        self.dones = self.dones.cpu().contiguous() if self.dones is not None else None
        self.terminations = self.terminations.cpu().contiguous() if self.terminations is not None else None
        self.truncations = self.truncations.cpu().contiguous() if self.truncations is not None else None
        self.rewards = (
            self.rewards.cpu().contiguous() if self.rewards is not None else None
        )

    def prepare_observations(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        wrist_image_tensor = None
        if self.simulator_type == "libero":
            image_tensor = torch.stack(
                [
                    value.clone().permute(2, 0, 1)
                    for value in obs["images_and_states"]["full_image"]
                ]
            )
            if "wrist_image" in obs["images_and_states"]:
                wrist_image_tensor = torch.stack(
                    [
                        value.clone().permute(2, 0, 1)
                        for value in obs["images_and_states"]["wrist_image"]
                    ]
                )
        elif self.simulator_type == "maniskill":
            image_tensor = obs["images"]
        elif self.simulator_type == "robotwin":
            image_tensor = obs["images"]
        elif self.simulator_type == "real":
            image_tensor = obs["images"]
        else:
            raise NotImplementedError

        states = None
        if "images_and_states" in obs and "state" in obs["images_and_states"]:
            states = obs["images_and_states"]["state"]
        if "states" in obs:
            states = obs["states"]

        task_descriptions = (
            list(obs["task_descriptions"]) if "task_descriptions" in obs else None
        )

        return {
            "images": image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
            "task_descriptions": task_descriptions,
        }

    def to_dict(self):
        env_output_dict = {}

        env_output_dict["obs"] = self.prepare_observations(self.obs)
        env_output_dict["final_obs"] = (
            self.prepare_observations(self.final_obs)
            if self.final_obs is not None
            else None
        )
        env_output_dict["dones"] = self.dones
        env_output_dict["terminations"] = self.terminations
        env_output_dict["truncations"] = self.truncations
        env_output_dict["rewards"] = self.rewards

        return env_output_dict