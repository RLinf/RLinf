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


from typing import Any, Callable, Optional

import torch
import torchvision.transforms as transforms
from lerobot.datasets.lerobot_dataset import LeRobotDataset


class FuncRegistry:
    """A registry for functions."""

    def __init__(self):
        self._func_map: dict[str, Callable] = {}

    def register(self, name: str) -> Callable:
        """Register a function with a given name."""

        def decorator(func: Callable) -> Callable:
            self._func_map[name] = func
            return func

        return decorator

    def __getitem__(self, name: str) -> Callable:
        """Get a function by name."""
        return self._func_map[name]

    def keys(self) -> list:
        """Get all registered function names."""
        return list(self._func_map.keys())


FUNC_MAPPING = FuncRegistry()


@FUNC_MAPPING.register("first_frame")
def first_frame(**kwargs: Any) -> list[int]:
    """Return the index of the first frame."""
    return [0]


@FUNC_MAPPING.register("last_frame")
def last_frame(**kwargs: Any) -> list[int]:
    """Return the index of the last frame."""
    return [kwargs["episode_frame_idxs"][-1].item()]


@FUNC_MAPPING.register("closest_timestamp")
def closest_timestamp(**kwargs: Any) -> list[int]:
    """Return the index of the frame closest to the target timestamp."""
    target_timestamp = kwargs["target_timestamp"]
    closest_idx = torch.argmin(
        torch.abs(kwargs["episode_timestamps"] - target_timestamp)
    )
    return [closest_idx.item()]


@FUNC_MAPPING.register("first_n_frames")
def first_n_frames(**kwargs: Any) -> list[int]:
    """Return the indices of the first n frames."""
    n = kwargs["start_n_frames"]
    return list(range(n))


@FUNC_MAPPING.register("last_n_frames")
def last_n_frames(**kwargs: Any) -> list[int]:
    """Return the indices of the last n frames."""
    n = kwargs["target_n_frames"]
    return list(range(-n, 0))


class LeRobotDatasetWrapper(torch.utils.data.Dataset):
    """
    A wrapper for the LeRobotDataset to provide custom frame selection policies.

    Args:
        repo_id: The repository ID of the dataset on the Hugging Face Hub.
        root: The root directory where the dataset is stored.
        start_select_policy: The policy to select the start frames.
        target_select_policy: The policy to select the target frames.
        target_timestamp: The target timestamp for the 'closest_timestamp' policy.
        start_n_frames: The number of start frames for the 'first_n_frames' policy.
        target_n_frames: The number of target frames for the 'last_n_frames' policy.
    """

    def __init__(
        self,
        repo_id: str,
        root: str,
        start_select_policy: str,
        target_select_policy: str,
        camera_names: list[str],
        target_timestamp: float = 10**4,
        start_n_frames: int = 1,
        target_n_frames: int = 1,
        camera_heights: Optional[int] = None,
        camera_widths: Optional[int] = None,
    ):
        if camera_heights is None or camera_widths is None:
            image_transforms = None
        else:
            image_transforms = transforms.Compose(
                [
                    transforms.Resize((camera_heights, camera_widths)),
                    transforms.Lambda(
                        lambda img: (img * 255).byte()
                        if img.dtype == torch.float32
                        else img
                    ),
                    transforms.Lambda(
                        lambda img: img.permute(1, 2, 0)
                        if len(img.shape) == 3 and img.shape[0] in [1, 3, 4]
                        else img
                    ),
                ]
            )
        self._lerobot_dataset = LeRobotDataset(
            repo_id, root, image_transforms=image_transforms
        )
        self.timestamps = torch.stack(
            self._lerobot_dataset.hf_dataset["timestamp"]
        ).numpy()
        self.action_dim = self._lerobot_dataset.features["action"]["shape"][0]
        for camera_name in camera_names:
            assert camera_name in self._lerobot_dataset.meta.camera_keys
        self.camera_names = camera_names
        self.camera_heights = camera_heights
        self.camera_widths = camera_widths

        assert start_select_policy in FUNC_MAPPING.keys(), (
            f"start_select_policy {start_select_policy} not in {FUNC_MAPPING.keys()}"
        )
        assert target_select_policy in FUNC_MAPPING.keys(), (
            f"target_select_policy {target_select_policy} not in {FUNC_MAPPING.keys()}"
        )
        self.start_select_policy = FUNC_MAPPING[start_select_policy]
        self.target_select_policy = FUNC_MAPPING[target_select_policy]
        self.target_timestamp = target_timestamp
        self.start_n_frames = start_n_frames
        self.target_n_frames = target_n_frames

    def __len__(self) -> int:
        """Return the total number of episodes."""
        return self._lerobot_dataset.meta.total_episodes

    def _get_frame_indices(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start_index = self._lerobot_dataset.episode_data_index["from"][index]
        end_index = self._lerobot_dataset.episode_data_index["to"][index]
        return torch.arange(start_index, end_index), self.timestamps[
            start_index:end_index
        ]

    def _select_frames(self, policy: Callable, **kwargs: Any) -> list[dict[str, Any]]:
        indices = policy(**kwargs)
        start_index = kwargs["episode_frame_idxs"][0]
        return [self._lerobot_dataset[int(start_index + idx)] for idx in indices]

    def __getitem__(self, index: int) -> dict[str, Any]:
        """
        Get an item from the dataset.

        Args:
            index: The index of the episode.

        Returns:
            A dictionary containing the start items, target items, episode index, task, and dataset metadata.
        """
        episode_frame_idxs, episode_timestamps = self._get_frame_indices(index)

        policy_kwargs = {
            "episode_frame_idxs": episode_frame_idxs - episode_frame_idxs[0],
            "episode_timestamps": episode_timestamps,
            "episode_reward": None,
            "target_timestamp": self.target_timestamp,
            "start_n_frames": self.start_n_frames,
            "target_n_frames": self.target_n_frames,
        }

        start_items = self._select_frames(self.start_select_policy, **policy_kwargs)
        target_items = self._select_frames(self.target_select_policy, **policy_kwargs)

        return {
            "start_items": start_items,
            "target_items": target_items,
            "episode_index": index,
            "task": self._lerobot_dataset[episode_frame_idxs[0].item()]["task"],
            "dataset_meta": self._lerobot_dataset.meta.episodes_stats[index],
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="unitreerobotics/G1_Dex1_MountCameraRedGripper_Dataset",
    )
    parser.add_argument(
        "--root",
        type=str,
        default="",
    )
    parser.add_argument("--start_select_policy", type=str, default="first_frame")
    parser.add_argument("--target_select_policy", type=str, default="last_frame")
    args = parser.parse_args()

    dataset = LeRobotDatasetWrapper(
        repo_id=args.repo_id,
        root=args.root,
        start_select_policy=args.start_select_policy,
        target_select_policy=args.target_select_policy,
    )

    data = dataset[0]
    print(data)
