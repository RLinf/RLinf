# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter


@pytest.mark.parametrize(
    ("use_videos", "expected_dtype"),
    [(False, "image"), (True, "video")],
)
def test_auto_generated_camera_features_follow_config(use_videos, expected_dtype):
    lerobot_module = ModuleType("lerobot")
    lerobot_module.__path__ = []
    datasets_module = ModuleType("lerobot.datasets")
    datasets_module.__path__ = []
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_cls = MagicMock()
    dataset_cls.create.return_value = MagicMock()
    dataset_module.LeRobotDataset = dataset_cls

    modules = {
        "lerobot": lerobot_module,
        "lerobot.datasets": datasets_module,
        "lerobot.datasets.lerobot_dataset": dataset_module,
    }
    with (
        patch.dict(sys.modules, modules),
        patch("rlinf.data.storage.lerobot.writer._silence_hf_datasets_progress_bars"),
    ):
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id="test_dataset",
            image_shape=(224, 224, 3),
            wrist_image_keys={"wrist_image": (224, 224, 3)},
            extra_view_image_keys={
                "extra_view_image-0": (224, 224, 3),
                "extra_view_image-1": (224, 224, 3),
            },
            use_videos=use_videos,
        )

    features = dataset_cls.create.call_args.kwargs["features"]
    camera_keys = [
        "image",
        "wrist_image",
        "extra_view_image-0",
        "extra_view_image-1",
    ]
    assert all(features[key]["dtype"] == expected_dtype for key in camera_keys)
    assert dataset_cls.create.call_args.kwargs["use_videos"] is use_videos
