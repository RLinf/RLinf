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

"""Export a LeRobot dataset to image/state/action-chunk arrays for CNN SFT.

Frames are decoded once, resized with the same ``cv2.INTER_AREA`` call the
simulator adapters use, and written as memory-mappable ``.npy`` files that
``rlinf.data.datasets.image_action_chunks`` reads during training. Past an
episode's end LeRobot repeats its last action, as OpenPI's loader does.
"""

import json
import os

import cv2
import numpy as np
import tqdm
import tyro
from torch.utils.data import DataLoader

from rlinf.data.storage.lerobot import resolve_lerobot_dataset_root
from rlinf.utils.logging import get_logger

logger = get_logger()


def main(
    repo_id: str,
    output: str,
    cameras: tuple[str, ...] = ("front", "wrist"),
    image_size: int = 128,
    chunk: int = 10,
    state_dim: int = 23,
    batch_size: int = 64,
    num_workers: int = 8,
) -> None:
    """Export ``repo_id`` (a local LeRobot directory or Hub id) to ``output``.

    The first camera becomes ``main_images`` and the others ``extra_view_images``;
    ``chunk`` future actions are taken at the dataset frame rate and only the first
    ``state_dim`` state entries (the robot proprioception) are kept.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    root = resolve_lerobot_dataset_root(repo_id)
    dataset = LeRobotDataset(repo_id, root=root)
    fps = dataset.fps
    dataset = LeRobotDataset(
        repo_id,
        root=root,
        delta_timestamps={"action": [i / fps for i in range(chunk)]},
    )
    image_keys = [f"observation.images.{name}" for name in cameras]
    num_frames = len(dataset)
    os.makedirs(output, exist_ok=True)

    def open_array(name, shape, dtype):
        return np.lib.format.open_memmap(
            os.path.join(output, f"{name}.npy"),
            mode="w+",
            dtype=dtype,
            shape=shape,
        )

    size = image_size
    main_images = open_array("main_images", (num_frames, size, size, 3), np.uint8)
    extra_view_images = None
    if len(image_keys) > 1:
        extra_view_images = open_array(
            "extra_view_images",
            (num_frames, len(image_keys) - 1, size, size, 3),
            np.uint8,
        )
    else:
        extra_path = os.path.join(output, "extra_view_images.npy")
        if os.path.exists(extra_path):
            os.remove(extra_path)
    states = open_array("states", (num_frames, state_dim), np.float32)
    actions = open_array(
        "actions",
        (num_frames, chunk, dataset.meta.shapes["action"][0]),
        np.float32,
    )
    episode_index = open_array("episode_index", (num_frames,), np.int64)

    def to_uint8(frames):
        frames = (frames.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)
        return np.stack(
            [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames]
        )

    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    start = 0
    for batch in tqdm.tqdm(loader, desc="export"):
        end = start + batch["index"].shape[0]
        main_images[start:end] = to_uint8(batch[image_keys[0]])
        if extra_view_images is not None:
            for i, key in enumerate(image_keys[1:]):
                extra_view_images[start:end, i] = to_uint8(batch[key])
        states[start:end] = batch["observation.state"][:, :state_dim].numpy()
        actions[start:end] = batch["action"].numpy()
        episode_index[start:end] = batch["episode_index"].numpy()
        start = end
    assert start == num_frames, f"exported {start} of {num_frames} frames"
    for array in (main_images, extra_view_images, states, actions, episode_index):
        if array is not None:
            array.flush()

    flat_actions = np.asarray(actions[:, 0])
    meta = {
        "repo_id": repo_id,
        "fps": fps,
        "cameras": cameras,
        "image_size": size,
        "chunk": chunk,
        "state_dim": state_dim,
        "num_frames": num_frames,
        "num_episodes": int(dataset.meta.total_episodes),
        "action_mean": flat_actions.mean(axis=0).tolist(),
        "action_std": flat_actions.std(axis=0).tolist(),
    }
    with open(os.path.join(output, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(
        "Exported image/action chunks: %s",
        json.dumps({k: v for k, v in meta.items() if "action_" not in k}, indent=2),
    )


if __name__ == "__main__":
    tyro.cli(main)
