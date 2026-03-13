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

"""Collect success / failure classifier data via teleoperation.

The operator teleoperates the robot through SpaceMouse + glove (same as
``collect_real_data.py``).  At each environment step the **physical
right button** of the SpaceMouse determines the frame label:

- **Right button pressed** → *success* frame
- **Right button NOT pressed** → *failure* frame

After the target number of success frames is reached, all frames
(success + failure) are saved to ``raw_frames.pkl`` together with
individual images and pickle files for ``train_reward_classifier.py``.

To review and filter frames interactively, run the companion script
``review_classifier_data.py`` on the same log directory afterwards.

Usage
-----
.. code-block:: bash

    bash examples/embodiment/collect_classifier_data.sh [config_name] [env_name]

Data is saved to ``logs/<timestamp>-reward-classifier-<env_name>/``.
The same directory can be passed to ``review_classifier_data.py`` for
visual review, or directly to ``train_reward_classifier.py``.
"""

from __future__ import annotations

import datetime
import os
import pickle

import cv2
import hydra
import numpy as np
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


# ======================================================================
# Teleoperation data collection (runs as Ray Worker)
# ======================================================================


class ClassifierDataCollector(Worker):
    """Collect labelled camera frames while teleoperating the robot."""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.successes_needed = cfg.runner.get(
            "successes_needed", 200
        )
        self.save_dir = cfg.runner.logger.log_path

        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

    def _init_image_keys(self, obs):
        """Determine camera key names from the first observation."""
        self.main_image_key = self.env.main_image_key

        extra = obs.get("extra_view_images", None)
        n_extra = 0
        if extra is not None:
            if isinstance(extra, (torch.Tensor, np.ndarray)):
                n_extra = extra.shape[1] if extra.ndim == 5 else 1

        n_total = 1 + n_extra
        all_keys = sorted(f"wrist_{i + 1}" for i in range(n_total))
        self.extra_image_keys = sorted(
            k for k in all_keys if k != self.main_image_key
        )
        self.image_keys = sorted(all_keys)
        self.log_info(
            f"Cameras: {self.image_keys} (main: {self.main_image_key})"
        )

    def _extract_frames(self, obs) -> dict[str, np.ndarray] | None:
        """Extract all camera frames from wrapped observation.

        Returns a dict mapping camera key (e.g. ``wrist_1``, ``wrist_2``)
        to a ``(H, W, 3)`` BGR uint8 numpy array, or ``None`` if no
        images are available.
        """
        main = obs.get("main_images", None)
        if main is None:
            return None
        if isinstance(main, torch.Tensor):
            main = main.cpu().numpy()
        if main.ndim == 4:
            main = main[0]

        frames = {self.main_image_key: main.copy()}

        extra = obs.get("extra_view_images", None)
        if extra is not None:
            if isinstance(extra, torch.Tensor):
                extra = extra.cpu().numpy()
            if extra.ndim == 5:
                extra = extra[0]  # (N_extra, H, W, 3)
            for i, key in enumerate(self.extra_image_keys):
                if i < extra.shape[0]:
                    frames[key] = extra[i].copy()

        return frames

    @staticmethod
    def _get_right_button(info) -> bool:
        """Safely extract the physical right button state from info.

        After SyncVectorEnv, ``info["right"]`` is a numpy bool array
        of shape ``(1,)``.
        """
        right = info.get("right", None)
        if right is None:
            return False
        if isinstance(right, np.ndarray):
            return bool(right[0])
        if isinstance(right, (list, tuple)):
            return bool(right[0])
        if isinstance(right, torch.Tensor):
            return bool(right.item())
        return bool(right)

    def run(self):
        from tqdm import tqdm

        max_steps = self.cfg.env.eval.max_episode_steps
        action_dim = self.env.env.single_action_space.shape[0]

        successes: list[dict] = []
        failures: list[dict] = []
        episode_count = 0

        self.log_info(
            f"Target: collect {self.successes_needed} success frames  "
            f"auto-reset every {max_steps} steps"
        )

        obs, _ = self.env.reset()
        self._init_image_keys(obs)

        while len(successes) < self.successes_needed:
            episode_count += 1
            self.log_info(
                f"\n{'#' * 50}\n"
                f"  Episode {episode_count}  "
                f"success: {len(successes)}/{self.successes_needed}  "
                f"failure: {len(failures)}\n"
                f"  >>> Start teleoperation -- right button marks success <<<\n"
                f"{'#' * 50}"
            )

            pbar = tqdm(
                range(max_steps),
                desc=f"Ep{episode_count} "
                     f"[S:{len(successes)}/{self.successes_needed}]",
                ncols=80,
                leave=False,
            )

            for step_in_ep in pbar:
                action = np.zeros((1, action_dim))
                next_obs, reward, terminated, truncated, info = (
                    self.env.step(action)
                )

                frames = self._extract_frames(next_obs)
                right_pressed = self._get_right_button(info)

                if frames is not None:
                    entry = {
                        "images": frames,
                        "label": "success" if right_pressed else "failure",
                    }
                    if right_pressed:
                        successes.append(entry)
                    else:
                        failures.append(entry)

                # Update progress bar description
                pbar.set_description(
                    f"Ep{episode_count} "
                    f"[S:{len(successes)}/{self.successes_needed} "
                    f"F:{len(failures)}]"
                )

                # Episode ends only by truncation (max steps)
                # terminated is ignored — we don't want target-pos success
                # to cut the episode short
                if bool(truncated):
                    break

                obs = next_obs

            pbar.close()
            self.log_info(
                f"Episode {episode_count} done  "
                f"success: {len(successes)}  failure: {len(failures)}"
            )

            # Check if we have enough before resetting
            if len(successes) >= self.successes_needed:
                break

            obs, _ = self.env.reset()

        # Save raw frames for optional review later
        all_frames = successes + failures
        os.makedirs(self.save_dir, exist_ok=True)
        raw_path = os.path.join(self.save_dir, "raw_frames.pkl")
        with open(raw_path, "wb") as f:
            pickle.dump(all_frames, f)

        self.log_info(
            f"Collection done: {len(successes)} success, {len(failures)} failure  "
            f"saved to: {raw_path}"
        )
        self.env.close()


# ======================================================================
# Save results (images + pickle files)
# ======================================================================


def save_results(frames: list[dict], save_dir: str) -> None:
    """Save frames as images + pickle files for train_reward_classifier."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_frames = [e for e in frames if e["label"] == "success"]
    failure_frames = [e for e in frames if e["label"] == "failure"]

    image_keys = sorted(frames[0]["images"].keys()) if frames else []

    # Individual images (per-camera subdirectories)
    for label_name, entries in [
        ("success", success_frames),
        ("failure", failure_frames),
    ]:
        for cam_key in image_keys:
            img_dir = os.path.join(save_dir, label_name, cam_key)
            os.makedirs(img_dir, exist_ok=True)
            for i, entry in enumerate(entries):
                cv2.imwrite(
                    os.path.join(img_dir, f"{stamp}_{i:05d}.png"),
                    entry["images"][cam_key],
                )

    # Separate pickle files (compatible with train_reward_classifier)
    for label_name, entries in [
        ("success", success_frames),
        ("failure", failure_frames),
    ]:
        if not entries:
            continue
        pkl = [
            {"observations": {"frames": {k: e["images"][k] for k in image_keys}}}
            for e in entries
        ]
        path = os.path.join(
            save_dir, f"{label_name}_{len(entries)}_{stamp}.pkl"
        )
        with open(path, "wb") as f:
            pickle.dump(pkl, f)

    camera_str = ", ".join(image_keys)
    print(f"\nSaved ({camera_str}):")
    print(f"  success: {len(success_frames)} groups -> {os.path.join(save_dir, 'success/')}")
    print(f"  failure: {len(failure_frames)} groups -> {os.path.join(save_dir, 'failure/')}")


# ======================================================================
# Main
# ======================================================================


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    save_dir = cfg.runner.logger.log_path

    # ── Teleoperation collection (Ray Worker) ────────────────────────
    print(f"[Classifier] save_dir={save_dir}")

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = ClassifierDataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()

    # ── Save images + pickle files ───────────────────────────────────
    raw_path = os.path.join(save_dir, "raw_frames.pkl")
    if not os.path.exists(raw_path):
        print(f"Raw data not found: {raw_path}")
        return

    with open(raw_path, "rb") as f:
        all_frames = pickle.load(f)

    save_results(all_frames, save_dir)

    n_success = sum(1 for f in all_frames if f["label"] == "success")
    n_failure = len(all_frames) - n_success
    image_keys = sorted(all_frames[0]["images"].keys()) if all_frames else ["wrist_1"]
    keys_str = " ".join(image_keys)
    print(
        f"\nCollection done: {len(all_frames)} frames total "
        f"({n_success} success + {n_failure} failure, "
        f"{len(image_keys)} cameras: {', '.join(image_keys)})"
    )
    print(
        f"\nNext steps:\n"
        f"  1. Review & filter:  python examples/embodiment/review_classifier_data.py"
        f" --log_dir {save_dir}\n"
        f"  2. Train classifier: python examples/embodiment/train_reward_classifier.py"
        f" --log_dir {save_dir} --image_keys {keys_str}"
    )


if __name__ == "__main__":
    main()
