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

"""Collect per-frame success/fail labels in a single episode and save as .pt.

Workflow (end-to-end, no intermediate pkl):
  1. Run one episode using the RealWorld env with SpaceMouse/keyboard.
  2. Label each step via keyboard: 'c' = success frame, 'a' = fail frame.
  3. Stop when both configured thresholds are reached (or max_steps exhausted).
  4. Apply fail:success ratio sampling and train/val split.
  5. Save train.pt / val.pt directly (no .pkl intermediate).

Usage:
    bash examples/reward/realworld_collect_process_dataset.sh
    # or with explicit config:
    bash examples/reward/realworld_collect_process_dataset.sh realworld_collect_dataset
"""

import json
import os
import random

import hydra
import numpy as np
import torch

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker
from rlinf.utils.logging import get_logger

logger = get_logger()


class FrameCollector(Worker):
    """Collects per-frame success/fail labels within a single episode.

    Uses keyboard keys 'c' (success) and 'a' (fail) to label each step.
    Collection stops when both configured thresholds are reached.
    On exit, frames are ratio-sampled, split into train/val, and saved as .pt.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.target_success = cfg.runner.num_success_frames
        self.target_fail = cfg.runner.num_fail_frames
        self.val_split = cfg.runner.get("val_split", 0.2)
        self.fail_success_ratio = cfg.runner.get("fail_success_ratio", 2.0)
        self.random_seed = cfg.runner.get("random_seed", 42)

        self.success_frames: list[torch.Tensor] = []
        self.fail_frames: list[torch.Tensor] = []

        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        self.listener = KeyboardListener()
        self.step_count = 0

    def _nhwc_to_chw(self, img: torch.Tensor) -> torch.Tensor:
        """Convert NHWC (H, W, C) image to CHW (C, H, W)."""
        if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
            return img.permute(2, 0, 1)
        return img

    def _extract_main_image(self, obs: dict) -> torch.Tensor | None:
        """Extract and normalize the main camera image from observation dict.

        Prioritizes 'main_images', falling back to 'images'.  Both inputs
        may arrive as [1, H, W, C] (batch dim = 1); this is squeezed to
        [H, W, C] so the resulting .pt stores NHWC images without a leading
        batch dimension, consistent with RewardBinaryDataset expectations.
        """
        obs = dict(obs)
        obs.pop("task_descriptions", None)

        img: torch.Tensor | None = None
        if "main_images" in obs:
            img = obs["main_images"]
        elif "images" in obs:
            img = self._nhwc_to_chw(obs["images"])

        if img is None:
            return None

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)
        img = img.cpu()
        if img.ndim == 4 and img.shape[0] == 1:
            img = img.squeeze(0)
        return img

    def _print_progress(self):
        s_ok = len(self.success_frames)
        f_ok = len(self.fail_frames)
        s_bar = "#" * s_ok + "-" * max(0, self.target_success - s_ok)
        f_bar = "#" * f_ok + "-" * max(0, self.target_fail - f_ok)
        print(
            f"\r  success: {s_ok}/{self.target_success} [{s_bar}]  "
            f"fail: {f_ok}/{self.target_fail} [{f_bar}]",
            end="",
            flush=True,
        )

    def _check_key(self):
        key = self.listener.get_key()
        if key == "c":
            return "success"
        elif key == "a":
            return "fail"
        return None

    def run(self):
        self._extract_main_image(self.env.reset()[0])
        max_steps = self.cfg.env.eval.max_episode_steps

        logger.info(
            f"Starting frame collection (single episode): "
            f"target {self.target_success} success frames, "
            f"{self.target_fail} fail frames | 'c'=success 'a'=fail"
        )

        while True:
            self._print_progress()

            s_ok = len(self.success_frames)
            f_ok = len(self.fail_frames)
            if s_ok >= self.target_success and f_ok >= self.target_fail:
                logger.info("Target frame counts reached, ending collection.")
                break

            if self.cfg.env.eval.get("no_gripper", True):
                action = np.zeros((1, 6))
            else:
                action = np.zeros((1, 7))
            next_obs, reward, done, _, info = self.env.step(action)

            if "intervene_action" in info:
                action = info["intervene_action"]

            img = self._extract_main_image(next_obs)
            if img is not None:
                label = self._check_key()
                if (
                    label == "success"
                    and len(self.success_frames) < self.target_success
                ):
                    self.success_frames.append(img.clone())
                elif label == "fail" and len(self.fail_frames) < self.target_fail:
                    self.fail_frames.append(img.clone())

            self.step_count += 1

            if self.step_count >= max_steps:
                logger.warning(
                    f"Max steps {max_steps} reached, exiting early. "
                    f"success {len(self.success_frames)}/{self.target_success}, "
                    f"fail {len(self.fail_frames)}/{self.target_fail}"
                )
                break

        print()
        self._save_pt()
        self.env.close()

    def _save_pt(self):
        out_dir = self.cfg.runner.logger.log_path
        os.makedirs(out_dir, exist_ok=True)

        success_frames = self.success_frames
        fail_frames = self.fail_frames

        total_frames = len(success_frames) + len(fail_frames)
        total_success = len(success_frames)

        logger.info(
            f"Loaded 1 episode, {total_frames} frames (all): "
            f"{total_success} success, {total_frames - total_success} fail"
        )

        rng = random.Random(self.random_seed)

        pairs = [(f, 1) for f in success_frames] + [(f, 0) for f in fail_frames]
        rng.shuffle(pairs)
        all_images, all_labels = zip(*pairs) if pairs else ([], [])

        n = len(all_images)
        n_val = max(1, int(n * self.val_split))
        val_images, val_labels = list(all_images[:n_val]), list(all_labels[:n_val])
        train_images, train_labels = (
            list(all_images[n_val:]),
            list(all_labels[n_val:]),
        )

        num_train_success = sum(train_labels)
        num_val_success = sum(val_labels)

        logger.info(
            f"Episode split: {1 if len(train_images) > 0 else 0} train eps, "
            f"{1 if len(val_images) > 0 else 0} val eps"
        )

        logger.info("Processing train set:")
        logger.info(
            f"  Raw: {num_train_success} success, "
            f"{len(train_labels) - num_train_success} fail"
        )

        if self.fail_success_ratio > 0 and num_train_success > 0:
            target_train_fail = int(num_train_success * self.fail_success_ratio)
            fail_indices = [i for i, l in enumerate(train_labels) if l == 0]
            rng.shuffle(fail_indices)
            fail_keep = set(fail_indices[:target_train_fail])
            train_keep = [i for i, l in enumerate(train_labels) if l == 1]
            train_keep += list(fail_keep)
            rng.shuffle(train_keep)
            train_images = [train_images[i] for i in train_keep]
            train_labels = [train_labels[i] for i in train_keep]
            num_train_success = sum(train_labels)
            logger.info(
                f"  After {self.fail_success_ratio}:1 ratio: {num_train_success} success, "
                f"{len(train_labels) - num_train_success} fail"
            )

        logger.info("Processing val set:")
        logger.info(
            f"  Raw: {num_val_success} success, "
            f"{len(val_labels) - num_val_success} fail"
        )

        metadata = {
            "num_success_frames": total_success,
            "num_fail_frames": total_frames - total_success,
            "total_frames": total_frames,
            "val_split": self.val_split,
            "fail_success_ratio": self.fail_success_ratio,
            "random_seed": self.random_seed,
            "num_train_samples": len(train_images),
            "num_val_samples": len(val_images),
        }

        train_path = f"{out_dir}/train.pt"
        val_path = f"{out_dir}/val.pt"

        RewardDatasetPayload(
            images=train_images, labels=train_labels, metadata=metadata
        ).save(train_path)
        RewardDatasetPayload(
            images=val_images, labels=val_labels, metadata=metadata
        ).save(val_path)

        logger.info(
            f"Episode-based split complete - "
            f"Train: {len(train_images)} frames "
            f"({num_train_success} success), "
            f"Val: {len(val_images)} frames ({sum(val_labels)} success)"
        )

        logger.info("=" * 60)
        logger.info("Reward dataset preprocessing complete")
        logger.info(
            f"Train split: {train_path} ({metadata['num_train_samples']} samples)"
        )
        logger.info(f"Val split:   {val_path} ({metadata['num_val_samples']} samples)")
        logger.info("Metadata:")
        logger.info(json.dumps(metadata, indent=2))
        logger.info("=" * 60)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_collect_dataset",
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = FrameCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
