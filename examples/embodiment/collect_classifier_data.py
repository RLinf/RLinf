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

After the target number of success frames is reached, a CV2 review
window is shown so the operator can manually refine labels.

Review controls
---------------
- **n** — next frame
- **p** — previous frame
- **g** — mark frame as *good* (keep)
- **b** — mark frame as *bad* (discard)
- **q / ESC** — finish review and save

Usage
-----
.. code-block:: bash

    bash examples/embodiment/collect_classifier_data.sh [config_name] [env_name]

Data is saved to ``logs/<timestamp>-reward-classifier-<env_name>/``.
The same directory can be passed to ``train_reward_classifier.py``.
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
# Phase 1 — Teleoperation data collection (runs as Ray Worker)
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

    def _extract_frame(self, obs) -> np.ndarray | None:
        """Extract camera frame from wrapped observation.

        ``obs["main_images"]`` is a torch tensor ``(1, H, W, 3)`` BGR
        uint8, produced by ``RealWorldEnv._wrap_obs``.
        """
        main = obs.get("main_images", None)
        if main is None:
            return None
        if isinstance(main, torch.Tensor):
            main = main.cpu().numpy()
        if main.ndim == 4:
            main = main[0]
        return main.copy()

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
            f"目标: 收集 {self.successes_needed} 帧成功样本  "
            f"每 episode {max_steps} 步后自动复位"
        )

        obs, _ = self.env.reset()

        while len(successes) < self.successes_needed:
            episode_count += 1
            self.log_info(
                f"\n{'#' * 50}\n"
                f"  Episode {episode_count}  "
                f"成功: {len(successes)}/{self.successes_needed}  "
                f"失败: {len(failures)}\n"
                f"  >>> 开始遥操作 — 右键标记成功帧 <<<\n"
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

                frame = self._extract_frame(next_obs)
                right_pressed = self._get_right_button(info)

                if frame is not None:
                    entry = {
                        "image": frame,
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
                f"Episode {episode_count} 完成  "
                f"成功: {len(successes)}  失败: {len(failures)}"
            )

            # Check if we have enough before resetting
            if len(successes) >= self.successes_needed:
                break

            obs, _ = self.env.reset()

        # Save raw frames for review in main process
        all_frames = successes + failures
        os.makedirs(self.save_dir, exist_ok=True)
        raw_path = os.path.join(self.save_dir, "raw_frames.pkl")
        with open(raw_path, "wb") as f:
            pickle.dump(all_frames, f)

        self.log_info(
            f"采集完成: {len(successes)} 成功, {len(failures)} 失败  "
            f"保存至: {raw_path}"
        )
        self.env.close()


# ======================================================================
# Phase 2 — Review UI (runs in main process, needs display)
# ======================================================================


def review_frames(frames: list[dict]) -> tuple[list[dict], list[dict]]:
    """Show an OpenCV window for manual frame review.

    Returns:
        ``(kept, discarded)``
    """
    if not frames:
        print("没有帧需要审核。")
        return [], []

    total = len(frames)
    decisions: list[bool | None] = [None] * total
    idx = 0

    WIN = "Review: g=keep  b=discard  n=next  p=prev  q=done"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 640, 520)

    def render(index: int) -> None:
        img = frames[index]["image"].copy()  # BGR uint8
        h, w = img.shape[:2]
        scale = max(1, 480 // min(h, w))
        display = cv2.resize(
            img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST
        )

        bar_h = 40
        canvas = np.zeros(
            (display.shape[0] + bar_h, display.shape[1], 3), dtype=np.uint8
        )
        canvas[bar_h:] = display

        dec = decisions[index]
        if dec is True:
            tag, color = "KEEP", (0, 200, 0)
        elif dec is False:
            tag, color = "DISCARD", (0, 0, 200)
        else:
            tag, color = "---", (180, 180, 180)

        reviewed = sum(1 for d in decisions if d is not None)
        label = frames[index]["label"]
        txt = (
            f"[{index + 1}/{total}]  "
            f"auto: {label}  |  mark: {tag}  |  "
            f"reviewed: {reviewed}/{total}"
        )
        cv2.putText(canvas, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.imshow(WIN, canvas)

    render(idx)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("n"):
            idx = min(idx + 1, total - 1)
        elif key == ord("p"):
            idx = max(idx - 1, 0)
        elif key == ord("g"):
            decisions[idx] = True
            idx = min(idx + 1, total - 1)
        elif key == ord("b"):
            decisions[idx] = False
            idx = min(idx + 1, total - 1)
        elif key in (ord("q"), 27):
            break
        render(idx)

    cv2.destroyAllWindows()

    kept = [f for f, d in zip(frames, decisions) if d is not False]
    discarded = [f for f, d in zip(frames, decisions) if d is False]
    n_unlabeled = sum(1 for d in decisions if d is None)
    print(
        f"审核结果: 保留 {len(kept)} (含 {n_unlabeled} 未标记), "
        f"丢弃 {len(discarded)}"
    )
    return kept, discarded


# ======================================================================
# Phase 3 — Save results
# ======================================================================


def save_results(kept: list[dict], save_dir: str) -> None:
    """Save reviewed frames as images + pickle files."""
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_frames = [e for e in kept if e["label"] == "success"]
    failure_frames = [e for e in kept if e["label"] == "failure"]

    # Individual images
    for label_name, entries in [
        ("success", success_frames),
        ("failure", failure_frames),
    ]:
        img_dir = os.path.join(save_dir, label_name)
        os.makedirs(img_dir, exist_ok=True)
        for i, entry in enumerate(entries):
            cv2.imwrite(
                os.path.join(img_dir, f"{stamp}_{i:05d}.png"),
                entry["image"],
            )

    # Separate pickle files (compatible with train_reward_classifier)
    for label_name, entries in [
        ("success", success_frames),
        ("failure", failure_frames),
    ]:
        if not entries:
            continue
        pkl = [
            {"observations": {"frames": {"wrist_1": e["image"]}}}
            for e in entries
        ]
        path = os.path.join(
            save_dir, f"{label_name}_{len(entries)}_{stamp}.pkl"
        )
        with open(path, "wb") as f:
            pickle.dump(pkl, f)

    print(f"\n保存完成:")
    print(f"  成功: {len(success_frames)} 张 → {os.path.join(save_dir, 'success/')}")
    print(f"  失败: {len(failure_frames)} 张 → {os.path.join(save_dir, 'failure/')}")


# ======================================================================
# Main
# ======================================================================


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    save_dir = cfg.runner.logger.log_path

    # ── Phase 1: Teleoperation collection (Ray Worker) ───────────────
    print(f"[Classifier] save_dir={save_dir}")

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = ClassifierDataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()

    # ── Phase 2: Review UI (main process) ────────────────────────────
    raw_path = os.path.join(save_dir, "raw_frames.pkl")
    if not os.path.exists(raw_path):
        print(f"未找到采集数据: {raw_path}")
        return

    with open(raw_path, "rb") as f:
        all_frames = pickle.load(f)

    n_success = sum(1 for f in all_frames if f["label"] == "success")
    n_failure = len(all_frames) - n_success
    print(
        f"\n进入审核阶段: 共 {len(all_frames)} 帧 "
        f"({n_success} 成功 + {n_failure} 失败)\n"
        "  n = 下一帧   p = 上一帧\n"
        "  g = 保留     b = 丢弃\n"
        "  q / ESC = 完成审核并保存\n"
    )
    kept, _ = review_frames(all_frames)

    # ── Phase 3: Save ────────────────────────────────────────────────
    save_results(kept, save_dir)
    print(
        f"\n全部完成。后续训练命令:\n"
        f"  python examples/embodiment/train_reward_classifier.py"
        f" --log_dir {save_dir}"
    )


if __name__ == "__main__":
    main()
