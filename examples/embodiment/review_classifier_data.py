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

"""Review and filter classifier data collected by collect_classifier_data.py.

This standalone script loads ``raw_frames.pkl`` from a log directory,
displays each frame in an OpenCV window, and lets you keep or discard
individual frames.  You can switch between viewing only *success* frames,
only *failure* frames, or *all* frames.

Keyboard controls
-----------------
- **n / →**       — next frame
- **p / ←**       — previous frame
- **g**           — mark frame as *good* (keep)
- **b**           — mark frame as *bad* (discard)
- **1**           — show only success frames
- **2**           — show only failure frames
- **0**           — show all frames
- **s**           — save & overwrite (will ask for confirmation)
- **q / ESC**     — quit (will ask whether to save if there are changes)

Usage
-----
.. code-block:: bash

    python examples/embodiment/review_classifier_data.py \\
        --log_dir logs/<timestamp>-reward-classifier-<env_name>
"""

from __future__ import annotations

import argparse
import datetime
import glob
import os
import pickle
import shutil
import sys

import cv2
import numpy as np


# ======================================================================
# Confirmation dialog (OpenCV-based, no extra dependency)
# ======================================================================


def confirm_dialog(message: str, title: str = "Confirm") -> bool:
    """Show a simple OpenCV confirmation dialog.

    Displays *message* on a small canvas with ``[Y] Yes  [N] No``
    and blocks until the user presses **y** or **n**.

    Returns:
        ``True`` if the user pressed **y**, ``False`` otherwise.
    """
    width, height = 480, 160
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (50, 50, 50)

    # Draw title bar
    cv2.rectangle(canvas, (0, 0), (width, 36), (80, 80, 80), -1)
    cv2.putText(
        canvas, title, (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
    )

    # Draw message (word-wrap by splitting on spaces)
    y = 70
    for line in message.split("\n"):
        cv2.putText(
            canvas, line, (16, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
        )
        y += 24

    # Draw buttons
    cv2.putText(
        canvas, "[Y] Yes        [N] No", (100, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1,
    )

    win_name = title
    try:
        cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    except cv2.error:
        # Fallback: terminal confirmation
        resp = input(f"{message} (y/n): ").strip().lower()
        return resp in ("y", "yes")

    cv2.imshow(win_name, canvas)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord("y"):
            cv2.destroyWindow(win_name)
            return True
        if key in (ord("n"), 27):  # n or ESC
            cv2.destroyWindow(win_name)
            return False


# ======================================================================
# Save helpers
# ======================================================================


def save_frames(frames: list[dict], save_dir: str) -> None:
    """Save reviewed frames as images + pickle files.

    This **overwrites** existing ``success/`` and ``failure/`` dirs
    and any existing ``success_*.pkl`` / ``failure_*.pkl`` files,
    then writes fresh ones from *frames*.
    """
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    success_frames = [e for e in frames if e["label"] == "success"]
    failure_frames = [e for e in frames if e["label"] == "failure"]

    image_keys = sorted(frames[0]["images"].keys()) if frames else []

    # Remove old image directories
    for label_name in ("success", "failure"):
        img_dir = os.path.join(save_dir, label_name)
        if os.path.isdir(img_dir):
            shutil.rmtree(img_dir)

    # Remove old pickle files
    for old_pkl in glob.glob(os.path.join(save_dir, "success_*.pkl")):
        os.remove(old_pkl)
    for old_pkl in glob.glob(os.path.join(save_dir, "failure_*.pkl")):
        os.remove(old_pkl)

    # Write new images (per-camera subdirectories)
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

    # Write new pickle files (compatible with train_reward_classifier)
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

    # Update raw_frames.pkl to reflect reviewed state
    raw_path = os.path.join(save_dir, "raw_frames.pkl")
    with open(raw_path, "wb") as f:
        pickle.dump(frames, f)

    camera_str = ", ".join(image_keys)
    print(f"\nSaved ({camera_str}):")
    print(f"  success: {len(success_frames)} groups -> {os.path.join(save_dir, 'success/')}")
    print(f"  failure: {len(failure_frames)} groups -> {os.path.join(save_dir, 'failure/')}")
    print(f"  raw_frames.pkl synced")


# ======================================================================
# Review UI
# ======================================================================


class ReviewUI:
    """Interactive OpenCV review window with filter/save support."""

    FILTER_ALL = "all"
    FILTER_SUCCESS = "success"
    FILTER_FAILURE = "failure"

    def __init__(self, frames: list[dict], save_dir: str):
        self.all_frames = frames  # full list — never mutated structurally
        self.save_dir = save_dir
        # True = keep, False = discard, None = not yet reviewed
        self.decisions: list[bool | None] = [None] * len(frames)
        self.filter_mode: str = self.FILTER_ALL
        self.dirty = False  # True when any decision differs from initial

        self._rebuild_view()
        self.view_idx = 0  # index into self.view

    # -- filtering helpers ------------------------------------------------

    def _rebuild_view(self) -> None:
        """Rebuild the filtered index list based on current filter_mode."""
        if self.filter_mode == self.FILTER_ALL:
            self.view = list(range(len(self.all_frames)))
        else:
            self.view = [
                i for i, f in enumerate(self.all_frames)
                if f["label"] == self.filter_mode
            ]

    def _clamp_view_idx(self) -> None:
        if not self.view:
            self.view_idx = 0
        else:
            self.view_idx = max(0, min(self.view_idx, len(self.view) - 1))

    # -- rendering --------------------------------------------------------

    def render(self) -> np.ndarray:
        """Render current frame + status bar onto a canvas."""
        if not self.view:
            canvas = np.zeros((120, 500, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                f"No {self.filter_mode} frames.",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1,
            )
            return canvas

        real_idx = self.view[self.view_idx]
        frame = self.all_frames[real_idx]
        images = frame["images"]

        # Build side-by-side display of all cameras
        display_parts = []
        for key in sorted(images.keys()):
            img = images[key].copy()  # BGR uint8
            h, w = img.shape[:2]
            scale = max(1, 360 // min(h, w))
            resized = cv2.resize(
                img, (w * scale, h * scale),
                interpolation=cv2.INTER_NEAREST,
            )
            # Camera label above image
            lbl_h = 22
            part = np.zeros(
                (resized.shape[0] + lbl_h, resized.shape[1], 3),
                dtype=np.uint8,
            )
            part[lbl_h:] = resized
            cv2.putText(
                part, key, (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 220, 255), 1,
            )
            display_parts.append(part)

        display = (
            np.concatenate(display_parts, axis=1)
            if display_parts
            else np.zeros((100, 300, 3), dtype=np.uint8)
        )

        bar_h = 60
        dh, dw = display.shape[:2]
        canvas = np.zeros((dh + bar_h, dw, 3), dtype=np.uint8)
        canvas[bar_h:] = display

        # Decision tag
        dec = self.decisions[real_idx]
        if dec is True:
            tag, color = "KEEP", (0, 200, 0)
        elif dec is False:
            tag, color = "DISCARD", (0, 0, 200)
        else:
            tag, color = "---", (180, 180, 180)

        # Stats
        total = len(self.all_frames)
        total_reviewed = sum(1 for d in self.decisions if d is not None)
        total_kept = sum(1 for d in self.decisions if d is not False)
        total_discarded = sum(1 for d in self.decisions if d is False)
        label = frame["label"]

        # Line 1: position + label + decision
        line1 = (
            f"[{self.view_idx + 1}/{len(self.view)}] "
            f"(global {real_idx + 1}/{total})  "
            f"label: {label}  |  mark: {tag}"
        )
        cv2.putText(
            canvas, line1, (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1,
        )

        # Line 2: summary stats + filter mode
        filter_display = {
            self.FILTER_ALL: "ALL",
            self.FILTER_SUCCESS: "SUCCESS only",
            self.FILTER_FAILURE: "FAILURE only",
        }[self.filter_mode]
        line2 = (
            f"filter: {filter_display}  |  "
            f"reviewed: {total_reviewed}/{total}  "
            f"keep: {total_kept}  discard: {total_discarded}"
        )
        cv2.putText(
            canvas, line2, (10, 46),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1,
        )

        return canvas

    # -- main loop --------------------------------------------------------

    def run(self) -> None:
        """Enter the interactive review loop."""
        win = "Review  |  n/p:nav  g:keep  b:discard  1/2/0:filter  s:save  q:quit"
        # Avoid X11 shared-memory issues in Docker.
        os.environ.setdefault("QT_X11_NO_MITSHM", "1")

        try:
            cv2.namedWindow(win, cv2.WINDOW_NORMAL)
            n_cams = len(self.all_frames[0]["images"]) if self.all_frames else 1
            cv2.resizeWindow(win, max(720, 400 * n_cams), 560)
        except cv2.error as e:
            print(f"Cannot create window (X11/display error): {e}")
            print("Please ensure an X11 display is available (export DISPLAY=...).")
            return

        cv2.imshow(win, self.render())

        while True:
            key = cv2.waitKey(0) & 0xFF

            if key in (ord("n"), 83):  # n or Right arrow
                self.view_idx = min(self.view_idx + 1, len(self.view) - 1)

            elif key in (ord("p"), 81):  # p or Left arrow
                self.view_idx = max(self.view_idx - 1, 0)

            elif key == ord("g"):  # keep
                if self.view:
                    self.decisions[self.view[self.view_idx]] = True
                    self.dirty = True
                    self.view_idx = min(self.view_idx + 1, len(self.view) - 1)

            elif key == ord("b"):  # discard
                if self.view:
                    self.decisions[self.view[self.view_idx]] = False
                    self.dirty = True
                    self.view_idx = min(self.view_idx + 1, len(self.view) - 1)

            elif key == ord("1"):  # success only
                self.filter_mode = self.FILTER_SUCCESS
                self._rebuild_view()
                self.view_idx = 0
                self._clamp_view_idx()

            elif key == ord("2"):  # failure only
                self.filter_mode = self.FILTER_FAILURE
                self._rebuild_view()
                self.view_idx = 0
                self._clamp_view_idx()

            elif key == ord("0"):  # all
                self.filter_mode = self.FILTER_ALL
                self._rebuild_view()
                self.view_idx = 0
                self._clamp_view_idx()

            elif key == ord("s"):  # save
                self._try_save()

            elif key in (ord("q"), 27):  # quit
                if self.dirty:
                    if confirm_dialog(
                        "You have unsaved changes.\nSave before quitting?",
                        "Save before quit?",
                    ):
                        self._try_save()
                break

            cv2.imshow(win, self.render())

        cv2.destroyAllWindows()

    # -- save logic -------------------------------------------------------

    def _try_save(self) -> None:
        """Prompt for confirmation, then save kept frames."""
        total_discarded = sum(1 for d in self.decisions if d is False)
        total_kept = sum(1 for d in self.decisions if d is not False)

        msg = (
            f"Keep {total_kept}, discard {total_discarded}.\n"
            f"Overwrite existing data?"
        )
        if confirm_dialog(msg, "Confirm overwrite"):
            kept = [
                f for f, d in zip(self.all_frames, self.decisions)
                if d is not False
            ]
            save_frames(kept, self.save_dir)
            self.dirty = False
            print("Saved.")
        else:
            print("Save cancelled.")


# ======================================================================
# CLI
# ======================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Review and filter classifier data (success/failure frames).",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help=(
            "Path to the log directory produced by "
            "collect_classifier_data.sh, e.g. "
            "logs/20260305-10:00:00-reward-classifier-dex_pnp"
        ),
    )
    args = parser.parse_args()

    log_dir = args.log_dir
    raw_path = os.path.join(log_dir, "raw_frames.pkl")

    if not os.path.exists(raw_path):
        print(f"Error: {raw_path} not found")
        print("Please verify --log_dir points to the directory created by collect_classifier_data.")
        sys.exit(1)

    with open(raw_path, "rb") as f:
        all_frames = pickle.load(f)

    # Backward compat: convert old single-image format to multi-camera
    if all_frames and "image" in all_frames[0] and "images" not in all_frames[0]:
        for fr in all_frames:
            fr["images"] = {"wrist_1": fr.pop("image")}

    n_success = sum(1 for fr in all_frames if fr["label"] == "success")
    n_failure = len(all_frames) - n_success
    image_keys = sorted(all_frames[0]["images"].keys()) if all_frames else []
    print(
        f"Loaded {len(all_frames)} frames "
        f"({n_success} success + {n_failure} failure, "
        f"{len(image_keys)} cameras: {', '.join(image_keys)}) "
        f"from {raw_path}"
    )
    print()
    print("Controls:")
    print("  n / ->     next frame          p / <-     previous frame")
    print("  g         mark as keep         b         mark as discard")
    print("  1         success only         2         failure only         0    show all")
    print("  s         save & overwrite (confirmation dialog)")
    print("  q / ESC   quit (prompts to save if changes exist)")
    print()

    ui = ReviewUI(all_frames, log_dir)
    ui.run()


if __name__ == "__main__":
    main()
