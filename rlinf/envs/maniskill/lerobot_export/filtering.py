# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from typing import Tuple
import numpy as np


def filter_small_actions(
    actions: np.ndarray,
    pos_thresh: float = 0.01,
    rot_thresh: float = 0.06,
    check_gripper: bool = True,
) -> np.ndarray:
    """
    Return mask of frames to keep based on action deltas.

    Follows the same semantics as rlinf's filter_small_actions: keep when
    position or rotation movement exceeds threshold, or when gripper toggles.
    """
    actions = np.asarray(actions)
    num_frames = actions.shape[0]
    valid_mask = np.zeros(num_frames, dtype=bool)
    for i in range(num_frames):
        act = actions[i]
        delta_xyz = act[:3]
        delta_euler = act[3:6]
        gripper = act[6]
        pos_movement = np.linalg.norm(delta_xyz)
        rot_movement = np.linalg.norm(delta_euler)

        if pos_thresh is None and rot_thresh is None:
            is_valid = True
        elif pos_thresh is None:
            is_valid = rot_movement > rot_thresh
        elif rot_thresh is None:
            is_valid = pos_movement > pos_thresh
        else:
            is_valid = (pos_movement > pos_thresh) or (rot_movement > rot_thresh)

        if check_gripper and i > 0 and actions[i - 1][6] != gripper:
            is_valid = True

        valid_mask[i] = is_valid
    return valid_mask


def filter_with_tail_keep(mask: np.ndarray, keep_last_n: int) -> np.ndarray:
    """
    Ensure the last N frames are always kept regardless of the filter result.
    """
    if keep_last_n <= 0:
        return mask
    mask = mask.copy()
    if mask.size == 0:
        return mask
    tail = min(keep_last_n, mask.size)
    mask[-tail:] = True
    return mask


def apply_action_filter(
    actions: np.ndarray,
    keep_last_n: int = 15,
    pos_thresh: float = 0.01,
    rot_thresh: float = 0.06,
    check_gripper: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper to produce final keep-mask, excluding last N frames
    from filtering, as requested in LeRobotCreation.md.
    """
    base_mask = filter_small_actions(
        actions,
        pos_thresh=pos_thresh,
        rot_thresh=rot_thresh,
        check_gripper=check_gripper,
    )
    return filter_with_tail_keep(base_mask, keep_last_n=keep_last_n)


