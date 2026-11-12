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

"""Fixed G1 tensor sizes, state slices and episode-end codes.

One episode-end kind per env step. A clip end outranks a timeout and both
outrank a failure, so a fall on the last step of a clip or of the episode
counts as that truncation:

* ``EPISODE_CONT``: the episode continues.
* ``EPISODE_FAILURE``: the pelvis left the reference by more than
  ``FAILURE_HEIGHT_M`` in height or ``FAILURE_TILT`` in gravity z.
* ``EPISODE_TIMEOUT``: the training episode cap.
* ``EPISODE_CLIP_END``: the step taken from the clip's last frame.
"""

# The simulator state: root position, root quaternion (wxyz), joint angles,
# world root linear velocity, body-frame root angular velocity, joint rates.
STATE_DIM = 71
ACTION_DIM = 29
# Unitree G1 29-dof order used by mjlab. A reference clip must name these
# joints in this order. The loader does not permute a mismatched list.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
CONTROL_FPS = 50
NUM_BODIES = 14

ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
JOINT_POS = slice(7, 36)
ROOT_LIN_VEL = slice(36, 39)
ROOT_ANG_VEL = slice(39, 42)
JOINT_VEL = slice(42, 71)

# Foot friction (1), torso CoM (3) and joint encoder offsets (29), each
# centered and scaled by its randomization range.
SIM_CONTEXT_DIM = 33

# The goal reads ten reference frames, five control steps apart.
GOAL_FRAMES = 10
GOAL_STRIDE = 5
GOAL_CONTENT_DIM = GOAL_FRAMES * 2 * ACTION_DIM
ACTOR_GOAL_DIM = GOAL_CONTENT_DIM + GOAL_FRAMES * 6
WORLD_MODEL_GOAL_DIM = ACTOR_GOAL_DIM + GOAL_FRAMES * 3

EPISODE_CONT = 0
EPISODE_FAILURE = 1
EPISODE_TIMEOUT = 2
EPISODE_CLIP_END = 3

DT = 0.02
FAILURE_HEIGHT_M = 0.25
FAILURE_TILT = 0.8
# Positive tracking weights sum to 5; times DT this is the per-step ceiling.
MAX_TRACKING_REWARD = 0.1
