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

"""Fixed SuperMimic tensor sizes and episode-end codes.

Episode-end kinds (mutually exclusive, priority failure > timeout > clip_end):

* ``EPISODE_CONT``: the episode continues.
* ``EPISODE_FAILURE``: pelvis height or tilt crossed the paper thresholds.
* ``EPISODE_TIMEOUT``: the 10 s training cap. Not a tracking failure.
* ``EPISODE_CLIP_END``: the reference clip ran out. Not a tracking failure.

How each surface uses the codes:

* Collect writes the true next state, then resets. The world-model BCE
  target is 1 only for ``FAILURE``. Timeout and clip-end are 0.
* An 8-step window may not cross any non-``CONT`` boundary.
* Imagination uses the learned termination head (soft) for survival
  weights. It does not apply the 10 s cap. Past the last clip frame the
  reference is held, not treated as termination.
* GAE on imagined trajectories bootstraps with ``V(s_H)`` and the soft
  survival weights. On real collect returns used for reset bins, timeout
  bootstraps with the critic and clip-end does not (CuboiLeo record).
* Evaluation has no 10 s cap. ``FAILURE`` is a fail; reaching clip-end
  is a finished rollout, not a fail.
"""

STATE_DIM = 71
RESIDUAL_DIM = 70
OBS_DIM = 93
GOAL_FRAME_DIM = 68
GOAL_FRAMES = 10
GOAL_REF_DIM = GOAL_FRAME_DIM * GOAL_FRAMES  # 680
ANCHOR_DIM = 9
XI_DIM = 33
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
HISTORY_LEN = 10
LATENT_DIM = 32
NUM_BODIES = 14
CRITIC_ROOT_ERR_DIM = 4

ROOT_POS = slice(0, 3)
ROOT_QUAT = slice(3, 7)
JOINT_POS = slice(7, 36)
ROOT_LIN_VEL = slice(36, 39)
ROOT_ANG_VEL = slice(39, 42)
JOINT_VEL = slice(42, 71)

# Residual: dpos(3) + drotvec(3) + dq(29) + dlin(3) + dang(3) + dqd(29)
RES_POS = slice(0, 3)
RES_ROT = slice(3, 6)
RES_JOINT = slice(6, 35)
RES_LIN = slice(35, 38)
RES_ANG = slice(38, 41)
RES_JOINT_VEL = slice(41, 70)

XI_FRICTION = slice(0, 1)
XI_ENCODER = slice(1, 30)
XI_COM = slice(30, 33)

EPISODE_CONT = 0
EPISODE_FAILURE = 1
EPISODE_TIMEOUT = 2
EPISODE_CLIP_END = 3

DT = 0.02
SUBSTEPS = 4
FAILURE_HEIGHT_M = 0.25
FAILURE_TILT = 0.8
MAX_TRACKING_REWARD = 0.1

# World-model state-loss scales (CuboiLeo, not the paper).
STATE_LOSS_SCALES = {
    "root_pos": 0.05,
    "root_rotvec": 0.1,
    "root_lin_vel": 0.5,
    "root_ang_vel": 1.0,
    "joint_pos": 0.1,
    "joint_vel": 2.0,
}

# Paper Table A.2 weights, applied before multiplying by dt.
# Positive terms sum to 5, so the per-step tracking cap after dt is 0.1.
# There is no separate joint-position or joint-velocity tracking term.
REWARD_WEIGHTS = {
    "body_pos": 1.0,
    "body_ori": 1.0,
    "body_lin_vel": 1.0,
    "body_ang_vel": 1.0,
    "root_pos": 0.5,
    "root_ori": 0.5,
    "joint_limit": -2.0,
    "action_rate": -0.1,
}

SENSOR_NOISE_STD = {
    "joint_pos": 0.01,
    "joint_vel": 0.5,
    "gyro": 0.2,
    "gravity": 0.05,
}


def critic_input_dim(goal_dim: int, critic_features: str) -> int:
    """Width of the motion-critic MLP input for one feature layout.

    ``root4`` is the heading-invariant state, the goal, ``xi``, and a 4-d
    root error. ``zita`` is zita-ch's layout: that invariant state, the joint
    error, 14 body-position errors, a 3-d root position error, a 6-d relative
    orientation, ``xi``, and the goal.
    """
    if critic_features == "root4":
        return STATE_DIM + goal_dim + XI_DIM + CRITIC_ROOT_ERR_DIM
    if critic_features == "zita":
        return STATE_DIM + ACTION_DIM + NUM_BODIES * 3 + 3 + 6 + XI_DIM + goal_dim
    raise ValueError(f"critic_features must be root4 or zita, got {critic_features!r}")
