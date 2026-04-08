# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Shared-memory layout constants for the GeneSim RL pipeline.
#
# This file is an RLinf-local copy of
#   geniesim/rl/renderer/shm_layout.py
# so that the RLinf host process requires NO geniesim Python package.
#
# Both sides of the SHM boundary must agree on these constants.
# Update here if the container-side shm_layout.py changes.
#
# This file intentionally has NO external dependencies.

# ---------------------------------------------------------------------------
# Frame SHM  (name = shm_name, e.g. "geniesim_frames")
# Written by Isaac Sim renderer; read by GenieSimShmClient on the host.
# ---------------------------------------------------------------------------

NUM_CAMS: int = 2
SHM_HEADER_BYTES: int = 4


def shm_total_bytes(
    num_envs: int, height: int, width: int, num_cams: int = NUM_CAMS
) -> int:
    return SHM_HEADER_BYTES + num_envs * num_cams * height * width * 3


# ---------------------------------------------------------------------------
# Control SHM  (name = ctrl_shm_name(shm_name, env_id))
# One segment per parallel environment.
# Written by MuJoCo node; read by GenieSimVectorEnv.
#
# Layout:
#   [0:4]             uint32   state_counter
#   [4:8]             uint32   reset_flag     (RESET_*)
#   [8:8+S]           float32  states         (state_dim,)
#   [8+S:8+S+A]       float32  actions        (action_dim,)
#   [8+S+A:8+S+A+I]   float32  info_buf       (info_dim,) body poses
#   [end-8:end-4]     uint32   mujoco_phase   (MUJOCO_PHASE_*)
#   [end-4:end]       uint32   steps_per_step
# ---------------------------------------------------------------------------

CTRL_HEADER_BYTES: int = 8
CTRL_SYNC_BYTES: int = 8

BODY_POSE_DIM: int = 7

EE_STATE_DIM: int = 12

RESET_IDLE: int = 0
RESET_REQUESTED: int = 1
RESET_DONE: int = 2

MUJOCO_PHASE_WAIT: int = 0
MUJOCO_PHASE_GO: int = 1
MUJOCO_PHASE_DONE: int = 2


def ctrl_shm_name(shm_name: str, env_id: int) -> str:
    return f"{shm_name}_ctrl_{env_id}"


def ctrl_total_bytes(state_dim: int, action_dim: int, info_dim: int = 0) -> int:
    return CTRL_HEADER_BYTES + (state_dim + action_dim + info_dim) * 4 + CTRL_SYNC_BYTES


# ---------------------------------------------------------------------------
# Step SHM  (name = step_shm_name(shm_name))
# Single global segment for request-reply between RLinf and GenieSimVectorEnv.
#
# Layout:
#   [0:4]              uint32   step_phase   (STEP_PHASE_*)
#   [4:4+N*4]          float32  reset_mask   (num_envs,) 1.0=reset, 0.0=keep
#   [+0:+A_ALL]        float32  actions_all  (num_envs * action_dim)
#   --- step output (written by GenieSimVectorEnv) ---
#   [+0  :+N]          float32  rewards        (num_envs,)
#   [+N  :+2N]         float32  terminated     (num_envs,) 0.0/1.0
#   [+2N :+3N]         float32  truncated      (num_envs,) 0.0/1.0
#   [+3N :+4N]         float32  elapsed_steps  (num_envs,)
#   [+4N :+5N]         float32  episode_returns(num_envs,)
#   [+5N :+6N]         float32  success_once   (num_envs,) 0.0/1.0
#   [+6N :+6N+I_ALL]   float32  info_body_poses(num_envs * info_dim)
# ---------------------------------------------------------------------------

STEP_HEADER_BYTES: int = 4

STEP_OUTPUT_SCALARS: int = 6

STEP_PHASE_IDLE: int = 0
STEP_PHASE_STEP_REQUEST: int = 1
STEP_PHASE_STEP_DONE: int = 2
STEP_PHASE_RESET_REQUEST: int = 3
STEP_PHASE_RESET_DONE: int = 4
STEP_PHASE_CLOSE: int = 5


def step_shm_name(shm_name: str) -> str:
    return f"{shm_name}_step"


def step_total_bytes(num_envs: int, action_dim: int, info_dim: int = 0) -> int:
    reset_mask_bytes = num_envs * 4
    actions_bytes = num_envs * action_dim * 4
    output_bytes = num_envs * (STEP_OUTPUT_SCALARS + info_dim) * 4
    return STEP_HEADER_BYTES + reset_mask_bytes + actions_bytes + output_bytes
