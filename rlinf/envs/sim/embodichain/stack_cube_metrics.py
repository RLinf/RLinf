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

"""Batched Stack Cube physical measurements."""

import torch


def stack_cube_metrics(
    positions: torch.Tensor,
    fingers: torch.Tensor,
    linear_speeds: torch.Tensor,
    angular_speeds: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Measure IsaacLab completion and stricter physical stability per scene.

    Args:
        positions: Blue, red, green centers, shaped ``[batch, 3, 3]``, in
            meters in each scene's frame with the table surface at z=0.
        fingers: Two independently actuated finger positions, ``[batch, 2]``.
        linear_speeds: Object speed magnitudes, ``[batch, 3]``, in m/s.
        angular_speeds: Object speed magnitudes, ``[batch, 3]``, in rad/s.

    Returns:
        Batched physical measurements. ``baseline_success`` reproduces the
        RLinf IsaacLab fork's ``cubes_stacked`` predicate, including its
        one-sided height bound. ``stable_frame`` additionally requires the
        intended layer heights and low object speeds; consecutive-frame
        counting belongs to the episode owner. ``failure`` marks a dropped
        object using the same -0.05 m threshold as the reference task.
    """
    batch = positions.shape[0]
    if positions.shape != (batch, 3, 3) or fingers.shape != (batch, 2):
        raise ValueError("Expected positions [batch, 3, 3] and fingers [batch, 2].")
    if linear_speeds.shape != (batch, 3) or angular_speeds.shape != (batch, 3):
        raise ValueError("Expected object speed magnitudes [batch, 3].")
    offsets = positions[:, 1:] - positions[:, :-1]
    xy_errors = offsets[..., :2].norm(dim=-1)
    heights = offsets[..., 2]
    height_errors = (heights - 0.0468).abs()
    layers = (xy_errors < 0.04) & (heights > 0) & (heights - 0.0468 < 0.005)
    released = torch.isclose(
        fingers, fingers.new_tensor(0.04), atol=0.0001, rtol=0.0001
    ).all(dim=-1)
    baseline_success = layers.all(dim=-1) & released
    stable_frame = (
        baseline_success
        & (height_errors < 0.005).all(dim=-1)
        & (linear_speeds <= 0.02).all(dim=-1)
        & (angular_speeds <= 0.5).all(dim=-1)
    )
    return {
        "positions": positions,
        "xy_errors": xy_errors,
        "height_errors": height_errors,
        "layers": layers,
        "released": released,
        "linear_speeds": linear_speeds,
        "angular_speeds": angular_speeds,
        "baseline_success": baseline_success,
        "stable_frame": stable_frame,
        "failure": (positions[..., 2] < -0.05).any(dim=-1),
    }
