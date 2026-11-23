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

"""Dream API policy for embodied evaluation."""

from typing import Any

import numpy as np
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy


class DreamPolicy(BasePolicy):
    """Adapt RLinf observation history to the official Dream SDK."""

    def __init__(self, sdk: Any, cfg: DictConfig) -> None:
        self.sdk = sdk
        sdk.policy(
            {
                "checkpoint": cfg.checkpoint,
                "robot": "other_embodiment",
                "strict": True,
                "_record": False,
                "timeout_ms": cfg.dream.timeout_ms,
                "action_hz": cfg.dream.action_hz,
                "obs_fps": 5,
                "obs_window": 9,
                "ik": {
                    "arm": cfg.dream.arm,
                    "head": "eef",
                    "solve": "traj",
                },
                "embodiment": {
                    "state": {
                        "left_arm": [0, 6],
                        "left_grip": [6, 7],
                        "left_gripper": [7, 8],
                        "left_eef_xyz": [8, 11],
                        "left_eef_rot6": [11, 17],
                        "right_arm": [17, 23],
                        "right_grip": [23, 24],
                        "right_gripper": [24, 25],
                        "right_eef_xyz": [25, 28],
                        "right_eef_rot6": [28, 34],
                    },
                    "action": {
                        "left_eef_xyz": [0, 3],
                        "left_eef_rot6": [3, 9],
                        "left_grip": [9, 10],
                        "right_eef_xyz": [10, 13],
                        "right_eef_rot6": [13, 19],
                        "right_grip": [19, 20],
                    },
                    "video": {
                        role: f"observation.images.{role}"
                        for role in ("overhead", "wrist_l", "wrist_r")
                    },
                },
            }
        )

    def default_forward(self, **kwargs: Any) -> None:
        """Reject training forwards for the evaluation-only API policy."""
        raise NotImplementedError("Dream API supports evaluation only.")

    def predict_action_batch(
        self, env_obs: dict[str, Any], **kwargs: Any
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Convert native policy input to SDK observations and joint targets."""
        sample = env_obs["history"][0]
        states = np.asarray(sample["states"], dtype=np.float32)
        if bool(sample["finished"][0]):
            return np.repeat(states[:, -1:], 48, axis=1), {}
        active_arms = sample["active_arms"][0]
        frames = sample["frames"][0]
        timestamps = np.asarray(sample["timestamps"])
        prompt = sample["task_descriptions"][0]
        for i, frame in enumerate(frames):
            q = states[0, i]
            self.sdk.set_observation(
                frame,
                {
                    "prompt": prompt,
                    "state.left_arm": q[:6].tolist(),
                    "state.left_grip": q[6:7].tolist(),
                    "state.right_arm": q[7:13].tolist(),
                    "state.right_grip": q[13:14].tolist(),
                },
                timestamp=float(timestamps[0, i]),
            )
        commanded = sample["commanded"]
        last = None
        if commanded is not None:
            q = np.asarray(commanded)[0]
            last = {"q_l_chunk": q[:7], "q_r_chunk": q[7:]}
        reply = self.sdk.get_action(commanded=last)
        arms = []
        for i, (side, key) in enumerate(
            (("left", "q_l_chunk"), ("right", "q_r_chunk"))
        ):
            arm = np.asarray(reply[key], dtype=np.float32)
            if arm.shape != (48, 7):
                raise ValueError("Dream must return two [48,7] joint chunks.")
            if not np.isfinite(arm).all():
                if side in active_arms:
                    raise ValueError(
                        f"Dream returned non-finite targets for the active {side} arm."
                    )
                arm = np.repeat(states[0, -1, i * 7 : i * 7 + 7][None], 48, axis=0)
            arms.append(arm)
        actions = np.concatenate(arms, axis=-1)[None]
        actions[..., [6, 13]] = np.clip(actions[..., [6, 13]], 0, 1)
        return actions, {}

    def close(self) -> None:
        """Release the SDK session."""
        self.sdk.close()


def get_model(cfg: DictConfig) -> DreamPolicy:
    """Build a Dream policy with the SDK in the rollout interpreter."""
    from dream import DreamClient

    return DreamPolicy(DreamClient(), cfg)
