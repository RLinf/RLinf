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

"""Dream API inference over the embodied evaluation channels."""

from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.utils.obs_compression import infer_obs_batch_size


class DreamActionClient:
    """Adapt RLinf observation history to the official Dream SDK."""

    def __init__(self, sdk: Any, cfg: DictConfig) -> None:
        self.sdk = sdk
        self.calls = 0
        self.inactive_arm_holds = 0
        self.episode_id = None
        self.closed = False
        try:
            sdk.policy(
                {
                    "checkpoint": cfg.model.checkpoint,
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
        except Exception:
            self.close()
            raise

    def predict(self, sample: dict[str, Any]) -> np.ndarray:
        """Return absolute joint targets; finished episodes consume no API call."""
        states = np.asarray(sample["states"], dtype=np.float32)
        if bool(sample["finished"][0]):
            return np.repeat(states[:, -1:], 48, axis=1)
        episode_id = int(sample["episode_id"][0])
        if episode_id != self.episode_id:
            self.sdk.reset_episode()
            self.episode_id = episode_id
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
                    "state.left_gripper": q[6:7].tolist(),
                    "state.right_arm": q[7:13].tolist(),
                    "state.right_grip": q[13:14].tolist(),
                    "state.right_gripper": q[13:14].tolist(),
                },
                timestamp=float(timestamps[0, i]),
            )
        self.calls += 1
        commanded = sample.get("commanded")
        last = (
            None
            if commanded is None
            else {
                "q_l_chunk": np.asarray(commanded)[0, :7],
                "q_r_chunk": np.asarray(commanded)[0, 7:],
            }
        )
        reply = self.sdk.get_action(commanded=last)
        arms = [
            np.asarray(reply[key], dtype=np.float32)
            for key in ("q_l_chunk", "q_r_chunk")
        ]
        active_arms = sample.get("active_arms", [["left", "right"]])[0]
        for i, (side, arm) in enumerate(zip(("left", "right"), arms, strict=True)):
            if arm.shape != (48, 7):
                raise ValueError("Dream must return two [48,7] joint chunks.")
            if not np.isfinite(arm).all():
                if side in active_arms:
                    raise ValueError(
                        f"Dream returned non-finite targets for the active {side} arm."
                    )
                arms[i] = np.repeat(states[0, -1, i * 7 : i * 7 + 7][None], 48, axis=0)
                self.inactive_arm_holds += 1
        actions = np.concatenate(arms, axis=-1)[None]
        actions[..., [6, 13]] = np.clip(actions[..., [6, 13]], 0, 1)
        return actions

    def close(self) -> None:
        """Release the SDK session once."""
        if not self.closed:
            self.closed = True
            self.sdk.close()


class DreamRolloutWorker(Worker):
    """Use the existing RLinf evaluation protocol for Dream inference."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        if (
            self._world_size != 1
            or cfg.env.eval.total_num_envs != 1
            or cfg.rollout.pipeline_stage_num != 1
            or cfg.env.eval.auto_reset
            or not cfg.runner.only_eval
            or (cfg.rollout.model.num_action_chunks, cfg.rollout.model.action_dim)
            != (48, 14)
        ):
            raise ValueError(
                "Dream requires one evaluation environment, auto_reset=false and 48x14 action chunks."
            )
        self.cfg = cfg
        self.client = None

    def init_worker(self) -> None:
        """Initialize the optional Dream SDK in the rollout process."""
        from dream import DreamClient

        self.client = DreamActionClient(DreamClient(), self.cfg.rollout)

    @staticmethod
    def merge_observations(batches: list[dict[str, Any]]) -> dict[str, Any]:
        """Preserve the single environment's observation time axis."""
        return batches[0]

    async def evaluate(
        self, input_channel: Channel, output_channel: Channel
    ) -> dict[str, torch.Tensor]:
        """Exchange observations and joint chunks through RLinf channels."""
        try:
            for _ in range(self.cfg.env.eval.rollout_epoch):
                for _ in range(self.cfg.env.eval.max_steps_per_rollout_epoch // 48):
                    batch = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=0,
                        async_op=True,
                        batch_size=1,
                        merge_fn=self.merge_observations,
                        infer_batch_size_fn=infer_obs_batch_size,
                    ).async_wait()
                    actions = self.client.predict(batch["dream_observations"])
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=torch.from_numpy(actions),
                        tag="eval_rollout_results",
                        route_key=0,
                        async_op=True,
                        batch_size=1,
                    )
            return {
                "dream_calls": torch.tensor([self.client.calls]),
                "dream_inactive_arm_holds": torch.tensor(
                    [self.client.inactive_arm_holds]
                ),
            }
        finally:
            self.close()

    def close(self) -> None:
        """Release the SDK session."""
        if self.client is not None:
            self.client.close()
