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

import os
import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.scheduler import Channel, Worker
from rlinf.utils.obs_compression import infer_obs_batch_size


def validate_dream_cfg(cfg: DictConfig) -> None:
    """Validate the synchronous, single-environment Dream contract."""
    env = cfg.env.eval
    if not cfg.runner.only_eval or env.env_type not in ("robotwin", "real"):
        raise ValueError(
            "Dream requires evaluation with RoboTwin or DualYamJointEnv-v1."
        )
    if (
        cfg.rollout.rollout_backend != "dream"
        or cfg.rollout.model.model_type != "dream"
    ):
        raise ValueError("Dream requires rollout_backend=dream and model_type=dream.")
    if env.total_num_envs != 1 or cfg.rollout.pipeline_stage_num != 1 or env.auto_reset:
        raise ValueError("Dream requires one environment/stage and auto_reset=false.")
    if cfg.runner.get("enable_decoupled_mode", False) or cfg.runner.get("rtc", {}).get(
        "enabled", False
    ):
        raise ValueError("Dream requires synchronous evaluation with RTC disabled.")
    if cfg.env.get("obs_compression", {}).get("enable", False):
        raise ValueError("Dream encodes RGB frames as JPEG; disable obs_compression.")
    if (cfg.rollout.model.num_action_chunks, cfg.rollout.model.action_dim) != (48, 14):
        raise ValueError("Dream requires 48-row, 14-dimensional joint chunks.")
    if env.max_steps_per_rollout_epoch % 48:
        raise ValueError("Dream max_steps_per_rollout_epoch must be divisible by 48.")
    if cfg.rollout.dream.head not in ("joint", "eef"):
        raise ValueError("Dream head must be joint or eef.")
    for key in ("action_hz", "timeout_ms", "max_calls"):
        if cfg.rollout.dream[key] <= 0:
            raise ValueError(f"Dream {key} must be positive.")
    required_calls = env.max_steps_per_rollout_epoch // 48 * env.rollout_epoch
    if cfg.rollout.dream.max_calls < required_calls:
        raise ValueError("Dream max_calls must cover the configured evaluation.")
    if env.env_type == "robotwin":
        if list(env.task_config.embodiment) not in (
            ["aloha-agilex"],
            ["piper", "piper", 0.6],
        ):
            raise ValueError("Dream RoboTwin examples support aloha-agilex and piper.")
        if cfg.rollout.dream.head != "eef":
            raise ValueError(
                "Dream RoboTwin transfer requires head=eef and an IK arm file."
            )
        if env.skip_intermediate_renders or env.center_crop:
            raise ValueError(
                "Dream requires intermediate RGB frames with center_crop=false."
            )
        if not (
            env.task_config.camera.collect_head_camera
            and env.task_config.camera.collect_wrist_camera
            and env.task_config.data_type.qpos
        ):
            raise ValueError(
                "Dream RoboTwin requires head/wrist RGB and qpos observations."
            )
    else:
        if env.init_params.id != "DualYamJointEnv-v1" or cfg.rollout.dream.arm != "yam":
            raise ValueError(
                "Dream real evaluation requires DualYamJointEnv-v1 and arm=yam."
            )
        if (
            env.main_image_key != "top_rgb"
            or env.override_cfg.step_frequency != cfg.rollout.dream.action_hz
        ):
            raise ValueError(
                "Dream YAM requires top_rgb and matching policy/control frequencies."
            )
        if env.ignore_terminations or env.get("teleop", "none") != "none":
            raise ValueError(
                "Dream YAM requires ignore_terminations=false and teleop=none."
            )


class DreamActionClient:
    """Adapt RLinf observation history to the official Dream SDK."""

    def __init__(self, sdk: Any, cfg: DictConfig) -> None:
        self.sdk = sdk
        self.cfg = cfg
        self.calls = 0
        self.request_s = 0.0
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
                        "head": cfg.dream.head,
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
                        "action": (
                            {
                                "left_eef_xyz": [0, 3],
                                "left_eef_rot6": [3, 9],
                                "left_grip": [9, 10],
                                "right_eef_xyz": [10, 13],
                                "right_eef_rot6": [13, 19],
                                "right_grip": [19, 20],
                            }
                            if cfg.dream.head == "eef"
                            else {
                                "left_arm": [0, 6],
                                "left_gripper": [6, 7],
                                "right_arm": [7, 13],
                                "right_gripper": [13, 14],
                            }
                        ),
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
        if states.ndim != 3 or states.shape[0] != 1 or states.shape[-1] != 14:
            raise ValueError("Dream expects states with shape [1,T,14].")
        if states.shape[1] == 0:
            raise ValueError("Dream requires at least one observation.")
        if not np.isfinite(states).all():
            raise ValueError("Dream observation states must be finite.")
        if bool(sample["finished"][0]):
            return np.repeat(states[:, -1:], 48, axis=1)
        episode_id = int(sample["episode_id"][0])
        if episode_id != self.episode_id:
            self.sdk.reset_episode()
            self.episode_id = episode_id
        frames = sample["frames"][0]
        timestamps = np.asarray(sample["timestamps"])
        if len(frames) != states.shape[1] or timestamps.shape != states.shape[:2]:
            raise ValueError(
                "Dream frames, states and timestamps must have matching lengths."
            )
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
        if self.calls >= self.cfg.dream.max_calls:
            raise RuntimeError("Dream request budget exhausted.")
        started = time.perf_counter()
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
        try:
            reply = self.sdk.get_action(commanded=last)
        finally:
            self.request_s += time.perf_counter() - started
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
        if self._world_size != 1:
            raise ValueError("Dream requires exactly one rollout worker.")
        self.cfg = cfg
        self.client = None

    def init_worker(self) -> None:
        """Initialize the optional Dream SDK in the rollout process."""
        if not os.environ.get("DREAM_API_KEY"):
            raise ValueError("Set DREAM_API_KEY before starting RLinf.")
        from dream import DreamClient

        self.client = DreamActionClient(DreamClient(), self.cfg.rollout)

    @staticmethod
    def merge_observations(batches: list[dict[str, Any]]) -> dict[str, Any]:
        """Preserve the single environment's observation time axis."""
        if len(batches) != 1:
            raise ValueError("Dream expects one environment source.")
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
                "dream_request_s": torch.tensor([self.client.request_s]),
            }
        finally:
            self.close()

    def close(self) -> None:
        """Release the SDK session."""
        if self.client is not None:
            self.client.close()
