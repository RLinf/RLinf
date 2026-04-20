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

"""Teleoperation script for RealWorld env using SpaceMouse, with reward model inference.

Usage:
    bash examples/reward/run_realworld_teleop.sh

Controls:
    - Move SpaceMouse to teleoperate the robot arm
    - Left button: close gripper
    - Right button: open gripper
    - Ctrl+C to stop
"""

import json
import signal
from dataclasses import dataclass, field
from typing import Optional

import hydra
import numpy as np
from omegaconf import OmegaConf

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import HybridComponentPlacement

logger = get_logger()


@dataclass
class RewardInjectionCfg:
    """Pre-computed configuration for launching the reward worker alongside TeleopWorker."""

    use_reward_model: bool = False
    reward_worker_cfg: dict = field(default_factory=dict)
    reward_worker_hardware_rank: Optional[int] = None
    reward_worker_node_rank: Optional[int] = None
    reward_worker_node_group: Optional[str] = None


def build_reward_injection(cfg, component_placement) -> RewardInjectionCfg:
    """Build the configuration needed to launch the reward worker alongside TeleopWorker."""
    injection = RewardInjectionCfg()

    if not (
        cfg.reward.get("use_reward_model", False)
        and cfg.reward.get("standalone_realworld", False)
    ):
        return injection

    injection.use_reward_model = True
    injection.reward_worker_cfg = OmegaConf.to_container(cfg.reward, resolve=True)

    reward_hardware_ranks = component_placement.get_hardware_ranks("reward")
    if not reward_hardware_ranks:
        raise ValueError("Reward placement must contain at least one hardware rank.")
    injection.reward_worker_hardware_rank = reward_hardware_ranks[0]

    reward_placements = component_placement.get_strategy("reward").get_placement(
        Cluster()
    )
    if not reward_placements:
        raise ValueError("Reward placement must contain at least one worker.")
    reward_placement = reward_placements[0]
    injection.reward_worker_node_rank = reward_placement.cluster_node_rank
    injection.reward_worker_node_group = reward_placement.node_group_label

    return injection


class TeleopWorker(Worker):
    """Worker that continuously teleoperates the robot arm via SpaceMouse.

    The RealWorldEnv is initialized with use_spacemouse=True, which wraps
    the gym env with SpacemouseIntervention. When SpaceMouse input is
    non-zero (or a button is pressed), the wrapper replaces the policy
    action with the SpaceMouse action for 0.5 seconds.

    The reward model (when enabled) runs on the GPU node and receives camera
    images each step for real-time inference.
    """

    def __init__(self, cfg, reward_injection: Optional[RewardInjectionCfg] = None):
        super().__init__()
        self.cfg = cfg
        self.reward_injection = reward_injection
        self._quit = False

        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )
        self._reward_worker = self._setup_reward_worker()

    def _setup_reward_worker(self):
        """Launch an EmbodiedRewardWorker on the GPU node for teleop-time inference."""
        if self.reward_injection is None or not self.reward_injection.use_reward_model:
            logger.info(
                "[TeleopWorker] Reward worker is NOT enabled "
                "(use_reward_model=True and standalone_realworld=True is required)."
            )
            return None

        logger.info(
            "[TeleopWorker] Launching EmbodiedRewardWorker on "
            "node_rank=%s node_group=%s hardware_rank=%s ...",
            self.reward_injection.reward_worker_node_rank,
            self.reward_injection.reward_worker_node_group,
            self.reward_injection.reward_worker_hardware_rank,
        )
        from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

        reward_worker = EmbodiedRewardWorker.launch_for_realworld(
            reward_cfg=self.reward_injection.reward_worker_cfg,
            node_rank=self.reward_injection.reward_worker_node_rank,
            node_group_label=self.reward_injection.reward_worker_node_group,
            hardware_rank=self.reward_injection.reward_worker_hardware_rank,
            env_idx=0,
            worker_rank=0,
        )
        reward_worker.init_worker().wait()
        logger.info(
            "[TeleopWorker] EmbodiedRewardWorker initialized successfully on %s.",
            type(reward_worker).__name__,
        )
        return reward_worker

    def _handle_signal(self, signum, frame):
        logger.info("Received signal %d, will quit after current step.", signum)
        self._quit = True

    def _setup_signal_handler(self):
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _run_teleop(self):
        logger.info("Starting teleoperation loop.")
        logger.info("  Move SpaceMouse to teleoperate the robot arm.")
        logger.info("  Left button: close gripper | Right button: open gripper.")
        logger.info("  Ctrl+C to stop.")
        if self._reward_worker is None:
            logger.warning(
                "[TeleopWorker] Reward worker NOT initialized — "
                "probs will not be logged. "
                "Ensure use_reward_model=True and standalone_realworld=True in config."
            )
        else:
            logger.info(
                "[TeleopWorker] Reward worker ready: type=%s | reward_threshold=%.3f",
                type(self._reward_worker).__name__,
                self.cfg.reward.get("reward_threshold", 0.6),
            )

        obs, _ = self.env.reset()
        step = 0

        while not self._quit:
            if self.cfg.env.eval.get("no_gripper", True):
                action = np.zeros((1, 6), dtype=np.float32)
            else:
                action = np.zeros((1, 7), dtype=np.float32)

            obs, _, terminated, truncated, _ = self.env.step(action)

            rm_reward = None
            if self._reward_worker is not None:
                raw_frames = obs.get("main_images", None)
                if raw_frames is not None:
                    image_np = (
                        raw_frames.numpy()
                        if hasattr(raw_frames, "numpy")
                        else np.array(raw_frames)
                    )
                    if image_np.ndim == 3:
                        image_np = np.expand_dims(image_np, axis=0)
                    result = self._reward_worker.compute_image_rewards(image_np).wait()[
                        0
                    ]
                    if hasattr(result, "numpy"):
                        result = result.numpy()
                    rm_reward = int(np.asarray(result).reshape(-1)[0])

            rm_str = (
                f" | rm_reward: {rm_reward} | success: {bool(rm_reward)}"
                if rm_reward is not None
                else ""
            )
            logger.info("Step %-6d%s", step, rm_str)

            if terminated or truncated:
                obs, _ = self.env.reset()
                logger.info("Episode ended, resetting.")

            step += 1

        logger.info("Teleoperation stopped after %d steps.", step)

    def run(self):
        self._setup_signal_handler()
        self._run_teleop()
        self.env.close()
        logger.info("TeleopWorker finished.")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_teleop",
)
def main(cfg) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")

    reward_injection = build_reward_injection(cfg, component_placement)
    if reward_injection.use_reward_model:
        logger.info(
            "[main] Reward injection configured: "
            "node_rank=%s node_group=%s hardware_rank=%s model_path=%s",
            reward_injection.reward_worker_node_rank,
            reward_injection.reward_worker_node_group,
            reward_injection.reward_worker_hardware_rank,
            reward_injection.reward_worker_cfg.get("model", {}).get(
                "model_path", "N/A"
            ),
        )
    else:
        logger.info(
            "[main] Reward injection NOT configured — "
            "skipping reward worker. "
            "Requires use_reward_model=True AND standalone_realworld=True in config."
        )

    teleop = TeleopWorker.create_group(cfg, reward_injection=reward_injection).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=env_placement,
    )

    teleop.run().wait()


if __name__ == "__main__":
    main()
