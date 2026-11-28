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

"""Continue a frozen Go2 policy with the RSL-RL PPO comparison recipe."""

import argparse
import os
import sys
from pathlib import Path
from types import MethodType

from rlinf.utils.logging import get_logger

from .evaluate import TASKS

logger = get_logger()
ITERATIONS = 1000


def _bound_rough_distribution(policy) -> None:
    import torch
    from torch.distributions import Normal

    def bound_std(value):
        return torch.nan_to_num(value, nan=1.0, posinf=5.0, neginf=1e-6).clamp(
            1e-6, 5.0
        )

    original_update = policy._update_distribution

    def update(self, obs):
        if not self.state_dependent_std and self.noise_std_type == "scalar":
            with torch.no_grad():
                self.std.copy_(bound_std(self.std))
        original_update(obs)
        mean = torch.nan_to_num(self.distribution.loc, nan=0.0, posinf=1e6, neginf=-1e6)
        self.distribution = Normal(mean, bound_std(self.distribution.scale))

    policy._update_distribution = MethodType(update, policy)


def _learn(runner, reward_manager, stairs: bool) -> None:
    if not stairs:
        runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=True)
        return

    # Preserve the stair recipe's repeated runner calls and reward updates.
    for iteration in range(0, ITERATIONS, 25):
        fraction = min(1.0, iteration / round(0.35 * ITERATIONS))
        for name, weight in (("base_height", -3.5), ("flat_orientation_l2", -2.0)):
            term = reward_manager.get_term_cfg(name)
            term.weight = (1.0 - fraction) * weight + fraction * weight
            reward_manager.set_term_cfg(name, term)
        runner.learn(num_learning_iterations=25, init_at_random_ep_len=iteration == 0)


def _train(args, env_cfg, agent_cfg) -> None:
    import gymnasium as gym
    import torch
    from isaaclab.utils.io import dump_yaml
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    from rlinf.envs.sim.isaaclab.tasks.go2 import (
        install_rough_action_bound,
        terrain_base_height_l2,
    )

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False

    agent_cfg.seed = args.seed
    agent_cfg.experiment_name = f"{args.task}_ppo"
    agent_cfg.run_name = f"{agent_cfg.experiment_name}_s{args.seed}"
    agent_cfg.max_iterations = ITERATIONS
    agent_cfg.device = args.device
    env_cfg.seed = args.seed
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = 1024 if args.task == "C" else 2048
    env_cfg.rewards.base_height.func = terrain_base_height_l2
    env_cfg.log_dir = str(args.output)

    env = gym.make(TASKS[args.task].gym_id, cfg=env_cfg)
    try:
        if args.task == "B":
            install_rough_action_bound(env.unwrapped)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = OnPolicyRunner(
            wrapped,
            agent_cfg.to_dict(),
            log_dir=str(args.output),
            device=agent_cfg.device,
        )
        runner.add_git_repo_to_log(__file__)
        if args.task == "B":
            _bound_rough_distribution(runner.alg.policy)
        runner.load(
            str(args.checkpoint), load_optimizer=False, map_location=args.device
        )
        runner.current_learning_iteration = 0
        dump_yaml(str(args.output / "params/env.yaml"), env_cfg)
        dump_yaml(str(args.output / "params/agent.yaml"), agent_cfg)
        _learn(runner, env.unwrapped.reward_manager, stairs=args.task == "C")
    finally:
        env.close()


def main() -> None:
    """Run the fixed PPO continuation recipe for one task and seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", choices=(0, 2), type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("checkpoint", type=Path, help="Frozen baseline weights")
    parser.add_argument("output", type=Path, help="New training output directory")
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve(strict=True)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)

    from isaaclab.app import AppLauncher

    kit_args = "--/app/extensions/excluded/0=omni.kit.telemetry"
    if args.asset_root is not None:
        kit_args += f" --/persistent/isaac/asset_root/cloud={args.asset_root.resolve()}"
    sys.argv = [sys.argv[0]]
    _simulation_app = AppLauncher(
        headless=True, device=args.device, kit_args=kit_args
    ).app
    exit_code = 0
    try:
        from isaaclab_tasks.utils.hydra import hydra_task_config

        from rlinf.envs.sim.isaaclab.tasks.go2 import register_go2_tasks

        register_go2_tasks()

        @hydra_task_config(TASKS[args.task].gym_id, "rsl_rl_cfg_entry_point")
        def train(env_cfg, agent_cfg):
            _train(args, env_cfg, agent_cfg)

        train()
    except KeyboardInterrupt:
        exit_code = 130
    except Exception:
        logger.exception("Go2 RSL-RL training failed")
        exit_code = 1
    # Kit can hang during headless interpreter finalization.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
