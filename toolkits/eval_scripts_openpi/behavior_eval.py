# Copyright 2025 The RLinf Authors.
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

"""
Single-threaded BEHAVIOR evaluation script for detailed metrics.

This script provides granular evaluation metrics for BEHAVIOR tasks,
similar to the OpenPI evaluation approach but integrated with RLinf models.
It runs single-threaded and provides detailed per-instance metrics.

For faster parallel evaluation, use the main RLinf evaluation pipeline:
    python examples/embodiment/eval_embodied_agent.py --config-name behavior_openvlaoft_eval
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

# OmniGibson imports
import omnigibson as og
from omnigibson.envs import VectorEnvironment
from omnigibson.learning.utils.eval_utils import (
    TASK_NAMES_TO_INDICES,
    TASK_INDICES_TO_NAMES,
    ROBOT_CAMERA_NAMES,
)
from omnigibson.learning.utils.obs_utils import create_video_writer, write_video
from omnigibson.macros import gm, create_module_macros
from omnigibson.utils.asset_utils import get_task_instance_path
from omnigibson.utils.python_utils import recursively_convert_to_torch
from omegaconf import OmegaConf
import omnigibson.utils.transform_utils as T

# Set up OmniGibson settings
gm.HEADLESS = True
gm.ENABLE_FLATCACHE = True
gm.USE_GPU_DYNAMICS = False
gm.ENABLE_TRANSITION_RULES = True
gm.ENABLE_OBJECT_STATES = True

m = create_module_macros(module_path=__file__)
m.NUM_EVAL_EPISODES = 1
m.NUM_EVAL_INSTANCES = 10

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BehaviorEvaluator:
    """
    BEHAVIOR task evaluator that provides detailed metrics for each task instance.
    
    This evaluator runs single-threaded and provides:
    - Per-instance success rates
    - Episode lengths
    - Task-specific metrics
    - Video recordings
    """
    
    def __init__(self, args):
        self.args = args
        self.task_name = args.task_name
        self.num_envs = 1  # Single-threaded evaluation
        
        # Set random seeds
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        
        # Load task configuration
        self.task_idx = TASK_NAMES_TO_INDICES[self.task_name]
        
        # Load human demo statistics
        self.human_stats = self._load_human_stats()
        
        # Load policy
        self.policy = self._load_policy()
        
        # Statistics
        self.total_trials = 0
        self.successful_trials = 0
        self.instance_results = []
        
        logger.info(f"Initialized BEHAVIOR evaluator for task: {self.task_name}")
        logger.info(f"Task index: {self.task_idx}")
        logger.info(f"Human demo avg length: {self.human_stats['length']:.1f}")
    
    def _load_human_stats(self):
        """Load human demonstration statistics for the task."""
        stats = {
            "length": [],
            "distance_traveled": [],
            "left_eef_displacement": [],
            "right_eef_displacement": [],
        }
        
        metadata_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-task-instances",
            "metadata",
            "episodes.jsonl"
        )
        
        with open(metadata_path, "r") as f:
            episodes = [json.loads(line) for line in f]
        
        for episode in episodes:
            if episode["episode_index"] // 1e4 == self.task_idx:
                for k in stats.keys():
                    stats[k].append(episode[k])
        
        # Compute averages
        for k in stats.keys():
            stats[k] = sum(stats[k]) / len(stats[k]) if stats[k] else 0
        
        return stats
    
    def _load_policy(self):
        """Load the policy model for evaluation."""
        # Import policy modules
        sys.path.insert(0, str(Path(self.args.pretrained_path).parent))
        
        if self.args.policy_type == "rlinf":
            return self._load_rlinf_policy()
        elif self.args.policy_type == "openpi":
            return self._load_openpi_policy()
        else:
            raise ValueError(f"Unknown policy type: {self.args.policy_type}")
    
    def _load_rlinf_policy(self):
        """Load RLinf-trained policy."""
        from rlinf.models import get_model_cls
        
        # Load config
        config_path = os.path.join(self.args.pretrained_path, "config.yaml")
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
        else:
            # Use default config
            logger.warning("No config found, using default OpenVLA-OFT config")
            from hydra import compose, initialize
            with initialize(config_path="../examples/embodiment/config"):
                cfg = compose(config_name="behavior_openvlaoft_eval")
        
        # Create model
        model_cls = get_model_cls(cfg.actor.model.model_type)
        model = model_cls(cfg.actor.model)
        
        # Load checkpoint
        checkpoint_path = os.path.join(self.args.pretrained_path, "model.pt")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location="cuda")
            model.load_state_dict(checkpoint)
        
        model = model.cuda().eval()
        
        return RLinfPolicyWrapper(
            model,
            cfg.actor.model,
            task_name=self.task_name,
            action_chunk=self.args.action_chunk
        )
    
    def _load_openpi_policy(self):
        """Load OpenPI-style policy."""
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        from openpi.shared.eval_b1k_wrapper import B1KPolicyWrapper
        
        # Load policy
        policy = _policy_config.create_trained_policy(
            _config.get_config(self.args.config_name),
            self.args.pretrained_path
        )
        
        # Wrap for BEHAVIOR evaluation
        policy = B1KPolicyWrapper(
            policy,
            task_name=self.task_name,
            control_mode=self.args.control_mode,
            max_len=self.args.max_len,
            action_horizon=self.args.action_chunk,
        )
        
        return policy
    
    def _create_env(self, instance_id=None):
        """Create OmniGibson environment for evaluation."""
        # Load base environment config
        config_path = os.path.join(og.example_config_path, "r1pro_behavior.yaml")
        cfg = OmegaConf.load(config_path)
        
        # Update task configuration
        cfg["task"]["activity_name"] = self.task_name
        if instance_id is not None:
            cfg["task"]["activity_definition_id"] = 0
            cfg["task"]["activity_instance_id"] = instance_id
        
        # Set timeout based on human demos
        cfg["task"]["termination_config"]["max_steps"] = int(
            self.human_stats["length"] * 2
        )
        
        # Configure observations
        cfg["robots"][0]["obs_modalities"] = ["proprio", "rgb"]
        
        # Create environment
        env = VectorEnvironment(
            self.num_envs,
            OmegaConf.to_container(cfg, resolve=True)
        )
        
        return env
    
    def _load_task_instance(self, env, instance_id):
        """Load a specific task instance into the environment."""
        robot = env.scene.object_registry("name", "robot_r1")
        scene_model = env.task.scene_name
        
        tro_filename = env.task.get_cached_activity_scene_filename(
            scene_model=scene_model,
            activity_name=env.task.activity_name,
            activity_definition_id=env.task.activity_definition_id,
            activity_instance_id=instance_id,
        )
        
        tro_file_path = os.path.join(
            get_task_instance_path(scene_model),
            f"json/{scene_model}_task_{env.task.activity_name}_instances/{tro_filename}-tro_state.json",
        )
        
        with open(tro_file_path, "r") as f:
            tro_state = recursively_convert_to_torch(json.load(f))
        
        for tro_key, state in tro_state.items():
            if tro_key == "robot_poses":
                robot_pos = state[robot.model_name][0]["position"]
                robot_quat = state[robot.model_name][0]["orientation"]
                robot.set_position_orientation(robot_pos, robot_quat)
                env.scene.write_task_metadata(key=tro_key, data=state)
            else:
                env.task.object_scope[tro_key].load_state(state, serialized=False)
        
        # Stabilize objects
        for _ in range(25):
            og.sim.step_physics()
            for entity in env.task.object_scope.values():
                if not entity.is_system and entity.exists:
                    entity.keep_still()
        
        env.scene.update_initial_file()
        env.scene.reset()
    
    def _preprocess_obs(self, obs, robot):
        """Preprocess raw observations for policy input."""
        obs = obs[0]  # Extract first env obs
        
        # Extract camera images
        images = {}
        for sensor_data in obs.values():
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    images["left_wrist"] = v["rgb"][..., :3]
                elif "right_realsense_link:Camera:0" in k:
                    images["right_wrist"] = v["rgb"][..., :3]
                elif "zed_link:Camera:0" in k:
                    images["head"] = v["rgb"][..., :3]
        
        # Get camera relative poses
        base_pose = robot.get_position_orientation()
        cam_rel_poses = []
        
        for camera_name in ROBOT_CAMERA_NAMES["R1Pro"].values():
            camera = robot.sensors[camera_name.split("::")[1]]
            direct_cam_pose = camera.camera_parameters["cameraViewTransform"]
            
            if np.allclose(direct_cam_pose, np.zeros(16)):
                cam_rel_poses.append(
                    torch.cat(T.relative_pose_transform(
                        *(camera.get_position_orientation()), *base_pose
                    ))
                )
            else:
                cam_pose = T.mat2pose(
                    torch.tensor(
                        np.linalg.inv(np.reshape(direct_cam_pose, [4, 4]).T),
                        dtype=torch.float32
                    )
                )
                cam_rel_poses.append(
                    torch.cat(T.relative_pose_transform(*cam_pose, *base_pose))
                )
        
        # Prepare policy input
        processed_obs = {
            "main_images": images["head"].unsqueeze(0),  # [1, H, W, C]
            "wrist_images": torch.stack(
                [images["left_wrist"], images["right_wrist"]]
            ).unsqueeze(0),  # [1, 2, H, W, C]
            "task_descriptions": [self._get_task_description()],
            "cam_rel_poses": torch.cat(cam_rel_poses, axis=-1).unsqueeze(0),
        }
        
        return processed_obs
    
    def _get_task_description(self):
        """Get natural language description for the task."""
        task_descriptions_path = os.path.join(
            os.path.dirname(__file__),
            "../../rlinf/envs/behavior/behavior_task.jsonl"
        )
        
        with open(task_descriptions_path, "r") as f:
            descriptions = [json.loads(x) for x in f.read().strip().split("\n") if x]
        
        description_map = {
            desc["task_name"]: desc["task"] for desc in descriptions
        }
        
        return description_map.get(self.task_name, f"Complete the {self.task_name} task")
    
    def _run_episode(self, env, instance_id, episode_id, save_video=False):
        """Run a single evaluation episode."""
        robot = env.scene.object_registry("name", "robot_r1")
        
        # Reset environment
        obs, info = env.reset()
        obs = self._preprocess_obs(obs, robot)
        
        # Initialize video writer if needed
        video_writer = None
        if save_video:
            video_dir = Path(self.args.log_path) / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = video_dir / f"video_{instance_id}_{episode_id}.mp4"
            video_writer = create_video_writer(
                fpath=str(video_path),
                resolution=(448, 672),
            )
        
        # Reset policy
        self.policy.reset()
        
        # Run episode
        done = False
        step = 0
        success = False
        
        while not done and step < self.args.max_steps:
            # Get action from policy
            action = self.policy.forward(obs=obs)
            
            # Step environment
            obs_raw, reward, terminated, truncated, info = env.step(action)
            obs = self._preprocess_obs(obs_raw, robot)
            
            # Write video frame
            if save_video:
                self._write_video_frame(obs_raw, video_writer)
            
            done = terminated[0] or truncated[0]
            success = info[0].get("done", {}).get("success", False)
            step += 1
        
        # Close video writer
        if video_writer is not None:
            container, stream = video_writer
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        
        return {
            "instance_id": instance_id,
            "episode_id": episode_id,
            "success": success,
            "episode_length": step,
            "terminated": terminated[0].item(),
            "truncated": truncated[0].item(),
        }
    
    def _write_video_frame(self, obs_raw, video_writer):
        """Write a single frame to the video."""
        obs = obs_raw[0]
        
        # Extract images
        for sensor_data in obs.values():
            for k, v in sensor_data.items():
                if "left_realsense_link:Camera:0" in k:
                    left_wrist_rgb = cv2.resize(v["rgb"].numpy(), (224, 224))
                elif "right_realsense_link:Camera:0" in k:
                    right_wrist_rgb = cv2.resize(v["rgb"].numpy(), (224, 224))
                elif "zed_link:Camera:0" in k:
                    head_rgb = cv2.resize(v["rgb"].numpy(), (448, 448))
        
        # Concatenate images
        frame = np.hstack([
            np.vstack([left_wrist_rgb, right_wrist_rgb]),
            head_rgb
        ])
        
        write_video(
            np.expand_dims(frame, 0),
            video_writer=video_writer,
            batch_size=1,
            mode="rgb",
        )
    
    def evaluate(self):
        """Run full evaluation across all test instances."""
        # Load test instances
        csv_path = os.path.join(
            gm.DATA_PATH,
            "2025-challenge-task-instances",
            "metadata",
            "test_instances.csv"
        )
        
        with open(csv_path, "r") as f:
            lines = list(csv.reader(f))[1:]
        
        task_line = lines[self.task_idx]
        assert task_line[1] == self.task_name, \
            f"Task name mismatch: {task_line[1]} vs {self.task_name}"
        
        test_instances = [int(x) for x in task_line[2].strip().split(",")]
        
        # Filter instances if specified
        if self.args.eval_instance_ids:
            test_instances = [test_instances[i] for i in self.args.eval_instance_ids]
        
        logger.info(f"Evaluating on {len(test_instances)} instances")
        logger.info(f"Instance IDs: {test_instances}")
        
        # Run evaluation
        all_results = []
        
        for instance_id in tqdm(test_instances, desc="Evaluating instances"):
            # Create environment for this instance
            env = self._create_env()
            self._load_task_instance(env, instance_id)
            
            instance_results = []
            
            for episode_id in range(self.args.num_episodes_per_instance):
                save_video = len(all_results) < self.args.num_save_videos
                
                result = self._run_episode(
                    env,
                    instance_id,
                    episode_id,
                    save_video=save_video
                )
                
                instance_results.append(result)
                all_results.append(result)
                
                self.total_trials += 1
                if result["success"]:
                    self.successful_trials += 1
                
                logger.info(
                    f"Instance {instance_id}, Episode {episode_id}: "
                    f"Success={result['success']}, Length={result['episode_length']}"
                )
            
            # Store instance-level results
            instance_success_rate = sum(r["success"] for r in instance_results) / len(instance_results)
            instance_avg_length = np.mean([r["episode_length"] for r in instance_results])
            
            self.instance_results.append({
                "instance_id": instance_id,
                "success_rate": instance_success_rate,
                "avg_episode_length": instance_avg_length,
                "num_episodes": len(instance_results),
            })
            
            # Clean up
            env.close()
            og.shutdown()
        
        return all_results
    
    def save_results(self, results):
        """Save evaluation results to files."""
        results_dir = Path(self.args.log_path) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = results_dir / f"{self.task_name}_detailed.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        
        # Save instance-level summary
        instance_file = results_dir / f"{self.task_name}_instances.json"
        with open(instance_file, "w") as f:
            json.dump(self.instance_results, f, indent=2)
        
        # Save overall summary
        summary = {
            "task_name": self.task_name,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "success_rate": self.successful_trials / self.total_trials if self.total_trials > 0 else 0,
            "avg_episode_length": np.mean([r["episode_length"] for r in results]),
            "num_instances": len(self.instance_results),
            "instance_success_rates": [r["success_rate"] for r in self.instance_results],
        }
        
        summary_file = results_dir / f"{self.task_name}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
        logger.info(f"Overall success rate: {summary['success_rate']:.2%}")
        logger.info(f"Average episode length: {summary['avg_episode_length']:.1f}")


class RLinfPolicyWrapper:
    """Wrapper for RLinf-trained policies to match evaluation interface."""
    
    def __init__(self, model, model_cfg, task_name, action_chunk=1):
        self.model = model
        self.model_cfg = model_cfg
        self.task_name = task_name
        self.action_chunk = action_chunk
        self.action_buffer = []
        self.last_obs = None
    
    def reset(self):
        """Reset policy state."""
        self.action_buffer = []
        self.last_obs = None
    
    def forward(self, obs):
        """Generate action from observation."""
        # Check if we have buffered actions
        if len(self.action_buffer) > 0:
            return self.action_buffer.pop(0)
        
        # Generate new action chunk
        with torch.no_grad():
            # Prepare input
            model_input = self._prepare_input(obs)
            
            # Forward pass
            output = self.model.forward(**model_input)
            
            # Extract actions
            actions = output["actions"]  # [1, action_chunk, action_dim]
            actions = actions[0].cpu().numpy()  # [action_chunk, action_dim]
            
            # Buffer actions
            for i in range(len(actions)):
                self.action_buffer.append(actions[i])
        
        # Return first action
        return self.action_buffer.pop(0) if self.action_buffer else np.zeros(23)
    
    def _prepare_input(self, obs):
        """Prepare observation for model input."""
        # Convert to model format
        model_input = {
            "images": obs["wrist_images"].cuda(),  # [1, 2, H, W, C]
            "main_images": obs["main_images"].cuda(),  # [1, H, W, C]
            "task_descriptions": obs["task_descriptions"],
        }
        
        if "cam_rel_poses" in obs:
            model_input["cam_rel_poses"] = obs["cam_rel_poses"].cuda()
        
        return model_input


def main():
    parser = argparse.ArgumentParser(description="BEHAVIOR single-threaded evaluation")
    
    # Task configuration
    parser.add_argument("--task_name", type=str, required=True,
                       help="BEHAVIOR task name (e.g., turning_on_radio)")
    parser.add_argument("--eval_instance_ids", type=int, nargs="+", default=None,
                       help="Specific instance IDs to evaluate (default: all 10)")
    parser.add_argument("--num_episodes_per_instance", type=int, default=1,
                       help="Number of episodes per instance")
    
    # Model configuration
    parser.add_argument("--policy_type", type=str, default="rlinf",
                       choices=["rlinf", "openpi"],
                       help="Type of policy to evaluate")
    parser.add_argument("--pretrained_path", type=str, required=True,
                       help="Path to pretrained model checkpoint")
    parser.add_argument("--config_name", type=str, default=None,
                       help="Config name for OpenPI policies")
    
    # Action configuration
    parser.add_argument("--action_chunk", type=int, default=32,
                       help="Action chunk size")
    parser.add_argument("--control_mode", type=str, default="receeding_horizon",
                       choices=["receeding_horizon", "temporal_ensemble", "receeding_temporal"],
                       help="Control mode for OpenPI policies")
    parser.add_argument("--max_len", type=int, default=32,
                       help="Max length for action generation")
    parser.add_argument("--max_steps", type=int, default=2000,
                       help="Maximum steps per episode")
    
    # Logging configuration
    parser.add_argument("--log_path", type=str, default="./eval_results",
                       help="Path to save evaluation results")
    parser.add_argument("--num_save_videos", type=int, default=10,
                       help="Number of videos to save")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Create evaluator and run
    evaluator = BehaviorEvaluator(args)
    results = evaluator.evaluate()
    evaluator.save_results(results)
    
    logger.info("Evaluation completed!")
    logger.info(f"Success rate: {evaluator.successful_trials / evaluator.total_trials:.2%}")


if __name__ == "__main__":
    main()

