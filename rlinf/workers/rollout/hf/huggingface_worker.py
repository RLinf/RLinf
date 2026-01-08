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

import copy
import gc
import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

if TYPE_CHECKING:
    from rlinf.algorithms.rewards.embodiment import RewardManager
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.io_struct import ChunkStepResult, EmbodiedRolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.metric_utils import compute_split_num
from rlinf.utils.nested_dict_process import put_tensor_device
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.rollout.hf.utils import init_real_obs

logger = logging.getLogger(__name__)


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = torch.cuda.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        actor_world_size = self.placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.hf_model.eval()

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

        # Initialize reward manager if configured
        self.reward_manager: Optional["RewardManager"] = None
        self._init_reward_manager()

        # Initialize reward data collector if configured
        self.data_collector = None
        self._init_data_collector()

    def _init_reward_manager(self) -> None:
        """Initialize the reward manager if reward model is configured.

        The reward manager provides model-based reward computation for
        embodied RL tasks. It is optional and only initialized when
        cfg.reward.use_reward_model is True.
        """
        reward_cfg = self.cfg.get("reward")
        if reward_cfg is None:
            return
        use_model = reward_cfg.get("use_reward_model", False)
        if not bool(use_model):
            return

        from rlinf.algorithms.rewards.embodiment import RewardManager
        self.reward_manager = RewardManager(reward_cfg)
        logger.info(
            f"Initialized reward manager with model type: "
            f"{self.reward_manager.model_type}"
        )

    def _init_data_collector(self) -> None:
        """Initialize the reward data collector if configured.

        The data collector saves labeled image data during rollouts for
        subsequent reward model training.
        """
        collect_cfg = self.cfg.get("reward_data_collection")
        if collect_cfg is None or not collect_cfg.get("enabled", False):
            return

        try:
            from rlinf.algorithms.rewards.embodiment.reward_data_collector import (
                RewardDataCollector,
            )
            self.data_collector = RewardDataCollector(collect_cfg)
            logger.info(
                f"Initialized reward data collector. "
                f"Save dir: {collect_cfg.get('save_dir')}"
            )
        except Exception as e:
            logger.warning(f"Failed to initialize data collector: {e}")
            self.data_collector = None

    def collect_reward_data(
        self,
        raw_obs: dict[str, Any],
        terminations: Optional[torch.Tensor] = None,
        successes: Optional[torch.Tensor] = None,
        dones: Optional[torch.Tensor] = None,
        step_id: int = 0,
        goal_positions_2d: Optional[list] = None,
    ) -> bool:
        """Collect data for reward model training if collector is enabled.

        Args:
            raw_obs: Raw observations containing images.
            terminations: Boolean tensor indicating terminal states (optional).
            successes: Optional tensor indicating success label.
            dones: Optional tensor indicating episode end (for resetting counters).
            step_id: Current step number.
            goal_positions_2d: Optional list of (x, y) goal positions in pixel coords.

        Returns:
            True if collection is complete, False otherwise.
        """
        if self.data_collector is None:
            return False

        # Get task descriptions if available
        task_descriptions = raw_obs.get("task_descriptions")

        self.data_collector.collect(
            observations=raw_obs,
            terminations=terminations if terminations is not None else torch.zeros(1),
            successes=successes,
            task_descriptions=task_descriptions,
            step_id=step_id,
            dones=dones,
            goal_positions_2d=goal_positions_2d,
        )

        # Check if collection is complete and save
        if self.data_collector.is_full:
            stats = self.data_collector.get_statistics()
            logger.info(
                f"Data collection complete! "
                f"Success: {stats['success_count']}, Failure: {stats['failure_count']}"
            )
            self.data_collector.save()
            return True

        return False

    def extract_goal_positions_2d(
        self, 
        raw_obs: dict[str, Any],
    ) -> Optional[list]:
        """Extract goal/cube positions and project to 2D pixel coordinates.

        For ManiSkill PickCube, projects the cube 3D position to 2D using camera params.

        Args:
            raw_obs: Raw observations from environment containing:
                - goal_pos: 3D goal position (extracted in maniskill_env)
                - sensor_param: Camera intrinsics/extrinsics

        Returns:
            List of (x, y) pixel coordinates, one per environment
        """
        try:
            import numpy as np
            
            # Get images for batch size and dimensions
            images = raw_obs.get("images")
            if images is None:
                images = raw_obs.get("main_images")
            if images is None:
                return None
            
            batch_size = images.shape[0]
            
            # Determine image dimensions
            if images.shape[-1] in [1, 3, 4]:  # [B, H, W, C]
                img_h, img_w = images.shape[1], images.shape[2]
            else:  # [B, C, H, W]
                img_h, img_w = images.shape[2], images.shape[3]
            
            # Get 3D goal position (extracted in maniskill_env from raw_obs["extra"])
            goal_pos_3d = raw_obs.get("goal_pos")
            
            # Get camera parameters
            sensor_param = raw_obs.get("sensor_param")
            
            goal_positions = []
            
            for i in range(batch_size):
                # Default position (center of image)
                goal_x, goal_y = img_w // 2, img_h // 2
                
                if goal_pos_3d is not None and sensor_param is not None and "base_camera" in sensor_param:
                    cam = sensor_param["base_camera"]
                    
                    # Get intrinsic matrix
                    intrinsic = cam.get("intrinsic_cv")
                    if intrinsic is None:
                        intrinsic = cam.get("intrinsic")
                    
                    # Get extrinsic (camera pose)
                    extrinsic = cam.get("extrinsic_cv")
                    if extrinsic is None:
                        extrinsic = cam.get("cam2world_gl")
                    
                    if intrinsic is not None and extrinsic is not None:
                        try:
                            # Convert to numpy
                            if isinstance(intrinsic, torch.Tensor):
                                K = intrinsic[i].cpu().numpy()
                            else:
                                K = np.array(intrinsic[i]) if len(np.array(intrinsic).shape) > 2 else np.array(intrinsic)
                                
                            if isinstance(extrinsic, torch.Tensor):
                                E = extrinsic[i].cpu().numpy()
                            else:
                                E = np.array(extrinsic[i]) if len(np.array(extrinsic).shape) > 2 else np.array(extrinsic)
                            
                            # Get 3D position for this env
                            if isinstance(goal_pos_3d, torch.Tensor):
                                pos_3d = goal_pos_3d[i].cpu().numpy()
                            else:
                                pos_3d = np.array(goal_pos_3d[i])
                            
                            # Project 3D -> 2D
                            # World to camera: p_cam = world2cam @ p_world
                            p_world = np.append(pos_3d[:3], 1.0)  # Homogeneous [x,y,z,1]
                            
                            # Camera extrinsic: cam2world, need world2cam
                            world2cam = np.linalg.inv(E)
                            p_cam = world2cam @ p_world
                            
                            # Camera to pixel: p_pixel = K @ p_cam[:3] / p_cam[2]
                            if p_cam[2] > 0:  # In front of camera
                                p_proj = K @ p_cam[:3]
                                goal_x = int(p_proj[0] / p_proj[2])
                                goal_y = int(p_proj[1] / p_proj[2])
                                
                                # Clamp to image bounds
                                goal_x = max(0, min(img_w - 1, goal_x))
                                goal_y = max(0, min(img_h - 1, goal_y))
                        except Exception as proj_err:
                            logger.debug(f"Projection failed for env {i}: {proj_err}")
                
                goal_positions.append((goal_x, goal_y))
            
            return goal_positions
            
        except Exception as e:
            logger.debug(f"Failed to extract goal positions: {e}")
            return None

    def compute_model_reward(
        self,
        observations: dict[str, Any],
        task_descriptions: Optional[list[str]] = None,
    ) -> Optional[torch.Tensor]:
        """Compute rewards using the reward manager if available.

        Args:
            observations: Dictionary containing observation data (images, states).
            task_descriptions: Optional task descriptions for VLM models.

        Returns:
            Reward tensor of shape [B] if reward manager is available,
            None otherwise.
        """
        if self.reward_manager is None or not self.reward_manager.is_enabled:
            return None

        try:
            return self.reward_manager.compute_rewards(observations, task_descriptions)
        except Exception as e:
            import traceback
            logger.warning(f"Reward computation failed: {e}\n{traceback.format_exc()}")
            return None

    def _save_resnet_debug_samples(
        self,
        observations: dict[str, Any],
        model_rewards: torch.Tensor,
    ) -> None:
        """Save one success and one failure sample from ResNet predictions for debugging.
        
        Saves one sample per step (overwrites previous to keep only latest).
        Args:
            observations: Dict with 'images' or 'main_images' key.
            model_rewards: Tensor of model reward predictions.
        """
        if not hasattr(self, "_debug_step_count"):
            self._debug_step_count = 0
        
        # Increment step counter
        self._debug_step_count += 1
        
        images = observations.get("images")
        if images is None:
            images = observations.get("main_images")
        if images is None:
            return
        
        # Threshold for success (default 0.5)
        threshold = float(self.cfg.get("reward", {}).get("resnet", {}).get("threshold", 0.5))
        
        # Ensure model_rewards is 1D
        rewards_1d = model_rewards.view(-1)
        success_mask = rewards_1d > threshold
        failure_mask = rewards_1d <= threshold
        
        import os
        from PIL import Image
        import numpy as np
        
        # Save to project root logs/resnet_debug
        log_path = self.cfg.get("runner", {}).get("logger", {}).get("log_path", None)
        
        if log_path:
            if "${oc.env:EMBODIED_PATH}" in log_path:
                embodied_path = os.environ.get("EMBODIED_PATH", "")
                if embodied_path:
                    log_path = log_path.replace("${oc.env:EMBODIED_PATH}", embodied_path)
            
            if os.path.isabs(log_path) and "logs" in log_path:
                parts = log_path.split(os.sep)
                if "logs" in parts:
                    logs_idx = parts.index("logs")
                    project_root = os.sep.join(parts[:logs_idx])
                    save_dir = os.path.join(project_root, "logs", "resnet_debug")
                else:
                    save_dir = os.path.join(log_path, "resnet_debug")
            elif log_path.startswith("../"):
                embodied_path = os.environ.get("EMBODIED_PATH", "")
                if embodied_path:
                    project_root = os.path.dirname(os.path.dirname(embodied_path))
                    save_dir = os.path.join(project_root, "logs", "resnet_debug")
                else:
                    save_dir = os.path.join(os.getcwd(), "logs", "resnet_debug")
            else:
                save_dir = os.path.join(log_path, "resnet_debug")
        else:
            embodied_path = os.environ.get("EMBODIED_PATH", "")
            if embodied_path:
                project_root = os.path.dirname(os.path.dirname(embodied_path))
                save_dir = os.path.join(project_root, "logs", "resnet_debug")
            else:
                save_dir = os.path.join(os.getcwd(), "logs", "resnet_debug")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save one success sample (overwrite to keep latest)
        has_success = bool(success_mask.any())
        has_failure = bool(failure_mask.any())
        
        step = self._debug_step_count
        
        if has_success:
            idx = int(success_mask.nonzero(as_tuple=True)[0][0])
            prob = float(rewards_1d[idx])
            img = images[idx].cpu().numpy()
            if img.shape[-1] != 3:  # NCHW -> NHWC
                img = np.transpose(img, (1, 2, 0))
            if float(img.max()) <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            # Overwrite single file to keep latest
            Image.fromarray(img).save(f"{save_dir}/latest_success_step{step}_p{prob:.3f}.png")
            logger.info(f"Saved ResNet success sample (step={step}, p={prob:.3f}) to {save_dir}")
        
        # Save one failure sample (overwrite to keep latest)
        if has_failure:
            idx = int(failure_mask.nonzero(as_tuple=True)[0][0])
            prob = float(rewards_1d[idx])
            img = images[idx].cpu().numpy()
            if img.shape[-1] != 3:  # NCHW -> NHWC
                img = np.transpose(img, (1, 2, 0))
            if float(img.max()) <= 1.0:
                img = (img * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)
            # Overwrite single file to keep latest
            Image.fromarray(img).save(f"{save_dir}/latest_failure_step{step}_p{prob:.3f}.png")
            logger.info(f"Saved ResNet failure sample (step={step}, p={prob:.3f}) to {save_dir}")

    def load_checkpoint(self, load_path):
        model_dict = torch.load(load_path)
        self.hf_model.load_state_dict(model_dict)

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
            "use_cache": True,
        }

        self._eval_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

    def predict(self, env_obs, mode="train"):
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.CNN_POLICY,
        ]:
            kwargs = {"mode": mode}

        kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        with torch.no_grad():
            actions, result = self.hf_model.predict_action_batch(
                env_obs=env_obs,
                **kwargs,
            )

        return actions, result

    def get_dones_and_rewards(
        self,
        env_output: dict[str, torch.Tensor],
        extracted_obs: dict[str, Any],
        raw_obs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, dict[str, Any] | None]:
        """
        Get dones and rewards from environment batch, handling auto_reset if needed.

        Args:
            env_output: Environment batch containing dones, rewards, and optionally final_obs
            extracted_obs: Preprocessed observations from the policy model
            raw_obs: Raw observations from environment (for reward model, contains images)

        Returns:
            Tuple of (dones, rewards, real_extracted_obs). dones and rewards are tensors.
        """
        # First step: no rewards yet, only dones
        real_extracted_obs = None
        if env_output["rewards"] is None:
            if hasattr(self.hf_model, "q_head"):
                real_extracted_obs = init_real_obs(extracted_obs)
            return (
                env_output["dones"].bool().cpu().contiguous(),
                None,
                real_extracted_obs,
            )

        dones = env_output["dones"].bool().cpu().contiguous()
        rewards = env_output["rewards"].cpu().contiguous()

        # Compute model-based reward if reward manager is available
        # Uses raw_obs which contains images for ResNet/VLM reward models
        # Note: In "episode_end" mode, images are only available when episode ends
        if self.reward_manager is not None and self.reward_manager.is_enabled:
            reward_mode = self.cfg.get("reward", {}).get("mode", "replace")
            render_mode = self.cfg.get("env", {}).get("train", {}).get("reward_render_mode", "always")
            
            # In episode_end mode with replace: sparse reward (0 for non-terminal, ResNet for terminal)
            if render_mode == "episode_end" and reward_mode == "replace":
                # Zero out non-terminal rewards (sparse reward setting)
                rewards = torch.zeros_like(rewards)
            
            obs_for_reward = raw_obs if raw_obs is not None else env_output.get("obs")
            has_images = (obs_for_reward is not None and 
                         (obs_for_reward.get("images") is not None or 
                          obs_for_reward.get("main_images") is not None))
            
            if has_images:
                if render_mode == "episode_end" and dones.any():
                    # Episode-end mode: only compute reward for done envs
                    done_mask = dones.squeeze(-1) if dones.dim() > 1 else dones
                    
                    # Use final_obs for done envs (before auto_reset)
                    final_obs = env_output.get("final_obs")
                    if final_obs is not None:
                        images = final_obs.get("images") or final_obs.get("main_images")
                    else:
                        images = obs_for_reward.get("images") or obs_for_reward.get("main_images")
                    
                    if images is not None and done_mask.sum() > 0:
                        done_images = images[done_mask]
                        done_obs = {"images": done_images}
                        done_rewards = self.compute_model_reward(done_obs)
                        if done_rewards is not None:
                            model_rewards = torch.zeros(images.shape[0], device=done_rewards.device)
                            model_rewards[done_mask] = done_rewards
                            
                            if reward_mode == "replace":
                                rewards[done_mask] = model_rewards[done_mask].cpu().unsqueeze(-1)
                            elif reward_mode == "add":
                                rewards[done_mask] = rewards[done_mask] + model_rewards[done_mask].cpu().unsqueeze(-1)
                else:
                    # Always mode: compute reward for all envs every step
                    model_rewards = self.compute_model_reward(obs_for_reward)
                    if model_rewards is not None:
                        if reward_mode == "replace":
                            rewards = model_rewards.cpu().unsqueeze(-1).expand_as(rewards).contiguous()
                        elif reward_mode == "add":
                            rewards = rewards + model_rewards.cpu().unsqueeze(-1).expand_as(rewards)

        # Handle auto_reset: add bootstrap value to rewards for done episodes
        # Note: currently this is not correct for chunk-size>1 with partial reset
        if dones.any() and self.cfg.env.train.auto_reset:
            if hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head"):
                final_obs = env_output["final_obs"]
                with torch.no_grad():
                    final_extracted_obs = self.hf_model.preprocess_env_obs(final_obs)
                    if hasattr(self.hf_model, "q_head"):
                        real_extracted_obs = init_real_obs(final_extracted_obs)
                    actions, result = self.predict(final_extracted_obs)
                    if "prev_values" in result:
                        _final_values = result["prev_values"]
                    else:
                        _final_values = torch.zeros_like(actions[:, 0])
                final_values = torch.zeros_like(_final_values[:, 0])  # [bsz, ]
                last_step_dones = dones[:, -1]  # [bsz, ]

                final_values[last_step_dones] = _final_values[:, 0][last_step_dones]

                # Add bootstrap value to the last step of done episodes
                rewards[:, -1] += self.cfg.algorithm.gamma * final_values.cpu()

        if real_extracted_obs is None and hasattr(self.hf_model, "q_head"):
            real_extracted_obs = init_real_obs(extracted_obs)
        return dones, rewards, real_extracted_obs

    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""
        param_state_dict = await self.recv(
            self.actor_group_name, src_rank=self.actor_weight_src_rank, async_op=True
        ).async_wait()

        self.hf_model.load_state_dict(param_state_dict)
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

    def update_intervene_actions(self, env_output, forward_inputs):
        intervene_actions = env_output["intervene_actions"]
        intervene_flags = env_output["intervene_flags"]
        if intervene_actions is not None:
            if "action" in forward_inputs:
                policy_action = forward_inputs["action"].to(intervene_actions.device)
                policy_action = policy_action.reshape(
                    policy_action.shape[0], self.hf_model.num_action_chunks, -1
                )
                intervene_actions = intervene_actions.reshape(
                    intervene_actions.shape[0], self.hf_model.num_action_chunks, -1
                )
                action = intervene_actions * intervene_flags[
                    ..., None
                ] + policy_action * (~intervene_flags[..., None])
                action = action.reshape(action.shape[0], -1)
                forward_inputs["action"] = action
            else:
                raise NotImplementedError(f"{forward_inputs.keys()=}")
        return forward_inputs

    async def generate(
        self, input_channel: Channel, output_channel: Channel, actor_channel: Channel
    ):
        if self.enable_offload:
            self.reload_model()

        self.buffer_list = [
            EmbodiedRolloutResult(rollout_epoch=self.cfg.algorithm.rollout_epoch)
            for _ in range(self.num_pipeline_stages)
        ]

        n_chunk_steps = (
            self.cfg.env.train.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )

        for _ in tqdm(
            range(self.cfg.algorithm.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            last_extracted_obs = [None for i in range(self.num_pipeline_stages)]
            last_forward_inputs = [
                None for i in range(self.num_pipeline_stages)
            ]  # save actions

            for _ in range(n_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel)

                    if last_forward_inputs[stage_id] is not None:
                        last_forward_inputs[stage_id] = self.update_intervene_actions(
                            env_output, last_forward_inputs[stage_id]
                        )

                    raw_obs = env_output["obs"]
                    extracted_obs = self.hf_model.preprocess_env_obs(raw_obs)
                    dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                        env_output, extracted_obs, raw_obs
                    )

                    # Collect data for reward model training if enabled
                    if self.data_collector is not None:
                        prev_total = self.data_collector.success_count + self.data_collector.failure_count
                        # In ManiSkill:
                        # - dones = terminations | truncations (episode ended)
                        # - infos["success"] = True when task actually succeeded
                        # IMPORTANT: After auto_reset, current obs is from NEW episode!
                        # Must use final_obs for success frames, current obs for failure frames
                        infos = env_output["infos"] if "infos" in env_output else None
                        if infos is None or "success" not in infos:
                            raise RuntimeError(
                                "infos['success'] is required for reward data collection! "
                                "Make sure env_worker passes infos with 'success' key."
                            )
                        # Real success = grasp AND success (must hold object at goal)
                        successes = infos["success"]
                        if "is_grasped" in infos:
                            successes = successes & infos["is_grasped"]
                        elif "grasp" in infos:
                            successes = successes & infos["grasp"]
                        episode_dones = env_output["dones"]  # For resetting per-episode counters
                        
                        # CRITICAL: Use final_obs for success frames (before auto_reset)
                        # Current raw_obs is AFTER reset, so success frames would have wrong images!
                        final_obs = env_output.get("final_obs")
                        obs_for_collection = raw_obs  # Default to current obs
                        
                        # If there are success frames and final_obs exists, we need to handle them separately
                        if successes.any() and final_obs is not None:
                            # Collect success frames from final_obs (pre-reset images)
                            collection_complete = self.collect_reward_data(
                                raw_obs=final_obs,
                                terminations=None,
                                successes=successes,
                                dones=episode_dones,
                                goal_positions_2d=None,
                            )
                        else:
                            # No success this step, use current obs for failure frames
                            goal_positions = None
                            if self.cfg.get("reward_data_collection", {}).get("draw_goal_marker", False):
                                goal_positions = self.extract_goal_positions_2d(raw_obs)
                            
                            collection_complete = self.collect_reward_data(
                                raw_obs=raw_obs,
                                terminations=None,
                                successes=successes,
                                dones=episode_dones,
                                goal_positions_2d=goal_positions,
                            )
                        # Log progress when new samples collected (every 100 samples)
                        curr_total = self.data_collector.success_count + self.data_collector.failure_count
                        if curr_total > prev_total and (curr_total // 100) > (prev_total // 100):
                            stats = self.data_collector.get_statistics()
                            logger.info(
                                f"Collected: {stats['success_count']}/{self.data_collector.target_success} success, "
                                f"{stats['failure_count']}/{self.data_collector.target_failure} failure"
                            )

                    actions, result = self.predict(extracted_obs)
                    chunk_step_result = ChunkStepResult(
                        prev_logprobs=result["prev_logprobs"],
                        prev_values=result["prev_values"],
                        dones=dones,
                        truncations=env_output["truncations"],
                        terminations=env_output["terminations"],
                        rewards=rewards,  # the first step is reset step, reward is none, which will not be appended to the buffer
                        forward_inputs=last_forward_inputs[stage_id],
                    )
                    self.buffer_list[stage_id].append_result(chunk_step_result)
                    if last_extracted_obs[stage_id] is not None and hasattr(
                        self.hf_model, "q_head"
                    ):
                        self.buffer_list[stage_id].add_transition(
                            last_extracted_obs[stage_id], real_extracted_obs
                        )
                    last_extracted_obs[stage_id] = extracted_obs
                    last_forward_inputs[stage_id] = result["forward_inputs"]

                    self.send_chunk_actions(output_channel, actions)

            for stage_id in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                last_forward_inputs[stage_id] = self.update_intervene_actions(
                    env_output, last_forward_inputs[stage_id]
                )

                raw_obs = env_output["obs"]
                extracted_obs = self.hf_model.preprocess_env_obs(raw_obs)
                # Get dones and rewards from environment batch (final step of epoch)
                dones, rewards, real_extracted_obs = self.get_dones_and_rewards(
                    env_output, extracted_obs, raw_obs
                )
                self.buffer_list[stage_id].dones.append(dones)
                self.buffer_list[stage_id].truncations.append(env_output["truncations"])
                self.buffer_list[stage_id].terminations.append(
                    env_output["terminations"]
                )
                self.buffer_list[stage_id].rewards.append(rewards)
                self.buffer_list[stage_id].forward_inputs.append(
                    put_tensor_device(last_forward_inputs[stage_id], "cpu")
                )

                with self.worker_timer():
                    actions, result = self.predict(extracted_obs)
                # For the final step, we only need prev_values for bootstrapping
                # This is a special case that doesn't create a full ChunkStepResult
                if "prev_values" in result:
                    self.buffer_list[stage_id].prev_values.append(
                        result["prev_values"].cpu().contiguous()
                    )
                if hasattr(self.hf_model, "q_head"):
                    self.buffer_list[stage_id].add_transition(
                        last_extracted_obs[stage_id], real_extracted_obs
                    )

        for i in range(self.num_pipeline_stages):
            self.send_rollout_batch(actor_channel, i)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()

        n_chunk_steps = (
            self.cfg.env.eval.max_steps_per_rollout_epoch
            // self.cfg.actor.model.num_action_chunks
        )
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(n_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    extracted_obs = self.hf_model.preprocess_env_obs(env_output["obs"])
                    actions, _ = self.predict(extracted_obs, mode="eval")
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        self.hf_model = self.hf_model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()

    def reload_model(self):
        self.hf_model = self.hf_model.to(self.device)

    async def recv_env_output(
        self, input_channel: Channel, mode="train"
    ) -> dict[str, torch.Tensor]:
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        # Use asyncio so that it can run alongside async weight syncing
        env_output = await input_channel.get(
            key=f"{self._rank}_{mode}", async_op=True
        ).async_wait()
        return env_output

    def send_chunk_actions(self, output_channel: Channel, chunk_actions, mode="train"):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        output_channel.put(
            item=chunk_actions, key=f"{self._rank}_{mode}", async_op=True
        )

    def send_rollout_batch(self, actor_channel: Channel, stage_id: int):
        # send rollout_batch to actor
        split_num = self.get_actor_split_num()
        splitted_rollout_result = self.buffer_list[stage_id].to_splitted_dict(split_num)
        for i in range(split_num):
            actor_channel.put(item=splitted_rollout_result[i], async_op=True)

    def get_actor_split_num(self):
        send_num = self.placement.get_world_size("rollout") * self.num_pipeline_stages
        recv_num = self.placement.get_world_size("actor")
        split_num = compute_split_num(recv_num, send_num)
        return split_num

    def set_global_step(self, global_step):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
