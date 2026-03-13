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


from diffsynth.pipelines.wan_video_action_multiview import ActionWanVideoMultiviewPipeline, ModelConfig
from diffsynth import save_video, VideoData
from collections import deque
from typing import Any, Optional
import imageio
import numpy as np
import os
import pdb
import torchvision
import torch
from PIL import Image
from gymnasium import spaces
from rlinf.models.worldmodel.base_fake_model import BaseFakeModelInference
# 导入工具函数
from rlinf.models.worldmodel.utils import euler2rotm, rotm2euler, generate_video_path, remote_server_reward
from omegaconf import OmegaConf
import torch.nn.functional as F
from safetensors.torch import load_file
from torchvision import transforms as T
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
# 导入 IRASim 模型
import sys
sys.path.append("/mnt/seed17/001688/qian.ren/qian.ren/diffsynth-studio-main")


class WanInference(BaseFakeModelInference):
    """
    This class implements the world model inference using a fake model,
    the purpose is to define the interaction logic with the env interface.
    """

    def __init__(self, cfg, dataset: Any, device: Any):
        """
        Initializes the world model backend.
        Args:
            cfg: Configuration dictionary.
            dataset: The dataset used by the world model.
            device: The device to run the model on.
        """

        self.cfg = cfg
        self.dataset = dataset
        self.device = device

        action_dim = self.dataset.action_dim
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )
        self.camera_names = self.dataset.camera_names
        self.max_episode_steps = self.cfg.max_episode_steps
        self.current_step = 0
        self.batch_size = self.cfg.batch_size
        self.num_prompt_frames = self.cfg.num_prompt_frames
        # 加载模型和创建pipeline
        self._init_episodes_structure()
        self._load_model()
        if self.cfg.vllm_base_url is not None:
            self._load_reward_model()

    def _load_model(self):
        # 加载配置文件
        config = self.cfg.model_cfg
        pipe = ActionWanVideoMultiviewPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device=self.device,
            model_configs=[
                ModelConfig(local_model_path=config.model_folder, model_id="Wan2.1-Fun-V1.1-1.3B-InP",
                            origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
                ModelConfig(local_model_path=config.model_folder, model_id="Wan2.1-Fun-V1.1-1.3B-InP",
                            origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
                ModelConfig(local_model_path=config.model_folder, model_id="Wan2.1-Fun-V1.1-1.3B-InP",
                            origin_file_pattern="models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth", offload_device="cpu"),
            ],
            tokenizer_config=ModelConfig(local_model_path=config.model_folder,
                                         model_id="Wan2.1-T2V-1.3B", origin_file_pattern="google/*"),
            checkpoint=config.checkpoint_file,
        )
        # action_dit_state_dict = load_file(config.checkpoint_file)
        # pipe.dit.load_state_dict(action_dit_state_dict, strict=True)
        pipe.enable_vram_management()
        self.videogen_pipeline = pipe

    def _load_reward_model(self):
        model_path = self.cfg.vllm_model
        self.reward_model = AutoModelForImageTextToText.from_pretrained(
            model_path, torch_dtype="auto", device_map=self.device
        )

        self.reward_processor = AutoProcessor.from_pretrained(model_path)

    def _compute_reward(self, task_text: str, video_path: str) -> str:
        """
        使用本地 Qwen-VL 模型评估视频。
        返回: "terminate", "Success", 或 "Failure"
        """

        # 2. 构建消息格式，与远程调用保持一致
        text_template = (
            f"You will be shown a video trajectory. First, evaluate the video quality and content:\n"
            f"1. If the video has severe quality issues (blurry, distorted, unrealistic, or clearly degraded visuals), "
            f"   output: terminate\n"
            f"2. If the video quality is acceptable, then evaluate if the robot succeeds at {task_text}.\n"
            f"   - If the robot clearly accomplishes the task, output: Success\n"
            f"   - If the robot fails to accomplish the task, output: Failure\n\n"
            f"Output strictly one of these three options: terminate, Success, or Failure\n"
            f"Do not provide explanations or additional text."
        )

        # 构建 content 列表，交替插入图片和文本
        content = []
        content.append({
            "type": "video",
            "video": video_path,
        })
        content.append({"type": "text", "text": text_template})

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # 3. 准备模型输入
        text_prompt = self.reward_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理图像输入
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.reward_processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # 移动到模型所在设备
        inputs = inputs.to(self.device)

        # 4. 生成回复
        with torch.no_grad():
            generated_ids = self.reward_model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.0,  # 贪婪解码
                do_sample=False,
            )

        # 5. 解码输出
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = self.reward_processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        )[0]

        return output_text.strip()

    def _init_episodes_structure(self):
        """
        Initializes the episodes structure.
        episodes_history: pay attention that conditional frames and generated frames have different data structures.
        """

        self.episodes = [None] * self.batch_size
        self.episodes_latest_frames: deque = deque(
            [
                [{} for _ in range(self.batch_size)]
                for _ in range(self.cfg.action_length)
            ],
            maxlen=self.cfg.action_length,
        )
        self.episodes_history = [
            deque(maxlen=self.num_prompt_frames + self.max_episode_steps)
            for _ in range(self.batch_size)
        ]

    def _get_latest_obs_from_deques(self) -> dict[str, Any]:
        """
        Retrieves the latest observations from the episode deques for all batch environments.

        This method processes the latest frames from the episodes_latest_frames deque,
        organizing camera images and state information into a structured format suitable
        for model inference. It handles multi-camera setups and maintains task descriptions.

        Returns:
            A list of dictionaries, each containing:
                - images_and_states: Dictionary with camera images and state tensors
                - task_descriptions: List of task descriptions for each batch element
        """

        return_obs = []
        for episodes_latest_frame in self.episodes_latest_frames:
            if not episodes_latest_frame:
                continue
            for obs in episodes_latest_frame:
                assert obs, "episode_latest_frame cannot empty"

            images_and_states = {}
            for camera_name in self.camera_names:
                imgs = torch.stack(
                    [obs[f"{camera_name}"] for obs in episodes_latest_frame], dim=0
                )
                imgs = F.interpolate(imgs, (256, 256)).permute(0, 3, 2, 1)
                images_and_states[camera_name] = imgs

            states = torch.stack(
                [obs["observation.state"] for obs in episodes_latest_frame], dim=0
            )

            images_and_states["states"] = states
            task_descriptions = [obs["task"] for obs in episodes_latest_frame]
            # 同样检查 image
            obs = {
                "images_0": images_and_states[self.camera_names[0]],
                "images_1": images_and_states[self.camera_names[1]],
                "images_2": images_and_states[self.camera_names[2]],
                "images_and_states": images_and_states,
                "states": images_and_states['states'],
                "task_descriptions": task_descriptions,
            }
            return_obs.append(obs)

        return return_obs

    def _init_reset_state_ids(self, seed: int):
        """
        Initializes the reset state IDs for batch environment initialization.

        This method creates a random number generator and generates random episode IDs
        from the dataset for resetting the environments. It ensures deterministic
        behavior when a seed is provided.

        Args:
            seed: The random seed for deterministic episode selection.
        """

        self._generator = torch.Generator()
        self._generator.manual_seed(seed)

        if self.dataset.select_episodes is not None:
            length = len(self.dataset.select_episodes)
            self._reset_state_ids = [self.dataset.select_episodes[i] for i in torch.randint(
                0, length, (self.batch_size,), generator=self._generator
            )]
        else:
            self._reset_state_ids = torch.randint(
                0, len(self.dataset), (self.batch_size,), generator=self._generator
            )

    def _apply_action_to_state(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """
        将动作应用到当前状态，计算新状态。

        注意：action是相对变化 [dx, dy, dz, droll, dpitch, dyaw, gripper]
        current_state是绝对状态 [x, y, z, roll, pitch, yaw, gripper]

        Args:
            current_state: 当前绝对状态 [x, y, z, roll, pitch, yaw, gripper]
            action: 相对动作 [dx, dy, dz, droll, dpitch, dyaw, gripper_action]

        Returns:
            新绝对状态 [x', y', z', roll', pitch', yaw', gripper']
        """
        # 解析当前状态
        current_xyz = current_state[:3]  # 绝对位置
        current_rpy = current_state[3:6]  # 绝对欧拉角
        pad = current_state[6]  # 绝对欧拉角
        # 解析动作
        rel_xyz = action[:3]  # 相对位置变化
        rel_rpy = action[3:6]  # 相对欧拉角变化

        gripper_action = action[6]  # 夹爪动作

        # 1. 更新位置：相对变化在当前坐标系下，需要转换到世界坐标系
        current_rotm = euler2rotm(current_rpy)  # 当前朝向的旋转矩阵
        world_delta_xyz = current_rotm @ rel_xyz  # 转换到世界坐标系
        new_xyz = current_xyz + world_delta_xyz  # 新绝对位置

        # 2. 更新朝向：相对欧拉角转换为旋转矩阵，然后与当前旋转矩阵相乘
        rel_rotm = euler2rotm(rel_rpy)  # 相对旋转的旋转矩阵
        new_rotm = current_rotm @ rel_rotm  # 新旋转矩阵
        new_rpy = rotm2euler(new_rotm)  # 新欧拉角

        # 3. 更新夹爪：动作中的gripper通常表示目标状态
        # 如果gripper_action > 0.5，则为打开(1.0)，否则为关闭(0.0)
        new_gripper = 1.0 if gripper_action > 0.5 else 0.0
        # 返回新状态
        new_state = np.array([new_xyz[0], new_xyz[1], new_xyz[2],
                             new_rpy[0], new_rpy[1], new_rpy[2], pad, new_gripper])
        return new_state

    def _process_generated_video(self, videos: torch.Tensor, actions: torch.Tensor, latents: Optional[torch.Tensor] = None) -> list[list[dict[str, Any]]]:
        """
        处理生成的视频，将其转换为观测格式。
        所有观测帧都使用完成所有动作后的最终状态。
        """
        B = self.batch_size
        N = len(self.camera_names)

        return_obs_list = []
        frame_per_step = 1
        for i in range(B):
            latest_obs = self.episodes_latest_frames[-1][i]

            # 获取初始状态
            current_state = latest_obs["observation.state"].cpu().numpy()

            # 获取动作序列
            # [chunk_action, action_dim]
            action_sequence = actions[i].cpu().numpy()

            # 应用所有动作，得到最终状态
            final_state = current_state.copy()
            # 假设 action 和输出的视频是均匀分布的
            for step, action in enumerate(action_sequence):
                if step >= len(return_obs_list):
                    return_obs_list.append([])
                frame_idx = int((step + 1) * frame_per_step)
                new_obs = {}
                # 为每个摄像头分配生成的图像
                for camera_name in self.camera_names:
                    new_obs[camera_name] = torch.from_numpy(
                        np.array(videos[i][camera_name][frame_idx]).transpose(2, 0, 1))
                final_state = self._apply_action_to_state(final_state, action)
                # 所有帧都使用相同的最终状态
                new_obs["observation.state"] = torch.from_numpy(
                    final_state).to(self.device)
                new_obs["task"] = latest_obs["task"]
                return_obs_list[step].append(new_obs)
        return return_obs_list

    def _infer_next_frames(self, actions: torch.Tensor) -> list[dict[str, Any]]:
        """
        使用视频生成模型生成下一帧。

        Args:
            actions: 要执行的动作，形状为 [batch_size, num_action,action_dim]

        Returns:
            新的观测列表
        """
        B = self.batch_size
        N = len(self.camera_names)
        with torch.no_grad():
            video_batch = []
            for i in range(B):
                latest_obs = self.episodes_latest_frames[-1][i]
                images = [Image.fromarray(latest_obs[camera_key].numpy().transpose(
                    1, 2, 0)) for camera_key in self.camera_names]
                output = self.videogen_pipeline(
                    prompt="",
                    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
                    action=actions[i],
                    input_image=images[0],
                    input_image1=images[1],
                    input_image2=images[2],
                    num_frames=self.cfg.model_cfg.num_frames,
                    height=self.cfg.model_cfg.video_size[0],
                    width=self.cfg.model_cfg.video_size[1],
                    num_inference_steps=self.cfg.model_cfg.infer_num_sampling_steps,
                )
                videos = {}
                for i, camera_key in enumerate(self.camera_names):
                    videos[camera_key] = output[i]
                video_batch.append(videos)
        # 处理生成的视频为观测格式，传入动作用于状态更新
        new_obs_list = self._process_generated_video(video_batch, actions)

        return new_obs_list, video_batch

    def _infer_next_rewards_and_terminate(
        self, videos: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算奖励并判断是否终止。

        逻辑流：
        1. 遍历 Batch 中的每个样本。
        2. 为每个样本的所有视角（N个）生成视频并请求奖励。
        3. 计算 N 个视角的平均奖励。
        4. 根据平均分和随机概率判定是否 Terminated。
        """
        os.makedirs(self.cfg.tmp_video, exist_ok=True)
        B = self.batch_size
        N = len(self.camera_names)

        rewards = torch.zeros(B, dtype=torch.float32, device=self.device)
        terminated = torch.zeros(B, dtype=torch.bool, device=self.device)
        info_list = []
        for i in range(B):
            batch_rewards_list = []
            latest_obs = self.episodes_latest_frames[-1][i]
            task_text = latest_obs["task"]
            info = {}
            # 对当前 Batch 的所有视角进行评价
            for camera_name in self.camera_names:
                video = videos[i][camera_name]
                # 重要：利用 generate_video_path 生成带唯一 ID 的路径
                # 这样可以确保远程服务器收到的每个视角文件都是唯一的
                unique_video_path = generate_video_path(self.cfg.tmp_video)
                save_video(video, unique_video_path, fps=8)
                if self.cfg.vllm_base_url is not None:
                    # 调用远程服务器，传入任务描述和唯一的视频路径
                    reward_text = remote_server_reward(task_text, unique_video_path,base_url=self.cfg.vllm_base_url)
                else:
                    # 调用本地模型
                    reward_text = self._compute_reward(
                        task_text, unique_video_path)

                # 解析逻辑：包含 success 记为 1.0
                cam_reward = 1.0 if 'success' in reward_text.lower() else 0.0
                info["success"] = ['success' in reward_text.lower()] * \
                    self.cfg.action_length
                info["fail"] = ['fail' in reward_text.lower()] * \
                    self.cfg.action_length
                terminated[i] = 'terminated' in reward_text.lower(
                ) or terminated[i]
                batch_rewards_list.append(cam_reward)
                # 获取奖励后删除临时文件
                if os.path.exists(unique_video_path):
                    os.remove(unique_video_path)

            info_list.append(info)

            # --- 计算平均奖励 ---
            if batch_rewards_list:
                rewards[i] = max(batch_rewards_list)

            # --- 终止逻辑 ---
            # 1. 奖励成功阈值判定
            success_terminated = rewards[i] > 0.5

            # 2.decayed 截断
            terminated[i] = success_terminated or terminated[i]

        return rewards, terminated, info_list

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[Any, dict[str, Any]]:
        """
        Resets the environment to its initial state.
        Args:
            seed: The random seed for the environment.
            options: Additional options for resetting the environment.
        Returns:
            A tuple containing the initial observation and a dictionary of info.
        """

        def _padding_or_truncate_start_items(start_items: list[dict[str, Any]]):
            """Padding or truncate the start_items to the gen_num_image_each_step."""
            if len(start_items) < self.cfg.action_length:
                start_items = start_items + [start_items[-1]] * (
                    self.cfg.action_length - len(start_items)
                )
            elif len(start_items) > self.cfg.action_length:
                start_items = start_items[-self.cfg.action_length:]
            return start_items

        if seed is None:
            seed = 0

        options = options or {}

        if "episode_id" not in options:
            self._init_reset_state_ids(seed)
            options["episode_id"] = self._reset_state_ids
        if "env_idx" in options:
            env_idx = options["env_idx"]
            episode_ids = options["episode_id"][: len(env_idx)]
            for i, episode_id in zip(env_idx, episode_ids):
                self.episodes[i] = self.dataset[int(episode_id)]
                for j in range(self.gen_num_image_each_step):
                    self.episodes_latest_frames[j][i] = {}
                self.episodes_history[i].clear()

                start_items = _padding_or_truncate_start_items(
                    self.episodes[i]["start_items"]
                )
                for j, frame in enumerate(start_items):
                    self.episodes_latest_frames[j][i] = frame
                for frame in self.episodes[i]["start_items"]:
                    self.episodes_history[i].append(frame)

            return self._get_latest_obs_from_deques(), {}

        self._init_episodes_structure()

        episode_ids = options["episode_id"]
        self.episodes = [self.dataset[int(episode_id)]
                         for episode_id in episode_ids]
        assert len(self.episodes) == self.batch_size

        for i, episode in enumerate(self.episodes):
            start_items = _padding_or_truncate_start_items(
                episode["start_items"])
            for j, frame in enumerate(start_items):
                self.episodes_latest_frames[j][i] = frame
            for frame in self.episodes[i]["start_items"]:
                self.episodes_history[i].append(frame)

        self.current_step = 0
        return self._get_latest_obs_from_deques(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """
        Takes a step in the environment for all batch elements.

        This method generates the next observations, calculates rewards, determines
        termination conditions, and updates the episode history for each batch element.
        It handles multiple frames per step and maintains proper episode tracking.

        Args:
            actions: The actions to take for each batch element.

        Returns:
            A tuple containing:
                - Latest observations from the updated deques
                - Stacked reward tensor for all generated frames
                - Stacked termination flags for all generated frames
                - Stacked truncation flags for all generated frames
                - List of info dictionaries for each generated frame
        """
        new_obs_list, videos = self._infer_next_frames(actions)

        reward, terminated, info_list = self._infer_next_rewards_and_terminate(
            videos)
        reward_list, terminated_list, truncated_list = [], [], []
        n_action = actions.shape[1]
        for i in range(n_action):
            self.episodes_latest_frames.append(new_obs_list[i])
            reward_list.append(reward)
            terminated_list.append(terminated)
            truncated_list.append(
                torch.zeros(self.batch_size, dtype=torch.bool,
                            device=self.device)
                if self.current_step <= self.max_episode_steps
                else torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
            )

            for j in range(self.batch_size):
                step_data = {
                    "observation": new_obs_list[i][j],
                    "action": actions[j],
                    "reward": reward_list[i][j].item(),
                    "terminated": terminated_list[i][j].item(),
                    "truncated": truncated_list[i][j].item(),
                }
                self.episodes_history[j].append(step_data)

            self.current_step += 1

        return (
            self._get_latest_obs_from_deques(),
            reward_list,
            terminated_list,
            truncated_list,
            info_list,
        )


if __name__ == "__main__":
    from rlinf.envs.worldmodel.dataset import LeRobotDatasetWrapper
    cfg = OmegaConf.load(
        "/mnt/seed17/001688/qian.ren/qian.ren/RLinf/logs/fast_wan-30step/tensorboard/config.yaml")

    dataset = LeRobotDatasetWrapper(
        **cfg.env.train.dataset_cfg
    )
    env_cfg = cfg.env.train
    device = 'musa'
    env = WanInference(env_cfg, dataset, device)

    options = {"episode_id": [0]}
    env.reset(options=options)
    # action = torch.tensor([-5.1146e-7, -1.2105e-8, -9.3151e-8,  2.5257e-07, -9.5845e-08,
    #     -5.6846e-08,  1.0000e+00])
    # actions = action.unsqueeze(0).unsqueeze(0).repeat(env_cfg.batch_size,15, 1)
    # env.step(actions)
    # env.step(actions)
