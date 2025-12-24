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


import torch


#daqi
import requests
import json
import re
import base64
import io
import numpy as np
from PIL import Image
#daqi




from omegaconf import DictConfig

from rlinf.algorithms.rewards import get_reward_class
from rlinf.data.io_struct import RolloutResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Channel, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement


class RewardWorker(Worker):
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self.cfg = cfg
        self.component_placement = placement
        self.tokenizer = hf_tokenizer(cfg.reward.tokenizer.tokenizer_model)
        self.total_batch_size_per_dp = (
            self.cfg.data.rollout_batch_size
            * self.cfg.algorithm.get("group_size", 1)
            // self._world_size
        )

    def init_worker(self):
        if self.cfg.reward.use_reward_model:
            
            #daqi，example for Qwen-VL
            self.api_url = "http://localhost:8000/v1/chat/completions"
            self.model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
            print(f"RewardWorker: Connected to API at {self.api_url}")
        else:
            self.reward = get_reward_class(self.cfg.reward.reward_type)(self.cfg.reward)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()
        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def compute_rewards(self, input_channel: Channel, output_channel: Channel):
        """Compute rewards.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
        """
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            rollout_result: RolloutResult = input_channel.get()
            recv_batch_size += rollout_result.num_sequence
            with self.worker_timer():
                if rollout_result.rewards is None:
                    if self.cfg.reward.use_reward_model:
                        with self.device_lock:
                         #   batch = rollout_result.to_actor_batch(
                         #       self.cfg.data.max_prompt_length,
                         #       self.cfg.actor.model.encoder_seq_length,
                         #       self.tokenizer.eos_token_id,
                         #   )
                            rollout_result.rewards = (
                                self.compute_batch_rewards_with_model(rollout_result)#batch->rollout_result
                            )
                    else:
                        rollout_result.rewards = self._compute_rule_based_rewards(
                            rollout_result
                        )

            output_channel.put(rollout_result)

        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )

    def _compute_rule_based_rewards(self, rollout_result: RolloutResult):
        # Decode only the generated tokens; response_ids are already the post-prompt tokens
        texts = rollout_result.response_texts
        if texts is None:
            texts = self.tokenizer.batch_decode(
                rollout_result.response_ids, skip_special_tokens=True
            )

        kwargs = {}
        if getattr(self.cfg.reward, "use_prompt", False):
            prompts = rollout_result.prompt_texts
            if prompts is None:
                prompts = self.tokenizer.batch_decode(
                    rollout_result.prompt_ids, skip_special_tokens=True
                )
            kwargs["prompts"] = prompts
        scores = self.reward.get_reward(texts, rollout_result.answers, **kwargs)
        return (
            torch.as_tensor(scores, dtype=torch.float, device=torch.device("cpu"))
            .view(-1, 1)
            .flatten()
        )


    #daqi
    # 参数从 batch 变成了 rollout_result
    def compute_batch_rewards_with_model(self, rollout_result: RolloutResult):
        rewards = []
        
        # 1. 尝试获取视频数据
        batch_videos = getattr(rollout_result, 'video_frames', [])
        
        # 2. 获取文本 Prompt
        batch_prompts = rollout_result.prompt_texts
        if batch_prompts is None:
             batch_prompts = self.tokenizer.batch_decode(
                rollout_result.prompt_ids, skip_special_tokens=True
            )

        # 3. 检查是否有视频
        if not batch_videos:
            print("[Warning] RewardWorker: No video frames found! Returning 0.0")
            return torch.zeros(len(batch_prompts), dtype=torch.float, device="cpu")

        # 4. 遍历 Batch 调用 API
        for i, frames in enumerate(batch_videos):
            task_text = batch_prompts[i]
            
            # 构造请求数据
            payload = self._build_api_payload(frames, task_text)
            
            # 发送请求并解析分数
            score = self._call_api_and_parse(payload)
            rewards.append(score)

        return torch.tensor(rewards, dtype=torch.float, device="cpu")

    #新增的辅助函数

    def _build_api_payload(self, frames, task_text):
        """构造 Qwen2.5-VL 格式的 API 请求"""
        content_list = []
        
        # 采样帧数,发给模型多少张图
        NUM_SAMPLES = 8 
        
        sampled_frames = self._sample_frames(frames, num_samples=NUM_SAMPLES)
        
        for frame in sampled_frames:
            b64_str = self._image_to_base64(frame)
            if b64_str:
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                })
        
        # prompt
        prompt = f"""
        Task: {task_text}
        The input is a video of a robot execution.
        Did the robot successfully complete the task?
        Output strictly in this format: [[SCORE]] where SCORE is 1.0 for success and -1.0 for failure.
        """
        content_list.append({"type": "text", "text": prompt})
        
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_list}],
            
            # 可修改
            "temperature": 0.0, 
            
            "max_tokens": 16 
        }

    def _call_api_and_parse(self, payload):
        """发送 HTTP 请求并解析结果"""
        try:
            # 超时时间 (秒)
            TIMEOUT_SECONDS = 60 
            
            resp = requests.post(self.api_url, json=payload, timeout=TIMEOUT_SECONDS)
            if resp.status_code != 200:
                print(f"Reward API Error: {resp.text}")
                return 0.0
                
            content = resp.json()['choices'][0]['message']['content']
            
            # 解析规则：如果模型输出格式变了，这里正则也要改
            match = re.search(r'\[\[([-+]?\d*\.?\d*)\]\]', content)
            if match:
                return float(match.group(1))
            
            # 简单兜底逻辑
            lower_content = content.lower()
            if "success" in lower_content: return 1.0
            if "fail" in lower_content: return -1.0
            
            return 0.0
        except Exception as e:
            print(f"Reward Logic Exception: {e}")
            return 0.0

    def _sample_frames(self, frames, num_samples=8):
        """均匀采样"""
        if not frames: return []
        if len(frames) <= num_samples: return frames
        
        indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
        if isinstance(frames, list):
            return [frames[i] for i in indices]
        return frames[indices]

    def _image_to_base64(self, img_array):
        """将 numpy array 转为 JPEG base64"""
        try:
            if hasattr(img_array, 'cpu'):
                img_array = img_array.cpu().numpy()
            
            if img_array.dtype != np.uint8:
                if img_array.max() <= 1.0:
                    img_array = (img_array * 255).astype(np.uint8)
                else:
                    img_array = img_array.astype(np.uint8)

            if img_array.ndim == 3 and img_array.shape[0] == 3 and img_array.shape[1] > 3:
                img_array = np.transpose(img_array, (1, 2, 0))

            img = Image.fromarray(img_array)
            buf = io.BytesIO()
            
            # 图片质量
            JPEG_QUALITY = 70 
            
            img.save(buf, format='JPEG', quality=JPEG_QUALITY)
            return base64.b64encode(buf.getvalue()).decode('utf-8')
        except Exception as e:
            print(f"Image encode error: {e}")
            return None
