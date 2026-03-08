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

"""TOPReward dense progress reward for embodied RL.

Computes a dense reward via True-token log-probability following the TOPReward
paper.  The prompt is constructed *without* a chat template: video frames +
prompt prefix are processed through ``apply_chat_template`` with
``add_generation_prompt=False``, the EOS is stripped, and the instruction suffix
+ "True" are appended raw.  The model then scores only the suffix tokens via
label masking.

The class receives the model and processor as constructor args — the
VLMPlannerWorker owns the model lifecycle.

Usage in YAML:
    reward:
      use_reward_model: False
      reward_type: top_reward
      reward_scale: 1.0
"""

import numpy as np
from omegaconf import DictConfig
from PIL import Image

_PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that "
    "completes the following task: "
)


class TOPReward:
    """Dense progress reward based on VLM True-token log-probability.

    Args:
        config: Reward config DictConfig.  Recognised keys:
            - ``reward_scale`` (float, default 1.0): multiplicative scale
              applied to the raw TOPReward delta.
            - ``top_reward_max_frames`` (int, default 16): max video frames.
        model: A HuggingFace causal VLM (e.g. Qwen-VL).  None when used as
            a registry-only entry.
        processor: Matching HuggingFace processor / tokenizer.
    """

    def __init__(self, config: DictConfig, model=None, processor=None):
        self.reward_scale = float(config.get("reward_scale", 1.0))
        self.max_frames = int(config.get("top_reward_max_frames", 16))
        self._model = model
        self._processor = processor

    def compute_score(
        self,
        frames: list[np.ndarray],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
    ) -> float:
        """Full TOPReward scoring: prompt build -> forward pass -> label-masked log-prob.

        Args:
            frames: List of uint8 RGB images ``(H, W, 3)`` representing the
                trajectory so far (most recent last).
            instruction: Task description string.
            reduction: How to aggregate per-token log-probs: ``"mean"`` or
                ``"sum"``.
            fps: Frames per second metadata for video input.

        Returns:
            Mean (or summed) log-probability of the instruction suffix + "True"
            tokens (a float, typically negative).
        """
        import torch
        import torch.nn.functional as F
        from qwen_vl_utils import process_vision_info

        pil_frames = [Image.fromarray(f.astype(np.uint8)) for f in frames]

        instruction_suffix = (
            f"{instruction} Decide whether the above statement is True or "
            "not. The answer is: True"
        )

        # Build messages with video + prompt prefix only (no instruction suffix).
        content = [
            {"type": "video", "video": pil_frames, "fps": fps},
            {"type": "text", "text": _PROMPT_PREFIX},
        ]
        user_messages = [{"role": "user", "content": content}]

        # Apply chat template WITHOUT generation prompt, then strip EOS.
        prompt_chat = self._processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=False
        )
        eos_token = self._processor.tokenizer.eos_token
        if eos_token is not None:
            prompt_chat = prompt_chat.split(eos_token)[0]

        # Append instruction suffix raw (no chat wrapping).
        full_text = f"{prompt_chat}{instruction_suffix}"

        image_inputs, video_inputs = process_vision_info(user_messages)
        video_kwargs = {}
        # Handle newer qwen_vl_utils that returns 3 values.
        try:
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                user_messages, return_video_metadata=True
            )
        except TypeError:
            pass

        device = next(self._model.parameters()).device
        inputs = self._processor(
            text=[full_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        ).to(device)

        labels = inputs["input_ids"].clone()

        # Mask all tokens except the last one — only score "True".
        # With the shifted-by-1 label convention the model predicts token i+1
        # from position i, so masking up to (seq_len - 1) leaves only the
        # final "True" token as the prediction target.
        prompt_length = inputs["input_ids"].shape[1] - 1
        labels[:, :prompt_length] = -100
        if "attention_mask" in inputs:
            labels = labels.masked_fill(inputs["attention_mask"] == 0, -100)

        self._model.eval()
        with torch.inference_mode():
            outputs = self._model(**inputs, labels=labels)

        # Compute per-token log-probabilities for the suffix.
        logits = outputs.logits[:, :-1, :]
        target_labels = labels[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        mask = target_labels != -100
        safe_targets = target_labels.masked_fill(~mask, 0)
        token_log_probs = log_probs.gather(-1, safe_targets.unsqueeze(-1)).squeeze(-1)
        masked_log_probs = token_log_probs[mask]

        if reduction == "sum":
            return masked_log_probs.sum().item()
        return masked_log_probs.mean().item()

    def get_reward(self, completions, answers, **kwargs) -> list[float]:
        """Not used — rewards come from env worker via VLM forward pass.

        Returns:
            Empty list.
        """
        return []
