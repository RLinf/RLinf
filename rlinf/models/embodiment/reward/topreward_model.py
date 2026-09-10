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

"""A frozen VLM as a world-model reward, read off one token's log-probability.

The prompt asserts that the task was completed rather than asking whether it was: an
invented "is the task complete:" wording scored an unrelated instruction higher than the
correct one on the same frames. Needs transformers >= 4.57 for Qwen3-VL.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from rlinf.models.embodiment.reward.qwen3vl_rocm_patch import patch_vision_patch_embed

PROMPT_PREFIX = (
    "The above video shows a robot manipulation trajectory that completes the "
    "following task: "
)
PROMPT_SUFFIX = (
    "{instruction} Decide whether the above statement is True or not. The answer is:"
)
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"


class TOPRewardModel(nn.Module):
    """Score generated frames with a frozen VLM, no training and no generation."""

    def __init__(
        self,
        model_path: str,
        chunk: int,
        success_prob_threshold: float = 0.46,
        window_frames: int = 16,
        fps: float = 2.0,
        attn_implementation: str = "eager",
        dtype: str = "bfloat16",
    ):
        super().__init__()
        self.chunk = int(chunk)
        self.threshold = float(success_prob_threshold)
        self.window_frames = int(window_frames)
        self.fps = float(fps)
        self._history: dict[int, np.ndarray] = {}

        from transformers import AutoProcessor

        try:
            from transformers import AutoModelForImageTextToText as AutoVLM
        except ImportError:
            from transformers import AutoModelForVision2Seq as AutoVLM

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoVLM.from_pretrained(
            model_path,
            dtype=getattr(torch, dtype),
            attn_implementation=attn_implementation,
        )
        patch_vision_patch_embed(self.model)
        self.model.requires_grad_(False)

        true_ids = self.processor.tokenizer.encode(" True", add_special_tokens=False)
        if len(true_ids) != 1:
            raise ValueError(
                f"' True' tokenizes to {len(true_ids)} tokens for {model_path}; "
                "the score reads a single next token"
            )
        self.true_id = true_ids[0]

    def reset_history(self, env_idx: Optional[list[int]] = None) -> None:
        """Drop cached frames, so a scoring window never spans two episodes."""
        if env_idx is None:
            self._history.clear()
            return
        for slot in env_idx:
            self._history.pop(int(slot), None)

    def _to_pil(self, frames_chw: np.ndarray):
        from PIL import Image

        if len(frames_chw) > self.window_frames:
            keep = np.linspace(0, len(frames_chw) - 1, self.window_frames).astype(int)
            frames_chw = frames_chw[keep]
        return [
            Image.fromarray(np.transpose(frame, (1, 2, 0))[..., :3]).convert("RGB")
            for frame in frames_chw
        ]

    @torch.no_grad()
    def _score(self, frames_chw: np.ndarray, instruction: str) -> float:
        text = (
            VIDEO_PLACEHOLDER
            + PROMPT_PREFIX
            + PROMPT_SUFFIX.format(instruction=instruction)
        )
        pil = self._to_pil(frames_chw)
        # Without video_metadata the processor assumes 24 fps, reads the clip as 0.67 s
        # and hands the model 4 of the 16 frames (grid_thw t=2 rather than t=8).
        inputs = self.processor(
            text=[text],
            videos=[pil],
            fps=self.fps,
            video_metadata=[
                {
                    "fps": self.fps,
                    "total_num_frames": len(pil),
                    "duration": len(pil) / self.fps,
                }
            ],
            return_tensors="pt",
        ).to(self.model.device)
        logits = self.model(**inputs).logits[0, -1].float()
        return torch.log_softmax(logits, dim=-1)[self.true_id].item()

    @torch.no_grad()
    def predict_rew(
        self, images: torch.Tensor, instructions: list[str]
    ) -> torch.Tensor:
        """Label each frame of ``[num_envs * chunk, 3, H, W]`` in ``[-1, 1]`` with 0/1.

        The label is what the ResNet reward model's ``round()`` returns, so it drives
        ``terminations`` and the loss mask unchanged.
        """
        num_envs = images.shape[0] // self.chunk
        frames = (
            (images.detach().float().clamp(-1.0, 1.0) + 1.0)
            .mul_(127.5)
            .to(torch.uint8)
            .cpu()
            .numpy()
            .reshape(num_envs, self.chunk, *images.shape[1:])
        )

        labels = torch.zeros(num_envs, dtype=images.dtype, device=images.device)
        for env in range(num_envs):
            chunk = frames[env]
            # A chunk scored on its own loses the separation between succeeding and
            # failing episodes, so the call also sees how the episode got here.
            past = self._history.get(env)
            window = chunk if past is None else np.concatenate([past, chunk])
            # Carrying the tail of the window rather than of this chunk lets a
            # window_frames above 2 * chunk fill up over successive calls.
            self._history[env] = window[-max(1, self.window_frames - self.chunk) :]

            logp = self._score(
                window[-self.window_frames :], instructions[env * self.chunk]
            )
            labels[env] = float(np.exp(logp) >= self.threshold)

        return labels.repeat_interleave(self.chunk)
