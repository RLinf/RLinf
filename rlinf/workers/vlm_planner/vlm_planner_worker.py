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

"""VLM Planner Worker for stage-aware embodied RL.

This Ray actor runs on the Beaker GPU node and serves two roles:

1. **Subtask generation** - given recent robot observations (images) and a
   rolling text memory of past actions/observations, it generates the current
   subtask instruction (e.g. "pick up the red block").

2. **Subtask reward evaluation** - given the same context plus the completed
   subtask description, it decides whether the subtask succeeded (1.0) or
   failed (0.0).

The worker loads a local Qwen3.5 multimodal model (no external API required)
and is placed on a node with a GPU.

Architecture
~~~~~~~~~~~~
::

    EnvWorker (YAM node)
        │  every subtask_interval steps
        │  images + memory ──────────────────▶  VLMPlannerWorker (Beaker)
        │                                           │
        │  ◀── new subtask text ─────────────────  │  get_next_subtask()
        │                                           │
        │  at subtask boundary:                     │
        │  images + memory + subtask ─────────────▶ │  evaluate_subtask()
        │  ◀── scalar reward (0.0 or 1.0) ────────  │

Configuration (under ``vlm_planner`` in the top-level YAML):
    model_path: str
        HuggingFace model ID or local path, e.g. "Qwen/Qwen3.5-9B".
    backend: str
        Inference backend: "transformers" (default) or "sglang".
    dtype: str
        Torch dtype string: "bfloat16" (default), "float16", or "float32".
    max_new_tokens_subtask: int
        Maximum tokens to generate for subtask instructions (default: 64).
    max_new_tokens_reward: int
        Maximum tokens to generate for reward verdicts (default: 16).
    max_memory_entries: int
        Maximum number of text entries kept in the rolling memory buffer.
    success_threshold: float
        Confidence threshold [0, 1] above which the VLM vote counts as success.

Example YAML::

    vlm_planner:
      model_path: "Qwen/Qwen3.5-9B"
      backend: "transformers"
      dtype: "bfloat16"
      max_new_tokens_subtask: 64
      max_new_tokens_reward: 16
      max_memory_entries: 20
      success_threshold: 0.5
"""

import warnings
from collections import deque
from typing import Optional

import numpy as np
import ray
from omegaconf import DictConfig
from PIL import Image

from rlinf.utils.logging import get_logger

_SUBTASK_SYSTEM_PROMPT = """\
You are an AI assistant controlling a bimanual robot arm. \
You will be shown images from the robot's cameras along with a history of past actions. \
Your job is to identify the single most appropriate next subtask for the robot to execute. \
Reply with ONLY the subtask instruction as a short imperative sentence (5-15 words). \
Do not add any explanation or formatting."""

_REWARD_SYSTEM_PROMPT = """\
You are an AI evaluator for a bimanual robot arm. \
You will be shown images from the robot's cameras and a description of the subtask that was attempted. \
Decide whether the robot successfully completed the subtask. \
Reply with ONLY "success" or "failure" — no other text."""


@ray.remote(num_gpus=1)
class VLMPlannerWorker:
    """Ray remote actor that hosts a local Qwen VLM for subtask planning and reward evaluation.

    Placement: Beaker GPU node.  The actor claims 1 GPU via ``@ray.remote(num_gpus=1)``.

    Args:
        cfg: Top-level Hydra config.  The worker reads ``cfg.vlm_planner``.
    """

    def __init__(self, cfg: DictConfig):
        self._logger = get_logger()
        planner_cfg = cfg.get("vlm_planner", {})

        # Backward-compat deprecation warnings for old Anthropic config fields.
        if "model" in planner_cfg:
            warnings.warn(
                "[VLMPlannerWorker] Config field 'vlm_planner.model' is deprecated. "
                "Use 'vlm_planner.model_path' instead (e.g. 'Qwen/Qwen3.5-9B').",
                DeprecationWarning,
                stacklevel=2,
            )
        if "api_key_env" in planner_cfg:
            warnings.warn(
                "[VLMPlannerWorker] Config field 'vlm_planner.api_key_env' is deprecated. "
                "The worker now uses a local Qwen model and requires no API key.",
                DeprecationWarning,
                stacklevel=2,
            )

        self._model_path: str = planner_cfg.get("model_path", "Qwen/Qwen3.5-9B")
        self._backend: str = planner_cfg.get("backend", "transformers")
        self._dtype_str: str = planner_cfg.get("dtype", "bfloat16")
        self._max_new_tokens_subtask: int = int(
            planner_cfg.get("max_new_tokens_subtask", 64)
        )
        self._max_new_tokens_reward: int = int(
            planner_cfg.get("max_new_tokens_reward", 16)
        )
        self._max_memory_entries: int = int(planner_cfg.get("max_memory_entries", 20))
        self._success_threshold: float = float(
            planner_cfg.get("success_threshold", 0.5)
        )

        # TOPReward configuration
        self._top_reward_enabled: bool = bool(
            planner_cfg.get("top_reward_enabled", False)
        )
        self._top_reward_max_frames: int = int(
            planner_cfg.get("top_reward_max_frames", 16)
        )

        if self._backend == "sglang":
            self._load_sglang_backend()
        else:
            self._load_transformers_backend()

        if self._top_reward_enabled:
            from rlinf.algorithms.rewards.top_reward import TOPReward

            self._top_reward = TOPReward(
                planner_cfg, model=self._model, processor=self._processor
            )
            self._logger.info("[VLMPlannerWorker] TOPReward enabled.")

        # Rolling text memory: deque of text entries logged each step.
        self._memory: deque[str] = deque(maxlen=self._max_memory_entries)

    # ------------------------------------------------------------------
    # Backend loading
    # ------------------------------------------------------------------

    def _load_transformers_backend(self) -> None:
        """Load Qwen model via HuggingFace Transformers."""
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self._dtype_str, torch.bfloat16)

        self._logger.info(
            f"[VLMPlannerWorker] Loading '{self._model_path}' "
            f"with dtype={self._dtype_str} via transformers."
        )
        self._processor = AutoProcessor.from_pretrained(self._model_path)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self._model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self._model.eval()
        self._logger.info("[VLMPlannerWorker] Model loaded.")

    def _load_sglang_backend(self) -> None:
        """Launch a local SGLang engine for Qwen inference.

        Requires sglang>=0.4.6.  Older versions have signal-handler issues
        inside Ray actors.
        """
        try:
            import sglang as sgl
        except ImportError as exc:
            raise ImportError(
                "Backend 'sglang' requires the 'sglang' package (>=0.4.6). "
                "Install it with: pip install 'sglang>=0.4.6'"
            ) from exc

        self._logger.info(
            f"[VLMPlannerWorker] Launching SGLang engine for '{self._model_path}'."
        )
        self._sgl_engine = sgl.Engine(
            model_path=self._model_path,
            dtype=self._dtype_str,
            mem_fraction_static=0.5,
        )
        # Qwen image placeholder token used when building SGLang prompts.
        self._sgl_image_token = "<image>"
        self._logger.info("[VLMPlannerWorker] SGLang engine ready.")

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def append_memory(self, entry: str) -> None:
        """Append a text entry to the rolling memory buffer.

        Args:
            entry: A brief description of what happened this step, e.g.
                "Step 5: moved left arm toward red block (joint delta: 0.12 rad)".
        """
        self._memory.append(entry)

    def clear_memory(self) -> None:
        """Clear the rolling memory buffer (call at episode start)."""
        self._memory.clear()

    def get_memory_text(self) -> str:
        """Return the current memory as a single newline-joined string."""
        return "\n".join(self._memory) if self._memory else "(no history yet)"

    # ------------------------------------------------------------------
    # Subtask generation
    # ------------------------------------------------------------------

    def get_next_subtask(
        self,
        images: list[np.ndarray],
        memory: Optional[str] = None,
    ) -> str:
        """Generate the next subtask instruction from observations.

        Args:
            images: List of uint8 RGB images (H, W, 3) from robot cameras.
            memory: Optional explicit memory string.  If None, uses internal
                rolling buffer.

        Returns:
            Subtask instruction string, e.g. "pick up the red block".
        """
        if memory is None:
            memory = self.get_memory_text()

        user_text = (
            f"History of past steps:\n{memory}\n\n"
            "What is the single best next subtask for the robot to execute?"
        )
        messages = self._build_qwen_messages(_SUBTASK_SYSTEM_PROMPT, images, user_text)

        try:
            subtask = self._generate(messages, self._max_new_tokens_subtask)
        except Exception as exc:
            self._logger.warning(
                f"[VLMPlannerWorker] get_next_subtask failed: {exc}. "
                "Returning empty subtask."
            )
            subtask = ""

        self._logger.info(f"[VLMPlannerWorker] Next subtask: '{subtask}'")
        return subtask

    # ------------------------------------------------------------------
    # Subtask reward evaluation
    # ------------------------------------------------------------------

    def evaluate_subtask(
        self,
        images: list[np.ndarray],
        subtask: str,
        memory: Optional[str] = None,
    ) -> float:
        """Evaluate whether a subtask was completed, returning a binary reward.

        Args:
            images: List of uint8 RGB images from robot cameras (post-subtask).
            subtask: The subtask instruction that was attempted.
            memory: Optional explicit memory string.  If None, uses internal
                rolling buffer.

        Returns:
            1.0 if the subtask was completed, 0.0 otherwise.
        """
        if memory is None:
            memory = self.get_memory_text()

        user_text = (
            f"History of steps during the subtask:\n{memory}\n\n"
            f'Subtask attempted: "{subtask}"\n\n'
            "Did the robot successfully complete this subtask?"
        )
        messages = self._build_qwen_messages(_REWARD_SYSTEM_PROMPT, images, user_text)

        try:
            verdict = self._generate(messages, self._max_new_tokens_reward).lower()
            reward = 1.0 if "success" in verdict else 0.0
        except Exception as exc:
            self._logger.warning(
                f"[VLMPlannerWorker] evaluate_subtask failed: {exc}. "
                "Returning 0.0 reward."
            )
            reward = 0.0

        self._logger.info(f"[VLMPlannerWorker] Subtask '{subtask}' → reward={reward}")
        return reward

    # ------------------------------------------------------------------
    # TOPReward: dense progress reward via True-token log-probability
    # ------------------------------------------------------------------

    def compute_top_reward(
        self,
        frames: list[np.ndarray],
        instruction: str,
        reduction: str = "mean",
        fps: float = 2.0,
    ) -> float:
        """Compute a TOPReward progress score for the given trajectory frames.

        Delegates to :class:`rlinf.algorithms.rewards.top_reward.TOPReward`
        for the actual scoring logic (prompt construction, label masking,
        log-prob extraction).

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
        if not self._top_reward_enabled:
            return 0.0

        if self._backend == "sglang":
            self._logger.warning(
                "[VLMPlannerWorker] TOPReward is not supported with the sglang "
                "backend (requires forward pass, not generation). Returning 0.0."
            )
            return 0.0

        # Trim frames to the configured maximum.
        if len(frames) > self._top_reward_max_frames:
            frames = frames[-self._top_reward_max_frames :]

        try:
            score = self._top_reward.compute_score(
                frames, instruction, reduction=reduction, fps=fps
            )
        except Exception as exc:
            self._logger.warning(
                f"[VLMPlannerWorker] compute_top_reward failed: {exc}. Returning 0.0."
            )
            score = 0.0

        return float(score)

    # ------------------------------------------------------------------
    # Inference dispatch
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict], max_new_tokens: int) -> str:
        """Dispatch to the active backend and return a stripped response string.

        Args:
            messages: ChatML message list (system + user with images).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Stripped generated text string.
        """
        if self._backend == "sglang":
            return self._generate_sglang(messages, max_new_tokens)
        return self._generate_transformers(messages, max_new_tokens)

    def _generate_transformers(self, messages: list[dict], max_new_tokens: int) -> str:
        """Run inference using HuggingFace Transformers.

        Args:
            messages: ChatML message list.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Stripped generated text string.
        """
        import torch
        from qwen_vl_utils import process_vision_info

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, return_video_metadata=True
        )
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            **video_kwargs,
        ).to(next(self._model.parameters()).device)

        with torch.inference_mode():
            generated_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens
            )

        # Trim the prompt prefix from each sequence before decoding.
        prompt_len = inputs["input_ids"].shape[1]
        trimmed = [seq[prompt_len:] for seq in generated_ids]
        return self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    def _generate_sglang(self, messages: list[dict], max_new_tokens: int) -> str:
        """Run inference using a local SGLang engine.

        Args:
            messages: ChatML message list (system + user with PIL images).
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Stripped generated text string.
        """
        # Extract PIL images and build a flat prompt string with placeholders.
        pil_images = []
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, str):
                prompt_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            else:
                part_text = f"<|im_start|>{role}\n"
                for item in content:
                    if item["type"] == "image":
                        pil_images.append(item["image"])
                        part_text += self._sgl_image_token
                    elif item["type"] == "text":
                        part_text += item["text"]
                part_text += "<|im_end|>\n"
                prompt_parts.append(part_text)
        prompt_parts.append("<|im_start|>assistant\n")
        prompt = "".join(prompt_parts)

        output = self._sgl_engine.generate(
            prompt=prompt,
            image_data=pil_images if pil_images else None,
            sampling_params={"max_new_tokens": max_new_tokens},
        )
        return output["text"].strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_pil_image(image: np.ndarray) -> Image.Image:
        """Convert a uint8 RGB numpy array to a PIL Image.

        Args:
            image: uint8 ndarray of shape (H, W, 3).

        Returns:
            PIL Image in RGB mode.
        """
        return Image.fromarray(image.astype(np.uint8))

    def _build_qwen_messages(
        self,
        system_prompt: str,
        images: list[np.ndarray],
        user_text: str,
    ) -> list[dict]:
        """Build a ChatML message list for Qwen multimodal inference.

        Args:
            system_prompt: System instruction string.
            images: List of uint8 RGB numpy arrays from robot cameras.
            user_text: User-facing text prompt (history + question).

        Returns:
            ChatML list with system and user roles; user content interleaves
            image dicts and a trailing text dict.
        """
        user_content = []
        for img in images:
            if img is None:
                continue
            user_content.append(
                {
                    "type": "image",
                    "image": self._to_pil_image(np.asarray(img, dtype=np.uint8)),
                }
            )
        user_content.append({"type": "text", "text": user_text})

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
