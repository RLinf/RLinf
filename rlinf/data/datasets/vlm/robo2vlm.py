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


import ast
from io import BytesIO
from typing import Any, Optional, Union

import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer

from rlinf.data.datasets.common.item import DatasetItem, SftDatasetItem
from rlinf.data.datasets.vlm.base import VLMBaseDataset
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry
from rlinf.utils.logging import get_logger

logger = get_logger()


@VLMDatasetRegistry.register("robo2vlm")
class Robo2VLMDataset(VLMBaseDataset):
    def __init__(
        self,
        data_paths: Union[list[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
    ) -> None:
        super().__init__(data_paths, config, tokenizer)
        self.system_prompt = (
            "You are a helpful robotic vision assistant specialized in "
            "answering questions about robotic manipulation tasks. "
            "Use <think></think> tags to show your reasoning process, "
            "then provide your final answer in <answer></answer> tags."
        )

    def get_image_list(self, dataitem: dict[str, Any]) -> list[Union[bytes, str, None]]:
        images: list[Any] = []
        if "images" in dataitem:
            v = dataitem.get("images")
            if isinstance(v, list):
                images = list(v)
            elif v is not None:
                images = [v]
            else:
                images = [None]
        elif "image" in dataitem:
            v = dataitem.get("image")
            if v is not None:
                images = [v]
            else:
                images = [None]
        else:
            return super().get_image_list(dataitem)

        normed: list[Union[bytes, str, None]] = []
        for v in images:
            if v is None:
                continue
            if isinstance(v, Image.Image):
                normed.append(v)
            elif isinstance(v, dict) and "bytes" in v:
                normed.append(v["bytes"])  # raw bytes
            else:
                normed.append(v)  # path/uri/string
        if not normed:
            normed = [None]
        return normed

    def build_prompt_text(self, data_item: dict[str, Any]) -> str:
        # Use 'question' and 'choices' if present; else fallback to base using configured prompt/choice keys
        question = data_item.get("question", None)
        choices = data_item.get("choices", None)
        if question is None:
            return super().build_prompt_text(data_item)
        # normalize choices
        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except Exception:
                choices = [choices]
        if not isinstance(choices, list):
            choices = [choices] if choices is not None else []

        text = f"{question}\n"
        if choices:
            text += "Choices:\n"
            for i, c in enumerate(choices):
                text += f"{chr(65 + i)}. {c}\n"
        return text

    def postprocess_dataset_item(
        self, item: DatasetItem, raw: dict[str, Any]
    ) -> DatasetItem:
        answer_dict = {
            "choices": raw.get("choices", None),
            "correct_answer": raw.get("correct_answer", None),
        }
        item.answer = answer_dict

        return item


@VLMDatasetRegistry.register("robo2vlmsft")
class Robo2VLMSFTDataset(Robo2VLMDataset):
    # The reason for overriding the current function is that the SFT dataset requires masking the attention for the answer separately.
    def __init__(
        self,
        data_paths: Union[list[str], str],
        config: DictConfig,
        tokenizer: AutoTokenizer,
        eval_dataset: bool = False,
    ) -> None:
        super().__init__(data_paths, config, tokenizer)
        self.apply_chat_template = config.data.apply_chat_template
        self.eval_dataset = eval_dataset

    def _process_raw_record(self, raw: dict[str, Any], idx: int) -> DatasetItem:
        images = self.get_image_list(raw)
        (
            prompt_ids,
            plen,
            prompt_text,
            answer,
            attention_mask,
            label_mask,
            multi_modal_inputs,
        ) = self.encode_prompt(raw)

        item = SftDatasetItem(
            prompt=prompt_ids,
            length=plen,
            idx=idx,
            image_data=images,
            answer=answer,
            prompt_text=prompt_text,
            attention_mask=attention_mask,
            label_mask=label_mask,
            meta=None,
            multi_modal_inputs=multi_modal_inputs,
        )
        return item

    def vlm_vision_convert(
        self,
        prompt: dict[str, Any],
        images_inputs: list[Any],
    ):
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.actor.model.model_path
            )

        prompt_text = self._processor.apply_chat_template(
            prompt, tokenize=False, add_generation_prompt=True
        )

        assert isinstance(prompt_text[0], str), (
            f"{type(prompt_text)} {prompt_text} is not str"
        )

        prompt_inputs = self._processor(
            text=prompt_text, images=images_inputs, return_tensors="pt", padding=True
        )

        return prompt_inputs

    def encode_prompt(
        self, raw: dict[str, Any]
    ) -> tuple[torch.Tensor, int, dict[str, Any], torch.Tensor, Optional[str]]:
        """
        Return (
            input_ids: torch.Tensor,      # [L] or [1, L] token ids
            int(input_ids.numel()),       # length of token ids
            prompt: dict[str, Any],       # raw prompt/messages
            answer: Optional[str],        # raw answer text (may be None)
            attention_mask: torch.Tensor, # [L] or [1, L] attention mask
            label_mask: torch.Tensor,     # [L] or [1, L] SFT label mask (prompt part masked)
            multi_modal_inputs: dict[str, Any], # processor outputs (e.g. pixel_values, image_grid_thw)
        ).

        If using chat template, encode with processor.
        Subclasses may override to support alternative prompting.

        Expected format:
        {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image1", "image1": <PIL.Image1>}
                    {"type": "image2", "image2": <PIL.Image2>}
                    {"type": "text", "text": "User message"},
                ]},
                {"role": "assistant", "content": [
                    {"type": "text", "text": "Assistant response"}
                ]}
            ]
        }

        """
        question = raw.get(self.prompt_key, None)
        assert question is not None, "question is required"
        choices = raw.get(self.choice_key, None)
        assert choices is not None, "choices is required"
        correct_answer = raw.get(self.answer_key, None)
        assert correct_answer is not None, "correct_answer is required"
        images = self.get_image_list(raw)

        str_images = []
        for k in self.image_keys:
            v = raw.get(k)
            if isinstance(v, str):
                str_images.append(v)
            elif isinstance(v, list):
                str_images.extend([x for x in v if isinstance(x, str)])

        # convert the images to PIL Image objects
        images_inputs = []
        for image in images:
            image_obj = None
            if isinstance(image, Image.Image):
                image_obj = image.convert("RGB")
            if isinstance(image, (bytes, bytearray)):
                image_obj = Image.open(BytesIO(image)).convert("RGB")
            images_inputs.append(image_obj)

        # Handle both letter format (A, B, C, D) and index format (0, 1, 2, 3)
        if isinstance(correct_answer, str):
            # Map the answer letter (A, B, C, D) to index (0, 1, 2, 3)
            letter_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
            correct_answer_idx = letter_to_idx.get(correct_answer, 0)
            correct_answer_letter = correct_answer
        else:
            # It's already an index
            correct_answer_idx = correct_answer
            correct_answer_letter = chr(
                65 + correct_answer_idx
            )  # Convert 0->A, 1->B, etc.

        # Parse choices if they're in string format
        if isinstance(choices, str):
            try:
                choices = ast.literal_eval(choices)
            except (ValueError, SyntaxError) as e:
                logger.warning(
                    "Failed to parse choices as list, fallback to single choice. "
                    "choices=%r, error=%s",
                    choices,
                    e,
                )
                choices = [str(choices)]

        # Create a formatted question with choices
        formatted_question = f"{question}\nChoices:\n"
        for i, choice in enumerate(choices):
            formatted_question += f"{chr(65 + i)}. {choice}\n"

        # Create the answer using the correct choice (matching finetune example format)
        answer = f"{correct_answer_letter}. {choices[correct_answer_idx]}"

        image_context = [{"type": "image", "image": img} for img in images_inputs]
        image_context.append({"type": "text", "text": formatted_question})

        prompt = [
            (
                {"role": "user", "content": image_context},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            )
        ]

        without_assistant_prompt = [({"role": "user", "content": image_context},)]

        # get the vlm model prompt inputs
        # prompt_inputs = input_ids, attention_mask, pixel_values, image_grid_thw
        prompt_inputs = self.vlm_vision_convert(prompt, images_inputs)
        # get the vlm model prompt inputs without assistant for label
        label_prompt_inputs = self.vlm_vision_convert(
            without_assistant_prompt, images_inputs
        )

        if self.eval_dataset:
            # for eval dataset, we need to use the label prompt inputs
            input_ids = label_prompt_inputs.pop("input_ids")
            attention_mask = label_prompt_inputs.pop("attention_mask")
            label_mask = attention_mask
        else:
            # for train dataset, we need to use the prompt prompt inputs
            input_ids = prompt_inputs.pop("input_ids")
            attention_mask = prompt_inputs.pop("attention_mask")
            label_mask = label_prompt_inputs.pop("attention_mask")

        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 2 and input_ids.size(0) == 1:
                input_ids = input_ids.squeeze(0)
            input_ids = input_ids.to(dtype=torch.long)
        else:
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        multi_modal_inputs = {}
        if self.eval_dataset:
            for k, v in label_prompt_inputs.items():
                multi_modal_inputs[k] = v
        else:
            for k, v in prompt_inputs.items():
                multi_modal_inputs[k] = v

        return (
            input_ids,
            int(input_ids.numel()),
            prompt,
            answer,
            attention_mask,
            label_mask,
            multi_modal_inputs,
        )
