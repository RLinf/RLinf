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

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from transformers import AutoProcessor, AutoTokenizer

from rlinf.data.datasets.common.item import SftDatasetItem
from rlinf.data.datasets.vlm.base import VLMBaseDataset
from rlinf.data.datasets.vlm.registry import VLMDatasetRegistry


def _chat_message(message: dict[str, Any], image_root: Path) -> dict[str, Any]:
    content = []
    for block in message["content"]:
        if block["type"] == "text":
            content.append({"type": "text", "text": block["text"]})
        elif block["type"] == "image":
            path = image_root / block["artifact"]["path"]
            content.append({"type": "image", "image": str(path)})
        else:
            raise ValueError(f"Unsupported RPent content type: {block['type']!r}")

    result = {"role": message["role"], "content": content}
    if message.get("tool_calls"):
        result["tool_calls"] = [
            {
                "type": "function",
                "id": call["id"],
                "function": {"name": call["name"], "arguments": call["arguments"]},
            }
            for call in message["tool_calls"]
        ]
    if message.get("tool_call_id") is not None:
        result["tool_call_id"] = message["tool_call_id"]
    return result


@VLMDatasetRegistry.register("rpent_planner_sft")
class RPentPlannerSFTDataset(VLMBaseDataset):
    """Encode RPent planner turns with supervision on the last reply and tool calls."""

    def __init__(
        self,
        data_paths: list[str] | str,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        eval_dataset: bool = False,
    ) -> None:
        self.model_path = config.actor.model.model_path
        self.max_prompt_length = config.data.max_prompt_length
        self.eval_dataset = eval_dataset
        self._processor = None
        self._load_data_files(data_paths, config.data.get("lazy_loading", False))

    def _process_raw_record(self, raw: dict[str, Any], idx: int) -> SftDatasetItem:
        image_root = Path(self._indices[idx][0]).absolute().parent
        messages = [
            _chat_message(message, image_root)
            for message in [*raw["messages"], raw["target"]]
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                },
            }
            for tool in raw["tools"]
        ]
        meta = {"episode_id": raw["episode_id"], "turn_id": raw["turn_id"]}

        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(self.model_path)
        template_args = {
            "tools": tools,
            "enable_thinking": False,
            "tokenize": True,
            "return_dict": True,
            "return_tensors": "pt",
        }
        prefix = self._processor.apply_chat_template(
            messages[:-1], add_generation_prompt=True, **template_args
        )
        full = self._processor.apply_chat_template(
            messages, add_generation_prompt=False, **template_args
        )
        prefix_ids, full_ids = prefix["input_ids"][0], full["input_ids"][0]
        start = prefix_ids.numel()
        if not torch.equal(prefix_ids, full_ids[:start]):
            raise ValueError(
                f"Generation prefix differs from the encoded chat for {meta}"
            )
        if full_ids.numel() > self.max_prompt_length:
            raise ValueError(
                f"RPent sample {meta} has {full_ids.numel()} tokens, "
                f"exceeding max_prompt_length={self.max_prompt_length}"
            )

        # Qwen closes the assistant turn with EOS and a newline after its payload.
        tokenizer = self._processor.tokenizer
        ending = tokenizer.encode(tokenizer.eos_token + "\n", add_special_tokens=False)
        stop = full_ids.numel() - len(ending)
        if full_ids[stop:].tolist() != ending or stop <= start:
            raise ValueError(
                f"Expected a nonempty reply followed by EOS and newline for {meta}"
            )

        inputs = prefix if self.eval_dataset else full
        ids = inputs.pop("input_ids")[0]
        label_mask = torch.ones_like(ids, dtype=torch.bool)
        label_mask[start:stop] = False
        return SftDatasetItem(
            prompt=ids,
            length=ids.numel(),
            idx=idx,
            answer=tokenizer.decode(
                full_ids[start:stop], clean_up_tokenization_spaces=False
            ),
            prompt_text={
                "messages": messages[:-1] if self.eval_dataset else messages,
                "tools": tools,
            },
            attention_mask=inputs.pop("attention_mask")[0],
            label_mask=label_mask,
            multi_modal_inputs=dict(inputs),
            meta=meta,
        )
