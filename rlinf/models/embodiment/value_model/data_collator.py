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

"""
Data Collator for value model training.

Handles batching and preprocessing of multimodal robot control data
with support for RL-specific fields (returns, target values, etc.).
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorMixin

logger = logging.getLogger(__name__)

# Module-level flag for one-time logging
_COLLATOR_VERIFIED = False


def stack_tensors(list_of_dicts: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Stack a list of dictionaries of tensors/values.

    Handles numpy booleans and other non-tensor types by converting to tensors first.
    Handles dicts with different keys by using the union of all keys and creating
    zero tensors for missing entries.
    """
    stacked_dict = {}
    if len(list_of_dicts) == 0:
        return stacked_dict

    all_keys = set()
    for d in list_of_dicts:
        all_keys.update(d.keys())

    for key in all_keys:
        tensors = []
        template_tensor = None
        for d in list_of_dicts:
            v = d.get(key)
            if v is None:
                tensors.append(None)
            elif isinstance(v, torch.Tensor):
                tensors.append(v)
                if template_tensor is None:
                    template_tensor = v
            elif isinstance(v, (np.bool_, bool)):
                t = torch.tensor(v, dtype=torch.bool)
                tensors.append(t)
                if template_tensor is None:
                    template_tensor = t
            elif isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                tensors.append(t)
                if template_tensor is None:
                    template_tensor = t
            else:
                t = torch.tensor(v)
                tensors.append(t)
                if template_tensor is None:
                    template_tensor = t

        if template_tensor is None:
            continue

        filled_tensors = []
        for t in tensors:
            if t is None:
                filled_tensors.append(torch.zeros_like(template_tensor))
            else:
                filled_tensors.append(t)
        stacked_dict[key] = torch.stack(filled_tensors)
    return stacked_dict


@dataclass
class ValueDataCollator(DataCollatorMixin):
    """Data collator for value model training."""

    processor: Any  # ValueProcessor
    max_length: int = 200
    return_tensors: str = "pt"
    train: bool = True

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate examples for value model training."""
        images_batch = []
        image_masks_batch = []
        prompts = []
        actions_list = []
        action_mask_list = []

        return_raw_list = []
        return_normalized_list = []
        return_bin_id_list = []
        target_values_list = []

        next_images_list = []
        next_states_list = []
        reward_sum_list = []
        dones_list = []

        for ex in examples:
            image_key = "image" if "image" in ex else "images"
            mask_key = "image_mask" if "image_mask" in ex else "image_masks"
            images_batch.append(ex[image_key])
            image_masks_batch.append(ex.get(mask_key, {}))
            prompts.append(ex["prompt"])

            actions = ex.get("actions")
            actions_list.append(actions)
            action_mask_list.append(1.0 if actions is not None else 0.0)
            return_raw_list.append(ex.get("return_raw"))
            return_normalized_list.append(ex.get("return_normalized"))
            return_bin_id_list.append(ex.get("return_bin_id"))
            target_values_list.append(ex.get("target_values"))

            next_images_list.append(ex.get("next_images"))
            next_states_list.append(ex.get("next_state"))
            reward_sum_list.append(ex.get("reward_sum"))
            dones_list.append(ex.get("dones"))

        images = stack_tensors(images_batch)
        image_masks = stack_tensors(image_masks_batch)

        processed_img = self.processor.image_processor(
            images=images,
            image_masks=image_masks,
            return_tensors="pt",
            train=self.train,
        )

        processed_txt = self.processor.process_text(
            prompts=prompts,
            max_length=self.max_length,
            return_tensors="pt",
        )

        lang_tokens = processed_txt["input_ids"]
        lang_masks = processed_txt["attention_mask"].bool()
        ar_masks = processed_txt["token_ar_mask"]

        global _COLLATOR_VERIFIED
        if not _COLLATOR_VERIFIED:
            _COLLATOR_VERIFIED = True
            logger.info("[Collator Verification] First batch prompts:")
            for i in range(min(len(prompts), 4)):
                logger.info(
                    "  [%d] prompt: %s", i, prompts[i] if prompts[i] else "None"
                )

        action_mask = torch.tensor(action_mask_list, dtype=torch.float32)

        observation = {
            "images": processed_img["pixel_values"],
            "image_masks": processed_img["image_masks"],
            "tokenized_prompt": lang_tokens,
            "tokenized_prompt_mask": lang_masks,
            "token_ar_mask": ar_masks,
            "action_mask": action_mask,
        }

        batch = {
            "input_ids": lang_tokens,
            "attention_mask": lang_masks,
            "observation": observation,
        }

        has_any_actions = any(a is not None for a in actions_list)
        if has_any_actions:
            batch_actions = []
            for a in actions_list:
                if a is None:
                    first_action = next(
                        (x for x in actions_list if x is not None), None
                    )
                    if first_action is not None:
                        shape = first_action.shape
                        a = torch.zeros(shape, dtype=torch.float32)
                    else:
                        a = torch.zeros(1, dtype=torch.float32)
                elif not isinstance(a, torch.Tensor):
                    a = torch.tensor(a, dtype=torch.float32)
                batch_actions.append(a)
            batch["actions"] = torch.stack(batch_actions)

        if return_raw_list[0] is not None:
            batch["return_raw"] = torch.tensor(return_raw_list, dtype=torch.float32)
        if return_normalized_list[0] is not None:
            batch["return_normalized"] = torch.tensor(
                return_normalized_list, dtype=torch.float32
            )
        if return_bin_id_list[0] is not None:
            batch["return_bin_id"] = torch.tensor(return_bin_id_list, dtype=torch.long)
        if target_values_list[0] is not None:
            batch["target_values"] = torch.tensor(
                target_values_list, dtype=torch.float32
            )

        if next_images_list[0] is not None:
            next_images = stack_tensors(next_images_list)
            next_processed_img = self.processor.image_processor(
                images=next_images, image_masks={}
            )
            batch["next_images"] = next_processed_img["pixel_values"]

        if next_states_list[0] is not None:
            next_states = []
            for ns in next_states_list:
                if ns is not None and isinstance(ns, torch.Tensor):
                    next_states.append(ns.cpu())
                elif ns is not None:
                    next_states.append(torch.tensor(ns, dtype=torch.float32))
                else:
                    next_states.append(
                        torch.zeros_like(next_states[0]) if next_states else None
                    )
            if all(s is not None for s in next_states):
                batch["next_states"] = torch.stack(next_states)

        if reward_sum_list[0] is not None:
            batch["reward_sum"] = torch.tensor(reward_sum_list, dtype=torch.float32)

        num_valid_rewards_list = [ex.get("num_valid_rewards") for ex in examples]
        if num_valid_rewards_list[0] is not None:
            batch["num_valid_rewards"] = torch.tensor(
                num_valid_rewards_list, dtype=torch.long
            )

        if dones_list[0] is not None:
            batch["dones"] = torch.tensor(dones_list, dtype=torch.bool)

        return batch


__all__ = ["ValueDataCollator", "stack_tensors"]
