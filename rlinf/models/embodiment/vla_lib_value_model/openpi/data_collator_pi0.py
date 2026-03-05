"""
PI0-specific data collator for VLA training using the new PI0Processor.

This collator handles the unique input format required by PI0 models:
- Images: Dict of camera views -> processed to List of [B, C, H, W] tensors
- Image masks: Dict of camera masks -> processed to List of [B] bool tensors  
- Language tokens: [B, seq_len] 
- Language masks: [B, seq_len] bool
- Robot state: [B, state_dim]
- Actions: [B, action_horizon, action_dim] (for training)
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional
from transformers.data.data_collator import DataCollatorMixin
from dataclasses import dataclass

from .processing_pi0 import PI0Processor


def stack_tensors(list_of_dicts: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """
    Stack a list of dictionaries of tensors/values.
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
class PI0DataCollator(DataCollatorMixin):
    """
    Data collator for PI0 Vision-Language-Action model using PI0Processor.
    
    Handles batching and preprocessing of multimodal robot control data.
    """
    
    processor: PI0Processor
    max_length: int = 48
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    train: bool = True  # Whether to apply training augmentations
    
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of examples for PI0 training using PI0Processor.
        
        Expected input format for each example (from dataset transforms):
        {
            'images': Dict[str, torch.Tensor] with OpenPI camera keys,
            'image_masks': Dict[str, torch.Tensor] with bool masks,
            'prompt': str (language instruction),
            'state': torch.Tensor [state_dim],
            'actions': torch.Tensor [action_horizon, action_dim] (for training)
        }
        
        Returns batch compatible with model expectations:
        {
            # For TRL compatibility
            'input_ids': [B, seq_len] (language tokens),
            'attention_mask': [B, seq_len] bool,
            
            # PI0 observation format (dict-style)
            'observation': Dict with processed multimodal data
        }
        """
        batch_size = len(examples)
        
        # Extract data
        images_batch = []
        image_masks_batch = []
        prompts_batch = []
        states_batch = []
        actions_batch = []
        
        for example in examples:
            # Handle both 'image' (OpenPI/LiberoInputs format) and 'images' keys
            image_key = 'image' if 'image' in example else 'images'
            mask_key = 'image_mask' if 'image_mask' in example else 'image_masks'
            images_batch.append(example[image_key])
            image_masks_batch.append(example.get(mask_key, {}))
            prompts_batch.append(example['prompt'])
            states_batch.append(example['state'])
            if 'actions' in example:
                actions_batch.append(example['actions'])
        
        images_batch = stack_tensors(images_batch)
        image_masks_batch = stack_tensors(image_masks_batch)

        processed_img = self.processor.image_processor(
            images_batch,
            image_masks_batch,
            return_tensors="pt",
            train=self.train
        )
        # Images are already normalized to [-1, 1] float by PI0ImageProcessor.process_images
        images = processed_img["pixel_values"]
        image_masks = processed_img["image_masks"]

        # Process text using PI0Processor
        processed_txt = self.processor.process_text(
            prompts_batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        lang_tokens = processed_txt["input_ids"]
        lang_masks = processed_txt["attention_mask"].bool()
        
        # Process states
        states = []
        for state in states_batch:
            if isinstance(state, torch.Tensor):
                states.append(state)
            else:
                states.append(torch.tensor(state, dtype=torch.float32))
        
        # Create observation dict (compatible with modeling_pi0.py)
        batch = {
            'attention_mask': lang_masks,
            'input_ids': lang_tokens,
            'observation': {
                'state': torch.stack(states),
                'images': images,
                'image_masks': image_masks,
                'tokenized_prompt': lang_tokens,
                'tokenized_prompt_mask': lang_masks,
            },
        }
        
        # Process actions if available (training mode)
        if actions_batch:
            actions = []
            for action in actions_batch:
                if isinstance(action, torch.Tensor):
                    actions.append(action)
                else:
                    actions.append(torch.tensor(action, dtype=torch.float32))
            
            batch['actions'] = torch.stack(actions)
        
        return batch
