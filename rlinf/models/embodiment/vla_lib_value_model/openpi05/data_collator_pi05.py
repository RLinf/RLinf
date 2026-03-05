"""
PI0.5 Data Collator with unified response format.

Expected dataset item format:
{
    'images': Dict[str, Tensor] with camera keys,
    'image_masks': Dict[str, Tensor] with bool masks,
    'prompt': str, # language instruction
    'prefix': Optional[str], # prefix string like "Subtask:", "Progress:"
    'response': Optional[str], # response string like "pick up the cup"
    'actions': Optional[Tensor [action_horizon, action_dim]], # action tensor
    'state': Tensor [state_dim], # state tensor
}

Note: prefix and response are paired - both None or both str.

Observation output format (matching modeling_pi05.py expectations):
{
    'images': Dict[str, Tensor],
    'image_masks': Dict[str, Tensor],
    'tokenized_prompt': Tensor[B, L],
    'tokenized_prompt_mask': BoolTensor[B, L],
    'token_ar_mask': Tensor[B, L],  # 0=bidirectional, 1=causal
    'token_loss_mask': BoolTensor[B, L],  # True=include in CE loss
    'token_kv_cache_mask': BoolTensor[B, L],  # True=include in KV cache for action expert
}
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from transformers.data.data_collator import DataCollatorMixin

from .processing_pi05 import PI05Processor
from ..openpi.data_collator_pi0 import stack_tensors

from rlinf.utils.dist_utils import get_logger

logger = get_logger(__name__)

# Module-level flag for one-time logging
_COLLATOR_VERIFIED = False


@dataclass
class PI05DataCollator(DataCollatorMixin):
    """
    Data collator for PI0.5 with unified response format.
    
    Expected dataset item format:
    {
        'images': Dict[str, Tensor] with camera keys,
        'image_masks': Dict[str, Tensor] with bool masks,
        'prompt': str, # language instruction
        'prefix': Optional[str], # prefix string like "Subtask:", "Progress:"
        'response': Optional[str], # response string like "pick up the cup"
        'actions': Optional[Tensor [action_horizon, action_dim]], # action tensor
        'state': Tensor [state_dim], # state tensor
    }
    """
    
    processor: PI05Processor
    max_length: int = 200
    return_tensors: str = "pt"
    train: bool = True
    
    def torch_call(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate examples for PI0.5 training."""
        images_batch = []
        image_masks_batch = []
        prompts = []
        prefixes = []
        responses = []
        states = []
        actions_list = []
        action_mask_list = []

        # Collect raw return values for metrics and expert mode training
        return_raw_list = []
        return_normalized_list = []
        return_bin_id_list = []
        target_values_list = []
        
        # Distributional RL fields (for Bellman backup)
        next_images_list = []
        next_states_list = []
        rewards_list = []
        reward_sum_list = []
        dones_list = []
        
        for ex in examples:
            image_key = 'image' if 'image' in ex else 'images'
            mask_key = 'image_mask' if 'image_mask' in ex else 'image_masks'
            images_batch.append(ex[image_key])
            image_masks_batch.append(ex.get(mask_key, {}))
            prompts.append(ex['prompt'])
            prefixes.append(ex.get('prefix'))
            responses.append(ex.get('response'))
            
            state = ex.get('state')
            if state is not None and isinstance(state, torch.Tensor):
                state = state.cpu().numpy()
            states.append(state)
            
            # Actions + per-example action mask (1.0 if actions present, else 0.0)
            actions = ex.get('actions')
            actions_list.append(actions)
            action_mask_list.append(1.0 if actions is not None else 0.0)
            # action_mask_list.append(1.0)
            return_raw_list.append(ex.get('return_raw'))
            return_normalized_list.append(ex.get('return_normalized'))
            return_bin_id_list.append(ex.get('return_bin_id'))
            target_values_list.append(ex.get('target_values'))
            
            # Distributional RL fields
            next_images_list.append(ex.get('next_images'))
            next_states_list.append(ex.get('next_state'))
            rewards_list.append(ex.get('rewards'))
            reward_sum_list.append(ex.get('reward_sum'))
            dones_list.append(ex.get('dones'))
        
        # Stack images
        images = stack_tensors(images_batch)
        image_masks = stack_tensors(image_masks_batch)
        
        # Process images
        processed_img = self.processor.image_processor(
            images=images,
            image_masks=image_masks,
            return_tensors="pt",
            train=self.train
        )
        
        # Process text with proper ar_mask and loss_mask
        processed_txt = self.processor.process_text(
            prompts=prompts,
            prefixes=prefixes,
            responses=responses,
            states=states,
            actions=actions_list,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        lang_tokens = processed_txt["input_ids"]
        lang_masks = processed_txt["attention_mask"].bool()
        ar_masks = processed_txt["token_ar_mask"]
        loss_masks = processed_txt["token_loss_mask"].bool()
        kv_cache_masks = processed_txt["token_kv_cache_mask"].bool()
        
        # One-time verification logging for prompt/prefix/response
        global _COLLATOR_VERIFIED
        if not _COLLATOR_VERIFIED:
            _COLLATOR_VERIFIED = True
            logger.info("[Collator Verification] First batch prompt/prefix/response:")
            for i in range(min(len(prompts), 4)):
                has_actions = action_mask_list[i] > 0.5
                sample_type = "VLA" if has_actions else "VLM"
                logger.info("  [%d] %s", i, sample_type)
                logger.info("      prompt: %s", prompts[i] if prompts[i] else "None")
                logger.info("      prefix: %s", prefixes[i] if prefixes[i] else "None")
                resp = responses[i] if responses[i] else "None"
                logger.info("      response: %s", resp)
                logger.info("      loss_mask sum: %d, kv_cache_mask sum: %d",
                           loss_masks[i].sum().item(), kv_cache_masks[i].sum().item())

        # Per-sample action mask: indicates which samples should contribute to
        # action (flow-matching) loss. This mirrors language token supervision
        # which is controlled via token_loss_mask, but at the sample level.
        action_mask = torch.tensor(action_mask_list, dtype=torch.float32)
        
        # Build observation dict matching modeling_pi05.py expectations
        observation = {
            'images': processed_img["pixel_values"],
            'image_masks': processed_img["image_masks"],
            'tokenized_prompt': lang_tokens,
            'tokenized_prompt_mask': lang_masks,
            'token_ar_mask': ar_masks,
            'token_loss_mask': loss_masks,
            'token_kv_cache_mask': kv_cache_masks,
            'action_mask': action_mask,
        }
        
        batch = {
            'input_ids': lang_tokens,
            'attention_mask': lang_masks,
            'observation': observation,
        }
        
        # Add actions if any example has actions
        has_any_actions = any(a is not None for a in actions_list)
        if has_any_actions:
            batch_actions = []
            for a in actions_list:
                if a is None:
                    # Placeholder for missing actions (will be masked in loss)
                    # Use shape from first non-None action or default
                    first_action = next((x for x in actions_list if x is not None), None)
                    if first_action is not None:
                        if isinstance(first_action, torch.Tensor):
                            shape = first_action.shape
                        else:
                            shape = first_action.shape
                        a = torch.zeros(shape, dtype=torch.float32)
                    else:
                        a = torch.zeros(1, dtype=torch.float32)
                elif not isinstance(a, torch.Tensor):
                    a = torch.tensor(a, dtype=torch.float32)
                batch_actions.append(a)
            batch['actions'] = torch.stack(batch_actions)
        
        # Add raw return values for metrics (if available)
        if return_raw_list[0] is not None:
            batch['return_raw'] = torch.tensor(return_raw_list, dtype=torch.float32)
        if return_normalized_list[0] is not None:
            batch['return_normalized'] = torch.tensor(return_normalized_list, dtype=torch.float32)
        if return_bin_id_list[0] is not None:
            batch['return_bin_id'] = torch.tensor(return_bin_id_list, dtype=torch.long)
        
        # Add target_values for expert/dual mode training
        if target_values_list[0] is not None:
            batch['target_values'] = torch.tensor(target_values_list, dtype=torch.float32)
        
        # =====================================================================
        # Distributional RL fields (for n-step TD target computation)
        # =====================================================================
        
        # Next observation images
        if next_images_list[0] is not None:
            next_images = stack_tensors(next_images_list)
            next_processed_img = self.processor.image_processor(
                images=next_images,
                image_masks={},  # No masks for next images
            )
            batch['next_images'] = next_processed_img["pixel_values"]
        
        # Next observation state
        if next_states_list[0] is not None:
            next_states = []
            for ns in next_states_list:
                if ns is not None and isinstance(ns, torch.Tensor):
                    next_states.append(ns.cpu())
                elif ns is not None:
                    next_states.append(torch.tensor(ns, dtype=torch.float32))
                else:
                    next_states.append(torch.zeros_like(next_states[0]) if next_states else None)
            if all(s is not None for s in next_states):
                batch['next_states'] = torch.stack(next_states)
        
        # Rewards (n-step reward chunk or sum)
        if reward_sum_list[0] is not None:
            batch['reward_sum'] = torch.tensor(reward_sum_list, dtype=torch.float32)
        
        # Number of valid rewards (for gamma power in Bellman backup)
        num_valid_rewards_list = [ex.get('num_valid_rewards') for ex in examples]
        if num_valid_rewards_list[0] is not None:
            batch['num_valid_rewards'] = torch.tensor(num_valid_rewards_list, dtype=torch.long)
        
        # Done flags
        if dones_list[0] is not None:
            batch['dones'] = torch.tensor(dones_list, dtype=torch.bool)
        
        return batch


__all__ = ["PI05DataCollator"]
