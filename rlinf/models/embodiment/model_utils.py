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

from typing import Any, Optional

import torch
import torch.nn.functional as F
from transformers.generation import TopKLogitsWarper


def default_logits_processor(logits, action_tokens, vocab_size, n_action_bins):
    logits = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]

    logits[:, : vocab_size - n_action_bins] = -torch.inf
    logits[:, vocab_size:] = -torch.inf

    logprobs = compute_logprobs_from_logits(logits=logits, target=action_tokens)

    entropy = compute_entropy_from_logits(logits)

    ret = {"logprobs": logprobs, "entropy": entropy}

    return ret


def compute_logprobs_from_logits(logits, target):
    logprobs = -F.cross_entropy(
        logits, target=target, reduction="none"
    )  # [B, action-dim]
    return logprobs


def compute_entropy_from_logits(logits, epsilon=1e-10):
    """
    Compute entropy by logits.

    Args:
        logits: [B, vocab-size, seq-len]
    Returns:
        entropy: [B, seq-len]
    """
    # Ensure epsilon is a tensor with correct dtype
    if not isinstance(epsilon, torch.Tensor):
        epsilon = torch.tensor(epsilon, dtype=logits.dtype, device=logits.device)
    
    all_probs = F.softmax(logits, dim=1)  # [B, vocab-size, seq-len]
    all_log_probs = torch.log(all_probs + epsilon)
    entropy = -torch.sum(all_probs * all_log_probs, dim=1)  # [B, seq-len]
    return entropy


def _convert_action_tokens_to_features(action_tokens, model):
    """Convert action tokens to features suitable for Q-network input"""
    batch_size = action_tokens.shape[0]
    model_dtype = next(model.parameters()).dtype
    
    # Convert to float first, then to model dtype
    action_features = action_tokens.view(batch_size, -1).float().to(dtype=model_dtype)
    
    # Normalize action features to a reasonable range
    # Ensure n_action_bins is also in the correct dtype
    n_action_bins = torch.tensor(model.config.n_action_bins, dtype=model_dtype, device=action_tokens.device)
    action_features = action_features / n_action_bins
    
    return action_features


def prepare_observations_for_vla(
    simulator_type: str,
    model_name: str,
    raw_obs: dict,
    use_proprio: bool,
    max_length: int,
    processor: Any,
    precision: torch.dtype,
    device: torch.device = torch.device("cuda:0"),
):
    task_descriptions = [
        f"In: What action should the robot take to {t.lower()}?\nOut: "
        for t in raw_obs["task_descriptions"]
    ]

    if simulator_type == "libero":
        image_tensor = torch.stack(
            [
                value.clone().to(device).permute(2, 0, 1)
                for value in raw_obs["images_and_states"]["full_image"]
            ]
        )
    elif simulator_type == "maniskill":
        images = raw_obs["images"]
        image_tensor = images.to(device=device, dtype=precision)
    elif simulator_type == "robotwin":
        images = raw_obs["images"]
        image_tensor = images.to(device=device, dtype=precision)
    else:
        raise NotImplementedError

    proprio_states = None
    if use_proprio:
        proprio_keys = [
            key for key in raw_obs["images_and_states"] if "image" not in key
        ]
        proprio_states = {
            key: torch.stack(
                [val.to(device) for val in raw_obs["images_and_states"][key]]
            )
            for key in proprio_keys
        }

    # Add num_images dimension
    if image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(1)
    assert image_tensor.ndim == 5

    if model_name == "openvla":
        processed_obs = processor(
            text=task_descriptions,
            images=image_tensor,
            padding="max_length",
            max_length=max_length,
        )
    elif model_name == "openvla_oft":
        images = {"images": image_tensor}
        processed_obs = processor(
            text=task_descriptions,
            images=images,
            proprio_states=proprio_states,
            padding="max_length",
            max_length=max_length,
        )

    processed_obs = processed_obs.to(device=device, dtype=precision)
    for key, value in processed_obs.items():
        processed_obs[key] = value.contiguous()

    return processed_obs

def prepare_observations(
    simulator_type: str,
    model_name: str,
    raw_obs: Any,
    use_proprio: bool = True,
    max_length: int = 512,
    processor: Optional[Any] = None,
    precision: str = "fp16",
    device: torch.device = torch.device("cuda:0"),
):
    if model_name == "openvla" or model_name == "openvla_oft":
        return prepare_observations_for_vla(
            simulator_type=simulator_type,
            model_name=model_name,
            raw_obs=raw_obs,
            use_proprio=use_proprio,
            max_length=max_length,
            processor=processor,
            precision=precision,
            device=device,
        )
    else:
        raise NotImplementedError


def gumbel_softmax_sample(logits, temperature=1.0, hard=True):
    """
    Sample from Gumbel-Softmax distribution.
    
    Args:
        logits: [batch_size, num_classes] logits
        temperature: Temperature parameter (higher = more uniform)
        hard: If True, use straight-through estimator (discrete samples)
              If False, use continuous relaxation
    
    Returns:
        samples: [batch_size, num_classes] sampled actions
        log_probs: [batch_size] log probabilities
    """
    # Ensure temperature is a tensor with correct dtype
    if not isinstance(temperature, torch.Tensor):
        temperature = torch.tensor(temperature, dtype=logits.dtype, device=logits.device)
    
    # Add Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
    
    # Apply temperature and softmax
    y = F.softmax((logits + gumbel_noise) / temperature, dim=-1)
    
    if hard:
        # Straight-through estimator: discrete samples in forward, continuous in backward
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(-1, torch.argmax(y, dim=-1, keepdim=True), 1.0)
        y = y_hard - y.detach() + y  # Straight-through gradient
    
    # Compute log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    
    return y, log_probs


def sample_logits_processor(logits, vocab_size, n_action_bins, use_gumbel_softmax=True, temperature=1.0):
    """
    Logits processor for sampling actions using Gumbel-Softmax or standard sampling.
    
    Args:
        logits: [B, action-dim, vocab-size] logits (from custom_forward)
        vocab_size: Vocabulary size
        n_action_bins: Number of action bins
        use_gumbel_softmax: Whether to use Gumbel-Softmax for gradient flow
        temperature: Temperature for Gumbel-Softmax
    
    Returns:
        Dictionary with action_tokens, logprobs, entropy
    """
    # logits is already [B, action-dim, vocab-size] from custom_forward
    batch_size, action_dim, vocab_size_dim = logits.shape

    # Mask non-action tokens
    logits[:, :, : vocab_size - n_action_bins] = -torch.inf
    logits[:, :, vocab_size:] = -torch.inf

    # Ensure numerical stability
    logits = torch.clamp(logits, min=-1e8, max=1e8)

    if use_gumbel_softmax:
        # Use Gumbel-Softmax for differentiable discrete sampling
        action_probs_list = []
        logprobs_list = []
        entropy_list = []
        
        for i in range(action_dim):
            action_logits = logits[:, i, :]  # [B, vocab-size]
            
            # Gumbel-Softmax sampling
            action_probs, log_probs = gumbel_softmax_sample(
                action_logits, temperature=temperature, hard=True
            )
            
            action_probs_list.append(action_probs)
            logprobs_list.append(log_probs)
            
            # Compute entropy
            epsilon = torch.tensor(1e-8, dtype=action_probs.dtype, device=action_probs.device)
            entropy = -(action_probs * torch.log(action_probs + epsilon)).sum(dim=-1)
            entropy_list.append(entropy)
        
        # Stack results
        action_probs = torch.stack(action_probs_list, dim=1)  # [B, action-dim, vocab-size]
        logprobs = torch.stack(logprobs_list, dim=1)  # [B, action-dim, vocab-size]
        entropy = torch.stack(entropy_list, dim=1)  # [B, action-dim]
        
        # Get discrete action tokens for compatibility
        action_tokens = torch.argmax(action_probs, dim=-1)  # [B, action-dim]
        
        # Compute log probabilities for the entire action sequence
        logits_transposed = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]
        logprobs_per_dim = compute_logprobs_from_logits(logits=logits_transposed, target=action_tokens)
        
        # Sum logprobs across action dimensions to get the probability of the entire sequence
        logprobs = logprobs_per_dim.sum(dim=1, keepdim=True)  # [B, action_dim] -> [B, 1]
        
        # Compute entropy for the entire sequence
        entropy_per_dim = compute_entropy_from_logits(logits_transposed)
        entropy = entropy_per_dim.sum(dim=1, keepdim=True)  # [B, action_dim] -> [B, 1]
        
        ret = {
            "action_tokens": action_tokens,
            "logprobs": logprobs,  # [B, 1] - probability of entire action sequence
            "entropy": entropy,    # [B, 1] - entropy of entire action sequence
            "gumbel_temperature": temperature
        }
        
    else:
        # Standard discrete sampling (original implementation)
        action_tokens = torch.zeros(batch_size, action_dim, dtype=torch.long, device=logits.device)
        
        for i in range(action_dim):
            action_logits = logits[:, i, :]  # [B, vocab-size]
            
            # Apply softmax to get probabilities
            probs = F.softmax(action_logits, dim=1)  # [B, vocab-size]
            
            # Check for valid probabilities
            prob_sums = probs.sum(dim=1)  # [B]
            valid_mask = prob_sums > 1e-8
            
            if valid_mask.any():
                # Sample for valid distributions
                valid_indices = torch.where(valid_mask)[0]
                valid_probs = probs[valid_indices]  # [valid_B, vocab-size]
                
                # Sample actions
                sampled_actions = torch.multinomial(valid_probs, 1).squeeze(1)  # [valid_B]
                action_tokens[valid_indices, i] = sampled_actions
            
            # For invalid distributions, sample uniformly from action tokens
            invalid_mask = ~valid_mask
            if invalid_mask.any():
                invalid_indices = torch.where(invalid_mask)[0]
                uniform_actions = torch.randint(
                    vocab_size - n_action_bins, 
                    vocab_size, 
                    (len(invalid_indices),), 
                    device=logits.device
                )
                action_tokens[invalid_indices, i] = uniform_actions
        
        # Compute logprobs for the entire action sequence
        # In SAC, we need the probability of the entire action sequence, not individual components
        logits_transposed = logits.permute(0, 2, 1)  # [B, vocab-size, action-dim]
        logprobs_per_dim = compute_logprobs_from_logits(logits=logits_transposed, target=action_tokens)
        
        # Sum logprobs across action dimensions to get the probability of the entire sequence
        # log P(action_sequence) = log P(action_1) + log P(action_2) + ... + log P(action_7)
        logprobs = logprobs_per_dim.sum(dim=1, keepdim=True)  # [B, action_dim] -> [B, 1]
        
        # Compute entropy for the entire sequence
        entropy_per_dim = compute_entropy_from_logits(logits_transposed)
        entropy = entropy_per_dim.sum(dim=1, keepdim=True)  # [B, action_dim] -> [B, 1]

        ret = {
            "action_tokens": action_tokens,
            "logprobs": logprobs,  # [B, 1] - probability of entire action sequence
            "entropy": entropy     # [B, 1] - entropy of entire action sequence
        }

    return ret

def custom_forward(
    model,
    input_ids,
    attention_mask,
    pixel_values,
    output_hidden_states=True,
    action_token_len=None,
    value_model=False,
    q_network=False,
    value_head_mode: str = "a",
    logits_processor=sample_logits_processor,
    temperature: int = 1.0,
    top_k: int = -1,
    logits_processor_args: Optional[dict] = None,
    action_features=None,  # For Q-network: action features to concatenate with state
    use_gumbel_softmax: bool = False,  # Enable Gumbel-Softmax for differentiable discrete sampling
    gumbel_temperature: float = 1.0,  # Temperature for Gumbel-Softmax
):
    # print("input_ids shape: ", input_ids.shape)
    
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=pixel_values,
        output_hidden_states=output_hidden_states,
    )
    logits = output.logits[:, -action_token_len - 1 : -1]  # [B, action_dim, vocab_size]

    processed_logits_tensor = logits / temperature
    top_k = min(top_k, processed_logits_tensor.size(-1))  # Safety check
    if top_k > 0:
        logits_warper = TopKLogitsWarper(
            top_k
        )  # since here is logprob instead of logits, we use 0 instead of -inf
        processed_logits_tensor = logits_warper(None, processed_logits_tensor)

    # Choose logits processor based on Gumbel-Softmax setting
    if use_gumbel_softmax and logits_processor == sample_logits_processor:
        # Use Gumbel-Softmax sampling
        vocab_size = logits_processor_args.get("vocab_size", model.vocab_size)
        n_action_bins = logits_processor_args.get("n_action_bins", model.config.n_action_bins)
        output_dict = sample_logits_processor(
            processed_logits_tensor, 
            vocab_size=vocab_size, 
            n_action_bins=n_action_bins,
            use_gumbel_softmax=True,
            temperature=gumbel_temperature
        )
    else:
        # Use standard logits processor
        output_dict = logits_processor(processed_logits_tensor, **logits_processor_args)

    # Handle value model (for PPO/GAE)
    if value_model:
        # NOTE: Here we subtract 1 because the input tokens do not include the EOS token.
        last_hidden_state = output.hidden_states[-1]  # [B, L, hidden_dim]
        if value_head_mode == "a0":
            hidden_features = last_hidden_state[
                :, -action_token_len - 1
            ]  # [batch_size, hidden_dim]
        elif value_head_mode == "a6":
            hidden_features = last_hidden_state[
                :, -action_token_len - 1 - 6
            ]  # [batch_size, hidden_dim]
        else:
            raise ValueError(f"Unknown value head mode: {value_head_mode}")

        if hasattr(model, "value_head"):
            values = model.value_head(hidden_features)
            output_dict.update({"values": values})
        else:
            raise ValueError("Model does not have value_head but value_model=True")

    # Handle Q-network (for SAC)
    if q_network:
        last_hidden_state = output.hidden_states[-1]
        if value_head_mode in ["a0", "q_network"]:
            state_features = last_hidden_state[:, -action_token_len - 1]
        else:
            raise ValueError(f"Unknown value head mode for Q-network: {value_head_mode}")
        
        if hasattr(model, 'q_value_head'):
            if action_features is not None:
                if hasattr(model.q_value_head, 'q1'): # Check for DoubleQValueHead
                    q1_values, q2_values = model.q_value_head(state_features, action_features)
                    output_dict.update({"q1_values": q1_values, "q2_values": q2_values, "q_values": torch.min(q1_values, q2_values)})
                else: # Single QValueHead
                    q_values = model.q_value_head(state_features, action_features)
                    output_dict.update({"q_values": q_values})
            elif "action_tokens" in logits_processor_args: # If actions are not provided directly, try to get from from logits_processor_args
                action_tokens = logits_processor_args["action_tokens"]
                action_features = _convert_action_tokens_to_features(action_tokens, model)
                if hasattr(model.q_value_head, 'q1'):
                    q1_values, q2_values = model.q_value_head(state_features, action_features)
                    output_dict.update({"q1_values": q1_values, "q2_values": q2_values, "q_values": torch.min(q1_values, q2_values)})
                else:
                    q_values = model.q_value_head(state_features, action_features)
                    output_dict.update({"q_values": q_values})
        else:
            raise ValueError("Model does not have q_value_head but q_network=True")
    return output_dict