from typing import Dict, List

import torch

# Model namespace configuration for buffer keys
MODEL_BUFFER_NAMESPACES = {
    "openvla": {
        "observation_keys": [
            "input_ids",
            "pixel_values",
            "attention_mask"
        ],
        "result_keys": [
            "chunk_action_tokens",
            "prev_logprobs",
            "prev_values"
        ]
    },
    "openvla_oft": {
        "observation_keys": [
            "input_ids",
            "pixel_values",
            "attention_mask"
        ],
        "result_keys": [
            "chunk_action_tokens",
            "prev_logprobs",
            "prev_values"
        ]
    },
    "pi0": {
        "observation_keys": [
            "observation.images.image",
            "observation.images.wrist_image",
            "observation.state",
            "lang_tokens",
            "lang_masks"
        ],
        "result_keys": [
            "chains",
            "prev_values",
            "prev_logprobs",
            "denoise_inds"
        ]
    }
}


def get_model_buffer_namespace(model_name: str) -> Dict[str, List[str]]:
    """Get buffer namespace configuration for a specific model."""
    if model_name not in MODEL_BUFFER_NAMESPACES:
        raise ValueError(f"Unsupported model: {model_name}. Supported models: {list(MODEL_BUFFER_NAMESPACES.keys())}")
    return MODEL_BUFFER_NAMESPACES[model_name]


def append_to_buffer(buffer_list: List[Dict], stage_idx: int, namespace: Dict[str, List[str]],
                    processed_obs: Dict[str, torch.Tensor], result: Dict[str, torch.Tensor]) -> None:
    """Automatically append data to buffer based on model namespace."""

    # Append observation keys
    for key in namespace["observation_keys"]:
        if key in processed_obs:
            if key == "attention_mask":
                # Special handling for attention_mask
                buffer_list[stage_idx][key].append(
                    processed_obs[key].bool().cpu().contiguous()
                )
            else:
                buffer_list[stage_idx][key].append(
                    processed_obs[key].cpu().contiguous()
                )

    # Append result keys
    for key in namespace["result_keys"]:
        if key in result:
            buffer_list[stage_idx][key].append(
                result[key].cpu().contiguous()
            )
