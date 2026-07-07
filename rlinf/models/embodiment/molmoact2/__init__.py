import types

import torch
from omegaconf import DictConfig


def _molmoact2_predict_action_batch(self, env_obs=None, mode="eval", **kwargs):
    """
    RLinf adapter for MolmoAct2.

    RLinf rollout worker calls:
        model.predict_action_batch(env_obs=env_obs, mode="eval")

    Official MolmoAct2 policy provides:
        model.select_action(batch)

    This adapter converts RLinf env_obs into MolmoAct2 batch format.
    """
    if env_obs is None:
        raise ValueError("MolmoAct2 predict_action_batch requires env_obs.")

    batch = {}

    # RLinf LIBERO provides:
    #   main_images  -> agent/front view
    #   wrist_images -> wrist view
    # MolmoAct2-LIBERO expects camera order: [agentview_rgb, wrist_rgb].
    #
    # Use separate observation.images.* keys instead of batch["images"].
    # Otherwise MolmoAct2 may infer the two views as batch_size=2.
    if "main_images" in env_obs:
        main_images = env_obs["main_images"]
        if "wrist_images" in env_obs and env_obs["wrist_images"] is not None:
            batch["observation.images.agentview"] = main_images
            batch["observation.images.wrist"] = env_obs["wrist_images"]
        else:
            batch["image"] = main_images
    elif "image" in env_obs:
        main_images = env_obs["image"]
        batch["image"] = main_images
    else:
        raise KeyError(f"Cannot find image in env_obs. Available keys: {list(env_obs.keys())}")

    batch_size = main_images.shape[0]

    # Language instruction
    if "task_descriptions" in env_obs:
        task_descriptions = env_obs["task_descriptions"]
        if isinstance(task_descriptions, str):
            task_descriptions = [task_descriptions]
        batch["language_instruction"] = task_descriptions
    elif "language_instruction" in env_obs:
        batch["language_instruction"] = env_obs["language_instruction"]
    elif "instruction" in env_obs:
        batch["language_instruction"] = env_obs["instruction"]
    else:
        batch["language_instruction"] = [""] * batch_size

    # Raw robot state for MolmoAct2-LIBERO.
    if "states" in env_obs:
        batch["state"] = env_obs["states"]
    elif "proprio" in env_obs:
        batch["state"] = env_obs["proprio"]
    elif "robot_states" in env_obs:
        batch["state"] = env_obs["robot_states"]

    with torch.no_grad():
        actions = self.select_action(batch)

    if isinstance(actions, torch.Tensor):
        actions = actions.detach().cpu()

    # RLinf expects action chunks: [B, num_chunks, action_dim]
    if actions.ndim == 2:
        chunk_actions = actions.unsqueeze(1)
    elif actions.ndim == 3:
        chunk_actions = actions
    else:
        raise ValueError(f"Unexpected MolmoAct2 action shape: {actions.shape}")

    batch_size = chunk_actions.shape[0]
    device = chunk_actions.device

    result = {
        "prev_logprobs": torch.zeros(batch_size, 1, device=device),
        "prev_values": torch.zeros(batch_size, 1, device=device),
        "forward_inputs": {
            "action": chunk_actions,
        },
    }

    return chunk_actions, result


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    """
    Minimal MolmoAct2 wrapper for RLinf.

    First-stage goal:
    - reuse official MolmoAct2 LeRobot policy
    - support inference / eval / rollout
    - do not implement PPO training yet
    """

    from lerobot.policies.molmoact2.configuration_molmoact2 import MolmoAct2Config
    from lerobot.policies.molmoact2.modeling_molmoact2 import MolmoAct2Policy

    checkpoint_path = cfg.get("checkpoint_path", None) or cfg.get("model_path", None)
    if checkpoint_path is None:
        raise ValueError("MolmoAct2 requires `checkpoint_path` or `model_path` in config.")

    molmo_cfg = MolmoAct2Config(
        checkpoint_path=checkpoint_path,
        seq_len=cfg.get("seq_len", None),
        num_steps=cfg.get("num_steps", None),
        inference_action_mode=cfg.get("inference_action_mode", "continuous"),
        discrete_action_tokenizer=cfg.get(
            "discrete_action_tokenizer",
            "allenai/MolmoAct2-FAST-Tokenizer",
        ),
        enable_depth_reasoning=cfg.get("enable_depth_reasoning", False),
        num_depth_tokens_per_image=cfg.get("num_depth_tokens_per_image", None),
        verbose=cfg.get("verbose", False),
        norm_tag=cfg.get("norm_tag", ""),
    )

    model = MolmoAct2Policy(molmo_cfg)

    # Add RLinf-required action interface
    model.predict_action_batch = types.MethodType(_molmoact2_predict_action_batch, model)

    return model
