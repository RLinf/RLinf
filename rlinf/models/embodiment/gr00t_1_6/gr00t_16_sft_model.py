import torch
from torch import nn
from typing import Any, Optional, Union, Literal
from pathlib import Path
import numpy as np
from PIL import Image

from transformers import AutoConfig, AutoModel, AutoProcessor
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

try:
    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
    print("[Module Pre-registration] Successfully registered Gr00tN1d6 architecture mapping to Transformers")
except Exception as e:
    pass

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

class GR00T_1_6_SFT_Model(Gr00tN1d6, BasePolicy):
    def __init__(
        self,
        config: Gr00tN1d6Config,
        embodiment_tag: Optional[str] = None,
        local_model_path: Optional[str] = None,
        compute_dtype: torch.dtype = torch.bfloat16,
        modality_config: Optional[Any] = None,
        modality_transform: Optional[Any] = None,
        **kwargs,
    ):
        if embodiment_tag is None:
            from gr00t.data.embodiment_tags import EmbodimentTag
            embodiment_tag = EmbodimentTag.ROBOCASA_PANDA_OMRON 
            
        if local_model_path is None:
            local_model_path = getattr(config, "_name_or_path", "/workspace/RLinf/GR00T-N1.6-3B")

        transformers_loading_kwargs = kwargs.pop("transformers_loading_kwargs", {"trust_remote_code": True})
        super().__init__(config, transformers_loading_kwargs=transformers_loading_kwargs, **kwargs)
        
        self.embodiment_tag = embodiment_tag
        self.compute_dtype = compute_dtype
        self.model_path = Path(local_model_path)
        
        if modality_config is None or modality_transform is None:
            print("Loading Processor...")
            processor = AutoProcessor.from_pretrained(str(local_model_path), trust_remote_code=True)
            modality_transform = processor
            modality_config = getattr(processor, "modality_config", None) 
            print("Processor loaded successfully.")

        self._modality_config = modality_config
        self._modality_transform = modality_transform
        
        self.to(self.compute_dtype)
        print(f"all model residual parameters have been forcibly unified and shuffled to: {self.compute_dtype}")
    
        
        self.requires_grad_(False) 
        
        if hasattr(self, "action_head") and self.action_head is not None:
            self.action_head.requires_grad_(True)
                
            if hasattr(self.action_head, "model") and hasattr(self.action_head.model, "timestep_encoder"):
                self.action_head.model.timestep_encoder.requires_grad_(False)
                print("Precisely disabled timestep_encoder gradient updates (to prevent NaN)")

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[SFT Override] Number of parameters participating in training: {trainable_params / 1e6:.2f} M")

    def forward(self, forward_type=ForwardType.SFT, **kwargs):
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: dict, **kwargs):
        obs = data["observation"]
        actions = data["actions"] 
        if actions.dim() == 2: actions = actions.unsqueeze(1)
            
        model_inputs = self.apply_transforms(obs)
        
        actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        
        target_dim = 128
        padded_actions = torch.zeros((actions.shape[0], actions.shape[1], target_dim), 
                                     dtype=self.compute_dtype, device=actions.device)
        padded_actions[:, :, :actions.shape[-1]] = actions.to(dtype=self.compute_dtype)
        
        # action_mask = torch.zeros((actions.shape[0], actions.shape[1], target_dim), 
        #                          dtype=self.compute_dtype, device=actions.device)
        # action_mask[:, :, :actions.shape[-1]] = 1.0
        action_mask = torch.ones((actions.shape[0], actions.shape[1], target_dim), 
                                 dtype=self.compute_dtype, device=actions.device)
        
        padded_actions = torch.nan_to_num(padded_actions, nan=0.0)
        
        model_inputs["action"] = padded_actions
        model_inputs["action_mask"] = action_mask
        
        for key in ["eagle_input_ids", "eagle_attention_mask", "eagle_pixel_values", "eagle_image_sizes"]:
            model_inputs.pop(key, None)

        with torch.amp.autocast('cuda', enabled=False):
            model_inputs_fp32 = {k: v.float() if isinstance(v, torch.Tensor) and torch.is_floating_point(v) else v 
                                for k, v in model_inputs.items()}
            
            backbone_in, action_in = super().prepare_input(model_inputs_fp32)
            backbone_out = self.backbone(backbone_in)
            outputs = self.action_head(backbone_out, action_in)

        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        
        if torch.isnan(loss):
            print("[Detected NaN Loss] Forcibly zeroing to protect weights")
            loss = torch.zeros([], device=loss.device, dtype=loss.dtype, requires_grad=True)
            
        return loss

    def apply_transforms(self, obs: dict) -> dict:
        class SimulationContent:
            def __init__(self, embodiment, states, actions, images, text):
                self.embodiment = embodiment
                self.states = states
                self.actions = actions
                self.images = images
                self.text = text

        batch_size = len(next(iter(obs.values())))
        
        text_key = None
        image_keys = []
        state_keys = []
        for k in obs.keys():
            k_lower = k.lower()
            if "task" in k_lower or "lang" in k_lower or "instruction" in k_lower:
                text_key = k
            elif "image" in k_lower or "rgb" in k_lower or "cam" in k_lower:
                image_keys.append(k)
            else:
                state_keys.append(k)
                
        processed_outputs = []
        
        for i in range(batch_size):
            text = obs[text_key][i] if text_key else ""
            if isinstance(text, (list, np.ndarray)):
                text = str(text[0]) if len(text) > 0 else ""
            else:
                text = str(text)
                
            states_dict = {}
            for k in state_keys:
                v = obs[k][i]
                if isinstance(v, torch.Tensor):
                    v = v.cpu().float().numpy()
                states_dict[k] = np.array(v)
                
            ref_T = next(iter(states_dict.values())).shape[0] if states_dict else 1
            robocasa_requirements = {
                "end_effector_position_relative": 3,
                "end_effector_rotation_relative": 4,
                "gripper_qpos": 2,
                "base_position": 3,
                "base_rotation": 4
            }
            for req_k, req_dim in robocasa_requirements.items():
                if req_k not in states_dict:
                    states_dict[req_k] = np.zeros((ref_T, req_dim), dtype=np.float32)

            raw_images_list = []
            for img_k in image_keys:
                img_data = obs[img_k][i]
                if isinstance(img_data, torch.Tensor):
                    img_data = img_data.cpu().float().numpy()
                
                frames = []
                if img_data.ndim == 3:
                    img_data = np.expand_dims(img_data, 0)
                
                for t in range(img_data.shape[0]):
                    frame = img_data[t]
                    if frame.shape[0] in [1, 3] and frame.shape[2] > 3:
                        frame = np.transpose(frame, (1, 2, 0))
                    if frame.dtype != np.uint8:
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    frames.append(Image.fromarray(frame))
                raw_images_list.append(frames)

            images_dict = {}
            req_img_keys = []
            try:
                tag_val = self.embodiment_tag.value if hasattr(self.embodiment_tag, "value") else str(self.embodiment_tag)
                req_img_keys = self._modality_transform.modality_configs[tag_val]["video"].modality_keys
            except Exception:
                req_img_keys = ["res256_image_side_0", "res256_image_wrist_0", "res256_image_front_0"]
            
            self.image_nums = len(req_img_keys)

            for idx, r_key in enumerate(req_img_keys):
                if idx < len(raw_images_list):
                    images_dict[r_key] = raw_images_list[idx]
                else:
                    images_dict[r_key] = raw_images_list[-1] if raw_images_list else []

            content = SimulationContent(
                embodiment=self.embodiment_tag,
                states=states_dict,
                actions=None,
                images=images_dict,
                text=text
            )

            is_training = False
            if hasattr(self._modality_transform, "training"):
                is_training = self._modality_transform.training
                if is_training:
                    self._modality_transform.eval()

            messages = [{"role": "user", "content": content}]
            out = self._modality_transform(messages=messages)
            
            if is_training:
                self._modality_transform.train()
            processed_outputs.append(out)
            
        collated_batch = self._modality_transform.collator(processed_outputs)
        
        if hasattr(collated_batch, "data") and "inputs" in collated_batch.data:
            batched_out = collated_batch.data["inputs"]
        elif "inputs" in collated_batch:
            batched_out = collated_batch["inputs"]
        else:
            batched_out = dict(collated_batch)
            
        if "input_ids" in batched_out:
            batched_out["eagle_input_ids"] = batched_out["input_ids"]
        if "attention_mask" in batched_out:
            batched_out["eagle_attention_mask"] = batched_out["attention_mask"]
        if "pixel_values" in batched_out:
            batched_out["eagle_pixel_values"] = batched_out["pixel_values"]
        if "image_sizes" in batched_out:
            batched_out["eagle_image_sizes"] = batched_out["image_sizes"]

        if "eagle_pixel_values" not in batched_out and "images" in batched_out:
            batched_out["eagle_pixel_values"] = batched_out["images"]
            batched_out["pixel_values"] = batched_out["images"]
                
        return batched_out

    def default_forward(self, *args, **kwargs):
        raise NotImplementedError("Pure SFT model does not support default_forward")

    def predict_action_batch(self, *args, **kwargs):
        raise NotImplementedError("Pure SFT model does not support predict_action_batch")