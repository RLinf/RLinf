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

import torch
from torch import nn
import torch.nn.functional as F
import types
from typing import Any, Optional, Union, Literal
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils import _pytree
from rlinf.utils.pytree import register_pytree_dataclasses

from transformers import AutoConfig, AutoModel, AutoProcessor, AutoModelForCausalLM
from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config

try:
    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
    print("[Module Pre-registration] Successfully registered Gr00tN1d6 architecture mapping to Transformers")
except Exception as e:
    pass

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

def patched_sample_time(self, batch_size, device, dtype):
    alpha = torch.tensor(self.config.noise_beta_alpha, dtype=torch.float32, device=device)
    beta_val = torch.tensor(self.config.noise_beta_beta, dtype=torch.float32, device=device)
    from torch.distributions import Beta
    temp_beta_dist = Beta(alpha, beta_val)
    
    sample = temp_beta_dist.sample([batch_size]).to(device, dtype=dtype)
    sample = (1 - sample) * self.config.noise_s
    return sample

def patched_process_backbone_output(self, backbone_output):
    backbone_features = backbone_output["backbone_features"]
    if hasattr(self, "vlln") and isinstance(self.vlln, nn.LayerNorm):
        orig_dtype = backbone_features.dtype
        weight = self.vlln.weight.float() if self.vlln.weight is not None else None
        bias = self.vlln.bias.float() if self.vlln.bias is not None else None
        
        backbone_features = F.layer_norm(
            backbone_features.float(),
            self.vlln.normalized_shape,
            weight,
            bias,
            self.vlln.eps
        ).to(orig_dtype) 
    else:
        backbone_features = self.vlln(backbone_features)

    backbone_output["backbone_features"] = backbone_features
    return backbone_output

def patched_action_head_forward(self, backbone_output, action_input):
    self.set_frozen_modules_to_eval_mode()
    backbone_output = self.process_backbone_output(backbone_output)

    vl_embeds = backbone_output.backbone_features
    device = vl_embeds.device
    embodiment_id = action_input.embodiment_id

    state_features = self.state_encoder(action_input.state, embodiment_id)

    if self.state_dropout_prob > 0:
        do_dropout = (torch.rand(state_features.shape[0], device=state_features.device) < self.state_dropout_prob)
        do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
        state_features = state_features * (1 - do_dropout) + self.mask_token * do_dropout

    if self.training and self.state_additive_noise_scale > 0:
        noise = torch.randn_like(state_features) * self.state_additive_noise_scale
        state_features = state_features + noise

    actions = action_input.action
    noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
    t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
    t = t[:, None, None]

    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise

    t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
    action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

    if self.config.add_pos_embed:
        pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
        pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
        action_features = action_features + pos_embs

    sa_embs = torch.cat((state_features, action_features), dim=1)
    vl_attn_mask = backbone_output.backbone_attention_mask

    if self.config.use_alternate_vl_dit:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
            image_mask=backbone_output.image_mask,
            backbone_attention_mask=backbone_output.backbone_attention_mask,
        )
    else:
        model_output, _ = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            encoder_attention_mask=vl_attn_mask,
            timestep=t_discretized,
            return_all_hidden_states=True,
        )

    pred = self.action_decoder(model_output, embodiment_id)
    pred_actions = pred[:, -actions.shape[1] :]
    action_mask = action_input.action_mask
    
    pred_actions = torch.nan_to_num(pred_actions, nan=0.0, posinf=1.0, neginf=-1.0)
    velocity = torch.nan_to_num(velocity, nan=0.0)

    pred_actions_fp32 = pred_actions.float()
    velocity_fp32 = velocity.float()
    action_mask_fp32 = action_mask.float()

    action_loss_fp32 = F.mse_loss(pred_actions_fp32, velocity_fp32, reduction="none") * action_mask_fp32
    loss_fp32 = action_loss_fp32.sum() / (action_mask_fp32.sum() + 1e-6)

    loss = loss_fp32.to(pred_actions.dtype)
    action_loss = action_loss_fp32.to(pred_actions.dtype)

    return {
        "loss": loss,
        "action_loss": action_loss,
        "action_mask": action_mask,
        "backbone_features": vl_embeds,
        "state_features": state_features,
    }

class GR00T_1_6_SFT_Model(Gr00tN1d6, BasePolicy):
    _supports_flash_attn_2 = True
    _supports_sdpa = True
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
        
        if hasattr(self, "action_head"):
            self.action_head.sample_time = types.MethodType(patched_sample_time, self.action_head)
            self.action_head.process_backbone_output = types.MethodType(patched_process_backbone_output, self.action_head)
            self.action_head.forward = types.MethodType(patched_action_head_forward, self.action_head)
            print("providing patch for Action Head (Monkey Patch)")

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

        print("[monitor loading] registering gradient monitors...")
        for name, param in self.named_parameters():
            if param.requires_grad:
                def create_check_grad_fn(layer_name):
                    def check_grad(grad):
                        if grad is not None:
                            if torch.isnan(grad).any() or torch.isinf(grad).any():
                                print(f"[gradient exposion] find NaN gradient! Layer: {layer_name}")
                                return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
                        return grad
                    return check_grad
                param.register_hook(create_check_grad_fn(name))
    def forward(self, forward_type=ForwardType.SFT, **kwargs):
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: dict, **kwargs):
        
        obs = data["observation"]
        actions = data["actions"] 

        for k, v in obs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                decoded_texts = []
                for row in v:
                    valid_bytes = row[row != 0].tolist()
                    decoded_texts.append(bytes(valid_bytes).decode('utf-8'))
                obs[k] = decoded_texts

        def safe_to_tensor(x):
            if x is None:
                return x
            if isinstance(x, str) or (isinstance(x, list) and len(x) > 0 and isinstance(x[0], str)):
                return x
            return torch.as_tensor(x, device=self.device).contiguous().clone()

        obs = _pytree.tree_map(safe_to_tensor, obs)
        
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.dim() == 2: 
            actions = actions.unsqueeze(1)
            
        model_inputs = self.apply_transforms(obs)
        
        actions = torch.nan_to_num(actions, nan=0.0)
        actions = torch.clamp(actions, min=-1.0, max=1.0)
        
        target_dim = 128
        
        padded_actions = torch.zeros((actions.shape[0], actions.shape[1], target_dim), 
                                     dtype=self.compute_dtype, device=actions.device)
        padded_actions[:, :, :actions.shape[-1]] = actions.to(dtype=self.compute_dtype)
        
        action_mask = torch.zeros((actions.shape[0], actions.shape[1], target_dim), 
                                 dtype=self.compute_dtype, device=actions.device)
        action_mask[:, :, :actions.shape[-1]] = 1.0
        
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
            print("[detect NaN Loss] force to zero and continue training to prevent collapse.")
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
try:
    AutoModel.register(Gr00tN1d6Config, GR00T_1_6_SFT_Model)
    AutoModelForCausalLM.register(Gr00tN1d6Config, GR00T_1_6_SFT_Model)
    AutoModelForCausalLM.register(Gr00tN1d6Config, GR00T_1_6_SFT_Model)
    print("[register model] successfully registered GR00T_1_6_SFT_Model！")
except Exception:
    # pass
    print(f"[register model] failed to register GR00T_1_6_SFT_Model: {e}")
def build_gr00t_dataloader(worker_instance, eval_dataset: bool = False):
    import torch
    from torch.utils.data import DataLoader
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    print(f"[Lazy Load] Successfully triggered GR00T model file registration! Loading LeRobot Dataset...")
    

    dataset = LeRobotDataset(
        repo_id=worker_instance.cfg.data.train_data_paths,
        # local_files_only=True
        video_backend="pyav"
    )

    def gr00t_collate_fn(batch):
        action_key = next((k for k in batch[0].keys() if "action" in k.lower()), None)
        if action_key is None:
            raise KeyError("Could not find an action key!")

        actions = torch.stack([item[action_key] for item in batch])
        if actions.dim() == 2:
            actions = actions.unsqueeze(1) 

        obs = {}
        for key in batch[0].keys():
            if key != action_key:
                item_val = batch[0][key]
                if isinstance(item_val, str) or (isinstance(item_val, (list, tuple)) and len(item_val)>0 and isinstance(item_val[0], str)):
                    byte_ts = []
                    for item in batch:
                        text = item[key]
                        if isinstance(text, (list, tuple)): text = text[0]
                        byte_ts.append(torch.tensor(list(text.encode('utf-8')), dtype=torch.uint8))
                    obs[key] = torch.nn.utils.rnn.pad_sequence(byte_ts, batch_first=True, padding_value=0)
                elif isinstance(item_val, torch.Tensor):
                    obs[key] = torch.stack([item[key] for item in batch])
                else:
                    try: obs[key] = torch.tensor([item[key] for item in batch])
                    except: obs[key] = [item[key] for item in batch]
        return obs, actions

    dataloader = DataLoader(
        dataset,
        batch_size=worker_instance.cfg.actor.micro_batch_size * worker_instance._world_size,
        shuffle=not eval_dataset,
        num_workers=4,
        collate_fn=gr00t_collate_fn,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader, None