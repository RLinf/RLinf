"""
Value Critic Models for Value Prediction.

Uses expert forward mode: Gemma expert + [CLS] -> categorical value prediction.
"""

import glob
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: N812
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .configuration import ValueCriticConfig, VLMBaseConfig, get_config
from .paligemma_with_multi_expert import (
    PaliGemmaWithMultiExpertModel,
    _requires_uniform_dtype,
)
from .siglip_gemma3_with_multi_expert import SiglipGemma3WithMultiExpert

logger = logging.getLogger(__name__)


# =============================================================================
# Free Functions (from openpi/modeling_pi0.py)
# =============================================================================


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from padding and AR masks.

    Tokens can attend to valid input tokens which have a cumulative mask_ar
    smaller or equal to theirs.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


# =============================================================================
# VLMPreTrainedModel (HF integration for from_pretrained)
# =============================================================================


class VLMPreTrainedModel(PreTrainedModel):
    """Base pretrained model with smart loading for VLM-based checkpoints."""

    config_class = VLMBaseConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True
    _tied_weights_keys = []

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def _set_gradient_checkpointing(
        self, enable: bool = True, gradient_checkpointing_func=None
    ):
        if gradient_checkpointing_func is None:
            gradient_checkpointing_func = torch.utils.checkpoint.checkpoint
        if hasattr(self, "model"):
            if enable:
                self.model.gradient_checkpointing_enable()
            else:
                self.model.gradient_checkpointing_disable()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Override from_pretrained to handle problematic safetensors files."""
        logger.info(
            f"VLM from_pretrained: Loading from {pretrained_model_name_or_path}"
        )

        single_file_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )
        index_file_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors.index.json"
        )
        sharded_pattern = os.path.join(
            pretrained_model_name_or_path, "model-*.safetensors"
        )
        has_single_file = os.path.exists(single_file_path)
        has_index_file = os.path.exists(index_file_path)
        sharded_parts = glob.glob(sharded_pattern)
        has_sharded_files = len(sharded_parts) > 0

        if has_single_file:
            try:
                with safe_open(single_file_path, framework="pt") as f:
                    metadata = f.metadata()
                invalid_metadata = (
                    not metadata
                    or "format" not in metadata
                    or metadata.get("format") is None
                )
            except Exception as exc:
                logger.warning(
                    "Failed to read single model.safetensors metadata (%s) - attempting manual loading",
                    exc,
                )
                return cls._load_from_safetensors(
                    pretrained_model_name_or_path, **kwargs
                )

            if invalid_metadata:
                logger.warning(
                    "Single model.safetensors has invalid metadata - attempting manual loading"
                )
                return cls._load_from_safetensors(
                    pretrained_model_name_or_path, **kwargs
                )
        elif has_index_file or has_sharded_files:
            logger.info(
                "Detected HF-sharded checkpoint - will rely on HuggingFace loader"
            )
        else:
            logger.warning(
                "No model.safetensors file detected; attempting standard HuggingFace loading."
            )

        safe_kwargs = kwargs.copy()
        original_device_map = kwargs.get("device_map")

        is_distributed = (
            os.environ.get("WORLD_SIZE") is not None
            or os.environ.get("LOCAL_RANK") is not None
            or os.environ.get("RANK") is not None
        )

        if is_distributed:
            logger.info(
                "Detected distributed training - letting Accelerate handle device placement"
            )
            safe_kwargs.update(
                {
                    "device_map": None,
                    "low_cpu_mem_usage": False,
                    "torch_dtype": kwargs.get("torch_dtype", torch.float32),
                }
            )
        else:
            if original_device_map is not None:
                logger.info(
                    f"Single GPU/inference mode - using device_map: {original_device_map}"
                )
                safe_kwargs.update(
                    {
                        "torch_dtype": kwargs.get("torch_dtype", torch.float32),
                    }
                )
            else:
                logger.info("No device_map specified - using default CPU loading")
                safe_kwargs.update(
                    {
                        "device_map": None,
                        "low_cpu_mem_usage": False,
                        "torch_dtype": kwargs.get("torch_dtype", torch.float32),
                    }
                )

        if "config" not in safe_kwargs or safe_kwargs["config"] is None:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as cfg_file:
                    config_dict = json.load(cfg_file)
                safe_kwargs["config"] = cls.config_class.from_dict(config_dict)
            else:
                loaded_config, _ = cls.config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    return_unused_kwargs=True,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )
                safe_kwargs["config"] = loaded_config

        model = super().from_pretrained(pretrained_model_name_or_path, **safe_kwargs)
        logger.info("Successfully loaded model using HuggingFace infrastructure")

        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        if torch_dtype == torch.bfloat16 and not _requires_uniform_dtype():
            cls._apply_mixed_precision(model)
        elif _requires_uniform_dtype():
            logger.info(
                "Parameter sharding detected (FSDP/Zero-3): using uniform dtype"
            )

        return model

    @classmethod
    def _apply_mixed_precision(cls, model):
        """Keep action projection layers in float32 for flow matching compatibility."""
        action_proj_patterns = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
            "state_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]
        for name, param in model.named_parameters():
            if any(pattern in name for pattern in action_proj_patterns):
                param.data = param.data.to(dtype=torch.float32)
                logger.debug(f"Kept {name} in float32 for flow matching compatibility")
        logger.info("Applied mixed precision: action projection layers kept in float32")

    @classmethod
    def _load_from_safetensors(cls, pretrained_model_name_or_path, **kwargs):
        """Manual loading from a single safetensors file with potentially corrupted metadata."""
        logger.info("Attempting manual loading from single safetensors file")

        config = kwargs.get("config")
        if config is None:
            config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config_dict = json.load(f)
                config = cls.config_class.from_dict(config_dict)
            else:
                raise ValueError(
                    f"No config.json found in {pretrained_model_name_or_path} and no config provided"
                )

        model = cls(config)

        safetensors_path = os.path.join(
            pretrained_model_name_or_path, "model.safetensors"
        )
        logger.info(f"Manually loading state dict from {safetensors_path}")

        try:
            state_dict = load_file(safetensors_path, device="cpu")
            logger.info(
                f"Successfully loaded {len(state_dict)} tensors from safetensors file"
            )

            sample_key = next(iter(state_dict.keys())) if state_dict else ""
            if sample_key.startswith("model."):
                missing_keys, unexpected_keys = model.load_state_dict(
                    state_dict, strict=False
                )
            else:
                missing_keys, unexpected_keys = model.model.load_state_dict(
                    state_dict, strict=False
                )

            if missing_keys:
                logger.warning(f"Missing keys when loading state dict: {missing_keys}")
            if unexpected_keys:
                logger.warning(
                    f"Unexpected keys when loading state dict: {unexpected_keys}"
                )

            logger.info("Successfully loaded model from single safetensors file")

            torch_dtype = kwargs.get("torch_dtype", torch.float32)
            if _requires_uniform_dtype():
                if torch_dtype == torch.bfloat16:
                    model.to(dtype=torch.bfloat16)
                else:
                    model.to(dtype=torch.float32)
            else:
                if torch_dtype == torch.bfloat16:
                    model.model.paligemma_with_expert.to_bfloat16_for_selected_params(
                        "bfloat16"
                    )
                elif torch_dtype == torch.float32:
                    model.model.paligemma_with_expert.to_bfloat16_for_selected_params(
                        "float32"
                    )

            return model
        except Exception as e:
            logger.error(f"Failed to manually load from safetensors file: {e}")
            raise ValueError(
                f"Could not load model from {pretrained_model_name_or_path}: {e}"
            )


class CriticPreTrainedModel(VLMPreTrainedModel):
    config_class = VLMBaseConfig


# =============================================================================
# Output Dataclasses
# =============================================================================


@dataclass
class CriticOutput(ModelOutput):
    """Output for critic models."""

    loss: Optional[torch.FloatTensor] = None
    predicted_values: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    expert_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    cat_acc_best: Optional[torch.FloatTensor] = None
    cat_acc_neighbor: Optional[torch.FloatTensor] = None
    cat_mae: Optional[torch.FloatTensor] = None


# =============================================================================
# VLMObservationEncoder (base class providing observation processing/embedding)
# =============================================================================


class VLMObservationEncoder(nn.Module):
    """Base class providing observation processing and embedding.

    Stripped to only the methods inherited by ValueCriticModel.
    ValueCriticModel bypasses __init__ with nn.Module.__init__(self),
    so no __init__ is defined here.
    """

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _get_model_dtype(self) -> torch.dtype:
        """Get the dtype of the backbone model's attention weights."""
        if getattr(self, "backbone_variant", "paligemma") == "siglip_gemma3":
            return self.paligemma_with_expert.gemma3.model.layers[
                0
            ].self_attn.q_proj.weight.dtype
        return self.paligemma_with_expert.paligemma.language_model.layers[
            0
        ].self_attn.q_proj.weight.dtype

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        dtype = self._get_model_dtype()
        return torch.where(
            att_2d_masks_4d,
            torch.tensor(0.0, dtype=dtype, device=att_2d_masks.device),
            torch.tensor(-2.3819763e38, dtype=dtype, device=att_2d_masks.device),
        )

    def _preprocess_observation(self, observation):
        """Extract observation components with unified masks (no state)."""
        images = observation["images"]
        image_masks = observation.get("image_masks", {})
        tokenized_prompt = observation["tokenized_prompt"]
        tokenized_prompt_mask = observation["tokenized_prompt_mask"]

        batch_size, seq_len = tokenized_prompt.shape
        device = tokenized_prompt.device

        token_ar_mask = observation.get("token_ar_mask")
        if token_ar_mask is None:
            token_ar_mask = torch.zeros(
                (batch_size, seq_len), dtype=torch.long, device=device
            )

        token_loss_mask = observation.get("token_loss_mask")
        if token_loss_mask is None:
            token_loss_mask = torch.zeros(
                (batch_size, seq_len), dtype=torch.bool, device=device
            )

        token_kv_cache_mask = observation.get("token_kv_cache_mask")
        if token_kv_cache_mask is None:
            token_kv_cache_mask = tokenized_prompt_mask.clone()

        sorted_keys = sorted(images.keys())
        img_list = [images[k] for k in sorted_keys]
        img_mask_list = [
            image_masks.get(k, torch.ones(batch_size, dtype=torch.bool, device=device))
            for k in sorted_keys
        ]

        return (
            img_list,
            img_mask_list,
            tokenized_prompt,
            tokenized_prompt_mask,
            token_ar_mask,
            token_loss_mask,
            token_kv_cache_mask,
        )

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks, token_ar_mask=None
    ):
        """Embed images and language tokens (with token_ar_mask)."""
        embs, pad_masks, ar_masks = [], [], []
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(
                self.paligemma_with_expert.embed_image, img
            )
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))
            ar_masks.append(
                torch.zeros(bsize, num_img_embs, dtype=torch.long, device=device)
            )

        def embed_lang(tokens):
            emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return emb * math.sqrt(emb.shape[-1])

        lang_emb = self._apply_checkpoint(embed_lang, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        if token_ar_mask is None:
            token_ar_mask = torch.zeros(
                bsize, lang_masks.shape[1], dtype=torch.long, device=device
            )
        ar_masks.append(token_ar_mask)

        return (
            torch.cat(embs, dim=1),
            torch.cat(pad_masks, dim=1),
            torch.cat(ar_masks, dim=1),
        )


# =============================================================================
# Value Head
# =============================================================================


class ValueHead(nn.Module):
    """Value prediction head with learnable CLS embedding and categorical projection."""

    def __init__(
        self,
        hidden_size: int,
        num_bins: int,
        v_min: float,
        v_max: float,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.cls_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.cls_embedding.weight, std=0.02)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.value_proj = nn.Linear(hidden_size, num_bins)
        self.register_buffer(
            "atoms", torch.linspace(v_min, v_max, num_bins), persistent=False
        )
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (num_bins - 1)

    def get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get CLS embedding expanded to batch size. Returns [B, 1, hidden_size]."""
        # Use embedding lookup instead of reading `.weight` directly.
        # Under FSDP, `.weight` can be a view into a flattened parameter and may
        # trigger autograd view/inplace conflicts during backward.
        cls_token_ids = torch.zeros(
            batch_size, 1, dtype=torch.long, device=self.cls_embedding.weight.device
        )
        return self.cls_embedding(cls_token_ids)

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to value logits."""
        hidden_states = hidden_states.to(self.value_proj.weight.dtype)
        hidden_states = self.dropout(hidden_states)
        return self.value_proj(hidden_states)


# =============================================================================
# ValueCriticModel (FSDP-enhanced, from top-level modeling_critic.py)
# =============================================================================


class ValueCriticModel(VLMObservationEncoder):
    """Value function V(s) with VLM observation encoding.

    Uses expert mode: Gemma expert + [CLS] -> categorical value prediction.
    """

    def __init__(self, config: ValueCriticConfig):
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = True
        self.backbone_variant = getattr(config, "backbone_variant", "paligemma")

        expert_config = get_config(config.critic_expert_variant)

        logger.info(
            f"Creating ValueCritic: expert={config.critic_expert_variant}, "
            f"backbone={self.backbone_variant}"
        )

        expert_configs = {"value": expert_config}

        if self.backbone_variant == "siglip_gemma3":
            siglip_path = getattr(config, "siglip_path", "")
            gemma3_path = getattr(config, "gemma3_path", "")
            if not siglip_path or not gemma3_path:
                raise ValueError(
                    "backbone_variant='siglip_gemma3' requires both "
                    f"siglip_path and gemma3_path. Got siglip_path='{siglip_path}', "
                    f"gemma3_path='{gemma3_path}'"
                )
            self.paligemma_with_expert = SiglipGemma3WithMultiExpert(
                expert_configs=expert_configs,
                siglip_path=siglip_path,
                gemma3_path=gemma3_path,
                precision=config.dtype,
                freeze_vision_encoder=getattr(config, "freeze_vision_encoder", False),
                freeze_vlm=getattr(config, "freeze_vlm", False),
            )
        else:
            paligemma_config = get_config(config.paligemma_variant)
            use_adarms = [False, {"value": False}]
            self.paligemma_with_expert = PaliGemmaWithMultiExpertModel(
                vlm_config=paligemma_config,
                expert_configs=expert_configs,
                use_adarms=use_adarms,
                precision=config.dtype,
                freeze_vision_encoder=getattr(config, "freeze_vision_encoder", False),
                freeze_vlm=getattr(config, "freeze_vlm", False),
            )

        self.gradient_checkpointing_enabled = False
        self._expert_config = expert_config

        self.expert_width = expert_config.width
        self.value_head = ValueHead(
            hidden_size=expert_config.width,
            num_bins=config.num_bins,
            v_min=config.v_min,
            v_max=config.v_max,
            dropout=getattr(config, "value_dropout", 0.0),
        )
        self.num_bins = config.num_bins
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.delta_z = (config.v_max - config.v_min) / (config.num_bins - 1)

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        if self.paligemma_with_expert.freeze_vlm:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                "ValueHead",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                "ValueHead",
            ]
        if self.backbone_variant == "siglip_gemma3":
            no_split_modules.extend(["Gemma3RMSNorm","Gemma3DecoderLayer"])
        return no_split_modules
            

    @property
    def _no_split_names(self) -> list[str]:
        return ["lm_head", "cls_embedding"]

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        if self.backbone_variant == "siglip_gemma3":
            self.paligemma_with_expert.gemma3.model.gradient_checkpointing = True
            self.paligemma_with_expert.vision_tower.gradient_checkpointing = True
        else:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = (
                True
            )
        for expert in self.paligemma_with_expert.experts.values():
            expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for ValueCritic")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        if self.backbone_variant == "siglip_gemma3":
            self.paligemma_with_expert.gemma3.model.gradient_checkpointing = False
            self.paligemma_with_expert.vision_tower.gradient_checkpointing = False
        else:
            self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
            self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = (
                False
            )
        for expert in self.paligemma_with_expert.experts.values():
            expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for ValueCritic")

    def get_cls_embedding(self, batch_size: int) -> Tensor:
        return self.value_head.get_cls_embedding(batch_size)

    def embed_suffix(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Create suffix with [CLS] token for value prediction."""
        cls_emb = self.get_cls_embedding(batch_size)
        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=cls_emb.device)
        ar_mask = torch.ones(batch_size, 1, dtype=torch.long, device=cls_emb.device)
        return cls_emb, pad_mask, ar_mask

    def forward(self, observation, target_values=None, **kwargs) -> CriticOutput:
        """Forward pass through expert mode."""
        (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            token_ar_mask,
            _,
            _,
        ) = self._preprocess_observation(observation)

        batch_size = lang_tokens.shape[0]

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        stop_gradient = getattr(self.config, "stop_gradient_to_vlm", False)
        values, hidden_states, logits, probs, backward_anchor = self._forward_expert(
            prefix_embs,
            prefix_pad_masks,
            prefix_ar_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_ar_masks,
            stop_gradient_to_vlm=stop_gradient,
        )

        expert_loss = None
        cat_metrics = None
        if target_values is not None:
            expert_loss, cat_metrics = self._compute_categorical_loss(
                logits, target_values
            )
            if backward_anchor is not None:
                expert_loss = expert_loss + backward_anchor

        expert_loss_mean = expert_loss.mean() if expert_loss is not None else None

        return CriticOutput(
            loss=expert_loss_mean,
            predicted_values=values,
            logits=logits,
            probs=probs,
            atoms=self.value_head.atoms,
            expert_loss=expert_loss_mean,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            cat_mae=cat_metrics["mae"] if cat_metrics else None,
        )

    def _forward_expert(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_ar_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ):
        """Forward through VLM + value expert."""
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # Use two-stage forward when VLM is frozen under FSDP to avoid
        # view/inplace errors in interleaved attention.
        # 800m always uses two-stage (no interleaved support for Gemma3).
        use_two_stage = self.backbone_variant == "siglip_gemma3" or getattr(
            self.paligemma_with_expert, "freeze_vlm", False
        )
        if use_two_stage:
            prefix_out, suffix_out = self._forward_expert_two_stage(
                prefix_embs=prefix_embs,
                prefix_pad_masks=prefix_pad_masks,
                prefix_ar_masks=prefix_ar_masks,
                suffix_embs=suffix_embs,
                suffix_pad_masks=suffix_pad_masks,
                suffix_ar_masks=suffix_ar_masks,
                stop_gradient_to_vlm=stop_gradient_to_vlm,
            )
            cls_hidden = suffix_out[:, -1, :].to(
                self.value_head.value_proj.weight.dtype
            )
            values, logits, probs = self._compute_value_from_hidden(cls_hidden)
            backward_anchor = self._build_vlm_backward_anchor(
                prefix_out=prefix_out, stop_gradient_to_vlm=stop_gradient_to_vlm
            )
            return values, cls_hidden, logits, probs, backward_anchor

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_masks = torch.cat([prefix_ar_masks, suffix_ar_masks], dim=1)
        attn_mask = make_att_2d_masks(pad_masks, ar_masks)
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(
            prefix_embs, suffix_embs, attn_mask_4d, position_ids, detach_kv
        ):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, None],
                expert_name="value",
                detach_prefix_for_suffix=detach_kv,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func,
            prefix_embs,
            suffix_embs,
            attn_mask_4d,
            position_ids,
            stop_gradient_to_vlm,
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        backward_anchor = self._build_vlm_backward_anchor(
            prefix_out=prefix_out, stop_gradient_to_vlm=stop_gradient_to_vlm
        )
        return values, cls_hidden, logits, probs, backward_anchor

    def _build_vlm_backward_anchor(
        self, prefix_out: Optional[Tensor], stop_gradient_to_vlm: bool
    ) -> Optional[Tensor]:
        """Build a zero-weight anchor so FSDP tracks two-stage Gemma3 backward."""
        if (
            not self.training
            or self.backbone_variant != "siglip_gemma3"
            or stop_gradient_to_vlm
            or prefix_out is None
            or not prefix_out.requires_grad
        ):
            return None
        # Gradients flow via DynamicCache in two-stage Gemma3 forward. Anchoring
        # an explicit output tensor keeps FSDP's forward/backward state aligned.
        return prefix_out.sum() * 0.0

    def _forward_expert_two_stage(
        self,
        prefix_embs,
        prefix_pad_masks,
        prefix_ar_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Two-stage expert forward with KV cache (for frozen VLM under FSDP)."""
        # Phase 1: prefill frozen VLM to get KV cache.
        prefix_attn = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            adarms_cond=[None, None],
        )

        if stop_gradient_to_vlm and past_kv is not None:
            from transformers.cache_utils import DynamicCache

            if isinstance(past_kv, DynamicCache):
                for i in range(len(past_kv.key_cache)):
                    past_kv.key_cache[i] = past_kv.key_cache[i].detach()
                    past_kv.value_cache[i] = past_kv.value_cache[i].detach()
            else:
                past_kv = tuple(
                    tuple(
                        t.detach() if isinstance(t, torch.Tensor) else t
                        for t in layer_kv
                    )
                    for layer_kv in past_kv
                )

        # Phase 2: run value expert with cached prefix keys/values.
        batch_size = prefix_pad_masks.shape[0]
        prefix_len, suffix_len = prefix_pad_masks.shape[1], suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_attn = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_2d, suffix_attn], dim=2)
        )
        suffix_pos = (
            prefix_pad_masks.sum(dim=-1)[:, None]
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None],
            expert_name="value",
        )
        return prefix_out, suffix_out

    def _compute_value_from_hidden(self, cls_hidden):
        """Compute value from [CLS] hidden state using categorical distribution."""
        logits = self.value_head(cls_hidden)
        probs = F.softmax(logits, dim=-1)
        values = (probs * self.value_head.atoms).sum(dim=-1)
        return values, logits, probs

    def _compute_categorical_loss(self, logits, target_values):
        """Compute categorical loss (Dirac delta projection onto bins).

        Returns:
            Tuple of (loss, metrics_dict).
        """
        target_values = target_values.clamp(self.v_min, self.v_max)
        b = (target_values - self.v_min) / self.delta_z
        l = b.floor().long().clamp(0, self.num_bins - 1)
        u = b.ceil().long().clamp(0, self.num_bins - 1)

        d_to_l, d_to_u = b - l.float(), u.float() - b
        same_bin = l == u
        d_to_l = torch.where(same_bin, torch.zeros_like(d_to_l), d_to_l)
        d_to_u = torch.where(same_bin, torch.ones_like(d_to_u), d_to_u)

        batch_size = target_values.shape[0]
        target_probs = torch.zeros(
            batch_size, self.num_bins, device=target_values.device
        )
        batch_idx = torch.arange(batch_size, device=target_values.device)
        target_probs[batch_idx, l] += d_to_u
        target_probs[batch_idx, u] += d_to_l

        loss = -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)

        pred_bin = logits.argmax(dim=-1)
        best_target_bin = torch.where(d_to_u >= d_to_l, l, u)
        acc_best = (pred_bin == best_target_bin).float().mean()
        acc_neighbor = ((pred_bin == l) | (pred_bin == u)).float().mean()

        dist_to_l = (pred_bin - l).abs()
        dist_to_u = (pred_bin - u).abs()
        min_dist = torch.min(dist_to_l, dist_to_u).float()
        mae = (min_dist * self.delta_z).mean()

        metrics = {
            "acc_best": acc_best,
            "acc_neighbor": acc_neighbor,
            "mae": mae,
        }
        return loss, metrics

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        """Inference with KV cache."""
        (images, img_masks, lang_tokens, lang_masks, token_ar_mask, _, _) = (
            self._preprocess_observation(observation)
        )
        batch_size = lang_tokens.shape[0]

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        # Phase 1: Prefill VLM
        prefix_attn = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Phase 2: Expert with cache
        prefix_len, suffix_len = prefix_pad_masks.shape[1], suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(
            batch_size, suffix_len, prefix_len
        )
        suffix_attn = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_4d = self._prepare_attention_masks_4d(
            torch.cat([prefix_2d, suffix_attn], dim=2)
        )
        suffix_pos = (
            prefix_pad_masks.sum(dim=-1)[:, None]
            + torch.cumsum(suffix_pad_masks, dim=1)
            - 1
        )

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, None],
            expert_name="value",
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return CriticOutput(
            predicted_values=values,
            logits=logits,
            probs=probs,
            atoms=self.value_head.atoms,
            hidden_states=cls_hidden,
        )


# =============================================================================
# ValueCritic (HF wrapper for training and inference via from_pretrained)
# =============================================================================


class ValueCritic(CriticPreTrainedModel):
    """Value Critic V(s) for RL fine-tuning."""

    config_class = ValueCriticConfig
    _no_split_modules = []

    def __init__(self, config: ValueCriticConfig):
        super().__init__(config)
        config.critic_expert_variant = getattr(
            config, "critic_expert_variant", "gemma_100m"
        )
        self.model = ValueCriticModel(config)
        self.post_init()

    def get_input_embeddings(self):
        if self.model.backbone_variant == "siglip_gemma3":
            return self.model.paligemma_with_expert.gemma3.get_input_embeddings()
        return self.model.paligemma_with_expert.paligemma.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        if self.model.backbone_variant == "siglip_gemma3":
            self.model.paligemma_with_expert.gemma3.set_input_embeddings(value)
        else:
            self.model.paligemma_with_expert.paligemma.language_model.set_input_embeddings(
                value
            )

    def get_output_embeddings(self):
        if self.model.backbone_variant == "siglip_gemma3":
            return self.model.paligemma_with_expert.gemma3.lm_head
        return self.model.paligemma_with_expert.paligemma.lm_head

    def set_output_embeddings(self, new_embeddings):
        if self.model.backbone_variant == "siglip_gemma3":
            self.model.paligemma_with_expert.gemma3.lm_head = new_embeddings
        else:
            self.model.paligemma_with_expert.paligemma.lm_head = new_embeddings

    def resize_token_embeddings(
        self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True
    ):
        if pad_to_multiple_of is not None:
            new_num_tokens = (
                (new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of
            ) * pad_to_multiple_of

        if self.model.backbone_variant == "siglip_gemma3":
            self.model.paligemma_with_expert.gemma3.resize_token_embeddings(
                new_num_tokens, pad_to_multiple_of, mean_resizing
            )
        else:
            lm = self.model.paligemma_with_expert.paligemma.language_model
            lm.resize_token_embeddings(
                new_num_tokens, pad_to_multiple_of, mean_resizing
            )

            old_lm_head = self.model.paligemma_with_expert.paligemma.lm_head
            old_num_tokens = old_lm_head.weight.shape[0]

            if old_num_tokens != new_num_tokens:
                new_lm_head = nn.Linear(
                    old_lm_head.in_features,
                    new_num_tokens,
                    bias=old_lm_head.bias is not None,
                    device=old_lm_head.weight.device,
                    dtype=old_lm_head.weight.dtype,
                )
                num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
                new_lm_head.weight.data[:num_tokens_to_copy] = old_lm_head.weight.data[
                    :num_tokens_to_copy
                ]
                if old_lm_head.bias is not None:
                    new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[
                        :num_tokens_to_copy
                    ]
                if new_num_tokens > old_num_tokens:
                    nn.init.normal_(
                        new_lm_head.weight.data[old_num_tokens:], mean=0.0, std=0.02
                    )
                    if old_lm_head.bias is not None:
                        new_lm_head.bias.data[old_num_tokens:].zero_()
                self.model.paligemma_with_expert.paligemma.lm_head = new_lm_head

        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

    def forward(self, observation, target_values=None, **kwargs) -> CriticOutput:
        return self.model.forward(
            observation,
            target_values=target_values,
            **kwargs,
        )

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        """Predict value for given observation. Returns scalar value."""
        return self.model.predict(observation).predicted_values

    @torch.no_grad()
    def predict_distribution(self, observation) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict value distribution. Returns (values, probs, atoms)."""
        out = self.model.predict(observation)
        return out.predicted_values, out.probs, out.atoms

    # =========================================================================
    # Inference API (from_checkpoint / infer / infer_batch)
    # =========================================================================

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir,
        *,
        device="cuda",
        env_type="libero",
        model_type="pi05",
        default_prompt=None,
        norm_stats=None,
        num_return_bins=201,
        return_min=-1.0,
        return_max=0.0,
        action_norm_skip_dims=None,
        critic_expert_variant="gemma_100m",
        tokenizer_path=None,
        backbone_variant="paligemma",
        siglip_path=None,
        gemma3_path=None,
        **kwargs,
    ):
        """Create a ValueCritic from a trained checkpoint, ready for inference.

        Args:
            checkpoint_dir: Path to checkpoint directory.
            device: Device to run inference on.
            env_type: Environment type (e.g., "libero").
            model_type: Model type ("pi0", "pi05").
            default_prompt: Default prompt to inject if none provided.
            norm_stats: Normalization stats (loaded from checkpoint if not provided).
            num_return_bins: Number of return bins.
            return_min: Minimum return value.
            return_max: Maximum return value.
            action_norm_skip_dims: Dims to skip in normalization.
            critic_expert_variant: Gemma variant (e.g., "gemma_100m").
            tokenizer_path: Explicit path to tokenizer. If not set, loads
                from checkpoint_dir. Raises if neither has tokenizer files.
            backbone_variant: Backbone type ("paligemma" or "siglip_gemma3").
            siglip_path: Path to SigLIP pretrained weights (siglip_gemma3 only).
            gemma3_path: Path to Gemma3 pretrained weights (siglip_gemma3 only).

        Returns:
            ValueCritic instance with transforms and processor attached.
        """
        import pathlib

        from transformers import AutoTokenizer

        from .checkpoint_utils import (
            build_input_transforms,
            has_tokenizer_files,
            load_norm_stats,
            load_state_dict_from_checkpoint,
        )
        from .processing import ValueProcessor

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        logger.info(f"Loading value model from {checkpoint_dir}")

        # Load processor: tokenizer_path > checkpoint tokenizer > error
        if tokenizer_path:
            logger.info("  Using explicit tokenizer_path: %s", tokenizer_path)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, add_bos_token=True, local_files_only=True
            )
        elif has_tokenizer_files(checkpoint_dir):
            logger.info("  Found tokenizer files in checkpoint")
            tokenizer = AutoTokenizer.from_pretrained(
                str(checkpoint_dir), add_bos_token=True, local_files_only=True
            )
        else:
            raise ValueError(
                f"No tokenizer found. Set tokenizer_path or ensure checkpoint "
                f"contains tokenizer files. checkpoint_dir={checkpoint_dir}"
            )
        processor = ValueProcessor(tokenizer=tokenizer, max_token_len=200)

        # Build config and load model
        critic_config = ValueCriticConfig(
            critic_expert_variant=critic_expert_variant,
            num_bins=num_return_bins,
            v_min=return_min,
            v_max=return_max,
            backbone_variant=backbone_variant,
            siglip_path=siglip_path,
            gemma3_path=gemma3_path,
        )

        try:
            model = cls.from_pretrained(
                str(checkpoint_dir),
                config=critic_config,
                torch_dtype=torch.bfloat16,
                device_map=None,
                trust_remote_code=True,
                ignore_mismatched_sizes=True,
            )
            logger.info("  Loaded model using from_pretrained")
        except (OSError, ValueError, AttributeError) as e:
            logger.info(
                f"  from_pretrained failed ({type(e).__name__}: {e}), "
                "loading state dict directly"
            )
            model = cls(critic_config)
            state_dict = load_state_dict_from_checkpoint(checkpoint_dir)

            # Handle key prefix mismatch (checkpoint may lack "model." prefix)
            model_keys = set(model.state_dict().keys())
            ckpt_keys = set(state_dict.keys())

            if len(model_keys & ckpt_keys) == 0:
                remapped = {}
                for k, v in state_dict.items():
                    new_key = f"model.{k}"
                    remapped[new_key if new_key in model_keys else k] = v

                if len(set(remapped.keys()) & model_keys) > len(
                    ckpt_keys & model_keys
                ):
                    logger.info("  Remapped checkpoint keys: added 'model.' prefix")
                    state_dict = remapped

            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            logger.info(
                f"  Loaded state dict (missing={len(missing)}, "
                f"unexpected={len(unexpected)})"
            )

        # Attach processor
        object.__setattr__(model, "processor", processor)

        model = model.to(device)
        model.eval()

        # Load norm stats
        if norm_stats is None:
            try:
                norm_stats = load_norm_stats(checkpoint_dir, env_type)
                logger.info(f"Loaded norm stats with asset_id={env_type}")
            except FileNotFoundError:
                logger.warning(
                    f"Could not find norm stats in {checkpoint_dir}, "
                    "proceeding without normalization"
                )

        # Exclude 'return' from normalization
        if norm_stats and "return" in norm_stats:
            norm_stats = {k: v for k, v in norm_stats.items() if k != "return"}

        use_quantile_norm = model_type.lower() != "pi0"

        transforms = build_input_transforms(
            env_type=env_type,
            model_type=model_type,
            action_dim=getattr(model.config, "action_dim", 32),
            default_prompt=default_prompt,
            norm_stats=norm_stats,
            use_quantile_norm=use_quantile_norm,
            action_norm_skip_dims=action_norm_skip_dims,
        )

        # Attach transforms and device for inference
        from rlinf.datasets.lerobot.transforms import compose

        object.__setattr__(model, "_input_transform", compose(transforms))
        object.__setattr__(model, "_device", device)

        logger.info("ValueCritic.from_checkpoint ready for inference")
        return model

    @staticmethod
    def _prepare_observation_cpu(inputs: dict, processor) -> dict:
        """CPU-only observation preparation (safe to run in DataLoader workers).

        Runs image_processor and tokenizer on CPU tensors, returns CPU tensors.
        No .to(device) calls, so this can be passed to multiprocessing workers
        via functools.partial.

        Args:
            inputs: Transformed observation dict (output of _input_transform).
            processor: ValueProcessor instance.

        Returns:
            Dict with CPU tensors: pixel_values, image_masks, tokens, mask, ar_mask.
        """
        import numpy as np

        # Get images
        if "image" in inputs and isinstance(inputs["image"], dict):
            images_dict = inputs["image"]
        elif "images" in inputs and isinstance(inputs["images"], dict):
            images_dict = inputs["images"]
        else:
            images_dict = {}
            for key in inputs:
                if "image" in key.lower() and isinstance(
                    inputs[key], (np.ndarray, torch.Tensor)
                ):
                    img_key = key
                    for prefix in [
                        "observation/",
                        "observation.",
                        "images/",
                        "images.",
                    ]:
                        img_key = img_key.replace(prefix, "")
                    images_dict[img_key] = inputs[key]

        # Get prompt
        prompt = inputs.get("prompt", "perform the task")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.size == 1 else "perform the task"
        elif not isinstance(prompt, str):
            prompt = "perform the task"

        # Convert to BHWC for image_processor (handles both CHW and HWC input)
        images_bhwc = {}
        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                if img.shape[0] == 3:
                    # CHW (3, H, W) -> BHWC (1, H, W, 3)
                    img = img.unsqueeze(0).permute(0, 2, 3, 1)
                else:
                    # HWC (H, W, 3) -> BHWC (1, H, W, 3)
                    img = img.unsqueeze(0)
            elif img.dim() == 4:
                if img.shape[1] == 3:
                    # BCHW -> BHWC
                    img = img.permute(0, 2, 3, 1)
                # else: already BHWC
            images_bhwc[cam_name] = img

        # Image masks
        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor(
                        [mask], dtype=torch.bool
                    )
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor(
                        [True], dtype=torch.bool
                    )
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        # Process images (CPU)
        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

        # Tokenize prompt (CPU)
        cleaned_prompt = processor._clean_text(prompt)
        cleaned_prompt = processor._strip_trailing_punctuation(cleaned_prompt)
        prefix_text = f"Task: {cleaned_prompt}."

        tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=True)
        seq_len = len(tokens)
        max_length = processor.max_token_len
        if seq_len < max_length:
            padding_len = max_length - seq_len
            tok_mask = [True] * seq_len + [False] * padding_len
            tokens = tokens + [0] * padding_len
            ar_mask = [0] * max_length
        else:
            tokens = tokens[:max_length]
            tok_mask = [True] * max_length
            ar_mask = [0] * max_length

        # Return CPU tensors only (no .to(device))
        return {
            "images": processed_img["pixel_values"],
            "image_masks": processed_img["image_masks"],
            "tokenized_prompt": torch.tensor([tokens], dtype=torch.long),
            "tokenized_prompt_mask": torch.tensor([tok_mask], dtype=torch.bool),
            "token_ar_mask": torch.tensor([ar_mask], dtype=torch.long),
        }

    def _prepare_observation(self, inputs: dict) -> dict:
        """Prepare observation dict for model forward.

        Tokenizes "Task: {prompt}." (matching training template).
        """
        import numpy as np

        processor = getattr(self, "processor", None)
        if processor is None:
            raise RuntimeError(
                "Model processor not attached. Use from_checkpoint() or attach manually."
            )

        device = getattr(self, "_device", "cuda")

        # Get images
        if "image" in inputs and isinstance(inputs["image"], dict):
            images_dict = inputs["image"]
        elif "images" in inputs and isinstance(inputs["images"], dict):
            images_dict = inputs["images"]
        else:
            images_dict = {}
            for key in inputs:
                if "image" in key.lower() and isinstance(
                    inputs[key], (np.ndarray, torch.Tensor)
                ):
                    img_key = key
                    for prefix in [
                        "observation/",
                        "observation.",
                        "images/",
                        "images.",
                    ]:
                        img_key = img_key.replace(prefix, "")
                    images_dict[img_key] = inputs[key]

        # Get prompt
        prompt = inputs.get("prompt", "perform the task")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.size == 1 else "perform the task"
        elif not isinstance(prompt, str):
            prompt = "perform the task"

        # Convert to BHWC for image_processor (handles both CHW and HWC input)
        images_bhwc = {}
        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                if img.shape[0] == 3:
                    # CHW (3, H, W) -> BHWC (1, H, W, 3)
                    img = img.unsqueeze(0).permute(0, 2, 3, 1)
                else:
                    # HWC (H, W, 3) -> BHWC (1, H, W, 3)
                    img = img.unsqueeze(0)
            elif img.dim() == 4:
                if img.shape[1] == 3:
                    # BCHW -> BHWC
                    img = img.permute(0, 2, 3, 1)
                # else: already BHWC
            images_bhwc[cam_name] = img

        # Image masks
        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor(
                        [mask], dtype=torch.bool
                    )
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor(
                        [True], dtype=torch.bool
                    )
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        # Process images
        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

        # Tokenize prompt
        cleaned_prompt = processor._clean_text(prompt)
        cleaned_prompt = processor._strip_trailing_punctuation(cleaned_prompt)
        prefix_text = f"Task: {cleaned_prompt}."

        tokens = processor.tokenizer.encode(prefix_text, add_special_tokens=True)
        seq_len = len(tokens)

        max_length = processor.max_token_len
        if seq_len < max_length:
            padding_len = max_length - seq_len
            mask = [True] * seq_len + [False] * padding_len
            tokens = tokens + [0] * padding_len
            ar_mask = [0] * max_length
        else:
            tokens = tokens[:max_length]
            mask = [True] * max_length
            ar_mask = [0] * max_length

        # Build observation dict
        pixel_values = processed_img["pixel_values"]
        image_masks = processed_img["image_masks"]

        if isinstance(pixel_values, dict):
            images_on_device = {k: v.to(device) for k, v in pixel_values.items()}
        else:
            images_on_device = pixel_values.to(device)

        if isinstance(image_masks, dict):
            masks_on_device = {k: v.to(device) for k, v in image_masks.items()}
        else:
            masks_on_device = image_masks.to(device)

        return {
            "images": images_on_device,
            "image_masks": masks_on_device,
            "tokenized_prompt": torch.tensor(
                [tokens], dtype=torch.long, device=device
            ),
            "tokenized_prompt_mask": torch.tensor(
                [mask], dtype=torch.bool, device=device
            ),
            "token_ar_mask": torch.tensor(
                [ar_mask], dtype=torch.long, device=device
            ),
        }

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        """Prepare batched observation dict from list of inputs."""
        all_images = []
        all_image_masks = []
        all_tokens = []
        all_masks = []
        all_ar_masks = []

        for inputs in inputs_list:
            single_obs = self._prepare_observation(inputs)
            all_images.append(single_obs["images"])
            all_image_masks.append(single_obs["image_masks"])
            all_tokens.append(single_obs["tokenized_prompt"])
            all_masks.append(single_obs["tokenized_prompt_mask"])
            all_ar_masks.append(single_obs["token_ar_mask"])

        if isinstance(all_images[0], dict):
            batched_images = {
                k: torch.cat([img[k] for img in all_images], dim=0)
                for k in all_images[0]
            }
            batched_masks = {
                k: torch.cat([m[k] for m in all_image_masks], dim=0)
                for k in all_image_masks[0]
            }
        else:
            batched_images = torch.cat(all_images, dim=0)
            batched_masks = torch.cat(all_image_masks, dim=0)

        return {
            "images": batched_images,
            "image_masks": batched_masks,
            "tokenized_prompt": torch.cat(all_tokens, dim=0),
            "tokenized_prompt_mask": torch.cat(all_masks, dim=0),
            "token_ar_mask": torch.cat(all_ar_masks, dim=0),
        }

    @torch.no_grad()
    def infer(self, obs: dict) -> dict:
        """Infer value from a single observation.

        Args:
            obs: Raw observation dictionary.

        Returns:
            Dictionary with "value" key.
        """
        import numpy as np

        inputs = {
            k: v.copy() if isinstance(v, np.ndarray) else v for k, v in obs.items()
        }
        inputs = self._input_transform(inputs)
        observation = self._prepare_observation(inputs)

        values = self.predict_value(observation)
        return {
            "value": float(values[0].item()),
            "state": obs.get("state", np.array([])),
        }

    @torch.no_grad()
    def infer_batch(
        self,
        obs_list: list[dict],
        *,
        batch_size: int = 64,
        pretransformed: bool = False,
        already_cpu_prepared: bool = False,
    ) -> list[dict]:
        """Batch inference for multiple observations.

        Args:
            obs_list: List of observation dictionaries.
            batch_size: Maximum batch size for single forward pass.
            pretransformed: If True, skip _input_transform (already applied).
            already_cpu_prepared: If True, obs are output of _prepare_observation_cpu.
                Skip all preprocessing, just stack tensors and move to device.

        Returns:
            List of dictionaries with "value" key.
        """
        import numpy as np

        if not obs_list:
            return []

        device = getattr(self, "_device", "cuda")
        all_outputs = []

        for batch_start in range(0, len(obs_list), batch_size):
            batch_end = min(batch_start + batch_size, len(obs_list))
            batch_obs = obs_list[batch_start:batch_end]

            if already_cpu_prepared:
                # Workers already ran _input_transform + _prepare_observation_cpu.
                # Just stack CPU tensors and move to device.
                first = batch_obs[0]
                if isinstance(first.get("images"), dict):
                    batched_images = {
                        k: torch.cat(
                            [obs["images"][k] for obs in batch_obs], dim=0
                        ).to(device)
                        for k in first["images"]
                    }
                    batched_masks = {
                        k: torch.cat(
                            [obs["image_masks"][k] for obs in batch_obs], dim=0
                        ).to(device)
                        for k in first["image_masks"]
                    }
                else:
                    batched_images = torch.cat(
                        [obs["images"] for obs in batch_obs], dim=0
                    ).to(device)
                    batched_masks = torch.cat(
                        [obs["image_masks"] for obs in batch_obs], dim=0
                    ).to(device)

                observation = {
                    "images": batched_images,
                    "image_masks": batched_masks,
                    "tokenized_prompt": torch.cat(
                        [obs["tokenized_prompt"] for obs in batch_obs], dim=0
                    ).to(device),
                    "tokenized_prompt_mask": torch.cat(
                        [obs["tokenized_prompt_mask"] for obs in batch_obs], dim=0
                    ).to(device),
                    "token_ar_mask": torch.cat(
                        [obs["token_ar_mask"] for obs in batch_obs], dim=0
                    ).to(device),
                }
            else:
                inputs_list = []
                for obs in batch_obs:
                    inputs = {
                        k: v.copy() if isinstance(v, np.ndarray) else v
                        for k, v in obs.items()
                    }
                    if not pretransformed:
                        inputs = self._input_transform(inputs)
                    inputs_list.append(inputs)

                observation = self._prepare_observation_batch(inputs_list)

            values = self.predict_value(observation).cpu()

            for i in range(len(batch_obs)):
                all_outputs.append({"value": float(values[i])})

        return all_outputs


__all__ = [
    "make_att_2d_masks",
    "VLMPreTrainedModel",
    "CriticPreTrainedModel",
    "VLMObservationEncoder",
    "ValueHead",
    "CriticOutput",
    "ValueCriticModel",
    "ValueCritic",
]
