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
Value Critic Models for Value Prediction.

Uses expert forward mode: Gemma expert + [CLS] -> categorical value prediction.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput

from .configuration import ValueCriticConfig, get_config
from .value_expert import ValueExpert

logger = logging.getLogger(__name__)


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


def prepare_target_values(
    target_values: torch.Tensor,
    *,
    v_min: float,
    v_max: float,
    expected_batch_size: int,
) -> torch.Tensor:
    """Flatten, validate, and clamp target values for categorical value loss."""
    target_values = target_values.float().view(-1)
    if expected_batch_size != target_values.shape[0]:
        raise ValueError(
            "ValueCriticModel target_values batch size does not match logits: "
            f"logits_batch={expected_batch_size}, targets_batch={target_values.shape[0]}"
        )
    return target_values.clamp(v_min, v_max)


@dataclass
class CriticOutput(ModelOutput):
    """Output for critic models."""

    loss: Optional[torch.FloatTensor] = None
    predicted_values: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    cat_acc_best: Optional[torch.FloatTensor] = None
    cat_acc_neighbor: Optional[torch.FloatTensor] = None
    mae: Optional[torch.FloatTensor] = None


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


class ValueCriticModel(nn.Module):
    """Value function V(s) with VLM observation encoding.

    Uses expert mode: Gemma expert + [CLS] -> categorical value prediction.
    """

    def __init__(self, config: ValueCriticConfig):
        super().__init__()
        self.config = config

        expert_config = get_config(config.critic_expert_variant)

        logger.info(f"Creating ValueCriticModel: expert={config.critic_expert_variant}")

        expert_configs = {"value": expert_config}

        siglip_path = getattr(config, "siglip_path", "")
        gemma3_path = getattr(config, "gemma3_path", "")
        if not siglip_path or not gemma3_path:
            raise ValueError(
                "ValueCriticModel requires both siglip_path and gemma3_path. "
                f"Got siglip_path='{siglip_path}', gemma3_path='{gemma3_path}'"
            )
        self.value_expert = ValueExpert(
            expert_configs=expert_configs,
            siglip_path=siglip_path,
            gemma3_path=gemma3_path,
            precision=config.precision,
            freeze_vision_encoder=getattr(config, "freeze_vision_encoder", False),
            freeze_vlm=getattr(config, "freeze_vlm", False),
        )

        self.gradient_checkpointing_enabled = False

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
        if self.value_expert.freeze_vlm:
            no_split_modules = [
                "GemmaDecoderLayer",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                "ValueHead",
                "Gemma3RMSNorm",
                "Gemma3DecoderLayer",
            ]
        else:
            no_split_modules = [
                "GemmaMLP",
                "SiglipVisionEmbeddings",
                "GemmaRMSNorm",
                "GemmaRotaryEmbedding",
                "ValueHead",
                "Gemma3RMSNorm",
                "Gemma3DecoderLayer",
            ]
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        return ["lm_head", "cls_embedding"]

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True
        self.value_expert.gemma3.model.gradient_checkpointing = True
        self.value_expert.vision_tower.gradient_checkpointing = True
        for expert in self.value_expert.experts.values():
            expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for ValueCriticModel")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing_enabled = False
        self.value_expert.gemma3.model.gradient_checkpointing = False
        self.value_expert.vision_tower.gradient_checkpointing = False
        for expert in self.value_expert.experts.values():
            expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for ValueCriticModel")

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _get_model_dtype(self) -> torch.dtype:
        """Get the dtype of the backbone model's attention weights."""
        return self.value_expert.gemma3.model.layers[0].self_attn.q_proj.weight.dtype

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
            token_loss_mask,
            token_kv_cache_mask,
        )

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks):
        """Embed images and language tokens into prefix embeddings."""
        embs, pad_masks = [], []
        bsize = lang_tokens.shape[0]

        for img, img_mask in zip(images, img_masks, strict=True):
            img_emb = self._apply_checkpoint(self.value_expert.embed_image, img)
            num_img_embs = img_emb.shape[1]
            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

        def embed_lang(tokens):
            return self.value_expert.embed_language_tokens(tokens)

        lang_emb = self._apply_checkpoint(embed_lang, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        return torch.cat(embs, dim=1), torch.cat(pad_masks, dim=1)

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
            _,
            _,
        ) = self._preprocess_observation(observation)

        batch_size = lang_tokens.shape[0]

        prefix_embs, prefix_pad_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        stop_gradient = getattr(self.config, "stop_gradient_to_vlm", False)
        values, hidden_states, logits, probs, backward_anchor = self._forward_expert(
            prefix_embs,
            prefix_pad_masks,
            suffix_embs,
            suffix_pad_masks,
            suffix_ar_masks,
            stop_gradient_to_vlm=stop_gradient,
        )

        loss = None
        cat_metrics = None
        if target_values is not None:
            loss, cat_metrics = self._compute_categorical_loss(logits, target_values)
            if backward_anchor is not None:
                loss = loss + backward_anchor

        loss_mean = loss.mean() if loss is not None else None

        return CriticOutput(
            loss=loss_mean,
            predicted_values=values,
            logits=logits,
            probs=probs,
            atoms=self.value_head.atoms,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            mae=cat_metrics["mae"] if cat_metrics else None,
        )

    def _forward_expert(
        self,
        prefix_embs,
        prefix_pad_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ):
        """Forward through VLM + value expert (two-stage)."""
        model_dtype = self._get_model_dtype()
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        prefix_out, suffix_out = self._forward_expert_two_stage(
            prefix_embs=prefix_embs,
            prefix_pad_masks=prefix_pad_masks,
            suffix_embs=suffix_embs,
            suffix_pad_masks=suffix_pad_masks,
            suffix_ar_masks=suffix_ar_masks,
            stop_gradient_to_vlm=stop_gradient_to_vlm,
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
        """Build a zero-weight anchor so FSDP tracks two-stage backward."""
        if (
            not self.training
            or stop_gradient_to_vlm
            or prefix_out is None
            or not prefix_out.requires_grad
        ):
            return None
        # Gradients flow via DynamicCache in two-stage forward. Anchoring
        # an explicit output tensor keeps FSDP's forward/backward state aligned.
        return prefix_out.sum() * 0.0

    def _forward_expert_two_stage(
        self,
        prefix_embs,
        prefix_pad_masks,
        suffix_embs,
        suffix_pad_masks,
        suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Two-stage expert forward with KV cache."""
        # Phase 1: prefill frozen VLM to get KV cache.
        # Prefix is fully bidirectional, so attention = pad_mask outer product.
        prefix_attn = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        (prefix_out, _), past_kv = self.value_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        if stop_gradient_to_vlm and past_kv is not None:
            from transformers.cache_utils import DynamicCache

            if not isinstance(past_kv, DynamicCache):
                raise TypeError(
                    f"Expected DynamicCache from ValueExpert, got {type(past_kv)}"
                )
            for i in range(len(past_kv.key_cache)):
                past_kv.key_cache[i] = past_kv.key_cache[i].detach()
                past_kv.value_cache[i] = past_kv.value_cache[i].detach()

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

        (_, suffix_out), _ = self.value_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
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
        target_values = prepare_target_values(
            target_values,
            v_min=self.v_min,
            v_max=self.v_max,
            expected_batch_size=logits.shape[0],
        )
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
        (images, img_masks, lang_tokens, lang_masks, _, _) = (
            self._preprocess_observation(observation)
        )
        batch_size = lang_tokens.shape[0]

        prefix_embs, prefix_pad_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        # Prefix is fully bidirectional, so attention = pad_mask outer product.
        prefix_attn = prefix_pad_masks[:, None, :] * prefix_pad_masks[:, :, None]
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_kv = self.value_expert.forward(
            attention_mask=prefix_attn_4d,
            position_ids=prefix_pos,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

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

        (_, suffix_out), _ = self.value_expert.forward(
            attention_mask=full_attn_4d,
            position_ids=suffix_pos,
            past_key_values=past_kv,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
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

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        """Predict value for given observation. Returns scalar value."""
        return self.predict(observation).predicted_values

    @torch.no_grad()
    def predict_distribution(self, observation) -> tuple[Tensor, Tensor, Tensor]:
        """Predict value distribution. Returns (values, probs, atoms)."""
        out = self.predict(observation)
        return out.predicted_values, out.probs, out.atoms

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
        critic_expert_variant="gemma_100m",
        tokenizer_path=None,
        siglip_path=None,
        gemma3_path=None,
        **kwargs,
    ):
        """Create a ValueCriticModel from a trained checkpoint, ready for inference.

        Delegates model creation and weight loading to :func:`get_model`,
        then attaches inference-specific components (processor, transforms,
        normalization stats).

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
            critic_expert_variant: Gemma variant (e.g., "gemma_100m").
            tokenizer_path: Explicit path to tokenizer. If not set, loads
                from checkpoint_dir. Raises if neither has tokenizer files.
            siglip_path: Path to SigLIP pretrained weights.
            gemma3_path: Path to Gemma3 pretrained weights.

        Returns:
            ValueCriticModel instance with transforms and processor attached.
        """
        import pathlib

        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        from . import get_model
        from .checkpoint_utils import (
            build_input_transforms,
            has_tokenizer_files,
            load_norm_stats,
        )
        from .processing import ValueProcessor

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        logger.info(f"Loading value model from {checkpoint_dir}")

        # Build model via shared factory (handles config + weight loading)
        cfg = OmegaConf.create(
            {
                "model_path": str(checkpoint_dir),
                "critic_expert_variant": critic_expert_variant,
                "num_bins": num_return_bins,
                "v_min": return_min,
                "v_max": return_max,
                "siglip_path": siglip_path,
                "gemma3_path": gemma3_path,
            }
        )
        model = get_model(cfg)

        # --- Inference-specific setup ---

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

        model.processor = processor
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
        )

        from openpi.transforms import compose

        model._input_transform = compose(transforms)
        model._device = device

        logger.info("ValueCriticModel.from_checkpoint ready for inference")
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
            Dict with CPU tensors: pixel_values, image_masks, tokens, mask.
        """
        import numpy as np

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

        prompt = inputs.get("prompt", "perform the task")
        if isinstance(prompt, np.ndarray):
            prompt = str(prompt.item()) if prompt.size == 1 else "perform the task"
        elif not isinstance(prompt, str):
            prompt = "perform the task"

        images_bhwc = {}
        for cam_name, img in images_dict.items():
            if isinstance(img, np.ndarray):
                img = torch.from_numpy(img)
            if img.dim() == 3:
                if img.shape[0] == 3:
                    img = img.unsqueeze(0).permute(0, 2, 3, 1)  # CHW -> BHWC
                else:
                    img = img.unsqueeze(0)  # HWC -> BHWC
            elif img.dim() == 4:
                if img.shape[1] == 3:
                    img = img.permute(0, 2, 3, 1)  # BCHW -> BHWC
            images_bhwc[cam_name] = img

        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor([mask], dtype=torch.bool)
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

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
        else:
            tokens = tokens[:max_length]
            tok_mask = [True] * max_length

        # Return CPU tensors only (no .to(device))
        return {
            "images": processed_img["pixel_values"],
            "image_masks": processed_img["image_masks"],
            "tokenized_prompt": torch.tensor([tokens], dtype=torch.long),
            "tokenized_prompt_mask": torch.tensor([tok_mask], dtype=torch.bool),
        }

    def _prepare_observation(self, inputs: dict) -> dict:
        """Prepare observation dict for model forward.

        Tokenizes "Task: {prompt}." (matching training template).
        Returns dict with images, image_masks, tokenized_prompt, tokenized_prompt_mask.
        """
        import numpy as np

        processor = getattr(self, "processor", None)
        if processor is None:
            raise RuntimeError(
                "Model processor not attached. Use from_checkpoint() or attach manually."
            )

        device = getattr(self, "_device", "cuda")

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

        input_masks = inputs.get("image_mask", inputs.get("image_masks", {}))
        image_masks_batch = {}
        for cam_name in images_bhwc:
            if cam_name in input_masks:
                mask = input_masks[cam_name]
                if isinstance(mask, (bool, np.bool_)):
                    image_masks_batch[cam_name] = torch.tensor([mask], dtype=torch.bool)
                elif isinstance(mask, torch.Tensor):
                    image_masks_batch[cam_name] = (
                        mask.unsqueeze(0) if mask.dim() == 0 else mask
                    )
                else:
                    image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)
            else:
                image_masks_batch[cam_name] = torch.tensor([True], dtype=torch.bool)

        processed_img = processor.image_processor(
            images=images_bhwc,
            image_masks=image_masks_batch if image_masks_batch else None,
            return_tensors="pt",
            train=False,
        )

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
        else:
            tokens = tokens[:max_length]
            mask = [True] * max_length

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
            "tokenized_prompt": torch.tensor([tokens], dtype=torch.long, device=device),
            "tokenized_prompt_mask": torch.tensor(
                [mask], dtype=torch.bool, device=device
            ),
        }

    def _prepare_observation_batch(self, inputs_list: list[dict]) -> dict:
        """Prepare batched observation dict from list of inputs."""
        all_images = []
        all_image_masks = []
        all_tokens = []
        all_masks = []

        for inputs in inputs_list:
            single_obs = self._prepare_observation(inputs)
            all_images.append(single_obs["images"])
            all_image_masks.append(single_obs["image_masks"])
            all_tokens.append(single_obs["tokenized_prompt"])
            all_masks.append(single_obs["tokenized_prompt_mask"])

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
                        k: torch.cat([obs["images"][k] for obs in batch_obs], dim=0).to(
                            device
                        )
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
    "ValueHead",
    "CriticOutput",
    "ValueCriticModel",
]
