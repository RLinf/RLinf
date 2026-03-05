"""
PI0.5 Critic Models: Value Function V(s) and Q Function Q(s, a).

Inherits from PI05FlowMatching to reuse:
- _preprocess_observation, embed_prefix, _forward_vlm, _compute_ce_loss
- Attention mask and position ID logic
- Gradient checkpointing infrastructure

Supports three forward modes (analogous to PI05's VLA/VLM/VLM+VLA):
1. VLM mode: Predict "Value: X" tokens with CE loss (inherited from PI05)
2. Expert mode: Gemma expert + [CLS] → continuous/distributional value
3. Dual mode: Both VLM + expert objectives (FAST-style dual learning)

These are designed for RL fine-tuning of VLA policies.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from transformers.modeling_outputs import ModelOutput

from rlinf.utils.dist_utils import get_logger
from ..openpi.configs import get_config
from ..openpi.modeling_pi0 import make_att_2d_masks
from .configuration_pi05 import PI05Config
from .modeling_pi05 import PI05PreTrainedModel, PI05FlowMatching, PI05Output
from .paligemma_with_multi_expert import PaliGemmaWithMultiExpertModel

logger = get_logger(__name__)


class ValueHead(nn.Module):
    """Value prediction head with learnable CLS embedding and projection.
    
    Notes:
    - CLS embedding is a plain nn.Embedding(1, hidden_size); fully learnable and
      FSDP-managed inside this small module.
    - atoms buffer is non-persistent because it's deterministically computed
      from (v_min, v_max, num_bins) and doesn't need to be saved.
    """
    
    def __init__(self, hidden_size: int, num_bins: int, loss_type: str, v_min: float, v_max: float):
        super().__init__()
        self.hidden_size = hidden_size
        self.loss_type = loss_type
        
        # Learnable CLS embedding (FSDP-friendly)
        self.cls_embedding = nn.Embedding(1, hidden_size)
        nn.init.normal_(self.cls_embedding.weight, std=0.02)
        
        # Value projection
        if loss_type == "mse":
            self.value_proj = nn.Linear(hidden_size, 1)
            self.register_buffer('atoms', None, persistent=False)
            self.num_bins = None
        else:
            self.value_proj = nn.Linear(hidden_size, num_bins)
            self.register_buffer('atoms', torch.linspace(v_min, v_max, num_bins), persistent=False)
            self.num_bins = num_bins
            self.v_min = v_min
            self.v_max = v_max
            self.delta_z = (v_max - v_min) / (num_bins - 1)
    
    def get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get CLS embedding expanded to batch size. Returns [B, 1, hidden_size]."""
        # Access weight directly (avoids creating regular Tensor idx that mixes with DTensor)
        cls_emb = self.cls_embedding.weight  # [1, hidden_size]
        cls_emb = cls_emb.unsqueeze(0)  # [1, 1, hidden_size]
        return cls_emb.expand(batch_size, -1, -1).contiguous()
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """Project hidden states to value logits."""
        return self.value_proj(hidden_states)


@dataclass
class CriticOutput(ModelOutput):
    """Output for critic models."""
    loss: Optional[torch.FloatTensor] = None
    # Expert outputs (avoid 'values' - conflicts with ModelOutput.values() method)
    predicted_values: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    expert_loss: Optional[torch.FloatTensor] = None
    # VLM outputs (inherited from PI05)
    language_loss: Optional[torch.FloatTensor] = None
    language_logits: Optional[torch.FloatTensor] = None
    language_token_acc: Optional[torch.FloatTensor] = None
    language_loss_mask: Optional[torch.BoolTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None


class PI05CriticConfig(PI05Config):
    """Configuration for PI05 Critic models.
    
    Attributes:
        critic_expert_variant: Gemma expert variant (e.g., "gemma_100m", "gemma_300m")
        critic_forward_mode: Which forward path to use
            - "vlm": VLM-only token prediction (inherited from PI05)
            - "expert": Expert-only value prediction
            - "dual": Both VLM and expert objectives
        expert_loss_type: Loss type for expert value prediction
            - "mse": Mean squared error on continuous values
            - "categorical": Cross-entropy on discretized bin index (Dirac delta target)
            - "distributional": Cross-entropy on projected Bellman distribution
        num_bins: Number of discretization bins (for categorical/distributional)
        v_min: Minimum value for atom support
        v_max: Maximum value for atom support
        vlm_loss_weight: Weight for VLM cross-entropy loss
        expert_loss_weight: Weight for expert value loss
    """
    
    def __init__(
        self,
        critic_expert_variant: str = "gemma_100m",
        critic_forward_mode: Literal["vlm", "expert", "dual"] = "expert",
        expert_loss_type: Literal["mse", "categorical", "distributional"] = "mse",
        num_bins: int = 201,
        v_min: float = -1.0,
        v_max: float = 0.0,
        vlm_loss_weight: float = 1.0,
        expert_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.critic_expert_variant = critic_expert_variant
        self.critic_forward_mode = critic_forward_mode
        self.expert_loss_type = expert_loss_type
        self.num_bins = num_bins
        self.v_min = v_min
        self.v_max = v_max
        self.vlm_loss_weight = vlm_loss_weight
        self.expert_loss_weight = expert_loss_weight


class ValueCriticModel(PI05FlowMatching):
    """Value function V(s) inheriting PI05 infrastructure.
    
    Reuses from PI05FlowMatching:
    - _preprocess_observation, embed_prefix, _forward_vlm, _compute_ce_loss
    - _prepare_attention_masks_4d, _apply_checkpoint
    
    Adds:
    - Value expert with [CLS] token for continuous/distributional value prediction
    """

    def __init__(self, config: PI05CriticConfig):
        nn.Module.__init__(self)
        self.config = config
        self.pi05 = True
        self.critic_forward_mode = getattr(config, 'critic_forward_mode', 'expert')
        self.expert_loss_type = getattr(config, 'expert_loss_type', 'mse')

        paligemma_config = get_config(config.paligemma_variant)
        expert_config = get_config(config.critic_expert_variant)

        logger.info(
            f"Creating ValueCritic: expert={config.critic_expert_variant}, "
            f"mode={self.critic_forward_mode}, loss_type={self.expert_loss_type}"
        )

        # Use PaliGemmaWithMultiExpertModel for flexibility
        if self.critic_forward_mode == "vlm":
            # VLM-only mode: no expert needed, use language model for token prediction
            expert_configs = {}
            use_adarms = [False, {}]
            logger.info(
                "  VLM-only mode: using language model for token prediction, "
                "no value expert created (gemma_expert weights from PI05 checkpoint will be ignored)"
            )
        else:
            expert_configs = {"value": expert_config}
            use_adarms = [False, {"value": False}]
            logger.info(f"  Expert mode: creating 'value' expert with {config.critic_expert_variant}")

        self.paligemma_with_expert = PaliGemmaWithMultiExpertModel(
            vlm_config=paligemma_config,
            expert_configs=expert_configs,
            use_adarms=use_adarms,
            precision=config.dtype,
            freeze_vision_encoder=getattr(config, 'freeze_vision_encoder', False),
            freeze_vlm=getattr(config, 'freeze_vlm', False),
        )

        self.gradient_checkpointing_enabled = False
        self._expert_config = expert_config  # Store for gradient checkpointing

        # Value-specific components (only for expert/dual modes)
        # Wrapped in ValueHead module for FSDP2 compatibility
        if self.critic_forward_mode != "vlm":
            self.expert_width = expert_config.width
            self.value_head = ValueHead(
                hidden_size=expert_config.width,
                num_bins=config.num_bins,
                loss_type=self.expert_loss_type,
                v_min=config.v_min,
                v_max=config.v_max,
            )
            # Expose attributes from value_head for backward compatibility
            if self.expert_loss_type != "mse":
                self.num_bins = config.num_bins
                self.v_min = config.v_min
                self.v_max = config.v_max
                self.delta_z = (config.v_max - config.v_min) / (config.num_bins - 1)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        # Enable for all experts in multi-expert model
        for expert in self.paligemma_with_expert.experts.values():
            expert.model.gradient_checkpointing = True
        logger.info("Enabled gradient checkpointing for ValueCritic")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        for expert in self.paligemma_with_expert.experts.values():
            expert.model.gradient_checkpointing = False
        logger.info("Disabled gradient checkpointing for ValueCritic")

    def get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get CLS embedding from ValueHead."""
        return self.value_head.get_cls_embedding(batch_size)

    def embed_suffix(self, batch_size: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Create suffix with [CLS] token for value prediction."""
        cls_emb = self.get_cls_embedding(batch_size)
        # Masks on same device as cls_emb (derived from model params, FSDP-managed)
        pad_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=cls_emb.device)
        ar_mask = torch.ones(batch_size, 1, dtype=torch.long, device=cls_emb.device)
        return cls_emb, pad_mask, ar_mask

    def forward(self, observation, target_values=None, target_distribution=None, **kwargs) -> CriticOutput:
        """Forward pass with auto mode detection.
        
        Args:
            observation: Observation dict from data collator
            target_values: Target values [B] for mse/categorical loss
            target_distribution: Target probability distribution [B, num_bins]
                                 for distributional loss (computed by trainer)
        """
        (images, img_masks, lang_tokens, lang_masks,
         token_ar_mask, token_loss_mask, _) = self._preprocess_observation(observation)

        batch_size = lang_tokens.shape[0]
        device = lang_tokens.device

        # VLM-only mode: use inherited _forward_vlm
        if self.critic_forward_mode == "vlm":
            result = self._forward_vlm(
                images, img_masks, lang_tokens, lang_masks, token_ar_mask, token_loss_mask
            )
            return CriticOutput(
                loss=result.loss,
                language_loss=result.language_loss,
                language_logits=result.logits,
                language_token_acc=result.language_token_acc,
                language_loss_mask=result.language_loss_mask,
            )

        # Expert or Dual mode
        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        # Forward through VLM + expert
        stop_gradient = getattr(self.config, 'stop_gradient_to_vlm', False)
        values, hidden_states, logits, probs, prefix_out = self._forward_expert(
            prefix_embs, prefix_pad_masks, prefix_ar_masks,
            suffix_embs, suffix_pad_masks, suffix_ar_masks,
            stop_gradient_to_vlm=stop_gradient,
        )

        # Compute losses
        expert_loss = None
        if target_values is not None or target_distribution is not None:
            expert_loss = self._compute_expert_loss(values, logits, target_values, target_distribution)

        language_loss, language_acc, language_logits = None, None, None
        if self.critic_forward_mode == "dual" and token_loss_mask.any():
            language_loss, language_acc, language_logits = self._compute_ce_loss(
                prefix_out[:, :-1], prefix_pad_masks, lang_tokens, token_loss_mask, truncated_input=False
            )

        # Combine losses
        total_loss = None
        if expert_loss is not None or language_loss is not None:
            total_loss = torch.zeros(batch_size, device=device)
            if expert_loss is not None:
                total_loss = total_loss + self.config.expert_loss_weight * expert_loss
            if language_loss is not None:
                total_loss = total_loss + self.config.vlm_loss_weight * language_loss
            total_loss = total_loss.mean()

        return CriticOutput(
            loss=total_loss,
            predicted_values=values,
            logits=logits,
            probs=probs,
            atoms=self.value_head.atoms,
            expert_loss=expert_loss.mean() if expert_loss is not None else None,
            language_loss=language_loss,
            language_logits=language_logits,
            language_token_acc=language_acc,
            language_loss_mask=token_loss_mask[:, 1:] if self.critic_forward_mode == "dual" else None,
            hidden_states=hidden_states,
        )

    def _forward_expert(
        self,
        prefix_embs, prefix_pad_masks, prefix_ar_masks,
        suffix_embs, suffix_pad_masks, suffix_ar_masks,
        stop_gradient_to_vlm: bool = False,
    ):
        """Forward through VLM + value expert."""
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_masks = torch.cat([prefix_ar_masks, suffix_ar_masks], dim=1)
        attn_mask = make_att_2d_masks(pad_masks, ar_masks)
        attn_mask_4d = self._prepare_attention_masks_4d(attn_mask)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, attn_mask_4d, position_ids, detach_kv):
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
            forward_func, prefix_embs, suffix_embs, attn_mask_4d, position_ids, stop_gradient_to_vlm
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return values, cls_hidden, logits, probs, prefix_out

    def _compute_value_from_hidden(self, cls_hidden):
        """Compute value from [CLS] hidden state."""
        if self.expert_loss_type == "mse":
            return self.value_head(cls_hidden).squeeze(-1), None, None
        else:
            # Both categorical and distributional compute expectation over bins
            logits = self.value_head(cls_hidden)
            probs = F.softmax(logits, dim=-1)
            values = (probs * self.value_head.atoms).sum(dim=-1)
            return values, logits, probs

    def _compute_expert_loss(self, values, logits, target_values, target_distribution=None):
        """Compute expert value loss.
        
        Args:
            values: Predicted values [B]
            logits: Predicted logits [B, num_bins] (None for mse mode)
            target_values: Target values [B] (used for mse and categorical)
            target_distribution: Target probability distribution [B, num_bins]
                                 (only for distributional mode, computed by trainer)
        """
        if self.expert_loss_type == "mse":
            return F.mse_loss(values, target_values, reduction='none')
        elif self.expert_loss_type == "categorical":
            # Cross-entropy on discretized bin index (Dirac delta target)
            return self._compute_categorical_loss(logits, target_values)
        else:
            # Distributional: CE on projected Bellman distribution
            if target_distribution is not None:
                return -(target_distribution * F.log_softmax(logits, dim=-1)).sum(dim=-1)
            # Fallback to categorical if no target distribution provided
            return self._compute_categorical_loss(logits, target_values)

    def _compute_categorical_loss(self, logits, target_values):
        """Compute categorical loss (Dirac delta projection onto bins).
        
        Projects a point target value onto the discretization grid using
        linear interpolation between adjacent bins.
        """
        target_values = target_values.clamp(self.v_min, self.v_max)
        b = (target_values - self.v_min) / self.delta_z
        l = b.floor().long().clamp(0, self.num_bins - 1)
        u = b.ceil().long().clamp(0, self.num_bins - 1)

        d_to_l, d_to_u = b - l.float(), u.float() - b
        same_bin = (l == u)
        d_to_l = torch.where(same_bin, torch.zeros_like(d_to_l), d_to_l)
        d_to_u = torch.where(same_bin, torch.ones_like(d_to_u), d_to_u)

        batch_size = target_values.shape[0]
        target_probs = torch.zeros(batch_size, self.num_bins, device=target_values.device)
        batch_idx = torch.arange(batch_size, device=target_values.device)
        target_probs[batch_idx, l] += d_to_u
        target_probs[batch_idx, u] += d_to_l

        return -(target_probs * F.log_softmax(logits, dim=-1)).sum(dim=-1)

    def project_distribution(self, next_probs, rewards, dones, gamma, num_steps=None):
        """Project n-step TD target distribution onto atom support.
        
        Computes the distributional TD target:
            T_z = r_sum + gamma^n * (1 - done) * z
        where z are the atoms of next-state distribution, then projects
        onto current atom support (C51-style projection).
        
        Args:
            next_probs: Next-state value distribution [B, num_bins]
            rewards: n-step discounted reward sum [B]
                     (r_0 + gamma*r_1 + ... + gamma^(n-1)*r_{n-1})
            dones: Terminal flags [B] (True if episode ended)
            gamma: Discount factor
            num_steps: Number of valid steps n [B] for gamma^n (defaults to 1)
            
        Returns:
            Projected target distribution [B, num_bins]
        """
        batch_size = next_probs.shape[0]
        device = next_probs.device
        
        # Compute gamma^n for n-step bootstrap
        if num_steps is not None:
            gamma_n = gamma ** num_steps.float()
        else:
            gamma_n = torch.full((batch_size,), gamma, device=device)
        
        # n-step TD target: T_z = r_sum + gamma^n * (1 - done) * z
        gamma_mask = (gamma_n * (1.0 - dones.float())).unsqueeze(-1)
        tz = rewards.unsqueeze(-1) + gamma_mask * self.value_head.atoms.unsqueeze(0)
        tz = tz.clamp(self.v_min, self.v_max)
        
        # Project onto current atoms
        b = (tz - self.v_min) / self.delta_z
        l = b.floor().long().clamp(0, self.num_bins - 1)
        u = b.ceil().long().clamp(0, self.num_bins - 1)
        
        # Distribute probability mass
        m = torch.zeros(batch_size, self.num_bins, device=device)
        offset = torch.arange(batch_size, device=device).unsqueeze(-1) * self.num_bins
        
        l_idx = (l + offset).flatten()
        u_idx = (u + offset).flatten()
        d_to_l = (b - l.float()).flatten()
        d_to_u = (u.float() - b).flatten()
        next_probs_flat = next_probs.flatten()
        
        m.flatten().index_add_(0, l_idx, next_probs_flat * d_to_u)
        m.flatten().index_add_(0, u_idx, next_probs_flat * d_to_l)
        
        return m

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        """Inference with KV cache."""
        if self.critic_forward_mode == "vlm":
            raise ValueError("predict() not supported for VLM mode")

        (images, img_masks, lang_tokens, lang_masks,
         token_ar_mask, _, _) = self._preprocess_observation(observation)
        batch_size, device = lang_tokens.shape[0], lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(batch_size)

        # Phase 1: Prefill VLM
        prefix_attn = make_att_2d_masks(prefix_pad_masks, prefix_ar_masks)
        prefix_attn_4d = self._prepare_attention_masks_4d(prefix_attn)
        prefix_pos = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_attn_4d, position_ids=prefix_pos,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
        )

        # Phase 2: Expert with cache
        prefix_len, suffix_len = prefix_pad_masks.shape[1], suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_attn = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_4d = self._prepare_attention_masks_4d(torch.cat([prefix_2d, suffix_attn], dim=2))
        suffix_pos = prefix_pad_masks.sum(dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d, position_ids=suffix_pos,
            past_key_values=past_kv, inputs_embeds=[None, suffix_embs],
            use_cache=False, adarms_cond=[None, None], expert_name="value",
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return CriticOutput(predicted_values=values, logits=logits, probs=probs, atoms=self.value_head.atoms, hidden_states=cls_hidden)


class QCriticModel(ValueCriticModel):
    """Q function Q(s, a) - extends ValueCriticModel with action input."""

    def __init__(self, config: PI05CriticConfig):
        # Override default expert variant for Q critic
        config.critic_expert_variant = getattr(config, 'critic_expert_variant', 'gemma_300m')
        super().__init__(config)

        if self.critic_forward_mode != "vlm":
            self.action_in_proj = nn.Linear(config.action_dim, self.expert_width)

        # Change expert name
        if hasattr(self.paligemma_with_expert, 'experts') and "value" in self.paligemma_with_expert.experts:
            self.paligemma_with_expert.experts["q"] = self.paligemma_with_expert.experts.pop("value")

    def get_cls_embedding(self, batch_size: int) -> Tensor:
        """Get CLS embedding from ValueHead."""
        return self.value_head.get_cls_embedding(batch_size)

    def embed_suffix(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Create suffix with action embeddings + [CLS] token."""
        batch_size, action_horizon, _ = actions.shape
        action_embs = self.action_in_proj(actions)
        cls_emb = self.get_cls_embedding(batch_size)
        # Concat on same device as cls_emb (FSDP-managed)
        suffix_embs = torch.cat([action_embs, cls_emb], dim=1)

        suffix_len = action_horizon + 1
        pad_mask = torch.ones(batch_size, suffix_len, dtype=torch.bool, device=suffix_embs.device)
        ar_mask = torch.zeros(batch_size, suffix_len, dtype=torch.long, device=suffix_embs.device)
        ar_mask[:, 0] = 1
        return suffix_embs, pad_mask, ar_mask

    def forward(self, observation, actions=None, target_values=None, target_distribution=None, **kwargs) -> CriticOutput:
        """Forward pass for Q value prediction.
        
        Args:
            observation: Observation dict from data collator
            actions: Action tensor [B, action_horizon, action_dim]
            target_values: Target Q values [B] for mse/categorical loss
            target_distribution: Target probability distribution [B, num_bins]
                                 for distributional loss
        """
        if self.critic_forward_mode == "vlm":
            return super().forward(observation, target_values=target_values, target_distribution=target_distribution, **kwargs)

        if actions is None:
            raise ValueError("actions required for expert/dual modes")

        (images, img_masks, lang_tokens, lang_masks,
         token_ar_mask, token_loss_mask, _) = self._preprocess_observation(observation)
        batch_size, device = lang_tokens.shape[0], lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(actions)

        stop_gradient = getattr(self.config, 'stop_gradient_to_vlm', False)
        values, hidden_states, logits, probs, prefix_out = self._forward_expert_q(
            prefix_embs, prefix_pad_masks, prefix_ar_masks,
            suffix_embs, suffix_pad_masks, suffix_ar_masks,
            stop_gradient_to_vlm=stop_gradient,
        )

        expert_loss = None
        if target_values is not None or target_distribution is not None:
            expert_loss = self._compute_expert_loss(values, logits, target_values, target_distribution)

        language_loss, language_acc, language_logits = None, None, None
        if self.critic_forward_mode == "dual" and token_loss_mask.any():
            language_loss, language_acc, language_logits = self._compute_ce_loss(
                prefix_out[:, :-1], prefix_pad_masks, lang_tokens, token_loss_mask, truncated_input=False
            )

        total_loss = None
        if expert_loss is not None or language_loss is not None:
            total_loss = torch.zeros(batch_size, device=device)
            if expert_loss is not None:
                total_loss = total_loss + self.config.expert_loss_weight * expert_loss
            if language_loss is not None:
                total_loss = total_loss + self.config.vlm_loss_weight * language_loss
            total_loss = total_loss.mean()

        return CriticOutput(
            loss=total_loss, predicted_values=values, logits=logits, probs=probs, atoms=self.value_head.atoms,
            expert_loss=expert_loss.mean() if expert_loss is not None else None,
            language_loss=language_loss, language_logits=language_logits,
            language_token_acc=language_acc,
            language_loss_mask=token_loss_mask[:, 1:] if self.critic_forward_mode == "dual" else None,
            hidden_states=hidden_states,
        )

    def _forward_expert_q(self, prefix_embs, prefix_pad_masks, prefix_ar_masks,
                          suffix_embs, suffix_pad_masks, suffix_ar_masks, stop_gradient_to_vlm=False):
        """Forward through VLM + Q expert (uses 'q' expert name)."""
        model_dtype = self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype == torch.bfloat16:
            prefix_embs = prefix_embs.to(torch.bfloat16)
            suffix_embs = suffix_embs.to(torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        ar_masks = torch.cat([prefix_ar_masks, suffix_ar_masks], dim=1)
        attn_mask_4d = self._prepare_attention_masks_4d(make_att_2d_masks(pad_masks, ar_masks))
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        def forward_func(prefix_embs, suffix_embs, attn_mask_4d, position_ids, detach_kv):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn_mask_4d, position_ids=position_ids,
                past_key_values=None, inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False, adarms_cond=[None, None], expert_name="q",
                detach_prefix_for_suffix=detach_kv,
            )
            return prefix_out, suffix_out

        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, attn_mask_4d, position_ids, stop_gradient_to_vlm
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return values, cls_hidden, logits, probs, prefix_out

    @torch.no_grad()
    def predict(self, observation, actions) -> CriticOutput:
        """Inference with KV cache."""
        if self.critic_forward_mode == "vlm":
            raise ValueError("predict() not supported for VLM mode")

        (images, img_masks, lang_tokens, lang_masks, token_ar_mask, _, _) = self._preprocess_observation(observation)
        batch_size, device = lang_tokens.shape[0], lang_tokens.device

        prefix_embs, prefix_pad_masks, prefix_ar_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, token_ar_mask
        )
        suffix_embs, suffix_pad_masks, suffix_ar_masks = self.embed_suffix(actions)

        # Phase 1: Prefill
        prefix_attn_4d = self._prepare_attention_masks_4d(make_att_2d_masks(prefix_pad_masks, prefix_ar_masks))
        _, past_kv = self.paligemma_with_expert.forward(
            attention_mask=prefix_attn_4d, position_ids=torch.cumsum(prefix_pad_masks, dim=1) - 1,
            past_key_values=None, inputs_embeds=[prefix_embs, None], use_cache=True,
        )

        # Phase 2: Expert
        prefix_len, suffix_len = prefix_pad_masks.shape[1], suffix_pad_masks.shape[1]
        prefix_2d = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_attn = make_att_2d_masks(suffix_pad_masks, suffix_ar_masks)
        full_attn_4d = self._prepare_attention_masks_4d(torch.cat([prefix_2d, suffix_attn], dim=2))
        suffix_pos = prefix_pad_masks.sum(dim=-1)[:, None] + torch.cumsum(suffix_pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=full_attn_4d, position_ids=suffix_pos,
            past_key_values=past_kv, inputs_embeds=[None, suffix_embs],
            use_cache=False, adarms_cond=[None, None], expert_name="q",
        )

        cls_hidden = suffix_out[:, -1, :].to(self.value_head.value_proj.weight.dtype)
        values, logits, probs = self._compute_value_from_hidden(cls_hidden)
        return CriticOutput(predicted_values=values, logits=logits, probs=probs, atoms=self.value_head.atoms, hidden_states=cls_hidden)


# =============================================================================
# High-level Model Classes
# =============================================================================

class PI05ValueCritic(PI05PreTrainedModel):
    """PI0.5 Value Critic V(s) for RL fine-tuning."""
    config_class = PI05CriticConfig
    _no_split_modules = []

    def __init__(self, config: PI05CriticConfig):
        super().__init__(config)
        config.critic_expert_variant = getattr(config, 'critic_expert_variant', 'gemma_100m')
        self.model = ValueCriticModel(config)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.paligemma_with_expert.paligemma.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.paligemma_with_expert.paligemma.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.model.paligemma_with_expert.paligemma.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.paligemma_with_expert.paligemma.lm_head = new_embeddings

    def resize_token_embeddings(self, new_num_tokens, pad_to_multiple_of=None, mean_resizing=True):
        """Resize token embeddings for both input and output."""
        if pad_to_multiple_of is not None:
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        lm = self.model.paligemma_with_expert.paligemma.language_model
        lm.resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)

        old_lm_head = self.model.paligemma_with_expert.paligemma.lm_head
        old_num_tokens = old_lm_head.weight.shape[0]

        if old_num_tokens != new_num_tokens:
            new_lm_head = nn.Linear(
                old_lm_head.in_features, new_num_tokens,
                bias=old_lm_head.bias is not None,
                device=old_lm_head.weight.device, dtype=old_lm_head.weight.dtype,
            )
            num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
            new_lm_head.weight.data[:num_tokens_to_copy] = old_lm_head.weight.data[:num_tokens_to_copy]
            if old_lm_head.bias is not None:
                new_lm_head.bias.data[:num_tokens_to_copy] = old_lm_head.bias.data[:num_tokens_to_copy]
            if new_num_tokens > old_num_tokens:
                nn.init.normal_(new_lm_head.weight.data[old_num_tokens:], mean=0.0, std=0.02)
                if old_lm_head.bias is not None:
                    new_lm_head.bias.data[old_num_tokens:].zero_()
            self.model.paligemma_with_expert.paligemma.lm_head = new_lm_head

        self.config.vocab_size = new_num_tokens
        return self.get_input_embeddings()

    def forward(self, observation, target_values=None, target_distribution=None, **kwargs) -> CriticOutput:
        return self.model.forward(observation, target_values=target_values, target_distribution=target_distribution, **kwargs)

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        """Predict value for given observation. Returns scalar value."""
        return self.model.predict(observation).predicted_values

    @torch.no_grad()
    def predict_distribution(self, observation) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict value distribution. Returns (values, probs, atoms)."""
        out = self.model.predict(observation)
        return out.predicted_values, out.probs, out.atoms


class PI05QCritic(PI05PreTrainedModel):
    """PI0.5 Q Critic Q(s, a) for RL fine-tuning."""
    config_class = PI05CriticConfig
    _no_split_modules = []

    def __init__(self, config: PI05CriticConfig):
        super().__init__(config)
        config.critic_expert_variant = getattr(config, 'critic_expert_variant', 'gemma_300m')
        self.model = QCriticModel(config)
        self.post_init()

    def forward(self, observation, actions=None, target_values=None, target_distribution=None, **kwargs) -> CriticOutput:
        return self.model.forward(observation, actions=actions, target_values=target_values, target_distribution=target_distribution, **kwargs)

    @torch.no_grad()
    def predict_q_value(self, observation, actions) -> Tensor:
        """Predict Q value for given observation and actions. Returns scalar value."""
        return self.model.predict(observation, actions).predicted_values

    @torch.no_grad()
    def predict_distribution(self, observation, actions) -> Tuple[Tensor, Tensor, Tensor]:
        """Predict Q value distribution. Returns (values, probs, atoms)."""
        out = self.model.predict(observation, actions)
        return out.predicted_values, out.probs, out.atoms
