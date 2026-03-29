from typing import Any, TypeAlias

from groot.vla.model.dreamzero.modules.wan2_1_attention import AttentionModule
from groot.vla.model.n1_5.modules.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)
from groot.vla.model.dreamzero.modules.wan2_1_submodule import (
    WanRMSNorm,
    rope_action_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, create_mask
from torch.nn.attention.flex_attention import BlockMask
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.distributed as dist
import os

ENABLE_TENSORRT = os.getenv("ENABLE_TENSORRT", "False").lower() == "true"


class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)


class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


def causal_rope_action_apply(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index):
    if ENABLE_TENSORRT:
        return causal_rope_action_apply_no_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)
    else:
        return causal_rope_action_apply_polar(x, freqs, freqs_action, freqs_state, action_register_length, num_action_per_block, num_state_per_block, action_state_index)


def causal_rope_action_apply_no_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, d = x.shape
    
    # (B, seq_len, n, d) -> (B, seq_len, n, d/2, 2)
    x = x.reshape(B, seq_len, n, -1, 2)
    x_real = x[..., 0] 
    x_imag = x[..., 1] 
    
    # Split freqs into cos and sin components
    freqs = freqs.unsqueeze(0).view(1, freqs.shape[0], 1, -1, 2)
    freqs_cos = freqs[..., 0] # Shape: (1, seq_len', 1, d/2)
    freqs_sin = freqs[..., 1] # Shape: (1, seq_len', 1, d/2)
    
    #  Handle the Action/State Register Frequencies
    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        
        freqs_action_slice = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state_slice = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        
        # Combine the action/state tokens for this frame
        freqs_1d = torch.cat([freqs_action_slice, freqs_state_slice], dim=0).view(
            action_register_length, 1, -1, 2
        )
        
        # Split the new action/state frequencies
        freqs_cos_1d = freqs_1d[..., 0]
        freqs_sin_1d = freqs_1d[..., 1]

        # Append the action/state register sin/cos to the main sequence sin/cos
        freqs_cos = torch.cat([freqs_cos[0], freqs_cos_1d], dim=0).unsqueeze(0)
        freqs_sin = torch.cat([freqs_sin[0], freqs_sin_1d], dim=0).unsqueeze(0)
    
    x_real_rotated = x_real * freqs_cos - x_imag * freqs_sin
    x_imag_rotated = x_real * freqs_sin + x_imag * freqs_cos
    
    x_rotated = torch.stack((x_real_rotated, x_imag_rotated), dim=-1)
    
    return x_rotated.flatten(3)

def causal_rope_action_apply_polar(
    x: torch.Tensor,
    freqs: torch.Tensor,
    freqs_action: torch.Tensor,
    freqs_state: torch.Tensor,
    action_register_length: int | None,
    num_action_per_block: int,
    num_state_per_block: int,
    action_state_index: int,
):
    B, seq_len, n, _ = x.shape

    # precompute multipliers
    x = torch.view_as_complex(
        x.to(torch.float64).reshape(B, seq_len, n, -1, 2)
    )

    if action_register_length is not None:
        assert action_register_length == (num_action_per_block + num_state_per_block)
        freqs_action = freqs_action[
            action_state_index * num_action_per_block:(action_state_index + 1) * num_action_per_block
        ]
        freqs_state = freqs_state[
            action_state_index * num_state_per_block:(action_state_index + 1) * num_state_per_block
        ]
        freqs_1d = torch.cat([freqs_action, freqs_state], dim=0).view(action_register_length, 1, -1)
        freqs = torch.cat([freqs, freqs_1d], dim=0)

    # apply rotary embedding
    freqs = freqs.unsqueeze(0)
    x = torch.view_as_real(x * freqs).flatten(3)

    return x


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.num_frame_per_block = num_frame_per_block
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 21 * frame_seqlen if local_attn_size == -1 else local_attn_size * frame_seqlen
        self.frame_seqlen = frame_seqlen
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block
        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim)
        self.causal_attn = AttentionModule(num_heads=self.num_heads, head_dim=self.head_dim, causal=True)

    def _visualize_attention_mask(self, total_len, first_image_len, image_blocks_len, 
                                   action_len, state_len, num_image_blocks, 
                                   num_action_blocks, num_state_blocks,
                                   num_frame_per_block, frame_seqlen,
                                   num_action_per_block, num_state_per_block):
        """
        Create and print a visualization of the attention mask pattern.
        Returns a binary mask [total_len, total_len] where 1 = can attend, 0 = cannot attend.
        """
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Create mask tensor
        mask = torch.zeros(total_len, total_len, dtype=torch.bool)
        
        # First image: self-attention only
        mask[first_image_start:first_image_end, first_image_start:first_image_end] = True
        
        # Image blocks
        for block_idx in range(num_image_blocks):
            block_start = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            
            # Attend to first image
            mask[block_start:block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[block_start:block_end, image_kv_start:block_end] = True
            
            # Attend to current action block
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            mask[block_start:block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[block_start:block_end, state_block_start:state_block_end] = True
        
        # Action blocks
        for block_idx in range(num_action_blocks):
            action_block_start = action_start + block_idx * num_action_per_block
            action_block_end = action_start + (block_idx + 1) * num_action_per_block
            
            # Attend to first image
            mask[action_block_start:action_block_end, first_image_start:first_image_end] = True
            
            # Attend to previous and current image blocks
            image_block_end = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            mask[action_block_start:action_block_end, image_kv_start:image_block_end] = True
            
            # Self-attention
            mask[action_block_start:action_block_end, action_block_start:action_block_end] = True
            
            # Attend to current state block
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[action_block_start:action_block_end, state_block_start:state_block_end] = True
        
        # State blocks: self-attention only
        for block_idx in range(num_state_blocks):
            state_block_start = state_start + block_idx * num_state_per_block
            state_block_end = state_start + (block_idx + 1) * num_state_per_block
            mask[state_block_start:state_block_end, state_block_start:state_block_end] = True
        
        return mask

    def _blockwise_causal_flash_attn(self, q, k, v, frame_seqlen, num_frame_per_block=1, 
                                       action_horizon=None, state_horizon=None, 
                                       num_action_per_block=None, num_state_per_block=None,
                                       visualize_mask=False):
        """
        Implement blockwise causal attention using flash_attention.
        Matches the pattern from _prepare_blockwise_causal_attn_mask:
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image blocks + current action block + current state block
        - Action blocks: can attend to previous image blocks + current image block + current state block + first image
        - State blocks: conditioning only, cannot attend to anything
        
        Args:
            q, k, v: Query, key, value tensors [B, L, num_heads, head_dim]
            frame_seqlen: Number of tokens per frame
            num_frame_per_block: Number of frames per attention block
            action_horizon: Total number of action tokens (if None, no action/state tokens)
            state_horizon: Total number of state tokens (if None, no action/state tokens)
            num_action_per_block: Number of action tokens per block
            num_state_per_block: Number of state tokens per block
            visualize_mask: If True, print the attention mask pattern
        
        Returns:
            Attention output [B, L, num_heads, head_dim]
        """
        b, total_len, n, d = q.shape
        
        # Check if we have action/state tokens
        has_action_state = (action_horizon is not None and state_horizon is not None)
        
        if not has_action_state:
            # OPTIMIZED: Simple blockwise causal attention (without action/state tokens)
            num_frames = total_len // frame_seqlen
            block_size = frame_seqlen * num_frame_per_block
            num_blocks = (num_frames - 1) // num_frame_per_block
            
            # Handle edge case when sequence is too short (no blocks to process)
            if num_blocks <= 0:
                # Process entire sequence as a single block
                return self.attn(q, k, v)
            
            # OPTIMIZATION: For global attention, process all blocks in one call with causal masking
            if self.local_attn_size == -1:
                # Single flash_attention call with causal=True for all blocks at once
                # This is much faster than looping!
                return self.causal_attn(q, k, v)
            
            # With local attention, still need loop but optimize it
            # Pre-allocate output tensor
            output = torch.empty_like(q)
            
            # Pre-compute block boundaries
            block_starts = [frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            kv_starts = [max(0, end - self.local_attn_size * frame_seqlen) for end in block_ends]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                kv_start = kv_starts[block_idx]
                
                output[:, block_start:block_end] = self.attn(
                    q[:, block_start:block_end],
                    k[:, kv_start:block_end],
                    v[:, kv_start:block_end]
                )
            
            return output

        assert action_horizon is not None and state_horizon is not None
        assert num_action_per_block is not None and num_state_per_block is not None

        # Multi-modal structure: [first image] [image blocks] [action blocks] [state blocks]
        # Calculate block structure
        first_image_len = frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        image_blocks_len = total_len - first_image_len - action_len - state_len
        
        num_image_blocks = image_blocks_len // (num_frame_per_block * frame_seqlen)
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block

        assert num_image_blocks == num_action_blocks == num_state_blocks
        
        # Token ranges
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Visualize attention mask if requested
        if visualize_mask:
            mask = self._visualize_attention_mask(
                total_len, first_image_len, image_blocks_len, 
                action_len, state_len, num_image_blocks,
                num_action_blocks, num_state_blocks,
                num_frame_per_block, frame_seqlen,
                num_action_per_block, num_state_per_block
            )
            
            print("\n" + "="*80)
            print("ATTENTION MASK VISUALIZATION")
            print("="*80)
            print(f"Total length: {total_len}")
            print(f"First image: [{first_image_start}:{first_image_end}] (len={first_image_len})")
            print(f"Image blocks: [{image_blocks_start}:{image_blocks_end}] (len={image_blocks_len}, num_blocks={num_image_blocks})")
            print(f"Action tokens: [{action_start}:{action_end}] (len={action_len}, num_blocks={num_action_blocks})")
            print(f"State tokens: [{state_start}:{state_end}] (len={state_len}, num_blocks={num_state_blocks})")
            print(f"Local attention size: {self.local_attn_size}")
            print("-"*80)
            
            # Print a downsampled version of the mask if it's too large
            if total_len <= 100:
                # Print full mask for small sequences
                print("Attention mask (1=can attend, 0=cannot attend):")
                print("Rows=Query tokens, Cols=Key tokens")
                for i in range(total_len):
                    row = "".join(["1" if mask[i, j] else "." for j in range(total_len)])
                    print(f"{i:4d}: {row}")
            else:
                # Print downsampled version for large sequences
                downsample = max(1, total_len // 100)
                print(f"Attention mask (downsampled by {downsample}x):")
                print("Rows=Query tokens, Cols=Key tokens (1=can attend, .=cannot attend)")
                for i in range(0, total_len, downsample):
                    row = "".join(["1" if mask[i, j] else "." for j in range(0, total_len, downsample)])
                    print(f"{i:4d}: {row}")
            
            # Save mask as image
            try:
                import cv2
                import numpy as np
                mask_np = mask.cpu().float().numpy()
                # Resize for visualization if needed
                if total_len > 1000:
                    mask_np = cv2.resize(mask_np, (1000, 1000), interpolation=cv2.INTER_NEAREST)
                mask_img = (mask_np * 255).astype(np.uint8)
                cv2.imwrite("attention_mask_blockwise_flash.png", mask_img)
                print(f"\nMask saved to: attention_mask_blockwise_flash.png")
            except Exception as e:
                print(f"Could not save mask image: {e}")
            
            print("="*80 + "\n")
        
        # OPTIMIZED: Pre-allocate output tensor and pre-compute all indices
        output = torch.empty_like(q)
        
        # Process first image (conditioning, can only self-attend)
        output[:, first_image_start:first_image_end] = self.attn(
            q[:, first_image_start:first_image_end],
            k[:, first_image_start:first_image_end],
            v[:, first_image_start:first_image_end]
        )
        
        # Pre-compute all block indices for image blocks
        image_block_starts = [image_blocks_start + i * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        image_block_ends = [image_blocks_start + (i + 1) * num_frame_per_block * frame_seqlen for i in range(num_image_blocks)]
        if self.local_attn_size != -1:
            image_kv_starts = [max(image_blocks_start, end - self.local_attn_size * frame_seqlen) for end in image_block_ends]
        else:
            image_kv_starts = [image_blocks_start] * num_image_blocks
        
        # Pre-compute action and state block indices
        action_block_starts = [action_start + i * num_action_per_block for i in range(num_action_blocks)]
        action_block_ends = [action_start + (i + 1) * num_action_per_block for i in range(num_action_blocks)]
        state_block_starts = [state_start + i * num_state_per_block for i in range(num_state_blocks)]
        state_block_ends = [state_start + (i + 1) * num_state_per_block for i in range(num_state_blocks)]
        
        # Process each image block
        for block_idx in range(num_image_blocks):
            block_start = image_block_starts[block_idx]
            block_end = image_block_ends[block_idx]
            image_kv_start = image_kv_starts[block_idx]
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Build context: first image + relevant image blocks + current action + current state
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, block_start:block_end] = self.attn(
                q[:, block_start:block_end], k_context, v_context
            )
        
        # Process each action block
        for block_idx in range(num_action_blocks):
            action_block_start = action_block_starts[block_idx]
            action_block_end = action_block_ends[block_idx]
            image_block_end = image_block_ends[block_idx]
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            # Determine image context range
            if self.local_attn_size != -1:
                image_kv_start = max(image_blocks_start, image_block_end - self.local_attn_size * frame_seqlen)
            else:
                image_kv_start = image_blocks_start
            
            # Build context
            k_context = torch.cat([
                k[:, first_image_start:first_image_end],  # First image
                k[:, image_kv_start:image_block_end],  # Image blocks
                k[:, action_block_start:action_block_end],  # Current action block
                k[:, state_block_start:state_block_end]  # Current state block
            ], dim=1)
            v_context = torch.cat([
                v[:, first_image_start:first_image_end],
                v[:, image_kv_start:image_block_end],
                v[:, action_block_start:action_block_end],
                v[:, state_block_start:state_block_end]
            ], dim=1)
            
            output[:, action_block_start:action_block_end] = self.attn(
                q[:, action_block_start:action_block_end], k_context, v_context
            )
        
        # Process state blocks (conditioning, can only self-attend)
        for block_idx in range(num_state_blocks):
            state_block_start = state_block_starts[block_idx]
            state_block_end = state_block_ends[block_idx]
            
            output[:, state_block_start:state_block_end] = self.attn(
                q[:, state_block_start:state_block_end],
                k[:, state_block_start:state_block_end],
                v[:, state_block_start:state_block_end]
            )
        
        return output

    def _process_clean_image_only(self, clean_image_q, clean_image_k, clean_image_v, clean_frames):
        """Process clean image blocks with causal attention pattern - OPTIMIZED
        
        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to first frame + previous blocks (0 to i-1) + current block
        
        OPTIMIZATION: Instead of looping through blocks, we batch process them together
        by using a single flash_attention call with properly structured KV cache.
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (clean_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            # Only first frame - single attention call
            return self.attn(
                clean_image_q[:, :self.frame_seqlen],
                clean_image_k[:, :self.frame_seqlen],
                clean_image_v[:, :self.frame_seqlen]
            )
        
        # Pre-allocate output tensor (avoids list append + cat overhead)
        b, total_len, n, d = clean_image_q.shape
        output = torch.empty_like(clean_image_q)
        
        # First frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            clean_image_q[:, :self.frame_seqlen],
            clean_image_k[:, :self.frame_seqlen],
            clean_image_v[:, :self.frame_seqlen]
        )
        
        # OPTIMIZATION: Process all blocks together with causal masking
        # For global attention (no local_attn_size), we can process all blocks in one call
        if self.local_attn_size == -1:
            # Single attention call for all blocks!
            # Each position can attend to first_frame + everything up to itself
            blocks_q = clean_image_q[:, self.frame_seqlen:]
            blocks_k = clean_image_k  # Can attend to everything including first frame
            blocks_v = clean_image_v
            
            # Use causal masking: each block token can see first frame + all previous tokens
            output[:, self.frame_seqlen:] = self.causal_attn(
                blocks_q, blocks_k, blocks_v
            )
        else:
            # With local attention, we still need to loop but with optimizations
            # Pre-compute all block boundaries to reduce overhead
            block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
            block_ends = [min(start + block_size, total_len) for start in block_starts]
            
            for block_idx in range(num_blocks):
                block_start = block_starts[block_idx]
                block_end = block_ends[block_idx]
                
                q_block = clean_image_q[:, block_start:block_end]
                
                # Context: first frame + recent blocks within local_attn_size
                image_kv_start = max(self.frame_seqlen, block_end - self.local_attn_size * self.frame_seqlen)
                k_context = torch.cat([
                    clean_image_k[:, :self.frame_seqlen],  # First frame
                    clean_image_k[:, image_kv_start:block_end]  # Recent blocks + current
                ], dim=1)
                v_context = torch.cat([
                    clean_image_v[:, :self.frame_seqlen],
                    clean_image_v[:, image_kv_start:block_end]
                ], dim=1)
                
                output[:, block_start:block_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_state_blocks(self, state_q, state_k, state_v, state_horizon):
        """Process state blocks: self-attention only - OPTIMIZED
        
        OPTIMIZATION: State blocks only do self-attention within each block.
        Instead of looping, we can process all blocks in a single call with block-diagonal masking,
        or even simpler: just one attention call since they're independent.
        """
        num_blocks = state_horizon // self.num_state_per_block
        
        if num_blocks == 1:
            # Single block - one attention call
            return self.attn(state_q, state_k, state_v)
        
        # OPTIMIZATION: Since each state block only attends to itself (no cross-block attention),
        # we can process all blocks in a single batched call. Flash attention will handle this
        # efficiently. The blocks are independent, so this is safe.
        # Alternative: reshape and process as separate batch items
        
        # Pre-allocate output
        output = torch.empty_like(state_q)
        
        # Process all blocks (keeping loop for now due to block-diagonal pattern)
        # This could be further optimized with custom masking
        for block_idx in range(num_blocks):
            state_block_start = block_idx * self.num_state_per_block
            state_block_end = state_block_start + self.num_state_per_block
            
            output[:, state_block_start:state_block_end] = self.attn(
                state_q[:, state_block_start:state_block_end],
                state_k[:, state_block_start:state_block_end],
                state_v[:, state_block_start:state_block_end]
            )
        
        return output
    
    def _process_noisy_image_blocks(self, noisy_image_q, noisy_image_k, noisy_image_v,
                                     clean_image_k, clean_image_v,
                                     noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                                     half_frames, action_horizon, state_horizon):
        """Process noisy image blocks with teacher forcing pattern - OPTIMIZED
        
        First frame: conditioning, cannot attend to anything (self-attention only)
        Block i: attends to action[i] + state[i] + first_clean_frame + clean_blocks[0:i] + current_noisy_block
        
        OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
        """
        block_size = self.frame_seqlen * self.num_frame_per_block
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        # Pre-allocate output tensor
        output = torch.empty_like(noisy_image_q)
        
        # First noisy frame: conditioning, self-attention only
        output[:, :self.frame_seqlen] = self.attn(
            noisy_image_q[:, :self.frame_seqlen],
            noisy_image_k[:, :self.frame_seqlen],
            noisy_image_v[:, :self.frame_seqlen]
        )
        
        if num_blocks == 0:
            return output
        
        # Pre-compute all block indices to reduce loop overhead
        noisy_block_starts = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        noisy_block_ends = [min(start + block_size, noisy_image_q.shape[1]) for start in noisy_block_starts]
        clean_context_ends = [self.frame_seqlen + i * block_size for i in range(num_blocks)]
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        # Process noisy image blocks
        for block_idx in range(num_blocks):
            noisy_start = noisy_block_starts[block_idx]
            noisy_end = noisy_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            q_block = noisy_image_q[:, noisy_start:noisy_end]
            
            # Build context: first_clean_frame + clean_blocks[0:i] + current_noisy_block + action[i] + state[i]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_start:noisy_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_start:noisy_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, noisy_start:noisy_end] = self.attn(q_block, k_context, v_context)
        
        return output
    
    def _process_noisy_action_blocks(self, noisy_action_q, noisy_action_k, noisy_action_v,
                                      clean_image_k, clean_image_v,
                                      noisy_image_k, noisy_image_v,
                                      noisy_state_k, noisy_state_v,
                                      half_frames, action_horizon, state_horizon):
        """Process noisy action blocks with teacher forcing pattern - OPTIMIZED
        
        First action (for first frame): cannot attend to anything (self-attention only)
        Action block i: attends to first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
        
        OPTIMIZATION: Pre-allocate output, pre-compute indices, reduce memory allocations
        """
        num_blocks = (half_frames - 1) // self.num_frame_per_block
        
        if num_blocks == 0:
            return torch.empty_like(noisy_action_q)
        
        # Pre-allocate output tensor
        output = torch.empty_like(noisy_action_q)
        
        # Pre-compute all block indices
        action_block_starts = [i * self.num_action_per_block for i in range(num_blocks)]
        action_block_ends = [start + self.num_action_per_block for start in action_block_starts]
        clean_context_ends = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_starts = [self.frame_seqlen + i * self.frame_seqlen * self.num_frame_per_block for i in range(num_blocks)]
        noisy_image_block_ends = [start + self.frame_seqlen * self.num_frame_per_block for start in noisy_image_block_starts]
        state_block_starts = [i * self.num_state_per_block for i in range(num_blocks)]
        state_block_ends = [start + self.num_state_per_block for start in state_block_starts]
        
        # Process noisy action blocks
        for block_idx in range(num_blocks):
            action_start = action_block_starts[block_idx]
            action_end = action_block_ends[block_idx]
            clean_end = clean_context_ends[block_idx]
            noisy_img_start = noisy_image_block_starts[block_idx]
            noisy_img_end = noisy_image_block_ends[block_idx]
            state_start = state_block_starts[block_idx]
            state_end = state_block_ends[block_idx]
            
            q_block = noisy_action_q[:, action_start:action_end]
            
            # Build context: first_clean_frame + clean_blocks[0:i] + noisy_image[i] + action[i] + state[i]
            k_context = torch.cat([
                clean_image_k[:, :clean_end],
                noisy_image_k[:, noisy_img_start:noisy_img_end],
                noisy_action_k[:, action_start:action_end],
                noisy_state_k[:, state_start:state_end]
            ], dim=1)
            v_context = torch.cat([
                clean_image_v[:, :clean_end],
                noisy_image_v[:, noisy_img_start:noisy_img_end],
                noisy_action_v[:, action_start:action_end],
                noisy_state_v[:, state_start:state_end]
            ], dim=1)
            
            output[:, action_start:action_end] = self.attn(q_block, k_context, v_context)
        
        return output

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        kv_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        updated_kv_cache: torch.Tensor | None = None

        if kv_cache is None:
            if is_tf:
                # Teacher forcing training.
                if action_register_length is not None:
                    q_context = q[:, :(s-action_register_length)//2]
                    k_context = k[:, :(s-action_register_length)//2]
                    q_noisy = q[:, (s-action_register_length)//2:]  
                    k_noisy = k[:, (s-action_register_length)//2:]
                else:
                    q_context = q[:, :s//2]
                    k_context = k[:, :s//2]
                    q_noisy = q[:, s//2:]
                    k_noisy = k[:, s//2:]
                roped_query = []
                roped_key = []

                # rope should be same for clean and noisy parts
                rq_context = rope_action_apply(
                    x=q_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)
                rk_context = rope_action_apply(
                    x=k_context,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=None,
                ).type_as(v)

                rq_noisy = rope_action_apply(
                    x=q_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                rk_noisy = rope_action_apply(
                    x=k_noisy,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                roped_query.append(rq_context)
                roped_key.append(rk_context)
                roped_query.append(rq_noisy)
                roped_key.append(rk_noisy)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)
                # Calculate sequence dimensions
                half_seq_len = (s - (action_register_length if action_register_length is not None else 0)) // 2
                
                if action_register_length is not None:
                    # Teacher forcing structure:
                    # Clean half: [image tokens only]
                    # Noisy half: [image tokens][action tokens][state tokens]
                    # Causality only applies to image blocks!
                    
                    # Clean half contains ONLY image tokens
                    clean_image_seq_len = half_seq_len
                    clean_frames = clean_image_seq_len // self.frame_seqlen
                    
                    # Noisy half contains image + action + state tokens
                    noisy_image_seq_len = half_seq_len
                    noisy_frames = noisy_image_seq_len // self.frame_seqlen
                    num_image_blocks = (noisy_frames - 1) // self.num_frame_per_block
                    action_horizon = num_image_blocks * self.num_action_per_block
                    state_horizon = num_image_blocks * self.num_state_per_block
                    
                    # Split clean and noisy parts
                    # Clean: [image tokens only]
                    clean_image_q = roped_query[:, :clean_image_seq_len]
                    clean_image_k = roped_key[:, :clean_image_seq_len]
                    clean_image_v = v[:, :clean_image_seq_len]

                    assert roped_query.shape[1] == half_seq_len + noisy_image_seq_len + action_horizon + state_horizon
                    
                    # Noisy: [image tokens][action tokens][state tokens]
                    noisy_image_q = roped_query[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_q = roped_query[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_q = roped_query[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    noisy_image_k = roped_key[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_k = roped_key[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_k = roped_key[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    noisy_image_v = v[:, half_seq_len:half_seq_len + noisy_image_seq_len]
                    noisy_action_v = v[:, half_seq_len + noisy_image_seq_len:half_seq_len + noisy_image_seq_len + action_horizon]
                    noisy_state_v = v[:, half_seq_len + noisy_image_seq_len + action_horizon:]
                    
                    # ========== Process CLEAN (context) image tokens ==========
                    # Clean images: simple blockwise causal attention (no action/state)
                    clean_image_outputs = self._process_clean_image_only(
                        clean_image_q, clean_image_k, clean_image_v, clean_frames)
                    
                    # ========== Process NOISY tokens ==========
                    # Noisy image blocks: attend to previous clean image blocks + current noisy image + current noisy action + current noisy state
                    noisy_image_outputs = self._process_noisy_image_blocks(
                        noisy_image_q, noisy_image_k, noisy_image_v,
                        clean_image_k, clean_image_v,
                        noisy_action_k, noisy_action_v, noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # Noisy action blocks: attend to previous clean image blocks (including first) + current noisy image + current noisy action + same state
                    noisy_action_outputs = self._process_noisy_action_blocks(
                        noisy_action_q, noisy_action_k, noisy_action_v,
                        clean_image_k, clean_image_v, 
                        noisy_image_k, noisy_image_v,
                        noisy_state_k, noisy_state_v,
                        noisy_frames, action_horizon, state_horizon)
                    
                    # Noisy state blocks: self-attention only
                    noisy_state_outputs = self._process_state_blocks(
                        noisy_state_q, noisy_state_k, noisy_state_v, state_horizon)
                    
                    # Concatenate all outputs in order: clean_img, noisy_img, noisy_act, noisy_state
                    x = torch.cat([
                        clean_image_outputs,
                        noisy_image_outputs, noisy_action_outputs, noisy_state_outputs
                    ], dim=1)
                else:
                    # No action/state tokens, fall back to simple image-only teacher forcing
                    half_frames = half_seq_len // self.frame_seqlen
                    clean_q = roped_query[:, :half_seq_len]
                    clean_k = roped_key[:, :half_seq_len]
                    clean_v = v[:, :half_seq_len]
                    noisy_q = roped_query[:, half_seq_len:]
                    noisy_k = roped_key[:, half_seq_len:]
                    noisy_v = v[:, half_seq_len:]
                    
                    # Process clean frames with blockwise causal attention
                    x_clean = self._blockwise_causal_flash_attn(
                        clean_q, clean_k, clean_v, self.frame_seqlen, self.num_frame_per_block,
                        action_horizon=None, state_horizon=None,
                        num_action_per_block=None, num_state_per_block=None,
                        visualize_mask=False)
                    
                    # Process noisy frames: attend to all clean frames + themselves
                    full_k = torch.cat([clean_k, noisy_k], dim=1)
                    full_v = torch.cat([clean_v, noisy_v], dim=1)
                    x_noisy = self.attn(noisy_q, full_k, full_v)
                    
                    x = torch.cat([x_clean, x_noisy], dim=1)

            else:
                roped_query = rope_action_apply(
                    x=q,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)
                roped_key = rope_action_apply(
                    x=k,
                    freqs=freqs,
                    freqs_action=freqs_action,
                    freqs_state=freqs_state,
                    action_register_length=action_register_length,
                    num_action_per_block=self.num_action_per_block,
                    num_state_per_block=self.num_state_per_block,
                ).type_as(v)

                # Calculate dynamic action and state horizons
                if action_register_length is not None:
                    chunk_size = action_register_length // (self.num_action_per_block + self.num_state_per_block)
                    action_horizon = chunk_size * self.num_action_per_block
                    state_horizon = chunk_size * self.num_state_per_block
                else:
                    action_horizon = None
                    state_horizon = None

                # Use blockwise causal flash attention without massive padding
                visualize = False
                x = self._blockwise_causal_flash_attn(
                    roped_query, roped_key, v, self.frame_seqlen, self.num_frame_per_block,
                    action_horizon=action_horizon,
                    state_horizon=state_horizon,
                    num_action_per_block=self.num_action_per_block if action_register_length else None,
                    num_state_per_block=self.num_state_per_block if action_register_length else None,
                    visualize_mask=visualize)

        else:
            action_state_index = (current_start_frame - 1) // self.num_frame_per_block

            roped_query = causal_rope_action_apply(
                x=q,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)
            roped_key = causal_rope_action_apply(
                x=k,
                freqs=freqs,
                freqs_action=freqs_action,
                freqs_state=freqs_state,
                action_register_length=action_register_length,
                num_action_per_block=self.num_action_per_block,
                num_state_per_block=self.num_state_per_block,
                action_state_index=action_state_index,
            ).type_as(v)

            # split roped_query and roped_action_query (the last action_register_length tokens)
            roped_action_query: torch.Tensor | None = None
            roped_action_key: torch.Tensor | None = None
            action_v: torch.Tensor | None = None

            if action_register_length is not None:
                roped_action_query = roped_query[:, -action_register_length:]
                roped_query = roped_query[:, :-action_register_length]
                roped_action_key = roped_key[:, -action_register_length:]
                roped_key = roped_key[:, :-action_register_length]
                action_v = v[:, -action_register_length:]
                v = v[:, :-action_register_length]
                assert roped_action_query is not None
                assert roped_action_key is not None
                assert action_v is not None

            num_new_tokens = roped_query.shape[1]
            assert roped_key.shape[1] == num_new_tokens
            assert v.shape[1] == num_new_tokens

            # If we are using local attention and the current KV cache size is larger
            # than the local attention size, we need to truncate the KV cache

            updated_kv_cache = kv_cache
            updated_k = updated_kv_cache[0]
            updated_v = updated_kv_cache[1]
            # Assign new keys/values directly up to current_end
            new_k = torch.cat([updated_k, roped_key], dim=1)
            new_v = torch.cat([updated_v, v], dim=1)

            # We may need to truncate the KV cache if it's size is larger than the max attention size.
            new_k = new_k[:, -self.max_attention_size:]
            new_v = new_v[:, -self.max_attention_size:]

            if action_register_length is not None:
                x = self.attn(
                    torch.cat([roped_query, roped_action_query], dim=1),
                    torch.cat([new_k, roped_action_key], dim=1),
                    torch.cat([new_v, action_v], dim=1),
                )
            else:
                x = self.attn(
                    roped_query,
                    new_k,
                    new_v,
                )
            updated_kv_cache = torch.stack([new_k, new_v], dim=0)


        # output
        x = x.flatten(2)
        x = self.o(x)
        return x, updated_kv_cache


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 frame_seqlen,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6,
                 num_action_per_block=32,
                 num_state_per_block=1):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(
            dim=dim,
            num_heads=num_heads,
            frame_seqlen=frame_seqlen,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            num_frame_per_block=num_frame_per_block,
            qk_norm=qk_norm,
            eps=eps,
            num_action_per_block=num_action_per_block,
            num_state_per_block=num_state_per_block,
        )
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
        freqs: torch.Tensor,
        freqs_action: torch.Tensor,
        freqs_state: torch.Tensor,
        action_register_length: int | None,
        context: torch.Tensor,
        kv_cache: torch.Tensor | None = None,
        crossattn_cache: torch.Tensor | None = None,
        current_start_frame: int = 0,
        is_tf: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        # self-attention
        y, updated_kv_cache = self.self_attn(
            x=(self.norm1(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)),
            freqs=freqs,
            freqs_action=freqs_action,
            freqs_state=freqs_state,
            action_register_length=action_register_length,
            kv_cache=kv_cache,
            is_tf=is_tf,
            current_start_frame=current_start_frame,
        )
        x = x + (y * e[2].squeeze(2))

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, e):
            x = x + self.cross_attn(self.norm3(x), context)
            y = self.ffn(
                (self.norm2(x) * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            )
            x = x + (y * e[5].squeeze(2))
            return x

        x = cross_attn_ffn(x, context, e)
        return x, updated_kv_cache


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 frame_seqlen=220,
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 max_chunk_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6,
                 num_frame_per_block=1, 
                 action_dim=32,
                 num_registers=8,
                 max_state_dim=64,
                 max_num_embodiments=32,
                 hidden_size=1024,
                 diffusion_model_pretrained_path=None,
                 num_action_per_block=32,
                 num_state_per_block=1):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.frame_seqlen = frame_seqlen
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = max_chunk_size * num_frame_per_block + 1 if max_chunk_size != -1 else -1
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.num_frame_per_block = num_frame_per_block
        self.diffusion_model_pretrained_path = diffusion_model_pretrained_path
        self.action_dim = action_dim
        self.num_registers = num_registers
        self.max_state_dim = max_state_dim
        self.max_num_embodiments = max_num_embodiments
        self.hidden_size = hidden_size
        self.num_action_per_block = num_action_per_block
        self.num_state_per_block = num_state_per_block

        max_num_embodiments = 1

        self.state_encoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=max_state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=action_dim,
            hidden_size=self.dim,
            num_embodiments=max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=max_num_embodiments,
            input_dim=dim,
            hidden_dim=self.hidden_size,
            output_dim=action_dim,
        )

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads, frame_seqlen,
                                    self.local_attn_size, sink_size, num_frame_per_block, qk_norm, cross_attn_norm, eps,
                                    num_action_per_block, num_state_per_block)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        
        self.freqs_action = rope_params(1024*10, d)
        self.freqs_state = rope_params(1024, d)
        self.freqs = [
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
        ]
        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = True
        self.independent_first_frame = False if self.num_frame_per_block == 1 else True


    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1, action_horizon=1, state_horizon=1, num_action_per_block=30, num_state_per_block=1
    ) -> BlockMask:
        """
        We will divide the token sequence into the following format:
        [first image (conditioning)] [image blocks] [action blocks] [state blocks]
        
        Structure:
        - First image: conditioning only, cannot attend to anything
        - Image blocks: can attend to first image + previous image block + current action block + current state block
        - Action blocks: can attend to previous image block + current image block + current state block
        - State blocks: conditioning only, cannot attend to anything
        
        Block alignment:
        - num_image_blocks = (num_frames - 1) // num_frame_per_block
        - num_action_blocks = action_horizon // num_action_per_block  
        - num_state_blocks = state_horizon // num_state_per_block
        - num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        """
        # Calculate block structure
        num_image_blocks = (num_frames - 1) // num_frame_per_block
        num_action_blocks = action_horizon // num_action_per_block
        num_state_blocks = state_horizon // num_state_per_block
        
        # Verify the relationship: num_image_blocks = num_action_blocks + 1 = num_state_blocks + 1
        assert num_image_blocks == num_action_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_action_blocks}"
        assert num_image_blocks == num_state_blocks, \
            f"image_blocks mismatch: {num_image_blocks} != {num_state_blocks}"
        
        # Token ranges
        first_image_len = frame_seqlen  # First image (conditioning)
        image_blocks_len = num_image_blocks * num_frame_per_block * frame_seqlen
        action_len = action_horizon
        state_len = state_horizon
        total_length = first_image_len + image_blocks_len + action_len + state_len
        
        # print("total_length", total_length, first_image_len, image_blocks_len, action_len, state_len)
        # Padding to multiple of 128
        # padded_length = math.ceil(total_length / 128) * 128 - total_length
        padded_length = math.ceil((local_attn_size * frame_seqlen + (local_attn_size - 1) + 32 * (local_attn_size - 1))/128) * 128 - total_length
        total_padded_length = total_length + padded_length
        # print("total_padded_length", total_padded_length, total_length, padded_length)
        
        # Define token ranges for each modality
        first_image_start = 0
        first_image_end = first_image_len
        image_blocks_start = first_image_end
        image_blocks_end = image_blocks_start + image_blocks_len
        action_start = image_blocks_end
        action_end = action_start + action_len
        state_start = action_end
        state_end = state_start + state_len
        
        # Precompute block indices for each token
        block_indices = torch.zeros(total_padded_length, device=device, dtype=torch.long)
        
        # First image gets special block index -1 (conditioning, cannot attend to anything)
        block_indices[first_image_start:first_image_end] = -1
        
        # Assign block indices for image blocks (0 to num_image_blocks-1)
        for block_idx in range(num_image_blocks):
            start_idx = image_blocks_start + block_idx * num_frame_per_block * frame_seqlen
            end_idx = image_blocks_start + (block_idx + 1) * num_frame_per_block * frame_seqlen
            block_indices[start_idx:end_idx] = block_idx
        
        # Assign block indices for action tokens (0 to num_action_blocks-1)
        for block_idx in range(num_action_blocks):
            start_idx = action_start + block_idx * num_action_per_block
            end_idx = action_start + (block_idx + 1) * num_action_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # Assign block indices for state tokens (0 to num_state_blocks-1)
        for block_idx in range(num_state_blocks):
            start_idx = state_start + block_idx * num_state_per_block
            end_idx = state_start + (block_idx + 1) * num_state_per_block
            block_indices[start_idx:end_idx] = block_idx
        
        # Padding tokens get block index of last block + 1 (won't attend to anything)
        block_indices[total_length:] = num_image_blocks
        
        def attention_mask(b, h, q_idx, kv_idx):
            # Self-attention
            self_attn = (q_idx == kv_idx)
            
            # Determine which modality q and kv belong to
            q_is_first_image = (q_idx >= first_image_start) & (q_idx < first_image_end)
            q_is_image_block = (q_idx >= image_blocks_start) & (q_idx < image_blocks_end)
            q_is_action = (q_idx >= action_start) & (q_idx < action_end)
            q_is_state = (q_idx >= state_start) & (q_idx < state_end)
            
            kv_is_first_image = (kv_idx >= first_image_start) & (kv_idx < first_image_end)
            kv_is_image_block = (kv_idx >= image_blocks_start) & (kv_idx < image_blocks_end)
            kv_is_action = (kv_idx >= action_start) & (kv_idx < action_end)
            kv_is_state = (kv_idx >= state_start) & (kv_idx < state_end)
            
            q_block = block_indices[q_idx]
            kv_block = block_indices[kv_idx]
            
            # First image query (conditioning) - cannot attend to anything
            first_image_mask = q_is_first_image & False
            
            # Image block query
            image_to_first = q_is_image_block & kv_is_first_image  # Image block to first image: always allowed
            image_to_image = q_is_image_block & kv_is_image_block & (kv_block <= q_block)  # Image block to image block: can attend to current and previous image blocks
            image_to_action = q_is_image_block & kv_is_action & (kv_block == q_block)  # Image block to action: can attend to current action block
            image_to_state = q_is_image_block & kv_is_state & (kv_block == q_block)  # Image block to state: can attend to current state block
            
            image_block_mask = image_to_first | image_to_image | image_to_action | image_to_state
            
            # Action query
            action_to_image = q_is_action & kv_is_image_block & (kv_block <= q_block)  # Action to image block: can attend to current and all previous image blocks
            action_to_action = q_is_action & kv_is_action & (kv_block == q_block)  # Action to action: only same block
            action_to_state = q_is_action & kv_is_state & (kv_block == q_block)  # Action to state: only same block
            action_to_first = q_is_action & kv_is_first_image  # Action to first image: always allowed
            
            action_mask = action_to_image | action_to_action | action_to_state | action_to_first
            
            # State query (conditioning) - cannot attend to anything
            state_mask = q_is_state & False
            
            # Combine all masks
            return self_attn | first_image_mask | image_block_mask | action_mask | state_mask
        
        block_mask = create_block_mask(
            attention_mask, B=None, H=None, 
            Q_LEN=total_padded_length,
            KV_LEN=total_padded_length, 
            _compile=False, device=device
        )
        
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f"Created blockwise causal attention mask:")
            print(f"  first_image_tokens={first_image_len} (conditioning)")
            print(f"  num_image_blocks={num_image_blocks} (blocks of {num_frame_per_block * frame_seqlen})")
            print(f"  num_action_blocks={num_action_blocks} (blocks of {num_action_per_block})")
            print(f"  num_state_blocks={num_state_blocks} (blocks of {num_state_per_block})")
            print(f"  total_length={total_length}, padded_length={padded_length}")
            print(block_mask)

            # Debug: materialize a small slice of the mask into 0/1 strings
            try:
                dense_mask = create_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=total_padded_length,
                    KV_LEN=total_padded_length,
                    device=device,
                )[0, 0]  # [Q, K]
                preview_q = min(979, dense_mask.shape[0])
                preview_k = min(979, dense_mask.shape[1])
                print("Block mask (preview):")
                for qi in range(preview_q):
                    row = dense_mask[qi, :preview_k].to(torch.int8).tolist()
                    print(" ".join(str(int(v)) for v in row))
            except Exception as err:
                print("[warn] Failed to materialize block mask preview:", err)
        
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(self.local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if DEBUG:
            print(block_mask)
            import imageio
            import numpy as np
            from torch.nn.attention.flex_attention import create_mask

            mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length +
                               padded_length, KV_LEN=total_length + padded_length, device=device)
            import cv2
            mask = cv2.resize(mask[0, 0].cpu().float().numpy(), (1024, 1024))
            imageio.imwrite("mask_%d.jpg" % (0), np.uint8(255. * mask))

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(local_attn_size * frame_seqlen/128) * 128 - total_length
        # padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_blocks(
        self,
        x: torch.Tensor,
        seq_len: int,
        freqs: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        clip_feature: torch.Tensor | None,
        embodiment_id: torch.Tensor | None,
        action: torch.Tensor | None,
        timestep_action: torch.Tensor | None,
        state: torch.Tensor | None,
        kv_cache: list[torch.Tensor],
        current_start_frame: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Forward pass through the diffusion model blocks.
        """
        x = x.flatten(start_dim=2).transpose(1, 2)

        B = x.shape[0]
        F = timestep.shape[1]

        if action is not None:
            embodiment_id = torch.tensor([0], device=x.device).repeat(x.shape[0])
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_length = action_features.shape[1]
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            state_features = None
            action_length = 0
            action_register_length = None

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        context = self.text_embedding(context)
        
        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        updated_kv_caches: list[torch.Tensor] = []
        for block_index, block in enumerate(self.blocks):
            x, updated_kv_cache = block(
                x=x,
                e=e0,
                freqs=freqs,
                freqs_action=self.freqs_action,
                freqs_state=self.freqs_state,
                context=context,
                action_register_length=action_register_length,
                kv_cache=kv_cache[block_index],
                current_start_frame=current_start_frame,
            )
            updated_kv_caches.append(updated_kv_cache)

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))

        return x_video, action_noise_pred, updated_kv_caches


    def _forward_inference_trt(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen 
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        ) 

        return x_video, action_noise_pred

    def _forward_inference_trt_droid(
        self,
        x,
        timestep,
        context,
        kv_cache_packed: torch.Tensor,
        y,
        clip_feature,
        action,
        timestep_action,
        state,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:


        frame_seqlen = 880
        seq_len = 2*frame_seqlen 
        kv_cache_seq_len = kv_cache_packed.shape[3]
        current_start_frame =  kv_cache_seq_len // frame_seqlen

        kv_cache_list = []
        for block_index in range(len(self.blocks)):
            kv_cache_list.append(kv_cache_packed[block_index])
        
        x_video, action_noise_pred, _ = self._forward_inference(
            x=x,
            timestep=timestep,
            context=context,
            seq_len=int(seq_len),
            kv_cache=kv_cache_list,
            crossattn_cache=None,
            y=y,
            clip_feature=clip_feature,
            action=action,
            timestep_action=timestep_action,
            state=state,
            current_start_frame = current_start_frame,
        ) 

        return x_video, action_noise_pred


    def _forward_inference(
        self,
        x,
        timestep,
        context,
        seq_len,
        kv_cache: list[torch.Tensor],
        crossattn_cache: list[torch.Tensor],
        current_start_frame: int,
        y=None,
        clip_feature=None,
        action=None,
        timestep_action=None,
        state=None,
        embodiment_id=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            timestep (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            action (Tensor, *optional*):
                Action tensor of shape [B, H, D]
            state (Tensor, *optional*):
                State tensor of shape [B, H, D]
            embodiment_id (Tensor, *optional*):
                Embodiment ID tensor of shape [B]
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x
            clip_feature (Tensor, *optional*):
                CLIP image features for image-to-video mode
            timestep_action (Tensor, *optional*):
                Action timestep tensor of shape [B]
        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """      
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None
        assert context.shape[1] == self.text_len

        if y is not None:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)
        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)

        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=current_start_frame,
        )

        x_video, action_noise_pred, updated_kv_caches = self._forward_blocks(
            x=x,
            seq_len=seq_len,
            freqs=freqs,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            embodiment_id=embodiment_id,
            action=action,
            timestep_action=timestep_action,
            state=state,
            kv_cache=kv_cache,
            current_start_frame=current_start_frame,
        )

        # Copy the updated KV caches back to the original KV cache.
        x_video = x_video.clone()
        if action_noise_pred is not None:
            action_noise_pred = action_noise_pred.clone()
        #for block_index, updated_kv_cache in enumerate(updated_kv_caches):
        #    kv_cache[block_index] = updated_kv_cache.clone()

        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred, updated_kv_caches

    def _forward_train(
        self,
        x,
        timestep,
        timestep_action,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        y=None,
        clip_feature=None,
        action=None,
        state=None,
        embodiment_id=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_feature is not None and y is not None

        if y is not None:
            x = torch.cat([x, y.to(dtype=x.dtype)], dim=1)

        # embeddings
        x = self.patch_embedding(x)

        grid_size = torch.tensor(x.shape[2:], dtype=torch.long)
        freqs = self._create_freqs(
            grid_size=grid_size,
            start_frame=0,
        )

        x = x.flatten(start_dim=2).transpose(1, 2)
        assert x.shape[1] == seq_len

        B = x.shape[0]
        F = timestep.shape[1]

        # time embeddings
        if action is not None:
            embodiment_id = torch.tensor([0]).repeat(x.shape[0]).to(device=embodiment_id.device)
            action_features = self.action_encoder(action, timestep_action, embodiment_id)
            action_length = action_features.shape[1]
            state_features = self.state_encoder(state, embodiment_id)
            action_register = torch.cat([action_features, state_features], dim=1)
            action_register_length = action_register.shape[1]
            x = torch.cat([x, action_register], dim=1)
        else:
            action_features = None
            action_length = None
            state_features = None
            action_register = None
            action_register_length = None

        # time embeddings
        timestep = timestep.unsqueeze(-1).expand(B, F, seq_len // F).reshape(B, -1)
        timestep_original = timestep.clone()

        if action is not None:
            assert timestep_action is not None
            assert state_features is not None
            stride = timestep_action.shape[1] // state_features.shape[1]
            timestep_state = timestep_action[:, ::stride]
            timestep = torch.cat([timestep, timestep_action, timestep_state], dim=1)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, timestep.flatten()).type_as(x))
        e = e.unflatten(dim=0, sizes=(B, -1))
        e0 = self.time_projection(e)
        e0 = e0.unflatten(dim=2, sizes=(6, self.dim))

        # context
        assert context.shape[1] == self.text_len
        context = self.text_embedding(context)

        if clip_feature is not None:
            clip_embedding = self.img_emb(clip_feature)
            context = torch.cat([clip_embedding, context], dim=1)

        if clean_x is not None:
            if y is not None:
                clean_x = torch.cat([clean_x, y.to(dtype=clean_x.dtype)], dim=1)
            clean_x = self.patch_embedding(clean_x)
            clean_x = clean_x.flatten(start_dim=2).transpose(1, 2)
            assert clean_x.shape[1] == seq_len

            x = torch.cat([clean_x, x], dim=1)

            if aug_t is None:
                aug_t = torch.zeros_like(timestep_original)
            assert aug_t is not None

            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t.flatten()).type_as(x))
            e_clean = e_clean.unflatten(dim=0, sizes=timestep_original.shape)
            e0_clean = self.time_projection(e_clean)
            e0_clean = e0_clean.unflatten(dim=2, sizes=(6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            freqs=freqs,
            freqs_action=self.freqs_action,
            freqs_state=self.freqs_state,
            action_register_length=action_register_length,
            context=context,
            is_tf=clean_x is not None,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                outputs, updated_kv_cache = module(*inputs, **kwargs)
                assert updated_kv_cache is None
                return outputs
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, clean_x.shape[1]:]

        if action is not None:
            action_noise_pred = x[:, seq_len: seq_len + action_length]
            action_noise_pred = self.action_decoder(action_noise_pred, embodiment_id)
        else:
            action_noise_pred = None

        # Build a tensor that contains only video tokens per sample with length = max(video_lens)
        x_video = x[:, :seq_len]
        e_video = e[:, :seq_len]

        # Unpatchify video-only tokens
        x_video = self.head(x_video, e_video.unsqueeze(2))
        video_noise_pred = self.unpatchify(x_video, grid_size)

        return video_noise_pred, action_noise_pred

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_size):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (Tensor):
                Patchified features, with shape [B, L, C_out * prod(patch_size)].
            grid_size (Tensor):
                Spatial-temporal grid dimensions before patching, with shape [3]
                (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            Tensor:
                Reconstructed video tensors with shape [B, C_out, F, H / 8, W / 8]
        """
        B = x.shape[0]
        c = self.out_dim
        grid_size = grid_size.tolist()
        assert x.shape[1] == math.prod(grid_size)
        x = x.view(B, *grid_size, *self.patch_size, c)
        x = torch.einsum('bfhwpqrc->bcfphqwr', x)
        x = x.reshape(B, c, *[i * j for i, j in zip(grid_size, self.patch_size)])
        return x

    def _create_freqs(
        self,
        grid_size: torch.Tensor,
        start_frame: int,
    ):
        device = self.patch_embedding.weight.device
        if any(freq.device != device for freq in self.freqs):
            self.freqs = [freq.to(device) for freq in self.freqs]
        if self.freqs_action.device != device:
            self.freqs_action = self.freqs_action.to(device)
        if self.freqs_state.device != device:
            self.freqs_state = self.freqs_state.to(device)

        f, h, w = grid_size.tolist()
        freqs = torch.cat(
            [
                self.freqs[0][start_frame:start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1
        ).reshape(f * h * w, 1, -1)

        return freqs

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
