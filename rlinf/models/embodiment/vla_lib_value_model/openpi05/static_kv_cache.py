"""Static KV Cache implementation for pre-allocated memory with indexed updates.

This module provides a StaticKVCache class that matches the JAX implementation's
behavior of pre-allocating cache memory and using indexed updates rather than
concatenation. Provides a Cache-compatible interface without inheriting from
transformers.Cache to avoid property conflicts in newer transformers versions.
"""

from typing import List, Optional, Tuple
import torch


class StaticKVCache:
    """
    Static KV cache with pre-allocated memory and indexed updates.
    
    This cache class pre-allocates memory for the maximum sequence length
    and uses indexed updates to insert new key-value pairs, matching the
    JAX implementation's behavior with jax.lax.dynamic_update_slice.
    
    JAX Reference Behavior (from gemma.py):
    1. _init_cache: Pre-allocates cache by padding k,v to cache_size (attn_mask.shape[-1])
       - Cache shape: [batch, prefill_len + max_decoding_steps, num_heads, head_dim]
       - Initial position idx = prefill_len (after prefill)
    
    2. _update_cache: Updates cache at specific position using dynamic_update_slice
       - Only for single-token updates (k.shape[1] == 1)
       - Writes at position idx, then increments: idx_new = idx + 1
    
    3. During action sampling: Concatenates cached k,v with new k,v WITHOUT updating cache
       - k = concatenate([k_cache, k], axis=1) when k.shape[1] > 1
    
    PyTorch Implementation:
    - Pre-allocates to max_cache_len during initialization (matching step 1)
    - Uses index_copy_ for updates instead of concatenation (matching step 2)
    - When use_cache=False in attention, uses __getitem__ + concatenation (matching step 3)
    """

    def __init__(
        self,
        max_batch_size: int = 1,
        max_cache_len: int = 4096,
        num_layers: int = 18,
        num_key_value_heads: int = 1,
        head_dim: int = 256,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        """Initialize pre-allocated static KV cache.
        
        Args:
            max_batch_size: Maximum batch size
            max_cache_len: Maximum sequence length (prefill_size + max_decoding_steps)
            num_layers: Number of transformer layers
            num_key_value_heads: Number of key-value attention heads
            head_dim: Dimension of each attention head
            dtype: Data type for cache tensors
            device: Device to allocate tensors on
        """
        # Don't call super().__init__() - newer transformers Cache has properties
        # that conflict with our direct attribute assignment
        self._max_batch_size = max_batch_size
        self._max_cache_len = max_cache_len
        self.num_layers = num_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # Pre-allocate cache for all layers
        # Shape: [batch, num_heads, max_seq_len, head_dim]
        cache_shape = (max_batch_size, num_key_value_heads, max_cache_len, head_dim)
        
        # Initialize empty cache for each layer
        # Use _key_cache/_value_cache to avoid conflict with parent class properties
        self._key_cache: List[torch.Tensor] = []
        self._value_cache: List[torch.Tensor] = []
        for _ in range(num_layers):
            self._key_cache.append(torch.zeros(cache_shape, dtype=dtype, device=device))
            self._value_cache.append(torch.zeros(cache_shape, dtype=dtype, device=device))
        
        # Track current position in cache for each batch element
        self.cache_position = torch.zeros((max_batch_size,), dtype=torch.long, device=device)
        self._seen_tokens = 0
    
    @property
    def key_cache(self) -> List[torch.Tensor]:
        """Return key cache (for compatibility with transformers Cache interface)."""
        return self._key_cache
    
    @property
    def value_cache(self) -> List[torch.Tensor]:
        """Return value cache (for compatibility with transformers Cache interface)."""
        return self._value_cache

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states using indexed update.
        
        Args:
            key_states: New key states [batch, num_heads, seq_len, head_dim]
            value_states: New value states [batch, num_heads, seq_len, head_dim]
            layer_idx: Index of the layer being updated
            cache_kwargs: Additional kwargs (contains cache_position from transformers)
        
        Returns:
            Tuple of (updated_keys, updated_values) containing full cache up to current position
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        if cache_kwargs is not None and "cache_position" in cache_kwargs:
            cache_position = cache_kwargs["cache_position"]
            if isinstance(cache_position, torch.Tensor):
                write_positions = cache_position
            else:
                write_positions = torch.arange(
                    cache_position, cache_position + seq_len, 
                    device=key_states.device, dtype=torch.long
                )
        else:
            write_pos = self.cache_position[0].item()
            write_positions = torch.arange(
                write_pos, write_pos + seq_len, 
                device=key_states.device, dtype=torch.long
            )
        
        if write_positions.numel() == 1:
            start_pos = write_positions.item()
            end_pos = start_pos + seq_len
            write_positions = torch.arange(start_pos, end_pos, device=key_states.device, dtype=torch.long)
        else:
            start_pos = write_positions[0].item()
            end_pos = write_positions[-1].item() + 1
        
        if end_pos > self._max_cache_len:
            raise RuntimeError(
                f"Cache position {end_pos} exceeds max cache length {self._max_cache_len}"
            )
        
        self.key_cache[layer_idx].index_copy_(2, write_positions, key_states)
        self.value_cache[layer_idx].index_copy_(2, write_positions, value_states)
        
        self.cache_position[0] = end_pos
        self._seen_tokens = end_pos
        end_pos_int = int(end_pos)
        
        return (
            self.key_cache[layer_idx][:, :, :end_pos_int, :].clone(),
            self.value_cache[layer_idx][:, :, :end_pos_int, :].clone(),
        )

    def get_seq_length(self, layer_idx: Optional[int] = None) -> int:
        """Return current sequence length in cache."""
        return int(self._seen_tokens)

    def get_max_length(self) -> int:
        """Return maximum cache length."""
        return self._max_cache_len

    def get_usable_length(self, new_seq_length: int, layer_idx: Optional[int] = None) -> int:
        """Return usable length considering new sequence length."""
        return int(min(self._seen_tokens, self._max_cache_len - new_seq_length))

    def reset(self):
        """Reset cache position counter without deallocating memory."""
        self.cache_position.zero_()
        self._seen_tokens = 0

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cached key-value pairs for a specific layer."""
        current_len = int(self._seen_tokens)
        return (
            self.key_cache[layer_idx][:, :, :current_len, :].clone(),
            self.value_cache[layer_idx][:, :, :current_len, :].clone(),
        )

    def __len__(self) -> int:
        """Return number of layers."""
        return self.num_layers


def left_to_right_align(x: torch.Tensor, input_mask: torch.Tensor, attn_mask: torch.Tensor):
    """
    Convert input from left-aligned to right-aligned for efficient autoregressive generation.
    
    Each example in the batch can have a different sequence length and will
    be rolled by a different amount to achieve right-alignment.
    
    Args:
        x: [batch, seq, dim] embeddings
        input_mask: [batch, seq] bool mask - True for valid tokens
        attn_mask: [batch, seq, seq] attention mask
    
    Returns:
        Tuple of (x_rolled, mask_rolled, attn_rolled) with same shapes as input
    """
    batch_size, seq_len = input_mask.shape
    device = x.device
    
    arange = torch.arange(seq_len, device=device).unsqueeze(0)
    seqlens = torch.max(input_mask.float() * arange, dim=1)[0] + 1
    seqlens = seqlens.long()
    
    seq_indices = torch.arange(seq_len, device=device).unsqueeze(0)
    rolled_seq_indices = (seq_indices + seqlens.unsqueeze(1)) % seq_len
    
    batch_indices = torch.arange(batch_size, device=device).unsqueeze(1)
    
    x_rolled = x[batch_indices, rolled_seq_indices]
    mask_rolled = input_mask[batch_indices, rolled_seq_indices]
    
    row_indices = rolled_seq_indices.unsqueeze(2)
    col_indices = rolled_seq_indices.unsqueeze(1)
    batch_indices_3d = batch_indices.unsqueeze(2)
    
    attn_rolled = attn_mask[batch_indices_3d, row_indices, col_indices]
    
    return x_rolled, mask_rolled, attn_rolled


__all__ = ["StaticKVCache", "left_to_right_align"]

