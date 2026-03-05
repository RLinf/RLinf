"""
PI0.5 Processor with unified tokenization format.

Unified template: Task: {prompt}. [State: {state}. ][{prefix} ][{response}.][EOS]

Parts are included based on inputs (state can be combined with any mode):
- State: only if discrete_state_input=True and state provided
- Prefix: only if prefix provided (e.g., "Subtask:", "FAST:", "Value:")
- Response: only for training (causal attention, has loss)
- EOS: only when response provided

Examples (state can be added to any):
- Pure VLA:           Task: pick cup.
- VLA + state:        Task: pick cup. State: 128 135.
- Subtask:            Task: pick cup. Subtask: grasp handle.[EOS]
- Subtask + state:    Task: pick cup. State: 128 135. Subtask: grasp handle.[EOS]
- FAST only:          Task: pick cup. FAST: 1 2 3 4 5[EOS]
- FAST + state:       Task: pick cup. State: 128 135. FAST: 1 2 3 4 5[EOS]
- Subtask + FAST:     Task: pick cup. Subtask: grasp. FAST: 1 2 3[EOS]
- Full (with state):  Task: pick cup. State: 128 135. Subtask: grasp. FAST: 1 2 3[EOS]

Masks:
- ar_mask: 0=bidirectional (prefix), 1=causal (response)
- loss_mask: True=include in CE loss (response tokens only)
- kv_cache_mask: True=include in KV cache for action expert (excludes FAST tokens, EOS)
"""

import os
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, BatchFeature
from transformers.processing_utils import ProcessorMixin

from rlinf.utils.dist_utils import get_logger, is_main_process
from rlinf.datasets.vla_lib.io_processing.value_tokens import (
    get_all_value_tokens,
)
from ..openpi.processing_pi0 import PI0ImageProcessor

logger = get_logger(__name__)


class PI05Processor(ProcessorMixin):
    """
    Processor for PI0.5 with unified tokenization.
    
    Handles three modes based on input:
    - VLA: prompt + state + actions (flow matching)
    - VLM: prompt + prefix + response (language generation with CE loss)
    - VLM+VLA: prompt + prefix + response + actions (combined)
    
    Key mask semantics:
    - ar_mask=0: Bidirectional attention (prefix/prompt tokens)
    - ar_mask=1: Causal attention (response/action tokens)
    - loss_mask=True: Include in cross-entropy loss (response tokens only)
    - kv_cache_mask=True: Include in KV cache for action expert (excludes EOS, etc.)
    """
    
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "PI0ImageProcessor"
    tokenizer_class = "AutoTokenizer"
    _tokenize_log_count = 0

    @staticmethod
    def _default_tokenizer_path() -> Optional[str]:
        # Walk up to find the project root (where pyproject.toml lives)
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                candidate = parent / "pretrained_models" / "paligemma-3b-mix-224"
                if candidate.exists():
                    return str(candidate)
                break
        return None

    def __init__(
        self,
        image_processor: Optional[PI0ImageProcessor] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_token_len: int = 200,
        tokenizer_name_or_path: Optional[str] = None,
        discrete_state_input: bool = False,
        image_keys: Optional[tuple] = None,
        exclude_cot_from_kv_cache: bool = False,
        **kwargs
    ):
        if image_processor is None:
            # Use custom image_keys if provided, otherwise use defaults
            image_processor = PI0ImageProcessor(image_keys=image_keys) if image_keys else PI0ImageProcessor()
            
        if tokenizer is None:
            tokenizer_path = (
                tokenizer_name_or_path
                or os.environ.get("VLA_TOKENIZER_PATH")
                or PI05Processor._default_tokenizer_path()  # TODO: zhihao: tokenizer_path最后也得改，不能默认用这个path，得赋值一个
            )
            tokenizer_kwargs = {"add_bos_token": True}
            if tokenizer_path and os.path.exists(tokenizer_path):
                tokenizer_kwargs["local_files_only"] = True
                tokenizer_source = tokenizer_path
            else:
                tokenizer_source = tokenizer_path or "google/paligemma-3b-pt-224"
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, **tokenizer_kwargs)
            
        self.image_processor = image_processor
        self.tokenizer: PreTrainedTokenizerBase = tokenizer
        self.max_token_len = max_token_len
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.discrete_state_input = discrete_state_input
        self.exclude_cot_from_kv_cache = exclude_cot_from_kv_cache
        # Required for save_pretrained compatibility with transformers ProcessorMixin
        self.chat_template = None
        self.audio_tokenizer = None
        # Value token IDs (populated by add_value_tokens)
        self._value_token_ids: Optional[Dict[int, int]] = None

    def add_value_tokens(self, num_bins: int = 201) -> int:
        """Add special tokens for value bins (<v0>, <v1>, ..., <v{num_bins-1}>).
        
        These tokens ensure single-token representation for all bin IDs,
        enabling proper softmax over bins for value prediction.
        
        Args:
            num_bins: Number of value bins (default: 201)
            
        Returns:
            Number of tokens added
        """
        value_tokens = get_all_value_tokens(num_bins)
        
        # Check if tokens already exist
        existing = self.tokenizer.convert_tokens_to_ids(value_tokens[0])
        if existing != self.tokenizer.unk_token_id:
            logger.info(f"Value tokens already exist in tokenizer (first token ID: {existing})")
            self._value_token_ids = {
                i: self.tokenizer.convert_tokens_to_ids(token) 
                for i, token in enumerate(value_tokens)
            }
            return 0
        
        num_added = self.tokenizer.add_special_tokens({"additional_special_tokens": value_tokens})
        logger.info(f"Added {num_added} value tokens to tokenizer")
        
        # Cache the token IDs mapping: bin_id -> token_id
        self._value_token_ids = {
            i: self.tokenizer.convert_tokens_to_ids(token) 
            for i, token in enumerate(value_tokens)
        }
        
        return num_added

    def get_value_token_ids(self) -> Optional[Dict[int, int]]:
        """Get mapping from bin ID to token ID.
        
        Returns:
            Dict mapping bin_id (0-200) to token_id, or None if not initialized
        """
        return self._value_token_ids

    def _clean_text(self, text: str) -> str:
        """Clean text by stripping and normalizing."""
        return text.lower().strip().replace("_", " ").replace("\n", " ")

    def _discretize_state(self, state: np.ndarray) -> str:
        """Discretize state into 256 bins and convert to string."""
        discretized = np.digitize(state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
        return " ".join(map(str, discretized))

    def _strip_trailing_punctuation(self, text: str) -> str:
        """Remove trailing punctuation from text, but preserve quotes.
        
        Quotes are preserved because they're structural in JSON-like responses
        such as ECOT annotations: "object": "[x, y]"
        """
        if text and text[-1] in string.punctuation and text[-1] not in '"\'':
            return text[:-1]
        return text

    def _build_prefix_text(
        self, cleaned_prompt: str, state: Optional[np.ndarray], prefix: Optional[str]
    ) -> str:
        """Build prefix text using unified template.
        
        Template: Task: {prompt}. [State: {state}. ][{prefix} ]
        
        All modes use this template - parts are included based on inputs:
        - State: only if discrete_state_input and state provided
        - Prefix: only if prefix provided (e.g., "Subtask:", "FAST:", "Value:")
        """
        parts = [f"Task: {cleaned_prompt}."]
        
        if self.discrete_state_input and state is not None:
            state_str = self._discretize_state(state)
            parts.append(f"State: {state_str}.")
        
        if prefix:
            parts.append(f"{prefix}")
        
        return " ".join(parts) + " " if prefix else " ".join(parts)

    def _append_tokens(
        self,
        tokens: List[int],
        ar_mask: List[int],
        loss_mask: List[bool],
        kv_cache_mask: List[bool],
        new_tokens: List[int],
        causal: bool = True,
        has_loss: bool = True,
        in_kv_cache: bool = True,
    ) -> None:
        """Append tokens with corresponding masks."""
        tokens.extend(new_tokens)
        ar_mask.extend([1 if causal else 0] * len(new_tokens))
        loss_mask.extend([has_loss] * len(new_tokens))
        kv_cache_mask.extend([in_kv_cache] * len(new_tokens))

    def _tokenize_single(
        self,
        prompt: str,
        prefix: Optional[str] = None,
        response: Optional[str] = None,
        state: Optional[np.ndarray] = None,
        has_actions: bool = False,
        max_length: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Tokenize using unified template.
        
        Template: Task: {prompt}. [State: {state}. ][{prefix} ][{response}][EOS]
        
        Parts included based on inputs:
        - State: only if discrete_state_input and state provided
        - Prefix: only if prefix provided (e.g., "Subtask:", "FAST:")
        - Response: only for training (causal attention, has loss)
        - EOS: only when response provided
        
        Response types:
        - Value token: "<v100>" → single special token, in KV cache
        - FAST only: "1 2 3 4 5" → NOT in KV cache (knowledge insulation)
        - Subtask + FAST: "text. FAST: 1 2 3" → text per exclude_cot flag, FAST excluded
        - Regular: "text" → in KV cache (or excluded if exclude_cot_from_kv_cache=True)
        
        Train-Eval Position ID Alignment:
        To ensure action tokens have consistent position IDs between training and eval,
        prefix labels that are present during training but absent during eval are
        EXCLUDED from the KV cache (kv_cache_mask=False). This applies to:
        
        1. FAST mode (prefix="FAST:"): Always excluded - eval uses sample_actions()
           without FAST tokens, so action expert shouldn't depend on "FAST:" label
        
        2. CoT mode (exclude_cot_from_kv_cache=True): "Subtask:" and reasoning text
           excluded - action expert only attends to core observation (images + prompt)
        
        Note: FAST tokens are ALWAYS excluded from KV cache (knowledge insulation).
        CoT reasoning exclusion is controlled INDEPENDENTLY by exclude_cot_from_kv_cache.
        
        This ensures the same rotary position embeddings during training and eval.
        """
        if max_length is None:
            max_length = self.max_token_len

        cleaned_prompt = self._clean_text(prompt)
        cleaned_prompt = self._strip_trailing_punctuation(cleaned_prompt)
        
        tokens: List[int] = []
        ar_mask: List[int] = []
        loss_mask: List[bool] = []
        kv_cache_mask: List[bool] = []

        # Determine if we should exclude the prefix label from action expert's KV cache.
        # This is needed for train-eval alignment when the prefix label is present during
        # training but absent during eval:
        # 1. FAST mode (prefix="FAST:"): Always excluded - eval uses sample_actions()
        # 2. CoT mode: Controlled by exclude_cot_from_kv_cache flag
        # Note: CoT reasoning exclusion is INDEPENDENT of FAST. When CoT + FAST:
        # - FAST tokens: always excluded (knowledge insulation)
        # - CoT reasoning: controlled by exclude_cot_from_kv_cache
        is_fast_prefix = prefix and prefix.strip().upper() == "FAST:"
        should_exclude_prefix_label = (self.exclude_cot_from_kv_cache or is_fast_prefix) and prefix
        
        if should_exclude_prefix_label:
            # Build core prefix WITHOUT the prefix label (e.g., without "FAST:" or "Subtask:")
            core_prefix_text = self._build_prefix_text(cleaned_prompt, state, None)
            core_tokens = self.tokenizer.encode(core_prefix_text, add_special_tokens=True)
            self._append_tokens(tokens, ar_mask, loss_mask, kv_cache_mask, core_tokens,
                                causal=False, has_loss=False, in_kv_cache=True)
            
            # Tokenize prefix label separately with in_kv_cache=False
            # This ensures action expert doesn't attend to tokens absent during eval
            prefix_label_text = f"{prefix} "
            prefix_label_tokens = self.tokenizer.encode(prefix_label_text, add_special_tokens=False)
            self._append_tokens(tokens, ar_mask, loss_mask, kv_cache_mask, prefix_label_tokens,
                                causal=False, has_loss=False, in_kv_cache=False)
        else:
            # Original behavior: build and tokenize full prefix together
            prefix_text = self._build_prefix_text(cleaned_prompt, state, prefix)
            prefix_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=True)
            self._append_tokens(tokens, ar_mask, loss_mask, kv_cache_mask, prefix_tokens,
                                causal=False, has_loss=False, in_kv_cache=True)
        
        # Add response if provided (training mode)
        if response is not None:
            self._tokenize_response(
                response, has_actions, prefix or "",
                tokens, ar_mask, loss_mask, kv_cache_mask
            )
            # Append EOS (causal, NOT in KV cache)
            # EOS loss depends on response type:
            # - FAST-only without actions: no CE loss (has_loss=False)
            # - All other cases: CE loss on EOS (has_loss=True)
            is_fast_only = prefix and prefix.strip().upper() == "FAST:"
            eos_has_loss = not (is_fast_only and not has_actions)
            if self.tokenizer.eos_token_id is not None:
                self._append_tokens(tokens, ar_mask, loss_mask, kv_cache_mask,
                                    [self.tokenizer.eos_token_id],
                                    causal=True, has_loss=eos_has_loss, in_kv_cache=False)

        # Debug logging
        self._maybe_log_example(
            prompt,
            prefix,
            response,
            state,
            has_actions,
            tokens,
            ar_mask,
            loss_mask,
            kv_cache_mask,
        )
        
        return self._pad_to_length(tokens, ar_mask, loss_mask, kv_cache_mask, max_length)

    def _tokenize_response(
        self,
        response: str,
        has_actions: bool,
        prefix: str,
        tokens: List[int],
        ar_mask: List[int],
        loss_mask: List[bool],
        kv_cache_mask: List[bool],
    ) -> None:
        """Tokenize the response part with appropriate masks.
        
        Response types (unified - all end with "."):
        - Value token: "<v100>" → single special token
        - FAST only (prefix="FAST:"): entire response is FAST tokens
        - Subtask + FAST: "{reasoning}. FAST: {tokens}"
        - Regular: "{response}."
        
        KV cache exclusion rules:
        - FAST tokens: Always excluded (knowledge insulation)
        - Reasoning: Excluded only if exclude_cot_from_kv_cache=True (independent of FAST)
        - Loss mask unchanged: CE loss always supervises reasoning
        """
        # Check response type
        is_value_token = (
            response.startswith("<v") and response.endswith(">") and response[2:-1].isdigit()
        )
        is_fast_only = prefix.strip().upper() == "FAST:"
        has_fast_in_response = "FAST:" in response
        
        # Determine if reasoning should be excluded from action expert's KV cache.
        # This applies to all reasoning tokens (non-FAST, non-value).
        # Note: This is INDEPENDENT of FAST tokens. FAST tokens are always excluded,
        # but CoT reasoning exclusion is controlled by exclude_cot_from_kv_cache.
        exclude_reasoning = self.exclude_cot_from_kv_cache
        
        # Value token: special single token (always in KV cache - it's the target, not reasoning)
        if is_value_token:
            token_id = self.tokenizer.convert_tokens_to_ids(response)
            if token_id == self.tokenizer.unk_token_id:
                if not getattr(self, '_warned_value_token', False):
                    logger.info(f"Value tokens not in tokenizer, encoding as text (this is expected for expert-only mode)")
                    self._warned_value_token = True
                resp_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            else:
                resp_tokens = [token_id]
            self._append_tokens(tokens, ar_mask, loss_mask, kv_cache_mask, resp_tokens,
                                causal=True, has_loss=True, in_kv_cache=True)
        
        # FAST only: entire response is FAST tokens (NOT in KV cache)
        elif is_fast_only:
            fast_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            # When has_actions=False (no action supervision), FAST tokens should
            # not contribute to language CE loss either. We still keep them as
            # causal tokens but set has_loss based on has_actions.
            self._append_tokens(
                tokens, ar_mask, loss_mask, kv_cache_mask, fast_tokens,
                causal=True, has_loss=has_actions, in_kv_cache=False
            )
        
        # Subtask + FAST: split into reasoning and FAST
        elif has_fast_in_response:
            fast_pos = response.find("FAST:")
            pre_fast = response[:fast_pos].strip()
            fast_part = response[fast_pos:]  # "FAST: {tokens}"
            
            # Reasoning part
            # When exclude_cot_from_kv_cache=True: NOT in KV cache (action expert doesn't see it)
            # Otherwise: in KV cache (original behavior)
            if pre_fast:
                clean_pre = self._strip_trailing_punctuation(self._clean_text(pre_fast))
                pre_tokens = self.tokenizer.encode(f"{clean_pre}. ", add_special_tokens=False)
                # Reasoning text is always supervised by CE loss.
                self._append_tokens(
                    tokens, ar_mask, loss_mask, kv_cache_mask, pre_tokens,
                    causal=True, has_loss=True, in_kv_cache=not exclude_reasoning
                )
            
            # FAST part (NOT in KV cache - knowledge insulation)
            fast_tokens = self.tokenizer.encode(fast_part, add_special_tokens=False)
            # For FAST tokens, tie CE supervision to presence of actions: if
            # we are not training flow-matching on this sample (no actions),
            # we also do not want FAST tokens to influence language CE loss.
            self._append_tokens(
                tokens, ar_mask, loss_mask, kv_cache_mask, fast_tokens,
                causal=True, has_loss=has_actions, in_kv_cache=False
            )
        
        # Regular response (unified: always ends with ".")
        # When exclude_cot_from_kv_cache=True: NOT in KV cache
        else:
            cleaned = self._strip_trailing_punctuation(self._clean_text(response))
            resp_tokens = self.tokenizer.encode(f"{cleaned}.", add_special_tokens=False)
            self._append_tokens(tokens, ar_mask, loss_mask, kv_cache_mask, resp_tokens,
                                causal=True, has_loss=True, in_kv_cache=not exclude_reasoning)

    def _maybe_log_example(
        self,
        prompt: str,
        prefix: Optional[str],
        response: Optional[str],
        state: Optional[np.ndarray],
        has_actions: bool,
        tokens: List[int],
        ar_mask: List[int],
        loss_mask: List[bool],
        kv_cache_mask: List[bool],
    ) -> None:
        """Log tokenization example for debugging (first 2 examples only)."""
        worker_info = torch.utils.data.get_worker_info()
        is_worker_0 = worker_info is None or worker_info.id == 0
        if is_worker_0 and is_main_process() and PI05Processor._tokenize_log_count < 2:
            PI05Processor._tokenize_log_count += 1
            self._log_tokenization_example(
                prompt=prompt,
                prefix=prefix,
                response=response,
                state=state,
                has_actions=has_actions,
                tokens=tokens,
                ar_mask=ar_mask,
                loss_mask=loss_mask,
                kv_cache_mask=kv_cache_mask,
            )

    def _pad_to_length(
        self, 
        tokens: List[int], 
        ar_mask: List[int], 
        loss_mask: List[bool],
        kv_cache_mask: List[bool],
        max_length: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pad sequences to max_length and return as numpy arrays."""
        tokens_len = len(tokens)
        
        if tokens_len < max_length:
            padding_len = max_length - tokens_len
            mask = [True] * tokens_len + [False] * padding_len
            tokens = tokens + [0] * padding_len
            ar_mask = ar_mask + [0] * padding_len
            loss_mask = loss_mask + [False] * padding_len
            kv_cache_mask = kv_cache_mask + [False] * padding_len
        else:
            if tokens_len > max_length:
                logger.warning(f"Token length ({tokens_len}) exceeds max ({max_length}), truncating.")
            tokens = tokens[:max_length]
            mask = [True] * max_length
            ar_mask = ar_mask[:max_length]
            loss_mask = loss_mask[:max_length]
            kv_cache_mask = kv_cache_mask[:max_length]

        return (
            np.asarray(tokens), np.asarray(mask), np.asarray(ar_mask),
            np.asarray(loss_mask), np.asarray(kv_cache_mask)
        )
    
    def _log_tokenization_example(
        self,
        prompt: str,
        prefix: Optional[str],
        response: Optional[str],
        state: Optional[np.ndarray],
        has_actions: bool,
        tokens: List[int],
        ar_mask: List[int],
        loss_mask: List[bool],
        kv_cache_mask: List[bool],
    ) -> None:
        """Log a tokenization example for debugging."""
        decoded = self.tokenizer.decode(tokens, skip_special_tokens=False)
        
        # Find positions where loss_mask is True
        loss_positions = [i for i, m in enumerate(loss_mask) if m]
        loss_tokens = [tokens[i] for i in loss_positions] if loss_positions else []
        loss_decoded = self.tokenizer.decode(loss_tokens, skip_special_tokens=False) if loss_tokens else ""
        
        # Find positions where ar_mask is 1 (causal)
        causal_positions = [i for i, m in enumerate(ar_mask) if m == 1]

        # Find positions where kv_cache_mask is True
        kv_true_positions = [i for i, m in enumerate(kv_cache_mask) if m]

        if state is not None:
            try:
                state_repr = state.tolist()
            except Exception:
                state_repr = str(state)
        else:
            state_repr = None

        logger.info("[Tokenization Example #%d]", self._tokenize_log_count)
        logger.info(
            "  Inputs: prompt=%r, prefix=%r, response=%r, state=%r, has_actions=%s",
            prompt,
            prefix,
            response,
            state_repr,
            has_actions,
        )
        logger.info("  token_ids (%d): %s", len(tokens), tokens)
        logger.info("  Tokens decoded: %r", decoded)
        logger.info(
            "  ar_mask (len=%d): %s  |  bidirectional=%d, causal=%d",
            len(ar_mask),
            ar_mask,
            len(tokens) - len(causal_positions),
            len(causal_positions),
        )
        logger.info(
            "  loss_mask (len=%d): %s  |  True positions=%s, tokens_decoded=%r",
            len(loss_mask),
            loss_mask,
            loss_positions,
            loss_decoded,
        )
        logger.info(
            "  kv_cache_mask (len=%d): %s  |  True positions=%s",
            len(kv_cache_mask),
            kv_cache_mask,
            kv_true_positions,
        )

    def process_text(
        self,
        prompts: List[str],
        prefixes: List[Optional[str]],
        responses: List[Optional[str]],
        states: List[Optional[np.ndarray]],
        actions: List[Optional[Any]],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: Optional[str] = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Process a batch of text inputs for PI0.5.
        
        Args:
            prompts: List of task descriptions
            prefixes: List of optional prefixes (paired with responses)
            responses: List of optional responses (paired with prefixes)
            states: List of optional state arrays
            actions: List of optional action tensors (used to determine has_actions)
            padding: Whether to pad sequences
            truncation: Whether to truncate sequences
            max_length: Maximum sequence length
            return_tensors: Output format ("pt" for PyTorch)
            
        Returns:
            Dict with input_ids, attention_mask, token_ar_mask, token_loss_mask
        """
        if max_length is None:
            max_length = self.max_token_len

        batch_tokens = []
        batch_masks = []
        batch_ar_masks = []
        batch_loss_masks = []
        batch_kv_cache_masks = []

        for i, prompt in enumerate(prompts):
            prefix = prefixes[i] if i < len(prefixes) else None
            response = responses[i] if i < len(responses) else None
            state = states[i] if i < len(states) else None
            has_actions = actions[i] is not None if i < len(actions) else False
            
            tokens, mask, ar_mask, loss_mask, kv_cache_mask = self._tokenize_single(
                prompt=prompt,
                prefix=prefix,
                response=response,
                state=state,
                has_actions=has_actions,
                max_length=max_length,
            )
            batch_tokens.append(tokens)
            batch_masks.append(mask)
            batch_ar_masks.append(ar_mask)
            batch_loss_masks.append(loss_mask)
            batch_kv_cache_masks.append(kv_cache_mask)

        result = {
            "input_ids": np.stack(batch_tokens),
            "attention_mask": np.stack(batch_masks),
            "token_ar_mask": np.stack(batch_ar_masks),
            "token_loss_mask": np.stack(batch_loss_masks),
            "token_kv_cache_mask": np.stack(batch_kv_cache_masks),
        }
        
        if return_tensors == "pt":
            result = {k: torch.tensor(v) for k, v in result.items()}
        
        return result

    def __call__(
        self,
        text: Union[str, List[str]] = None,
        images: Union[Dict[str, torch.Tensor], List[torch.Tensor], torch.Tensor] = None,
        image_masks: Optional[Dict[str, torch.Tensor]] = None,
        return_tensors: Optional[str] = 'pt',
        train: bool = False,
        state: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        prefix: Optional[Union[str, List[str]]] = None,
        response: Optional[Union[str, List[str]]] = None,
        actions: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        **kwargs
    ) -> BatchFeature:
        """
        Process text and images for PI0.5 model.
        
        Args:
            text: Input text (prompt)
            images: Image dict with camera keys
            image_masks: Optional image masks
            return_tensors: Output tensor format
            train: Whether in training mode
            state: Robot state (will be discretized if discrete_state_input=True)
            prefix: Optional prefix for VLM mode
            response: Optional response for VLM mode
            actions: Optional actions for VLA mode
        """
        if text is None and images is None:
            raise ValueError("You must provide either text or images")
        
        result_data = {}
        
        if text is not None:
            is_batched = isinstance(text, list)
            texts = text if is_batched else [text]
            batch_size = len(texts)
            
            # Normalize inputs to lists
            states = state if isinstance(state, list) else [state] * batch_size
            prefixes = prefix if isinstance(prefix, list) else [prefix] * batch_size
            responses = response if isinstance(response, list) else [response] * batch_size
            actions_list = actions if isinstance(actions, list) else [actions] * batch_size
            
            processed = self.process_text(
                prompts=texts,
                prefixes=prefixes,
                responses=responses,
                states=states,
                actions=actions_list,
                return_tensors=return_tensors,
            )
            result_data.update(processed)
            
            if not is_batched:
                for key in result_data:
                    if result_data[key].dim() > 0:
                        result_data[key] = result_data[key][0]
        
        if images is not None:
            image_inputs = self.image_processor(
                images, 
                image_masks=image_masks,
                return_tensors=return_tensors,
                train=train
            )
            result_data.update(image_inputs)
        
        return BatchFeature(data=result_data, tensor_type=return_tensors)

    def decode(self, token_ids: Union[List[int], torch.Tensor], **kwargs) -> str:
        """Decode tokens to text."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        token_ids = [t for t in token_ids if t != 0]
        return self.tokenizer.decode(token_ids, **kwargs)
    
    def batch_decode(self, token_ids_batch: Union[List[List[int]], torch.Tensor], **kwargs) -> List[str]:
        """Decode batch of tokens to text."""
        if isinstance(token_ids_batch, torch.Tensor):
            token_ids_batch = token_ids_batch.tolist()
        return [self.decode(tokens, **kwargs) for tokens in token_ids_batch]

    @property
    def model_input_names(self):
        return [
            "pixel_values",
            "image_masks",
            "input_ids",
            "attention_mask",
            "token_ar_mask",
            "token_loss_mask",
            "token_kv_cache_mask",
        ]


__all__ = ["PI05Processor", "PI0ImageProcessor"]
