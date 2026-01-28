# Copyright 2025 The RLinf Authors.
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

from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import (
    OpenVLAForActionPrediction as OpenVLAOFTForActionPrediction,
)
from prismatic.models.projectors import ProprioProjector
from prismatic.vla.constants import (
    IGNORE_INDEX,
)
from torch.nn.utils.rnn import pad_sequence
from transformers.generation import TopKLogitsWarper

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.models.embodiment.openvla_oft.openvla_utils import (
    find_checkpoint_file,
    load_component_state_dict,
    normalize_proprio,
)
from rlinf.utils.torch_functionals import pad_tensor_to_length
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)


class OpenVLAOFTRLConfig(OpenVLAConfig):
    def __init__(
        self,
        action_dim: int = 7,
        num_action_chunks: int = 8,
        add_value_head: bool = False,
        value_type: str = "chunk_level",
        proprio_dim: int = 7,
        use_proprio: bool = False,
        use_film: bool = False,
        max_prompt_length: int = 128,
        **kwargs,
    ) -> None:
        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.add_value_head = add_value_head
        self.value_type = value_type
        self.proprio_dim = proprio_dim
        self.use_proprio = use_proprio
        self.use_film = use_film
        self.max_prompt_length = max_prompt_length

        super().__init__(**kwargs)


class OpenVLAOFTForRLActionPrediction(OpenVLAOFTForActionPrediction, BasePolicy):
    config_class = OpenVLAOFTRLConfig

    def __init__(
        self,
        config: OpenVLAOFTRLConfig,
    ) -> None:
        super().__init__(config)

        self.action_dim = config.action_dim
        self.num_action_chunks = config.num_action_chunks
        self.use_proprio = config.use_proprio
        self.proprio_projector = None
        if self.use_proprio:
            self.proprio_projector = ProprioProjector(
                llm_dim=config.text_config.hidden_size, proprio_dim=config.proprio_dim
            )
        self.use_film = config.use_film

        self.unnorm_key = config.unnorm_key

        if config.add_value_head:
            self.hidden_size = self.config.text_config.hidden_size
            output_dim = (
                1 if self.config.value_type == "chunk_level" else self.num_action_chunks
            )
            self.value_head = ValueHead(
                input_dim=self.hidden_size,
                hidden_sizes=(512, 128),
                output_dim=output_dim,
                activation="gelu",
                bias_last=False,
            )

        self.max_prompt_length = config.max_prompt_length

    def set_processor(self, processor):
        self.processor = processor

    def load_proprio_projector_weights(self, checkpoint_path_or_repo_id: str):
        """
        Load pre-trained weights for the proprio projector.

        Args:
            checkpoint_path_or_repo_id: Either a local path to checkpoint file or HF Hub repo ID
        """
        if self.proprio_projector is None:
            raise ValueError("Model was not initialized with use_proprio=True")

        checkpoint_path = find_checkpoint_file(
            checkpoint_path_or_repo_id, "proprio_projector"
        )
        state_dict = load_component_state_dict(checkpoint_path)
        self.proprio_projector.load_state_dict(state_dict)

    def prepare_inputs(self, env_obs):
        task_descriptions = [
            f"In: What action should the robot take to {t.lower()}?\nOut: "
            for t in env_obs["task_descriptions"]
        ]

        batchdata = {"input_ids": [], "attention_mask": [], "pixel_values": []}
        if self.config.use_proprio:
            batchdata["proprio"] = []

        for i in range(len(task_descriptions)):
            image = np.array(env_obs["main_images"][i])
            image = Image.fromarray(image).convert("RGB")
            prompt = task_descriptions[i]
            batch_feature = self.processor(prompt, image)

            pixel_values_list = [batch_feature["pixel_values"]]
            if self.vision_backbone.get_num_images_in_input() > 1:
                wrist_images = np.array(env_obs["wrist_images"][i])
                for idx in range(wrist_images.shape[0]):
                    wrist_image = Image.fromarray(wrist_images[idx]).convert("RGB")
                    wrist_batch_feature = self.processor(prompt, wrist_image)
                    pixel_values_list.append(wrist_batch_feature["pixel_values"])

            batch_feature["pixel_values"] = torch.cat(pixel_values_list, dim=1)

            input_ids = batch_feature["input_ids"]
            attention_mask = batch_feature["attention_mask"]
            pixel_values = batch_feature["pixel_values"]

            if not torch.all(input_ids[:, -1] == 29871):
                input_ids = torch.cat(
                    (
                        input_ids,
                        torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(
                            input_ids.device
                        ),
                    ),
                    dim=1,
                )
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        torch.unsqueeze(torch.Tensor([True]).bool(), dim=0).to(
                            attention_mask.device
                        ),
                    ),
                    dim=1,
                )

            batchdata["input_ids"].append(input_ids)
            batchdata["attention_mask"].append(attention_mask)
            batchdata["pixel_values"].append(pixel_values)

            # Process proprioception
            if self.use_proprio:
                state = env_obs["states"][i]
                proprio_norm_stats = self.norm_stats[self.unnorm_key]["proprio"]
                norm_proprio = normalize_proprio(state, proprio_norm_stats)
                batchdata["proprio"].append(torch.from_numpy(norm_proprio))

        device = next(self.parameters()).device
        precision = next(self.parameters()).dtype

        batchdata["input_ids"] = [x.transpose(0, 1) for x in batchdata["input_ids"]]
        batchdata["attention_mask"] = [
            x.transpose(0, 1) for x in batchdata["attention_mask"]
        ]
        batchdata["input_ids"] = (
            pad_sequence(
                batchdata["input_ids"],
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            )
            .squeeze(-1)
            .to(device)
        )
        batchdata["attention_mask"] = (
            pad_sequence(batchdata["attention_mask"], batch_first=True, padding_value=0)
            .squeeze(-1)
            .to(device)
        )

        padding_mask = batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
        assert torch.all(padding_mask == batchdata["attention_mask"].ne(0))
        padding_mask = ~padding_mask
        padding_mask = padding_mask.int()
        sorted_indices = torch.argsort(
            padding_mask, dim=1, descending=True, stable=True
        )
        batchdata["input_ids"] = torch.gather(batchdata["input_ids"], 1, sorted_indices)
        batchdata["attention_mask"] = torch.gather(
            batchdata["attention_mask"], 1, sorted_indices
        )

        batchdata["pixel_values"] = (
            torch.cat(batchdata["pixel_values"], dim=0).to(device).to(precision)
        )

        if self.use_proprio:
            batchdata["proprio"] = torch.stack(batchdata["proprio"], dim=0).to(device)

        assert torch.all(
            batchdata["attention_mask"].ne(0)
            == batchdata["input_ids"].ne(self.processor.tokenizer.pad_token_id)
        )

        input_ids = batchdata["input_ids"]
        attention_mask = batchdata["attention_mask"]
        pixel_values = batchdata["pixel_values"]
        proprio = batchdata.get("proprio", None)

        input_ids = pad_tensor_to_length(
            input_ids,
            max_seq_len=self.max_prompt_length,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            left_pad=True,
        )
        attention_mask = pad_tensor_to_length(
            attention_mask,
            max_seq_len=self.max_prompt_length,
            pad_token_id=0,
            left_pad=True,
        )
        assert torch.all(
            attention_mask.ne(0) == input_ids.ne(self.processor.tokenizer.pad_token_id)
        ), (
            "Invalid padding alignment after argsort/gather reordering: "
            "attention_mask and input_ids are no longer consistent."
        )

        return input_ids, attention_mask, pixel_values, proprio

    def _compute_logprobs_and_entropy(
        self,
        action_logits: torch.Tensor,
        action_tokens,
        compute_entropy: bool = False,
    ):
        """
        Compute logprobs and entropy from action logits and action tokens.

        Args:
            action_logits: Action logits, shape [B, action_dim * num_action_chunks, 256]
            action_tokens: Action token indices, relative indices [0, 256)
            compute_entropy: Whether to compute entropy

        Returns:
            logprobs: Computed logprobs, shape [B, action_dim * num_action_chunks]
            entropy: Computed entropy (optional), shape [B, action_dim * num_action_chunks]
        """

        logprobs = compute_logprobs_from_logits(
            logits=action_logits, target=action_tokens
        )

        result = {"logprobs": logprobs}

        if compute_entropy:
            entropy = compute_entropy_from_logits(logits=action_logits)
            result["entropy"] = entropy

        return result

    def _compute_values(
        self,
        last_hidden_states: torch.Tensor,
    ):
        """
        Compute values from action tokens' hidden states.

        Args:
            last_hidden_states: Last hidden states for action tokens, shape [B, seq_len, hidden_size]

        Returns:
            values: Computed values, shape depends on value_type:
                - chunk_level: [batch_size, 1]
                - action_level: [batch_size, num_action_chunks]
            Returns None if value_head is not available
        """
        if not hasattr(self, "value_head"):
            return None

        hidden_features = last_hidden_states[
            :, -self.action_dim * self.num_action_chunks - 1
        ]  # [batch_size, hidden_dim]
        values = self.value_head(hidden_features)  # [batch_size, 1]

        return values

    def _discrete_prediction(
        self,
        multimodal_embeddings,
        multimodal_attention_mask,
        multimodal_position_ids,
        num_patches,
        num_prompt_tokens,
        do_sample=True,
        temperature=1,
        top_k=-1,
    ):
        """Run discrete action tokens prediction."""

        # Forward pass through language model
        language_model_output = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract hidden states for action tokens
        last_hidden_states = language_model_output.hidden_states[
            -1
        ]  # (B, seq_len, hidden_size)

        # Prepare seq_indices for extracting action tokens (both logits and hidden states)
        batch_size = language_model_output.logits.shape[0]
        device = language_model_output.logits.device
        num_prompt_tokens_total = num_patches + num_prompt_tokens
        start_indices = num_prompt_tokens_total.unsqueeze(1)
        position_offsets = torch.arange(
            self.action_dim * self.num_action_chunks, device=device
        ).unsqueeze(0)
        seq_indices = (
            start_indices + position_offsets
        )  # Shape: [batch_size, action_dim * num_action_chunks]

        # Extract logits using seq_indices
        reponse_ids_logits = language_model_output.logits[
            torch.arange(batch_size, device=device).unsqueeze(-1), seq_indices, :
        ]  # Shape: [batch_size, action_dim * num_action_chunks, vocab_size]
        action_logits = reponse_ids_logits[
            ...,
            -self.config.n_action_bins
            - self.config.pad_to_multiple_of : -self.config.pad_to_multiple_of,
        ]  # Shape: [batch_size, action_dim * num_action_chunks, 256]

        if not do_sample:
            reponse_ids = action_logits.argmax(dim=-1)
            action_tokens = reponse_ids
        else:
            assert temperature > 0

            action_logits = action_logits / temperature
            top_k = min(top_k, action_logits.size(-1))  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                action_logits = logits_warper(None, action_logits)

            probs = torch.softmax(action_logits, dim=-1)

            probs_flat = probs.reshape(-1, probs.shape[-1])
            reponse_ids = torch.multinomial(probs_flat, num_samples=1, replacement=True)

            reponse_ids = reponse_ids.view(
                action_logits.shape[0], action_logits.shape[1]
            )
            action_tokens = reponse_ids

        final_reponse_ids = reponse_ids + (self.vocab_size - self.config.n_action_bins)

        assert torch.all(
            final_reponse_ids >= self.vocab_size - self.config.n_action_bins
        ) and torch.all(final_reponse_ids < self.vocab_size)

        predicted_action_token_ids = final_reponse_ids.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = self.bin_centers[discretized_actions]  # [B, dim]
        normalized_actions = normalized_actions.reshape(-1, self.action_dim)

        return (
            normalized_actions,
            action_logits,
            action_tokens,
            last_hidden_states,
        )

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        proprio=None,
        env_obs=None,
        calulate_logprobs=True,
        calulate_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")

        if env_obs is not None:
            input_ids, attention_mask, pixel_values, proprio = self.prepare_inputs(
                env_obs
            )

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "proprio": proprio,
        }

        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # Get number of tokens in prompt (excluding the start token)
        num_prompt_tokens = (
            input_ids.ne(self.processor.tokenizer.pad_token_id).sum(dim=1) - 1
        )

        multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = (
            self._build_embedding(
                input_ids, attention_mask, pixel_values, labels, proprio
            )
        )

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        num_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )
        use_proprio = self.proprio_projector is not None and proprio is not None
        if use_proprio:
            num_patches += 1

        normalized_actions, action_logits, action_tokens, last_hidden_states = (
            self._discrete_prediction(
                multimodal_embeddings,
                multimodal_attention_mask,
                multimodal_position_ids,
                num_patches,
                num_prompt_tokens,
                do_sample=do_sample,
                temperature=kwargs["temperature"],
                top_k=kwargs["top_k"],
            )
        )

        # Unnormalize predicted actions
        actions = self._unnormalize_actions(normalized_actions, self.unnorm_key)
        actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)

        # Compute chunk logprobs (and critic values)
        logprobs_result = self._compute_logprobs_and_entropy(
            action_logits=action_logits,
            action_tokens=action_tokens,
            compute_entropy=False,
        )
        chunk_logprobs = logprobs_result["logprobs"]

        chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)

        chunk_values = None
        if calulate_values:
            chunk_values = self._compute_values(last_hidden_states)
        if chunk_values is None:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        forward_inputs["action_tokens"] = action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }

        return chunk_actions, result

    def _build_embedding(
        self, input_ids, attention_mask, pixel_values, labels, proprio=None
    ):
        # Prepare inputs by adding necessary tokens
        input_ids, attention_mask = self._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )

        # Update labels tensor for action mask computation later
        labels = self._prepare_labels_for_action_prediction(labels, input_ids)

        padding_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        assert torch.all(padding_mask == attention_mask.ne(0))
        padding_mask = padding_mask.int()
        sorted_indices = torch.argsort(
            padding_mask, dim=1, descending=True, stable=True
        )
        input_ids = torch.gather(input_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)

        # Get input embeddings and action masks
        input_embeddings = self.get_input_embeddings()(input_ids)
        all_actions_mask = self._process_action_masks(labels)

        # Extract language embeddings
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )

        # Process vision features
        projected_patch_embeddings = self._process_vision_features(
            pixel_values, language_embeddings, self.use_film
        )

        # Add proprioceptive features if provided
        use_proprio = self.proprio_projector is not None and proprio is not None
        if use_proprio:
            proprio = torch.Tensor(proprio).to(
                projected_patch_embeddings.device,
                dtype=projected_patch_embeddings.dtype,
            )
            projected_patch_embeddings = self._process_proprio_features(
                projected_patch_embeddings, proprio, self.proprio_projector
            )

        # Zero out action token embeddings
        all_actions_mask = all_actions_mask.unsqueeze(-1)  # (B, seq_len, 1)
        input_embeddings = input_embeddings * ~all_actions_mask

        # Build multimodal embeddings and attention mask
        multimodal_embeddings, multimodal_attention_mask = (
            self._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
        )
        multimodal_position_ids = (
            multimodal_attention_mask.long().cumsum(-1) - 1
        ).masked_fill(multimodal_attention_mask == 0, 1)
        return multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        pixel_values: torch.FloatTensor = None,
        proprio: torch.FloatTensor = None,
        output_hidden_states: bool = False,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: Optional[bool] = None,
        **kwargs,
    ):
        if forward_inputs is not None:
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]
            proprio = forward_inputs["proprio"]
            action_tokens = forward_inputs["action_tokens"]

        # Create fake labels tensor (needed for action mask)
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX

        # num_prompt_tokens = input_ids.shape[-1] - 1
        num_prompt_tokens = (
            input_ids.ne(self.processor.tokenizer.pad_token_id).sum(dim=1) - 1
        )

        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        attention_mask = attention_mask.to(torch.long)

        multimodal_embeddings, multimodal_attention_mask, multimodal_position_ids = (
            self._build_embedding(
                input_ids, attention_mask, pixel_values, labels, proprio
            )
        )

        # Calculate number of patches (including proprio token and/or diffusion timestep embedding if present)
        num_patches = (
            self.vision_backbone.get_num_patches()
            * self.vision_backbone.get_num_images_in_input()
        )
        use_proprio = self.proprio_projector is not None and proprio is not None
        if use_proprio:
            num_patches += 1

        if compute_values:
            output_hidden_states = True

        # Forward pass through language model
        outputs = self.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            position_ids=multimodal_position_ids,
            past_key_values=None,
            inputs_embeds=multimodal_embeddings,
            labels=None,
            use_cache=use_cache,
            output_attentions=False,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        # Prepare seq_indices for extracting action tokens (both logits and hidden states)
        batch_size = outputs.logits.shape[0]
        device = outputs.logits.device
        num_prompt_tokens_total = num_patches + num_prompt_tokens
        start_indices = num_prompt_tokens_total.unsqueeze(1)
        position_offsets = torch.arange(
            self.action_dim * self.num_action_chunks, device=device
        ).unsqueeze(0)
        seq_indices = (
            start_indices + position_offsets
        )  # Shape: [batch_size, action_dim * num_action_chunks]

        logprobs = None
        entropy = None
        if compute_logprobs:
            # Extract logits using seq_indices
            reponse_ids_logits = outputs.logits[
                torch.arange(batch_size, device=device).unsqueeze(-1), seq_indices, :
            ]  # Shape: [batch_size, action_dim * num_action_chunks, vocab_size]
            action_logits = reponse_ids_logits[
                ...,
                -self.config.n_action_bins
                - self.config.pad_to_multiple_of : -self.config.pad_to_multiple_of,
            ]  # Shape: [batch_size, action_dim * num_action_chunks, 256]

            action_logits = action_logits / kwargs["temperature"]
            top_k = min(kwargs["top_k"], action_logits.size(-1))  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                action_logits = logits_warper(None, action_logits)

            logprobs_result = self._compute_logprobs_and_entropy(
                action_logits=action_logits,
                action_tokens=action_tokens,
                compute_entropy=compute_entropy,
            )
            logprobs = logprobs_result["logprobs"]
            entropy = logprobs_result.get("entropy")

        if compute_values:
            last_hidden_states = outputs.hidden_states[-1]
            values = self._compute_values(last_hidden_states)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result
