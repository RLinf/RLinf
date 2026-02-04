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

# You may need to install flashattn (you should wait for ~10 minutes):
# pip install flash-attn --no-build-isolation
#
# Expected `transformers==4.40.1` and `tokenizers==0.19.1`

from typing import Any, ClassVar, Optional, Union

import numpy as np
import torch
import torchvision.transforms.functional as TVF
from prismatic.extern.hf.modeling_prismatic import (
    IGNORE_INDEX,
    OpenVLAForActionPrediction,
    PrismaticCausalLMOutputWithPast,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticImageProcessor as PrismaticImageProcessorOrginal,
)
from prismatic.extern.hf.processing_prismatic import (
    PrismaticProcessor as PrismaticProcessorOriginal,
)
from transformers.generation import (
    GenerateDecoderOnlyOutput,
    LogitsProcessor,
    LogitsProcessorList,
    TopKLogitsWarper,
)
from transformers.image_processing_utils import BatchFeature
from transformers.tokenization_utils import (
    PaddingStrategy,
    PreTokenizedInput,
    TextInput,
    TruncationStrategy,
)
from transformers.utils import TensorType

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.utils import (
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
)


class OpenVLAForBatchActionPrediction(OpenVLAForActionPrediction):
    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, PrismaticCausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a PrismaticCausalLMOutputWithPast instance."""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        output_projector_features = (
            output_projector_features
            if output_projector_features is not None
            else False
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache and not self.training

        # Instantiate Placeholder for Projector Features
        projected_patch_embeddings = None

        # Note :: We only support forward passes with the following cases:
        #   => Cached Generation :: (input_ids.shape[1] == 1) and (past_key_values is not None)
        #   => Unimodal Forward :: (pixel_values is None)
        #   => Multimodal Forward :: (pixel_values is not None) and (input_ids/embeds.shape[0] == pixel_values.shape[0])

        # === Handle Generation with Cache (`input_ids.shape[1] == 1`) =>> requires `past_keys_values` ===
        if input_ids.shape[1] == 1:
            assert past_key_values is not None, (
                "You must provide `past_key_values` during cached generation!"
            )
            assert labels is None, (
                "Unexpected key `labels` provided during cached generation!"
            )

            multimodal_attention_mask = None
            new_position_ids = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (attention_mask.shape[0], 256),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                multimodal_attention_mask = torch.cat(
                    [
                        attention_mask[:, :1],
                        projected_patch_attention_mask,
                        attention_mask[:, 1:],
                    ],
                    dim=1,
                )  # [B, L]

                new_position_ids = multimodal_attention_mask.cumsum(dim=1) - 1  # [B, L]
                new_position_ids = new_position_ids[:, -1:]  # [B, 1]

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=multimodal_attention_mask,
                position_ids=new_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Unimodal Forward ===
        elif pixel_values is None:
            assert (input_ids is not None) and (inputs_embeds is None), (
                "Missing `input_ids` in language-only forward!"
            )
            assert past_key_values is None, (
                "Unexpected key `past_key_values` provided during language-only forward!"
            )

            language_model_output = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Handle Multimodal Forward ===
        elif (input_ids.shape[0] == pixel_values.shape[0]) or (
            inputs_embeds.shape[0] == pixel_values.shape[0]
        ):
            assert past_key_values is None, (
                "Unexpected key `past_key_values` provided during language-only forward!"
            )

            # Visual Feature Extraction
            pixel_values = pixel_values.reshape(-1, *pixel_values.shape[2:])
            patch_features = self.vision_backbone(pixel_values)

            # Projection Logic =>> Update Attention Mask
            projected_patch_embeddings = self.projector(patch_features)
            projected_patch_embeddings = projected_patch_embeddings.reshape(
                input_ids.shape[0], -1, *projected_patch_embeddings.shape[2:]
            )

            projected_patch_attention_mask = None
            if attention_mask is not None:
                projected_patch_attention_mask = torch.full(
                    (
                        projected_patch_embeddings.shape[0],
                        projected_patch_embeddings.shape[1],
                    ),
                    fill_value=True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )

            # Get Input Embeddings (from Language Model Embeddings)
            input_embeddings = self.get_input_embeddings()(input_ids)

            # Build Multimodal Embeddings & Attention Mask =>> Prismatic defaults to inserting after <BOS> token (1:)
            assert torch.all(input_ids[:, 0] == 1)
            multimodal_embeddings = torch.cat(
                [
                    input_embeddings[:, :1, :],
                    projected_patch_embeddings,
                    input_embeddings[:, 1:, :],
                ],
                dim=1,
            )

            multimodal_attention_mask = None
            if attention_mask is not None:
                assert torch.all(attention_mask[:, 0] == 1)
                multimodal_attention_mask = torch.cat(
                    [
                        attention_mask[:, :1],
                        projected_patch_attention_mask,
                        attention_mask[:, 1:],
                    ],
                    dim=1,
                )

            # position_ids
            multimodal_position_ids = None
            if attention_mask is not None:
                multimodal_position_ids = multimodal_attention_mask.cumsum(dim=1) - 1

            # Build Labels (if specified) =>> Ignore Labels for Patch Embeddings
            multimodal_labels = None
            if labels is not None:
                projected_patch_labels = torch.full(
                    (
                        projected_patch_embeddings.shape[0],
                        projected_patch_embeddings.shape[1],
                    ),
                    fill_value=IGNORE_INDEX,
                    dtype=labels.dtype,
                    device=labels.device,
                )
                multimodal_labels = torch.cat(
                    [labels[:, :1], projected_patch_labels, labels[:, 1:]], dim=1
                )

            # Dispatch to Language Model
            language_model_output = self.language_model(
                input_ids=None,
                attention_mask=multimodal_attention_mask,
                position_ids=multimodal_position_ids,
                past_key_values=None,
                inputs_embeds=multimodal_embeddings,
                labels=multimodal_labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # === Otherwise =>> Assume Invalid! ===
        elif (input_ids.shape[0] != pixel_values.shape[0]) or (
            inputs_embeds.shape[0] != pixel_values.shape[0]
        ):
            raise ValueError(
                "Non-homogenous batch of (text, image) input -- forward() does not support mixed batches!"
            )

        else:
            raise ValueError(
                "Invalid PrismaticForConditionalGeneration `forward()` call with provided arguments:\n"
                f"=> `input_ids` = {input_ids is not None}\n"
                f"=> `attention_mask` = {attention_mask is not None}\n"
                f"=> `pixel_values` = {pixel_values is not None}\n"
                f"=> `labels` = {labels is not None}\n"
                f"=> `input_embeds` = {inputs_embeds is not None}\n"
                f"=> `past_key_values` = {past_key_values is not None}\n"
                f"=> `use_cache` = {use_cache}"
            )

        # Unpack `language_model_output` and return PrismaticCausalLMOutputWithPast (or tuple if not `return_dict`)
        if not return_dict:
            if output_projector_features and (projected_patch_embeddings is not None):
                return *language_model_output, projected_patch_embeddings

            return language_model_output

        return PrismaticCausalLMOutputWithPast(
            loss=language_model_output.loss,
            logits=language_model_output.logits,
            past_key_values=language_model_output.past_key_values,
            hidden_states=language_model_output.hidden_states,
            attentions=language_model_output.attentions,
            projector_features=projected_patch_embeddings,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: str,
    ) -> dict[str, torch.Tensor]:
        """Borrowed from `LlamaForCausalLM` and simplified for batch size = 1; mirrors original PrismaticVLM logic."""
        # Handle `past_key_values` (cache) =>> assume `input_ids` just has unprocessed tokens
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        # If `input_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"input_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
            }
        )

        return model_inputs


class PrismaticImageProcessor(PrismaticImageProcessorOrginal):
    def apply_transform(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])
        img: [B, C, H, W]
        """
        if self.tvf_do_letterbox:
            raise NotImplementedError("Letterbox padding is not yet supported!")

        # [Contract] Fused Backbones expect "channel-stacked" inputs; we'll unpack on the model side!
        imgs_t = []
        for idx in range(len(self.input_sizes)):
            img_idx = TVF.resize(img, **self.tvf_resize_params[idx])
            img_idx = TVF.center_crop(img_idx, **self.tvf_crop_params[idx])
            # img_idx = TVF.to_tensor(img_idx)
            img_idx = img_idx / 255.0
            img_idx = TVF.normalize(img_idx, **self.tvf_normalize_params[idx])

            imgs_t.append(img_idx)

        # [Contract] `imgs_t` is a list of Tensors of shape [B, C, H, W]; stack along dim C
        img_t = torch.cat(imgs_t, dim=1)  # [B, C * n, H, W]

        return img_t

    def preprocess(
        self,
        images: torch.Tensor,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image.Image instances for simplicity.
        @param images: [B, C, H, W]
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray
        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = self.apply_transform(images)

        # Dict[str, torch.Tensor]
        return BatchFeature(
            data={"pixel_values": pixel_values}, tensor_type=return_tensors
        )

    def __call__(self, images: torch.Tensor, **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


class PrismaticProcessor(PrismaticProcessorOriginal):
    attributes: ClassVar[list[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __call__(
        self,
        text: Union[
            TextInput, PreTokenizedInput, list[TextInput], list[PreTokenizedInput]
        ],
        images: torch.Tensor,
        proprio_states: torch.Tensor,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.
        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: torch.Tensor [B, C, H, W].
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)
        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        assert self.tokenizer.padding_side == "left", (
            "Required: Init tokenizer with padding_side='left'"
        )

        pixel_values = self.image_processor(images, return_tensors=return_tensors)[
            "pixel_values"
        ]
        text_inputs = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )

        input_ids = text_inputs["input_ids"]  # [B, L]
        attention_mask = text_inputs["attention_mask"]  # [B, L]

        first_nonzero_indices = torch.argmax(attention_mask, dim=1).unsqueeze(
            1
        )  # [B, 1]
        # assert first token is BOS token
        assert torch.all(
            input_ids.gather(1, first_nonzero_indices) == self.tokenizer.bos_token_id
        )
        # assert left padding
        assert torch.all(input_ids[:, -1] != self.tokenizer.pad_token_id)

        input_ids.scatter_(1, first_nonzero_indices, self.tokenizer.pad_token_id)
        attention_mask.scatter_(1, first_nonzero_indices, 0)

        input_ids[:, 0] = self.tokenizer.bos_token_id
        attention_mask[:, 0] = 1

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError(
                "Batch is malformed; expected same number of images and text inputs!"
            )

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})


class VLALogitsProcessor(LogitsProcessor):
    """
    Only sample the action token.
    """

    def __init__(self, action_num_bins, filter_value: float = -torch.inf):
        super().__init__()
        self.action_num_bins = action_num_bins
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        """
        - scores: [B, vocab-size]
        """
        scores_processed = scores.clone()
        # scores_processed[:, :-self.action_num_bins] = self.filter_value
        scores_processed[:, : 32000 - self.action_num_bins] = self.filter_value
        scores_processed[:, 32000:] = self.filter_value
        return scores_processed


class OpenVLAForRLActionPrediction(OpenVLAForBatchActionPrediction, BasePolicy):
    def __init__(
        self,
        config,
        hidden_size,
        unnorm_key,
        action_dim,
        num_action_chunks,
        add_value_head,
        max_prompt_length,
    ):
        super().__init__(config)
        self._init_logits_processor()

        action_norm_stats = self.get_action_stats(unnorm_key)
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])

        self.hidden_size = hidden_size
        if add_value_head:
            self.value_head = ValueHead(
                input_dim=hidden_size,
                hidden_sizes=(512, 128),
                output_dim=1,
                activation="gelu",
                bias_last=False,
            )

        self.action_dim = action_dim
        self.num_action_chunks = num_action_chunks
        self.unnorm_key = unnorm_key
        self.max_prompt_length = max_prompt_length

    def _init_logits_processor(self):
        self.logits_processors = LogitsProcessorList()
        self.logits_processors.append(VLALogitsProcessor(self.config.n_action_bins))

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        else:
            raise NotImplementedError

    def default_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_projector_features: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        **kwargs,
    ):
        if forward_inputs is not None:
            forward_inputs = self.preprocess_for_train(forward_inputs)
            input_ids = forward_inputs["input_ids"]
            attention_mask = forward_inputs["attention_mask"]
            pixel_values = forward_inputs["pixel_values"]

            action_tokens = forward_inputs["action_tokens"]

        if compute_values:
            output_hidden_states = True

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states,
            output_projector_features=output_projector_features,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )

        if not compute_logprobs and not compute_values:
            return outputs

        if compute_logprobs:
            logits = outputs.logits[
                :, -self.action_dim * self.num_action_chunks - 1 : -1
            ]  # [B, action-dim, vocab-size]

            processed_logits_tensor = logits / kwargs["temperature"]
            top_k = min(
                kwargs["top_k"], processed_logits_tensor.size(-1)
            )  # Safety check
            if top_k > 0:
                logits_warper = TopKLogitsWarper(
                    top_k
                )  # since here is logprob instead of logits, we use 0 instead of -inf
                processed_logits_tensor = logits_warper(None, processed_logits_tensor)

            action_logits = processed_logits_tensor
            action_logits[
                ..., : self.vocab_size - self.config.n_action_bins
            ] = -torch.inf
            action_logits[..., self.vocab_size :] = -torch.inf

            logprobs = compute_logprobs_from_logits(
                logits=action_logits, target=action_tokens
            )

            entropy = None
            if compute_entropy:
                entropy = compute_entropy_from_logits(logits=action_logits)

        if hasattr(self, "value_head") and compute_values:
            last_hidden_state = outputs.hidden_states[-1]
            hidden_features = last_hidden_state[
                :, -self.action_dim * self.num_action_chunks - 1
            ]  # [batch_size, hidden_dim]
            values = self.value_head(hidden_features)
        else:
            values = None

        result = {
            "logprobs": logprobs,
            "entropy": entropy,
            "values": values,
        }

        return result

    @torch.no_grad()
    def predict_action_batch(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        env_obs=None,
        calculate_logprobs=True,
        calculate_values=True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        do_sample = kwargs.pop("do_sample")

        if env_obs is not None:
            task_descriptions = [
                f"In: What action should the robot take to {t.lower()}?\nOut: "
                for t in env_obs["task_descriptions"]
            ]
            image_tensor = env_obs["main_images"].permute(
                0, 3, 1, 2
            )  # [B, H, W, C] -> [B, C, H, W]
            if image_tensor.ndim == 4:
                image_tensor = image_tensor.unsqueeze(1)
            assert image_tensor.ndim == 5

            max_length = self.max_prompt_length
            device = next(self.parameters()).device
            precision = next(self.parameters()).dtype
            processed_obs = self.input_processor(
                text=task_descriptions,
                images=image_tensor,
                padding="max_length",
                max_length=max_length,
            )

            input_ids = processed_obs["input_ids"].to(device=device, dtype=torch.long)
            attention_mask = processed_obs["attention_mask"].to(
                device=device, dtype=torch.bool
            )
            pixel_values = processed_obs["pixel_values"].to(
                device=device, dtype=precision
            )

        forward_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        # assert first token is 1
        assert torch.all(input_ids[:, 0] == 1)
        assert torch.all(attention_mask[:, 0] == 1)
        # last token is space ` `
        assert torch.all(input_ids[:, -1] == 29871)
        assert torch.all(attention_mask[:, -1] == 1)

        generated_results: GenerateDecoderOnlyOutput = self.generate(
            input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_scores=True,
            output_logits=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            do_sample=do_sample,
            logits_processor=self.logits_processors,
            **kwargs,
        )
        action_tokens = generated_results.sequences
        action_tokens = action_tokens[:, -self.action_dim :]

        token_logits = (
            generated_results.scores
        )  # ([B, vocab-size], ...), after logits processor and warper results
        token_logits_tensor = torch.stack(
            token_logits, dim=1
        )  # [B, action-dim, vocab-size]

        last_hidden_states = torch.stack(
            [
                token_hidden_states[-1][:, -1]
                for token_hidden_states in generated_results.hidden_states
            ],
            dim=1,
        )  # [B, hidden_states] -> [B, action-dim, hidden_states]

        predicted_action_token_ids = action_tokens.cpu().numpy()
        discretized_actions = self.vocab_size - predicted_action_token_ids
        discretized_actions = np.clip(
            discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1
        )
        normalized_actions = np.asarray(
            [self.bin_centers[da] for da in discretized_actions]
        )  # [B, dim]

        # Unnormalize actions
        action_norm_stats = self._get_action_stats()
        mask = action_norm_stats.get(
            "mask", np.ones_like(action_norm_stats["q01"], dtype=bool)
        )
        mask = (
            np.array(mask).reshape(1, -1).repeat(action_tokens.shape[0], axis=0)
        )  # [B, dim]
        action_high, action_low = (
            np.array(action_norm_stats["q99"]),
            np.array(action_norm_stats["q01"]),
        )
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8)
            + action_low,
            normalized_actions,
        )

        action_logits = token_logits_tensor
        action_logits[..., : self.vocab_size - self.config.n_action_bins] = -torch.inf
        action_logits[..., self.vocab_size :] = -torch.inf

        chunk_logprobs = compute_logprobs_from_logits(
            logits=action_logits, target=action_tokens
        )

        if hasattr(self, "value_head") and calculate_values:
            hidden_features = last_hidden_states[
                :, -self.action_dim * self.num_action_chunks
            ]  # [batch_size, hidden_dim]

            chunk_values = self.value_head(hidden_features)  # [batch_size, 1]
        else:
            chunk_values = torch.zeros_like(chunk_logprobs[..., :1])

        chunk_actions = actions.reshape(-1, self.num_action_chunks, self.action_dim)
        chunk_action_tokens = action_tokens.reshape(
            -1, self.num_action_chunks, self.action_dim
        )

        forward_inputs["action_tokens"] = chunk_action_tokens

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": chunk_values,
            "forward_inputs": forward_inputs,
        }

        return chunk_actions, result

    def _check_unnorm_key(
        self, norm_stats: dict[str, dict[str, Any]], unnorm_key: Optional[str]
    ) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key

    def _get_action_stats(self) -> dict[str, Any]:
        """Get all the logged statistics for the given dataset."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, self.unnorm_key)
        return self.norm_stats[unnorm_key]["action"]

    def preprocess_for_train(self, data):
        input_ids = data["input_ids"]
        action_tokens = data["action_tokens"]
        attention_mask = data["attention_mask"]

        action_tokens = action_tokens.reshape(action_tokens.shape[0], self.action_dim)

        data["input_ids"] = torch.cat(
            [input_ids, action_tokens], dim=-1
        )  # [B, seq-len+action-dim]
        data["attention_mask"] = torch.cat(
            [attention_mask, torch.ones_like(action_tokens).to(attention_mask.dtype)],
            dim=-1,
        )
        data["action_tokens"] = action_tokens
        return data

    def setup_config_and_processor(self, model_config, input_processor):
        self.vocab_size = (
            model_config.text_config.vocab_size - model_config.pad_to_multiple_of
        )
        self.bins = np.linspace(-1, 1, model_config.n_action_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0
        self.norm_stats = model_config.norm_stats

        action_norm_stats = self._get_action_stats()
        self.min_action = np.array(action_norm_stats["q01"])
        self.max_action = np.array(action_norm_stats["q99"])
        self.action_scale = 1.0

        self.input_processor = input_processor
