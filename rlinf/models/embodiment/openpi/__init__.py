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
# openpi model configs

import json
import logging
import os
import re
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.openpi.adarms_expert import (
    AdaRMSGemmaRMSNorm,
    enable_openpi_adarms_expert,
)


def _load_hf_export_norm_stats(checkpoint_dir):
    """Recover normalization stats from Hugging Face processor exports.

    Some OpenPI checkpoints are exported in LeRobot/Hugging Face format and do
    not bundle the original ``norm_stats.json`` files. Instead, they store the
    same statistics inside processor safetensors referenced by
    ``policy_preprocessor.json`` / ``policy_postprocessor.json``.  RLinf still
    expects OpenPI-style norm stats, so we reconstruct the subset it needs.
    """

    from openpi.shared.normalize import NormStats
    from safetensors.torch import load_file

    checkpoint_dir = Path(checkpoint_dir)
    processor_specs = [
        checkpoint_dir / "policy_preprocessor.json",
        checkpoint_dir / "policy_postprocessor.json",
    ]
    if not any(path.exists() for path in processor_specs):
        return None

    feature_aliases = {
        "action": "actions",
        "actions": "actions",
        "observation.state": "state",
        "state": "state",
    }
    stat_suffixes = ("mean", "std", "q01", "q99")
    recovered_stats = {}

    for spec_path in processor_specs:
        if not spec_path.exists():
            continue
        spec = json.loads(spec_path.read_text())
        for step in spec.get("steps", []):
            state_file = step.get("state_file")
            if not state_file:
                continue
            state_path = checkpoint_dir / state_file
            if not state_path.exists():
                continue

            tensors = load_file(str(state_path), device="cpu")
            for feature_name, alias in feature_aliases.items():
                if alias in recovered_stats:
                    continue
                stats = {}
                for suffix in stat_suffixes:
                    tensor = tensors.get(f"{feature_name}.{suffix}")
                    if tensor is None:
                        stats = {}
                        break
                    stats[suffix] = tensor.detach().cpu().numpy()
                if stats:
                    recovered_stats[alias] = NormStats(**stats)

    return recovered_stats or None


def _load_hf_visual_feature_order(checkpoint_dir: str) -> tuple[str, ...] | None:
    """Read the visual input feature order from a Hugging Face-exported config."""

    config_path = Path(checkpoint_dir) / "config.json"
    if not config_path.exists():
        return None

    config = json.loads(config_path.read_text())
    input_features = config.get("input_features", {})
    image_feature_order = []
    for key in input_features:
        if not key.startswith("observation.images."):
            continue
        camera_name = key.removeprefix("observation.images.")
        if camera_name in {"left", "right", "top"}:
            image_feature_order.append(camera_name)

    if len(image_feature_order) < 3:
        return None
    return tuple(image_feature_order[:3])


def _ensure_openpi_transformers_overlay() -> None:
    """Relax OpenPI's strict Transformers version gate in RLinf workers."""
    try:
        from transformers.models.siglip import check
    except ImportError:
        pass
        return
    check.check_whether_transformers_replace_is_installed_correctly = lambda: True


def _patch_transformers_for_openpi(model: torch.nn.Module) -> None:
    """Monkey-patch stock transformers 4.57+ to match OpenPI's expected behaviour.

    OpenPI was originally built against a patched transformers 4.53.2 that:
    1. Removed the ``sqrt(hidden_size)`` normalizer in ``GemmaModel.forward``.
    2. Did NOT divide image features by ``sqrt(hidden_size)`` in PaliGemma.
    These two changes cancel when *both* are present in stock 4.57+ but break
    OpenPI when present individually.  This function patches the specific model
    *instances* used by the OpenPI runtime so the numerical path is identical
    to the original patched 4.53.2.
    """
    _patch_gemma_model_no_normalizer(model)
    _patch_paligemma_no_image_scaling(model)


def _patch_gemma_model_no_normalizer(model: torch.nn.Module) -> None:
    """Remove the ``sqrt(hidden_size)`` normalizer from the VLM GemmaModel."""
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask
    from transformers.modeling_outputs import BaseModelOutputWithPast

    gemma_model = model.paligemma_with_expert.paligemma.language_model

    def _forward_no_normalizer(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # NOTE: normalizer (hidden_states *= sqrt(hidden_size)) intentionally
        # skipped — OpenPI embeds language tokens with that scale already
        # applied in embed_prefix().

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    gemma_model.forward = MethodType(_forward_no_normalizer, gemma_model)
    logging.info("Patched VLM GemmaModel.forward to skip sqrt(hidden_size) normalizer.")


def _patch_paligemma_no_image_scaling(model: torch.nn.Module) -> None:
    """Remove the ``image_features / sqrt(hidden_size)`` scaling added in 4.57+."""
    paligemma_model = model.paligemma_with_expert.paligemma.model

    def _get_image_features_no_scaling(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        # NOTE: division by sqrt(hidden_size) intentionally skipped — it was
        # added in transformers 4.57+ to compensate for the GemmaModel
        # normalizer, which we also skip for OpenPI compatibility.
        return image_features

    paligemma_model.get_image_features = MethodType(
        _get_image_features_no_scaling, paligemma_model
    )
    # PaliGemmaForConditionalGeneration.get_image_features delegates to
    # self.model.get_image_features, so patching the inner model is sufficient.
    logging.info(
        "Patched PaliGemmaModel.get_image_features to skip sqrt(hidden_size) scaling."
    )


def _normalize_openpi_state_dict_keys(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Normalize checkpoint keys exported by external OpenPI conversions.

    Hugging Face / LeRobot exports often wrap model parameters under a leading
    ``model.`` prefix, while RLinf's OpenPI module expects the raw parameter
    names. Strip that prefix when present so pretrained weights actually land
    in the model instead of being silently treated as unexpected under
    ``strict=False``.
    """

    if not state_dict:
        return state_dict

    if all(key.startswith("model.") for key in state_dict):
        state_dict = {key[len("model.") :]: value for key, value in state_dict.items()}

    # torch.compile() wraps the model under _orig_mod.*; strip that prefix too.
    if all(key.startswith("_orig_mod.") for key in state_dict):
        state_dict = {
            key[len("_orig_mod.") :]: value for key, value in state_dict.items()
        }

    return state_dict


def _inject_missing_paligemma_embeddings(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Backfill missing PaliGemma input embeddings from tied lm_head weights.

    OpenPI checkpoints commonly serialize the PaliGemma output head but omit
    ``embed_tokens.weight``. RLinf's integrated runtime materializes a distinct
    input embedding parameter, so we provide the missing key explicitly before
    ``load_state_dict``.
    """

    normalized_state_dict = dict(state_dict)
    lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
    embed_tokens_key = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    if (
        lm_head_key in normalized_state_dict
        and embed_tokens_key not in normalized_state_dict
    ):
        normalized_state_dict[embed_tokens_key] = normalized_state_dict[lm_head_key]
    return normalized_state_dict


def _retie_paligemma_embeddings(model: torch.nn.Module) -> None:
    """Retie PaliGemma input/output embeddings after checkpoint load."""

    paligemma = model.paligemma_with_expert.paligemma
    input_embeddings = paligemma.model.language_model.embed_tokens
    lm_head = paligemma.lm_head

    if input_embeddings.weight.data_ptr() != lm_head.weight.data_ptr():
        input_embeddings.weight = lm_head.weight
    paligemma.tie_weights()


def _get_model_weight_paths(checkpoint_dir: str) -> list[str]:
    """Return only actual model weight files from a checkpoint directory."""

    checkpoint_path = Path(checkpoint_dir)
    preferred = [
        checkpoint_path / "model.safetensors",
        checkpoint_path / "model-00001-of-00001.safetensors",
    ]
    preferred_paths = [str(path) for path in preferred if path.exists()]
    if preferred_paths:
        return preferred_paths

    weight_paths = []
    for path in sorted(checkpoint_path.glob("*.safetensors")):
        # Processor/export stats safetensors are not model weights.
        if re.search(r"(pre|post)processor", path.name):
            continue
        weight_paths.append(str(path))
    return weight_paths


def _load_pretrained_state_dict(
    model: torch.nn.Module, model_state_dict: dict[str, torch.Tensor], source: str
) -> None:
    """Load a state dict and log a compact match summary."""

    normalized_state_dict = _normalize_openpi_state_dict_keys(model_state_dict)
    normalized_state_dict = _inject_missing_paligemma_embeddings(normalized_state_dict)
    load_result = model.load_state_dict(normalized_state_dict, strict=False)
    _retie_paligemma_embeddings(model)
    missing = len(load_result.missing_keys)
    unexpected = len(load_result.unexpected_keys)
    logging.info(
        "Loaded OpenPI weights from %s with %d state keys (%d missing, %d unexpected).",
        source,
        len(normalized_state_dict),
        missing,
        unexpected,
    )


def get_model(cfg: DictConfig, torch_dtype=None):
    _ensure_openpi_transformers_overlay()

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    try:
        from transformers.modeling_utils import no_init_weights
    except ImportError:
        no_init_weights = nullcontext

    # Resolve model_path
    model_path = str(cfg.model_path)
    if not os.path.exists(model_path):
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(repo_id=model_path)

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(model_path)

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    # These checkpoints are loaded immediately afterwards, so skipping the
    # default random initialization avoids spending minutes filling large
    # PaliGemma weights that will be overwritten anyway.
    with no_init_weights():
        model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
            actor_model_config
        )
    if getattr(actor_model_config, "pi05", False):
        enable_openpi_adarms_expert(model.paligemma_with_expert)
    _patch_transformers_for_openpi(model)
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        _load_pretrained_state_dict(model, model_state_dict, full_weights_path)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        _load_pretrained_state_dict(model, model_state_dict, actor_full_weights_path)
    else:
        # Original model directory with safetensors files
        weight_paths = _get_model_weight_paths(checkpoint_dir)
        for weight_path in weight_paths:
            # Pretrained OpenPI checkpoints do not contain RL-specific heads
            # such as value/noise heads. Load the raw state dict and rely on
            # ``strict=False`` so those task-specific weights stay randomly
            # initialized instead of failing inside safetensors' stricter helper.
            model_state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            _load_pretrained_state_dict(model, model_state_dict, weight_path)

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if actor_model_config.train_expert_only:
        logging.info(
            "Keeping OpenPI Gemma expert in float32 for training stability while the frozen VLM stays mixed-precision."
        )
        model.paligemma_with_expert.gemma_expert.to(dtype=torch.float32)

        def _make_stable_gemma_rmsnorm_forward(forward_fn):
            def _stable_gemma_rmsnorm_forward(self, hidden_states, *args, **kwargs):
                hidden_states = torch.nan_to_num(
                    hidden_states, nan=0.0, posinf=1e4, neginf=-1e4
                )
                outputs = forward_fn(hidden_states, *args, **kwargs)
                if torch.is_tensor(outputs):
                    return torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
                if isinstance(outputs, tuple) and outputs:
                    sanitized = []
                    for value in outputs:
                        if torch.is_tensor(value):
                            value = torch.nan_to_num(
                                value, nan=0.0, posinf=1e4, neginf=-1e4
                            )
                        sanitized.append(value)
                    return tuple(sanitized)
                return outputs

            return _stable_gemma_rmsnorm_forward

        def _sanitize_decoder_outputs(outputs):
            if torch.is_tensor(outputs):
                return torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)
            if isinstance(outputs, tuple) and outputs:
                first = outputs[0]
                if torch.is_tensor(first):
                    first = torch.nan_to_num(first, nan=0.0, posinf=1e4, neginf=-1e4)
                    return (first, *outputs[1:])
            return outputs

        def _make_stable_gemma_decoder_forward(forward_fn):
            def _stable_gemma_decoder_forward(self, *args, **kwargs):
                return _sanitize_decoder_outputs(forward_fn(*args, **kwargs))

            return _stable_gemma_decoder_forward

        for (
            module_name,
            module,
        ) in model.paligemma_with_expert.gemma_expert.model.named_modules():
            if (
                isinstance(module, AdaRMSGemmaRMSNorm)
                or module.__class__.__name__ == "GemmaRMSNorm"
            ):
                module.forward = MethodType(
                    _make_stable_gemma_rmsnorm_forward(module.forward), module
                )
                logging.info(
                    "Patched GemmaRMSNorm for stable training: %s", module_name
                )
            elif module.__class__.__name__ == "GemmaDecoderLayer":
                module.forward = MethodType(
                    _make_stable_gemma_decoder_forward(module.forward), module
                )
                logging.info(
                    "Patched GemmaDecoderLayer for stable training: %s", module_name
                )

    vision_model = model.paligemma_with_expert.paligemma.vision_tower.vision_model
    original_vision_embeddings_forward = vision_model.embeddings.forward
    vision_encoder_dtype = vision_model.encoder.layers[0].layer_norm1.weight.dtype

    def _vision_embeddings_forward_with_consistent_dtype(
        self, pixel_values, interpolate_pos_encoding=False
    ):
        embeddings = original_vision_embeddings_forward(
            pixel_values, interpolate_pos_encoding=interpolate_pos_encoding
        )
        # SigLIP patch and position embeddings stay in float32, while later
        # encoder layer norms often live in bfloat16. Casting the embedding
        # output to the encoder dtype keeps the runtime path uniform without
        # changing the model's parameter dtypes, so FSDP can still flatten it.
        if torch.is_tensor(embeddings) and embeddings.dtype != vision_encoder_dtype:
            embeddings = embeddings.to(dtype=vision_encoder_dtype)
        return embeddings

    vision_model.embeddings.forward = MethodType(
        _vision_embeddings_forward_with_consistent_dtype,
        vision_model.embeddings,
    )

    original_embed_image = model.paligemma_with_expert.embed_image
    language_embed_dtype = model.paligemma_with_expert.paligemma.language_model.get_input_embeddings().weight.dtype

    def _embed_image_with_float32(self, image):
        # Run the entire vision tower in float32, then cast the projected image
        # tokens back to the language embedding dtype so the text stack keeps
        # the precision it expects.
        if torch.is_tensor(image) and image.dtype != torch.float32:
            image = image.float()
        image_embeds = original_embed_image(image)
        if torch.is_tensor(image_embeds) and image_embeds.dtype != language_embed_dtype:
            image_embeds = image_embeds.to(dtype=language_embed_dtype)
        return image_embeds

    model.paligemma_with_expert.embed_image = MethodType(
        _embed_image_with_float32, model.paligemma_with_expert
    )
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    asset_norm_stats = data_config.norm_stats
    checkpoint_norm_stats = None
    processor_export_norm_stats = _load_hf_export_norm_stats(checkpoint_dir)

    if data_config.asset_id is not None:
        try:
            checkpoint_norm_stats = _checkpoints.load_norm_stats(
                checkpoint_dir, data_config.asset_id
            )
        except FileNotFoundError:
            logging.info(
                "Norm stats were not bundled with checkpoint %s; "
                "falling back to processor export stats or asset config stats.",
                checkpoint_dir,
            )
    elif processor_export_norm_stats is None and asset_norm_stats is None:
        raise ValueError("Asset id is required to load norm stats.")

    if checkpoint_norm_stats is not None:
        norm_stats = checkpoint_norm_stats
    elif processor_export_norm_stats is not None:
        norm_stats = processor_export_norm_stats
        logging.info(
            "Recovered OpenPI norm stats from Hugging Face processor export in %s.",
            checkpoint_dir,
        )
    else:
        norm_stats = asset_norm_stats
    if norm_stats is None:
        raise FileNotFoundError(
            f"Norm stats were not found in checkpoint {checkpoint_dir!r} or asset config "
            f"for asset_id {data_config.asset_id!r}."
        )
    if getattr(actor_model_config, "config_name", None) == "pi05_yam_follower":
        model._yam_camera_order = _load_hf_visual_feature_order(checkpoint_dir)
        if model._yam_camera_order is not None:
            logging.info(
                "Using checkpoint-driven YAM camera order: %s",
                model._yam_camera_order,
            )
        else:
            logging.warning(
                "Could not infer YAM camera order from %s/config.json; "
                "falling back to the default OpenPI slot order.",
                checkpoint_dir,
            )
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
