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
from contextlib import nullcontext
from pathlib import Path
from types import MethodType

import torch
from omegaconf import DictConfig


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


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

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
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        for weight_path in weight_paths:
            # Pretrained OpenPI checkpoints do not contain RL-specific heads
            # such as value/noise heads. Load the raw state dict and rely on
            # ``strict=False`` so those task-specific weights stay randomly
            # initialized instead of failing inside safetensors' stricter helper.
            model_state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            model.load_state_dict(model_state_dict, strict=False)

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
                return torch.nan_to_num(outputs, nan=0.0, posinf=1e4, neginf=-1e4)

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
            if module.__class__.__name__ == "GemmaRMSNorm":
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
    # Prefer the asset-config norm stats if already available, and only
    # override them from the checkpoint when that metadata exists there.
    norm_stats = data_config.norm_stats
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    try:
        checkpoint_norm_stats = _checkpoints.load_norm_stats(
            checkpoint_dir, data_config.asset_id
        )
        if checkpoint_norm_stats is not None:
            norm_stats = checkpoint_norm_stats
    except FileNotFoundError:
        logging.info(
            "Norm stats were not bundled with checkpoint %s; falling back to asset config stats.",
            checkpoint_dir,
        )
    if norm_stats is None:
        norm_stats = _load_hf_export_norm_stats(checkpoint_dir)
        if norm_stats is not None:
            logging.info(
                "Recovered OpenPI norm stats from Hugging Face processor export in %s.",
                checkpoint_dir,
            )
    if norm_stats is None:
        raise FileNotFoundError(
            f"Norm stats were not found in checkpoint {checkpoint_dir!r} or asset config "
            f"for asset_id {data_config.asset_id!r}."
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
