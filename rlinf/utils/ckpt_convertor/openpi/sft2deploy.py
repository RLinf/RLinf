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

"""Convert an openpi_pytorch SFT checkpoint to legacy OpenPI deploy weights.

This mode converts a consolidated RLinf SFT ``full_weights.pt`` checkpoint into
the legacy OpenPI deploy ``full_weights.pt`` layout. It first strips the SFT
wrapper prefixes to recover the bare Pi0 checkpoint, converts that checkpoint to
the old ``paligemma_with_expert.*`` layout with a reference old-format OpenPI
model, then packs the resulting state dict as one deploy ``full_weights.pt``.

``--ckpt`` can be a ``global_step_*`` directory, an ``actor`` directory, a
``model_state_dict`` directory, or a direct ``full_weights.pt`` file. ``--output``
can be either a direct ``full_weights.pt`` path or a deploy directory. When a
directory is provided, this writes
``<output>/actor/model_state_dict/full_weights.pt``.
"""

from __future__ import annotations

import argparse
import pathlib
import tempfile
from collections import Counter
from typing import Any

from rlinf.utils.logging import get_logger

logger = get_logger()

EMBED_TOKENS_KEY = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
EMBEDDER_KEYS = (
    "model.llm.embedder.embedding.weight",
    "llm.embedder.embedding.weight",
)

DEPLOY_WEIGHTS_REL = pathlib.Path("actor") / "model_state_dict" / "full_weights.pt"
WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)


def resolve_full_weights(ckpt: str | pathlib.Path) -> pathlib.Path:
    """Find the consolidated ``full_weights.pt`` for an SFT checkpoint."""
    ckpt = pathlib.Path(ckpt)
    if ckpt.is_file():
        return ckpt

    candidates = [ckpt / relative_path for relative_path in WEIGHTS_CANDIDATES]
    weights = next((candidate for candidate in candidates if candidate.is_file()), None)
    if weights is None:
        raise FileNotFoundError(
            f"No full_weights.pt found under {ckpt}; looked at "
            f"{[str(candidate) for candidate in candidates]}."
        )
    return weights


def resolve_output_pt(output: str | pathlib.Path) -> pathlib.Path:
    """Resolve OUTPUT to the final deploy ``full_weights.pt`` path."""
    output = pathlib.Path(output)
    if output.suffix == ".pt":
        return output
    return output / DEPLOY_WEIGHTS_REL


def load_full_weights_pt(path: str | pathlib.Path) -> Any:
    """Load a consolidated ``full_weights.pt`` state dict onto CPU."""
    import torch

    return torch.load(str(path), map_location="cpu", weights_only=False)


def load_reference_state_dict(reference: str | pathlib.Path) -> dict[str, Any]:
    """Load a reference state dict from a deploy pt or old-format model."""
    from rlinf.utils.ckpt_convertor.openpi._core import load_safetensors

    reference = pathlib.Path(reference)
    if reference.suffix == ".pt":
        return load_full_weights_pt(reference)
    if reference.suffix == ".safetensors":
        return load_safetensors(reference)

    model_safetensors = reference / "model.safetensors"
    if model_safetensors.exists():
        return load_safetensors(model_safetensors)

    for relative_path in WEIGHTS_CANDIDATES:
        deploy_pt = reference / relative_path
        if deploy_pt.exists():
            return load_full_weights_pt(deploy_pt)

    raise FileNotFoundError(
        f"No reference checkpoint found under {reference}; expected a deploy "
        "full_weights.pt or an old-format model.safetensors."
    )


def sft_to_new_safetensors(
    input_ckpt: str | pathlib.Path,
    output_model: str | pathlib.Path,
) -> pathlib.Path:
    """Strip SFT wrapper prefixes and write a bare Pi0 safetensors checkpoint."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi._core import (
        as_state_dict,
        save_safetensors,
        strip_wrapper_prefix,
    )

    weights_path = resolve_full_weights(input_ckpt)
    loaded = torch.load(
        str(weights_path), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    bare_state = strip_wrapper_prefix(state_dict, cast_dtype=torch.bfloat16)

    output_model = pathlib.Path(output_model)
    save_safetensors(bare_state, output_model / "model.safetensors")
    logger.info(
        "Converted %s -> %s (%d bf16 tensors)",
        weights_path,
        output_model,
        len(bare_state),
    )
    return output_model / "model.safetensors"


def extract_embed_tokens(
    input_ckpt: str | pathlib.Path,
    new_safetensors: str | pathlib.Path,
) -> Any:
    """Extract the shared token-embedding table from SFT or new-format weights."""
    from rlinf.utils.ckpt_convertor.openpi._core import load_safetensors

    raw = load_full_weights_pt(resolve_full_weights(input_ckpt))
    for key in EMBEDDER_KEYS:
        if key in raw:
            return raw[key].detach().cpu()

    new_sd = load_safetensors(new_safetensors)
    for key in EMBEDDER_KEYS:
        if key in new_sd:
            return new_sd[key].detach().cpu()

    raise KeyError(
        f"Could not find an embedder tensor. Expected one of {EMBEDDER_KEYS}."
    )


def load_dtype_profile(
    dtype_reference: str | pathlib.Path,
) -> dict[str, Any]:
    """Return the per-key dtype map from a reference deploy checkpoint."""
    ref = load_reference_state_dict(dtype_reference)
    return {key: tensor.dtype for key, tensor in ref.items()}


def load_shape_reference(
    dtype_reference: str | pathlib.Path,
) -> dict[str, Any]:
    """Load the state dict used to validate deploy keys and shapes."""
    return load_reference_state_dict(dtype_reference)


def build_deploy_state_dict(
    old_sd: dict[str, Any],
    embed_tokens: Any,
    dtype_profile: dict[str, Any],
) -> dict[str, Any]:
    """Assemble a deploy state dict using the built-in dtype profile."""
    if EMBED_TOKENS_KEY not in dtype_profile:
        raise KeyError(
            f"dtype reference is missing {EMBED_TOKENS_KEY!r}; cannot pack deploy "
            "weights."
        )

    expected_keys = set(dtype_profile)
    missing_from_old = expected_keys - set(old_sd) - {EMBED_TOKENS_KEY}
    if missing_from_old:
        raise KeyError(
            f"old-format state dict is missing {len(missing_from_old)} deploy keys; "
            f"first few: {sorted(missing_from_old)[:5]}"
        )

    merged = dict(old_sd)
    merged[EMBED_TOKENS_KEY] = embed_tokens.detach().cpu()

    deploy = {}
    for key, dtype in dtype_profile.items():
        if key not in merged:
            raise KeyError(f"Missing tensor for deploy key {key!r}")
        deploy[key] = merged[key].to(dtype=dtype).contiguous()
    return deploy


def validate_deploy_state_dict(
    deploy_sd: dict[str, Any],
    dtype_profile: dict[str, Any],
    shape_reference: dict[str, Any],
) -> None:
    """Validate key set, shapes, and dtypes against the supplied reference."""
    expected_keys = set(dtype_profile)
    got_keys = set(deploy_sd)
    missing = expected_keys - got_keys
    extra = got_keys - expected_keys
    if missing or extra:
        raise RuntimeError(
            f"Deploy key mismatch: missing={sorted(missing)} extra={sorted(extra)}"
        )

    dtype_bad = [
        key for key in expected_keys if deploy_sd[key].dtype != dtype_profile[key]
    ]
    if dtype_bad:
        raise RuntimeError(
            f"Deploy dtype mismatch on {len(dtype_bad)} keys; examples: {dtype_bad[:5]}"
        )

    shape_bad = [
        (key, tuple(deploy_sd[key].shape), tuple(shape_reference[key].shape))
        for key in expected_keys
        if tuple(deploy_sd[key].shape) != tuple(shape_reference[key].shape)
    ]
    if shape_bad:
        raise RuntimeError(f"Deploy shape mismatch vs reference: {shape_bad[:5]}")

    nbytes = sum(
        tensor.numel() * tensor.element_size() for tensor in deploy_sd.values()
    )
    dtype_counts = Counter(str(tensor.dtype) for tensor in deploy_sd.values())
    logger.info(
        "Validated deploy checkpoint: %d keys, tensor_bytes=%.3f GiB, dtypes=%s",
        len(deploy_sd),
        nbytes / 1024**3,
        dict(dtype_counts),
    )


def convert_sft_to_deploy_pt(
    input_ckpt: str | pathlib.Path,
    output: str | pathlib.Path,
    reference_model: str | pathlib.Path,
    dtype_reference: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an SFT checkpoint into one legacy deploy ``full_weights.pt``."""
    import torch

    from rlinf.utils.ckpt_convertor.openpi import new2old
    from rlinf.utils.ckpt_convertor.openpi._core import load_safetensors

    output_pt = resolve_output_pt(output)
    output_pt.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="sft2deploy_", dir=str(output_pt.parent)
    ) as work_dir:
        work_dir = pathlib.Path(work_dir)
        new_dir = work_dir / "new"
        old_dir = work_dir / "old"

        logger.info("Step 1/3: sft -> new")
        new_safetensors = sft_to_new_safetensors(input_ckpt, new_dir)

        logger.info("Step 2/3: new -> old")
        new2old.convert_trained_ckpt(
            input_ckpt=str(new_safetensors),
            output_dir=str(old_dir),
            reference_model=str(reference_model),
            norm_stats=None,
        )

        logger.info("Step 3/3: old -> deploy pt")
        old_sd = load_safetensors(old_dir / "model.safetensors")
        dtype_profile = load_dtype_profile(dtype_reference)
        shape_reference = load_shape_reference(dtype_reference)
        embed_tokens = extract_embed_tokens(input_ckpt, new_safetensors)
        deploy_sd = build_deploy_state_dict(old_sd, embed_tokens, dtype_profile)
        validate_deploy_state_dict(deploy_sd, dtype_profile, shape_reference)

        torch.save(deploy_sd, output_pt)
        logger.info(
            "Wrote deploy weights -> %s (%.3f GiB)",
            output_pt,
            output_pt.stat().st_size / 1024**3,
        )
        return output_pt


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``sft2deploy`` mode arguments on ``parser``."""
    parser.add_argument(
        "--ckpt",
        required=True,
        help="saved SFT checkpoint or consolidated full_weights.pt",
    )
    parser.add_argument(
        "--output",
        required=True,
        help=(
            "output full_weights.pt path, or deploy directory where "
            "actor/model_state_dict/full_weights.pt will be written"
        ),
    )
    parser.add_argument(
        "--reference-model",
        required=True,
        help=(
            "reference old-format OpenPI model dir used by new2old to source the "
            "old-only action-expert lm_head and validate keys/shapes"
        ),
    )
    parser.add_argument(
        "--dtype-reference",
        required=True,
        help=(
            "reference deploy full_weights.pt (or its checkpoint directory) used "
            "for the final key set, shapes, and per-key dtype profile"
        ),
    )


def run(args: argparse.Namespace) -> None:
    """Execute the ``sft2deploy`` mode from parsed ``args``."""
    convert_sft_to_deploy_pt(
        args.ckpt,
        args.output,
        args.reference_model,
        args.dtype_reference,
    )
