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

"""Convert RLinf Pi0/Pi0.5 SFT PT weights to real-world PT in FP32.

This is deliberately a direct PT-to-PT path. It never creates a bf16
intermediate.  The Pi0/Pi0.5 variant is detected from the trained projection
keys, and the existing lossless new-to-old layout mapper is reused for key and
tensor-layout conversion.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import tempfile
from collections import Counter
from collections.abc import Mapping
from typing import Any

import torch

from rlinf.utils.ckpt_convertor.openpi import openpi_rlinf_to_openpi_pytorch
from rlinf.utils.ckpt_convertor.openpi._core import as_state_dict, strip_wrapper_prefix

ACTION_EXPERT_LM_HEAD = "paligemma_with_expert.gemma_expert.lm_head.weight"
ACTION_EXPERT_PREFIX = "paligemma_with_expert.gemma_expert.model."
EMBED_TOKENS_KEY = (
    "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
)
WEIGHTS_CANDIDATES = (
    "actor/model_state_dict/full_weights.pt",
    "model_state_dict/full_weights.pt",
    "full_weights.pt",
)
DEPLOY_WEIGHTS_REL = pathlib.Path("actor/model_state_dict/full_weights.pt")


def resolve_full_weights(path: str | pathlib.Path) -> pathlib.Path:
    """Resolve a checkpoint file or directory to ``full_weights.pt``.

    Args:
        path: Checkpoint file or directory containing a supported weights path.

    Returns:
        The resolved path to ``full_weights.pt``.

    Raises:
        FileNotFoundError: If no supported weights file exists under ``path``.
    """
    path = pathlib.Path(path)
    if path.is_file():
        return path
    for relative in WEIGHTS_CANDIDATES:
        candidate = path / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No full_weights.pt found under {path}")


def resolve_output_pt(output: str | pathlib.Path) -> pathlib.Path:
    """Resolve an output file or deploy directory to ``full_weights.pt``."""
    output = pathlib.Path(output)
    if output.suffix == ".pt":
        return output
    return output / DEPLOY_WEIGHTS_REL


def detect_variant(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Detect whether a state dictionary contains Pi0 or Pi0.5 weights.

    Args:
        state_dict: Model state dictionary or reference shape schema.

    Returns:
        ``"pi0"`` for Pi0 weights or ``"pi05"`` for Pi0.5 weights.

    Raises:
        ValueError: If the projection keys do not identify exactly one variant.
    """
    keys = set(state_dict)
    is_pi0 = {
        "state_proj.weight",
        "action_time_mlp_in.weight",
        "action_time_mlp_out.weight",
    }.issubset(keys)
    is_pi05 = {
        "time_mlp_in.weight",
        "time_mlp_out.weight",
    }.issubset(keys) and "state_proj.weight" not in keys
    if is_pi0 == is_pi05:
        raise ValueError(
            "Cannot unambiguously detect Pi0/Pi0.5 from projection keys: "
            f"state_proj={'state_proj.weight' in keys}, "
            f"action_time_mlp={'action_time_mlp_in.weight' in keys}, "
            f"time_mlp={'time_mlp_in.weight' in keys}"
        )
    return "pi0" if is_pi0 else "pi05"


def reference_uses_action_expert_adarms(
    reference_schema: Mapping[str, tuple[int, ...]],
) -> bool:
    """Return whether the reference action expert uses adaptive RMSNorm."""
    return any(
        key.startswith(ACTION_EXPERT_PREFIX)
        and (
            ".input_layernorm.dense." in key
            or ".post_attention_layernorm.dense." in key
            or key.startswith(ACTION_EXPERT_PREFIX + "norm.dense.")
        )
        for key in reference_schema
    )


def load_reference(
    reference: str | pathlib.Path,
) -> tuple[dict[str, tuple[int, ...]], torch.Tensor, str]:
    """Load the old reference schema, old-only expert head, and model variant.

    The reference itself is the source of truth for deploy keys and shapes.  The
    final torch deploy checkpoint has one additional ``EMBED_TOKENS_KEY`` that
    is sourced from the trained SFT checkpoint and added by :func:`convert`.
    """
    reference = pathlib.Path(reference)
    if reference.is_dir():
        safetensors_path = reference / "model.safetensors"
        if safetensors_path.is_file():
            reference = safetensors_path
        else:
            reference = resolve_full_weights(reference)

    if reference.suffix == ".safetensors":
        from safetensors import safe_open

        with safe_open(str(reference), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            if ACTION_EXPERT_LM_HEAD not in keys:
                raise KeyError(f"Reference {reference} lacks {ACTION_EXPERT_LM_HEAD!r}")
            schema = {key: tuple(handle.get_slice(key).get_shape()) for key in keys}
            head = handle.get_tensor(ACTION_EXPERT_LM_HEAD).float().contiguous()
            variant = detect_variant(schema)
            return schema, head, variant

    loaded = torch.load(
        str(reference), map_location="cpu", weights_only=False, mmap=True
    )
    state_dict = as_state_dict(loaded)
    if ACTION_EXPERT_LM_HEAD not in state_dict:
        raise KeyError(f"Reference {reference} lacks {ACTION_EXPERT_LM_HEAD!r}")
    schema = {key: tuple(tensor.shape) for key, tensor in state_dict.items()}
    head = state_dict[ACTION_EXPERT_LM_HEAD].detach().float().cpu().contiguous()
    variant = detect_variant(schema)
    return schema, head, variant


def validate_source_fp32(state_dict: Mapping[str, torch.Tensor]) -> None:
    """Validate that every floating-point source tensor uses FP32.

    Args:
        state_dict: Source SFT state dictionary to validate.

    Raises:
        TypeError: If any floating-point tensor does not use ``torch.float32``.
    """
    bad = {
        key: str(tensor.dtype)
        for key, tensor in state_dict.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    }
    if bad:
        examples = list(bad.items())[:10]
        raise TypeError(
            f"Source is not an all-fp32 SFT checkpoint; {len(bad)} floating "
            f"tensors are non-fp32, examples={examples}"
        )


def validate_output(
    state_dict: Mapping[str, torch.Tensor],
    shape_schema: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """Validate converted keys, shapes, and dtypes against a reference schema.

    Args:
        state_dict: Converted deployment state dictionary.
        shape_schema: Expected tensor shapes keyed by parameter name.

    Returns:
        A report containing tensor count, dtype counts, and storage size.

    Raises:
        RuntimeError: If keys or shapes differ from the reference, or if any
            floating-point output tensor is not FP32.
    """
    expected = set(shape_schema)
    actual = set(state_dict)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise RuntimeError(
            f"Deploy key mismatch: missing={missing[:20]}, extra={extra[:20]}"
        )

    shape_bad = [
        (key, tuple(state_dict[key].shape), shape_schema[key])
        for key in sorted(expected)
        if tuple(state_dict[key].shape) != shape_schema[key]
    ]
    if shape_bad:
        raise RuntimeError(f"Deploy shape mismatch: {shape_bad[:20]}")

    dtype_bad = [
        (key, str(tensor.dtype))
        for key, tensor in state_dict.items()
        if tensor.is_floating_point() and tensor.dtype != torch.float32
    ]
    if dtype_bad:
        raise RuntimeError(f"Non-fp32 deploy tensors: {dtype_bad[:20]}")

    dtype_counts = Counter(str(tensor.dtype) for tensor in state_dict.values())
    tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in state_dict.values()
    )
    return {
        "keys": len(state_dict),
        "dtype_counts": dict(dtype_counts),
        "tensor_bytes": tensor_bytes,
        "tensor_gib": tensor_bytes / 1024**3,
    }


def convert(
    checkpoint: str | pathlib.Path,
    reference_model: str | pathlib.Path,
    output: str | pathlib.Path,
) -> pathlib.Path:
    """Convert an RLinf Pi0/Pi0.5 SFT checkpoint to deployment FP32 PT.

    The source and reference variants are detected automatically. The reference
    supplies the expected legacy layout and the action-expert language-model
    head that is absent from the trained SFT checkpoint.

    Args:
        checkpoint: SFT ``full_weights.pt`` file or checkpoint directory.
        reference_model: Matching Pi0 or Pi0.5 reference model in legacy PT or
            safetensors format.
        output: Destination ``full_weights.pt`` path or deployment directory.

    Returns:
        The path of the converted ``full_weights.pt`` file.

    Raises:
        FileExistsError: If the resolved output file already exists.
        ValueError: If a model variant or normalization layout is ambiguous or
            incompatible between the source and reference.
        KeyError: If a required source or reference tensor is missing.
    """
    source_path = resolve_full_weights(checkpoint)
    output_path = resolve_output_pt(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_path}")

    loaded = torch.load(
        str(source_path), map_location="cpu", weights_only=False, mmap=True
    )
    wrapped = as_state_dict(loaded)
    bare = strip_wrapper_prefix(wrapped, cast_dtype=None)
    validate_source_fp32(bare)
    variant = detect_variant(bare)

    reference_schema, reference_head, reference_variant = load_reference(
        reference_model
    )
    if reference_variant != variant:
        raise ValueError(
            "Source/reference model variant mismatch: "
            f"source={variant}, reference={reference_variant}"
        )

    action_expert_uses_adarms = reference_uses_action_expert_adarms(reference_schema)
    expected_adarms = variant == "pi05"
    if action_expert_uses_adarms != expected_adarms:
        raise ValueError(
            "Reference normalization layout does not match the detected model "
            f"variant: variant={variant}, "
            f"action_expert_uses_adarms={action_expert_uses_adarms}"
        )

    deploy = openpi_rlinf_to_openpi_pytorch.new_to_old_state_dict(
        bare,
        action_expert_uses_adarms=action_expert_uses_adarms,
    )
    if "llm.embedder.embedding.weight" not in bare:
        raise KeyError("SFT checkpoint lacks llm.embedder.embedding.weight")
    deploy[EMBED_TOKENS_KEY] = (
        bare["llm.embedder.embedding.weight"].detach().float().cpu().contiguous()
    )
    deploy[ACTION_EXPERT_LM_HEAD] = reference_head

    # Structural operations above are lossless.  Enforce FP32 storage for every
    # floating tensor without ever passing through bf16.
    deploy = {
        key: (
            tensor.detach().float().cpu().contiguous()
            if tensor.is_floating_point()
            else tensor.detach().cpu().contiguous()
        )
        for key, tensor in deploy.items()
    }
    # The legacy reference supplies every expected deploy key except the token
    # embedding stored separately by the torch deployment checkpoint.
    reference_schema[EMBED_TOKENS_KEY] = tuple(deploy[EMBED_TOKENS_KEY].shape)
    report = validate_output(deploy, reference_schema)
    report.update(
        {
            "variant": variant,
            "source": str(source_path),
            "reference": str(reference_model),
            "output": str(output_path),
        }
    )

    # Write beside the destination and atomically publish only after torch.save
    # succeeds, so an interruption cannot leave a partial final checkpoint.
    fd, temporary_name = tempfile.mkstemp(
        prefix=output_path.name + ".tmp.", dir=str(output_path.parent)
    )
    os.close(fd)
    temporary_path = pathlib.Path(temporary_name)
    try:
        torch.save(deploy, temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    report_path = output_path.with_suffix(output_path.suffix + ".report.json")
    report["file_bytes"] = output_path.stat().st_size
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return output_path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register arguments for the unified checkpoint converter."""
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--reference-model",
        required=True,
        help="Old-layout reference containing the untrained action-expert lm_head",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output full_weights.pt path or deploy directory",
    )


def run(args: argparse.Namespace) -> None:
    """Execute the FP32 real-world PT conversion."""
    convert(args.ckpt, args.reference_model, args.output)


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return parser


def main() -> int:
    """Run the standalone converter command-line interface.

    Returns:
        Zero after a successful conversion.
    """
    args = build_parser().parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
