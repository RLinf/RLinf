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

"""Export a trained RPent policy with its original VLM and processor."""

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

VLM_LANGUAGE = "model.language_model."
VLM_VISION = "model.visual."
TEXT_MODEL = "model."


def _load(directory: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, "pt") as stream:
            tensors.update({key: stream.get_tensor(key) for key in stream.keys()})
    if not tensors:
        raise FileNotFoundError(f"no safetensors weights in {directory}")
    return tensors


def _language_name(key: str) -> str | None:
    """Return one exported weight name without its model prefix.

    A language-backbone export uses ``model.<name>``; a full-model export nests
    the language weights as ``model.language_model.<name>``. Unknown
    prefixes return ``None``.
    """
    for prefix in (VLM_LANGUAGE, TEXT_MODEL):
        if key.startswith(prefix):
            return key.removeprefix(prefix)
    return None


def export(checkpoint: Path, vlm: Path, output: Path) -> None:
    """Write a Megatron checkpoint's policy into a copy of the served VLM."""
    served = _load(vlm)
    trained = _load(checkpoint / "actor/hf_checkpoint")
    lm_head = trained.pop("lm_head.weight", None)
    unexpected = [
        key
        for key in trained
        if not key.startswith(
            (VLM_LANGUAGE, VLM_VISION, TEXT_MODEL, TEXT_MODEL + "visual.")
        )
    ]
    if unexpected:
        raise ValueError(f"expected language weights only, got {unexpected[:5]}")
    trained = {
        TEXT_MODEL + name: value
        for key, value in trained.items()
        if (name := _language_name(key)) is not None
        and not key.removeprefix(TEXT_MODEL).startswith("visual.")
    }
    embedding = trained.get(TEXT_MODEL + "embed_tokens.weight")
    tied = json.loads((vlm / "config.json").read_text())["text_config"].get(
        "tie_word_embeddings", False
    )
    if lm_head is not None and not (tied and torch.equal(lm_head, embedding)):
        raise ValueError("lm_head.weight is only accepted as a copy of tied embeddings")
    expected = {key for key in served if key.startswith(VLM_LANGUAGE)}
    replaced = {
        VLM_LANGUAGE + key.removeprefix(TEXT_MODEL): value
        for key, value in trained.items()
    }
    if set(replaced) != expected or not all(
        key.startswith(TEXT_MODEL) for key in trained
    ):
        missing = sorted(expected - set(replaced))[:5]
        extra = sorted(set(replaced) - expected)[:5]
        raise ValueError(
            f"language weights differ: missing {missing}, unexpected {extra}"
        )
    for key, value in replaced.items():
        if value.shape != served[key].shape:
            raise ValueError(
                f"{key}: trained shape {tuple(value.shape)}, served {tuple(served[key].shape)}"
            )
    merged = {
        key: value for key, value in served.items() if not key.startswith(VLM_LANGUAGE)
    }
    merged.update(
        {
            key: value.to(served[key].dtype).contiguous()
            for key, value in replaced.items()
        }
    )
    output.mkdir(parents=True, exist_ok=False)
    save_file(merged, output / "model.safetensors", metadata={"format": "pt"})
    for path in vlm.iterdir():
        # The weights, and their index, were rewritten above.
        if (
            path.is_file()
            and path.suffix != ".safetensors"
            and path.name != "model.safetensors.index.json"
        ):
            shutil.copy2(path, output / path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--vlm", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.checkpoint, args.vlm, args.output)


if __name__ == "__main__":
    main()
