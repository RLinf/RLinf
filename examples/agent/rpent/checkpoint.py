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

"""Move Qwen3.5 planner weights between SGLang serving and Megatron training.

Megatron trains the language model alone through Megatron-Bridge's
``Qwen3_5ForCausalLM`` bridge, while SGLang serves the full
``Qwen3_5ForConditionalGeneration`` checkpoint.

``extract`` writes the language model of the served model as the text-only
checkpoint the actor and critic start from. ``export`` writes a training
checkpoint's policy into a copy of the served model, keeping its vision tower,
config and processor, so SGLang can serve the trained policy.
"""

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
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
)


def _load(directory: Path) -> dict[str, torch.Tensor]:
    tensors = {}
    for path in sorted(directory.glob("*.safetensors")):
        with safe_open(path, "pt") as stream:
            tensors.update({key: stream.get_tensor(key) for key in stream.keys()})
    if not tensors:
        raise FileNotFoundError(f"no safetensors weights in {directory}")
    return tensors


def extract(vlm: Path, output: Path) -> None:
    """Write the language model of a Qwen3.5 VLM as a ``Qwen3_5ForCausalLM``."""
    tensors = _load(vlm)
    # Original checkpoints may include MTP weights, unused by the PPO model.
    other = [
        key for key in tensors if not key.startswith((VLM_LANGUAGE, VLM_VISION, "mtp."))
    ]
    if other:
        raise ValueError(f"expected language, vision or MTP weights, got {other[:5]}")
    config = json.loads((vlm / "config.json").read_text())
    text_config = {**config["text_config"], "architectures": ["Qwen3_5ForCausalLM"]}
    output.mkdir(parents=True, exist_ok=False)
    save_file(
        {
            TEXT_MODEL + key.removeprefix(VLM_LANGUAGE): value.contiguous()
            for key, value in tensors.items()
            if key.startswith(VLM_LANGUAGE)
        },
        output / "model.safetensors",
        metadata={"format": "pt"},
    )
    (output / "config.json").write_text(json.dumps(text_config, indent=2) + "\n")
    for name in TOKENIZER_FILES:
        if (vlm / name).is_file():
            shutil.copy2(vlm / name, output / name)


def export(checkpoint: Path, vlm: Path, output: Path) -> None:
    """Write a Megatron checkpoint's policy into a copy of the served VLM."""
    served = _load(vlm)
    trained = _load(checkpoint / "actor/hf_checkpoint")
    embedding = trained.get(TEXT_MODEL + "embed_tokens.weight")
    lm_head = trained.pop("lm_head.weight", None)
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
    extract_parser = commands.add_parser("extract")
    extract_parser.add_argument("--vlm", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    export_parser = commands.add_parser("export")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--vlm", type=Path, required=True)
    export_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "extract":
        extract(args.vlm, args.output)
    else:
        export(args.checkpoint, args.vlm, args.output)


if __name__ == "__main__":
    main()
