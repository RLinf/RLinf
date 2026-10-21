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

"""Convert an RLinf Cosmos3 SFT checkpoint (FSDP2 ``full_weights.pt``) into a
diffusers component directory (``model_diffusers``) for SGLang evaluation.

With ``--base-model-path``, replace the transformer tensors in a matching full
HF snapshot after validating every key and shape. This supports Nano and Edge
in the SGLang environment using only torch and safetensors.

Without a template, the conversion is four steps:

  1. ``full_weights.pt``  -> ``model_safetensors``  (strip the ``omni.`` prefix)
  2. ``model_safetensors`` -> ``model_dcp``         (``torch.distributed.checkpoint``)
  3. ``model_dcp``         -> ``model_hf``           (``cosmos_framework.scripts.export_model``)
  4. ``model_hf``          -> ``model_diffusers``    (``cosmos_framework.scripts.convert_model_to_diffusers``)

Steps 3-4 shell out to cosmos-framework's CLI scripts, so this legacy path must
run in the cosmos-framework environment (``transformers 4.57.x`` + ``diffusers
0.39.0`` with ``Cosmos3OmniTransformer``). Set ``LIBERO_ROOT`` before running
(the export step reads it).
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_EXPERIMENT = "action_policy_libero_all_nano"
DEFAULT_CONFIG = "cosmos_framework/configs/base/config.py"
PIPELINE_CLASS = "Cosmos3OmniDiffusersPipeline"
# Match transformers' default max_shard_size so the DCP shards land near 5 GB.
SHARD_BYTES = 5 * 1024**3

# Training-only / default config nodes that cosmos export_model has no public alias for.
EXPORT_DROPPED_CONFIG_KEYS = ("lbl", "multiview_attention")
# export_model with those nodes removed; argv[1] lists them, the rest is its CLI.
_EXPORT_MODEL_CODE = """
import sys

import cosmos_framework.scripts.export_model as export_model

dropped = sys.argv[1].split(",")
coerce_to_base_model = export_model._coerce_to_base_model


def coerce_and_drop(model_dict):
    coerce_to_base_model(model_dict)
    for key in dropped:
        model_dict["config"].pop(key, None)


export_model._coerce_to_base_model = coerce_and_drop
sys.argv = ["export_model", *sys.argv[2:]]
export_model.main()
"""


# Same attention names as cosmos-framework's native -> Diffusers converter.
_ATTENTION_NAMES = (
    ("q_proj_moe_gen", "add_q_proj"),
    ("k_proj_moe_gen", "add_k_proj"),
    ("v_proj_moe_gen", "add_v_proj"),
    ("o_proj_moe_gen", "to_add_out"),
    ("q_norm_moe_gen", "norm_added_q"),
    ("k_norm_moe_gen", "norm_added_k"),
    ("q_proj", "to_q"),
    ("k_proj", "to_k"),
    ("v_proj", "to_v"),
    ("o_proj", "to_out"),
    ("q_norm", "norm_q"),
    ("k_norm", "norm_k"),
)
_ROOT_NAMES = (
    ("time_embedder.mlp.0.", "time_embedder.linear_1."),
    ("time_embedder.mlp.2.", "time_embedder.linear_2."),
    ("vae2llm.", "proj_in."),
    ("llm2vae.", "proj_out."),
    ("action2llm.", "action_proj_in."),
    ("llm2action.", "action_proj_out."),
)


def _diffusers_key(name: str) -> str:
    if name.startswith("language_model."):
        name = name.removeprefix("language_model.").removeprefix("model.")
        for native, exported in _ATTENTION_NAMES:
            name = name.replace(f".self_attn.{native}.", f".self_attn.{exported}.")
        return name
    for native, exported in _ROOT_NAMES:
        if name.startswith(native):
            return exported + name.removeprefix(native)
    return name


def export_from_template(
    src: str, base_model_path: str, output_path: str, *, use_ema: bool = False
) -> Path:
    """Replace every transformer tensor in a complete, matching HF snapshot.

    The snapshot supplies architecture, tokenizer, VAE and frozen vision assets.
    Require exact transformer key and shape coverage: a base tensor must never
    silently replace a missing SFT tensor. The output is published atomically.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    base = Path(base_model_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if output.is_relative_to(base):
        raise ValueError(
            f"Export directory must be outside the base snapshot: {output}"
        )
    if output.exists():
        raise FileExistsError(f"Expected a new export directory, found {output}")
    for name in ("model_index.json", "transformer/config.json"):
        if not (base / name).is_file():
            raise FileNotFoundError(f"Cosmos3 snapshot is missing {base / name}")
    for name in ("text_tokenizer", "vae", "scheduler"):
        if not (base / name).is_dir():
            raise FileNotFoundError(f"Cosmos3 snapshot is missing {base / name}")

    source = torch.load(src, map_location="cpu", mmap=True, weights_only=True)
    prefix = "omni.net_ema." if use_ema else "omni.net."
    state = {}
    for name, tensor in source.items():
        if not name.startswith(prefix):
            continue
        key = _diffusers_key(name.removeprefix(prefix))
        if key in state:
            raise ValueError(f"Multiple SFT tensors map to {key}")
        state[key] = tensor
    if not state:
        raise ValueError(f"Expected SFT tensors with prefix {prefix!r} in {src}")

    shard_keys = {}
    shapes = {}
    dtypes = {}
    for shard in sorted((base / "transformer").glob("*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as reader:
            shard_keys[shard.name] = list(reader.keys())
            for key in reader.keys():
                if key in shapes:
                    raise ValueError(f"Duplicate template transformer tensor: {key}")
                shapes[key] = tuple(reader.get_slice(key).get_shape())
                dtypes[key] = reader.get_tensor(key).dtype
    missing = shapes.keys() - state.keys()
    unexpected = state.keys() - shapes.keys()
    if missing or unexpected:
        raise ValueError(
            f"SFT/template transformer keys differ: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    for key, shape in shapes.items():
        tensor = state[key]
        # FSDP2 promotes scalar Parameters to one-element tensors.
        if shape == () and tensor.shape == (1,):
            state[key] = tensor.reshape(())
        elif tuple(tensor.shape) != shape:
            raise ValueError(
                f"SFT/template shape mismatch for {key}: {tuple(tensor.shape)} != {shape}"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{output.name}-", dir=output.parent
    ) as tmp:
        staging = Path(tmp) / "model"
        shutil.copytree(
            base,
            staging,
            ignore=shutil.ignore_patterns(".cache", ".git", "*.safetensors"),
        )
        # Copy frozen components, excluding the transformer shards we replace.
        for path in base.rglob("*.safetensors"):
            relative = path.relative_to(base)
            if relative.parts[0] not in ("transformer", ".cache", ".git"):
                (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, staging / relative)
        total_size = 0
        for name, keys in shard_keys.items():
            tensors = {
                key: state[key].to(dtype=dtypes[key]).clone().contiguous()
                for key in keys
            }
            total_size += sum(t.numel() * t.element_size() for t in tensors.values())
            save_file(
                tensors, staging / "transformer" / name, metadata={"format": "pt"}
            )
            del tensors
        index = (
            staging / "transformer" / "diffusion_pytorch_model.safetensors.index.json"
        )
        index.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": total_size},
                    "weight_map": {
                        key: name for name, keys in shard_keys.items() for key in keys
                    },
                },
                indent=2,
            )
        )
        (staging / "rlinf_export.json").write_text(
            json.dumps(
                {
                    "sft_checkpoint": str(Path(src).resolve()),
                    "base_model_path": str(base),
                    "weight_source": "ema" if use_ema else "net",
                    "transformer_tensor_count": len(state),
                },
                indent=2,
            )
        )
        staging.rename(output)
    return output


def step1_strip_omni_prefix(src: str, out: str) -> str:
    """full_weights.pt -> model_safetensors (strip ``omni.`` -> ``net.`` / ``net_ema.``)."""
    import torch
    from safetensors.torch import save_file

    out_dir = os.path.join(out, "model_safetensors")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {src} (~91GB)...")
    # weights_only=False: RLinf stores a state dict carrying metadata.
    sd = torch.load(src, map_location="cpu", weights_only=False)
    print(f"Loaded {len(sd)} keys")

    stripped = {
        (k[5:] if k.startswith("omni.") else k): v.contiguous() for k, v in sd.items()
    }
    print(f"Stripped to {len(stripped)} keys (e.g. {list(stripped)[:3]})")

    dst = os.path.join(out_dir, "model.safetensors")
    save_file(stripped, dst)
    print(f"Saved {os.path.getsize(dst) / 1e9:.1f} GB -> {dst}")
    return dst


def step2_save_dcp(safetensors_path: str, out: str) -> str:
    """model.safetensors -> model_dcp (export_model only reads DCP format)."""
    import torch
    import torch.distributed.checkpoint as dcp
    from cosmos_framework.checkpoint.dcp import CustomSavePlanner
    from safetensors.torch import load_file
    from torch.distributed.checkpoint.filesystem import FileSystemWriter

    out_dir = os.path.join(out, "model_dcp", "model")
    os.makedirs(out_dir, exist_ok=True)

    state_dict = load_file(safetensors_path)
    print(f"Loaded {len(state_dict)} keys")

    nbytes = sum(
        v.numel() * v.element_size()
        for v in state_dict.values()
        if isinstance(v, torch.Tensor)
    )
    nshards = max(1, math.ceil(nbytes / SHARD_BYTES))
    writer = FileSystemWriter(out_dir, thread_count=nshards)
    dcp.save(state_dict=state_dict, storage_writer=writer, planner=CustomSavePlanner())
    print(f"Saved DCP -> {out_dir}")
    return out_dir


def step3_export_to_hf(
    dcp_dir: str, out: str, config_file: str, experiment: str, no_ema: bool
) -> str:
    """model_dcp -> model_hf (cosmos export_model, DCP by training config)."""
    out_dir = os.path.join(out, "model_hf")
    cmd = [
        sys.executable,
        "-c",
        _EXPORT_MODEL_CODE,
        ",".join(EXPORT_DROPPED_CONFIG_KEYS),
        "--checkpoint-path",
        dcp_dir,
        "--config-file",
        config_file,
        "--experiment",
        experiment,
    ]
    if no_ema:
        cmd.append("--no-use-ema-weights")
    cmd += ["-o", out_dir]
    print(
        "DCP -> HF: cosmos_framework.scripts.export_model",
        " ".join(cmd[4:]),
        f"(without model.config.{{{','.join(EXPORT_DROPPED_CONFIG_KEYS)}}})",
    )
    subprocess.run(cmd, check=True)
    return out_dir


def step4_convert_to_diffusers(hf_dir: str, out: str) -> str:
    """model_hf -> model_diffusers (split into components; fix _class_name)."""
    out_dir = os.path.join(out, "model_diffusers")
    cmd = [
        sys.executable,
        "-m",
        "cosmos_framework.scripts.convert_model_to_diffusers",
        "--checkpoint-path",
        hf_dir,
        "-o",
        out_dir,
    ]
    print("HF -> diffusers:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # convert_model_to_diffusers writes a non-standard _class_name; set the
    # pipeline class diffusers expects.
    index = os.path.join(out_dir, "model_index.json")
    data = json.load(open(index))
    if data.get("_class_name") != PIPELINE_CLASS:
        data["_class_name"] = PIPELINE_CLASS
        json.dump(data, open(index, "w"), indent=2)
        print(f"Set _class_name={PIPELINE_CLASS} in {index}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        required=True,
        help="RLinf SFT output .../full_weights.pt (FSDP2, omni.net.* prefix).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Conversion working dir; steps 1-4 write subdirs model_safetensors/"
        "model_dcp/model_hf/model_diffusers under here.",
    )
    parser.add_argument("--config-file", default=DEFAULT_CONFIG)
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT)
    parser.add_argument(
        "--base-model-path",
        help="Matching complete Cosmos3 HF snapshot. Export SFT tensors directly "
        "into its transformer layout, retaining its tokenizer/VAE/config assets.",
    )
    parser.add_argument(
        "--use-ema-weights",
        action="store_true",
        help="Keep the EMA weights (default drops them).",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.base_model_path:
        output = export_from_template(
            args.src,
            args.base_model_path,
            os.path.join(args.out, "model_diffusers"),
            use_ema=args.use_ema_weights,
        )
        print(f"Done. Point rollout.model.model_path at {output}")

    else:
        no_ema = not args.use_ema_weights

        safetensors = step1_strip_omni_prefix(args.src, args.out)
        dcp_dir = step2_save_dcp(safetensors, args.out)
        hf_dir = step3_export_to_hf(
            dcp_dir, args.out, args.config_file, args.experiment, no_ema
        )
        diffusers = step4_convert_to_diffusers(hf_dir, args.out)
        print(f"Done. model_diffusers -> {diffusers}")
        print("Point rollout.model.model_path at this directory for eval.")

    return


if __name__ == "__main__":
    main()
