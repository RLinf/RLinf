# OpenPI checkpoint convertors

Consolidated convertors for the self-contained OpenPI Pi0 and Pi0.5 checkpoints
used by the `openpi_rlinf` model package. Five conversion modes share one core
(`_core.py`) that owns the common plumbing: locating `model.safetensors` inside a
checkpoint directory, safetensors load/save, `config.json` read/write, the
wrapper/FSDP prefix strip, and the single `copy_norm_stats` helper.

Unified entry point:

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode {jax_to_openpi_rlinf,openpi_pytorch_to_openpi_rlinf,sft_to_openpi_rlinf,openpi_rlinf_to_openpi_pytorch,sft2deploy} ...
```

Two named checkpoint layouts are referenced throughout:

- **OpenPI_RLinf** — the bare `Pi0` layout this package loads: a directory with
  `model.safetensors` (keys like `img.*`, `llm.*`, `action_in_proj.*`) plus a
  `config.json`, and a norm-stats asset under
  `physical-intelligence/behavior/norm_stats.json`.
- **OpenPI PyTorch** — the upstream PyTorch / BEHAVIOR-eval layout, with keys under
  `paligemma_with_expert.*` in `model.safetensors`.

The norm-stats file is never modified: every mode copies the input
`norm_stats.json` verbatim to the requested output path.

---

## `jax_to_openpi_rlinf`

JAX Pi0/Pi05 orbax checkpoint -> OpenPI_RLinf bare `Pi0` layout.

- **Input**: a JAX checkpoint directory containing a `params/` subdir (orbax
  pytree). The `--input-norm-stats` path points at the matching
  `norm_stats.json`. Requires `jax` / `orbax` installed (imported lazily, only
  when this mode runs).
- **Output**: `<output-model>/model.safetensors` + `<output-model>/config.json`;
  norm-stats copied to `--output-norm-stats`.
- **Dtype policy**: weights are written in **fp32**, but the emitted
  `config.json` records a `"dtype": "bfloat16"` hint for the eval loader.
- **Norm-stats**: input copied verbatim to the output path.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax_to_openpi_rlinf \
    --input-model       /path/to/pi05_base \
    --input-norm-stats  /path/to/norm_stats.json \
    --output-model      /path/to/pi05_base_openpi_rlinf \
    --output-norm-stats /path/to/pi05_base_openpi_rlinf/physical-intelligence/behavior/norm_stats.json
```

Optional shape flags: `--no-pi05`, `--action-dim`, `--action-horizon`,
`--max-token-len`, `--paligemma-variant`, `--action-expert-variant`.

---

## `openpi_pytorch_to_openpi_rlinf`

OpenPI PyTorch `paligemma_with_expert.*` checkpoint -> OpenPI_RLinf bare `Pi0` layout.

- **Input**: `--input-model` is an OpenPI PyTorch checkpoint directory or a direct
  `model.safetensors` file.
- **Output**: `<output-model>/model.safetensors`; if the input dir carries a
  `config.json` it is copied verbatim into the output; norm-stats copied to
  `--output-norm-stats`.
- **Dtype policy**: weights are passed through unchanged (no cast); only keys and
  weight layouts are transformed (SigLIP Q/K/V concat, MLP transpose+stack,
  norm-prefix rewrites).
- **Norm-stats**: input copied verbatim to the output path.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode openpi_pytorch_to_openpi_rlinf \
    --input-model       /path/to/pi05_base_pytorch \
    --input-norm-stats  /path/to/norm_stats.json \
    --output-model      /path/to/pi05_base_openpi_rlinf \
    --output-norm-stats /path/to/pi05_base_openpi_rlinf/physical-intelligence/behavior/norm_stats.json
```

---

## `sft_to_openpi_rlinf`

RLinf SFT-trained checkpoint -> OpenPI_RLinf bare `Pi0` layout.

- **Input**: `--ckpt` points at a saved SFT checkpoint — the `global_step_<N>`
  dir, its `actor/` subdir, the `model_state_dict/` dir, or the consolidated
  `full_weights.pt` file directly. The convertor strips the wrapper/FSDP key
  prefixes (`model.`, `_fsdp_wrapped_module.`, `_orig_mod.`, `module.`) to recover
  the bare `Pi0` keys.
- **Model configuration**: `--config-name` is required and is resolved through
  `rlinf.models.embodiment.openpi.dataconfig.get_openpi_config`. It is the same
  source used by SFT and eval, and supplies Pi0/Pi0.5 selection, action horizon,
  model action dimension, token length, and state-input semantics. For example,
  use `pi05_behavior`, `pi0_aloha_robotwin`, or `pi05_aloha_robotwin`. This mode
  therefore requires the OpenPI/RLinf config dependencies to be importable.
- **Output**: `<output-model>/model.safetensors` + `<output-model>/config.json`;
  `config.json` is derived from `--config-name`. Norm-stats are copied to
  `--output-norm-stats`.
- **Storage dtype**: `--dtype {fp32,bf16}` is required. It controls an actual
  cast before `model.safetensors` is written; it is not merely metadata. Use
  `fp32` to preserve a full-precision SFT checkpoint and choose `bf16` only when
  a smaller, lossy artifact is intended.
- **Validation**: `--reference-model` optionally checks all keys and tensor
  shapes against a matching OpenPI_RLinf base model.
- **Norm-stats**: input copied verbatim to the output path.

### Behavior and RoboTwin precision

The two SFT configurations use the same mixed-precision policy for training:

| Configuration | Base/model checkpoint | FSDP `param_dtype` | FSDP reduction and buffer dtype |
| --- | --- | --- | --- |
| `behavior_pi05_vla.yaml` | fp32 | bf16 | fp32 |
| `robotwin_sft_openpi_rlinf.yaml` | fp32 | bf16 | fp32 |

This training compute policy is separate from converter storage dtype. The
BEHAVIOR recipe retains its existing bf16 artifact, while the RoboTwin recipe
uses fp32 to preserve its full-precision SFT weights. Eval may still use bf16
compute through its runtime `precision` setting.

```bash
# BEHAVIOR Pi0.5, retaining the existing bf16 artifact.
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi_rlinf \
    --config-name       pi05_behavior \
    --dtype              bf16 \
    --ckpt              /path/to/logs/.../checkpoints/global_step_30000 \
    --input-norm-stats  /path/to/norm_stats.json \
    --output-model      /path/to/pi05_sft_openpi_rlinf \
    --output-norm-stats /path/to/pi05_sft_openpi_rlinf/physical-intelligence/behavior/norm_stats.json
```

```bash
# RoboTwin Pi0, preserving SFT weights in fp32.
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft_to_openpi_rlinf \
    --config-name       pi0_aloha_robotwin \
    --dtype              fp32 \
    --ckpt              /path/to/checkpoints/global_step_30000 \
    --input-norm-stats /path/to/robotwin/norm_stats.json \
    --output-model      /path/to/pi0_robotwin_sft_openpi_rlinf \
    --output-norm-stats /path/to/pi0_robotwin_sft_openpi_rlinf/physical-intelligence/robotwin/norm_stats.json \
    --reference-model   /path/to/pi0_base_openpi_rlinf
```

---

## `openpi_rlinf_to_openpi_pytorch`

OpenPI_RLinf bare `Pi0` layout -> OpenPI PyTorch `paligemma_with_expert.*` layout.

OpenPI_RLinf carries only PaliGemma's single 2048-wide shared embedder. OpenPI
PyTorch additionally requires the separate 1024-wide action-expert head
`paligemma_with_expert.gemma_expert.lm_head.weight`, which OpenPI_RLinf does not
carry and cannot be reconstructed. Therefore:

- **`--reference-model` is mandatory in practice.** With it, the head is sourced
  from the reference OpenPI PyTorch model and the converted state dict is validated
  against the reference (keys and shapes must match exactly) to produce a
  **complete** OpenPI PyTorch checkpoint. The reference `config.json` is copied to the output.
- **Without `--reference-model`, this mode fails loudly** (`RuntimeError`) before
  writing anything, rather than emit an incomplete OpenPI PyTorch checkpoint missing the
  action-expert head.

- **Input**: `--input-model` is an OpenPI_RLinf checkpoint dir, a `model.safetensors`,
  or a torch `model.pt`. `--reference-model` is an OpenPI PyTorch model dir.
- **Output**: `<output-model>/model.safetensors` (+ `config.json` from the
  reference); norm-stats copied to `--output-norm-stats`.
- **Dtype policy**: with a reference model, all output tensors are cast to bf16.
- **Norm-stats**: input copied verbatim to the output path.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode openpi_rlinf_to_openpi_pytorch \
    --input-model       /path/to/pi05_sft_openpi_rlinf/model.safetensors \
    --input-norm-stats  /path/to/pi05_sft_openpi_rlinf/physical-intelligence/behavior/norm_stats.json \
    --output-model      /path/to/pi05_sft_openpi_rlinf_to_openpi_pytorch \
    --output-norm-stats /path/to/pi05_sft_openpi_rlinf_to_openpi_pytorch/physical-intelligence/behavior/norm_stats.json \
    --reference-model   /path/to/pi05_base_pytorch
```

---

## `sft2deploy`

RLinf SFT-trained checkpoint -> OpenPI PyTorch deploy `full_weights.pt` only.
This mode does not copy norm-stats or other assets.

- **Input**: `--ckpt` accepts a saved SFT checkpoint directory or its
  `full_weights.pt`.
- **Output**: `--output` accepts a direct `.pt` path or a deploy directory. For a
  directory, the converter writes `actor/model_state_dict/full_weights.pt`.
- **Reference model**: `--reference-model` is the OpenPI PyTorch model used to
  supply the action-expert `lm_head`, which OpenPI_RLinf cannot reconstruct.
- **Dtype reference**: `--dtype-reference` is an existing deploy
  `full_weights.pt` or its checkpoint directory. Its key set, shapes, and
  per-key dtypes define the output checkpoint.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2deploy \
    --ckpt            /path/to/checkpoints/global_step_20000 \
    --output          /path/to/checkpoints/global_step_20000_openpi_deploy \
    --reference-model /path/to/pi05_base_pytorch \
    --dtype-reference /path/to/existing_deploy/actor/model_state_dict/full_weights.pt
```
