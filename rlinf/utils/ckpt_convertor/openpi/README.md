# OpenPI 0.5 checkpoint convertors

Consolidated convertors for the self-contained OpenPI 0.5 (`Pi0`) checkpoints
used by the `openpi_pytorch` model package. Four conversion modes share one core
(`_core.py`) that owns the common plumbing: locating `model.safetensors` inside a
checkpoint directory, safetensors load/save, `config.json` read/write, the
wrapper/FSDP prefix strip, and the single `copy_norm_stats` helper.

Unified entry point:

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode {jax2new,old2new,sft2new,new2old} ...
```

Two checkpoint layouts are referenced throughout:

- **new** — the bare `Pi0` layout this package loads: a directory with
  `model.safetensors` (keys like `img.*`, `llm.*`, `action_in_proj.*`) plus a
  `config.json`, and a norm-stats asset under
  `physical-intelligence/behavior/norm_stats.json`.
- **old** — the previous PyTorch / BEHAVIOR-eval layout, with keys under
  `paligemma_with_expert.*` in `model.safetensors`.

The norm-stats file is never modified: every mode copies the input
`norm_stats.json` verbatim to the requested output path.

---

## `jax2new`

JAX Pi0/Pi05 orbax checkpoint -> new bare `Pi0` layout.

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
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode jax2new \
    --input-model       /path/to/pi05_base \
    --input-norm-stats  /path/to/norm_stats.json \
    --output-model      /path/to/pi05_base_pytorch_new \
    --output-norm-stats /path/to/pi05_base_pytorch_new/physical-intelligence/behavior/norm_stats.json
```

Optional shape flags: `--no-pi05`, `--action-dim`, `--action-horizon`,
`--max-token-len`, `--paligemma-variant`, `--action-expert-variant`.

---

## `old2new`

Old `paligemma_with_expert.*` checkpoint -> new bare `Pi0` layout.

- **Input**: `--input-model` is an old-format checkpoint directory or a direct
  `model.safetensors` file.
- **Output**: `<output-model>/model.safetensors`; if the input dir carries a
  `config.json` it is copied verbatim into the output; norm-stats copied to
  `--output-norm-stats`.
- **Dtype policy**: weights are passed through unchanged (no cast); only keys and
  weight layouts are transformed (SigLIP Q/K/V concat, MLP transpose+stack,
  norm-prefix rewrites).
- **Norm-stats**: input copied verbatim to the output path.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode old2new \
    --input-model       /path/to/pi05_base_pytorch \
    --input-norm-stats  /path/to/norm_stats.json \
    --output-model      /path/to/pi05_base_pytorch_new \
    --output-norm-stats /path/to/pi05_base_pytorch_new/physical-intelligence/behavior/norm_stats.json
```

---

## `sft2new`

RLinf SFT-trained checkpoint -> new bare `Pi0` layout.

- **Input**: `--ckpt` points at a saved SFT checkpoint — the `global_step_<N>`
  dir, its `actor/` subdir, the `model_state_dict/` dir, or the consolidated
  `full_weights.pt` file directly. The convertor strips the wrapper/FSDP key
  prefixes (`model.`, `_fsdp_wrapped_module.`, `_orig_mod.`, `module.`) to recover
  the bare `Pi0` keys.
- **Output**: `<output-model>/model.safetensors` + `<output-model>/config.json`
  (the fixed BEHAVIOR pi0.5 architecture config); norm-stats copied to
  `--output-norm-stats`.
- **Dtype policy**: floating-point tensors are **cast to bf16** (the new-format
  eval loader validates that every checkpoint tensor is bf16); integer/bool
  buffers pass through. The `config.json` records `"dtype": "bfloat16"`.
- **Norm-stats**: input copied verbatim to the output path.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode sft2new \
    --ckpt              /path/to/logs/.../checkpoints/global_step_30000 \
    --input-norm-stats  /path/to/norm_stats.json \
    --output-model      /path/to/pi05_sft_pytorch_new \
    --output-norm-stats /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json
```

---

## `new2old`

New bare `Pi0` layout -> old `paligemma_with_expert.*` layout.

The new format carries only PaliGemma's single 2048-wide shared embedder. The old
format additionally requires the separate 1024-wide action-expert head
`paligemma_with_expert.gemma_expert.lm_head.weight`, which the new format does not
carry and cannot be reconstructed. Therefore:

- **`--reference-model` is mandatory in practice.** With it, the head is sourced
  from the reference old-format model and the converted state dict is validated
  against the reference (keys and shapes must match exactly) to produce a
  **complete** old checkpoint. The reference `config.json` is copied to the output.
- **Without `--reference-model`, this mode fails loudly** (`RuntimeError`) before
  writing anything, rather than emit an incomplete checkpoint missing the
  action-expert head.

- **Input**: `--input-model` is a new-format checkpoint dir, a `model.safetensors`,
  or a torch `model.pt`. `--reference-model` is an old-format model dir.
- **Output**: `<output-model>/model.safetensors` (+ `config.json` from the
  reference); norm-stats copied to `--output-norm-stats`.
- **Dtype policy**: with a reference model, all output tensors are cast to bf16.
- **Norm-stats**: input copied verbatim to the output path.

```bash
python -m rlinf.utils.ckpt_convertor.openpi.convert --mode new2old \
    --input-model       /path/to/pi05_sft_pytorch_new/model.safetensors \
    --input-norm-stats  /path/to/pi05_sft_pytorch_new/physical-intelligence/behavior/norm_stats.json \
    --output-model      /path/to/pi05_sft_pytorch_new_2_old \
    --output-norm-stats /path/to/pi05_sft_pytorch_new_2_old/physical-intelligence/behavior/norm_stats.json \
    --reference-model   /path/to/pi05_base_pytorch
```
