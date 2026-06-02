#!/usr/bin/env python
# Copyright 2025 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License").
"""Temporary OpenVLA-OFT attention patch for Ascend NPU.

It is a *temporary* workaround for an NPU operator bug:
the fused attention kernel (``aclnn`` FlashAttention) accumulates bf16 matmuls
in a different order than the GPU kernels the OpenVLA-OFT policy was trained
with, which was empirically responsible for ~80% of GPU<->NPU output divergence
(token agreement on a 32-sample parity test went 20% -> ~98% with this patch).

Once the NPU operator is fixed, uninstall this patch and delete the directory;
nothing in RLinf needs to change.
"""

from __future__ import annotations

import math
import os
import sys

_MARKER = "__oft_npu_attn_patch_installed__"
_PTH_NAME = "_oft_npu_attn_patch.pth"
_VERBOSE = os.environ.get("OFT_NPU_ATTN_VERBOSE", "0") == "1"


def _log(msg: str) -> None:
    print(f"[oft-npu-attn] {msg} (PID {os.getpid()})", file=sys.stderr, flush=True)


def _resolve_compute_dtype():
    """Return the torch dtype to run attention in (default bf16)."""
    import torch

    name = os.environ.get("OFT_NPU_ATTN_DTYPE", "bf16").lower()
    table = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
    }
    dt = table.get(name)
    if dt is None:
        _log(f"WARNING: OFT_NPU_ATTN_DTYPE={name!r} not recognised; using bf16")
        return torch.bfloat16
    return dt


def _npu_stack_present() -> bool:
    """True if the Ascend NPU stack is installed in this interpreter."""
    if os.environ.get("OFT_NPU_ATTN_FORCE") == "1":
        return True
    try:
        import torch_npu  # noqa: F401

        return True
    except Exception:
        return False


def apply() -> None:
    """Monkeypatch transformers' LLaMA SDPA attention with the decomposition.

    A no-op off NPU, when disabled, or when already applied (idempotent).
    """
    if os.environ.get("OFT_NPU_ATTN_DISABLE") == "1":
        if _VERBOSE:
            _log("disabled via OFT_NPU_ATTN_DISABLE=1")
        return
    if not _npu_stack_present():
        if _VERBOSE:
            _log(
                "NPU stack not detected; skipping (set OFT_NPU_ATTN_FORCE=1 to override)"
            )
        return

    try:
        import torch  # noqa: F401
        import torch.nn.functional as F
    except ImportError:
        return
    try:
        from transformers.models.llama import modeling_llama as llama_mod
    except ImportError:
        if _VERBOSE:
            _log("transformers not importable yet; skipping")
        return

    # ── Path A: transformers <=4.45 — LlamaSdpaAttention subclass ────────────
    try:
        from transformers.models.llama.modeling_llama import (
            LlamaSdpaAttention,
            apply_rotary_pos_emb,
            repeat_kv,
        )

        has_sdpa_cls = True
    except ImportError:
        has_sdpa_cls = False

    if has_sdpa_cls:
        if getattr(LlamaSdpaAttention.forward, _MARKER, False):
            if _VERBOSE:
                _log("LlamaSdpaAttention already patched; skipping")
            return

        def _decomposed_forward(
            self,
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position=None,
            **kwargs,
        ):
            if output_attentions:
                return super(LlamaSdpaAttention, self).forward(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            bsz, q_len, _ = hidden_states.size()
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
                1, 2
            )
            v = v.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(
                1, 2
            )

            cos, sin = self.rotary_emb(v, position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            pkv = getattr(self, "past_key_value", past_key_value)
            if pkv is not None:
                cache_kwargs = {
                    "sin": sin,
                    "cos": cos,
                    "cache_position": cache_position,
                }
                k, v = pkv.update(k, v, self.layer_idx, cache_kwargs)

            k = repeat_kv(k, self.num_key_value_groups)
            v = repeat_kv(v, self.num_key_value_groups)

            causal_mask = attention_mask
            if causal_mask is not None:
                causal_mask = causal_mask[:, :, :, : k.shape[-2]]
                depth = causal_mask.shape[-1]
                last = causal_mask[:, :, -1, :].clone()
                causal_mask = last.unsqueeze(2).expand(-1, -1, depth, -1)

            compute_dtype = _resolve_compute_dtype()
            qf, kf, vf = q.to(compute_dtype), k.to(compute_dtype), v.to(compute_dtype)
            scale = 1.0 / math.sqrt(self.head_dim)
            attn = torch.matmul(qf, kf.transpose(-2, -1)) * scale
            if causal_mask is not None:
                attn = attn + causal_mask.to(compute_dtype)
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attention_dropout > 0.0:
                attn = F.dropout(attn, p=self.attention_dropout)
            out = torch.matmul(attn, vf).to(q.dtype)

            out = out.transpose(1, 2).contiguous()
            out = out.view(bsz, q_len, self.hidden_size)
            out = self.o_proj(out)
            return out, None, past_key_value

        setattr(_decomposed_forward, _MARKER, True)
        LlamaSdpaAttention.forward = _decomposed_forward
        _log(
            f"attention patch installed (transformers <=4.45 path, "
            f"compute dtype = {os.environ.get('OFT_NPU_ATTN_DTYPE', 'bf16')})"
        )
        return

    # ── Path B: transformers >=4.46 — module-level sdpa_attention_forward ──
    target = None
    for name in ("sdpa_attention_forward", "eager_attention_forward"):
        if hasattr(llama_mod, name):
            target = name
            break
    if target is None:
        _log("no LlamaSdpaAttention / sdpa_attention_forward found; patch SKIPPED")
        return
    if getattr(getattr(llama_mod, target), _MARKER, False):
        if _VERBOSE:
            _log(f"{target} already patched; skipping")
        return

    def _decomposed_attn(
        module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
    ):
        import torch.nn.functional as F

        compute_dtype = _resolve_compute_dtype()
        qf = query.to(compute_dtype)
        kf = key.to(compute_dtype)
        vf = value.to(compute_dtype)
        attn = torch.matmul(qf, kf.transpose(-2, -1)) * scaling
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn = attn + causal_mask.to(compute_dtype)
        attn = F.softmax(attn, dim=-1)
        if module.training and getattr(module, "attention_dropout", 0.0) > 0.0:
            attn = F.dropout(attn, p=module.attention_dropout)
        out = torch.matmul(attn, vf).to(query.dtype)
        return out, attn

    setattr(_decomposed_attn, _MARKER, True)
    setattr(llama_mod, target, _decomposed_attn)
    _log(
        f"attention patch installed (transformers >=4.46 path, replaced {target}, "
        f"compute dtype = {os.environ.get('OFT_NPU_ATTN_DTYPE', 'bf16')})"
    )


# ──────────────────────────────────────────────────────────────────────────
# Install / uninstall: drop a .pth into the active venv so the patch loads in
# every Python process (head + Ray workers) automatically.
# ──────────────────────────────────────────────────────────────────────────
def _pth_path() -> str:
    import sysconfig

    return os.path.join(sysconfig.get_paths()["purelib"], _PTH_NAME)


def _install() -> int:
    here = os.path.abspath(__file__)
    pth = _pth_path()
    # Single-line .pth: exec this file by absolute path at interpreter startup.
    with open(pth, "w") as f:
        f.write(f"import runpy; runpy.run_path(r'{here}')\n")
    print(f"[oft-npu-attn] installed: {pth}")
    print(f"               -> runs {here} at every Python startup in this venv")
    print("               uninstall with: python patch.py --uninstall")
    return 0


def _uninstall() -> int:
    pth = _pth_path()
    if os.path.exists(pth):
        os.remove(pth)
        print(f"[oft-npu-attn] removed: {pth}")
    else:
        print(f"[oft-npu-attn] nothing to remove (no {pth})")
    return 0


def _status() -> int:
    pth = _pth_path()
    print(f"[oft-npu-attn] .pth installed: {os.path.exists(pth)} ({pth})")
    print(f"[oft-npu-attn] NPU stack present: {_npu_stack_present()}")
    print(
        f"[oft-npu-attn] compute dtype: {os.environ.get('OFT_NPU_ATTN_DTYPE', 'bf16')}"
    )
    try:
        from transformers.models.llama import modeling_llama as m

        for attr in (
            "LlamaSdpaAttention",
            "sdpa_attention_forward",
            "eager_attention_forward",
        ):
            obj = getattr(m, attr, None)
            if obj is None:
                continue
            fn = obj.forward if attr == "LlamaSdpaAttention" else obj
            print(f"[oft-npu-attn] {attr}: patched={getattr(fn, _MARKER, False)}")
    except Exception as exc:
        print(f"[oft-npu-attn] transformers not importable: {exc}")
    return 0


def _cli() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Temporary OpenVLA-OFT NPU attention patch."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--install", action="store_true", help="install the venv .pth (default)"
    )
    group.add_argument("--uninstall", action="store_true", help="remove the venv .pth")
    group.add_argument(
        "--status", action="store_true", help="show install + patch state"
    )
    group.add_argument(
        "--apply", action="store_true", help="apply the patch in THIS process only"
    )
    args = parser.parse_args()

    if args.uninstall:
        return _uninstall()
    if args.status:
        return _status()
    if args.apply:
        apply()
        return 0
    return _install()  # default


if __name__ == "__main__":
    raise SystemExit(_cli())
else:
    # Imported (e.g. via the installed .pth at interpreter startup): apply.
    # Guard hard: this runs at the start of EVERY Python process in the venv, so
    # a patch failure must never break the interpreter -- log and continue.
    try:
        apply()
    except Exception as _exc:  # noqa: BLE001
        _log(f"patch failed, continuing unpatched: {_exc!r}")