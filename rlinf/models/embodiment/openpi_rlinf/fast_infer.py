# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Opt-in fast path for the Stage2 RLT feature model's rollout forward.

``OpenPiPytorchEvalActionModel.extract_rlt_obs`` calls :func:`rlt_forward`
instead of its eager prefix / RLT / Euler block only when ``TIANJI_FAST_INFER``
is set; unset (or ``0``) nothing here is imported and nothing changes.

``TIANJI_FAST_INFER=1`` (exact): the eager computation, op for op, replayed
from a CUDA graph (one per RTC guided / unguided). The eager path is CPU-bound:
about 7k kernel launches, and 90 blocking host-to-device copies of the
attention-mask constant (one per Gemma layer and flow step). Outputs are
bit-identical to the eager path.

``TIANJI_FAST_INFER=2`` (fused): 1 plus kernel changes with the same math,
chosen by ``TIANJI_FAST_INFER_OPTS`` (default ``siglip,rlt``):
  siglip  the three images in one SigLIP batch, fused MHA (no attention-weight
          output that the eager call computes and discards);
  rlt     the RLT encoder's last layer for the RL token only (the only row
          the encoder returns; LayerNorm / attention queries / MLP are per row);
  sdpa    Gemma attention through scaled_dot_product_attention with the same
          additive -2.38e38 mask (not a default: slower on Blackwell, untested
          on the 3090).
Outputs then differ at bf16 rounding level.

``TIANJI_FAST_INFER_GRAPH=0`` runs the fast path eagerly (debugging).

Not changed in any mode: weights, dtypes, image size, token count, flow steps,
noise (the same torch.randn call as the eager sampler), RTC guidance.
"""

from __future__ import annotations

import os
import threading
import time

import einops
import torch
import torch.nn.functional as F

from rlinf.models.embodiment.openpi_rlinf.pi0_model import gemma as _gemma
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import make_attn_mask, posemb_sincos

_BIG_NEG = -2.3819763e38  # gemma.Attention's mask value
_MODES = {"1": 1, "exact": 1, "2": 2, "fused": 2}


def fast_infer_mode() -> int:
    value = os.environ.get("TIANJI_FAST_INFER", "").strip().lower()
    if value in ("", "0"):  # the same test as the branch in extract_rlt_obs
        return 0
    if value not in _MODES:
        raise ValueError(f"TIANJI_FAST_INFER={value!r}: use 0, 1 (exact) or 2 (fused)")
    return _MODES[value]


_ALL_OPTS = ("sdpa", "siglip", "rlt")
_DEFAULT_OPTS = frozenset({"siglip", "rlt"})


def fused_opts(mode: int) -> frozenset:
    """Mode 2's kernel changes: TIANJI_FAST_INFER_OPTS, a comma list of sdpa,
    siglip, rlt (default siglip,rlt)."""
    if mode < 2:
        return frozenset()
    text = os.environ.get("TIANJI_FAST_INFER_OPTS", "").strip()
    opts = frozenset(o.strip() for o in text.split(",") if o.strip()) if text else _DEFAULT_OPTS
    if not opts <= set(_ALL_OPTS):
        raise ValueError(f"TIANJI_FAST_INFER_OPTS={text!r}: choose from {_ALL_OPTS}")
    return opts


# --------------------------------------------------------------------------
# Gemma (prefix + action expert): gemma.Module / Block / Attention, op for op.
# The only eager-path change is the mask constant: torch.where(m, x, scalar)
# instead of torch.where(m, x, torch.tensor(scalar, device=cuda)) -- the same
# float32 value, without a blocking host-to-device copy per layer (which a
# CUDA graph cannot capture).
# --------------------------------------------------------------------------


def _sdpa_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(B, 1, T, S) bool -> additive mask (0 / -2.38e38) whose rows are 16-aligned
    (the memory-efficient kernel would otherwise pad a copy in every layer).
    A fully masked row stays uniform, as in the eager softmax."""
    b, _, t, s = mask.shape
    s_pad = (s + 15) // 16 * 16
    bias = torch.zeros((b, 1, t, s_pad), dtype=dtype, device=mask.device)[..., :s]
    return bias.masked_fill_(~mask, _BIG_NEG)


def _attention(attn, xs, positions, attn_mask, kv_cache, fused, bias):
    dtype = next(x.dtype for x in xs if x is not None)
    q_parts, k_parts, v_parts = [], [], []
    for i, x in enumerate(xs):
        if x is None:
            q_parts.append(None)
            k_parts.append(None)
            v_parts.append(None)
            continue
        B, T, _ = x.shape
        if attn.k_proj[i] is None:
            qkv = attn.q_proj[i](x)
            qkv = qkv.reshape(B, T, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.unbind(dim=2)
        else:
            q = attn.q_proj[i](x).reshape(B, T, attn.num_heads, attn.head_dim)
            k = attn.k_proj[i](x).reshape(B, T, attn.num_kv_heads, attn.head_dim)
            v = attn.v_proj[i](x).reshape(B, T, attn.num_kv_heads, attn.head_dim)
        q_parts.append(q)
        k_parts.append(k)
        v_parts.append(v)

    q = torch.cat([q for q in q_parts if q is not None], dim=1)
    k = torch.cat([k for k in k_parts if k is not None], dim=1)
    v = torch.cat([v for v in v_parts if v is not None], dim=1)
    q = _gemma._apply_rope(q, positions=positions)
    k = _gemma._apply_rope(k, positions=positions)
    if kv_cache is not None:
        cache_k, cache_v = kv_cache
        k = torch.cat([cache_k, k], dim=1)
        v = torch.cat([cache_v, v], dim=1)
    new_kv_cache = (k, v)

    K = attn.num_kv_heads
    G = attn.num_heads // K
    if fused:
        qh = q.transpose(1, 2)
        kh = k.transpose(1, 2)
        vh = v.transpose(1, 2)
        if G > 1:
            kh = kh.repeat_interleave(G, dim=1)
            vh = vh.repeat_interleave(G, dim=1)
        out = F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=bias, scale=attn.head_dim**-0.5
        )
        encoded = out.transpose(1, 2)
    else:
        if attn_mask.dim() == 4:
            attn_mask = attn_mask[:, 0:1, :, :]
        q = q * (attn.head_dim**-0.5)
        q_r = q.reshape(q.shape[0], q.shape[1], K, G, attn.head_dim)
        k_r = k.reshape(k.shape[0], k.shape[1], K, attn.head_dim)
        v_r = v.reshape(v.shape[0], v.shape[1], K, attn.head_dim)
        logits = torch.einsum("BTKGH,BSKH->BKGTS", q_r.float(), k_r.float())
        mask_for_logits = attn_mask[:, :, None, :, :].expand_as(logits).bool()
        masked_logits = torch.where(mask_for_logits, logits, _BIG_NEG)
        probs = F.softmax(masked_logits, dim=-1).to(dtype)
        encoded = torch.einsum("BKGTS,BSKH->BTKGH", probs, v_r.to(dtype))
        encoded = encoded.reshape(encoded.shape[0], encoded.shape[1], K * G, attn.head_dim)

    outputs = []
    start = 0
    for i, x in enumerate(xs):
        if x is not None:
            end = start + x.shape[1]
            expert_out = encoded[:, start:end].reshape(B, x.shape[1], -1)
            outputs.append(attn.o_proj[i](expert_out))
            start = end
        else:
            outputs.append(None)
    return outputs, new_kv_cache


def _block(block, xs, kv_cache, positions, attn_mask, adarms_cond, fused, bias):
    pre_attn, gates = [], []
    for i, x in enumerate(xs):
        if x is not None:
            x_norm, gate = block.pre_attention_norms[i](x, adarms_cond[i])
            pre_attn.append(x_norm)
            gates.append(gate)
        else:
            pre_attn.append(None)
            gates.append(None)
    post_attn, kv_cache = _attention(
        block.attn, pre_attn, positions, attn_mask, kv_cache, fused, bias
    )
    post_attn = [block.dropout(p) if p is not None else None for p in post_attn]
    xs = [
        _gemma._gated_residual(x, y, gate)
        for x, y, gate in zip(xs, post_attn, gates, strict=True)
    ]
    out, gates = [], []
    for i, x in enumerate(xs):
        if x is not None:
            x_norm, gate = block.pre_ffw_norms[i](x, adarms_cond[i])
            out.append(block.mlps[i](x_norm))
            gates.append(gate)
        else:
            out.append(None)
            gates.append(None)
    out = [block.dropout(o) if o is not None else None for o in out]
    xs = [_gemma._gated_residual(x, y, gate) for x, y, gate in zip(xs, out, gates, strict=True)]
    return xs, kv_cache


def _llm(llm, embedded, positions, mask, adarms_cond=None, kv_cache=None, fused=False):
    if adarms_cond is None:
        adarms_cond = [None] * len(llm.configs)
    mask = mask.unsqueeze(1)
    xs = [e.to(llm.embed_dtype) if e is not None else None for e in embedded]
    caches = list(kv_cache) if kv_cache is not None else [None] * len(llm.layers)
    dtype = next(x.dtype for x in xs if x is not None)
    bias = _sdpa_bias(mask, dtype) if fused else None
    new_caches = []
    for i, layer in enumerate(llm.layers):
        xs, new_kv = _block(layer, xs, caches[i], positions, mask, adarms_cond, fused, bias)
        new_caches.append(new_kv)
    outputs = []
    for i in range(len(llm.configs)):
        if xs[i] is not None:
            out, _ = llm.final_norms[i](xs[i], adarms_cond[i])
            outputs.append(out)
        else:
            outputs.append(None)
    return outputs, tuple(new_caches)


# --------------------------------------------------------------------------
# SigLIP (fused mode): SigLIPViT.forward with the images of one request in a
# single batch and need_weights=False (the fused MHA kernel; the eager call
# also computes and averages the attention weights it then discards).
# --------------------------------------------------------------------------


def _siglip(vit, image: torch.Tensor) -> torch.Tensor:
    x = image.permute(0, 3, 1, 2).float()
    x = F.conv2d(
        x,
        vit.stem.weight.float(),
        vit.stem.bias.float() if vit.stem.bias is not None else None,
        stride=vit.stem.stride,
        padding=vit.stem.padding,
    )
    B, C, h, w = x.shape
    x = x.reshape(B, C, h * w).permute(0, 2, 1)
    x = x + vit.pos_embedding.float()
    if vit.pool_type != "none":
        raise NotImplementedError("fast SigLIP path covers pool_type='none' only")
    x = x.to(vit.dtype_mm)
    for layer in vit.encoder.layers:
        norm_dtype = layer.norm1.weight.dtype
        y = layer.norm1(x.to(norm_dtype))
        y, _ = layer.attn(y, y, y, need_weights=False)
        x = x + layer.dropout1(y)
        y = layer.mlp(layer.norm2(x.to(norm_dtype)))
        x = x + layer.dropout2(y)
    x = vit.encoder.norm(x.to(vit.encoder.norm.weight.dtype))
    if vit.head is not None:
        x = vit.head(x)
    return x


def _embed_prefix(pi0, obs, fused):
    """Pi0.embed_prefix (images + language); ar_mask built on the device."""
    tokens, input_mask = [], []
    names = list(obs.images)
    if fused and len(names) > 1:
        shapes = {tuple(obs.images[n].shape) for n in names}
        if len(shapes) != 1:
            raise ValueError(f"fused SigLIP needs equal image shapes, got {shapes}")
        batch = obs.images[names[0]].shape[0]
        all_tokens = _siglip(pi0.img, torch.cat([obs.images[n] for n in names], dim=0))
        image_tokens = list(all_tokens.split(batch, dim=0))
    else:
        image_tokens = [pi0.img(obs.images[n])[0] for n in names]
    for name, tok in zip(names, image_tokens):
        tokens.append(tok)
        input_mask.append(einops.repeat(obs.image_masks[name], "b -> b s", s=tok.shape[1]))
    if obs.tokenized_prompt is not None:
        tokens.append(pi0.llm.embed(obs.tokenized_prompt))
        input_mask.append(obs.tokenized_prompt_mask)
    if pi0.pcd and obs.pcd_xyz is not None:
        raise NotImplementedError("fast path has no point-cloud tokens")
    tokens = torch.cat(tokens, dim=1)
    input_mask = torch.cat(input_mask, dim=1)
    ar_mask = torch.zeros(tokens.shape[1], dtype=torch.bool, device=tokens.device)
    return tokens, input_mask, ar_mask


def _embed_suffix(pi0, obs, noisy_actions, timestep):
    """Pi0.embed_suffix; ``ar_mask[:n] = True`` (a host-to-device copy of the
    Python bool) becomes ``arange < n``, the same bool tensor."""
    input_mask, tokens = [], []
    B = noisy_actions.shape[0]
    noisy_actions = noisy_actions.to(pi0.embed_dtype)
    timestep = timestep.to(pi0.embed_dtype)
    if not pi0.pi05:
        state_token = pi0.state_proj(obs.state.to(pi0.embed_dtype))[:, None, :]
        tokens.append(state_token)
        input_mask.append(torch.ones(B, 1, dtype=torch.bool, device=state_token.device))
    action_tokens = pi0.action_in_proj(noisy_actions)
    time_emb = posemb_sincos(
        timestep, pi0.action_in_proj.out_features, min_period=4e-3, max_period=4.0
    )
    if pi0.pi05:
        time_emb = pi0.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = pi0.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)
        action_expert_tokens = action_tokens
        adarms_cond = time_emb
    else:
        time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=pi0.action_horizon)
        action_time_tokens = torch.cat([action_tokens, time_tokens], dim=-1)
        action_time_tokens = pi0.action_time_mlp_in(action_time_tokens)
        action_time_tokens = F.silu(action_time_tokens)
        action_expert_tokens = pi0.action_time_mlp_out(action_time_tokens)
        adarms_cond = None
    tokens.append(action_expert_tokens)
    input_mask.append(
        torch.ones(
            action_expert_tokens.shape[:2], dtype=torch.bool, device=action_expert_tokens.device
        )
    )
    tokens = torch.cat(tokens, dim=1)
    input_mask = torch.cat(input_mask, dim=1)
    n_ar = 1 if pi0.pi05 else 2
    ar_mask = torch.arange(input_mask.shape[1], device=tokens.device) < n_ar
    return tokens, input_mask, ar_mask, adarms_cond


# --------------------------------------------------------------------------
# RLT encoder (fused mode): the last layer only for the RL token, the one row
# RLTTokenEncoder.forward returns (LayerNorm, attention queries and the MLP are
# per row, so the other rows of that layer never reach the output).
# --------------------------------------------------------------------------


def _rlt_encode_flat(wrapper, prefix_output, prefix_mask, last_row_only):
    module = wrapper.rlt_module
    if not last_row_only:
        return wrapper._encode_rlt_flat(prefix_output, prefix_mask)
    param = next(module.parameters())
    prefix_embs = prefix_output.to(device=param.device, dtype=param.dtype)
    mask = prefix_mask if wrapper.rlt_cfg.rlt_use_mask else None
    enc = module.encoder
    prefix_embs = enc.input_proj(prefix_embs)
    seq_len = prefix_embs.shape[-2]
    if seq_len > enc.prefix_seq_len:
        raise ValueError(f"prefix sequence length {seq_len} exceeds {enc.prefix_seq_len}")
    batch = prefix_embs.shape[0]
    prefix_tokens = prefix_embs + enc.prefix_pos_enc[:seq_len].to(
        device=prefix_embs.device, dtype=prefix_embs.dtype
    )
    rl_tokens = (
        enc.rl_token_embed.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
        .unsqueeze(0)
        .expand(batch, -1, -1)
    )
    rl_pos = enc.rl_token_pos_enc.to(device=prefix_embs.device, dtype=prefix_embs.dtype)
    x = torch.cat([prefix_tokens, rl_tokens.to(dtype=prefix_embs.dtype) + rl_pos], dim=1)
    if mask is not None:
        mask = mask.to(device=prefix_embs.device, dtype=torch.bool)
        mask = torch.cat(
            [mask, torch.ones(batch, 1, device=prefix_embs.device, dtype=torch.bool)], dim=1
        )
    for layer in enc.layers[:-1]:
        x = layer(x, mask=mask)
    layer = enc.layers[-1]
    x = x.to(device=layer.self_norm.weight.device, dtype=layer.self_norm.weight.dtype)
    key_padding_mask = None if mask is None else ~mask
    x_norm = layer.self_norm(x)
    attn = layer.self_attn(
        x_norm[:, -1:],
        x_norm,
        x_norm,
        key_padding_mask=key_padding_mask,
        need_weights=False,
    )[0]
    last = x[:, -1:] + attn
    last = last + layer.mlp(layer.mlp_norm(last))
    return last.reshape(batch, -1)


# --------------------------------------------------------------------------
# The forward replayed by the graph: prefix, z_rl, Euler sampler with RTC.
# --------------------------------------------------------------------------


def _forward(wrapper, obs, noise, rtc_target, rtc_mask, *, guided, opts):
    pi0 = wrapper.model
    fused = "sdpa" in opts
    prefix_tokens, prefix_mask, prefix_ar = _embed_prefix(pi0, obs, "siglip" in opts)
    prefix_attn = make_attn_mask(prefix_mask, prefix_ar)
    positions = torch.cumsum(prefix_mask.int(), dim=1) - 1
    outputs, kv_cache = _llm(pi0.llm, [prefix_tokens, None], positions, prefix_attn, fused=fused)
    rlt_out, rlt_mask = wrapper._select_rlt_prefix_embeddings(
        outputs[0], prefix_mask, obs.tokenized_prompt
    )
    z_rl = _rlt_encode_flat(wrapper, rlt_out, rlt_mask, last_row_only="rlt" in opts).to(
        dtype=torch.float32
    )

    batch_size = obs.state.shape[0]
    device = obs.state.device
    x_t = noise
    dt = -1.0 / wrapper.num_steps
    t = 1.0
    while t >= -dt / 2:
        t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
        suffix_tokens, suffix_mask, suffix_ar, adarms_cond = _embed_suffix(pi0, obs, x_t, t_tensor)
        suffix_len = suffix_tokens.shape[1]
        suffix_attn = make_attn_mask(suffix_mask, suffix_ar)
        prefix_to_suffix = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_len)
        full_attn = torch.cat([prefix_to_suffix, suffix_attn], dim=-1)
        suffix_positions = (
            torch.sum(prefix_mask, dim=-1)[:, None] + torch.cumsum(suffix_mask.int(), dim=-1) - 1
        )
        out, _ = _llm(
            pi0.llm,
            [None, suffix_tokens],
            suffix_positions,
            full_attn,
            adarms_cond=[None, adarms_cond],
            kv_cache=kv_cache,
            fused=fused,
        )
        v_t = pi0.velocity_from_suffix(out[1][:, -pi0.action_horizon :])
        if guided:
            x0 = x_t - t * v_t
            x0 = x0 + rtc_mask * (rtc_target - x0)
            v_t = (x_t - x0) / t
        x_t = x_t + dt * v_t
        t += dt
    return z_rl, x_t


class _Graph:
    def __init__(self, graph, inputs, outputs):
        self.graph, self.inputs, self.outputs = graph, inputs, outputs


def _obs_inputs(obs, noise, rtc_target, rtc_mask):
    return {
        "images": dict(obs.images),
        "image_masks": dict(obs.image_masks),
        "state": obs.state,
        "tokenized_prompt": obs.tokenized_prompt,
        "tokenized_prompt_mask": obs.tokenized_prompt_mask,
        "noise": noise,
        "rtc_target": rtc_target,
        "rtc_mask": rtc_mask,
    }


def _signature(inputs):
    leaves, _ = torch.utils._pytree.tree_flatten(inputs)
    return (tuple(inputs["images"]),) + tuple((tuple(x.shape), x.dtype, x.device) for x in leaves)


class FastRLTRunner:
    """CUDA graphs of :func:`_forward`, one per (RTC guided, input signature)."""

    def __init__(self, wrapper, mode: int, use_graph: bool):
        self.wrapper = wrapper
        self.mode = mode
        self.opts = fused_opts(mode)
        self.use_graph = use_graph
        self.failed = False
        self.graphs: dict = {}
        self.pool = None
        self.param_ptr = None
        self.lock = threading.Lock()

    def _call(self, inputs, guided):
        from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation

        obs = Observation(
            images=inputs["images"],
            image_masks=inputs["image_masks"],
            state=inputs["state"],
            tokenized_prompt=inputs["tokenized_prompt"],
            tokenized_prompt_mask=inputs["tokenized_prompt_mask"],
        )
        return _forward(
            self.wrapper,
            obs,
            inputs["noise"],
            inputs["rtc_target"],
            inputs["rtc_mask"],
            guided=guided,
            opts=self.opts,
        )

    def _capture(self, inputs, guided):
        static = torch.utils._pytree.tree_map(lambda x: x.clone(), inputs)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):  # kernel selection, compiled helpers, allocator
                self._call(static, guided)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        # thread_local: CUDA calls of other threads in the rollout process
        # (Ray, weight sync) cannot invalidate this capture.
        with torch.cuda.graph(
            graph, pool=self.pool, stream=stream, capture_error_mode="thread_local"
        ):
            outputs = self._call(static, guided)
        self.pool = graph.pool()
        torch.cuda.current_stream().wait_stream(stream)
        return _Graph(graph, static, outputs)

    def prepare(self, inputs):
        """Capture both RTC variants for this input signature (once). Returns
        False when graphs are off or capture failed (eager fallback)."""
        if not self.use_graph or self.failed:
            return False
        ptr = next(self.wrapper.parameters()).data_ptr()
        if ptr != self.param_ptr:  # moved / offloaded: old graphs point at freed weights
            self.graphs.clear()
            self.pool = None
            self.param_ptr = ptr
        sig = _signature(inputs)
        for guided in (False, True):
            if (guided, sig) in self.graphs:
                continue
            start = time.perf_counter()
            try:
                self.graphs[(guided, sig)] = self._capture(inputs, guided)
            except Exception as exc:  # keep serving: same math, no graph
                self.failed = True
                self.graphs.clear()
                print(f"TIANJI_FAST_INFER graph capture failed, eager fallback: {exc!r}", flush=True)
                return False
            print(
                f"TIANJI_FAST_INFER mode={self.mode} opts={','.join(sorted(self.opts)) or '-'} "
                f"captured guided={guided} in {time.perf_counter() - start:.2f} s",
                flush=True,
            )
        return True

    @torch.no_grad()
    def run(self, obs, noise, rtc_target, rtc_mask, guided):
        if rtc_target is None:  # same buffers for both variants; unused unguided
            rtc_target = torch.zeros_like(noise)
            rtc_mask = torch.zeros_like(noise[..., :1])
        inputs = _obs_inputs(obs, noise, rtc_target, rtc_mask)
        with self.lock:
            if not noise.is_cuda or not self.prepare(inputs):
                return self._call(inputs, guided)
            entry = self.graphs[(bool(guided), _signature(inputs))]
            torch.utils._pytree.tree_map(lambda dst, src: dst.copy_(src), entry.inputs, inputs)
            entry.graph.replay()
            # The graph overwrites its outputs on the next replay; z_rl is kept
            # in the replay buffer, so hand out copies.
            return tuple(o.clone() for o in entry.outputs)


def _runner(wrapper):
    mode = fast_infer_mode()
    use_graph = os.environ.get("TIANJI_FAST_INFER_GRAPH", "1") != "0"
    runner = getattr(wrapper, "_tianji_fast_runner", None)
    if (
        runner is None
        or runner.mode != mode
        or runner.opts != fused_opts(mode)
        or runner.use_graph != use_graph
    ):
        runner = FastRLTRunner(wrapper, mode, use_graph)
        object.__setattr__(wrapper, "_tianji_fast_runner", runner)
    return runner


@torch.no_grad()
def warmup(wrapper) -> None:
    """Capture the graphs before the first request (rollout worker init), so
    the first chunks do not pay for warm-up and capture. Synthetic inputs of
    the deploy shapes; no RNG is drawn."""
    if fast_infer_mode() == 0 or not getattr(wrapper, "rlt_cfg", None) or not wrapper.rlt_cfg.use_rlt:
        return
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation

    device = wrapper.device
    if device.type != "cuda":
        return
    pi0 = wrapper.model
    h, w = pi0_model_module.IMAGE_RESOLUTION
    tokens = torch.zeros(1, pi0.max_token_len, dtype=torch.long, device=device)
    token_mask = torch.zeros(1, pi0.max_token_len, dtype=torch.bool, device=device)
    token_mask[:, :8] = True
    obs = Observation(
        images={k: torch.zeros(1, h, w, 3, device=device) for k in pi0_model_module.IMAGE_KEYS},
        image_masks={k: torch.ones(1, dtype=torch.bool, device=device) for k in pi0_model_module.IMAGE_KEYS},
        state=torch.zeros(1, pi0.action_dim, device=device),
        tokenized_prompt=tokens,
        tokenized_prompt_mask=token_mask,
    )
    noise = torch.zeros(1, pi0.action_horizon, pi0.action_dim, device=device)
    runner = _runner(wrapper)
    with runner.lock:
        runner.prepare(_obs_inputs(obs, noise, torch.zeros_like(noise), torch.zeros_like(noise[..., :1])))
    torch.cuda.synchronize()


def rlt_forward(wrapper, observation, previous, previous_model, *, noise=None, rng=None):
    """Fast replacement of extract_rlt_obs's model block.

    Returns ``(z_rl float32 [B, z], model_actions float32 [B, H, A])`` as the
    eager block computes them; ``previous`` / ``previous_model`` are the RTC
    inputs extract_rlt_obs prepared.
    """
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module

    runner = _runner(wrapper)
    prepared = pi0_model_module.preprocess_observation(observation, train=False)
    batch_size = prepared.state.shape[0]
    device = prepared.state.device
    rtc_target = rtc_mask = None
    guided = False
    if previous is not None:
        rtc_target, rtc_mask = wrapper._rtc_guidance(previous_model, *previous[1:], device)
        # eager: bool((rtc_mask > 0).any()). The mask's first row is 1 for every
        # valid batch row (rows >= 1), so that is valid.any(); no device sync.
        guided = bool(previous[3].any())
    if noise is None:
        # The eager sampler's draw (same call, same generator state).
        noise = torch.randn(
            batch_size,
            wrapper.model.action_horizon,
            wrapper.model.action_dim,
            device=device,
            generator=rng,
        )
    return runner.run(prepared, noise, rtc_target, rtc_mask, guided)
