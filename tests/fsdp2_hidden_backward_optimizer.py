#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist


# ----------------------------
# Device / distributed helpers
# ----------------------------

def try_import_torch_npu():
    try:
        import torch_npu  # noqa: F401
        return True
    except Exception:
        return False


def get_device_type():
    if hasattr(torch, "npu") and try_import_torch_npu():
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass

    if torch.cuda.is_available():
        return "cuda"

    raise RuntimeError("Neither NPU nor CUDA is available.")


def get_device_api(device_type):
    if device_type == "npu":
        return torch.npu
    if device_type == "cuda":
        return torch.cuda
    raise ValueError(device_type)


def synchronize(device_type):
    try:
        get_device_api(device_type).synchronize()
    except Exception:
        pass


def empty_cache(device_type):
    try:
        get_device_api(device_type).empty_cache()
    except Exception:
        pass


def reset_peak(device_type):
    try:
        get_device_api(device_type).reset_peak_memory_stats()
    except Exception:
        pass


def gib(x):
    return x / 1024**3


def mem_report(tag, device_type, rank, log_all_ranks=False, barrier=False):
    if barrier and dist.is_initialized():
        dist.barrier()

    synchronize(device_type)
    api = get_device_api(device_type)

    try:
        alloc = api.memory_allocated()
    except Exception:
        alloc = -1

    try:
        reserved = api.memory_reserved()
    except Exception:
        reserved = -1

    try:
        max_alloc = api.max_memory_allocated()
    except Exception:
        max_alloc = -1

    try:
        max_reserved = api.max_memory_reserved()
    except Exception:
        max_reserved = -1

    if log_all_ranks or rank == 0:
        print(
            f"[MEM][rank={rank}] {tag}: "
            f"alloc={gib(alloc):.3f} GiB, "
            f"reserved={gib(reserved):.3f} GiB, "
            f"max_alloc={gib(max_alloc):.3f} GiB, "
            f"max_reserved={gib(max_reserved):.3f} GiB",
            flush=True,
        )


def init_dist(args):
    device_type = get_device_type()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if device_type == "npu":
        torch.npu.set_device(local_rank)
        backend = args.backend or os.environ.get("BACKEND", "hccl")
    else:
        torch.cuda.set_device(local_rank)
        backend = args.backend or os.environ.get("BACKEND", "nccl")

    if not dist.is_initialized():
        dist.init_process_group(backend=backend)

    from torch.distributed.device_mesh import init_device_mesh
    mesh = init_device_mesh(device_type, (world_size,))

    return device_type, rank, local_rank, world_size, mesh


# ----------------------------
# dtype / FSDP2 helpers
# ----------------------------

def parse_dtype(s):
    s = str(s).lower()
    if s in ("none", "null"):
        return None
    if s in ("fp32", "float32", "torch.float32"):
        return torch.float32
    if s in ("fp16", "float16", "half", "torch.float16"):
        return torch.float16
    if s in ("bf16", "bfloat16", "torch.bfloat16"):
        return torch.bfloat16
    raise ValueError(f"Unknown dtype: {s}")


def import_fsdp2():
    try:
        from torch.distributed.fsdp import fully_shard
    except Exception:
        from torch.distributed._composable.fsdp import fully_shard

    try:
        from torch.distributed.fsdp import MixedPrecisionPolicy
    except Exception:
        MixedPrecisionPolicy = None

    try:
        from torch.distributed.fsdp import CPUOffloadPolicy
    except Exception:
        CPUOffloadPolicy = None

    return fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy


def apply_fsdp2(module, mesh, args):
    fully_shard, MixedPrecisionPolicy, CPUOffloadPolicy = import_fsdp2()

    mp_policy = None
    mp_param_dtype = parse_dtype(args.mp_param_dtype)
    mp_reduce_dtype = parse_dtype(args.mp_reduce_dtype)

    if MixedPrecisionPolicy is not None and (
        mp_param_dtype is not None or mp_reduce_dtype is not None
    ):
        mp_policy = MixedPrecisionPolicy(
            param_dtype=mp_param_dtype,
            reduce_dtype=mp_reduce_dtype,
            cast_forward_inputs=True,
        )

    offload_policy = None
    if args.cpu_offload:
        if CPUOffloadPolicy is None:
            raise RuntimeError("CPUOffloadPolicy is not available in this PyTorch build.")
        try:
            offload_policy = CPUOffloadPolicy(pin_memory=args.pin_memory)
        except TypeError:
            offload_policy = CPUOffloadPolicy()

    kwargs = {
        "mesh": mesh,
        "reshard_after_forward": args.reshard_after_forward,
    }

    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy

    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy

    try:
        return fully_shard(module, **kwargs)
    except TypeError as e:
        # 兼容不同 PyTorch/FSDP2 版本的参数名。
        kwargs2 = dict(kwargs)
        if "mp_policy" in kwargs2:
            kwargs2["mixed_precision"] = kwargs2.pop("mp_policy")
        try:
            return fully_shard(module, **kwargs2)
        except TypeError:
            raise e


# ----------------------------
# Model structure helpers
# ----------------------------

def get_layers(model):
    if hasattr(model, "layers"):
        return model.layers
    if hasattr(model, "h"):
        return model.h
    if hasattr(model, "blocks"):
        return model.blocks
    raise RuntimeError("Cannot find transformer layers, e.g. model.layers")


def get_embed_tokens(model):
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    if hasattr(model, "wte"):
        return model.wte
    return None


def get_final_norm(model):
    for name in ("norm", "ln_f", "final_layernorm"):
        if hasattr(model, name):
            return getattr(model, name), name
    return None, None


def set_config_attr_if_exists(config, name, value):
    try:
        setattr(config, name, value)
    except Exception:
        pass


def build_backbone_model_on_meta(config, param_dtype, rank):
    from transformers import AutoModel

    if rank == 0:
        print("[BUILD] constructing AutoModel backbone on meta device", flush=True)

    # 尽量兼容 transformers 新旧版本：新版推荐 dtype，旧版可能只认 torch_dtype。
    with torch.device("meta"):
        try:
            model = AutoModel.from_config(
                config,
                trust_remote_code=True,
                dtype=param_dtype,
            )
        except TypeError:
            model = AutoModel.from_config(
                config,
                trust_remote_code=True,
                torch_dtype=param_dtype,
            )

    return model


# ----------------------------
# Debug / report helpers
# ----------------------------

def init_params_constant(model, rank):
    if rank == 0:
        print("[INIT_PARAM] filling local parameters with constants...", flush=True)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.is_meta:
                raise RuntimeError(f"Parameter is still meta after to_empty: {name}")

            if "norm" in name and p.ndim == 1:
                p.fill_(1.0)
            else:
                p.fill_(0.001)


def report_params(model, rank):
    logical_total = 0
    trainable_total = 0

    for _, p in model.named_parameters():
        logical_total += p.numel()
        if p.requires_grad:
            trainable_total += p.numel()

    if rank == 0:
        print(
            f"[PARAM_SUM] logical_total_numel={logical_total:,}, "
            f"requires_grad_numel={trainable_total:,}",
            flush=True,
        )


def print_fsdp_units(model, rank, topk=100):
    rows = []

    for name, mod in model.named_modules():
        cls = mod.__class__.__name__
        if "FSDP" not in cls:
            continue

        direct_numel = 0
        direct_nparams = 0
        for _, p in mod.named_parameters(recurse=False):
            direct_nparams += 1
            direct_numel += p.numel()

        rows.append((name, cls, direct_nparams, direct_numel))

    if rank == 0:
        print(f"[FSDP_UNITS] num_units={len(rows)}", flush=True)
        for name, cls, direct_nparams, direct_numel in rows[:topk]:
            print(
                f"[FSDP_UNITS] name={name}, cls={cls}, "
                f"direct_nparams={direct_nparams}, direct_numel={direct_numel:,}",
                flush=True,
            )


def safe_register_pre_hook(module, hook_fn, prepend=False):
    try:
        return module.register_forward_pre_hook(hook_fn, prepend=prepend)
    except TypeError:
        return module.register_forward_pre_hook(hook_fn)


def safe_register_forward_hook(module, hook_fn, prepend=False):
    try:
        return module.register_forward_hook(hook_fn, prepend=prepend)
    except TypeError:
        return module.register_forward_hook(hook_fn)


def add_memory_hooks(model, device_type, rank, args):
    """
    hook-style:
      simple:
        pre/post 各一次。输出较少。
      detailed:
        PRE_EARLY: 尽量在 FSDP pre-hook 之前打印
        PRE_LATE : 尽量在 FSDP pre-hook 之后打印，即可能已经 all-gather full params
        POST_EARLY: 尽量在 FSDP post-hook 之前打印
        POST_LATE : 尽量在 FSDP post-hook 之后打印，即可能已经 reshard/release
    """
    handles = []

    def should_hook_layer(i):
        return args.hook_every > 0 and (i % args.hook_every == 0)

    def make_pre(tag):
        def hook(mod, inputs):
            mem_report(tag, device_type, rank, log_all_ranks=args.log_all_ranks, barrier=False)
        return hook

    def make_post(tag):
        def hook(mod, inputs, output):
            mem_report(tag, device_type, rank, log_all_ranks=args.log_all_ranks, barrier=False)
        return hook

    def add_one_module_hooks(name, module):
        if args.hook_style == "none":
            return

        if args.hook_style == "simple":
            handles.append(
                safe_register_pre_hook(
                    module,
                    make_pre(f"PRE {name}"),
                    prepend=False,
                )
            )
            handles.append(
                safe_register_forward_hook(
                    module,
                    make_post(f"POST {name}"),
                    prepend=False,
                )
            )
            return

        if args.hook_style == "detailed":
            # prepend=True 的 pre hook 尽量在 FSDP unshard/all-gather 之前打。
            handles.append(
                safe_register_pre_hook(
                    module,
                    make_pre(f"PRE_EARLY {name}"),
                    prepend=True,
                )
            )
            # prepend=False 的 pre hook 通常在 FSDP pre-hook 之后，可能能看到 full-param all-gather 后的显存。
            handles.append(
                safe_register_pre_hook(
                    module,
                    make_pre(f"PRE_LATE {name}"),
                    prepend=False,
                )
            )
            # forward hook 的 prepend=True/False 可以帮助区分 module forward 刚结束与 FSDP post-hook 之后。
            handles.append(
                safe_register_forward_hook(
                    module,
                    make_post(f"POST_EARLY {name}"),
                    prepend=True,
                )
            )
            handles.append(
                safe_register_forward_hook(
                    module,
                    make_post(f"POST_LATE {name}"),
                    prepend=False,
                )
            )
            return

        raise ValueError(f"Unknown hook_style: {args.hook_style}")

    embed = get_embed_tokens(model)
    if embed is not None and args.hook_embed:
        add_one_module_hooks("embed_tokens", embed)

    layers = get_layers(model)
    for i, layer in enumerate(layers):
        if should_hook_layer(i):
            add_one_module_hooks(f"layer.{i}", layer)

    norm, norm_name = get_final_norm(model)
    if norm is not None and args.hook_final_norm:
        add_one_module_hooks(norm_name, norm)

    if rank == 0:
        print(
            f"[HOOK] style={args.hook_style}, hook_every={args.hook_every}, "
            f"num_handles={len(handles)}",
            flush=True,
        )

    return handles


# ----------------------------
# FSDP wrapping
# ----------------------------

def apply_fsdp2_use(model, mesh, args, rank):
    embed = get_embed_tokens(model)
    if embed is not None and args.wrap_embed:
        if rank == 0:
            print("[FSDP_WRAP] wrap embed_tokens", flush=True)
        apply_fsdp2(embed, mesh, args)

    layers = get_layers(model)
    if rank == 0:
        print(f"[FSDP_WRAP] wrap {len(layers)} decoder layers", flush=True)

    for i, layer in enumerate(layers):
        apply_fsdp2(layer, mesh, args)
        if rank == 0 and (i % args.wrap_log_every == 0 or i == len(layers) - 1):
            print(f"[FSDP_WRAP] wrapped layer {i}/{len(layers)-1}", flush=True)

    norm, norm_name = get_final_norm(model)
    if norm is not None and args.wrap_final_norm:
        if rank == 0:
            print(f"[FSDP_WRAP] wrap final norm: {norm_name}", flush=True)
        apply_fsdp2(norm, mesh, args)
        
    if args.wrap_root:
        if rank == 0:
            print("[FSDP_WRAP] wrap root AutoModel backbone, reshard_after_forward=False", flush=True)
        old_reshard = args.reshard_after_forward
        args.reshard_after_forward = False
        apply_fsdp2(model, mesh, args)
        args.reshard_after_forward = old_reshard
    else:
        if rank == 0:
            print("[FSDP_WRAP] intentionally skip root model", flush=True)

    return model



# ----------------------------
# Optimizer helpers
# ----------------------------

def parse_optional_bool(s):
    s = str(s).lower()
    if s in ("auto", "none", "null"):
        return None
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Unknown bool value: {s}")


def tensor_local_numel_nbytes(t):
    """
    FSDP2 参数/优化器状态可能是 DTensor。
    对 DTensor 优先统计 local shard，避免把 logical/global numel 当成单卡显存。
    """
    lt = t
    if hasattr(t, "to_local"):
        try:
            lt = t.to_local()
        except Exception:
            lt = t

    try:
        numel = lt.numel()
        nbytes = numel * lt.element_size()
        device = str(lt.device)
        dtype = str(lt.dtype)
    except Exception:
        numel = t.numel()
        nbytes = numel * t.element_size()
        device = str(getattr(t, "device", "unknown"))
        dtype = str(getattr(t, "dtype", "unknown"))

    return int(numel), int(nbytes), dtype, device


def build_optimizer(model, args, rank):
    params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(
            f"[OPT] building optimizer={args.optimizer}, "
            f"num_param_tensors={len(params)}, lr={args.lr}, "
            f"weight_decay={args.weight_decay}",
            flush=True,
        )

    opt_name = args.optimizer.lower()

    if opt_name == "adamw":
        kwargs = {
            "lr": args.lr,
            "betas": (args.adam_beta1, args.adam_beta2),
            "eps": args.adam_eps,
            "weight_decay": args.weight_decay,
        }

        foreach = parse_optional_bool(args.optimizer_foreach)
        fused = parse_optional_bool(args.optimizer_fused)

        # foreach/fused 在不同 torch / torch_npu 版本上兼容性不同，所以做成可选。
        if foreach is not None:
            kwargs["foreach"] = foreach
        if fused is not None:
            kwargs["fused"] = fused

        try:
            return torch.optim.AdamW(params, **kwargs)
        except TypeError as e:
            # 兼容旧版本：如果不支持 foreach/fused，就去掉重试。
            kwargs.pop("foreach", None)
            kwargs.pop("fused", None)
            if rank == 0:
                print(
                    f"[OPT] AdamW does not accept foreach/fused in this build, "
                    f"retry without them. original_error={repr(e)}",
                    flush=True,
                )
            return torch.optim.AdamW(params, **kwargs)

    if opt_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.sgd_momentum,
            weight_decay=args.weight_decay,
        )

    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def report_optimizer_state(optimizer, rank, tag, log_all_ranks=False):
    """
    打印 optimizer.state 中 tensor 的 local shard 规模。
    AdamW 第一次 step 前通常为 0，第一次 step 后会出现 exp_avg / exp_avg_sq / step。
    """
    state_entries = 0
    tensor_entries = 0
    local_numel = 0
    local_bytes = 0
    by_dtype_device = {}

    for state in optimizer.state.values():
        state_entries += 1
        for _, v in state.items():
            if torch.is_tensor(v):
                tensor_entries += 1
                n, b, dtype, device = tensor_local_numel_nbytes(v)
                local_numel += n
                local_bytes += b
                key = (dtype, device)
                old_n, old_b, old_cnt = by_dtype_device.get(key, (0, 0, 0))
                by_dtype_device[key] = (old_n + n, old_b + b, old_cnt + 1)

    if log_all_ranks or rank == 0:
        print(
            f"[OPT_STATE][rank={rank}] {tag}: "
            f"param_state_entries={state_entries}, "
            f"tensor_entries={tensor_entries}, "
            f"local_numel={local_numel:,}, "
            f"local_bytes={gib(local_bytes):.3f} GiB",
            flush=True,
        )
        for (dtype, device), (n, b, cnt) in sorted(by_dtype_device.items()):
            print(
                f"[OPT_STATE][rank={rank}]   dtype={dtype}, device={device}, "
                f"tensors={cnt}, local_numel={n:,}, local_bytes={gib(b):.3f} GiB",
                flush=True,
            )



# ----------------------------
# Run
# ----------------------------

def run(args, device_type, rank, local_rank, world_size, mesh):
    from transformers import AutoConfig

    device = torch.device(f"{device_type}:{local_rank}")
    param_dtype = parse_dtype(args.param_dtype)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    set_config_attr_if_exists(config, "use_cache", False)

    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = args.attn_impl

    # 新旧 transformers 兼容。即使新版提示 torch_dtype deprecated，也不影响这个测试。
    set_config_attr_if_exists(config, "torch_dtype", param_dtype)

    if rank == 0:
        print(
            f"[CONFIG] model_type={getattr(config, 'model_type', None)}, "
            f"hidden_size={getattr(config, 'hidden_size', None)}, "
            f"intermediate_size={getattr(config, 'intermediate_size', None)}, "
            f"num_attention_heads={getattr(config, 'num_attention_heads', None)}, "
            f"num_key_value_heads={getattr(config, 'num_key_value_heads', None)}, "
            f"vocab_size={getattr(config, 'vocab_size', None)}, "
            f"num_hidden_layers={getattr(config, 'num_hidden_layers', None)}, "
            f"attn_impl={getattr(config, '_attn_implementation', None)}",
            flush=True,
        )

    mem_report("start", device_type, rank, args.log_all_ranks, barrier=True)

    model = build_backbone_model_on_meta(config, param_dtype, rank)
    model.config.use_cache = False

    if args.gradient_checkpointing:
        if rank == 0:
            print("[CHECKPOINT] enabling gradient checkpointing", flush=True)
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            model.gradient_checkpointing_enable()

    mem_report("after meta AutoModel construct", device_type, rank, args.log_all_ranks, barrier=True)

    model = apply_fsdp2_use(model, mesh, args, rank)

    mem_report("after FSDP2 wrap on meta", device_type, rank, args.log_all_ranks, barrier=True)

    if rank == 0:
        print(f"[MATERIALIZE] model.to_empty(device={device})", flush=True)

    try:
        model.to_empty(device=device)
    except TypeError:
        model.to_empty(device)

    mem_report("after to_empty materialize local shards", device_type, rank, args.log_all_ranks, barrier=True)

    init_params_constant(model, rank)

    mem_report("after local param init", device_type, rank, args.log_all_ranks, barrier=True)

    report_params(model, rank)
    print_fsdp_units(model, rank)

    model.train()

    gc.collect()
    empty_cache(device_type)
    reset_peak(device_type)

    mem_report("before input allocation", device_type, rank, args.log_all_ranks, barrier=True)

    input_ids = torch.randint(
        low=0,
        high=int(getattr(config, "vocab_size")),
        size=(args.batch_size, args.seq_len),
        dtype=torch.long,
        device=device,
    )

    attention_mask = torch.ones(
        args.batch_size,
        args.seq_len,
        dtype=torch.long,
        device=device,
    )

    mem_report("after input allocation / before hooks", device_type, rank, args.log_all_ranks, barrier=True)

    hook_handles = add_memory_hooks(model, device_type, rank, args)

    mem_report("after hook registration / before forward", device_type, rank, args.log_all_ranks, barrier=True)

    if args.mode == "forward_only":
        if rank == 0:
            print("[RUN] forward_only: AutoModel backbone forward, no lm_head, no logits, no backward", flush=True)

        ctx = torch.no_grad()
        with ctx:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state
            dummy = hidden.float().mean()

        mem_report("after forward_only", device_type, rank, args.log_all_ranks, barrier=True)

        if rank == 0:
            print(f"[RESULT] dummy={float(dummy.detach().cpu())}", flush=True)

    elif args.mode in ("hidden_backward", "hidden_backward_optimizer"):
        if args.mode == "hidden_backward":
            if rank == 0:
                print("[RUN] hidden_backward: AutoModel backbone forward + dummy hidden loss backward", flush=True)
            optimizer = None
        else:
            if rank == 0:
                print(
                    "[RUN] hidden_backward_optimizer: AutoModel backbone forward "
                    "+ dummy hidden loss backward + optimizer.step",
                    flush=True,
                )

            mem_report("before optimizer construction", device_type, rank, args.log_all_ranks, barrier=True)
            optimizer = build_optimizer(model, args, rank)
            mem_report("after optimizer construction", device_type, rank, args.log_all_ranks, barrier=True)
            report_optimizer_state(
                optimizer,
                rank,
                "after optimizer construction",
                log_all_ranks=args.log_all_ranks,
            )

            optimizer.zero_grad(set_to_none=True)
            mem_report("after initial optimizer.zero_grad(set_to_none=True)", device_type, rank, args.log_all_ranks, barrier=True)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
            return_dict=True,
        )
        hidden = outputs.last_hidden_state
        loss = hidden.float().sum() * args.loss_scale

        mem_report("after forward / before backward", device_type, rank, args.log_all_ranks, barrier=True)

        loss.backward()

        if optimizer is None:
            mem_report("after backward", device_type, rank, args.log_all_ranks, barrier=True)
        else:
            mem_report("after backward / before optimizer.step", device_type, rank, args.log_all_ranks, barrier=True)
            report_optimizer_state(
                optimizer,
                rank,
                "before optimizer.step",
                log_all_ranks=args.log_all_ranks,
            )

            if args.reset_peak_before_optimizer_step:
                reset_peak(device_type)
                mem_report(
                    "before optimizer.step / after reset_peak",
                    device_type,
                    rank,
                    args.log_all_ranks,
                    barrier=True,
                )

            optimizer.step()

            mem_report("after optimizer.step", device_type, rank, args.log_all_ranks, barrier=True)
            report_optimizer_state(
                optimizer,
                rank,
                "after optimizer.step",
                log_all_ranks=args.log_all_ranks,
            )

            optimizer.zero_grad(set_to_none=args.zero_grad_set_to_none)
            mem_report(
                f"after optimizer.zero_grad(set_to_none={args.zero_grad_set_to_none})",
                device_type,
                rank,
                args.log_all_ranks,
                barrier=True,
            )
            report_optimizer_state(
                optimizer,
                rank,
                "after optimizer.zero_grad",
                log_all_ranks=args.log_all_ranks,
            )

    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    for h in hook_handles:
        h.remove()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--backend", type=str, default=None)

    parser.add_argument(
        "--mode",
        choices=["forward_only", "hidden_backward", "hidden_backward_optimizer"],
        default="forward_only",
    )

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)

    parser.add_argument("--param-dtype", type=str, default="fp32")
    parser.add_argument("--mp-param-dtype", type=str, default="fp16")
    parser.add_argument("--mp-reduce-dtype", type=str, default="fp16")
    parser.add_argument("--attn-impl", type=str, default="eager")

    parser.add_argument("--loss-scale", type=float, default=1.0e-8)

    parser.add_argument(
        "--optimizer",
        choices=["adamw", "sgd"],
        default="adamw",
        help="Only used by mode=hidden_backward_optimizer.",
    )
    parser.add_argument("--lr", type=float, default=1.0e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--adam-eps", type=float, default=1.0e-8)
    parser.add_argument("--sgd-momentum", type=float, default=0.0)
    parser.add_argument(
        "--optimizer-foreach",
        choices=["auto", "true", "false"],
        default="false",
        help="Use false by default to reduce extra foreach tensor-list temporary memory.",
    )
    parser.add_argument(
        "--optimizer-fused",
        choices=["auto", "true", "false"],
        default="auto",
        help="Keep auto by default for compatibility, especially on NPU.",
    )
    parser.add_argument(
        "--zero-grad-set-to-none",
        action="store_true",
        default=True,
        help="After optimizer.step(), call zero_grad(set_to_none=True).",
    )
    parser.add_argument(
        "--no-zero-grad-set-to-none",
        dest="zero_grad_set_to_none",
        action="store_false",
    )
    parser.add_argument(
        "--reset-peak-before-optimizer-step",
        action="store_true",
        help="Reset peak stats right before optimizer.step() to isolate step peak.",
    )

    parser.add_argument("--gradient-checkpointing", action="store_true")

    parser.add_argument("--wrap-embed", action="store_true", default=True)
    parser.add_argument("--no-wrap-embed", dest="wrap_embed", action="store_false")

    parser.add_argument("--wrap-final-norm", action="store_true", default=False)

    parser.add_argument("--reshard-after-forward", action="store_true", default=True)
    parser.add_argument(
        "--no-reshard-after-forward",
        dest="reshard_after_forward",
        action="store_false",
    )

    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")

    parser.add_argument(
        "--hook-style",
        choices=["none", "simple", "detailed"],
        default="detailed",
    )
    parser.add_argument("--hook-every", type=int, default=1)
    parser.add_argument("--hook-embed", action="store_true", default=True)
    parser.add_argument("--no-hook-embed", dest="hook_embed", action="store_false")
    parser.add_argument("--hook-final-norm", action="store_true", default=True)
    parser.add_argument("--no-hook-final-norm", dest="hook_final_norm", action="store_false")

    parser.add_argument("--log-all-ranks", action="store_true")
    parser.add_argument("--wrap-log-every", type=int, default=8)
    
    parser.add_argument("--wrap-root", action="store_true")

    args = parser.parse_args()

    device_type, rank, local_rank, world_size, mesh = init_dist(args)

    if rank == 0:
        print(
            f"[INIT] mode={args.mode}, device_type={device_type}, "
            f"world_size={world_size}, backend={dist.get_backend()}",
            flush=True,
        )
        print(f"[ARGS] {args}", flush=True)

    try:
        run(args, device_type, rank, local_rank, world_size, mesh)
    finally:
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception:
                pass
            dist.destroy_process_group()


if __name__ == "__main__":
    main()