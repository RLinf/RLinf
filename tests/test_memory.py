#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Usage examples:

1) Test pure FSDP2 all-gather/backward for one logical layer param size:
BACKEND=hccl torchrun --standalone --nproc_per_node=16 test_memory.py \
  --mode blob \
  --param-numel 487605248 \
  --param-dtype fp32 \
  --mp-param-dtype fp16 \
  --mp-reduce-dtype fp16

2) Test a real Qwen2DecoderLayer from the same model config:
BACKEND=hccl torchrun --standalone --nproc_per_node=16 test_memory.py \
  --mode qwen_layer \
  --model-path /path/to/your/Qwen2-32B-or-checkpoint \
  --batch-size 1 \
  --seq-len 1024 \
  --param-dtype fp32 \
  --input-dtype fp16 \
  --mp-param-dtype fp16 \
  --mp-reduce-dtype fp16

3) With activation checkpointing:
BACKEND=hccl torchrun --standalone --nproc_per_node=16 test_memory.py \
  --mode qwen_layer \
  --model-path /path/to/model \
  --batch-size 1 \
  --seq-len 1024 \
  --use-checkpoint
"""

import argparse
import gc
import inspect
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn


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


def gib(x):
    return x / 1024**3


def synchronize(device_type):
    api = get_device_api(device_type)
    try:
        api.synchronize()
    except Exception:
        pass


def reset_peak(device_type):
    api = get_device_api(device_type)
    try:
        api.reset_peak_memory_stats()
    except Exception:
        pass


def empty_cache(device_type):
    api = get_device_api(device_type)
    try:
        api.empty_cache()
    except Exception:
        pass


def mem_report(tag, device_type, rank, barrier=False):
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

    if rank == 0:
        print(
            f"[MEM][rank={rank}] {tag}: "
            f"alloc={gib(alloc):.3f} GiB, "
            f"reserved={gib(reserved):.3f} GiB, "
            f"max_alloc={gib(max_alloc):.3f} GiB, "
            f"max_reserved={gib(max_reserved):.3f} GiB",
            flush=True,
        )


def tensor_bytes(t):
    return t.numel() * t.element_size()


def report_param_grad_optim(model, optimizer, tag, rank):
    param_dev = 0
    param_cpu = 0
    grad_dev = 0
    grad_cpu = 0
    optim_dev = 0
    optim_cpu = 0

    for _, p in model.named_parameters():
        try:
            b = tensor_bytes(p)
            dev_type = p.device.type
            if dev_type in ("cuda", "npu"):
                param_dev += b
            elif dev_type == "cpu":
                param_cpu += b
        except Exception:
            pass

        if p.grad is not None:
            try:
                b = tensor_bytes(p.grad)
                dev_type = p.grad.device.type
                if dev_type in ("cuda", "npu"):
                    grad_dev += b
                elif dev_type == "cpu":
                    grad_cpu += b
            except Exception:
                pass

    if optimizer is not None:
        for state in optimizer.state.values():
            for v in state.values():
                if hasattr(v, "numel") and hasattr(v, "element_size"):
                    b = tensor_bytes(v)
                    dev_type = v.device.type
                    if dev_type in ("cuda", "npu"):
                        optim_dev += b
                    elif dev_type == "cpu":
                        optim_cpu += b

    if rank == 0:
        print(
            f"[PGO][rank={rank}] {tag}: "
            f"param_dev={gib(param_dev):.3f} GiB, "
            f"param_cpu={gib(param_cpu):.3f} GiB, "
            f"grad_dev={gib(grad_dev):.3f} GiB, "
            f"grad_cpu={gib(grad_cpu):.3f} GiB, "
            f"optim_dev={gib(optim_dev):.3f} GiB, "
            f"optim_cpu={gib(optim_cpu):.3f} GiB",
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

    try:
        from torch.distributed.device_mesh import init_device_mesh
        mesh = init_device_mesh(device_type, (world_size,))
    except Exception as e:
        if rank == 0:
            print(f"[WARN] init_device_mesh failed: {repr(e)}. Use mesh=None.", flush=True)
        mesh = None

    return device_type, rank, local_rank, world_size, mesh


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

    if MixedPrecisionPolicy is not None and (mp_param_dtype is not None or mp_reduce_dtype is not None):
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
        "reshard_after_forward": args.reshard_after_forward,
    }
    if mesh is not None:
        kwargs["mesh"] = mesh
    if mp_policy is not None:
        kwargs["mp_policy"] = mp_policy
    if offload_policy is not None:
        kwargs["offload_policy"] = offload_policy

    try:
        return fully_shard(module, **kwargs)
    except TypeError as e:
        # Some PyTorch versions use slightly different argument names.
        if "mp_policy" in kwargs:
            kwargs["mixed_precision"] = kwargs.pop("mp_policy")
        try:
            return fully_shard(module, **kwargs)
        except TypeError:
            raise e


class ParamBlob(nn.Module):
    """
    A single huge parameter module.

    This mode tries to isolate:
    - FSDP2 pre-forward / pre-backward all-gather
    - full parameter materialization
    - gradient materialization
    - reduce-scatter-related temporary memory

    It does NOT model attention/MLP activation.
    """

    def __init__(self, numel, dtype, device):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(numel, dtype=dtype, device=device))
        # Avoid expensive random normal over many GB. Fill is enough.
        with torch.no_grad():
            self.weight.fill_(1.0 / 1024)

    def forward(self):
        # This touches the whole parameter and creates dense gradient.
        # The scale avoids huge loss value.
        return self.weight.float().sum() * 1.0e-8


def run_blob(args, device_type, rank, local_rank, world_size, mesh):
    device = torch.device(f"{device_type}:{local_rank}")
    param_dtype = parse_dtype(args.param_dtype)

    mem_report("start", device_type, rank, barrier=True)

    model = ParamBlob(args.param_numel, dtype=param_dtype, device=device)
    mem_report("after construct full ParamBlob before FSDP2", device_type, rank, barrier=True)

    model = apply_fsdp2(model, mesh, args)
    mem_report("after FSDP2 wrap ParamBlob", device_type, rank, barrier=True)

    gc.collect()
    empty_cache(device_type)
    reset_peak(device_type)
    mem_report("before forward", device_type, rank, barrier=True)

    loss = model()
    mem_report("after forward / before backward", device_type, rank, barrier=True)
    report_param_grad_optim(model, None, "before backward", rank)

    loss.backward()

    mem_report("after backward", device_type, rank, barrier=True)
    report_param_grad_optim(model, None, "after backward", rank)


def make_causal_mask(batch_size, seq_len, dtype, device):
    # Shape expected by Qwen2DecoderLayer is usually [bsz, 1, q_len, kv_len].
    min_val = torch.finfo(dtype).min
    mask = torch.full((seq_len, seq_len), min_val, dtype=dtype, device=device)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, seq_len, seq_len)
    return mask


def run_qwen_layer(args, device_type, rank, local_rank, world_size, mesh):
    try:
        from transformers import AutoConfig
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding
    except Exception as e:
        raise RuntimeError(
            "qwen_layer mode needs transformers with Qwen2 support. "
            f"Import failed: {repr(e)}"
        )

    device = torch.device(f"{device_type}:{local_rank}")

    param_dtype = parse_dtype(args.param_dtype)
    input_dtype = parse_dtype(args.input_dtype)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config.use_cache = False

    # Prefer eager attention for debuggability and stable memory attribution.
    # If your environment requires flash/sdpa, change this.
    if hasattr(config, "_attn_implementation"):
        config._attn_implementation = args.attn_impl

    if rank == 0:
        print(
            f"[CONFIG] hidden_size={getattr(config, 'hidden_size', None)}, "
            f"intermediate_size={getattr(config, 'intermediate_size', None)}, "
            f"num_attention_heads={getattr(config, 'num_attention_heads', None)}, "
            f"num_key_value_heads={getattr(config, 'num_key_value_heads', None)}, "
            f"vocab_size={getattr(config, 'vocab_size', None)}, "
            f"num_hidden_layers={getattr(config, 'num_hidden_layers', None)}",
            flush=True,
        )

    mem_report("start", device_type, rank, barrier=True)

    with torch.no_grad():
        layer = Qwen2DecoderLayer(config, layer_idx=0).to(device=device, dtype=param_dtype)
        rotary = Qwen2RotaryEmbedding(config=config).to(device=device)

    mem_report("after construct full Qwen2DecoderLayer before FSDP2", device_type, rank, barrier=True)

    layer = apply_fsdp2(layer, mesh, args)

    mem_report("after FSDP2 wrap Qwen2DecoderLayer", device_type, rank, barrier=True)

    batch_size = args.batch_size
    seq_len = args.seq_len
    hidden_size = config.hidden_size

    hidden_states = torch.randn(
        batch_size,
        seq_len,
        hidden_size,
        dtype=input_dtype,
        device=device,
        requires_grad=True,
    )

    position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

    attention_mask = make_causal_mask(batch_size, seq_len, input_dtype, device)

    # Newer Qwen2 passes rotary embeddings explicitly.
    try:
        position_embeddings = rotary(hidden_states, position_ids)
    except Exception:
        position_embeddings = None

    def layer_forward(h):
        kwargs = {
            "hidden_states": h,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "output_attentions": False,
            "use_cache": False,
        }

        sig = inspect.signature(layer.forward)
        if "position_embeddings" in sig.parameters and position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings

        out = layer(**kwargs)
        if isinstance(out, tuple):
            out = out[0]
        return out

    gc.collect()
    empty_cache(device_type)
    reset_peak(device_type)
    mem_report("before forward", device_type, rank, barrier=True)

    if args.use_checkpoint:
        from torch.utils.checkpoint import checkpoint
        out = checkpoint(layer_forward, hidden_states, use_reentrant=False)
    else:
        out = layer_forward(hidden_states)

    # Keep loss simple; this exercises full backward through the layer.
    loss = out.float().square().mean()

    mem_report("after forward / before backward", device_type, rank, barrier=True)
    report_param_grad_optim(layer, None, "before backward", rank)

    loss.backward()

    mem_report("after backward", device_type, rank, barrier=True)
    report_param_grad_optim(layer, None, "after backward", rank)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["blob", "qwen_layer"], required=True)

    parser.add_argument("--backend", type=str, default=None)

    # FSDP2 / dtype options
    parser.add_argument("--param-dtype", type=str, default="fp32")
    parser.add_argument("--input-dtype", type=str, default="fp16")
    parser.add_argument("--mp-param-dtype", type=str, default="none")
    parser.add_argument("--mp-reduce-dtype", type=str, default="none")
    parser.add_argument("--reshard-after-forward", action="store_true", default=True)
    parser.add_argument("--no-reshard-after-forward", dest="reshard_after_forward", action="store_false")
    parser.add_argument("--cpu-offload", action="store_true")
    parser.add_argument("--pin-memory", action="store_true")

    # blob mode
    parser.add_argument("--param-numel", type=int, default=487_605_248)

    # qwen_layer mode
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--attn-impl", type=str, default="eager")
    parser.add_argument("--use-checkpoint", action="store_true")

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
        if args.mode == "blob":
            run_blob(args, device_type, rank, local_rank, world_size, mesh)
        elif args.mode == "qwen_layer":
            if args.model_path is None:
                raise ValueError("--model-path is required for qwen_layer mode.")
            run_qwen_layer(args, device_type, rank, local_rank, world_size, mesh)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
