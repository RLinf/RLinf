# Copyright 2025 The RLinf Authors.
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

from typing import Callable, Literal

import torch

from .fp8_utils import fp8Tensor


class Operation:
    local_idx = 0
    global_device = "cpu"
    global_tp = 1
    global_tpe = 1
    global_ep = 1
    global_pp = 1

    def execute(self):
        raise NotImplementedError()


class Load(Operation):
    def __init__(
        self,
        read: Callable[[str], torch.Tensor],
        name: str,
        dtype_trans: Literal["auto", "fp8_bf16", "bf16_bf16", "fp8_fp8"] = "auto",
    ):
        super().__init__()
        assert dtype_trans in ["auto", "fp8_bf16", "bf16_bf16", "fp8_fp8", "not_tensor"]
        self.read = read
        self.name = name
        self.dtype_trans = dtype_trans

    def execute(self):
        scale_inv_name = f"{self.name}_scale_inv"
        src_tensor = self.read(self.name)

        if self.dtype_trans == "auto":
            if src_tensor.dtype == torch.float8_e4m3fn:
                scale_inv = self.read(scale_inv_name)
                assert scale_inv.dtype == torch.float32, (
                    f"scale_inv is not float32 but {scale_inv.dtype}. name is {scale_inv_name}"
                )
                return fp8Tensor(src_tensor, scale_inv)
            else:
                return src_tensor
        elif self.dtype_trans == "fp8_bf16":
            # # trans to float32 and gpu, compute and trans to cpu
            # assert src_tensor.dtype == torch.float8_e4m3fn, f'fp8 tensor is not e4m3fn but {src_tensor.dtype}. name is {self.name}'
            # scale_inv = self.read(scale_inv_name)
            # assert scale_inv.dtype == torch.float32, f'scale_inv is not float32 but {scale_inv.dtype}. name is {scale_inv_name}'
            # src_tensor = src_tensor.to(device=Operation.global_device, dtype=torch.float32)
            # scale_inv = scale_inv.to(device=Operation.global_device)
            # with torch.cuda.device(Operation.global_device):
            #     return weight_dequant(src_tensor, scale_inv).to(dtype=torch.bfloat16)
            pass
        elif self.dtype_trans == "bf16_bf16":
            assert src_tensor.dtype == torch.bfloat16, (
                f"tensor is not bf16 but {src_tensor.dtype}. name is {self.name}"
            )
            return src_tensor
        elif self.dtype_trans == "fp8_fp8":
            assert src_tensor.dtype == torch.float8_e4m3fn, (
                f"fp8 tensor is not e4m3fn but {src_tensor.dtype}. name is {self.name}"
            )
            scale_inv = self.read(scale_inv_name)
            assert scale_inv.dtype == torch.float32, (
                f"scale_inv is not float32 but {scale_inv.dtype}. name is {scale_inv_name}"
            )
            return fp8Tensor(src_tensor, scale_inv)
        elif self.dtype_trans == "fp32_fp32":
            assert src_tensor.dtype == torch.float32, (
                f"tensor is not fp32 but {src_tensor.dtype}. name is {self.name}"
            )
            return src_tensor
        elif self.dtype_trans == "bf16_fp8":
            assert src_tensor.dtype == torch.bfloat16, (
                f"tensor is not bf16 but {src_tensor.dtype}. name is {self.name}"
            )
            return fp8Tensor.from_bf16(src_tensor)
        elif self.dtype_trans == "not_tensor":
            return src_tensor
        else:
            assert False, f"not supported load trans {self.dtype_trans}"


class SplitTpTpe(Operation):
    def __init__(
        self,
        src: Load,
        type_trans: Literal[
            "dense_fc1", "dense_fc1_glu", "dense_fc2", "moe_fc1_glu", "moe_fc2"
        ],
    ):
        super().__init__()
        self.src = src
        self.type_trans = type_trans

    def execute(self):
        src_tensor = self.src.execute()
        if self.type_trans.startswith("dense_") and Operation.global_tp == 1:
            return [src_tensor]
        if self.type_trans.startswith("moe_") and Operation.global_tpe == 1:
            return [src_tensor]

        if isinstance(src_tensor, fp8Tensor):
            raise RuntimeError("fp8 tensor cannot split tp or tpe")

        if self.type_trans == "dense_fc1":
            tgt_tensors = torch.chunk(src_tensor, Operation.global_tp, dim=0)
        elif self.type_trans == "dense_fc1_glu":
            viewed = src_tensor.view(2, -1, src_tensor.shape[-1])
            tgt_tensors = torch.chunk(viewed, Operation.global_tp, dim=1)
            tgt_tensors = [i.reshape(-1, src_tensor.shape[-1]) for i in tgt_tensors]
        elif self.type_trans == "dense_fc2":
            tgt_tensors = torch.chunk(src_tensor, Operation.global_tp, dim=1)
        elif self.type_trans == "moe_fc1_glu":
            viewed = src_tensor.view(2, -1, src_tensor.shape[-1])
            tgt_tensors = torch.chunk(viewed, Operation.global_tpe, dim=1)
            tgt_tensors = [i.reshape(-1, src_tensor.shape[-1]) for i in tgt_tensors]
        elif self.type_trans == "moe_fc2":
            tgt_tensors = torch.chunk(src_tensor, Operation.global_tpe, dim=1)
        else:
            assert False, f"bad type_trans: {self.type_trans}"
        return tgt_tensors


def de_dup(tensors):
    if len(tensors) == 1:
        return tensors[0]
    stack_tensor = torch.stack(tensors)
    for i in tensors[1:]:
        if not torch.allclose(tensors[0], i):
            tensor_max, _ = torch.max(stack_tensor, dim=0)
            tensor_min, _ = torch.min(stack_tensor, dim=0)
            # tensor_avg = torch.sum(stack_tensor, dim=0) / len(tensors)
            diff = tensor_max - tensor_min
            avg_diff = torch.sum(abs(diff)) / diff.numel()
            max_diff = torch.sum(abs(diff.view(-1)))
            assert False, (
                f"avg diff is {avg_diff}, max diff is {max_diff}, diff = {diff}, "
            )
    return tensors[0]


class MergeTpTpe(Operation):
    def __init__(
        self,
        srcs: list[Load],
        type_trans: Literal[
            "copy", "dense_fc1", "dense_fc1_glu", "dense_fc2", "moe_fc1_glu", "moe_fc2"
        ],
    ):
        super().__init__()
        self.srcs = srcs
        self.type_trans = type_trans

    def execute(self):
        src_tensors = [i.execute() for i in self.srcs]

        if self.type_trans == "copy":
            return de_dup(src_tensors)
        elif self.type_trans.startswith("dense_"):
            new_src_tensors = []
            dp = len(src_tensors) // Operation.global_tp
            for i in range(0, len(src_tensors), dp):
                new_src_tensors.append(de_dup(src_tensors[i : i + dp]))
            assert len(new_src_tensors) == Operation.global_tp
        elif self.type_trans.startswith("moe_"):
            new_src_tensors = []
            dpe = len(src_tensors) // Operation.global_tpe
            for i in range(0, len(src_tensors), dpe):
                new_src_tensors.append(de_dup(src_tensors[i : i + dpe]))
            assert len(new_src_tensors) == Operation.global_tpe

        src_tensors = new_src_tensors
        if len(src_tensors) == 1:
            return src_tensors[0]

        if isinstance(src_tensors[0], fp8Tensor):
            raise RuntimeError("fp8 tensor cannot merge tp or tpe")

        if self.type_trans == "dense_fc1":
            tgt_tensor = torch.cat(src_tensors, dim=0)
        elif self.type_trans == "dense_fc1_glu":
            tgt_tensor = torch.stack(src_tensors)
            tgt_tensor = tgt_tensor.view(
                Operation.global_tp, 2, -1, tgt_tensor.shape[-1]
            )
            tgt_tensor = tgt_tensor.transpose(0, 1)
            tgt_tensor = tgt_tensor.reshape(-1, tgt_tensor.shape[-1])
        elif self.type_trans == "dense_fc2":
            tgt_tensor = torch.cat(src_tensors, dim=1)
        elif self.type_trans == "moe_fc1_glu":
            tgt_tensor = torch.stack(src_tensors)
            tgt_tensor = tgt_tensor.view(
                Operation.global_tpe, 2, -1, tgt_tensor.shape[-1]
            )
            tgt_tensor = tgt_tensor.transpose(0, 1)
            tgt_tensor = tgt_tensor.reshape(-1, tgt_tensor.shape[-1])
        elif self.type_trans == "moe_fc2":
            tgt_tensor = torch.cat(src_tensors, dim=1)
        else:
            assert False, f"bad type_trans: {self.type_trans}"
        return tgt_tensor


class MergeTensors(Operation):
    def __init__(self, srcs: list[Load]):
        super().__init__()
        assert len(srcs) > 0
        self.srcs = srcs

    def execute(self):
        tensors = [i.execute() for i in self.srcs]
        if isinstance(tensors[0], fp8Tensor):
            return fp8Tensor.cat(tensors, dim=0)
        else:
            return torch.cat(tensors)


class CopyEquals(Operation):
    def __init__(self, srcs: list[Load]):
        super().__init__()
        assert len(srcs) > 0
        self.srcs = srcs

    def execute(self):
        tensors = [i.execute() for i in self.srcs]
        if isinstance(tensors[0], fp8Tensor):
            assert False
        else:
            return de_dup(tensors)


class MergeGlu(Operation):
    def __init__(self, src_gate: Load, src_fc1: Load):
        super().__init__()
        self.src_gate = src_gate
        self.src_fc1 = src_fc1

    def execute(self):
        tensor_gate = self.src_gate.execute()
        tensor_fc1 = self.src_fc1.execute()
        if isinstance(tensor_gate, fp8Tensor):
            return fp8Tensor.cat([tensor_gate, tensor_fc1], dim=0)
        else:
            return torch.cat([tensor_gate, tensor_fc1], dim=0)


class SplitGlu(Operation):
    def __init__(self, src: Load):
        super().__init__()
        self.src = src

    def execute(self):
        tensor_merged = self.src.execute()
        if isinstance(tensor_merged, fp8Tensor):
            assert False
        else:
            return torch.chunk(tensor_merged, 2, dim=0)


class MergeQKV(Operation):
    def __init__(
        self,
        src_q: Load,
        src_k: Load,
        src_v: Load,
        num_query_groups,
        num_attention_heads,
        head_dim,
        w_or_b,
    ):
        super().__init__()
        self.src_q = src_q
        self.src_k = src_k
        self.src_v = src_v
        self.num_query_groups = num_query_groups
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        assert w_or_b in ("w", "b")
        self.w_or_b = w_or_b

    def execute(self):
        tensor_q = self.src_q.execute()
        tensor_k = self.src_k.execute()
        tensor_v = self.src_v.execute()
        split_sizes = [self.num_attention_heads // self.num_query_groups, 1, 1]
        if isinstance(tensor_q, fp8Tensor):
            assert False
        if self.w_or_b == "w":
            assert (
                len(tensor_q.shape) == 2
                and len(tensor_k.shape) == 2
                and len(tensor_v.shape) == 2
            )
            assert tensor_k.shape[0] == tensor_v.shape[0]
            assert (
                tensor_q.shape[1] == tensor_k.shape[1]
                and tensor_q.shape[1] == tensor_v.shape[1]
            )
            if self.head_dim is None:
                assert tensor_q.shape[0] == tensor_q.shape[1], (
                    f"should tensor_q.shape[0] == tensor_q.shape[1], but tensor_q.shape: {tensor_q.shape}"
                )
                self.head_dim = tensor_q.shape[-1] // self.num_attention_heads
            # qkv weight
            tensor_q = tensor_q.view(
                self.num_query_groups, split_sizes[0], self.head_dim, tensor_q.shape[-1]
            )
            tensor_k = tensor_k.view(
                self.num_query_groups, split_sizes[1], self.head_dim, tensor_k.shape[-1]
            )
            tensor_v = tensor_v.view(
                self.num_query_groups, split_sizes[2], self.head_dim, tensor_v.shape[-1]
            )
            tgt_tensor = torch.cat([tensor_q, tensor_k, tensor_v], dim=1)
            return tgt_tensor.view(-1, tgt_tensor.shape[-1]).contiguous()
        else:
            assert (
                len(tensor_q.shape) == 1
                and len(tensor_k.shape) == 1
                and len(tensor_v.shape) == 1
            )
            assert tensor_k.shape[0] == tensor_v.shape[0]
            if self.head_dim is None:
                self.head_dim = tensor_q.shape[0] // self.num_attention_heads
            # qkv bias
            tensor_q = tensor_q.view(
                self.num_query_groups, split_sizes[0], self.head_dim
            )
            tensor_k = tensor_k.view(
                self.num_query_groups, split_sizes[1], self.head_dim
            )
            tensor_v = tensor_v.view(
                self.num_query_groups, split_sizes[2], self.head_dim
            )
            return (
                torch.cat([tensor_q, tensor_k, tensor_v], dim=1).view(-1).contiguous()
            )


class SplitQKV(Operation):
    def __init__(
        self, src: Load, num_query_groups, num_attention_heads, head_dim, w_or_b
    ):
        super().__init__()
        self.src = src
        self.num_query_groups = num_query_groups
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        assert w_or_b in ("w", "b")
        self.w_or_b = w_or_b

    def execute(self):
        tensor_merged = self.src.execute()
        if isinstance(tensor_merged, fp8Tensor):
            assert False
        split_sizes = [self.num_attention_heads // self.num_query_groups, 1, 1]
        num_channel_qkv = sum(split_sizes)
        if self.w_or_b == "w":
            # qkv weight
            assert len(tensor_merged.shape) == 2
            if self.head_dim is None:
                self.head_dim = tensor_merged.shape[1] // self.num_attention_heads
            qkvw = tensor_merged.view(
                self.num_query_groups,
                num_channel_qkv,
                self.head_dim,
                tensor_merged.shape[-1],
            )
            return [
                i.reshape(-1, i.shape[-1])
                for i in torch.split(qkvw, split_sizes, dim=1)
            ]
        else:
            # qkv bias
            assert len(tensor_merged.shape) == 1, (
                f"tensor_merged.shape: {tensor_merged.shape}"
            )
            if self.head_dim is None:
                self.head_dim = (
                    tensor_merged.shape[0] // num_channel_qkv // self.num_query_groups
                )
            qkvb = tensor_merged.view(
                self.num_query_groups, num_channel_qkv, self.head_dim
            )
            return [i.reshape(-1) for i in torch.split(qkvb, split_sizes, dim=1)]
