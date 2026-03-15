# Copyright 2025 Bytedance Ltd. and/or its affiliates
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


import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, Literal, Optional, Union

import torch
from torch import distributed as dist

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather(tensor: "torch.Tensor", world_size: int) -> "torch.Tensor":
    """
    Gathers the tensor from all ranks and concats them along the first dim.
    """
    output_tensor = torch.empty(
        world_size * tensor.numel(), dtype=tensor.dtype, device="cuda"
    )
    dist.all_gather_into_tensor(output_tensor, tensor)
    return output_tensor.view(-1, *tensor.size()[1:])


def all_reduce(
    data: Union[int, float, list[Union[int, float]], "torch.Tensor"],
    op: Literal["mean", "sum", "max"] = "mean",
    group: Optional["ProcessGroup"] = None,
) -> Union[int, float, list[Union[int, float]]]:
    """
    Performs all reduce in the given process group.
    """
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float, device="cuda")

    reduce_ops = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
    }
    dist.all_reduce(data, op=reduce_ops[op], group=group)
    if op == "mean":  # ReduceOp.AVG is not supported by the NPU backend
        data /= dist.get_world_size(group=group)

    if data.numel() == 1:
        return data.item()
    else:
        return data.tolist()


@contextmanager
def main_process_first(local_only: bool = True) -> None:
    """
    A context manager for torch distributed environment to do something on the main process firstly.
    """
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        is_main_process = (
            int(os.getenv("LOCAL_RANK")) == 0
            if local_only
            else int(os.getenv("RANK")) == 0
        )
        try:
            if not is_main_process:
                dist.barrier()
            yield
        finally:
            if is_main_process:
                dist.barrier()
    else:
        yield


def execute_in_order(task: Callable, *, local_only: bool = True, **kwargs) -> Any:
    """
    Executes the task in the order of rank.
    """
    world_size = int(
        os.getenv("LOCAL_WORLD_SIZE", "1")
        if local_only
        else os.getenv("WORLD_SIZE", "1")
    )
    rank = int(os.getenv("LOCAL_RANK", "1") if local_only else os.getenv("RANK", "1"))
    if world_size > 1:
        dist.barrier()
        for i in range(world_size):
            if rank == i:
                result = task(**kwargs)
                dist.barrier()
            else:
                dist.barrier()

        return result
    else:
        return task(**kwargs)
