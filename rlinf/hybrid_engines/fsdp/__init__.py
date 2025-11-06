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


import torch
from packaging import version

try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor  # noqa: F401

if version.parse(torch.__version__) >= version.parse("2.6.0"):
    from torch.distributed.fsdp import (
        BackwardPrefetch,
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        OffloadPolicy,
        ShardingStrategy,
        fully_shard,
    )
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

elif version.parse(torch.__version__) >= version.parse("2.4.0"):
    from torch.distributed._composable.fsdp import (  # noqa: F401
        CPUOffloadPolicy,
        FSDPModule,
        MixedPrecisionPolicy,
        OffloadPolicy,
        fully_shard,
    )
    from torch.distributed.fsdp import BackwardPrefetch, ShardingStrategy  # noqa: F401
    from torch.distributed.fsdp.fully_sharded_data_parallel import (
        FullyShardedDataParallel as FSDP,  # noqa: F401
    )
else:
    raise ImportError(f"Unsupport torch version: {version.parse(torch.__version__)}")
