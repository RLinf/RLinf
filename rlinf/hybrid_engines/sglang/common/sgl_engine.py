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

import asyncio
import logging
import multiprocessing as mp
import os
import threading
from importlib.metadata import version

import uvloop
from packaging.version import parse
from sglang.srt.entrypoints.engine import Engine as _Engine
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    assert_pkg_version,
    set_prometheus_multiproc_dir,
    set_ulimit,
)

from rlinf.utils.patcher import Patcher

from .io_struct import (
    AbortGenerationInput,
    SyncHFWeightInput,
    TaskMethodInput,
)

HAVE_TRITON_CACHE_MANAGER = True
try:
    from sglang.srt.utils import maybe_set_triton_cache_manager
except ImportError:
    HAVE_TRITON_CACHE_MANAGER = False

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class Engine(_Engine):
    def sync_hf_weight(self):
        obj = SyncHFWeightInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.sync_hf_weight(obj))

    def abort_generation(self):
        obj = AbortGenerationInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.abort_generation(obj))

    def run_task_method(self, method_name: str, *args, **kwargs):
        """Run a method in the tokenizer manager."""
        obj = TaskMethodInput(method_name=method_name, args=args, kwargs=kwargs)
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.run_task_method(obj))


# disable signal handler registration in sglang
# patch to avoid issue https://github.com/sgl-project/sglang/issues/6723
def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = "0"
    os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["TORCH_NCCL_AVOID_RECORD_STREAMS"] = "1"
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Fix triton bugs
    if server_args.tp_size * server_args.dp_size > 1 and HAVE_TRITON_CACHE_MANAGER:
        # FIXME: remove this after https://github.com/triton-lang/triton/pull/4295 is used as a dependency.
        maybe_set_triton_cache_manager()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.2.3",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )

    # Set mp start method
    mp.set_start_method("spawn", force=True)


Patcher.clear()
Patcher.add_patch(
    "sglang.srt.entrypoints.engine._set_envs_and_config",
    "rlinf.hybrid_engines.sglang.common.sgl_engine._set_envs_and_config",
)
Patcher.add_patch(
    "sglang.srt.managers.tokenizer_manager.TokenizerManager",
    "rlinf.hybrid_engines.sglang.common.tokenizer_manager.TokenizerManager",
)

try:
    sglang_version = parse(version("sglang"))
except Exception as e:
    raise ValueError(f"sglang version not supported: {e}")
# for sglang < 0.5.0, support to get output_ids in result
# for sglang < 0.5.5, fix a bug in sglang to get correct output_ids
if sglang_version <= parse("0.5.5"):
    Patcher.add_patch(
        "sglang.srt.managers.detokenizer_manager.DetokenizerManager",
        "rlinf.hybrid_engines.sglang.common.detokenizer_manager.DetokenizerManager",
    )
    Patcher.add_patch(
        "sglang.srt.managers.detokenizer_manager.run_detokenizer_process",
        "rlinf.hybrid_engines.sglang.common.detokenizer_manager.run_detokenizer_process",
    )

Patcher.add_patch(
    "sglang.srt.managers.scheduler.run_scheduler_process",
    "rlinf.hybrid_engines.sglang.common.sgl_scheduler.run_scheduler_process",
)
Patcher.apply()
