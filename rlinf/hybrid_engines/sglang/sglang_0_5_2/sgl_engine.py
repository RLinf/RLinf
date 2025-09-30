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
import atexit
import logging
import multiprocessing as mp
import os
import random
import signal
import threading
import time
from typing import Dict, Optional, Tuple

import uvloop
import zmq
from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine as _Engine
from sglang.srt.managers.data_parallel_controller import (
    run_data_parallel_controller_process,
)
from sglang.srt.managers.detokenizer_manager import run_detokenizer_process
from sglang.srt.managers.multi_tokenizer_mixin import MultiTokenizerRouter

# from sglang.srt.managers.scheduler import run_scheduler_process
# from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.srt.managers.template_manager import TemplateManager
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.srt.utils import (
    assert_pkg_version,
    configure_logger,
    get_bool_env_var,
    get_zmq_socket,
    is_cuda,
    kill_process_tree,
    launch_dummy_health_check_server,
    prepare_model_and_tokenizer,
    set_prometheus_multiproc_dir,
    set_ulimit,
)

from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ComponentPlacement

from .io_struct import OffloadReqInput, SyncHFWeightInput, SyncWeightInput
from .sgl_scheduler import run_scheduler_process
from .tokenizer_manager import TokenizerManager

# Fix a bug of Python threading
setattr(threading, "_register_atexit", lambda *args, **kwargs: None)

logger = logging.getLogger(__name__)
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

_is_cuda = is_cuda()


class Engine(_Engine):
    def __init__(
        self,
        parent_address: WorkerAddress,
        placement: ComponentPlacement,
        config: DictConfig,
        dp_rank: int,
        **kwargs,
    ):
        """
        The arguments of this function is the same as `sglang/srt/server_args.py::ServerArgs`.
        Please refer to `ServerArgs` for the documentation.
        """
        if "server_args" in kwargs:
            # Directly load server_args
            server_args = kwargs["server_args"]
        else:
            # Construct server_args from kwargs
            if "log_level" not in kwargs:
                # Do not print logs by default
                kwargs["log_level"] = "error"
            server_args = ServerArgs(**kwargs)

        # Shutdown the subprocesses automatically when the program exits
        atexit.register(self.shutdown)

        # Allocate ports for inter-process communications
        self.port_args = PortArgs.init_new(server_args)

        # Launch subprocesses
        tokenizer_manager, template_manager, scheduler_info = _launch_subprocesses(
            parent_address=parent_address,
            placement=placement,
            config=config,
            dp_rank=dp_rank,
            server_args=server_args,
            port_args=self.port_args,
        )

        self.server_args = server_args
        self.tokenizer_manager = tokenizer_manager
        self.scheduler_info = scheduler_info
        self.template_manager = template_manager

        context = zmq.Context(2)
        self.send_to_rpc = get_zmq_socket(
            context, zmq.DEALER, self.port_args.rpc_ipc_name, True
        )

    def offload_model_weights(self):
        """Offload model weights to meta."""
        obj = OffloadReqInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.tokenizer_manager.offload_model_weights(obj, None)
        )

    def sync_hf_weight(self):
        obj = SyncHFWeightInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.sync_hf_weight(obj))

    def sync_weight(self):
        obj = SyncWeightInput()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.tokenizer_manager.sync_weight(obj))


def _set_envs_and_config(server_args: ServerArgs):
    # Set global environments
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["NCCL_CUMEM_ENABLE"] = str(int(server_args.enable_symm_mem))
    if not server_args.enable_symm_mem:
        os.environ["NCCL_NVLS_ENABLE"] = str(int(server_args.enable_nccl_nvls))
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "4"
    os.environ["CUDA_MODULE_LOADING"] = "AUTO"
    # flashinfer uses this environment variable for various kernels from MoE to quant kernels
    os.environ["TRTLLM_ENABLE_PDL"] = "1"

    # Can also be passed as argument
    os.environ["SGLANG_RUN_ID"] = (
        f"sglang-run-{time.time()}-{random.randint(0, 100000000)}"
    )

    # Set prometheus env vars
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    # Set ulimit
    set_ulimit()

    # Check flashinfer version
    if server_args.attention_backend == "flashinfer":
        assert_pkg_version(
            "flashinfer_python",
            "0.3.0",
            "Please uninstall the old version and "
            "reinstall the latest version by following the instructions "
            "at https://docs.flashinfer.ai/installation.html.",
        )
    if _is_cuda and not get_bool_env_var("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK"):
        assert_pkg_version(
            "sgl-kernel",
            "0.3.8",
            "Please reinstall the latest version with `pip install sgl-kernel --force-reinstall`",
        )

    if True:  # Keep this check for internal code compatibility
        # Register the signal handler.
        # The child processes will send SIGQUIT to this process when any error happens
        # This process then clean up the whole process tree
        # Note: This sigquit handler is used in the launch phase, and may be replaced by
        # the running_phase_sigquit_handler in the tokenizer manager after the grpc server is launched.
        def launch_phase_sigquit_handler(signum, frame):
            logger.error(
                "Received sigquit from a child process. It usually means the child failed."
            )
            kill_process_tree(os.getpid())

        signal.signal(signal.SIGQUIT, launch_phase_sigquit_handler)

    # Set mp start method
    mp.set_start_method("spawn", force=True)


def _launch_subprocesses(
    parent_address: WorkerAddress,
    placement: ComponentPlacement,
    config: DictConfig,
    dp_rank: int,
    server_args: ServerArgs,
    port_args: Optional[PortArgs] = None,
) -> Tuple[TokenizerManager, TemplateManager, Dict]:
    """
    Launch the TokenizerManager in the main process, the Scheduler in a subprocess, and the DetokenizerManager in another subprocess.
    """

    assert server_args.pp_size == 1, (
        "RLinf currently only supports and validates pp_size=1."
    )

    # Configure global environment
    configure_logger(server_args)
    server_args.check_server_args()
    _set_envs_and_config(server_args)

    # Allocate ports for inter-process communications
    if port_args is None:
        port_args = PortArgs.init_new(server_args)
        logger.info(f"{server_args=}")

    # If using model from www.modelscope.cn, first download the model.
    server_args.model_path, server_args.tokenizer_path = prepare_model_and_tokenizer(
        server_args.model_path, server_args.tokenizer_path
    )

    scheduler_procs = []
    if server_args.dp_size == 1:
        memory_saver_adapter = TorchMemorySaverAdapter.create(
            enable=server_args.enable_memory_saver
        )
        scheduler_pipe_readers = []

        nnodes_per_tp_group = max(server_args.nnodes // server_args.pp_size, 1)
        tp_size_per_node = server_args.tp_size // nnodes_per_tp_group
        tp_rank_range = range(
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group),
            tp_size_per_node * (server_args.node_rank % nnodes_per_tp_group + 1),
        )

        pp_size_per_node = max(server_args.pp_size // server_args.nnodes, 1)
        pp_rank_range = range(
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group),
            pp_size_per_node * (server_args.node_rank // nnodes_per_tp_group + 1),
        )

        for pp_rank in pp_rank_range:
            for tp_rank in tp_rank_range:
                reader, writer = mp.Pipe(duplex=False)
                gpu_id = (
                    server_args.base_gpu_id
                    + ((pp_rank % pp_size_per_node) * tp_size_per_node)
                    + (tp_rank % tp_size_per_node) * server_args.gpu_id_step
                )
                moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
                proc = mp.Process(
                    target=run_scheduler_process,
                    args=(
                        parent_address,
                        placement,
                        config,
                        server_args.tp_size * server_args.pp_size,
                        tp_rank + pp_rank * server_args.pp_size,
                        server_args,
                        port_args,
                        gpu_id,
                        tp_rank,
                        moe_ep_rank,
                        pp_rank,
                        None,
                        writer,
                        None,
                    ),
                )

                with memory_saver_adapter.configure_subprocess():
                    proc.start()
                scheduler_procs.append(proc)
                scheduler_pipe_readers.append(reader)
    else:
        # Launch the data parallel controller
        reader, writer = mp.Pipe(duplex=False)
        scheduler_pipe_readers = [reader]
        proc = mp.Process(
            target=run_data_parallel_controller_process,
            args=(server_args, port_args, writer),
        )
        proc.start()
        scheduler_procs.append(proc)

    if server_args.node_rank >= 1:
        # In multi-node cases, non-zero rank nodes do not need to run tokenizer or detokenizer,
        # so they can just wait here.

        for reader in scheduler_pipe_readers:
            data = reader.recv()
            assert data["status"] == "ready"

        if os.getenv("SGLANG_BLOCK_NONZERO_RANK_CHILDREN") == "0":
            # When using `Engine` as a Python API, we don't want to block here.
            return None, None, None

        launch_dummy_health_check_server(
            server_args.host, server_args.port, server_args.enable_metrics
        )

        for proc in scheduler_procs:
            proc.join()
            logger.error(
                f"Scheduler or DataParallelController {proc.pid} terminated with {proc.exitcode}"
            )
        return None, None, None

    # Launch detokenizer process
    detoken_proc = mp.Process(
        target=run_detokenizer_process,
        args=(
            server_args,
            port_args,
        ),
    )
    detoken_proc.start()
    if server_args.tokenizer_worker_num > 1:
        # Launch multi-tokenizer router
        tokenizer_manager = MultiTokenizerRouter(server_args, port_args)

        # Initialize templates
        template_manager = None
    else:
        # Launch tokenizer process
        tokenizer_manager = TokenizerManager(server_args, port_args)

        # Initialize templates
        template_manager = TemplateManager()
        template_manager.initialize_templates(
            tokenizer_manager=tokenizer_manager,
            model_path=server_args.model_path,
            chat_template=server_args.chat_template,
            completion_template=server_args.completion_template,
        )

    # Wait for the model to finish loading
    scheduler_infos = []
    for i in range(len(scheduler_pipe_readers)):
        try:
            data = scheduler_pipe_readers[i].recv()
        except EOFError:
            logger.error(
                f"Rank {i} scheduler is dead. Please check if there are relevant logs."
            )
            scheduler_procs[i].join()
            logger.error(f"Exit code: {scheduler_procs[i].exitcode}")
            raise

        if data["status"] != "ready":
            raise RuntimeError(
                "Initialization failed. Please see the error messages above."
            )
        scheduler_infos.append(data)

    # Assume all schedulers have the same scheduler_info
    scheduler_info = scheduler_infos[0]
    tokenizer_manager.max_req_input_len = scheduler_info["max_req_input_len"]
    return tokenizer_manager, template_manager, scheduler_info
