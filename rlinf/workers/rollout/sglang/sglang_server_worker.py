from __future__ import annotations

import os
import socket
import time
import multiprocessing as mp
from typing import Literal, Optional

import requests
import yaml
from omegaconf import DictConfig, OmegaConf

from sglang.srt.server_args import ServerArgs

from rlinf.scheduler import Worker
from rlinf.scheduler import WorkerAddress
from rlinf.utils.placement import ModelParallelComponentPlacement


def _build_server_args_from_config(
    cfg: DictConfig,
    placement: ModelParallelComponentPlacement,
    rank_or_idx: int,
    *,
    host: str,
    base_port: int,
    weight_reload: Literal["sync", "cpu", None] = "sync",
) -> ServerArgs:
    """根据 config 和 placement 构造 ServerArgs，供 __init__ 与 create_group 复用。含 EP/DPA 参数。"""
    from rlinf.config import torch_dtype_from_precision

    rollout_cfg = cfg.rollout
    port = base_port + rank_or_idx
    use_cudagraph = not rollout_cfg.get("enforce_eager", True)
    # 与 SGLangWorker._init_engine 一致：sync 且未 validate 时用 dummy，否则 auto
    if weight_reload == "sync":
        load_format = (
            "auto"
            if getattr(rollout_cfg, "validate_weight", False)
            or getattr(rollout_cfg, "validate_weight_first_sync", False)
            else "dummy"
        )
    else:
        load_format = "auto"
    cuda_graph_max_bs = min(
        getattr(rollout_cfg, "cuda_graph_max_bs", rollout_cfg.max_running_requests),
        rollout_cfg.max_running_requests,
    )
    torch_compile_max_bs = min(
        rollout_cfg.sglang.torch_compile_max_bs,
        rollout_cfg.max_running_requests,
    )
    dtype_obj = torch_dtype_from_precision(rollout_cfg.model.precision)
    dtype_str = getattr(dtype_obj, "name", str(dtype_obj).replace("torch.", ""))

    sglang = getattr(rollout_cfg, "sglang", {})
    get_sglang = getattr(sglang, "get", lambda k, d=None: d)

    return ServerArgs(
        model_path=rollout_cfg.model.model_path,
        host=host,
        port=port,
        disable_cuda_graph=not use_cudagraph,
        cuda_graph_max_bs=cuda_graph_max_bs,
        tp_size=rollout_cfg.tensor_parallel_size,
        mem_fraction_static=rollout_cfg.gpu_memory_utilization,
        enable_memory_saver=use_cudagraph,
        enable_torch_compile=rollout_cfg.sglang.use_torch_compile,
        torch_compile_max_bs=torch_compile_max_bs,
        load_format=load_format,
        dtype=dtype_str,
        skip_tokenizer_init=not rollout_cfg.detokenize,
        decode_log_interval=rollout_cfg.sglang.decode_log_interval,
        attention_backend=rollout_cfg.sglang.attention_backend,
        log_level="warning",
        max_running_requests=rollout_cfg.max_running_requests,
        dp_size=1,
        tool_call_parser=rollout_cfg.sglang.get("tool_call_parser", None),
        ep_size=get_sglang("ep_size", 1),
        moe_a2a_backend=get_sglang("moe_a2a_backend", "none"),
        moe_runner_backend=get_sglang("moe_runner_backend", "auto"),
        deepep_mode=get_sglang("deepep_mode", "auto"),
    )


class SGLangServerWorker(Worker):
    """与 SGLangWorker 一致：接受 (config, placement [, weight_reload, config_rollout])，使用基类 create_group(cfg, placement)。"""

    def __init__(
        self,
        config: DictConfig,
        placement: ModelParallelComponentPlacement,
        weight_reload: Literal["sync", "cpu", None] = "sync",
        config_rollout: Optional[DictConfig] = None,
    ):
        super().__init__()
        rank = getattr(self, "_rank", None)
        if rank is None:
            rank = int(os.environ.get("RANK", 0))
        rank = int(rank)

        s = config.rollout.sglang.server
        host, base_port = s.host, s.port

        self._server_index = rank
        self.weight_reload = weight_reload
        self._server_args = _build_server_args_from_config(
            config, placement, rank, host=host, base_port=base_port, weight_reload=weight_reload
        )
        self._placement = placement
        self._cfg = config
        self._cfg_rollout = config_rollout if config_rollout is not None else getattr(config, "rollout", None)
        self._server_proc: Optional[mp.Process] = None
        self._return_logprobs = self._cfg_rollout.return_logprobs

    @staticmethod
    def _is_port_in_use(host: str, port: int) -> bool:
        """检测端口是否已被占用"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                s.bind(("127.0.0.1", port))
                return False
        except OSError:
            return True

    def server_start(self):
        if self._server_proc is not None and self._server_proc.is_alive():
            self.log_warning("SGLangServerWorker server already started")
            return

        server_args = self._server_args

        def _run():
            from rlinf.hybrid_engines.sglang.server.http_server import (
                launch_server,
            )

            launch_server(server_args, rollout_return_logprobs=self._return_logprobs)

        # daemon=False：launch_server 内会 spawn detokenizer/scheduler 子进程
        proc = mp.Process(target=_run, daemon=False)
        proc.start()
        self._server_proc = proc

    def get_server_address(self) -> str:
        host = self._server_args.host
        if host == "0.0.0.0":
            try:
                import ray.util
                host = ray.util.get_node_ip_address()
            except Exception:
                # 回退到本地回环地址
                host = "127.0.0.1"
        return f"{host}:{self._server_args.port}"

    def _base_url(self) -> str:
        return "http://" + self.get_server_address()

    @staticmethod
    def _placement_to_minimal_dict(
        placement: ModelParallelComponentPlacement,
    ) -> dict:
   
        cfg = getattr(placement, "_config", None)
        actor_group_name = (
            getattr(getattr(cfg, "actor", None), "group_name", None) if cfg else None
        )
        rollout_group_name = (
            getattr(getattr(cfg, "rollout", None), "group_name", None) if cfg else None
        )
        return {
            "rollout_dp_size": placement.rollout_dp_size,
            "actor_group_name": actor_group_name,
            "rollout_group_name": rollout_group_name,
            "actor_tp_size": getattr(placement, "actor_tp_size", 1),
            "actor_pp_size": getattr(placement, "actor_pp_size", 1),
            "actor_world_size": getattr(placement, "actor_world_size", 1),
            "rollout_tp_size": placement.rollout_tp_size,
            "rollout_world_size": placement.rollout_world_size,
            "placement_mode": placement.placement_mode.value
            if hasattr(placement.placement_mode, "value")
            else int(placement.placement_mode),
        }

    @staticmethod
    def _cfg_to_resolved_yaml(cfg: DictConfig) -> str:
        data = OmegaConf.to_container(cfg, resolve=True)
        return yaml.safe_dump(
            data, default_flow_style=False, allow_unicode=True, sort_keys=False
        )

    def sync_model_from_actor(self):
        url = self._base_url() + "/sync_hf_weight"
        resp = requests.post(url, timeout=1800)
        resp.raise_for_status()

    def offload_engine(self):
        assert self.weight_reload is not None, "offload_engine requires weight_reload is not None"
        url = self._base_url() + "/release_memory_occupation"
        resp = requests.post(url, json={}, timeout=600)
        resp.raise_for_status()

    def onload_engine(self):
        assert self.weight_reload == "cpu", "onload_engine requires weight_reload == 'cpu'"
        url = self._base_url() + "/resume_memory_occupation"
        resp = requests.post(url, json={}, timeout=600)
        resp.raise_for_status()

    async def http_server_stop(self):
        if self._server_proc is None:
            return
        if self._server_proc.is_alive():
            self._server_proc.terminate()
            self._server_proc.join(timeout=30)
        self._server_proc = None

    def init_rlinf_worker_http(
        self,
        parent_address: WorkerAddress,
        placement: ModelParallelComponentPlacement,
        cfg: DictConfig,
        weight_reload: str = "sync",
    ):
        url = self._base_url() + "/init_rlinf_worker"
        payload = {
            "parent_worker_name": parent_address.get_name(),
            "weight_reload": weight_reload,
            "placement": self._placement_to_minimal_dict(placement),
            "cfg_yaml": self._cfg_to_resolved_yaml(cfg),
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()

    def _wait_server_ready(self, timeout: float = 180, interval: float = 2) -> None:
        """轮询直到 SGLang HTTP 开始监听，再返回。避免 server_start() 后立刻调 init_rlinf_worker 出现 Connection refused。"""
        url = self._base_url()
        deadline = time.monotonic() + timeout
        last_err = None
        while time.monotonic() < deadline:
            try:
                resp = requests.get(url, timeout=5)
                if resp.status_code in (200, 404, 405):
                    return
            except (requests.exceptions.ConnectionError, OSError) as e:
                last_err = e
            time.sleep(interval)
        raise TimeoutError(
            f"SGLang server at {url} did not become ready within {timeout}s"
        ) from last_err

    def init_worker(self, start_http_server: bool = False, ready_channels=None):
        self._ready_channels = ready_channels  # list[Channel] 或 None
        if start_http_server:
            self.server_start()
            self._wait_server_ready()
        if self._placement is not None and self._cfg is not None:
            self.init_rlinf_worker_http(
                parent_address=self.worker_address,
                placement=self._placement,
                cfg=self._cfg,
                weight_reload=self.weight_reload,
            )


__all__ = ["SGLangServerWorker"]

