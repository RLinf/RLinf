from __future__ import annotations

import asyncio
import copy
import multiprocessing as mp
import socket
from typing import Any, List, Optional, Union

import requests
from omegaconf import OmegaConf
from sglang_router.router import Router as SmgRouter
from sglang_router.router_args import RouterArgs

from rlinf.data.io_struct import (
    RolloutRequest,
    RolloutResult,
    SeqGroupInfo,
)
from rlinf.scheduler import Channel, Worker
from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
from rlinf.workers.rollout.utils import MetaInfoStatsCollector, RunningStatusManager


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


class SGLangRouterWorker(Worker):

    def __init__(
        self,
        config: Any,
        placement: Any = None,
        server_group: Any = None,
        weight_reload: Optional[str] = None,
        config_rollout: Optional[Any] = None,
    ):
        super().__init__()
        self._router_process: Optional[mp.Process] = None
        self._server_base_urls: List[str] = []
        self._server_group = server_group

        if placement is not None:
            cfg = config
            sglang_router = _get(_get(cfg, "rollout", {}).get("sglang"), "router") or {}
            self._config = {
                "host": _get(sglang_router, "host", "0.0.0.0"),
                "port": int(_get(sglang_router, "port", 30000)),
                "policy": _get(sglang_router, "policy", "round_robin"),
            }
            if _get(sglang_router, "advertised_host"):
                self._config["advertised_host"] = _get(sglang_router, "advertised_host")
            self._cfg_rollout = _get(cfg, "rollout")
            self._placement = placement
            self.weight_reload = weight_reload if weight_reload is not None else _get(
                _get(cfg, "rollout"), "weight_reload"
            )
        else:
            raw = (
                OmegaConf.to_container(config, resolve=True)
                if hasattr(config, "get") and not isinstance(config, dict)
                else config
            )
            self._config = raw if isinstance(raw, dict) else {}
            self._cfg_rollout = self._config.get("rollout")
            self._placement = self._config.get("placement")
            self.weight_reload = self._config.get("weight_reload")

        self._cached_router_base_url: Optional[str] = None
        self._cached_server_addrs: List[str] = []

        if self._server_group is not None:
            cr = config_rollout or _get(config, "rollout")
            self._return_logprobs = _get(cr, "return_logprobs", False)
            sampling_params = _get(cr, "sampling_params")
            if sampling_params is None:
                sampling_params = _get(config, "algorithm", {}).get("sampling_params") if config else None
            if sampling_params is not None and isinstance(sampling_params, dict):
                sampling_params = OmegaConf.create(sampling_params)
            self._sampling_params = (
                SGLangWorker.get_sampling_param_from_config(sampling_params)
                if sampling_params is not None
                else {"temperature": 0, "max_new_tokens": 256}
            )
            self.status_manager = RunningStatusManager()
            self._use_auto_scheduler = False
            self._collect_meta_stats = _get(cr, "collect_meta_stats", False)
            if self._collect_meta_stats:
                self._init_meta_stats_collector()

    def _init_meta_stats_collector(self) -> None:
        async_stats_file = _get(
            self._cfg_rollout, "async_meta_stats_file", "sglang_meta_stats_async_router.jsonl"
        )
        self.async_meta_stats_collector = MetaInfoStatsCollector(async_stats_file)
        self.async_batch_counter = 0

    def _collect_stats(self, engine_results: list) -> None:
        self.async_meta_stats_collector.collect_batch_stats(
            engine_results, self.async_batch_counter
        )
        self.async_batch_counter += 1

    @staticmethod
    def _run_router_in_subprocess(args_dict: dict) -> None:
        router_args = RouterArgs(
            worker_urls=args_dict["worker_urls"],
            host=args_dict["host"],
            port=args_dict["port"],
            policy=args_dict["policy"],
        )
        router_args._validate_router_args()
        router = SmgRouter.from_args(router_args)
        router.start()

    def server_start(self, worker_urls: List[str]):
        if self._router_process is not None:
            return
        port = self._config["port"]
        try:
            import setproctitle

            setproctitle.setproctitle("sglang::router")
        except Exception:
            pass
        router_args = RouterArgs(
            worker_urls=worker_urls,
            host=self._config["host"],
            port=self._config["port"],
            policy=self._config["policy"],
        )
        router_args._validate_router_args()
        args_dict = {
            "worker_urls": router_args.worker_urls,
            "host": router_args.host,
            "port": router_args.port,
            "policy": router_args.policy,
        }
        self._router_process = mp.Process(
            target=SGLangRouterWorker._run_router_in_subprocess,
            args=(args_dict,),
            daemon=True,
        )
        self._router_process.start()
        self._server_base_urls = [
            u if u.startswith("http") else "http://" + u for u in worker_urls
        ]

    def _compute_router_address_str(self) -> str:
        port = self._config["port"]
        if self._config.get("advertised_host"):
            return f"{self._config['advertised_host']}:{port}"
        host = self._config["host"]
        if host == "0.0.0.0":
            try:
                host = socket.gethostbyname(socket.gethostname())
            except OSError:
                host = "127.0.0.1"
        return f"{host}:{port}"

    def get_server_address(self) -> str:
        """返回 Router 的 ``host:port`` 字符串。"""
        return self._compute_router_address_str()

    def _router_base_url(self) -> str:
        if self._cached_router_base_url is None:
            addr = self._compute_router_address_str()
            self._cached_router_base_url = (
                ("http://" + addr) if addr and not str(addr).startswith("http") else str(addr)
            )
        return self._cached_router_base_url

    def _post_generate(self, payload: dict) -> dict:
        url = self._router_base_url() + "/generate"
        resp = requests.post(url, json=payload, timeout=600)
        resp.raise_for_status()
        return resp.json()

    async def async_generate(
        self,
        prompt: list[str] | str | None = None,
        sampling_params: list[dict] | dict | None = None,
        input_ids: list[list[int]] | list[int] | None = None,
        image_data: list | None = None,
        return_logprob: list[bool] | bool | None = False,
        request_info: Any | None = None,
    ):
        payload: dict = {"return_logprob": bool(return_logprob)}
        if prompt is not None:
            payload["text"] = prompt
        if input_ids is not None:
            payload["input_ids"] = input_ids
        if sampling_params is not None:
            payload["sampling_params"] = sampling_params
        if image_data is not None and any(image_data):
            payload["image_data"] = image_data
        result = await asyncio.to_thread(self._post_generate, payload)
        return result, request_info

    async def generate_and_send(
        self,
        output_channel: Channel,
        channel_key: str,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
    ):
        assert self._server_group is not None
        final_sampling_params = self._sampling_params
        if sampling_params and len(sampling_params) > 0:
            final_sampling_params = copy.deepcopy(self._sampling_params)
            for key, value in sampling_params.items():
                final_sampling_params[key] = value
        result, _ = await self.async_generate(
            input_ids=prompt_ids,
            sampling_params=final_sampling_params,
            return_logprob=self._return_logprobs,
            request_info=None,
        )
        result_dict = {
            "output_ids": result["output_ids"],
            "finish_reason": result["meta_info"]["finish_reason"]["type"],
        }
        if self._return_logprobs:
            result_dict["logprobs"] = [
                item[0] for item in result["meta_info"]["output_token_logprobs"]
            ]
        await output_channel.put(result_dict, key=channel_key, async_op=True).async_wait()

    async def rollout_serverless(self, input_channel: Channel, output_channel: Channel):
        assert self._server_group is not None
        while True:
            rollout_request = await input_channel.get(async_op=True).async_wait()
            asyncio.create_task(
                self.generate_and_send(
                    output_channel=output_channel,
                    channel_key=rollout_request["channel_key"],
                    prompt_ids=rollout_request["prompt_ids"],
                    sampling_params=rollout_request.get("sampling_params", None),
                )
            )

    def init_worker(self, **kwargs):
        kwargs.pop("start_http_server", None)
        if self._server_group is None:
            return
        self._server_group.init_worker().wait()
        addrs = self._server_group.get_server_address().wait()
        worker_urls = [
            a if str(a).startswith("http") else "http://" + a
            for a in (addrs if isinstance(addrs, list) else [addrs])
        ]
        self._cached_server_addrs = [
            a.replace("http://", "").replace("https://", "") if isinstance(a, str) else str(a)
            for a in (addrs if isinstance(addrs, list) else [addrs])
        ]
        addr = self._compute_router_address_str()
        router_base_url = ("http://" + addr) if addr and not str(addr).startswith("http") else str(addr)
        self._cached_router_base_url = router_base_url
        self.server_start(worker_urls=worker_urls)

    def sync_model_from_actor(self):
        if self._server_group is None:
            return None
        print("sync_model_from_actor start in rollout worker")
        self._server_group.sync_model_from_actor().wait()
        return None

    def offload_engine(self):
        assert self._server_group is not None, "offload_engine only in router_server mode"
        return self._server_group.offload_engine()

    def onload_engine(self):
        assert self._server_group is not None, "onload_engine only in router_server mode"
        return self._server_group.onload_engine()


__all__ = ["SGLangRouterWorker"]
