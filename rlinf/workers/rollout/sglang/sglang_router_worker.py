from __future__ import annotations

import asyncio
import copy
import multiprocessing as mp
import socket
from typing import Any, List, Optional

import requests
from omegaconf import OmegaConf
from sglang_router.router import Router as SmgRouter
from sglang_router.router_args import RouterArgs
from sglang_router.sglang_router_rs import PolicyType

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
    """从 rollout.sglang.router 读监听配置。"""

    def __init__(self, config: Any, placement: Any = None):
        super().__init__()
        self._router_process: Optional[mp.Process] = None
        self._server_base_urls: List[str] = []

        if placement is not None:
            cfg = config
            sglang_router = _get(_get(cfg, "rollout", {}).get("sglang"), "router") or {}
            self._config = {
                "host": _get(sglang_router, "host", "0.0.0.0"),
                "port": int(_get(sglang_router, "port", 30000)),
                "policy": _get(sglang_router, "policy", "cache_aware"),
            }
            if _get(sglang_router, "advertised_host"):
                self._config["advertised_host"] = _get(sglang_router, "advertised_host")
            self._cfg_rollout = _get(cfg, "rollout")
            self._placement = placement
            self.weight_reload = _get(_get(cfg, "rollout"), "weight_reload")
        else:
            raw = OmegaConf.to_container(config, resolve=True) if hasattr(config, "get") and not isinstance(config, dict) else config
            self._config = raw if isinstance(raw, dict) else {}
            self._cfg_rollout = self._config.get("rollout")
            self._placement = self._config.get("placement")
            self.weight_reload = self._config.get("weight_reload")

        self._return_logprobs = _get(self._cfg_rollout, "return_logprobs", False)
        sampling_params = _get(self._cfg_rollout, "sampling_params")
        if sampling_params is None:
            alg = _get(config, "algorithm")
            sampling_params = _get(alg, "sampling_params") if alg else None
        if sampling_params is not None and isinstance(sampling_params, dict):
            sampling_params = OmegaConf.create(sampling_params)
        self._sampling_params = (
            SGLangWorker.get_sampling_param_from_config(sampling_params)
            if sampling_params is not None
            else {"temperature": 0, "max_new_tokens": 256}
        )
        self.status_manager = RunningStatusManager()
        self._use_auto_scheduler = False
        self._collect_meta_stats = _get(self._cfg_rollout, "collect_meta_stats", False)
        if self._collect_meta_stats:
            self._init_meta_stats_collector()

    def _init_meta_stats_collector(self) -> None:
        async_stats_file = _get(self._cfg_rollout, "async_meta_stats_file", "sglang_meta_stats_async_router.jsonl")
        self.async_meta_stats_collector = MetaInfoStatsCollector(async_stats_file)
        self.async_batch_counter = 0

    def _collect_stats(self, engine_results: list) -> None:
        self.async_meta_stats_collector.collect_batch_stats(
            engine_results, self.async_batch_counter
        )
        self.async_batch_counter += 1

    @staticmethod
    def _policy_to_enum(name: str) -> PolicyType:
        name = name.lower()
        if name == "random":
            return PolicyType.Random
        if name == "round_robin":
            return PolicyType.RoundRobin
        if name == "power_of_two":
            return PolicyType.PowerOfTwo
        return PolicyType.CacheAware

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

    def get_router_address(self) -> str:
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

    def _router_base_url(self) -> str:
        return "http://" + self.get_router_address()

    def offload_engine(self) -> None:
        """向所有后端 server 发 /release_memory_occupation，与 SGLangWorker 语义对齐。"""
        assert self.weight_reload is not None, "offload_engine requires weight_reload is not None"
        for base_url in self._server_base_urls:
            url = base_url + "/release_memory_occupation"
            resp = requests.post(url, json={}, timeout=600)
            resp.raise_for_status()

    def _post_generate(self, payload: dict) -> dict:
        """向 Router 发 POST /generate，Router 会负载均衡到后端 server。"""
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
        """向 Router 发 /generate，等结果后返回。"""
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

    async def _async_generate_group(self, seq_group_info: SeqGroupInfo):
        """Generate a group of responses for a request (for GRPO-like behavior)."""
        if seq_group_info.num_aborted == 0:
            # No aborted sequences, repeat the input for group_size times
            assert seq_group_info.num_returned == 0
            seq_idx_list = list(range(seq_group_info.group_size))
            input_batch = [seq_group_info.input_ids] * seq_group_info.group_size
            sampling_params_list = [self._sampling_params] * seq_group_info.group_size
            image_data_list = [seq_group_info.image_data] * seq_group_info.group_size
        else:
            # Have aborted sequences (e.g., migrated from other engines)
            # Continue generation for the aborted group
            idx_aborted = seq_group_info.idx_aborted.copy()
            seq_idx_list: list[int] = []
            seq_group_info.idx_aborted.clear()
            input_batch: list[list[int]] = []
            sampling_params_list: list[dict] = []
            image_data_list: list = []
            for idx in idx_aborted:
                generated_ids: list[int] = seq_group_info.results[idx]["output_ids"]
                if len(generated_ids) >= self._sampling_params["max_new_tokens"]:
                    # avoid genererating for sequences that have already meet their max_new_tokens
                    self.log_warning(
                        f"SeqGroup {seq_group_info.id} idx {idx} "
                        f"has generated {len(generated_ids)} tokens, "
                        f"exceeding max_new_tokens={self._sampling_params['max_new_tokens']}, "
                        f"it will be truncatured."
                    )
                    result = seq_group_info.results[idx]
                    seq_group_info.results[idx] = None
                    result["meta_info"]["finish_reason"]["type"] = "length"
                    seq_group_info.record_sglang_result(idx, result)
                    continue
                seq_idx_list.append(idx)
                input_batch.append(seq_group_info.input_ids + generated_ids)
                params = self._sampling_params.copy()
                params["max_new_tokens"] -= len(generated_ids)
                sampling_params_list.append(params)
                image_data_list.append(seq_group_info.image_data)

        tasks = [
            asyncio.create_task(
                self.async_generate(
                    input_ids=input_ids,
                    image_data=image_data,
                    sampling_params=sampling_params,
                    return_logprob=self._return_logprobs,
                    request_info={
                        "seq_idx": seq_idx,
                    },
                )
            )
            for seq_idx, input_ids, sampling_params, image_data in zip(
                seq_idx_list,
                input_batch,
                sampling_params_list,
                image_data_list,
                strict=True,
            )
        ]
        for future in asyncio.as_completed(tasks):
            result, request_info = await future
            seq_group_info.record_sglang_result(
                request_info["seq_idx"], result, self._logger
            )

        return seq_group_info

    async def rollout(self, input_channel: Channel, output_channel: Channel):
        self.log_on_first_rank("Start generation...")
        request: RolloutRequest = input_channel.get()
        groups = request.to_seq_group_infos()
        async_wait_type = asyncio.ALL_COMPLETED
        with self.device_lock, self.worker_timer():
            num_residual = self.status_manager.num_seq_group
            assert num_residual == 0, (
                f"There are {num_residual} "
                f"sequence group{'' if num_residual == 1 else 's'} before rollout."
            )

            for group in groups:
                task = asyncio.create_task(self._async_generate_group(group))
                self.status_manager.add_task(group, task)

            all_rollout_results = []
            while pending := self.status_manager.get_running_tasks():
                done, pending = await asyncio.wait(pending, return_when=async_wait_type)
                returned_seq_groups: list[SeqGroupInfo] = [
                    task.result() for task in done
                ]
                for group in returned_seq_groups:
                    if group.all_completed:
                        rollout_result = RolloutResult.from_sglang_seq_group(
                            group,
                            self._return_logprobs,
                        )
                        all_rollout_results.append(rollout_result)
                        await output_channel.put(
                            item=rollout_result, async_op=True
                        ).async_wait()
                        self.status_manager.mark_done(group)
                    else:
                        self.status_manager.mark_aborted(group)

                if (
                    self._use_auto_scheduler
                    and self.status_manager.num_seq_group_running == 0
                ):
                    # rollout should not exit immediately when using auto scheduler
                    # because there might be migrations
                    # if so, `pending` will not be empty in while loop condition
                    await self.status_manager.wait_notification()

            self.status_manager.clear()

            if self._collect_meta_stats:
                self._collect_stats(all_rollout_results)

            if self.weight_reload is not None and self._placement is not None and (
                getattr(self._placement, "is_collocated", False)
                or getattr(self._placement, "is_auto", False)
            ):
                await asyncio.to_thread(self.offload_engine)
                if self._use_auto_scheduler and getattr(self, "_scheduler", None) is not None:
                    await self._scheduler.report_offloaded()

    async def generate_and_send(
        self,
        output_channel: Channel,
        channel_key: str,
        prompt_ids: list[int],
        sampling_params: Optional[dict] = None,
    ):
        final_sampling_params = self._sampling_params
        if sampling_params is not None and len(sampling_params) > 0:
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
        await output_channel.put(
            result_dict, key=channel_key, async_op=True
        ).async_wait()

    async def rollout_serverless(self, input_channel: Channel, output_channel: Channel):
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