# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.scheduler.placement import ComponentPlacement
from rlinf.utils.logging import get_logger

logger = get_logger()

_RLINF_SERVER_CONFIG_KEYS = {
    "api_base",
    "lora_adapter_name",
    "lora_name",
    "server_startup_timeout",
}
_RLINF_ROUTER_CONFIG_KEYS = {"router_node_rank"}


def _to_plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _normalize_key(key: Any) -> str:
    return str(key).replace("-", "_")


def _normalize_config_dict(value: Any) -> dict[str, Any]:
    config_dict = _to_plain(value) or {}
    if not isinstance(config_dict, dict):
        raise TypeError(f"Expected a dict config, got {type(config_dict).__name__}.")
    return {_normalize_key(key): val for key, val in config_dict.items()}


def _get_reward_model_cfg(cfg: DictConfig) -> DictConfig:
    return cfg.get("reward", {}).get("model", {})


def _get_sglang_server_args(reward_model_cfg: DictConfig) -> dict[str, Any]:
    return _normalize_config_dict(reward_model_cfg.get("sglang_server_args", {}))


def _get_sglang_router_args(reward_model_cfg: DictConfig) -> dict[str, Any]:
    return _normalize_config_dict(reward_model_cfg.get("sglang_router_args", {}))


def _infer_lora_adapter_name(
    server_args: dict[str, Any],
    served_model_name: str,
) -> str:
    adapter_name = server_args.get("lora_name") or server_args.get("lora_adapter_name")
    if adapter_name:
        return str(adapter_name)

    lora_paths = server_args.get("lora_paths")
    if isinstance(lora_paths, str):
        lora_paths = [lora_paths]
    if isinstance(lora_paths, list) and lora_paths:
        first_lora_path = lora_paths[0]
        if isinstance(first_lora_path, dict):
            adapter_name = first_lora_path.get("lora_name")
            if adapter_name:
                return str(adapter_name)
            first_lora_path = first_lora_path.get("lora_path")
        if isinstance(first_lora_path, str):
            if "=" in first_lora_path:
                return first_lora_path.split("=", 1)[0]
            return Path(first_lora_path).name

    return served_model_name


def should_launch_sglang_reward_server(cfg: DictConfig) -> bool:
    reward_cfg = cfg.get("reward", {})
    if not reward_cfg.get("use_reward_model", False):
        return False
    model_cfg = reward_cfg.get("model", {})
    if model_cfg.get("model_type") != "history_vlm":
        return False
    if str(model_cfg.get("inference_backend", "")).lower() != "sglang":
        return False
    return not bool(_get_sglang_server_args(model_cfg).get("api_base", None))


@dataclass
class SGLangRewardServerStack:
    """Ray-managed SGLang reward server and router groups."""

    server_group: Any
    router_group: Any
    api_base: str

    def stop(self) -> None:
        """Shut down router first, then the server group."""
        try:
            if self.router_group is not None:
                self.router_group.shutdown().wait()
                self.router_group = None
        finally:
            if self.server_group is not None:
                self.server_group.shutdown().wait()
                self.server_group = None


def _load_sglang_server_worker_classes():
    try:
        from rlinf.workers.rollout.sglang_server import (
            SGLangRouterWorker,
            SGLangServerWorker,
        )
    except ImportError as exc:
        raise ImportError(
            "Ray-managed SGLang reward server requires SGLang server/router "
            "workers from rlinf.workers.rollout.sglang_server."
        ) from exc
    return SGLangServerWorker, SGLangRouterWorker


def _infer_reward_server_tp_size(component_placement: ComponentPlacement) -> int:
    component_name = "reward_server"
    try:
        hardware_ranks = component_placement.get_hardware_ranks(component_name)
        world_size = component_placement.get_world_size(component_name)
    except AssertionError as exc:
        raise ValueError(
            "Auto-launched SGLang reward server requires "
            "cluster.component_placement.reward_server."
        ) from exc

    if not hardware_ranks:
        raise ValueError("reward_server placement must allocate at least one GPU.")
    if world_size <= 0 or len(hardware_ranks) % world_size != 0:
        raise ValueError(
            "reward_server placement must allocate the same number of GPUs to "
            "each server worker."
        )
    return len(hardware_ranks) // world_size


def _build_server_cfg(
    reward_model_cfg: DictConfig,
    component_placement: ComponentPlacement,
) -> DictConfig:
    model_path = reward_model_cfg.get("model_path")
    if not model_path:
        raise ValueError(
            "reward.model.model_path must be set to launch SGLang reward server."
        )

    inferred_tp_size = _infer_reward_server_tp_size(component_placement)
    server_args = _get_sglang_server_args(reward_model_cfg)
    explicit_tp_size = server_args.get("tp_size", None)
    if explicit_tp_size is not None and int(explicit_tp_size) != inferred_tp_size:
        raise ValueError(
            "reward.model.sglang_server_args.tp_size must match the number of "
            f"GPUs per reward_server worker ({inferred_tp_size}), got "
            f"{explicit_tp_size}."
        )

    served_model_name = (
        server_args.get("served_model_name")
        or Path(str(model_path)).name
        or "history_vlm_reward"
    )
    defaults: dict[str, Any] = {
        "model_path": model_path,
        "served_model_name": served_model_name,
        "tp_size": inferred_tp_size,
        "trust_remote_code": True,
        "enable_multimodal": True,
        "dtype": "bfloat16",
        "disable_cuda_graph": True,
        "attention_backend": "triton",
        "sampling_backend": "pytorch",
        "grammar_backend": "none",
    }
    lora_path = reward_model_cfg.get("lora_path")
    if lora_path:
        defaults["enable_lora"] = True
        if not server_args.get("lora_paths"):
            lora_name = _infer_lora_adapter_name(server_args, served_model_name)
            defaults["lora_paths"] = [f"{lora_name}={lora_path}"]
    server_cfg = {
        **defaults,
        **{
            key: value
            for key, value in server_args.items()
            if key not in _RLINF_SERVER_CONFIG_KEYS and value is not None
        },
    }
    server_cfg["tp_size"] = inferred_tp_size
    return OmegaConf.create(server_cfg)


def _build_router_cfg(reward_model_cfg: DictConfig) -> tuple[DictConfig, int]:
    router_args = _get_sglang_router_args(reward_model_cfg)
    router_node_rank = int(router_args.pop("router_node_rank", 0))
    defaults: dict[str, Any] = {
        "policy": "cache_aware",
        "log_level": "warn",
        "worker_startup_timeout_secs": 1800,
        "request_timeout_secs": 1800,
    }
    router_cfg = {
        **defaults,
        **{
            key: value
            for key, value in router_args.items()
            if key not in _RLINF_ROUTER_CONFIG_KEYS and value is not None
        },
    }
    return OmegaConf.create(router_cfg), router_node_rank


def launch_sglang_reward_server_stack(
    cfg: DictConfig,
    cluster: Cluster,
    component_placement: ComponentPlacement,
) -> SGLangRewardServerStack | None:
    """Launch Ray-managed SGLang reward server and router groups if needed."""
    if not should_launch_sglang_reward_server(cfg):
        return None

    model_cfg = _get_reward_model_cfg(cfg)
    server_args = _get_sglang_server_args(model_cfg)
    server_startup_timeout = float(server_args.get("server_startup_timeout", 600.0))
    server_cfg = _build_server_cfg(model_cfg, component_placement)
    router_cfg, router_node_rank = _build_router_cfg(model_cfg)
    SGLangServerWorker, SGLangRouterWorker = _load_sglang_server_worker_classes()

    server_group = None
    router_group = None
    try:
        server_group = SGLangServerWorker.create_group(
            config=cfg,
            sglang_cfg=server_cfg,
        ).launch(
            cluster=cluster,
            name=cfg.reward.get("server_group_name", "SGLangRewardServerGroup"),
            placement_strategy=component_placement.get_strategy("reward_server"),
        )
        router_group = SGLangRouterWorker.create_group(
            config=cfg,
            router_cfg=router_cfg,
        ).launch(
            cluster=cluster,
            name=cfg.reward.get("router_group_name", "SGLangRewardRouterGroup"),
            placement_strategy=NodePlacementStrategy(node_ranks=[router_node_rank]),
        )

        router_handle = router_group.init_router()
        server_handle = server_group.init_server()
        server_handle.wait()
        router_handle.wait()

        for server_url in server_group.get_server_url().wait():
            router_group.register_server(
                server_url,
                timeout=server_startup_timeout,
            ).wait()

        router_url = router_group.get_router_url().wait()[0].rstrip("/")
        api_base = f"{router_url}/v1"
        with open_dict(model_cfg):
            if model_cfg.get("sglang_server_args", None) is None:
                model_cfg.sglang_server_args = {}
            model_cfg.sglang_server_args.api_base = api_base

        logger.info("SGLang reward router is ready at %s", api_base)
        return SGLangRewardServerStack(
            server_group=server_group,
            router_group=router_group,
            api_base=api_base,
        )
    except Exception:
        stack = SGLangRewardServerStack(
            server_group=server_group,
            router_group=router_group,
            api_base="",
        )
        stack.stop()
        raise
