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

"""Top-level helper that wires placement → server group → router group.

Single entrypoint :func:`launch_sglang_router_and_server` keeps the
launch script short. The function:

1. Takes the flat list of hardware ranks the sglang engines should
   occupy (typically ``ComponentPlacement.get_hardware_ranks("<name>")``)
   and builds a ``PackedPlacementStrategy`` whose process width is
   ``tensor_parallel_size * pipeline_parallel_size`` GPUs — one sglang
   engine per process.
2. Launches a ``SGLangServerWorker`` group on that strategy.
3. Spawns each server's sglang process and collects its URL.
4. Launches a single ``SGLangRouterWorker`` on the chosen node and
   points it at the server URLs.

"""

from __future__ import annotations

from omegaconf import DictConfig

from rlinf.scheduler import (
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)

from .router_launcher import SGLangRouterWorker
from .server_launcher import SGLangServerWorker


def launch_sglang_router_and_server(
    config: DictConfig,
    cluster: Cluster,
    rollout_hardware_ranks: list[int],
    *,
    router_node_rank: int = 0,
) -> tuple["object | None", "object | None"]:
    """Launch the sglang server group and a single router worker.

    Args:
        config: Full RLinf ``DictConfig``. The launcher reads
            ``config.rollout.{tensor_parallel_size, pipeline_parallel_size,
            server, router, group_name, router_group_name, launch_server,
            launch_router}`` directly — the caller doesn't break the
            rollout block apart.
        cluster: Active :class:`rlinf.scheduler.Cluster`.
        rollout_hardware_ranks: Flat list of global hardware ranks the
            sglang engines should occupy — typically
            ``ComponentPlacement.get_hardware_ranks("<name>")``. The
            launcher repacks them into engines and asserts the ranks form a
            contiguous range (the prerequisite for a
            ``PackedPlacementStrategy``). Ignored when
            ``launch_server`` is ``False``.
        router_node_rank: Cluster-global node rank on which to place the
            router. Defaults to node 0 (the head).

    Returns:
        ``(server_group, router_group)`` — two ``WorkerGroup`` handles
        (or ``None`` for any side gated off by config). The router's URL
        can be retrieved with ``router_group.get_router_url().wait()[0]``.
    """
    rollout_cfg = config.rollout
    launch_server = rollout_cfg.get("launch_server", True)
    launch_router = rollout_cfg.get("launch_router", True)

    server_group = None
    if launch_server:
        num_accelerators_per_engine = int(rollout_cfg.tensor_parallel_size) * int(
            rollout_cfg.pipeline_parallel_size
        )
        ranks = sorted(int(r) for r in rollout_hardware_ranks)
        assert ranks, "rollout_hardware_ranks must not be empty."
        assert ranks == list(range(ranks[0], ranks[-1] + 1)), (
            f"rollout_hardware_ranks must be contiguous to repack as a "
            f"PackedPlacementStrategy; got {ranks}."
        )
        rollout_placement = PackedPlacementStrategy(
            start_hardware_rank=ranks[0],
            end_hardware_rank=ranks[-1],
            num_hardware_per_process=num_accelerators_per_engine,
        )

        server_group = SGLangServerWorker.create_group(
            config=config,
            sglang_cfg=rollout_cfg.server,
        ).launch(
            cluster=cluster,
            name=rollout_cfg.group_name,
            placement_strategy=rollout_placement,
        )

    # Bring up the router subprocess first WITHOUT any workers, then
    # spin up the servers in parallel. This decouples router startup
    # from server warmup: the router is reachable immediately, and we
    # register each server as soon as its /health goes 200.
    router_group = None
    if launch_router:
        router_placement = NodePlacementStrategy(node_ranks=[router_node_rank])
        router_group = SGLangRouterWorker.create_group(
            config=config,
            router_cfg=rollout_cfg.router,
        ).launch(
            cluster=cluster,
            name=rollout_cfg.router_group_name,
            placement_strategy=router_placement,
        )

    router_handle = router_group.init_router() if router_group is not None else None
    server_handle = server_group.init_server() if server_group is not None else None

    if server_handle is not None:
        server_handle.wait()
    if router_handle is not None:
        router_handle.wait()

    # Register every server with the running router. Done from the
    # driver (serialized) so the order of attached workers is stable —
    # if you want concurrent registration, call from N workers instead.
    if server_group is not None and router_group is not None:
        server_urls = server_group.get_server_url().wait()
        for url in server_urls:
            router_group.register_server(url).wait()

    return server_group, router_group
