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

Entrypoints:

* :func:`launch_sglang_router_and_server` — returns the worker-group
  handles.
* :func:`launch_sglang_api` — same launch, but also resolves the
  OpenAI-compatible API base URL for callers that only need an endpoint.

``launch_sglang_router_and_server``:

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

from typing import Sequence

from omegaconf import DictConfig

from rlinf.scheduler import (
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)
from rlinf.scheduler.placement import PlacementStrategy

from .router_worker import SGLangRouterWorker
from .server_worker import SGLangServerWorker


def launch_sglang_router_and_server(
    config: DictConfig,
    cluster: Cluster,
    rollout_hardware_ranks: list[int] | None,
    router_server_args: DictConfig,
    *,
    rollout_node_group: str | Sequence[str] | None = None,
    placement_strategy: PlacementStrategy | None = None,
    router_node_rank: int = 0,
) -> tuple["object | None", "object | None"]:
    """Launch the sglang server group and a single router worker.

    Args:
        config: Full RLinf ``DictConfig`` passed through to the worker
            groups for cluster/runtime context.
        cluster: Active :class:`rlinf.scheduler.Cluster`.
        rollout_hardware_ranks: Flat list of global hardware ranks the
            sglang engines should occupy — typically
            ``ComponentPlacement.get_hardware_ranks("<name>")``. The
            launcher repacks them into engines and asserts the ranks form a
            contiguous range (the prerequisite for a
            ``PackedPlacementStrategy``). Ignored when
            ``placement_strategy`` is provided or when ``launch_server``
            is ``False``.
        router_server_args: ``DictConfig`` carrying the
            ``{tensor_parallel_size, pipeline_parallel_size, server,
            router, group_name, router_group_name, launch_server,
            launch_router}`` keys the launcher consumes directly
            (typically ``config.rollout``).
        rollout_node_group: Optional node-group label(s) to forward to
            the repacked ``PackedPlacementStrategy``. Lets the caller
            preserve the node-group selection (e.g. when the rollout
            component is pinned to a non-default node group in
            ``cluster.component_placement``). Pass the same value
            originally given to ``component_placement[<name>].node_group``,
            e.g.  ``"rollout_gpu"`` or ``["a800", "4090"]``. Ignored when
            ``placement_strategy`` is provided.
        placement_strategy: A fully-built placement strategy to use as-is
            for the sglang server group — typically
            ``ComponentPlacement.get_strategy("<name>")``. When provided,
            this short-circuits the repacking path: it preserves the
            original node-group selection, process-to-resource mapping,
            and non-contiguous rank layouts (e.g. a
            ``FlexiblePlacementStrategy`` produced from a fancy
            ``placement: 0-1:0-3,3-5`` config). The caller is responsible
            for ensuring the strategy already encodes
            ``tensor_parallel_size * pipeline_parallel_size`` accelerators
            per process.
        router_node_rank: Cluster-global node rank on which to place the
            router. Defaults to node 0 (the head).

    Returns:
        ``(server_group, router_group)`` — two ``WorkerGroup`` handles
        (or ``None`` for any side gated off by config). The router's URL
        can be retrieved with ``router_group.get_router_url().wait()[0]``.
    """
    launch_server = router_server_args.get("launch_server", True)
    launch_router = router_server_args.get("launch_router", True)

    server_group = None
    if launch_server:
        if placement_strategy is not None:
            server_ps = placement_strategy
        else:
            assert rollout_hardware_ranks is not None, (
                "rollout_hardware_ranks must be provided when "
                "placement_strategy is not."
            )
            num_accelerators_per_engine = int(
                router_server_args.tensor_parallel_size
            ) * int(router_server_args.pipeline_parallel_size)
            ranks = sorted(int(r) for r in rollout_hardware_ranks)
            assert ranks, "rollout_hardware_ranks must not be empty."
            assert ranks == list(range(ranks[0], ranks[-1] + 1)), (
                f"rollout_hardware_ranks must be contiguous to repack as a "
                f"PackedPlacementStrategy; got {ranks}."
            )
            server_ps = PackedPlacementStrategy(
                start_hardware_rank=ranks[0],
                end_hardware_rank=ranks[-1],
                num_hardware_per_process=num_accelerators_per_engine,
                node_group=rollout_node_group,
            )

        server_group = SGLangServerWorker.create_group(
            config=config,
            sglang_cfg=router_server_args.server,
        ).launch(
            cluster=cluster,
            name=router_server_args.group_name,
            placement_strategy=server_ps,
        )

    # Bring up the router subprocess first WITHOUT any workers, then
    # spin up the servers in parallel. This decouples router startup
    # from server warmup: the router is reachable immediately, and we
    # register each server as soon as its /health goes 200.
    router_group = None
    if launch_router:
        router_ps = NodePlacementStrategy(node_ranks=[router_node_rank])
        router_group = SGLangRouterWorker.create_group(
            config=config,
            router_cfg=router_server_args.router,
        ).launch(
            cluster=cluster,
            name=router_server_args.router_group_name,
            placement_strategy=router_ps,
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


def get_sglang_api_url(server_group, router_group) -> str:
    """Resolve the OpenAI-compatible API base URL from launched groups.

    Prefers the router URL when a router was launched; otherwise falls
    back to the first server URL.
    """
    if router_group is not None:
        return router_group.get_router_url().wait()[0].rstrip("/")
    if server_group is not None:
        server_urls = server_group.get_server_url().wait()
        if server_urls:
            return str(server_urls[0]).rstrip("/")
    raise RuntimeError(
        "Unable to resolve SGLang API URL: neither router nor server group "
        "is available."
    )


def launch_sglang_api(
    config: DictConfig,
    cluster: Cluster,
    rollout_hardware_ranks: list[int] | None,
    router_server_args: DictConfig,
    *,
    rollout_node_group: str | Sequence[str] | None = None,
    placement_strategy: PlacementStrategy | None = None,
    router_node_rank: int = 0,
) -> tuple[str, "object | None", "object | None"]:
    """Launch SGLang and return ``(api_url, server_group, router_group)``.

    Same arguments as :func:`launch_sglang_router_and_server`. Use this when
    the caller needs a ready-to-use OpenAI-compatible base URL (for example
    ``reward.api.api_base``) rather than only the worker-group handles.
    """
    server_group, router_group = launch_sglang_router_and_server(
        config=config,
        cluster=cluster,
        rollout_hardware_ranks=rollout_hardware_ranks,
        router_server_args=router_server_args,
        rollout_node_group=rollout_node_group,
        placement_strategy=placement_strategy,
        router_node_rank=router_node_rank,
    )
    return get_sglang_api_url(server_group, router_group), server_group, router_group
