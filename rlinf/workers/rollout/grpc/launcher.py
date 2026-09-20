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

"""Launch and clean up the optional Ray-managed policy service."""

from contextlib import contextmanager
from typing import Iterator

from omegaconf import OmegaConf

from rlinf.scheduler import Cluster, ComponentPlacement

from .server_worker import PolicyServerWorker


@contextmanager
def policy_server_endpoint(
    cfg, cluster: Cluster, placement: ComponentPlacement
) -> Iterator[str]:
    """Yield an endpoint; external services remain owned by their deployer."""
    grpc_cfg = cfg.rollout.grpc
    if str(grpc_cfg.mode).lower() == "external":
        yield grpc_cfg.server_address
        return
    model_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.rollout.model, resolve=True)
    )
    serving_cfg = OmegaConf.create(
        {
            "model": model_cfg,
            "ckpt_path": cfg.runner.get("ckpt_path"),
            "server": {
                **OmegaConf.to_container(grpc_cfg.server, resolve=True),
                "policy_id": grpc_cfg.policy_id,
            },
        }
    )
    group = PolicyServerWorker.create_group(
        serving_cfg, grpc_cfg.startup_timeout_s
    ).launch(
        cluster,
        name=grpc_cfg.group_name,
        placement_strategy=placement.get_strategy("policy_server"),
    )
    try:
        group.init_server().wait()
        addresses = group.get_server_address().wait()
        if len(addresses) != 1:
            raise ValueError("gRPC evaluation currently requires one policy server")
        yield addresses[0]
    finally:
        group.shutdown().wait()
