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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.config import validate_fsdp_weight_sync_cfg
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import resolve_fsdp_mesh


def _strategy_with(sharding_strategy):
    return SimpleNamespace(
        cfg=OmegaConf.create({"fsdp_config": {"sharding_strategy": sharding_strategy}})
    )


@pytest.mark.parametrize(
    "sharding_strategy", ["full_shard", "shard_grad_op", "no_shard"]
)
def test_non_hybrid_strategies_keep_the_one_dimensional_mesh(sharding_strategy):
    assert resolve_fsdp_mesh(8, sharding_strategy=sharding_strategy) == (
        (8,),
        ("fsdp",),
    )


def test_hybrid_shard_replicates_on_dim0_and_shards_on_dim1():
    # PyTorch reads mesh dim 0 as the replicate group and dim 1 as the shard
    # group for both FSDP1 HYBRID_SHARD and FSDP2 fully_shard, so the order and
    # the pairing of names to sizes are part of the contract, not cosmetic.
    mesh_shape, mesh_dim_names = resolve_fsdp_mesh(
        16, sharding_strategy="hybrid_shard", hybrid_shard_size=8
    )

    assert mesh_dim_names == ("ddp", "fsdp")
    assert mesh_shape == (2, 8)
    assert dict(zip(mesh_dim_names, mesh_shape)) == {"ddp": 2, "fsdp": 8}


def test_hybrid_shard_size_sets_the_shard_degree():
    assert resolve_fsdp_mesh(
        16, sharding_strategy="hybrid_shard", hybrid_shard_size=4
    ) == ((4, 4), ("ddp", "fsdp"))


@pytest.mark.parametrize("hybrid_shard_size", [-1, 0, None])
def test_hybrid_shard_without_a_configured_size_is_rejected(hybrid_shard_size):
    # Deriving the shard degree per rank (e.g. from LOCAL_WORLD_SIZE) would let
    # ranks disagree on the mesh shape, and init_device_mesh is collective.
    with pytest.raises(ValueError, match="requires"):
        resolve_fsdp_mesh(
            16,
            sharding_strategy="hybrid_shard",
            hybrid_shard_size=hybrid_shard_size,
        )


def test_hybrid_shard_size_that_does_not_divide_the_world_is_rejected():
    with pytest.raises(ValueError, match="must divide"):
        resolve_fsdp_mesh(12, sharding_strategy="hybrid_shard", hybrid_shard_size=8)


@pytest.mark.parametrize(
    ("world_size", "hybrid_shard_size"),
    [
        (8, 8),  # replicate degree 1: nothing is replicated
        (8, 1),  # shard degree 1: nothing is sharded
    ],
)
def test_hybrid_shard_degenerating_to_a_single_group_is_rejected(
    world_size, hybrid_shard_size
):
    # A silent fall back to full_shard here would let a mis-sized multi-node run
    # train under a different strategy than the config asks for.
    with pytest.raises(ValueError, match="both shard and replicate"):
        resolve_fsdp_mesh(
            world_size,
            sharding_strategy="hybrid_shard",
            hybrid_shard_size=hybrid_shard_size,
        )


@pytest.mark.parametrize(
    "sharding_strategy", ["full_shard", "shard_grad_op", "no_shard"]
)
def test_weight_sync_accepts_single_shard_group_strategies(sharding_strategy):
    FSDPStrategyBase._assert_single_shard_group(_strategy_with(sharding_strategy))


def test_weight_sync_rejects_hybrid_shard():
    # The actor<->inference handshake advertises each rank's shard as
    # rank * ceil(numel / world_size), which is wrong once shards repeat along
    # the replicate dim, so it must fail loudly instead of syncing wrong bytes.
    with pytest.raises(NotImplementedError, match="hybrid_shard"):
        FSDPStrategyBase._assert_single_shard_group(_strategy_with("hybrid_shard"))


@pytest.mark.parametrize("sharding_strategy", ["hybrid-shard", "HYBRID_SHARD", ""])
def test_unknown_sharding_strategy_is_rejected(sharding_strategy):
    # Only FSDP1 routes the name through get_sharding_strategy(), so without this
    # check a typo would take the 1-D branch and silently train under full_shard.
    with pytest.raises(ValueError, match="Unknown fsdp_config.sharding_strategy"):
        resolve_fsdp_mesh(16, sharding_strategy=sharding_strategy)


def _run_cfg(actor_strategy, inference_strategy, load_from_actor=True):
    return OmegaConf.create(
        {
            "actor": {"fsdp_config": {"sharding_strategy": actor_strategy}},
            "inference": {
                "load_from_actor": load_from_actor,
                "fsdp_config": {"sharding_strategy": inference_strategy},
            },
        }
    )


def test_weight_sync_cfg_accepts_full_shard_on_both_sides():
    validate_fsdp_weight_sync_cfg(_run_cfg("full_shard", "full_shard"))


@pytest.mark.parametrize(
    ("actor_strategy", "inference_strategy"),
    [
        ("hybrid_shard", "full_shard"),  # actor overridden inline
        ("full_shard", "hybrid_shard"),  # shared fsdp group file edited
        ("hybrid_shard", "hybrid_shard"),
    ],
)
def test_weight_sync_cfg_rejects_hybrid_shard_on_either_side(
    actor_strategy, inference_strategy
):
    # An asymmetric config is the dangerous one: the handshake guard would raise
    # on one side while the other blocks in the paired send/recv.
    with pytest.raises(ValueError, match="hybrid_shard"):
        validate_fsdp_weight_sync_cfg(_run_cfg(actor_strategy, inference_strategy))


def test_weight_sync_cfg_ignores_runs_that_do_not_load_from_the_actor():
    validate_fsdp_weight_sync_cfg(
        _run_cfg("hybrid_shard", "full_shard", load_from_actor=False)
    )
