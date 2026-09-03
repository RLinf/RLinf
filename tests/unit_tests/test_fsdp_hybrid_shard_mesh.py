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

import pytest

from rlinf.hybrid_engines.fsdp.utils import resolve_fsdp_mesh


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


def test_hybrid_shard_defaults_to_one_shard_group_per_node():
    assert resolve_fsdp_mesh(
        16, sharding_strategy="hybrid_shard", local_world_size=8
    ) == ((2, 8), ("ddp", "fsdp"))


def test_hybrid_shard_size_overrides_the_node_local_default():
    assert resolve_fsdp_mesh(
        16, sharding_strategy="hybrid_shard", hybrid_shard_size=4, local_world_size=8
    ) == ((4, 4), ("ddp", "fsdp"))


def test_hybrid_shard_without_a_size_or_local_world_size_is_rejected():
    with pytest.raises(ValueError, match="LOCAL_WORLD_SIZE"):
        resolve_fsdp_mesh(16, sharding_strategy="hybrid_shard", local_world_size=0)


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
