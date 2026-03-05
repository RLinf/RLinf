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

import pytest
from omegaconf import DictConfig

from rlinf.scheduler import (
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)
from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)


class FakeCluster:
    """Minimal Cluster stub exposing just the APIs placement strategies rely on."""

    def __init__(self, nodes: list[NodeInfo], node_groups: dict[str, NodeGroupInfo]):
        self._nodes = nodes
        self._node_groups = node_groups

    def get_node_group(
        self, label: str | None = NodeGroupInfo.DEFAULT_GROUP_LABEL
    ) -> NodeGroupInfo:
        resolved = NodeGroupInfo.DEFAULT_GROUP_LABEL if label is None else str(label)
        assert resolved in self._node_groups, (
            f"Node group '{resolved}' not found. Available groups: {list(self._node_groups.keys())}."
        )
        return self._node_groups[resolved]

    def get_node_info(self, node_rank: int) -> NodeInfo:
        return self._nodes[node_rank]

    def get_node_num_accelerators(self, node_rank: int) -> int:
        return self._nodes[node_rank].num_accelerators

    def get_node_id_from_accel_id(self, accel_id: int) -> int:
        node_info = self.get_node_group().get_node_by_hardware_rank(accel_id)
        assert node_info is not None, (
            f"Accelerator rank {accel_id} does not belong to any node in the default group."
        )
        return node_info.node_rank

    def global_accel_id_to_local_accel_id(self, accel_id: int) -> int:
        local_rank = self.get_node_group().get_local_hardware_rank(accel_id)
        assert local_rank is not None, (
            f"Accelerator rank {accel_id} does not map to a local rank in the default group."
        )
        return local_rank

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_accelerators(self) -> int:
        return sum(node.num_accelerators for node in self._nodes)

    @property
    def num_accelerators_in_cluster(self) -> int:
        return self.num_accelerators


def _make_node_info(node_rank: int, num_accelerators: int) -> NodeInfo:
    resources: list[HardwareResource] = []
    if num_accelerators > 0:
        resources.append(
            HardwareResource(
                type=Accelerator.HW_TYPE,
                infos=[
                    HardwareInfo(type=Accelerator.HW_TYPE, model="NV_GPU:Mock")
                    for _ in range(num_accelerators)
                ],
            )
        )
    return NodeInfo(
        node_labels=[],
        node_rank=node_rank,
        ray_id=f"ray-node-{node_rank}",
        node_ip=f"10.0.0.{node_rank + 1}",
        num_cpus=32,
        python_interpreter_path="/usr/bin/python3",
        default_env_vars={},
        env_vars={},
        hardware_resources=resources,
    )


def create_fake_cluster(
    num_nodes: int,
    accelerators_per_node: int | list[int],
    extra_group_mapping: dict[str, list[int]] | None = None,
) -> FakeCluster:
    if isinstance(accelerators_per_node, int):
        accel_counts = [accelerators_per_node] * num_nodes
    else:
        accel_counts = list(accelerators_per_node)
        assert len(accel_counts) == num_nodes, (
            "Length of accelerators_per_node list must match num_nodes."
        )

    nodes = [_make_node_info(i, accel_counts[i]) for i in range(num_nodes)]
    node_groups: dict[str, NodeGroupInfo] = {}

    default_group = NodeGroupInfo(label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes)
    if default_group.hardware_type is None and any(
        node.num_accelerators for node in nodes
    ):
        default_group.hardware_type = Accelerator.HW_TYPE
    node_groups[default_group.label] = default_group

    node_only_group = NodeGroupInfo(
        label=NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL,
        nodes=nodes,
    )
    node_only_group.hardware_type = None
    node_groups[node_only_group.label] = node_only_group

    if extra_group_mapping:
        for label, node_indices in extra_group_mapping.items():
            group_nodes = [nodes[idx] for idx in node_indices]
            group = NodeGroupInfo(label=label, nodes=group_nodes)
            if group.hardware_type is None and any(
                node.num_accelerators for node in group_nodes
            ):
                group.hardware_type = Accelerator.HW_TYPE
            node_groups[label] = group

    return FakeCluster(nodes, node_groups)


def mock_cluster(
    num_nodes: int, num_accelerators_per_node: int | list[int]
) -> FakeCluster:
    return create_fake_cluster(num_nodes, num_accelerators_per_node)


class TestPackedPlacementStrategy:
    def test_spans_requested_hardware(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        strategy = PackedPlacementStrategy(0, 3)

        placements = strategy.get_placement(cluster, isolate_accelerator=True)
        placements = sorted(placements, key=lambda p: p.rank)

        assert len(placements) == 4
        for idx, placement in enumerate(placements):
            assert placement.cluster_node_rank == 0
            assert placement.local_accelerator_rank == idx
            assert placement.visible_accelerators == [str(idx)]
            assert placement.isolate_accelerator is True
        assert {placement.local_world_size for placement in placements} == {4}

    def test_chunked_allocation(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        strategy = PackedPlacementStrategy(
            start_hardware_rank=0,
            end_hardware_rank=3,
            num_hardware_per_process=2,
        )

        placements = strategy.get_placement(cluster, isolate_accelerator=True)
        placements = sorted(placements, key=lambda p: p.rank)

        assert len(placements) == 2
        assert placements[0].visible_accelerators == ["0", "1"]
        assert placements[1].visible_accelerators == ["2", "3"]
        assert {placement.local_world_size for placement in placements} == {2}

    def test_strided_allocation(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=8)
        strategy = PackedPlacementStrategy(
            start_hardware_rank=0,
            end_hardware_rank=7,
            stride=2,
            num_hardware_per_process=2,
        )

        placements = strategy.get_placement(cluster, isolate_accelerator=True)
        placements = sorted(placements, key=lambda p: p.rank)

        assert len(placements) == 4
        expected = [["0", "2"], ["1", "3"], ["4", "6"], ["5", "7"]]
        assert [placement.visible_accelerators for placement in placements] == expected

    def test_invalid_stride_raises(self):
        with pytest.raises(AssertionError):
            PackedPlacementStrategy(0, 3, stride=5)

    def test_invalid_num_hardware_per_process_raises(self):
        with pytest.raises(AssertionError):
            PackedPlacementStrategy(0, 2, num_hardware_per_process=4)

    def test_invalid_master_gpu_range(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        strategy = PackedPlacementStrategy(4, 7)
        with pytest.raises(AssertionError):
            strategy.get_placement(cluster, isolate_accelerator=True)


class TestFlexiblePlacementStrategy:
    def test_single_process_single_gpu(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0]])

        placements = strategy.get_placement(cluster)
        assert len(placements) == 1
        placement = placements[0]
        assert placement.cluster_node_rank == 0
        assert placement.local_accelerator_rank == 0
        assert placement.visible_accelerators == ["0"]
        assert placement.local_world_size == 1

    def test_single_process_multiple_gpus_sorted(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[2, 1, 3]])

        placements = strategy.get_placement(cluster)
        assert placements[0].visible_accelerators == ["1", "2", "3"]
        assert placements[0].local_accelerator_rank == 1

    def test_multiple_processes_same_node(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0], [1], [2]])

        placements = strategy.get_placement(cluster)
        assert len(placements) == 3
        assert [p.local_rank for p in placements] == [0, 1, 2]
        assert {p.local_world_size for p in placements} == {3}

    def test_multiple_nodes(self):
        cluster = create_fake_cluster(num_nodes=3, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0], [4], [8]])

        placements = strategy.get_placement(cluster)
        assert [p.cluster_node_rank for p in placements] == [0, 1, 2]
        assert {p.local_world_size for p in placements} == {1}

    def test_cross_node_gpu_ids_raises(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0, 4]])
        with pytest.raises(AssertionError, match="same node"):
            strategy.get_placement(cluster)

    def test_out_of_range_gpu_id_raises(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0], [8]])
        with pytest.raises(AssertionError, match="out of range"):
            strategy.get_placement(cluster)

    def test_duplicate_gpu_ids_in_process_raises(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0, 0]])
        with pytest.raises(AssertionError, match="must be unique"):
            strategy.get_placement(cluster)

    def test_empty_gpu_ids_raises(self):
        with pytest.raises(AssertionError, match="must not be empty"):
            FlexiblePlacementStrategy([])


class TestNodePlacementStrategy:
    def test_single_node_multiple_processes(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=[2, 2])
        strategy = NodePlacementStrategy(
            [0, 0], NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL
        )

        placements = strategy.get_placement(cluster)
        assert len(placements) == 2
        assert [p.cluster_node_rank for p in placements] == [0, 0]
        assert [p.local_rank for p in placements] == [0, 1]
        assert {p.local_world_size for p in placements} == {2}

    def test_multiple_nodes(self):
        cluster = create_fake_cluster(num_nodes=3, accelerators_per_node=1)
        strategy = NodePlacementStrategy(
            [0, 1, 1, 2], NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL
        )

        placements = strategy.get_placement(cluster)
        assert [p.cluster_node_rank for p in placements] == [0, 1, 1, 2]
        assert [p.local_rank for p in placements] == [0, 0, 1, 0]
        assert [p.local_world_size for p in placements] == [1, 2, 2, 1]

    def test_no_accelerators(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=0)
        strategy = NodePlacementStrategy([0], NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL)

        placements = strategy.get_placement(cluster)
        assert placements[0].local_accelerator_rank == -1
        assert placements[0].visible_accelerators == []

    def test_isolate_accelerator_false(self):
        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
        strategy = NodePlacementStrategy([0], NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL)

        placements = strategy.get_placement(cluster, isolate_accelerator=False)
        assert placements[0].isolate_accelerator is False
        assert placements[0].visible_accelerators == ["0", "1"]

    def test_invalid_node_rank_raises(self):
        cluster = create_fake_cluster(num_nodes=2, accelerators_per_node=1)
        strategy = NodePlacementStrategy(
            [0, 2], NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL
        )
        with pytest.raises(IndexError):
            strategy.get_placement(cluster)


class TestHybridComponentPlacement:
    def test_parses_gpu_and_node_groups(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 2,
                    "component_placement": {
                        "actor": {"node_group": "train", "placement": "0-3"},
                        "env": {
                            "node_group": NodeGroupInfo.NODE_PLACEMENT_GROUP_LABEL,
                            "placement": "0-1:0-3",
                        },
                        "reward": {"node_group": "train", "placement": "0-1"},
                    },
                },
            }
        )

        cluster = create_fake_cluster(
            num_nodes=2,
            accelerators_per_node=4,
            extra_group_mapping={"train": [0, 1]},
        )
        placement = HybridComponentPlacement(config, cluster)

        actor_strategy = placement.get_strategy("actor")
        actor_ranks = placement.get_hardware_ranks("actor")
        assert actor_ranks == [0, 1, 2, 3]
        actor_placements = actor_strategy.get_placement(cluster)
        assert len(actor_placements) == 4

        env_strategy = placement.get_strategy("env")
        assert isinstance(env_strategy, NodePlacementStrategy)
        env_placements = env_strategy.get_placement(cluster)
        assert len(env_placements) == 4
        assert [p.cluster_node_rank for p in env_placements] == [0, 0, 1, 1]

    def test_all_keyword_expands_to_all_gpus(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor,reward": "all"},
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        placement = HybridComponentPlacement(config, cluster)

        actor_strategy = placement.get_strategy("actor")
        placements = actor_strategy.get_placement(cluster)
        assert len(placements) == 4
        assert sorted(p.local_accelerator_rank for p in placements) == [0, 1, 2, 3]

    def test_gpu_ranges_are_respected(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-2,5,7",
                        "inference": "3-4",
                        "reward": "0-2",
                    },
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=8)
        placement = HybridComponentPlacement(config, cluster)

        actor_strategy = placement.get_strategy("actor")
        actor_placements = actor_strategy.get_placement(cluster)
        assert sorted(p.local_accelerator_rank for p in actor_placements) == [
            0,
            1,
            2,
            5,
            7,
        ]

        inference_strategy = placement.get_strategy("inference")
        inference_placements = inference_strategy.get_placement(cluster)
        assert sorted(p.local_accelerator_rank for p in inference_placements) == [3, 4]

    def test_component_grouping_shares_strategy(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor,inference": "0-1",
                    },
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
        placement = HybridComponentPlacement(config, cluster)

        actor_strategy = placement.get_strategy("actor")
        assert actor_strategy is placement.get_strategy("inference")
        actor_placements = actor_strategy.get_placement(cluster)
        assert len(actor_placements) == 2

    def test_invalid_duplicate_resource_ranks(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1,1-2",
                    },
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        with pytest.raises(AssertionError, match="Resource ranks must be unique"):
            HybridComponentPlacement(config, cluster)

    def test_invalid_non_continuous_process_ranks(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-0:0,1-1:2",
                    },
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=2)
        with pytest.raises(
            AssertionError, match="Process ranks must be in ascending order"
        ):
            HybridComponentPlacement(config, cluster)

    def test_node_group_specific_allocations(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 2,
                    "component_placement": {
                        "actor": {
                            "node_group": "train",
                            "placement": "0-1",
                        },
                        "rollout": {
                            "node_group": "infer",
                            "placement": "0",
                        },
                    },
                },
            }
        )

        cluster = create_fake_cluster(
            num_nodes=2,
            accelerators_per_node=4,
            extra_group_mapping={"train": [0], "infer": [1]},
        )
        placement = HybridComponentPlacement(config, cluster)

        actor_placements = placement.get_strategy("actor").get_placement(cluster)
        assert {p.cluster_node_rank for p in actor_placements} == {0}

        rollout_placements = placement.get_strategy("rollout").get_placement(cluster)
        assert [p.cluster_node_rank for p in rollout_placements] == [1]


class TestModelParallelComponentPlacement:
    def test_collocated_mode(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor,rollout,reward": "0-3",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 4,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        placement = ModelParallelComponentPlacement(config, cluster)

        assert placement.is_collocated
        assert placement.get_hardware_ranks("actor") == [0, 1, 2, 3]
        assert placement.get_hardware_ranks("rollout") == [0, 1, 2, 3]
        assert placement.has_dedicated_inference is False
        assert placement.actor_world_size == 4
        assert placement.rollout_world_size == 4

        actor_strategy = placement.get_strategy("actor")
        rollout_strategy = placement.get_strategy("rollout")
        assert isinstance(actor_strategy, PackedPlacementStrategy)
        assert isinstance(rollout_strategy, PackedPlacementStrategy)

    def test_disaggregated_mode(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-5",
                        "inference": "6-7",
                        "reward": "2-5",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": True},
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=8)
        placement = ModelParallelComponentPlacement(config, cluster)

        assert placement.is_disaggregated
        assert placement.get_hardware_ranks("actor") == [0, 1]
        assert placement.get_hardware_ranks("rollout") == [2, 3, 4, 5]
        assert placement.get_hardware_ranks("inference") == [6, 7]
        assert placement.has_dedicated_inference is True

    def test_missing_actor_gpus_raises(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "rollout,reward": "0-3",
                    },
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        with pytest.raises(AssertionError, match="Actor GPUs"):
            ModelParallelComponentPlacement(config, cluster)

    def test_inference_tp_size_constraint(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-5",
                        "inference": "6-7",
                        "reward": "2-5",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 4,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": True},
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=8)
        with pytest.raises(AssertionError, match="Inference TP size"):
            ModelParallelComponentPlacement(config, cluster)

    def test_recompute_logprobs_false_with_inference_raises(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-5",
                        "inference": "6-7",
                        "reward": "2-5",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": False},
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=8)
        with pytest.raises(AssertionError, match="recompute_logprobs"):
            ModelParallelComponentPlacement(config, cluster)

    def test_reward_required_raises(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-3",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=4)
        with pytest.raises(AssertionError, match="Reward GPUs"):
            ModelParallelComponentPlacement(config, cluster)

    def test_auto_mode_actor_expands_to_all_gpus(self):
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "auto_scheduler": True,
                    "component_placement": {
                        "actor": "0-3",
                        "rollout": "4-7",
                        "inference": "8-9",
                        "reward": "4-7",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 4,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {
                    "tensor_parallel_size": 2,
                    "pipeline_parallel_size": 1,
                },
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": True},
            }
        )

        cluster = create_fake_cluster(num_nodes=1, accelerators_per_node=10)
        placement = ModelParallelComponentPlacement(config, cluster)

        assert placement.is_auto
        assert placement.has_dedicated_inference
        actor_strategy = placement.get_strategy("actor")
        actor_placements = actor_strategy.get_placement(cluster)
        assert len(actor_placements) == cluster.num_accelerators
        assert actor_placements[0].visible_accelerators == ["0"]
        rollout_strategy = placement.get_strategy("rollout")
        rollout_placements = rollout_strategy.get_placement(cluster)
        assert all(
            p.local_world_size == placement.rollout_tp_size for p in rollout_placements
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
