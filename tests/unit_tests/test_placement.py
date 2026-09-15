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

import os
import sys
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import (
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
)
from rlinf.scheduler.cluster.config import NodeGroupEnvConfig
from rlinf.scheduler.cluster.node import NodeGroupInfo, NodeInfo
from rlinf.scheduler.hardware import Accelerator, HardwareInfo, HardwareResource
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../toolkits/auto_placement")
)
from auto_placement_worker import AutoPlacementWorker, get_workflow_graph  # noqa: E402
from node import ComponentNode, MegatronNode, RolloutNode, SccNode  # noqa: E402
from placement import ScheduleMode, ScheduleResult  # noqa: E402
from util import init_global_config  # noqa: E402
from workflow import Workflow, traverse_st_cuts  # noqa: E402


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


class TestHeteroMultiNodeGroupPlacement:
    def test_flexible_placement_across_hetero_node_groups(self):
        # Fake cluster with one node exposing both GPU and Franka hardware plus per-group env/interpreter.
        nodes = [
            NodeInfo(
                node_labels=[],
                node_rank=0,
                ray_id="ray-node-0",
                node_ip="10.0.0.1",
                num_cpus=32,
                python_interpreter_path="/usr/bin/python3",
                default_env_vars={},
                env_vars={},
                hardware_resources=[
                    HardwareResource(
                        type=Accelerator.HW_TYPE,
                        infos=[
                            HardwareInfo(type=Accelerator.HW_TYPE, model="GPU:MockGPU")
                        ],
                    ),
                    HardwareResource(
                        type="Franka",
                        infos=[HardwareInfo(type="Franka", model="Franka")],
                    ),
                ],
            )
        ]

        gpu_group = NodeGroupInfo(
            label="gpu",
            nodes=nodes,
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"GPU_ENV": "1"}]),
                    python_interpreter_path="/opt/gpu/python",
                )
            ],
        )
        franka_group = NodeGroupInfo(
            label="franka",
            nodes=nodes,
            hardware_type="Franka",
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"FRANKA_ENV": "1"}]),
                    python_interpreter_path="/opt/franka/python",
                )
            ],
        )
        node_groups = {
            gpu_group.label: gpu_group,
            franka_group.label: franka_group,
            NodeGroupInfo.DEFAULT_GROUP_LABEL: NodeGroupInfo(
                label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes
            ),
        }
        cluster = FakeCluster(nodes, node_groups)

        strategy = FlexiblePlacementStrategy(
            hardware_ranks_list=[[0], [1]],
            node_group_label=["gpu", "franka"],
        )
        placements = strategy.get_placement(cluster, isolate_accelerator=True)

        assert len(placements) == 2
        gpu_placement, franka_placement = placements

        assert gpu_placement.node_group_label == "gpu"
        assert gpu_placement.local_hardware_ranks == [0]
        assert gpu_placement.visible_accelerators == ["0"]

        assert franka_placement.node_group_label == "franka"
        assert franka_placement.local_hardware_ranks == [0]
        assert franka_placement.visible_accelerators == ["0"]

        assert gpu_group.get_node_env_vars(0) == {"GPU_ENV": "1"}
        assert franka_group.get_node_env_vars(0) == {"FRANKA_ENV": "1"}
        assert gpu_group.get_node_python_interpreter_path(0) == "/opt/gpu/python"
        assert franka_group.get_node_python_interpreter_path(0) == "/opt/franka/python"

    def test_packed_placement_across_hetero_node_groups(self):
        nodes = [
            NodeInfo(
                node_labels=[],
                node_rank=0,
                ray_id="ray-node-0",
                node_ip="10.0.0.1",
                num_cpus=32,
                python_interpreter_path="/usr/bin/python3",
                default_env_vars={},
                env_vars={},
                hardware_resources=[
                    HardwareResource(
                        type=Accelerator.HW_TYPE,
                        infos=[
                            HardwareInfo(type=Accelerator.HW_TYPE, model="GPU:MockGPU")
                        ],
                    ),
                    HardwareResource(
                        type="Franka",
                        infos=[HardwareInfo(type="Franka", model="Franka")],
                    ),
                ],
            )
        ]

        gpu_group = NodeGroupInfo(
            label="gpu",
            nodes=nodes,
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"GPU_ENV": "1"}]),
                    python_interpreter_path="/opt/gpu/python",
                )
            ],
        )
        franka_group = NodeGroupInfo(
            label="franka",
            nodes=nodes,
            hardware_type="Franka",
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"FRANKA_ENV": "1"}]),
                    python_interpreter_path="/opt/franka/python",
                )
            ],
        )
        node_groups = {
            gpu_group.label: gpu_group,
            franka_group.label: franka_group,
            NodeGroupInfo.DEFAULT_GROUP_LABEL: NodeGroupInfo(
                label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes
            ),
        }
        cluster = FakeCluster(nodes, node_groups)

        strategy = PackedPlacementStrategy(
            start_hardware_rank=0, end_hardware_rank=1, node_group=["gpu", "franka"]
        )
        placements = strategy.get_placement(cluster, isolate_accelerator=True)
        assert len(placements) == 2
        p0, p1 = placements

        assert p0.node_group_label == "gpu"
        assert p0.local_hardware_ranks == [0]
        assert p0.visible_accelerators == ["0"]

        assert p1.node_group_label == "franka"
        assert p1.local_hardware_ranks == [0]
        # Franka is non-accelerator; visible accelerators remain the node's GPUs
        assert p1.visible_accelerators == ["0"]

        assert gpu_group.get_node_env_vars(0) == {"GPU_ENV": "1"}
        assert franka_group.get_node_env_vars(0) == {"FRANKA_ENV": "1"}
        assert gpu_group.get_node_python_interpreter_path(0) == "/opt/gpu/python"
        assert franka_group.get_node_python_interpreter_path(0) == "/opt/franka/python"

    def test_node_placement_multi_node_groups(self):
        nodes = [
            _make_node_info(0, 1),
            _make_node_info(1, 2),
        ]
        group_a = NodeGroupInfo(
            label="a",
            nodes=[nodes[0]],
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"A_ENV": "1"}]),
                    python_interpreter_path="/opt/a/python",
                )
            ],
        )
        group_b = NodeGroupInfo(
            label="b",
            nodes=[nodes[1]],
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[1],
                    env_vars=OmegaConf.create([{"B_ENV": "1"}]),
                    python_interpreter_path="/opt/b/python",
                )
            ],
        )
        node_groups = {
            group_a.label: group_a,
            group_b.label: group_b,
            NodeGroupInfo.DEFAULT_GROUP_LABEL: NodeGroupInfo(
                label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes
            ),
        }
        cluster = FakeCluster(nodes, node_groups)

        strategy = NodePlacementStrategy([0, 1], node_group_label=["a", "b"])
        placements = strategy.get_placement(cluster, isolate_accelerator=False)
        assert len(placements) == 2
        p0, p1 = placements
        assert p0.node_group_label == "a"
        assert p0.cluster_node_rank == 0
        assert p0.visible_accelerators == ["0"]

        assert p1.node_group_label == "b"
        assert p1.cluster_node_rank == 1
        assert p1.visible_accelerators == ["0", "1"]

        assert group_a.get_node_env_vars(0) == {"A_ENV": "1"}
        assert group_b.get_node_env_vars(1) == {"B_ENV": "1"}
        assert group_a.get_node_python_interpreter_path(0) == "/opt/a/python"
        assert group_b.get_node_python_interpreter_path(1) == "/opt/b/python"

    def test_same_node_different_node_groups_env_and_interpreter_isolated(self):
        nodes = [
            NodeInfo(
                node_labels=[],
                node_rank=0,
                ray_id="ray-node-0",
                node_ip="10.0.0.1",
                num_cpus=32,
                python_interpreter_path="/usr/bin/python3",
                default_env_vars={},
                env_vars={},
                hardware_resources=[
                    HardwareResource(
                        type=Accelerator.HW_TYPE,
                        infos=[
                            HardwareInfo(
                                type=Accelerator.HW_TYPE, model="NV_GPU:MockGPU"
                            )
                        ],
                    ),
                    HardwareResource(
                        type="Franka",
                        infos=[HardwareInfo(type="Franka", model="Franka")],
                    ),
                ],
            )
        ]

        gpu_group = NodeGroupInfo(
            label="gpu",
            nodes=nodes,
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"GPU_ENV": "1"}]),
                    python_interpreter_path="/opt/gpu/python",
                )
            ],
        )
        franka_group = NodeGroupInfo(
            label="franka",
            nodes=nodes,
            hardware_type="Franka",
            env_configs=[
                NodeGroupEnvConfig(
                    node_ranks=[0],
                    env_vars=OmegaConf.create([{"FRANKA_ENV": "1"}]),
                    python_interpreter_path="/opt/franka/python",
                )
            ],
        )
        node_groups = {
            gpu_group.label: gpu_group,
            franka_group.label: franka_group,
            NodeGroupInfo.DEFAULT_GROUP_LABEL: NodeGroupInfo(
                label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes
            ),
        }
        cluster = FakeCluster(nodes, node_groups)

        strategy = FlexiblePlacementStrategy(
            hardware_ranks_list=[[0], [1]],
            node_group_label=["gpu", "franka"],
        )
        placements = strategy.get_placement(cluster, isolate_accelerator=True)
        assert len(placements) == 2
        assert placements[0].node_group_label == "gpu"
        assert placements[1].node_group_label == "franka"

        assert gpu_group.get_node_env_vars(0) == {"GPU_ENV": "1"}
        assert franka_group.get_node_env_vars(0) == {"FRANKA_ENV": "1"}
        assert gpu_group.get_node_python_interpreter_path(0) == "/opt/gpu/python"
        assert franka_group.get_node_python_interpreter_path(0) == "/opt/franka/python"

    def test_single_process_mixed_node_groups_raises(self):
        # One process cannot span different hardware/node groups
        nodes = [
            NodeInfo(
                node_labels=[],
                node_rank=0,
                ray_id="ray-node-0",
                node_ip="10.0.0.1",
                num_cpus=32,
                python_interpreter_path="/usr/bin/python3",
                default_env_vars={},
                env_vars={},
                hardware_resources=[
                    HardwareResource(
                        type=Accelerator.HW_TYPE,
                        infos=[
                            HardwareInfo(
                                type=Accelerator.HW_TYPE, model="NV_GPU:MockGPU"
                            )
                        ],
                    ),
                    HardwareResource(
                        type="Franka",
                        infos=[HardwareInfo(type="Franka", model="Franka")],
                    ),
                ],
            )
        ]
        node_groups = {
            "gpu": NodeGroupInfo(label="gpu", nodes=nodes),
            "franka": NodeGroupInfo(
                label="franka", nodes=nodes, hardware_type="Franka"
            ),
            NodeGroupInfo.DEFAULT_GROUP_LABEL: NodeGroupInfo(
                label=NodeGroupInfo.DEFAULT_GROUP_LABEL, nodes=nodes
            ),
        }
        cluster = FakeCluster(nodes, node_groups)

        strategy = FlexiblePlacementStrategy(
            hardware_ranks_list=[[0, 1]],
            node_group_label=["gpu", "franka"],
        )
        with pytest.raises(AssertionError):
            strategy.get_placement(cluster, isolate_accelerator=True)


if __name__ == "__main__":
    pytest.main(["-v", __file__])


def get_mock_config_reasoning():
    mock_cfg = MagicMock()
    mock_cfg.runner.task_type = "reasoning"

    # get_workflow_graph
    mock_cfg.algorithm.recompute_logprobs = True

    # Batch size
    mock_cfg.algorithm.group_size = 16
    mock_cfg.algorithm.n_minibatches = 4
    mock_cfg.data.rollout_batch_size = 512
    mock_cfg.runner.seq_length = 28 * 1024

    # Rollout config
    mock_cfg.rollout.max_running_requests = 128
    mock_cfg.rollout.gpu_memory_utilization = 0.55

    # Profile data
    mock_cfg.profile_data.actor_cost = 101
    mock_cfg.profile_data.rollout_cost = 224
    mock_cfg.profile_data.inference_cost = 10

    # Model size
    mock_component_placement = MagicMock()
    mock_component_placement._cluster_num_gpus = 16 * 8
    mock_component_placement._components = [
        "actor",
        "rollout",
        "inference",
    ]
    world_size = 16 * 8
    mock_component_placement.actor_dp_size = world_size // 2
    mock_component_placement.actor_world_size = world_size
    mock_component_placement.rollout_dp_size = world_size
    mock_component_placement.rollout_world_size = world_size
    mock_component_placement.inference_dp_size = world_size // 2
    mock_component_placement.inference_world_size = world_size

    # cluster
    mock_cluster = MagicMock()
    mock_cluster.num_accelerators = 16 * 8

    return mock_cfg, mock_component_placement, mock_cluster


def get_mock_config_embodiment(env_type: str):
    mock_cfg = MagicMock()
    mock_cfg.runner.task_type = "embodied"

    mock_cfg.data.rollout_batch_size = 1024
    if env_type == "libero":
        mock_cfg.data.env_num = 64
        mock_cfg.profile_data.env_profile_data = {
            4: 0.61,
            8: 1.23,
            16: 2.46,
            32: 4.66,
            64: 18.5,
        }
        mock_cfg.profile_data.rollout_profile_data = {
            4: 0.6,
            8: 1.01,
            16: 2.12,
            32: 3.72,
            64: 15.3,
        }
    elif env_type == "maniskill":
        mock_cfg.data.env_num = 40
        mock_cfg.profile_data.env_profile_data = {
            10: 0.8,
            20: 0.8,
            30: 0.85,
            40: 0.85,
        }
        mock_cfg.profile_data.rollout_profile_data = {
            10: 0.4,
            20: 0.6,
            30: 0.85,
            40: 1.15,
        }

    # Model size
    mock_component_placement = MagicMock()
    mock_component_placement._components = ["env", "rollout", "actor"]
    mock_component_placement.get_world_size.side_effect = lambda component: {
        "env": 4,
        "rollout": 4,
        "actor": 4,
    }[component]
    mock_cfg.algorithm.group_size = 1
    mock_cfg.profile_data.actor_cost = 100

    # cluster
    mock_cluster = MagicMock()
    mock_cluster.num_accelerators = 4

    return mock_cfg, mock_component_placement, mock_cluster


init_global_config(*get_mock_config_reasoning())


class TestNode:
    """Tests for node class."""

    def test_node_creation(self):
        """Test basic node creation and methods."""
        actor_node = MegatronNode("actor")
        inference_node = MegatronNode("inference")
        rollout_node = RolloutNode()

        assert actor_node.role == "actor"
        assert inference_node.role == "inference"
        assert rollout_node.role == "rollout"

    def test_node_validation(self):
        """Test node validation."""
        valid_gpu_nums = [1, 2, 4, 8]
        actor_node = MegatronNode(role="actor", valid_gpu_nums=valid_gpu_nums)

        for gpu_num in range(10):
            if gpu_num in valid_gpu_nums:
                assert actor_node._validate_gpu_num(gpu_num)
            else:
                assert not actor_node._validate_gpu_num(gpu_num)


class TestWorkflow:
    """Tests for the Workflow class."""

    _name_to_node_dict = {
        "rollout": RolloutNode(),
        "inference": MegatronNode("inference"),
        "actor": MegatronNode("actor"),
    }

    def get_node(self, name: str) -> ComponentNode:
        return self._name_to_node_dict[name]

    def test_workflow_graph(self):
        """Test workflow creation and basic properties."""
        cfg = MagicMock()
        cfg.runner.task_type = "reasoning"
        cfg.algorithm.recompute_logprobs = True
        workflow_graph = get_workflow_graph(cfg)
        assert workflow_graph == {
            "rollout": ["inference"],
            "inference": ["actor"],
            "actor": [],
        }

        cfg.algorithm.recompute_logprobs = False
        workflow_graph = get_workflow_graph(cfg)
        assert workflow_graph == {
            "rollout": ["actor"],
            "actor": [],
        }

    def test_workflow_creation(self):
        """Test workflow creation."""
        graph = {
            "rollout": ["inference"],
            "inference": ["actor"],
            "actor": [],
        }

        workflow_graph = {}
        for node, neighbors in graph.items():
            workflow_graph[self.get_node(node)] = [
                self.get_node(neighbor) for neighbor in neighbors
            ]
        workflow = Workflow(workflow_graph)
        assert set(workflow.nodes) == {
            self.get_node("rollout"),
            self.get_node("inference"),
            self.get_node("actor"),
        }
        assert workflow.topological_order == [
            self.get_node("rollout"),
            self.get_node("inference"),
            self.get_node("actor"),
        ]

    def test_traverse_st_cuts(self):
        """Test traverse st cuts of workflow."""
        graph = {
            "rollout": ["inference"],
            "inference": ["actor"],
            "actor": [],
        }
        workflow = Workflow(graph)
        cuts = traverse_st_cuts(workflow)
        assert len(cuts) == 2
        assert cuts[0][0].is_node() and cuts[0][0].nodes[0] == "rollout"
        assert cuts[1][1].is_node() and cuts[1][1].nodes[0] == "actor"

        cuts = traverse_st_cuts(cuts[0][1])
        assert len(cuts) == 1
        assert cuts[0][0].is_node() and cuts[0][0].nodes[0] == "inference"
        assert cuts[0][1].is_node() and cuts[0][1].nodes[0] == "actor"

    def test_compress_sccs(self):
        """Test SCC compression."""
        graph = {
            self.get_node("inference"): [self.get_node("rollout")],
            self.get_node("rollout"): [
                self.get_node("inference"),
                self.get_node("actor"),
            ],
            self.get_node("actor"): [],
        }
        workflow = Workflow(graph)
        compressed_workflow = workflow.compress_sccs()

        assert len(workflow.nodes) == 3 and len(compressed_workflow.nodes) == 2

        topological_order = compressed_workflow.topological_order
        assert isinstance(topological_order[0], SccNode)
        assert topological_order[0].role in [
            "inference - rollout",
            "rollout - inference",
        ]


class TestAutoPlacementWorkerForReasoning:
    """Tests for the SchedulerTask class."""

    def test_auto_placement_worker(self):
        """Test SchedulerTask initialization."""
        # Create a mock config
        mock_cfg, mock_component_placement, mock_cluster = get_mock_config_reasoning()

        graph = {
            "rollout": ["inference"],
            "inference": ["actor"],
            "actor": [],
        }

        init_global_config(mock_cfg, mock_component_placement, mock_cluster)

        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert isinstance(res, ScheduleResult)
        assert res.total_gpu_num == mock_cluster.num_accelerators
        assert res.mode == ScheduleMode.DISAGGREGATED

        assert len(res.placement[auto_placement_worker.get_node("rollout")]) == 80, (
            f"{res.placement_str}"
        )
        assert len(res.placement[auto_placement_worker.get_node("inference")]) == 16, (
            f"{res}"
        )
        assert len(res.placement[auto_placement_worker.get_node("actor")]) == 32


class TestAutoPlacementWorkerForEmbodiment:
    """Tests for the SchedulerTask class."""

    def test_libero_embodiment(self):
        """Test SchedulerTask initialization."""
        # Create a mock config
        mock_cfg, mock_component_placement, mock_cluster = get_mock_config_embodiment(
            env_type="libero"
        )

        init_global_config(mock_cfg, mock_component_placement, mock_cluster)

        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }

        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert res.total_gpu_num == mock_cluster.num_accelerators
        assert isinstance(res, ScheduleResult)
        assert res.mode == ScheduleMode.COLLOCATED

    def test_maniskill_embodiment(self):
        mock_cfg, mock_component_placement, mock_cluster = get_mock_config_embodiment(
            env_type="maniskill"
        )

        init_global_config(mock_cfg, mock_component_placement, mock_cluster)

        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }
        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert res.total_gpu_num == mock_cluster.num_accelerators
        assert res.placement[auto_placement_worker.get_node("actor")] == range(4)
        assert res.placement[auto_placement_worker.get_node("env")] == range(0, 1)
        assert res.placement[auto_placement_worker.get_node("env_rollout")] == range(
            1, 4
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
