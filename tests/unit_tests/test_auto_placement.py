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

# Add auto_placement tools to path for testing
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../toolkits/auto_placement")
)
from auto_placement_worker import AutoPlacementWorker, get_workflow_graph
from node import ComponentNode, MegatronNode, RolloutNode, SccNode
from placement import ScheduleMode, ScheduleResult
from util import init_global_config
from workflow import Workflow, traverse_st_cuts


def get_mock_config_reasoning():
    mock_cfg = MagicMock()
    mock_cfg.runner.task_type = "reasoning"

    # 设置 get_workflow_graph 需要的属性
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
    return mock_cfg, mock_component_placement


def get_mock_config_embodiment(env_type: str):
    mock_cfg = MagicMock()
    mock_cfg.runner.task_type = "embodiment"

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
    mock_component_placement._cluster_num_gpus = 4
    mock_component_placement._components = ["env", "rollout", "actor"]
    mock_component_placement.rollout_dp_size = (
        mock_component_placement._cluster_num_gpus
    )
    mock_component_placement.rollout_world_size = (
        mock_component_placement._cluster_num_gpus
    )

    mock_cfg.algorithm.group_size = 1
    mock_cfg.profile_data.actor_cost = 100
    mock_component_placement.actor_dp_size = (
        mock_component_placement._cluster_num_gpus // 2
    )
    mock_component_placement.actor_world_size = (
        mock_component_placement._cluster_num_gpus
    )

    return mock_cfg, mock_component_placement


mock_cfg, mock_component_placement = get_mock_config_reasoning()
init_global_config(mock_cfg, mock_component_placement)


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
        mock_cfg, mock_component_placement = get_mock_config_reasoning()

        graph = {
            "rollout": ["inference"],
            "inference": ["actor"],
            "actor": [],
        }
        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert isinstance(res, ScheduleResult)
        assert res.total_gpu_num == mock_component_placement._cluster_num_gpus
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
        mock_cfg, mock_component_placement = get_mock_config_embodiment(
            env_type="libero"
        )

        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }

        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert res.total_gpu_num == mock_component_placement._cluster_num_gpus
        assert isinstance(res, ScheduleResult)
        assert res.mode == ScheduleMode.COLLOCATED

    def test_maniskill_embodiment(self):
        mock_cfg, mock_component_placement = get_mock_config_embodiment(
            env_type="maniskill"
        )
        graph = {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }
        auto_placement_worker = AutoPlacementWorker(
            mock_cfg, mock_component_placement, graph
        )
        res = auto_placement_worker.run()
        assert res.total_gpu_num == mock_component_placement._cluster_num_gpus
        assert res.placement[auto_placement_worker.get_node("actor")] == range(4)
        assert res.placement[auto_placement_worker.get_node("env")] == range(0, 1)
        assert res.placement[auto_placement_worker.get_node("env_rollout")] == range(
            1, 4
        )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
