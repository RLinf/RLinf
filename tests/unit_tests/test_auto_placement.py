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
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add auto_placement tools to path for testing
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "../../toolkits/auto_placement")
)
# Add RLinf to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from resource_allocator import (
    AllocationStates,
    ComponentParallelState,
    ResourcePlanner,
    get_valid_dp_sizes,
    resource_allocate,
)
from workflow import (
    ComponentNode,
    Node,
    PipelineCostCacl,
    SccComponentNode,
    Workflow,
    WorkflowPartitioner,
    get_workflow_cost,
    get_workflow_partition,
)

try:
    # Mock the missing dependencies for scheduler_task
    sys.modules["rlinf.config"] = Mock()
    sys.modules["rlinf.config"].validate_cfg = Mock()
    from scheduler_task import SchedulerTask, get_profile_data

    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    pytestmark = pytest.mark.skip(f"Auto placement modules not available: {e}")


class TestComponentParallelState:
    """Tests for the ComponentParallelState class."""

    def test_component_parallel_state_initialization(self):
        """Test ComponentParallelState initialization and post_init calculations."""
        state = ComponentParallelState(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=4
        )

        assert state.tensor_model_parallel_size == 2
        assert state.pipeline_model_parallel_size == 4
        assert state.model_parallel_size == 8
        assert state.world_size == 0
        assert state.data_parallel_size == 0
        assert state.valid_dp_sizes == []

    def test_allocation_without_valid_dp_sizes(self):
        """Test resource allocation without valid DP size constraints."""
        state = ComponentParallelState(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )

        # Test normal allocation
        idle_gpus = state.allocation(16)
        assert state.world_size == 16
        assert state.data_parallel_size == 4
        assert idle_gpus == 0

    def test_allocation_with_valid_dp_sizes(self):
        """Test resource allocation with valid DP size constraints."""
        state = ComponentParallelState(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )
        state.set_valid_dp_sizes([1, 2])

        # Test allocation respects valid DP sizes
        idle_gpus = state.allocation(16)
        assert state.data_parallel_size == 2  # Limited by valid_dp_sizes
        assert state.world_size == 8
        assert idle_gpus == 8

    def test_allocation_insufficient_gpus(self):
        """Test allocation when insufficient GPUs are available."""
        state = ComponentParallelState(
            tensor_model_parallel_size=4, pipeline_model_parallel_size=2
        )

        # Not enough GPUs for even one instance
        idle_gpus = state.allocation(4)
        assert state.world_size == 0
        assert state.data_parallel_size == 0
        assert idle_gpus == 4


class TestAllocationStates:
    """Tests for the AllocationStates class."""

    def test_allocation_states_initialization(self):
        """Test AllocationStates initialization."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }

        allocation = AllocationStates(components_config)

        assert "actor" in allocation.states
        assert "rollout" in allocation.states
        assert allocation.idle_gpus == 0

    def test_get_component(self):
        """Test getting component states."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
        }

        allocation = AllocationStates(components_config)
        actor_state = allocation.get_component("actor")

        assert actor_state is not None
        assert actor_state.tensor_model_parallel_size == 2
        assert allocation.get_component("nonexistent") is None

    def test_total_and_used_gpus(self):
        """Test GPU counting methods."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }

        allocation = AllocationStates(components_config)
        allocation.get_component("actor").allocation(4)
        allocation.get_component("rollout").allocation(2)
        allocation.idle_gpus = 2

        assert allocation.used_gpus() == 6
        assert allocation.total_gpus() == 8


class TestResourcePlanner:
    """Tests for the ResourcePlanner class."""

    def test_resource_planner_initialization(self):
        """Test ResourcePlanner initialization."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }

        planner = ResourcePlanner(
            components_config=components_config,
            total_gpus=8,
            valid_actor_dp_sizes=[1, 2, 4],
            valid_inference_dp_sizes=[1, 2],
        )

        assert planner.total_gpus == 8
        assert len(planner.valid_components) == 2
        assert "actor" in planner.valid_components
        assert "rollout" in planner.valid_components

    def test_generate_states_for_single_component(self):
        """Test generating states for a single component."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
        }

        planner = ResourcePlanner(
            components_config=components_config,
            total_gpus=8,
            valid_actor_dp_sizes=[1, 2, 4],
            valid_inference_dp_sizes=[],
        )

        init_allocation = AllocationStates(components_config)
        states = planner.generate_states_for_single_component(init_allocation, "actor")

        assert len(states) > 0
        # Should generate states with different data parallel sizes
        for state in states:
            actor_state = state.get_component("actor")
            assert actor_state.world_size > 0

    def test_generate_all_states(self):
        """Test generating all possible allocation states."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }

        planner = ResourcePlanner(
            components_config=components_config,
            total_gpus=8,
            valid_actor_dp_sizes=[1, 2],
            valid_inference_dp_sizes=[],
        )

        planner.generate_all_states()
        assert hasattr(planner, "all_states")
        assert len(planner.all_states) > 0

    def test_generate_static_states(self):
        """Test generating static states that use all GPUs."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }

        planner = ResourcePlanner(
            components_config=components_config,
            total_gpus=8,
            valid_actor_dp_sizes=[1, 2, 4],
            valid_inference_dp_sizes=[],
        )

        static_states = planner.generate_static_states()

        # All static states should use all available GPUs
        for state in static_states:
            assert state.used_gpus() == 8


class TestResourceAllocationFunctions:
    """Tests for standalone resource allocation functions."""

    def test_get_valid_dp_sizes(self):
        """Test getting valid data parallel sizes."""
        parallel_config = {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
        }

        valid_sizes = get_valid_dp_sizes(
            total_gpus=8,
            parallel_config=parallel_config,
            group_size=16,
            rollout_batch_size=128,
            n_minibatches=4,
        )

        assert valid_sizes == [1, 2, 4]

    def test_resource_allocate(self):
        """Test main resource allocation function."""
        components_config = {
            "actor": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "inference": {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 1,
            },
            "rollout": {
                "tensor_model_parallel_size": 1,
                "pipeline_model_parallel_size": 1,
            },
        }

        allocation_states = resource_allocate(
            components_config=components_config,
            total_gpus=8,
            group_size=16,
            rollout_batch_size=128,
            n_minibatches=4,
        )

        assert isinstance(allocation_states, list)
        assert len(allocation_states) > 0
        assert all(isinstance(state, AllocationStates) for state in allocation_states)

        for allocation_state in allocation_states:
            assert allocation_state.used_gpus() == 8
            actor_state = allocation_state.get_component("actor")
            inference_state = allocation_state.get_component("inference")
            rollout_state = allocation_state.get_component("rollout")
            assert actor_state.model_parallel_size == 2
            assert inference_state.model_parallel_size == 2
            assert rollout_state.model_parallel_size == 1
            assert (
                actor_state.world_size != 0
                and inference_state.world_size != 0
                and rollout_state.world_size != 0
            )


class TestWorkflowNodes:
    """Tests for workflow node classes."""

    def test_node_creation(self):
        """Test basic node creation and methods."""
        node = Node("test_node")
        node.set_single_batch_instance_cost(10.0)
        node.set_instance_num(2)

        assert node.name == "test_node"
        assert node.get_single_batch_cost() == 5.0

    def test_component_node(self):
        """Test ComponentNode creation."""
        component = ComponentNode("actor")
        component.set_single_batch_instance_cost(20.0)
        component.set_instance_num(4)

        assert component.name == "actor"
        assert component.get_single_batch_cost() == 5.0

    def test_scc_component_node(self):
        """Test SccComponentNode with multiple components."""
        component1 = ComponentNode("actor")
        component1.set_single_batch_instance_cost(20.0)
        component1.set_instance_num(4)

        component2 = ComponentNode("rollout")
        component2.set_single_batch_instance_cost(30.0)
        component2.set_instance_num(3)

        scc_node = SccComponentNode([component1, component2])

        assert "actor - rollout" in scc_node.name
        assert scc_node.get_single_batch_cost() == 15.0  # 5.0 + 10.0


class TestWorkflow:
    """Tests for the Workflow class."""

    def test_workflow_creation(self):
        """Test workflow creation and basic properties."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")

        workflow_dict = {node1: [node2], node2: []}

        workflow = Workflow(workflow_dict)

        assert len(workflow.nodes) == 2
        assert node1 in workflow.nodes
        assert node2 in workflow.nodes

    def test_topological_sort(self):
        """Test topological sorting of workflow."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")
        node3 = ComponentNode("inference")

        workflow_dict = {node1: [node2], node2: [node3], node3: []}

        workflow = Workflow(workflow_dict)
        topo_order = workflow.topological_sort()

        assert len(topo_order) == 3
        # node1 should come before node2, which should come before node3
        assert topo_order.index(node1) < topo_order.index(node2)
        assert topo_order.index(node2) < topo_order.index(node3)

    def test_find_sccs(self):
        """Test finding strongly connected components."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")

        # Create a simple cycle
        workflow_dict = {node1: [node2], node2: [node1]}

        workflow = Workflow(workflow_dict)
        sccs = workflow.find_sccs()

        assert len(sccs) >= 1
        # The cycle should form an SCC
        assert any(len(scc) == 2 for scc in sccs)

    def test_compress_sccs(self):
        """Test SCC compression."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")
        node3 = ComponentNode("inference")

        workflow_dict = {
            node1: [node2],
            node2: [node1, node3],  # Creates a cycle between node1 and node2
            node3: [],
        }

        workflow = Workflow(workflow_dict)
        compressed = workflow.compress_sccs()

        assert isinstance(compressed, Workflow)
        # After compression, should have fewer nodes if there were cycles


class TestWorkflowPartitioner:
    """Tests for the WorkflowPartitioner class."""

    def test_workflow_partitioner_creation(self):
        """Test WorkflowPartitioner creation."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")

        workflow_dict = {node1: [node2], node2: []}

        workflow = Workflow(workflow_dict)
        partitioner = WorkflowPartitioner(workflow)

        assert partitioner.workflow is not None

    def test_partition_generation(self):
        """Test partition generation."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")
        node3 = ComponentNode("inference")

        workflow_dict = {node1: [node2], node2: [node3], node3: []}

        workflow = Workflow(workflow_dict)
        partitioner = WorkflowPartitioner(workflow)
        partitions = partitioner.partition()

        assert isinstance(partitions, list)
        assert len(partitions) > 0
        # Should have different partition options
        assert all(isinstance(partition, dict) for partition in partitions)


class TestPipelineCostCacl:
    """Tests for the PipelineCostCacl class."""

    def test_pipeline_cost_calculation(self):
        """Test pipeline cost calculation."""
        node1 = ComponentNode("actor")
        node1.set_single_batch_instance_cost(10.0)
        node1.set_instance_num(1)

        node2 = ComponentNode("rollout")
        node2.set_single_batch_instance_cost(20.0)
        node2.set_instance_num(1)

        workflow_dict = {node1: [node2], node2: []}

        workflow = Workflow(workflow_dict)
        cost_calc = PipelineCostCacl(workflow)

        result = cost_calc.calculate_total_time(total_data_size=100, batch_size=10)

        assert "total_time" in result
        assert "startup_time" in result
        assert "steady_time" in result
        assert "num_batches" in result
        assert "critical_path" in result
        assert "throughput" in result

        assert result["total_time"] == 30.0 + 20.0 * 9
        assert result["num_batches"] == 10

    def test_critical_path_finding(self):
        """Test critical path finding."""
        node1 = ComponentNode("actor")
        node1.set_single_batch_instance_cost(10.0)
        node1.set_instance_num(1)

        node2 = ComponentNode("rollout")
        node2.set_single_batch_instance_cost(5.0)
        node2.set_instance_num(1)

        workflow_dict = {node1: [node2], node2: []}

        workflow = Workflow(workflow_dict)
        cost_calc = PipelineCostCacl(workflow)

        assert len(cost_calc.critical_path) > 0
        assert node1 in cost_calc.critical_path and node2 in cost_calc.critical_path


class TestWorkflowUtilityFunctions:
    """Tests for workflow utility functions."""

    def test_get_workflow_cost(self):
        """Test get_workflow_cost function."""

        node1 = ComponentNode("rollout")
        node1.set_single_batch_instance_cost(10.0)
        node1.set_instance_num(1)

        node2 = ComponentNode("actor")
        node2.set_single_batch_instance_cost(10.0)
        node2.set_instance_num(2)

        workflow_dict = {node1: [node2], node2: []}

        workflow = Workflow(workflow_dict)
        cost = get_workflow_cost(workflow, batch_size=10, total_data_size=100)

        assert isinstance(cost, (int, float))
        assert int(cost) == 15 + 90

    def test_get_workflow_partition(self):
        """Test get_workflow_partition function."""
        node1 = ComponentNode("actor")
        node2 = ComponentNode("rollout")

        workflow_dict = {node1: [node2], node2: []}

        workflow = Workflow(workflow_dict)
        partitions = get_workflow_partition(workflow)

        assert isinstance(partitions, list)
        assert len(partitions) > 0


@pytest.mark.skipif(
    not IMPORTS_AVAILABLE, reason="Auto placement modules not available"
)
class TestSchedulerTask:
    """Tests for the SchedulerTask class."""

    def test_get_profile_data(self):
        """Test get_profile_data function."""
        # Create a mock config
        mock_cfg = MagicMock()
        mock_cfg.runner.task_type = "math"
        mock_cfg.actor.model.tensor_model_parallel_size = 2
        mock_cfg.actor.model.pipeline_model_parallel_size = 1
        mock_cfg.rollout.tensor_parallel_size = 1
        mock_cfg.rollout.pipeline_parallel_size = 1
        mock_cfg.cluster.num_nodes = 1
        mock_cfg.algorithm.group_size = 4
        mock_cfg.algorithm.n_minibatches = 4
        mock_cfg.data.rollout_batch_size = 16
        mock_cfg.runner.seq_length = 2048

        mock_cluster = MagicMock()
        mock_cluster.num_accelerators = 8

        profile_data = get_profile_data(
            mock_cfg,
            cluster=mock_cluster,
            actor_cost=100.0,
            inference_cost=50.0,
            rollout_cost=75.0,
        )

        assert profile_data["actor"] == (4, 100.0)
        assert profile_data["rollout"] == (8, 75.0)
        assert profile_data["inference"] == (4, 50.0)

    @patch("scheduler_task.validate_cfg")
    def test_scheduler_task_initialization(self, mock_validate):
        """Test SchedulerTask initialization."""
        # Create a mock config
        mock_cfg = MagicMock()
        mock_cfg.runner.task_type = "reasoning"
        mock_cfg.actor.model.tensor_model_parallel_size = 2
        mock_cfg.actor.model.pipeline_model_parallel_size = 1
        mock_cfg.rollout.tensor_parallel_size = 1
        mock_cfg.rollout.pipeline_parallel_size = 1
        mock_cfg.cluster.num_nodes = 1
        mock_cfg.algorithm.group_size = 4
        mock_cfg.algorithm.n_minibatches = 4
        mock_cfg.data.rollout_batch_size = 16
        mock_cfg.runner.seq_length = 2048

        # Mock the inference config to avoid exceptions
        mock_cfg.inference.model.tensor_model_parallel_size = 2
        mock_cfg.inference.model.pipeline_model_parallel_size = 1

        mock_validate.return_value = mock_cfg

        mock_cluster = MagicMock()
        mock_cluster.num_accelerators = 8

        scheduler_task = SchedulerTask(mock_cfg, mock_cluster)

        assert scheduler_task.is_reasoning is True
        assert scheduler_task.total_gpus == 8
        assert scheduler_task.group_size == 4
        assert "actor" in scheduler_task.components_config
        assert "rollout" in scheduler_task.components_config


if __name__ == "__main__":
    pytest.main(["-v", __file__])
