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
import torch

from rlinf.scheduler import (
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
    WorkerAddress,
)


# Fixture to provide a ClusterResource instance for the test session
@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    # Use a small, fixed number of GPUs for consistent testing
    return Cluster(num_nodes=1)


# A basic Worker class for testing purposes
class BasicTestWorker(Worker):
    """A simple Worker implementation for testing basic functionality."""

    def __init__(self, arg1=None):
        super().__init__()
        self.arg1 = arg1
        self.initialized = True

    def get_rank(self):
        return self._rank

    def get_world_size(self):
        return self._world_size

    def get_init_arg(self):
        return self.arg1


# A WorkerGroup-decorated class for testing distributed functionality
class DistributedTestWorker(Worker):
    """A WorkerGroup for testing distributed operations."""

    def __init__(self):
        super().__init__()

    def get_env_info(self):
        """Returns a dictionary of environment information for the worker."""
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "node_id": self._cluster_node_rank,
            "gpu_id": self._local_accelerator_rank,
            "node_local_rank": self._node_local_rank,
        }

    def sum_with_rank(self, value):
        """Adds the worker's rank to the given value."""
        return value + self._rank


class TestClusterResource:
    """Tests for the ClusterResource class."""

    def test_cluster_initialization(self, cluster: Cluster):
        """Verify that the cluster is initialized with correct properties."""
        assert cluster._num_nodes == 1
        if torch.cuda.is_available():
            assert cluster.num_accelerators >= 1


class TestWorkerAddress:
    """Tests for the WorkerAddress class."""

    def test_worker_address_naming(self):
        """Verify that WorkerAddress generates correct names."""
        addr = WorkerAddress("MyWorkerGroup", 5)
        assert addr.root_group_name == "MyWorkerGroup"
        assert addr.rank == 5
        assert addr.get_name() == "MyWorkerGroup:5"


class TestWorkerGroup:
    """Tests for the WorkerGroup class and its interactions."""

    def test_worker_group_creation(self, cluster: Cluster):
        """Verify that a WorkerGroup can be created successfully."""
        if torch.cuda.is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="dist_test_1"
        )

        # Check that the correct number of actors were created
        assert len(worker_group.worker_info_list) == num_workers

        # Verify that we can get results from the workers
        results = worker_group.get_env_info().wait()
        assert len(results) == num_workers
        ranks = sorted([info["rank"] for info in results])
        assert ranks == list(range(num_workers))

    def test_execute_on_all_workers(self, cluster: Cluster):
        """Test calling a method on all workers in a group."""
        if torch.cuda.is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="dist_test_2"
        )

        base_value = 10
        results = worker_group.sum_with_rank(base_value).wait()

        assert len(results) == num_workers
        expected_results = sorted([base_value + i for i in range(num_workers)])
        assert sorted(results) == expected_results

    def test_execute_on_specific_ranks(self, cluster: Cluster):
        """Test calling a method on a subset of workers in a group."""
        if torch.cuda.is_available():
            placement = PackedPlacementStrategy(0, cluster.num_accelerators - 1)
        else:
            placement = NodePlacementStrategy([0] * 8)
        worker_group = DistributedTestWorker.create_group().launch(
            cluster=cluster, placement_strategy=placement, name="dist_test_3"
        )

        target_ranks = (0, 1)
        base_value = 20
        results = (
            worker_group.execute_on(*target_ranks).sum_with_rank(base_value).wait()
        )

        assert len(results) == len(target_ranks)
        expected_results = sorted([base_value + rank for rank in target_ranks])
        assert sorted(results) == expected_results

    def test_multiple_worker_groups(self, cluster: Cluster):
        """Test the creation and operation of multiple independent worker groups."""
        if torch.cuda.is_available():
            num_workers = cluster.num_accelerators
        else:
            num_workers = 1
        group1 = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="multi_group_1"
        )
        group2 = DistributedTestWorker.create_group().launch(
            cluster=cluster, name="multi_group_2"
        )

        # Call a method on group 1
        results1 = group1.sum_with_rank(100).wait()
        assert len(results1) == num_workers
        assert sorted(results1) == [100 + i for i in range(num_workers)]

        # Call a method on group 2
        results2 = group2.sum_with_rank(200).wait()
        assert len(results2) == num_workers
        assert sorted(results2) == [200 + i for i in range(num_workers)]


if __name__ == "__main__":
    pytest.main(["-v", __file__])
