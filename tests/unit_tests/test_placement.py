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
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import DictConfig

from rlinf.scheduler import (
    AcceleratorType,
    Cluster,
    FlexiblePlacementStrategy,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)


# Fixture to provide a ClusterResource instance for the test session
@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    # Use a fixed number of GPUs for consistent testing.
    # A warning will be issued if an instance already exists, which is fine for testing.
    return Cluster(num_nodes=1)


def mock_cluster(num_nodes, num_accelerators_per_node):
    cluster = MagicMock()
    cluster.num_accelerators_in_cluster = num_nodes * num_accelerators_per_node
    cluster.num_nodes = num_nodes
    cluster.get_node_id_from_accel_id = MagicMock(
        side_effect=lambda x: x // num_accelerators_per_node
    )
    cluster.global_accel_id_to_local_accel_id = MagicMock(
        side_effect=lambda x: x % num_accelerators_per_node
    )
    cluster.get_node_num_accelerators = MagicMock(
        return_value=num_accelerators_per_node
    )
    # accelerator_type can be retrieved from node_info
    node_info = MagicMock()
    node_info.accelerator_type = AcceleratorType.NV_GPU
    cluster.get_node_info = MagicMock(return_value=node_info)
    return cluster


# A WorkerGroup-decorated class for testing placement strategies
class PlacementTestWorker(Worker):
    """A WorkerGroup for testing placement strategies by reporting environment info."""

    def __init__(self):
        super().__init__()

    def get_placement_info(self):
        """Returns a dictionary of placement-related information for the worker."""
        return {
            "rank": self._rank,
            "world_size": self._world_size,
            "node_id": self._node_id,
            "gpu_id": self._local_accelerator_id,
            "node_local_rank": self._node_local_rank,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "local_rank": self._local_rank,
            "local_world_size": self._local_world_size,
        }

    def get_available_gpus(self):
        """Returns the number of GPUs visible to the torch runtime inside the worker."""
        return torch.cuda.device_count()


class TestPlacementStrategies:
    """End-to-end tests for different placement strategies."""

    def test_packed_placement(self, cluster: Cluster):
        """Verify that PackedPlacementStrategy places workers on all available GPUs."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping accelerator placement test on CPU-only environment.")
        num_gpus = cluster.num_accelerators_in_cluster
        placement = PackedPlacementStrategy(
            start_accelerator_id=0, end_accelerator_id=num_gpus - 1
        )
        worker_group = PlacementTestWorker.create_group().launch(
            cluster=cluster, name="packed_test", placement_strategy=placement
        )

        results = worker_group.get_placement_info().wait()

        assert len(results) == num_gpus
        ranks = sorted([info["rank"] for info in results])
        assert ranks == list(range(num_gpus))

        for info in results:
            assert info["world_size"] == num_gpus
            # In packed mode, local rank equals the global rank (for a single node)
            assert info["node_local_rank"] == info["rank"]
            # Ray sets CUDA_VISIBLE_DEVICES to the specific GPU ID for the worker
            assert info["cuda_visible_devices"] == str(info["node_local_rank"])
            # Worker._env_setup_before_init sets LOCAL_RANK based on node_local_rank
            assert info["local_rank"] == 0

        worker_group2 = PlacementTestWorker.create_group().launch(
            cluster=cluster,
            name="packed_test_no_isolate",
            placement_strategy=placement,
            isolate_gpu=False,
        )

        results = worker_group2.get_placement_info().wait()

        assert len(results) == num_gpus
        ranks = sorted([info["rank"] for info in results])
        assert ranks == list(range(num_gpus))

        for info in results:
            assert info["world_size"] == num_gpus
            # In packed mode, local rank equals the global rank (for a single node)
            assert info["node_local_rank"] == info["rank"]
            # Ray sets CUDA_VISIBLE_DEVICES to the specific GPU ID for the worker
            assert info["cuda_visible_devices"] == ",".join(
                [str(i) for i in range(num_gpus)]
            )
            # For packed placement with isolate_gpu=False
            # Worker._env_setup_before_init sets LOCAL_RANK based on node_local_rank
            assert info["local_rank"] == info["node_local_rank"]

    def test_fine_grained_packed_placement(self, cluster: Cluster):
        """Verify that FineGrainedPackedPlacementStrategy places a specific number of workers."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping accelerator placement test on CPU-only environment.")
        num_workers_to_place = 2
        placement = PackedPlacementStrategy(
            start_accelerator_id=0, end_accelerator_id=num_workers_to_place - 1
        )
        worker_group = PlacementTestWorker.create_group().launch(
            cluster=cluster, name="fine_grained_test", placement_strategy=placement
        )

        results = worker_group.get_placement_info().wait()

        assert len(results) == num_workers_to_place
        ranks = sorted([info["rank"] for info in results])
        assert ranks == list(range(num_workers_to_place))

        for info in results:
            assert info["world_size"] == num_workers_to_place
            assert info["rank"] in [0, 1]
            assert info["gpu_id"] in [0, 1]

    def test_chunked_placement(self, cluster: Cluster):
        """Verify that ChunkedPlacementStrategy allocates multiple GPUs per worker."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping accelerator placement test on CPU-only environment.")
        stride = 2
        num_gpus = cluster.num_accelerators_in_cluster
        expected_num_workers = num_gpus // stride

        placement = PackedPlacementStrategy(
            start_accelerator_id=0,
            end_accelerator_id=cluster.num_accelerators_in_cluster - 1,
            num_accelerators_per_process=stride,
        )
        worker_group = PlacementTestWorker.create_group().launch(
            cluster=cluster, name="chunked_test", placement_strategy=placement
        )

        # Test the number of available GPUs inside each worker
        gpu_counts = worker_group.get_available_gpus().wait()
        assert len(gpu_counts) == expected_num_workers
        assert all(count == stride for count in gpu_counts)

        # Test the environment variables
        results = worker_group.get_placement_info().wait()
        assert len(results) == expected_num_workers

        for i in range(expected_num_workers):
            info = next(r for r in results if r["rank"] == i)
            assert info["world_size"] == expected_num_workers

            # Check that CUDA_VISIBLE_DEVICES is set correctly for each stride
            expected_gpus = [str(g) for g in range(i * stride, (i + 1) * stride)]
            assert info["cuda_visible_devices"] == ",".join(expected_gpus)

            # For strided placement, isolate_gpu is True, so LOCAL_RANK should be 0
            assert info["local_rank"] == 0
            assert info["local_world_size"] == 1

    def test_strided_placement(self, cluster: Cluster):
        """Verify that stride allocates GPUs in a strided manner."""
        if not torch.cuda.is_available():
            pytest.skip("Skipping accelerator placement test on CPU-only environment.")
        num_gpus = cluster.num_accelerators_in_cluster
        stride = 2
        num_gpus_per_process = 2

        # Only run if the number of GPUs is divisible by stride * num_gpus_per_process
        if (
            num_gpus < stride * num_gpus_per_process
            or num_gpus % (stride * num_gpus_per_process) != 0
        ):
            pytest.skip(
                "Strided test requires num_gpus to be divisible by stride * num_gpus_per_process."
            )

        expected_num_workers = (num_gpus // (stride * num_gpus_per_process)) * stride

        placement = PackedPlacementStrategy(
            start_accelerator_id=0,
            end_accelerator_id=cluster.num_accelerators_in_cluster - 1,
            stride=stride,
            num_accelerators_per_process=num_gpus_per_process,
        )
        worker_group = PlacementTestWorker.create_group().launch(
            cluster=cluster, name="strided_test", placement_strategy=placement
        )

        # Test the number of available GPUs inside each worker
        gpu_counts = worker_group.get_available_gpus().wait()
        assert len(gpu_counts) == expected_num_workers
        assert all(count == num_gpus_per_process for count in gpu_counts)

        # Test the environment variables and CUDA_VISIBLE_DEVICES
        results = worker_group.get_placement_info().wait()
        assert len(results) == expected_num_workers

        # Check that each worker has the correct CUDA_VISIBLE_DEVICES and local_rank
        for info in results:
            rank = info["rank"]
            group_rank = rank // stride
            process_rank_in_group = rank % stride
            local_gpu_id = (
                group_rank * stride * num_gpus_per_process + process_rank_in_group
            )
            expected_devices = [
                str(local_gpu_id + i * stride) for i in range(num_gpus_per_process)
            ]
            assert info["cuda_visible_devices"] == ",".join(expected_devices)
            assert (
                info["local_rank"] == 0
            )  # isolate_gpu=True, so local_rank should be 0
            assert info["local_world_size"] == 1

    def test_packed_placement_strategy_multiple_nodes(self):
        """Test PackedPlacementStrategy with 2 nodes and 4 GPUs per node, 1 GPU per process."""
        # Mock a cluster with 2 nodes and 4 GPUs per node
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=7)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 8
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == i // 4
            assert p.node_rank == i // 4
            assert p.local_accelerator_id == i % 4
            assert p.local_rank == i % 4
            assert p.local_world_size == 4
            assert p.visible_accelerators == [str(i % 4)]
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_multiple_nodes_multiple_gpus_per_process(self):
        """Test PackedPlacementStrategy with 2 nodes, 4 GPUs per node, 2 GPUs per process."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(
            start_accelerator_id=0, end_accelerator_id=7, num_accelerators_per_process=2
        )
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == i // 2
            assert p.node_rank == i // 2
            assert p.local_accelerator_id == (i % 2) * 2
            assert p.local_rank == i % 2
            assert p.local_world_size == 2
            expected_devices = [str((i % 2) * 2), str((i % 2) * 2 + 1)]
            assert p.visible_accelerators == expected_devices
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_multiple_nodes_no_isolate(self):
        """Test PackedPlacementStrategy with 2 nodes, 4 GPUs per node, isolate_gpu=False."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=7)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=False)

        assert len(placements) == 8
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == i // 4
            assert p.node_rank == i // 4
            assert p.local_accelerator_id == i % 4
            assert p.local_rank == i % 4
            assert p.local_world_size == 4
            assert p.visible_accelerators == [str(j) for j in range(4)]
            assert p.isolate_accelerator is False

    def test_packed_placement_strategy_node_offset(self):
        """Test PackedPlacementStrategy with node offset (start from node 1)."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=4, end_accelerator_id=7)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 1
            assert p.node_rank == 0
            assert p.local_accelerator_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.visible_accelerators == [str(i)]
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_partial_node(self):
        """Test PackedPlacementStrategy with 2 nodes, 4 GPUs per node, but only 1 node used."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=3)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_accelerator_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.visible_accelerators == [str(i)]
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_single_node_single_gpu_per_process(self):
        """Test PackedPlacementStrategy with 1 node, 4 GPUs per node, 1 GPU per process."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=3)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_accelerator_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.visible_accelerators == [str(i)]
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_single_node_multiple_gpus_per_process(self):
        """Test PackedPlacementStrategy with 1 node, 4 GPUs per node, 2 GPUs per process."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(
            start_accelerator_id=0, end_accelerator_id=3, num_accelerators_per_process=2
        )
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 2
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_accelerator_id == i * 2
            assert p.local_rank == i
            assert p.local_world_size == 2
            expected_devices = [str(i * 2), str(i * 2 + 1)]
            assert p.visible_accelerators == expected_devices
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_isolate_gpu_false(self):
        """Test PackedPlacementStrategy with isolate_gpu=False."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=3)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=False)

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_accelerator_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.visible_accelerators == [str(j) for j in range(4)]
            assert p.isolate_accelerator is False

    def test_packed_placement_strategy_master_gpu_offset(self):
        """Test PackedPlacementStrategy with master_gpu offset (start from GPU 2)."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=2, end_accelerator_id=3)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 2
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_accelerator_id == i + 2
            assert p.local_rank == i
            assert p.local_world_size == 2
            assert p.visible_accelerators == [str(i + 2)]
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_num_processes(self):
        """Test PackedPlacementStrategy with num_processes specified."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=0, end_accelerator_id=2)
        placements = strategy.get_placement(cluster=cluster, isolate_accelerator=True)

        assert len(placements) == 3
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_accelerator_id == i
            assert p.local_rank == i
            assert p.local_world_size == 3
            assert p.visible_accelerators == [str(i)]
            assert p.isolate_accelerator is True

    def test_packed_placement_strategy_invalid_start_end_gpu(self):
        """Test that specifying both start_gpu_id and end_gpu_id raises ValueError."""
        with pytest.raises(AssertionError):
            PackedPlacementStrategy(start_accelerator_id=4, end_accelerator_id=1)
            PackedPlacementStrategy(start_accelerator_id=4, end_accelerator_id=-1)
            PackedPlacementStrategy(start_accelerator_id=-4, end_accelerator_id=-1)

    def test_packed_placement_strategy_invalid_master_gpu(self):
        """Test that specifying master_gpu >= num_accelerators_per_node raises AssertionError."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = PackedPlacementStrategy(start_accelerator_id=4, end_accelerator_id=7)
        with pytest.raises(AssertionError):
            strategy.get_placement(cluster=cluster, isolate_accelerator=True)

    def test_packed_placement_strategy_invalid_num_gpus_per_process(self):
        """Test that specifying num_gpus_per_process that doesn't fit raises AssertionError."""
        with pytest.raises(AssertionError):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            strategy = PackedPlacementStrategy(
                start_accelerator_id=0,
                end_accelerator_id=2,
                num_accelerators_per_process=4,
            )
            strategy.get_placement(cluster=cluster, isolate_accelerator=True)

    def test_packed_placement_strategy_invalid_stride(self):
        """Test that specifying stride that doesn't fit raises AssertionError."""
        with pytest.raises(AssertionError):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            strategy = PackedPlacementStrategy(
                start_accelerator_id=0, end_accelerator_id=2, stride=7
            )
            strategy.get_placement(cluster=cluster, isolate_accelerator=True)

    def test_flex_placement_single_process_single_gpu(self):
        """Test placement for a single process with a single GPU."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0]])
        placements = strategy.get_placement(cluster=cluster)

        assert len(placements) == 1
        placement = placements[0]
        assert placement.rank == 0
        assert placement.node_id == 0
        assert placement.node_rank == 0
        assert placement.local_accelerator_id == 0
        assert placement.local_rank == 0
        assert placement.local_world_size == 1
        assert placement.visible_accelerators == ["0"]
        assert placement.isolate_accelerator is True

    def test_flex_placement_single_process_multiple_gpus(self):
        """Test placement for a single process with multiple GPUs."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[2, 1, 3]])
        placements = strategy.get_placement(cluster=cluster)

        assert len(placements) == 1
        placement = placements[0]
        assert placement.rank == 0
        assert placement.node_id == 0
        assert placement.node_rank == 0
        assert placement.local_accelerator_id == 1  # First GPU after sorting
        assert placement.local_rank == 0
        assert placement.local_world_size == 1
        assert placement.visible_accelerators == ["1", "2", "3"]  # Sorted local GPU IDs
        assert placement.isolate_accelerator is True

    def test_flex_placement_multiple_processes_same_node(self):
        """Test placement for multiple processes on the same node."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0], [1], [2]])
        placements = strategy.get_placement(cluster=cluster)

        assert len(placements) == 3

        # Check first process
        assert placements[0].rank == 0
        assert placements[0].node_id == 0
        assert placements[0].node_rank == 0
        assert placements[0].local_rank == 0
        assert placements[0].local_world_size == 3

        # Check second process
        assert placements[1].rank == 1
        assert placements[1].node_id == 0
        assert placements[1].node_rank == 0
        assert placements[1].local_rank == 1
        assert placements[1].local_world_size == 3

        # Check third process
        assert placements[2].rank == 2
        assert placements[2].node_id == 0
        assert placements[2].node_rank == 0
        assert placements[2].local_rank == 2
        assert placements[2].local_world_size == 3

    def test_flex_placement_multiple_processes_different_nodes(self):
        """Test placement for multiple processes across different nodes."""
        cluster = mock_cluster(num_nodes=3, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[0], [4], [8]])
        placements = strategy.get_placement(cluster=cluster)

        assert len(placements) == 3

        # Check first process (node 0)
        assert placements[0].rank == 0
        assert placements[0].node_id == 0
        assert placements[0].node_rank == 0
        assert placements[0].local_rank == 0
        assert placements[0].local_world_size == 1

        # Check second process (node 1)
        assert placements[1].rank == 1
        assert placements[1].node_id == 1
        assert placements[1].node_rank == 1
        assert placements[1].local_rank == 0
        assert placements[1].local_world_size == 1

        # Check third process (node 2)
        assert placements[2].rank == 2
        assert placements[2].node_id == 2
        assert placements[2].node_rank == 2
        assert placements[2].local_rank == 0
        assert placements[2].local_world_size == 1

    def test_flex_placement_mixed_nodes_and_processes(self):
        """Test placement for mixed scenario with multiple processes per node."""
        cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[1], [2], [4], [5]])
        placements = strategy.get_placement(cluster=cluster)

        assert len(placements) == 4

        # Node 0 processes
        assert placements[0].node_id == 0
        assert placements[0].node_rank == 0
        assert placements[0].local_rank == 0
        assert placements[0].local_world_size == 2

        assert placements[1].node_id == 0
        assert placements[1].node_rank == 0
        assert placements[1].local_rank == 1
        assert placements[1].local_world_size == 2

        # Node 1 processes
        assert placements[2].node_id == 1
        assert placements[2].node_rank == 1
        assert placements[2].local_rank == 0
        assert placements[2].local_world_size == 2

        assert placements[3].node_id == 1
        assert placements[3].node_rank == 1
        assert placements[3].local_rank == 1
        assert placements[3].local_world_size == 2

    def test_flex_placement_sorting_gpu_ids_within_process(self):
        """Test that GPU IDs are sorted within each process."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[3, 1, 2]])
        placements = strategy.get_placement(cluster=cluster)

        assert placements[0].visible_accelerators == ["1", "2", "3"]
        assert placements[0].local_accelerator_id == 1

    def test_flex_placement_sorting_processes_by_first_gpu(self):
        """Test that processes are sorted by their first GPU ID."""
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        strategy = FlexiblePlacementStrategy([[2], [0], [1]])
        placements = strategy.get_placement(cluster=cluster)

        assert len(placements) == 3
        assert placements[0].local_accelerator_id == 0  # Process with GPU 0
        assert placements[1].local_accelerator_id == 1  # Process with GPU 1
        assert placements[2].local_accelerator_id == 2  # Process with GPU 2

    def test_empty_gpu_ids_list_raises_error(self):
        """Test that empty GPU IDs list raises assertion error."""
        with pytest.raises(
            AssertionError, match="The accelerator_id_list must not be empty"
        ):
            FlexiblePlacementStrategy([])

    def test_overlapping_gpu_ids_raises_error(self):
        """Test that overlapping GPU IDs raise assertion error."""
        with pytest.raises(AssertionError, match="should not overlap"):
            FlexiblePlacementStrategy([[0, 1], [1, 2]])

    def test_out_of_range_gpu_id_raises_error(self):
        """Test that out of range GPU ID raises assertion error."""
        with pytest.raises(AssertionError, match="Accelerator ID 8 is out of range"):
            cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
            strategy = FlexiblePlacementStrategy([[0], [8]])
            strategy.get_placement(cluster=cluster)

    def test_cross_node_gpu_ids_raises_error(self):
        """Test that GPU IDs spanning multiple nodes raise assertion error."""
        with pytest.raises(AssertionError, match="must be on the same node"):
            cluster = mock_cluster(num_nodes=2, num_accelerators_per_node=4)
            strategy = FlexiblePlacementStrategy([[0, 4]])
            strategy.get_placement(cluster=cluster)

    def test_duplicate_gpu_ids_in_process_raises_error(self):
        """Test that duplicate GPU IDs within a process raise assertion error."""
        with pytest.raises(AssertionError, match="should not overlap"):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            strategy = FlexiblePlacementStrategy([[0, 0]])
            strategy.get_placement(cluster=cluster)

    def test_negative_gpu_id_raises_error(self):
        """Test that negative GPU IDs raise assertion error."""
        with pytest.raises(
            AssertionError, match="The start accelerator ID -1 must be non-negative"
        ):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            strategy = FlexiblePlacementStrategy([[-1]])
            strategy.get_placement(cluster=cluster)

    def test_init_valid_node_ids(self):
        """Test initialization with valid node IDs."""
        strategy = NodePlacementStrategy([0, 1, 2])
        assert strategy._node_ids == [0, 1, 2]
        assert strategy._placement_strategy == "NODE"

    def test_init_sorts_node_ids(self):
        """Test that node IDs are sorted during initialization."""
        strategy = NodePlacementStrategy([2, 0, 1])
        assert strategy._node_ids == [0, 1, 2]

    def test_init_empty_node_ids_raises_assertion(self):
        """Test that empty node IDs list raises assertion error."""
        with pytest.raises(AssertionError, match="The node_ids list must not be empty"):
            NodePlacementStrategy([])

    def test_get_placement_single_node(self):
        """Test placement generation for single node."""
        # Mock cluster
        cluster = MagicMock(spec=Cluster)
        cluster.num_nodes = 2
        cluster.get_node_num_accelerators.return_value = 2

        node_info = MagicMock()
        node_info.accelerator_type = "cuda"
        cluster.get_node_info.return_value = node_info

        strategy = NodePlacementStrategy([0, 0])
        placements = strategy.get_placement(cluster)

        assert len(placements) == 2
        assert placements[0].rank == 0
        assert placements[0].node_id == 0
        assert placements[0].node_rank == 0
        assert placements[0].local_rank == 0
        assert placements[0].local_world_size == 2

        assert placements[1].rank == 1
        assert placements[1].node_id == 0
        assert placements[1].node_rank == 0
        assert placements[1].local_rank == 1
        assert placements[1].local_world_size == 2

    def test_get_placement_multiple_nodes(self):
        """Test placement generation across multiple nodes."""
        # Mock cluster
        cluster = MagicMock(spec=Cluster)
        cluster.num_nodes = 3
        cluster.get_node_num_accelerators.return_value = 1

        node_info = MagicMock()
        node_info.accelerator_type = "cuda"
        cluster.get_node_info.return_value = node_info

        strategy = NodePlacementStrategy([0, 1, 1, 2])
        placements = strategy.get_placement(cluster)

        assert len(placements) == 4

        # Node 0
        assert placements[0].node_id == 0
        assert placements[0].node_rank == 0
        assert placements[0].local_rank == 0
        assert placements[0].local_world_size == 1

        # Node 1
        assert placements[1].node_id == 1
        assert placements[1].node_rank == 1
        assert placements[1].local_rank == 0
        assert placements[1].local_world_size == 2

        assert placements[2].node_id == 1
        assert placements[2].node_rank == 1
        assert placements[2].local_rank == 1
        assert placements[2].local_world_size == 2

        # Node 2
        assert placements[3].node_id == 2
        assert placements[3].node_rank == 2
        assert placements[3].local_rank == 0
        assert placements[3].local_world_size == 1

    def test_get_placement_no_accelerators(self):
        """Test placement generation when no accelerators available."""
        # Mock cluster
        cluster = MagicMock(spec=Cluster)
        cluster.num_nodes = 2
        cluster.get_node_num_accelerators.return_value = 0

        node_info = MagicMock()
        node_info.accelerator_type = "cpu"
        cluster.get_node_info.return_value = node_info

        strategy = NodePlacementStrategy([0])
        placements = strategy.get_placement(cluster)

        assert len(placements) == 1
        assert placements[0].local_accelerator_id == -1
        assert placements[0].visible_accelerators == []

    def test_get_placement_node_id_exceeds_cluster_size(self):
        """Test that node ID exceeding cluster size raises assertion error."""
        # Mock cluster
        cluster = MagicMock(spec=Cluster)
        cluster.num_nodes = 2
        cluster.get_node_num_accelerators.return_value = 2

        strategy = NodePlacementStrategy([0, 2])

        with pytest.raises(
            AssertionError, match="Node ID 2 exceeds number of available nodes 2"
        ):
            strategy.get_placement(cluster)

    def test_get_placement_isolate_accelerator_false(self):
        """Test placement generation with isolate_accelerator=False."""
        # Mock cluster
        cluster = MagicMock(spec=Cluster)
        cluster.num_nodes = 1
        cluster.get_node_num_accelerators.return_value = 2

        node_info = MagicMock()
        node_info.accelerator_type = "cuda"
        cluster.get_node_info.return_value = node_info

        strategy = NodePlacementStrategy([0])
        placements = strategy.get_placement(cluster, isolate_accelerator=False)

        assert len(placements) == 1
        assert not placements[0].isolate_accelerator
        assert placements[0].visible_accelerators == ["0", "1"]

    def test_hybrid_component_placement_generate_placements(self):
        """Test HybridComponentPlacement._generate_placements method."""

        # Create a mock config
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor,rollout": "0,1,2", "inference": "3"},
                }
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        placement = HybridComponentPlacement(config, cluster)

        # Initially placements should be empty
        assert len(placement._placements) == 0

        # Call _generate_placements
        placement._generate_placements()

        # Check that placements are generated for all components
        assert "actor" in placement._placements
        assert "rollout" in placement._placements
        assert "inference" in placement._placements

        # Check that FlexiblePlacementStrategy is created with correct GPU lists
        actor_strategy = placement._placements["actor"]
        rollout_strategy = placement._placements["rollout"]
        inference_strategy = placement._placements["inference"]

        # Verify that each GPU ID is in its own list (as per the implementation)
        # actor and rollout share GPUs 0,1,2
        expected_actor_gpu_lists = [0, 1, 2]
        expected_rollout_gpu_lists = [0, 1, 2]
        expected_inference_gpu_lists = [3]
        assert placement._component_gpu_map["actor"] == expected_actor_gpu_lists
        assert placement._component_gpu_map["rollout"] == expected_rollout_gpu_lists
        assert placement._component_gpu_map["inference"] == expected_inference_gpu_lists

        # We can't directly access the internal structure, but we can test the placement results
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        actor_placements = actor_strategy.get_placement(cluster=cluster)
        rollout_placements = rollout_strategy.get_placement(cluster=cluster)
        inference_placements = inference_strategy.get_placement(cluster=cluster)

        # Check that correct number of processes are created
        assert len(actor_placements) == 3  # GPUs 0,1,2
        assert len(rollout_placements) == 3  # GPUs 0,1,2
        assert len(inference_placements) == 1  # GPU 3

        # Check GPU assignments
        actor_gpus = sorted([p.local_accelerator_id for p in actor_placements])
        rollout_gpus = sorted([p.local_accelerator_id for p in rollout_placements])
        inference_gpus = [p.local_accelerator_id for p in inference_placements]

        assert actor_gpus == [0, 1, 2]
        assert rollout_gpus == [0, 1, 2]
        assert inference_gpus == [3]

    def test_hybrid_component_placement_generate_placements_all_gpus(self):
        """Test HybridComponentPlacement._generate_placements with 'all' GPU specification."""

        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor": "all"},
                }
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        placement = HybridComponentPlacement(config, cluster)
        placement._generate_placements()

        assert "actor" in placement._placements

        actor_strategy = placement._placements["actor"]
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        actor_placements = actor_strategy.get_placement(cluster=cluster)

        # Should create one process per GPU
        assert len(actor_placements) == 4

        actor_gpus = sorted([p.local_accelerator_id for p in actor_placements])
        assert actor_gpus == [0, 1, 2, 3]

    def test_hybrid_component_placement_generate_placements_gpu_ranges(self):
        """Test HybridComponentPlacement._generate_placements with GPU ranges."""

        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor": "0-2,5,7", "inference": "3-4"},
                }
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
        placement = HybridComponentPlacement(config, cluster)
        placement._generate_placements()

        # Check actor placement (GPUs 0,1,2,5,7)
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
        actor_strategy = placement._placements["actor"]
        actor_placements = actor_strategy.get_placement(cluster=cluster)
        assert len(actor_placements) == 5

        actor_gpus = sorted([p.local_accelerator_id for p in actor_placements])
        assert actor_gpus == [0, 1, 2, 5, 7]

        # Check inference placement (GPUs 3,4)
        inference_strategy = placement._placements["inference"]
        inference_placements = inference_strategy.get_placement(cluster=cluster)
        assert len(inference_placements) == 2

        inference_gpus = sorted([p.local_accelerator_id for p in inference_placements])
        assert inference_gpus == [3, 4]

    def test_hybrid_component_placement_generate_placements_single_gpu(self):
        """Test HybridComponentPlacement._generate_placements with single GPU per component."""

        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0",
                        "rollout": "1",
                        "inference": "2",
                    },
                }
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        placement = HybridComponentPlacement(config, cluster)
        placement._generate_placements()

        # Each component should have exactly one process
        for component in ["actor", "rollout", "inference"]:
            strategy = placement._placements[component]
            placements = strategy.get_placement(cluster)
            assert len(placements) == 1

        # Check GPU assignments
        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        actor_gpu = (
            placement._placements["actor"]
            .get_placement(cluster=cluster)[0]
            .local_accelerator_id
        )
        rollout_gpu = (
            placement._placements["rollout"]
            .get_placement(cluster=cluster)[0]
            .local_accelerator_id
        )
        inference_gpu = (
            placement._placements["inference"]
            .get_placement(cluster=cluster)[0]
            .local_accelerator_id
        )

        assert actor_gpu == 0
        assert rollout_gpu == 1
        assert inference_gpu == 2

    def test_hybrid_component_placement_generate_placements_called_by_get_strategy(
        self,
    ):
        """Test that _generate_placements is automatically called when get_strategy is used."""

        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor": "0,1"},
                }
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        placement = HybridComponentPlacement(config, cluster)

        # Initially placements should be empty
        assert len(placement._placements) == 0

        # Calling get_strategy should trigger _generate_placements
        strategy = placement.get_strategy("actor")

        # Now placements should be populated
        assert len(placement._placements) > 0
        assert "actor" in placement._placements
        assert placement._placements["actor"] == strategy

    def test_model_parallel_component_placement_init_collocated_mode(self):
        """Test ModelParallelComponentPlacement initialization in collocated mode."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor,rollout": "0-3"},
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 4,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
        placement = ModelParallelComponentPlacement(config, cluster)

        assert placement._placement_mode.name == "COLLOCATED"
        assert placement._actor_gpus == [0, 1, 2, 3]
        assert placement._rollout_gpus == [0, 1, 2, 3]
        assert placement._inference_gpus is None
        assert placement._actor_num_gpus == 4
        assert placement._rollout_num_gpus == 4
        assert placement._inference_num_gpus == 0
        assert placement.is_disaggregated is False

    def test_model_parallel_component_placement_init_disaggregated_mode(self):
        """Test ModelParallelComponentPlacement initialization in disaggregated mode."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-5",
                        "inference": "6-7",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": True},
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
        placement = ModelParallelComponentPlacement(config, cluster)

        assert placement._placement_mode.name == "DISAGGREGATED"
        assert placement._actor_gpus == [0, 1]
        assert placement._rollout_gpus == [2, 3, 4, 5]
        assert placement._inference_gpus == [6, 7]
        assert placement._actor_num_gpus == 2
        assert placement._rollout_num_gpus == 4
        assert placement._inference_num_gpus == 2
        assert placement.is_disaggregated is True
        assert placement.has_dedicated_inference is True

    def test_model_parallel_component_placement_init_missing_actor_gpus(self):
        """Test ModelParallelComponentPlacement raises error when actor GPUs are missing."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"rollout": "0-3"},
                }
            }
        )

        with pytest.raises(AssertionError, match="Actor GPUs must be specified"):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_missing_rollout_gpus(self):
        """Test ModelParallelComponentPlacement raises error when rollout GPUs are missing."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor": "0-3"},
                }
            }
        )

        with pytest.raises(AssertionError, match="Rollout GPUs must be specified"):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_collocated_mode_with_inference_gpus(
        self,
    ):
        """Test ModelParallelComponentPlacement raises error when inference GPUs are specified in collocated mode."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor,rollout": "0-1", "inference": "2-3"},
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            }
        )

        with pytest.raises(
            AssertionError,
            match="Inference GPUs must not be specified in collocated mode",
        ):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_disaggregated_mode_invalid_inference_tp_size(
        self,
    ):
        """Test ModelParallelComponentPlacement raises error when inference TP size > inference world size."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-5",
                        "inference": "6-7",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 4,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": True},
            }
        )

        with pytest.raises(
            AssertionError,
            match="Inference TP size 4 must be less than or equal to Inference world size 2",
        ):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_disaggregated_mode_recompute_logprobs_false(
        self,
    ):
        """Test ModelParallelComponentPlacement raises error when recompute_logprobs is False but inference GPUs specified."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-1",
                        "rollout": "2-5",
                        "inference": "6-7",
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
                "inference": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "algorithm": {"recompute_logprobs": False},
            }
        )

        with pytest.raises(
            AssertionError, match="algorithm.recompute_logprobs has been set to false"
        ):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_invalid_placement_mode(self):
        """Test ModelParallelComponentPlacement raises error for invalid placement (neither collocated nor disaggregated)."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {
                        "actor": "0-2",
                        "rollout": "1-4",  # Overlapping GPUs
                    },
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            }
        )

        with pytest.raises(
            ValueError,
            match="The specified placement does not match either the collocated mode",
        ):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_actor_tp_size_exceeds_world_size(
        self,
    ):
        """Test ModelParallelComponentPlacement raises error when actor TP size > actor world size."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor,rollout": "0-1"},
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 4,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            }
        )

        with pytest.raises(
            AssertionError,
            match="Actor TP size 4 must be less than or equal to Actor world size 2",
        ):
            cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=4)
            ModelParallelComponentPlacement(config, cluster)

    def test_model_parallel_component_placement_init_disaggregated_mode_without_inference(
        self,
    ):
        """Test ModelParallelComponentPlacement initialization in disaggregated mode without inference GPUs."""
        config = DictConfig(
            {
                "cluster": {
                    "num_nodes": 1,
                    "component_placement": {"actor": "0-1", "rollout": "2-5"},
                },
                "actor": {
                    "model": {
                        "tensor_model_parallel_size": 2,
                        "context_parallel_size": 1,
                        "pipeline_model_parallel_size": 1,
                    }
                },
                "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
            }
        )

        cluster = mock_cluster(num_nodes=1, num_accelerators_per_node=8)
        placement = ModelParallelComponentPlacement(config, cluster)

        assert placement._placement_mode.name == "DISAGGREGATED"
        assert placement._inference_gpus is None
        assert placement._inference_num_gpus == 0
        assert placement.has_dedicated_inference is False


if __name__ == "__main__":
    pytest.main(["-v", __file__])
