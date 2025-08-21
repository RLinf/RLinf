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

import pytest
import torch

from rlinf.scheduler import (
    Cluster,
    PackedPlacementStrategy,
    StridedPlacementStrategy,
    Worker,
)


# Fixture to provide a ClusterResource instance for the test session
@pytest.fixture(scope="module")
def cluster():
    """Provides a ClusterResource instance for the tests."""
    # Use a fixed number of GPUs for consistent testing.
    # A warning will be issued if an instance already exists, which is fine for testing.
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 4
    if num_gpus < 4:
        pytest.skip("Placement tests require at least 4 GPUs.")
    return Cluster(num_nodes=1, num_gpus_per_node=num_gpus)


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
            "gpu_id": self._gpu_id,
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
        num_gpus = cluster.num_gpus_per_node
        placement = PackedPlacementStrategy(master_node=0, num_nodes=1)
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
        num_workers_to_place = 2
        placement = PackedPlacementStrategy(
            master_node=0, master_gpu=0, num_processes=num_workers_to_place
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
        stride = 2
        num_gpus = cluster.num_gpus_per_node
        expected_num_workers = num_gpus // stride

        placement = PackedPlacementStrategy(
            master_node=0, num_nodes=1, num_gpus_per_process=stride
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
        """Verify that StridedPlacementStrategy allocates GPUs in a strided manner."""

        num_gpus = cluster.num_gpus_per_node
        stride = 2
        num_gpus_per_process = 2

        # Only run if the number of GPUs is divisible by stride * num_gpus_per_process
        if (
            num_gpus < stride * num_gpus_per_process
            or num_gpus % (stride * num_gpus_per_process) != 0
        ):
            pytest.skip(
                "StridedPlacementStrategy test requires num_gpus to be divisible by stride * num_gpus_per_process."
            )

        expected_num_workers = (num_gpus // (stride * num_gpus_per_process)) * stride

        placement = StridedPlacementStrategy(
            master_node=0,
            num_nodes=1,
            stride=stride,
            num_gpus_per_process=num_gpus_per_process,
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
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=2)
        placements = strategy.get_placement(
            num_nodes_in_cluster=2, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 8
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == i // 4
            assert p.node_rank == i // 4
            assert p.local_gpu_id == i % 4
            assert p.local_rank == i % 4
            assert p.local_world_size == 4
            assert p.cuda_visible_devices == [str(i % 4)]
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_multiple_nodes_multiple_gpus_per_process(self):
        """Test PackedPlacementStrategy with 2 nodes, 4 GPUs per node, 2 GPUs per process."""
        strategy = PackedPlacementStrategy(
            master_node=0, num_nodes=2, num_gpus_per_process=2
        )
        placements = strategy.get_placement(
            num_nodes_in_cluster=2, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == i // 2
            assert p.node_rank == i // 2
            assert p.local_gpu_id == (i % 2) * 2
            assert p.local_rank == i % 2
            assert p.local_world_size == 2
            expected_devices = [str((i % 2) * 2), str((i % 2) * 2 + 1)]
            assert p.cuda_visible_devices == expected_devices
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_multiple_nodes_no_isolate(self):
        """Test PackedPlacementStrategy with 2 nodes, 4 GPUs per node, isolate_gpu=False."""
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=2)
        placements = strategy.get_placement(
            num_nodes_in_cluster=2, num_gpus_per_node=4, isolate_gpu=False
        )

        assert len(placements) == 8
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == i // 4
            assert p.node_rank == i // 4
            assert p.local_gpu_id == i % 4
            assert p.local_rank == i % 4
            assert p.local_world_size == 4
            assert p.cuda_visible_devices == [str(j) for j in range(4)]
            assert p.isolate_gpu is False

    def test_packed_placement_strategy_master_node_offset(self):
        """Test PackedPlacementStrategy with master_node offset (start from node 1)."""
        strategy = PackedPlacementStrategy(master_node=1, num_nodes=1)
        placements = strategy.get_placement(
            num_nodes_in_cluster=2, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 1
            assert p.node_rank == 0
            assert p.local_gpu_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.cuda_visible_devices == [str(i)]
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_partial_node(self):
        """Test PackedPlacementStrategy with 2 nodes, 4 GPUs per node, but only 1 node used."""
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=1)
        placements = strategy.get_placement(
            num_nodes_in_cluster=2, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_gpu_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.cuda_visible_devices == [str(i)]
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_single_node_single_gpu_per_process(self):
        """Test PackedPlacementStrategy with 1 node, 4 GPUs per node, 1 GPU per process."""
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=1)
        placements = strategy.get_placement(
            num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_gpu_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.cuda_visible_devices == [str(i)]
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_single_node_multiple_gpus_per_process(self):
        """Test PackedPlacementStrategy with 1 node, 4 GPUs per node, 2 GPUs per process."""
        strategy = PackedPlacementStrategy(
            master_node=0, num_nodes=1, num_gpus_per_process=2
        )
        placements = strategy.get_placement(
            num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 2
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_gpu_id == i * 2
            assert p.local_rank == i
            assert p.local_world_size == 2
            expected_devices = [str(i * 2), str(i * 2 + 1)]
            assert p.cuda_visible_devices == expected_devices
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_isolate_gpu_false(self):
        """Test PackedPlacementStrategy with isolate_gpu=False."""
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=1)
        placements = strategy.get_placement(
            num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=False
        )

        assert len(placements) == 4
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_gpu_id == i
            assert p.local_rank == i
            assert p.local_world_size == 4
            assert p.cuda_visible_devices == [str(j) for j in range(4)]
            assert p.isolate_gpu is False

    def test_packed_placement_strategy_master_gpu_offset(self):
        """Test PackedPlacementStrategy with master_gpu offset (start from GPU 2)."""
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=1, master_gpu=2)
        placements = strategy.get_placement(
            num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 2
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_gpu_id == i + 2
            assert p.local_rank == i
            assert p.local_world_size == 2
            assert p.cuda_visible_devices == [str(i + 2)]
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_num_processes(self):
        """Test PackedPlacementStrategy with num_processes specified."""
        strategy = PackedPlacementStrategy(master_node=0, num_processes=3)
        placements = strategy.get_placement(
            num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=True
        )

        assert len(placements) == 3
        for i, p in enumerate(placements):
            assert p.rank == i
            assert p.node_id == 0
            assert p.node_rank == 0
            assert p.local_gpu_id == i
            assert p.local_rank == i
            assert p.local_world_size == 3
            assert p.cuda_visible_devices == [str(i)]
            assert p.isolate_gpu is True

    def test_packed_placement_strategy_invalid_both_num_nodes_and_num_processes(self):
        """Test that specifying both num_nodes and num_processes raises ValueError."""
        with pytest.raises(ValueError):
            PackedPlacementStrategy(master_node=0, num_nodes=1, num_processes=2)

    def test_packed_placement_strategy_invalid_neither_num_nodes_nor_num_processes(
        self,
    ):
        """Test that not specifying num_nodes or num_processes raises ValueError."""
        with pytest.raises(ValueError):
            PackedPlacementStrategy(master_node=0)

    def test_packed_placement_strategy_invalid_master_gpu(self):
        """Test that specifying master_gpu >= num_gpus_per_node raises AssertionError."""
        strategy = PackedPlacementStrategy(master_node=0, num_nodes=1, master_gpu=4)
        with pytest.raises(AssertionError):
            strategy.get_placement(
                num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=True
            )

    def test_packed_placement_strategy_invalid_num_gpus_per_process(self):
        """Test that specifying num_gpus_per_process that doesn't fit raises AssertionError."""
        strategy = PackedPlacementStrategy(
            master_node=0, num_processes=2, num_gpus_per_process=3
        )
        with pytest.raises(AssertionError):
            strategy.get_placement(
                num_nodes_in_cluster=1, num_gpus_per_node=4, isolate_gpu=True
            )


if __name__ == "__main__":
    pytest.main(["-v", __file__])
