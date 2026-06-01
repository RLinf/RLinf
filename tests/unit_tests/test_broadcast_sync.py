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

import os
from types import SimpleNamespace

import pytest
import torch

from rlinf.scheduler import (
    Cluster,
    NodePlacementStrategy,
    PackedPlacementStrategy,
    Worker,
)
from rlinf.scheduler.collective.collective_group import CollectiveGroup

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def accelerator_is_available():
    return (
        Worker.torch_platform is not None
        and hasattr(Worker.torch_platform, "is_available")
        and Worker.torch_platform.is_available()
    )


def accelerator_device_count():
    if Worker.torch_platform is None or not hasattr(
        Worker.torch_platform, "device_count"
    ):
        return 0
    return Worker.torch_platform.device_count()


def is_accel():
    """Return True only on CUDA; IPC broadcast is not supported on NPU."""
    return accelerator_is_available()


# ---------------------------------------------------------------------------
# Unit tests for _classify_broadcast_ranks – no hardware needed
# ---------------------------------------------------------------------------


def _make_fake_group(worker_specs):
    """Minimal stub satisfying _classify_broadcast_ranks' _group_info.workers access.

    worker_specs: list of (cluster_node_rank, available_accelerators) tuples.
    """
    workers = [
        SimpleNamespace(cluster_node_rank=node, available_accelerators=list(devs))
        for node, devs in worker_specs
    ]
    return SimpleNamespace(_group_info=SimpleNamespace(workers=workers))


def _classify(worker_specs, src_rank=0):
    fake = _make_fake_group(worker_specs)
    return CollectiveGroup._classify_broadcast_ranks(fake, src_rank)


class TestClassifyBroadcastRanks:
    """Unit tests for CollectiveGroup._classify_broadcast_ranks.

    Verifies all three output buckets:
      definitely_same – both workers have exactly one accelerator and it matches
      uncertain       – overlapping multi-device sets that need a runtime exchange
      definitely_diff – no device overlap or different cluster node
    """

    def test_single_device_same_gpu(self):
        """Two workers, same node, identical single GPU → definitely_same."""
        same, uncertain, diff = _classify([(0, [0]), (0, [0])])
        assert same == [1] and uncertain == [] and diff == []

    def test_single_device_different_gpu(self):
        """Two workers, same node, distinct single GPUs → definitely_diff."""
        same, uncertain, diff = _classify([(0, [0]), (0, [1])])
        assert same == [] and uncertain == [] and diff == [1]

    def test_different_nodes_same_device_index(self):
        """Same device index but different cluster nodes → definitely_diff."""
        same, uncertain, diff = _classify([(0, [0]), (1, [0])])
        assert same == [] and uncertain == [] and diff == [1]

    def test_multi_device_overlap_both_multi(self):
        """src=[0,1] dst=[0,1]: overlapping multi-device sets → uncertain."""
        same, uncertain, diff = _classify([(0, [0, 1]), (0, [0, 1])])
        assert same == [] and uncertain == [1] and diff == []

    def test_src_multi_dst_single_overlap(self):
        """src=[0,1] dst=[0]: overlap exists but src has multiple devices → uncertain."""
        same, uncertain, diff = _classify([(0, [0, 1]), (0, [0])])
        assert same == [] and uncertain == [1] and diff == []

    def test_src_single_dst_multi_overlap(self):
        """src=[0] dst=[0,1]: overlap exists but dst has multiple devices → uncertain."""
        same, uncertain, diff = _classify([(0, [0]), (0, [0, 1])])
        assert same == [] and uncertain == [1] and diff == []

    def test_multi_device_no_overlap(self):
        """src=[0,1] dst=[2,3]: both multi-device but no intersection → definitely_diff."""
        same, uncertain, diff = _classify([(0, [0, 1]), (0, [2, 3])])
        assert same == [] and uncertain == [] and diff == [1]

    def test_multi_worker_mixed_classification(self):
        """Five workers covering all three buckets simultaneously."""
        # rank0 = src [0]
        # rank1 = same node, [0]     → definitely_same
        # rank2 = same node, [1]     → definitely_diff
        # rank3 = same node, [0, 1]  → uncertain
        # rank4 = node 1,   [0]      → definitely_diff (different node)
        specs = [(0, [0]), (0, [0]), (0, [1]), (0, [0, 1]), (1, [0])]
        same, uncertain, diff = _classify(specs, src_rank=0)
        assert same == [1]
        assert uncertain == [3]
        assert set(diff) == {2, 4}

    def test_non_zero_src_rank(self):
        """Classification is correct when src is not rank 0."""
        # rank0=[0], rank1=[0] (src), rank2=[1]
        same, uncertain, diff = _classify([(0, [0]), (0, [0]), (0, [1])], src_rank=1)
        assert same == [0] and uncertain == [] and diff == [2]


# ---------------------------------------------------------------------------
# Integration tests – CUDA only; skipped on NPU
# ---------------------------------------------------------------------------

_ACTOR_SAME = "bcast_sync_actor_same"
_ROLLOUT_SAME = "bcast_sync_rollout_same"
_ACTOR_DIFF = "bcast_sync_actor_diff"
_ROLLOUT_DIFF = "bcast_sync_rollout_diff"
_ACTOR_MIXED = "bcast_sync_actor_mixed"
_ROLLOUT_MIXED = "bcast_sync_rollout_mixed"


class _BroadcastWorker(Worker):
    def __init__(self):
        super().__init__()
        Worker.torch_platform.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def run(self, groups, value, is_src):
        device = f"{Worker.torch_device_type}:{Worker.torch_platform.current_device()}"
        payload = torch.full((4, 4), float(value), device=device) if is_src else None
        return self.broadcast(payload, groups=groups)


@pytest.fixture(scope="module")
def cluster():
    if not is_accel():
        pytest.skip("Hybrid broadcast IPC integration tests require CUDA.")
    return Cluster(num_nodes=1)


@pytest.fixture(scope="class")
def same_gpu_groups(cluster):
    """Both worker groups pinned to GPU 0 – exercises the IPC (definitely-same) path."""
    placement = NodePlacementStrategy([0])
    actor = _BroadcastWorker.create_group().launch(
        cluster=cluster, placement_strategy=placement, name=_ACTOR_SAME
    )
    rollout = _BroadcastWorker.create_group().launch(
        cluster=cluster, placement_strategy=placement, name=_ROLLOUT_SAME
    )
    yield actor, rollout
    actor._close()
    rollout._close()


@pytest.fixture(scope="class")
def diff_gpu_groups(cluster):
    """Actor on GPU 0, rollout on GPU 1 – exercises the NCCL sub-group (diff-device) path."""
    if accelerator_device_count() < 2:
        pytest.skip("Different-GPU broadcast test requires at least 2 GPUs.")
    actor = _BroadcastWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=PackedPlacementStrategy(0, 0),
        name=_ACTOR_DIFF,
    )
    rollout = _BroadcastWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=PackedPlacementStrategy(1, 1),
        name=_ROLLOUT_DIFF,
    )
    yield actor, rollout
    actor._close()
    rollout._close()


@pytest.fixture(scope="class")
def mixed_gpu_groups(cluster):
    """Actor on GPU 0; rollout group with one worker on GPU 0 and one on GPU 1.

    The rollout receiver on GPU 0 is same-device as the actor (IPC path),
    while the rollout receiver on GPU 1 is different-device (collective
    sub-group path). Exercises the hybrid path that uses both routes in a
    single broadcast call.
    """
    if accelerator_device_count() < 2:
        pytest.skip("Mixed-GPU broadcast test requires at least 2 GPUs.")
    actor = _BroadcastWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=PackedPlacementStrategy(0, 0),
        name=_ACTOR_MIXED,
    )
    rollout = _BroadcastWorker.create_group().launch(
        cluster=cluster,
        placement_strategy=PackedPlacementStrategy(0, 1),
        name=_ROLLOUT_MIXED,
    )
    yield actor, rollout
    actor._close()
    rollout._close()


class TestBroadcastHybridSync:
    """Integration tests for the hybrid IPC / NCCL-sub-group broadcast routing.

    Skipped entirely on NPU because CUDA IPC is not available there.
    """

    def _run(self, actor_g, rollout_g, actor_name, rollout_name, value=42.0):
        groups = [(actor_name, [0]), (rollout_name, [0])]
        actor_h = actor_g.run(groups, value, is_src=True)
        rollout_h = rollout_g.run(groups, value, is_src=False)
        return actor_h.wait()[0], rollout_h.wait()[0]

    def test_same_gpu_broadcast_value(self, same_gpu_groups):
        """Both groups on GPU 0: broadcast takes the IPC path and delivers the correct value."""
        actor_g, rollout_g = same_gpu_groups
        actor_r, rollout_r = self._run(actor_g, rollout_g, _ACTOR_SAME, _ROLLOUT_SAME)
        expected = torch.full((4, 4), 42.0)
        assert torch.equal(actor_r.cpu(), expected)
        assert torch.equal(rollout_r.cpu(), expected)

    def test_same_gpu_broadcast_result_on_accelerator(self, same_gpu_groups):
        """Received tensor stays on the accelerator (not migrated to CPU by IPC path)."""
        actor_g, rollout_g = same_gpu_groups
        _, rollout_r = self._run(
            actor_g, rollout_g, _ACTOR_SAME, _ROLLOUT_SAME, value=7.0
        )
        assert rollout_r.device.type == Worker.torch_device_type

    def test_same_gpu_broadcast_repeated(self, same_gpu_groups):
        """Multiple consecutive broadcasts stay correct (IPC comm_id counters stay in sync)."""
        actor_g, rollout_g = same_gpu_groups
        for v in [1.0, 2.0, 3.0]:
            _, rollout_r = self._run(
                actor_g, rollout_g, _ACTOR_SAME, _ROLLOUT_SAME, value=v
            )
            expected = torch.full((4, 4), v)
            assert torch.equal(rollout_r.cpu(), expected), f"Mismatch at value={v}"

    def test_diff_gpu_broadcast_value(self, diff_gpu_groups):
        """Actor on GPU 0, rollout on GPU 1: broadcast uses the NCCL sub-group path."""
        actor_g, rollout_g = diff_gpu_groups
        actor_r, rollout_r = self._run(actor_g, rollout_g, _ACTOR_DIFF, _ROLLOUT_DIFF)
        expected = torch.full((4, 4), 42.0)
        assert torch.equal(actor_r.cpu(), expected)
        assert torch.equal(rollout_r.cpu(), expected)

    def _run_mixed(self, actor_g, rollout_g, value=17.0):
        """Drive a broadcast with one same-GPU receiver and one diff-GPU receiver."""
        groups = [(_ACTOR_MIXED, [0]), (_ROLLOUT_MIXED, [0, 1])]
        actor_h = actor_g.run(groups, value, is_src=True)
        rollout_h = rollout_g.run(groups, value, is_src=False)
        return actor_h.wait(), rollout_h.wait()

    def test_mixed_gpu_broadcast_value(self, mixed_gpu_groups):
        """Hybrid path: src + one same-GPU receiver (IPC) + one diff-GPU receiver
        (collective sub-group), all in a single broadcast call.
        """
        actor_g, rollout_g = mixed_gpu_groups
        actor_results, rollout_results = self._run_mixed(actor_g, rollout_g)
        expected = torch.full((4, 4), 17.0)
        assert len(actor_results) == 1
        assert torch.equal(actor_results[0].cpu(), expected)
        assert len(rollout_results) == 2, (
            "expected 2 rollout receivers for the mixed topology"
        )
        for r in rollout_results:
            assert torch.equal(r.cpu(), expected)

    def test_mixed_gpu_broadcast_repeated(self, mixed_gpu_groups):
        """Multiple consecutive mixed broadcasts stay correct so IPC and
        sub-group comm_id counters remain in sync across calls.
        """
        actor_g, rollout_g = mixed_gpu_groups
        for v in [4.0, 5.0, 6.0]:
            _, rollout_results = self._run_mixed(actor_g, rollout_g, value=v)
            expected = torch.full((4, 4), v)
            for r in rollout_results:
                assert torch.equal(r.cpu(), expected), f"Mismatch at value={v}"


if __name__ == "__main__":
    pytest.main(["-v", __file__])
