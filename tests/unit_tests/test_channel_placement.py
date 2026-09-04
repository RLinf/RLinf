# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Placement resolution for non-distributed channels.

A channel that defaults to node 0 puts itself a network hop away from both ends
of its traffic whenever the workers it serves run elsewhere. These tests cover
the resolution that picks a node from the channel's own producers and consumers.

Groups are resolved by name through the worker manager, because a worker group's
own ``worker_info_list`` holds ``WorkerRank`` entries -- an actor handle and a
rank -- and carries no placement. The stub below therefore mirrors the manager,
not the group, and the fake group exposes only ``worker_group_name``, the single
attribute ``resolve_group_names`` reads in production.
"""

from types import SimpleNamespace

import pytest

from rlinf.scheduler.cluster import resolve_colocation_node_rank

# Where each named group is placed, for the stubbed manager to report.
NODE_OF_GROUP = {"Rollout": 2, "Actor": 3, "Env": 1}


class _FakeGroup:
    """A worker group as ``resolve_group_names`` sees it: just a name."""

    def __init__(self, name):
        self.worker_group_name = name


@pytest.fixture(autouse=True)
def stub_manager(monkeypatch):
    """Report placements by group name, as the real worker manager does."""

    class _Proxy:
        def get_worker_info(self, address):
            assert address.rank == 0, "placement is read from rank 0"
            node = NODE_OF_GROUP.get(address.root_group_name)
            return None if node is None else SimpleNamespace(cluster_node_rank=node)

    import rlinf.scheduler.manager as manager_mod

    monkeypatch.setattr(
        manager_mod.WorkerManager, "get_proxy", staticmethod(lambda: _Proxy())
    )


@pytest.fixture
def group_type(monkeypatch):
    """Make the name resolver accept _FakeGroup in place of WorkerGroup."""
    import rlinf.scheduler.cluster.utils as utils_mod
    import rlinf.scheduler.worker as worker_mod

    monkeypatch.setattr(worker_mod, "WorkerGroup", _FakeGroup)
    monkeypatch.setattr(utils_mod, "WorkerGroup", _FakeGroup, raising=False)


def test_resolves_group_object_to_its_node(group_type):
    """A group passed as an object resolves through its name."""
    assert resolve_colocation_node_rank(_FakeGroup("Rollout")) == 2


def test_resolves_group_name(group_type):
    """A group passed as a bare name resolves the same way."""
    assert resolve_colocation_node_rank("Actor") == 3


def test_prefers_producer_over_consumer(group_type):
    """Producers win: the collector runs channel-side, before the consumer hop."""
    assert resolve_colocation_node_rank("Rollout", "Actor") == 2


def test_falls_back_to_consumer(group_type):
    """With no producer given, the consumer still pins the channel."""
    assert resolve_colocation_node_rank(None, "Actor") == 3


def test_accepts_iterables_and_skips_unknown_groups(group_type):
    """A group the manager does not know is skipped, not chosen."""
    assert resolve_colocation_node_rank(["NotLaunched", _FakeGroup("Env")]) == 1


def test_returns_none_when_nothing_resolves(group_type):
    """No producers and no consumers leaves the caller to pick a default."""
    assert resolve_colocation_node_rank(None, None) is None


def test_unregistered_group_name_is_skipped(group_type):
    """An unknown name yields None instead of raising, so creation can proceed."""
    assert resolve_colocation_node_rank("Missing") is None
