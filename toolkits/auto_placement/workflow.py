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


from collections import defaultdict, deque
from typing import Optional

from node import ComponentNode, SccNode


class Workflow:
    """The Workflow class represents a directed acyclic graph (DAG) of nodes.

    Args:
        workflow: A dictionary of nodes and their neighbors.

    Methods:
        _topological_sort: Perform topological sort on the workflow(graph).
        _find_sccs: Find strongly connected components (SCCs) using Tarjan's algorithm.
        compress_sccs: Compress strongly connected components (SCCs) into single nodes to build a directed acyclic graph (DAG).

    """

    def __init__(self, graph: dict[ComponentNode, list[ComponentNode]]):
        node_set: set[ComponentNode] = set()
        for node, neighbors in graph.items():
            node_set.add(node)
            for neighbor in neighbors:
                node_set.add(neighbor)

        self.nodes: list[ComponentNode] = list(node_set)
        self.graph: dict[ComponentNode, list[ComponentNode]] = graph
        self.topological_order: list[ComponentNode] = self._topological_sort()

        # gpu_num -> time
        self._profile_cache: dict[int, float] = {}

    def _find_sccs(self) -> list[list[ComponentNode]]:
        """Find strongly connected components (SCCs) using Tarjan's algorithm."""

        def tarjan_dfs(node, disc, low, stack, in_stack, time):
            disc[node] = low[node] = time[0]
            time[0] += 1
            stack.append(node)
            in_stack.add(node)

            for neighbor in self.get_neighbors(node):
                if neighbor not in disc:
                    tarjan_dfs(neighbor, disc, low, stack, in_stack, time)
                    low[node] = min(low[node], low[neighbor])
                elif neighbor in in_stack:
                    low[node] = min(low[node], disc[neighbor])

            if low[node] == disc[node]:
                scc = []
                while True:
                    top = stack.pop()
                    in_stack.remove(top)
                    scc.append(top)
                    if top == node:
                        break
                sccs.append(scc)

        sccs = []
        disc = {}
        low = {}
        stack = []
        in_stack = set()
        time = [0]

        for node in self.nodes:
            if node not in disc:
                tarjan_dfs(node, disc, low, stack, in_stack, time)

        return sccs

    def compress_sccs(self) -> "Workflow":
        """Compress strongly connected components (SCCs) into single nodes to build a directed acyclic graph (DAG)"""
        sccs: list[list[ComponentNode]] = self._find_sccs()
        node_to_scc: dict[ComponentNode, int] = {}
        for scc_idx, scc in enumerate(sccs):
            for node in scc:
                node_to_scc[node] = scc_idx

        # Build compressed graph using Workflow format
        compressed_workflow: dict[ComponentNode, list[ComponentNode]] = {}

        # Create compressed node for each SCC
        for scc_idx, scc in enumerate(sccs):
            if len(scc) == 1:
                compressed_node = scc[0]
            else:
                compressed_node = SccNode(scc)

            compressed_workflow[compressed_node] = []

            for node in scc:
                for neighbor in self.get_neighbors(node):
                    neighbor_scc = node_to_scc[neighbor]
                    if neighbor_scc != scc_idx:
                        # Find corresponding compressed node
                        target_compressed_node = None
                        for existing_node in compressed_workflow.keys():
                            if existing_node in compressed_workflow:
                                if len(sccs[neighbor_scc]) == 1:
                                    if existing_node == sccs[neighbor_scc][0]:
                                        target_compressed_node = existing_node
                                        break
                                else:
                                    if (
                                        isinstance(existing_node, SccNode)
                                        and existing_node.nodes == sccs[neighbor_scc]
                                    ):
                                        target_compressed_node = existing_node
                                        break

                        if (
                            target_compressed_node
                            and target_compressed_node
                            not in compressed_workflow[compressed_node]
                        ):
                            compressed_workflow[compressed_node].append(
                                target_compressed_node
                            )

        return Workflow(compressed_workflow)

    def _topological_sort(self) -> list[ComponentNode]:
        """Perform topological sort on the workflow(graph)"""
        in_degree = defaultdict(int)
        for node in self.nodes:
            for neighbor in self.get_neighbors(node):
                in_degree[neighbor] += 1

        queue = deque([node for node in self.nodes if in_degree[node] == 0])
        topological_order = []

        while queue:
            node = queue.popleft()
            topological_order.append(node)

            for neighbor in self.get_neighbors(node):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return topological_order

    def get_neighbors(self, node: ComponentNode) -> list[ComponentNode]:
        return self.graph.get(node, [])

    def is_node(self) -> bool:
        return len(self.nodes) == 1

    def profile(self, gpu_num: int) -> Optional[float]:
        assert self.is_node()
        return self.nodes[0].profile(gpu_num)

    def __hash__(self):
        # Include both nodes and graph structure in hash for consistency with __eq__
        # Create a frozenset of (node, tuple of neighbors) pairs to represent the graph
        graph_edges = frozenset(
            (node, tuple(sorted(neighbors, key=lambda n: n.role)))
            for node, neighbors in self.graph.items()
        )
        return hash((tuple(sorted(self.nodes, key=lambda n: n.role)), graph_edges))

    def __eq__(self, other):
        if not isinstance(other, Workflow):
            return False
        if set(self.nodes) != set(other.nodes):
            return False
        if set(self.graph.keys()) != set(other.graph.keys()):
            return False
        for node in self.graph:
            if set(self.graph[node]) != set(other.graph.get(node, [])):
                return False
        return True

    def __str__(self):
        return ", ".join([f"{node} -> {self.graph[node]}" for node in self.graph])

    def __repr__(self):
        return self.__str__()


def traverse_st_cuts(workflow: Workflow) -> list[tuple[Workflow, Workflow]]:
    cuts: list[tuple[Workflow, Workflow]] = []
    topological_order = workflow.topological_order
    if len(topological_order) <= 1:
        return []

    def get_sub_workflow(sub_nodes: set[ComponentNode]) -> Workflow:
        sub_graph: dict[ComponentNode, list[ComponentNode]] = {}
        for node in sub_nodes:
            sub_node_neighbors = []
            for neighbor in workflow.get_neighbors(node):
                if neighbor in sub_nodes:
                    sub_node_neighbors.append(neighbor)
            sub_graph[node] = sub_node_neighbors
        return Workflow(sub_graph)

    def has_edge(
        source_nodes: set[ComponentNode], sink_nodes: set[ComponentNode]
    ) -> bool:
        for node in source_nodes:
            for neighbor in workflow.get_neighbors(node):
                if neighbor in sink_nodes:
                    return True
        return False

    for cut_idx in range(len(topological_order) - 1):
        source_nodes = set(topological_order[: cut_idx + 1])
        sink_nodes = set(topological_order[cut_idx + 1 :])

        if not source_nodes or not sink_nodes:
            continue

        if not has_edge(source_nodes, sink_nodes):
            continue

        cuts.append((get_sub_workflow(source_nodes), get_sub_workflow(sink_nodes)))
    return cuts
