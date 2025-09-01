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

import itertools
import math
from collections import defaultdict, deque
from typing import Dict, List


class Node:
    def __init__(self, name: str):
        self.name = name
        self.neighbors = []

    def add_neighbor(self, neighbor: "Node"):
        self.neighbors.append(neighbor)

    def set_single_batch_instance_cost(self, single_batch_instance_cost: float):
        self.single_batch_instance_cost = single_batch_instance_cost

    def set_instance_num(self, instance_num: int):
        self.instance_num = instance_num

    def get_single_batch_cost(self) -> float:
        return self.single_batch_instance_cost / self.instance_num

    def __repr__(self):
        return self.name


class ComponentNode(Node):
    def __init__(self, name: str):
        super().__init__(name)


class SccComponentNode(Node):
    """The SccComponentNode denotes a strongly connected component (SCC) and is comprised of multiple constituent ComponentNodes."""

    def __init__(self, Components: List[ComponentNode]):
        super().__init__(" - ".join([component.name for component in Components]))
        self.components = Components

    def get_single_batch_cost(self) -> float:
        return sum(component.get_single_batch_cost() for component in self.components)


class Workflow:
    def __init__(self, workflow: Dict[Node, List[Node]]):
        self.nodes = list(set(workflow.keys()))
        for neighbors in workflow.values():
            for neighbor in neighbors:
                if neighbor not in self.nodes:
                    self.nodes.add(neighbor)
                    workflow[neighbor] = []
        self.workflow = workflow

        self.sccs = self.find_sccs()

        self.topological_order = None

    def find_sccs(self) -> List[List[Node]]:
        """Find strongly connected components (SCCs) using Tarjan's algorithm."""

        def tarjan_dfs(node, disc, low, stack, in_stack, time):
            disc[node] = low[node] = time[0]
            time[0] += 1
            stack.append(node)
            in_stack.add(node)

            for neighbor in self.workflow.get(node, []):
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

        node_to_scc = {}
        for scc_idx, scc in enumerate(self.sccs):
            for node in scc:
                node_to_scc[node] = scc_idx

        # Build compressed graph using Workflow format
        compressed_workflow = {}

        # Create compressed node for each SCC
        for scc_idx, scc in enumerate(self.sccs):
            if len(scc) == 1:
                compressed_node = scc[0]
            else:
                compressed_node = SccComponentNode(scc)

            compressed_workflow[compressed_node] = []

            for node in scc:
                for neighbor in self.workflow.get(node, []):
                    neighbor_scc = node_to_scc[neighbor]
                    if neighbor_scc != scc_idx:
                        # Find corresponding compressed node
                        target_compressed_node = None
                        for existing_node in compressed_workflow.keys():
                            if existing_node in compressed_workflow:
                                if len(self.sccs[neighbor_scc]) == 1:
                                    if existing_node == self.sccs[neighbor_scc][0]:
                                        target_compressed_node = existing_node
                                        break
                                else:
                                    if (
                                        isinstance(existing_node, SccComponentNode)
                                        and existing_node.components
                                        == self.sccs[neighbor_scc]
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

    def topological_sort(self) -> List[Node]:
        """Perform topological sort on the workflow(graph)"""
        if self.topological_order is not None:
            return self.topological_order

        in_degree = defaultdict(int)
        for node in self.workflow:
            for neighbor in self.workflow[node]:
                in_degree[neighbor] += 1

        queue = deque([node for node in self.workflow if in_degree[node] == 0])
        self.topological_order = []

        while queue:
            node = queue.popleft()
            self.topological_order.append(node)

            for neighbor in self.workflow[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return self.topological_order


class WorkflowPartitioner:
    def __init__(self, workflow: Workflow):
        self.workflow = workflow.compress_sccs()

    def partition(self) -> List[Dict[str, Workflow]]:
        """Enumerate possible partitioning ways

        Returns:
            List of all possible partitioning ways, each is a dictionary with subgraph names as keys and Workflow objects as values
        """
        if self.workflow.topological_order is None:
            self.workflow.topological_sort()

        partitions: List[Dict[str, Workflow]] = []

        # Try different numbers of partitions
        for num_partitions in range(1, len(self.workflow.sccs) + 1):
            # Generate all possible cut point combinations
            if num_partitions == 1:
                subgraph_nodes = self._extract_nodes_from_compressed_workflow()
                subgraph_workflow = self._create_subgraph_workflow(subgraph_nodes)
                partition_graph = {"SUBGRAPH_0": subgraph_workflow}
                partitions.append(partition_graph)
            else:
                # Insert cut points between SCCs in topological order
                for cut_points in itertools.combinations(
                    range(len(self.workflow.topological_order) - 1), num_partitions - 1
                ):
                    partition_graph = {}
                    start_idx = 0
                    subgraph_id = 0
                    for cut_point in cut_points:
                        sccs_in_partition = []
                        for i in range(start_idx, cut_point + 1):
                            compressed_node = self.workflow.topological_order[i]
                            if isinstance(compressed_node, SccComponentNode):
                                sccs_in_partition.extend(compressed_node.components)
                            else:
                                sccs_in_partition.append(compressed_node)

                        # Create subgraph
                        subgraph_workflow = self._create_subgraph_workflow(
                            sccs_in_partition
                        )
                        partition_graph[f"SUBGRAPH_{subgraph_id}"] = subgraph_workflow
                        subgraph_id += 1
                        start_idx = cut_point + 1

                    # Add the last subgraph
                    sccs_in_partition = []
                    for i in range(start_idx, len(self.workflow.topological_order)):
                        compressed_node = self.workflow.topological_order[i]
                        if isinstance(compressed_node, SccComponentNode):
                            sccs_in_partition.extend(compressed_node.components)
                        else:
                            sccs_in_partition.append(compressed_node)

                    subgraph_workflow = self._create_subgraph_workflow(
                        sccs_in_partition
                    )
                    partition_graph[f"SUBGRAPH_{subgraph_id}"] = subgraph_workflow

                    partitions.append(partition_graph)

        return partitions

    def _extract_nodes_from_compressed_workflow(self) -> List[Node]:
        """Extract all original nodes from the compressed workflow"""
        nodes = []
        for compressed_node in self.workflow.topological_order:
            if isinstance(compressed_node, SccComponentNode):
                nodes.extend(compressed_node.components)
            else:
                nodes.append(compressed_node)
        return nodes

    def _create_subgraph_workflow(self, nodes: List[Node]) -> Workflow:
        """
        Create subgraph Workflow from node list

        Args:
            nodes: List of nodes in the subgraph

        Returns:
            Subgraph Workflow object
        """
        subgraph_dict = {}
        for node in nodes:
            neighbors = []
            for neighbor in self.workflow.workflow.get(node, []):
                if neighbor in nodes:
                    neighbors.append(neighbor)
            subgraph_dict[node] = neighbors
        return Workflow(subgraph_dict)


class PipelineCostCacl:
    def __init__(
        self,
        workflow: Workflow,
    ):
        self.workflow = workflow

        # Check if contains strongly connected component compressed nodes
        has_scc_compressed = any(
            isinstance(node, SccComponentNode) for node in self.workflow.nodes
        )

        if has_scc_compressed:
            raise NotImplementedError(
                "SCC compressed nodes are not yet supported in PipelineCostCacl"
            )

        self.topological_order = self.workflow.topological_sort()
        self.critical_path = self._find_critical_path()

    def _find_critical_path(self):
        """Find critical path (longest path)"""
        dp = dict.fromkeys(self.workflow.nodes, 0)
        parent = {}

        for node in self.topological_order:
            for neighbor in self.workflow.workflow.get(node, []):
                # For SCC compressed nodes, calculate total time of all internal components
                node_time = node.get_single_batch_cost()

                new_cost = dp[node] + node_time
                if new_cost > dp[neighbor]:
                    dp[neighbor] = new_cost
                    parent[neighbor] = node

        # Find end nodes
        end_nodes = [
            node
            for node in self.workflow.nodes
            if not self.workflow.workflow.get(node, [])
        ]
        if not end_nodes:
            end_node = max(dp.keys(), key=lambda x: dp[x])
        else:
            end_node = max(end_nodes, key=lambda x: dp[x])

        # Reconstruct path
        path = []
        current = end_node
        while current in parent:
            path.append(current)
            current = parent[current]
        path.append(current)

        return list(reversed(path))

    def calculate_total_time(self, total_data_size: int, batch_size: int):
        """Calculate total time to process N data items"""
        num_batches = math.ceil(total_data_size / batch_size)

        # Pipeline startup time (critical path length)
        startup_time = 0
        for node in self.critical_path:
            startup_time += node.get_single_batch_cost()

        # Pipeline steady state running time
        max_node_time = 0
        for node in self.workflow.nodes:
            node_time = node.get_single_batch_cost()
            max_node_time = max(max_node_time, node_time)

        steady_time = (num_batches - 1) * max_node_time

        # Total time
        total_time = startup_time + steady_time

        return {
            "total_time": total_time,
            "startup_time": startup_time,
            "steady_time": steady_time,
            "num_batches": num_batches,
            "critical_path": self.critical_path,
            "throughput": total_data_size / total_time if total_time > 0 else 0,
        }

    def print_analysis(self, total_data_size: int, batch_size: int):
        """Print time analysis"""
        result = self.calculate_total_time(total_data_size, batch_size)

        print(f"Subgraph: {self.workflow.nodes}")
        print(
            f"Data size: {total_data_size}, Number of batches: {result['num_batches']}"
        )
        critical_path_str = ""
        for path in result["critical_path"]:
            if isinstance(path, SccComponentNode):
                critical_path_str += f"{path}"
            else:
                critical_path_str += f"{path}"
            if path != self.critical_path[-1]:
                critical_path_str += " -> "
        print(f"Critical path: {critical_path_str}")
        print(
            f"Startup time: {result['startup_time']:.2f}, Steady time: {result['steady_time']:.2f}"
        )
        print(
            f"Total time: {result['total_time']:.2f}, Throughput: {result['throughput']:.2f}"
        )


def get_workflow_cost(
    workflow: Workflow,
    batch_size: int,
    total_data_size: int,
) -> float:
    """Calculate total cost of workflow"""
    return PipelineCostCacl(workflow).calculate_total_time(total_data_size, batch_size)[
        "total_time"
    ]


def get_workflow_partition(workflow: Workflow) -> List[Dict[str, Workflow]]:
    """Get workflow partitions"""
    return WorkflowPartitioner(workflow).partition()
