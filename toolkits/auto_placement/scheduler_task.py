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

from typing import Optional

import hydra
from omegaconf.omegaconf import OmegaConf
from resource_allocator import AllocationStates, resource_allocate
from workflow import ComponentNode, Workflow, get_workflow_cost, get_workflow_partition

from rlinf.config import validate_cfg
from rlinf.scheduler import Cluster


class SchedulerTask:
    def __init__(
        self,
        cfg,
        cluster,
        workflow_graph: Optional[dict[ComponentNode, list[ComponentNode]]] = None,
    ):
        self.cfg = cfg
        self.is_reasoning = cfg.runner.task_type == "reasoning"
        assert self.is_reasoning, (
            f"Only reasoning task is supported, current task type: {cfg.runner.task_type}"
        )

        self.components_config = {
            "actor": {
                "tensor_model_parallel_size": cfg.actor.model.tensor_model_parallel_size,
                "pipeline_model_parallel_size": cfg.actor.model.pipeline_model_parallel_size,
            },
            "rollout": {
                "tensor_model_parallel_size": cfg.rollout.tensor_parallel_size,
                "pipeline_model_parallel_size": cfg.rollout.pipeline_parallel_size,
            },
        }
        try:
            inference_tensor_model_parallel_size = (
                cfg.inference.model.tensor_model_parallel_size
            )
            inference_pipeline_model_parallel_size = (
                cfg.inference.model.pipeline_model_parallel_size
            )
        except Exception:
            inference_tensor_model_parallel_size = self.components_config["actor"][
                "tensor_model_parallel_size"
            ]
            inference_pipeline_model_parallel_size = self.components_config["actor"][
                "pipeline_model_parallel_size"
            ]
        self.components_config["inference"] = {
            "tensor_model_parallel_size": inference_tensor_model_parallel_size,
            "pipeline_model_parallel_size": inference_pipeline_model_parallel_size,
        }

        self.total_gpus = cluster.num_accelerators
        self.group_size = cfg.algorithm.group_size
        self.n_minibatches = cfg.algorithm.n_minibatches
        self.rollout_batch_size = cfg.data.rollout_batch_size
        self.seq_length = cfg.runner.seq_length
        self.global_step_batch_size = self.rollout_batch_size * self.group_size

        if workflow_graph is None:
            if self.is_reasoning:
                actor = ComponentNode("actor")
                inference = ComponentNode("inference")
                rollout = ComponentNode("rollout")
                workflow_graph = {
                    rollout: [inference],
                    inference: [actor],
                    actor: [],
                }

            else:
                actor = ComponentNode("actor")
                inference = ComponentNode("inference")
                rollout = ComponentNode("rollout")
                env = ComponentNode("env")
                workflow_graph = {
                    env: [rollout],
                    rollout: [env, inference],
                    inference: [actor],
                    actor: [],
                }

        self.workflow = Workflow(workflow_graph)
        self.profile_data_registed = False

    def register_profile_data(self, profile_data: dict[str, float]):
        for component_node in self.workflow.nodes:
            component_node_name = component_node.name
            instance_num, signle_iter_cost = profile_data[component_node_name]
            signle_batch_cost = (
                signle_iter_cost * instance_num / self.rollout_batch_size
            )

            assert signle_batch_cost > 0

            component_node.set_single_batch_instance_cost(signle_batch_cost)

        self.profile_data_registed = True

    def run(self) -> str:
        assert self.profile_data_registed, "Profile data not registered"
        partitions: list[dict[str, Workflow]] = self.time_division_multiplexing()

        min_partition_cost = float("inf")
        min_partition_allocations = None

        for i, partition in enumerate(partitions):
            cur_partition_cost = 0
            cur_partition_allocations = {}
            for workflow in partition.values():
                workflow_allocation, workflow_cost = self.space_division_multiplexing(
                    workflow
                )
                cur_partition_cost += workflow_cost
                cur_partition_allocations[workflow] = workflow_allocation

            if cur_partition_cost < min_partition_cost:
                min_partition_cost = cur_partition_cost
                min_partition_allocations = cur_partition_allocations

        best_placement = self.parse_partition_allocation_to_cfg(
            min_partition_allocations
        )
        return best_placement

    def parse_partition_allocation_to_cfg(
        self, partition_allocation: dict[Workflow, AllocationStates]
    ) -> str:
        new_cfg = OmegaConf.create()
        new_cfg.cluster = {}
        new_cfg.cluster.num_nodes = self.cfg.cluster.num_nodes
        new_cfg.cluster.component_placement = {}

        is_collocated = len(self.workflow.nodes) == len(partition_allocation)
        if is_collocated:
            for workflow, allocation in partition_allocation.items():
                assert len(workflow.nodes) == 1, (
                    "Only one component is allowed in collocated mode"
                )
                assert (
                    allocation.get_component(workflow.nodes[0].name).world_size
                    == self.total_gpus
                ), (
                    "The total number of GPUs in collocated mode must be equal to the total number of GPUs in the cluster"
                )

            valid_collocated_components = [
                node.name for node in self.workflow.nodes if node.name != "inference"
            ]
            collocated_components = ",".join(valid_collocated_components)
            new_cfg.cluster.component_placement[collocated_components] = "all"
            return OmegaConf.to_yaml(new_cfg)

        is_disaggregated = len(partition_allocation) == 1
        if is_disaggregated:
            start_gpu_id = 0
            for workflow, allocation in partition_allocation.items():
                for component in workflow.nodes:
                    new_cfg.cluster.component_placement[component.name] = (
                        f"{start_gpu_id}-{start_gpu_id + allocation.get_component(component.name).world_size - 1}"
                    )
                    start_gpu_id += allocation.get_component(component.name).world_size
            return OmegaConf.to_yaml(new_cfg)

        raise NotImplementedError("hybrid mode is not supported")

    def time_division_multiplexing(self) -> list[dict[str, Workflow]]:
        partitions: list[dict[str, Workflow]] = get_workflow_partition(self.workflow)
        if self.is_reasoning:
            valid_partitions = [
                i for i in partitions if len(i) in [1, len(self.components_config)]
            ]
        else:
            valid_partitions = partitions

        return valid_partitions

    def space_division_multiplexing(self, workflow: Workflow):
        sub_components_config = {}
        workflow_components_name = [component.name for component in workflow.nodes]
        for component_name, parallel_config in self.components_config.items():
            if component_name in workflow_components_name:
                sub_components_config[component_name] = parallel_config

        if (
            len(sub_components_config) == 1
            and sub_components_config.get("inference") is not None
        ):
            inference_instance_max_num = self.total_gpus // (
                sub_components_config["inference"]["tensor_model_parallel_size"]
                * sub_components_config["inference"]["pipeline_model_parallel_size"]
            )
        else:
            inference_instance_max_num = 2

        allocations: list[AllocationStates] = resource_allocate(
            sub_components_config,
            self.total_gpus,
            self.group_size,
            self.rollout_batch_size,
            self.n_minibatches,
            inference_instance_max_num=inference_instance_max_num,
        )

        min_cost = float("inf")
        min_cost_allocation = None
        for allocation in allocations:
            for component in workflow.nodes:
                component.set_instance_num(
                    allocation.get_component(component.name).data_parallel_size
                )

            cost = get_workflow_cost(
                workflow, self.group_size, self.global_step_batch_size
            )

            if cost < min_cost:
                min_cost = cost
                min_cost_allocation = allocation
        return min_cost_allocation, min_cost


def get_profile_data(
    cfg, cluster: Cluster, actor_cost=None, inference_cost=None, rollout_cost=None
):
    total_gpus = cluster.num_accelerators
    collocated_actor_instance_num = total_gpus // (
        cfg.actor.model.tensor_model_parallel_size
        * cfg.actor.model.pipeline_model_parallel_size
    )
    collocated_inference_instance_num = collocated_actor_instance_num
    collocated_rollout_instance_num = total_gpus // (
        cfg.rollout.tensor_parallel_size * cfg.rollout.pipeline_parallel_size
    )

    profile_data = {
        "actor": (collocated_actor_instance_num, actor_cost),
        "inference": (collocated_inference_instance_num, inference_cost),
        "rollout": (collocated_rollout_instance_num, rollout_cost),
    }

    return profile_data


@hydra.main(version_base="1.1")
def main(cfg):
    cfg = validate_cfg(cfg)

    actor_cost = getattr(cfg.profile_data, "actor_cost", None)
    inference_cost = getattr(cfg.profile_data, "inference_cost", None)
    rollout_cost = getattr(cfg.profile_data, "rollout_cost", None)

    if actor_cost is None or inference_cost is None or rollout_cost is None:
        raise ValueError("Profile data is not provided")

    cluster = Cluster(cfg.cluster.num_nodes)
    profile_data = get_profile_data(
        cfg, cluster, actor_cost, inference_cost, rollout_cost
    )
    scheduler_task = SchedulerTask(cfg, cluster)
    scheduler_task.register_profile_data(profile_data)
    res = scheduler_task.run()

    print("=" * 50)
    print("Best placement for this task is:\n")
    print(res)


if __name__ == "__main__":
    main()
