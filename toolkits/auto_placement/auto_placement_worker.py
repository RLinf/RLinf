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

import logging
from typing import Optional

import hydra
from node import (
    ComponentNode,
    EnvNode,
    EnvProfiler,
    EnvRolloutNode,
    MegatronNode,
    RolloutNode,
)
from placement import (
    PlacementStrategy,
    ScheduleResult,
    SingleNodeScheduleResult,
)
from util import get_global_config, get_valid_gpu_num_list, init_global_config
from workflow import Workflow, traverse_st_cuts

from rlinf.scheduler import Cluster
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)

_CANDIDATE_RESULTS: list[dict] = []


class AutoPlacementWorker:
    def __init__(
        self,
        cfg,
        component_placement,
        graph: Optional[dict[str, list[str]]] = None,
    ):
        """Initialize the AutoPlacementWorker."""
        self.config = get_global_config()
        self.components_config = self.config.components_config
        self._name_to_node_dict: dict[str, ComponentNode] = {}
        self._init_workflow(graph)

    def get_node(self, component_name: str) -> ComponentNode:
        if component_name in self._name_to_node_dict:
            return self._name_to_node_dict[component_name]

        if component_name == "rollout":
            node = RolloutNode()
        elif component_name in ["actor", "inference"]:
            valid_gpu_num_list: list[int] = get_valid_gpu_num_list(component_name)
            node = MegatronNode(
                role=component_name,
                valid_gpu_nums=valid_gpu_num_list,
            )
        elif component_name == "env":
            node = EnvNode(
                profiler=EnvProfiler(
                    self.config.profile_time.env_profile_time
                    if hasattr(self.config, "profile_time")
                    else self.config.profile_data.env_profile_data,
                    self.config.env_num,
                )
            )
        elif component_name == "env_rollout":
            node = EnvRolloutNode(
                profiler=EnvProfiler(
                    self.config.profile_time.rollout_profile_time
                    if hasattr(self.config, "profile_time")
                    else self.config.profile_data.rollout_profile_data,
                    self.config.env_num,
                ),
                model_parallel_size=self.components_config[
                    "env_rollout"
                ].model_parallel_size,
            )
        else:
            raise ValueError(f"{component_name=} is not supported")

        self._name_to_node_dict[component_name] = node
        return node

    def _init_workflow(self, graph: dict[str, list[str]]):
        # Create ComponentNode and conver graph(str) to graph(ComponentNode)
        workflow_graph: dict[ComponentNode, list[ComponentNode]] = {}
        for component_name, neighbors in graph.items():
            node = self.get_node(component_name)
            workflow_graph[node] = [self.get_node(neighbor) for neighbor in neighbors]

        # Compress strongly connected components
        workflow = Workflow(workflow_graph)
        self.workflow = workflow.compress_sccs()

    def _find_schedule(
        self, workflow: Workflow, gpu_num: int
    ) -> Optional[ScheduleResult]:
        key = (workflow, gpu_num)
        if key in self._result_cache:
            return self._result_cache[key]

        if workflow.is_node():
            cost = workflow.profile(gpu_num)
            if cost is None:
                return None

            # For reasoning task, the cost is the cost per group batch.
            if self.config.task_type == "reasoning":
                cost_per_group_batch = cost
                total_cost = cost * self.config.rollout_batch_size
            else:  # For embodiment task, the cost means total cost.
                cost_per_group_batch = cost / self.config.env_num
                total_cost = cost

            self._result_cache[key] = SingleNodeScheduleResult(
                total_gpu_num=gpu_num,
                node=workflow.nodes[0],
                cost_per_group_batch=cost_per_group_batch,
                total_cost=total_cost,
            )

            return self._result_cache[key]

        best_res = None

        cuts = traverse_st_cuts(workflow)
        for source_workflow, sink_workflow in cuts:
            source_res: ScheduleResult = self._find_schedule(source_workflow, gpu_num)
            sink_res: ScheduleResult = self._find_schedule(sink_workflow, gpu_num)
            collocated_res = ScheduleResult.merger_schedule_results(
                gpu_num, source_res, sink_res, is_collocated=True
            )
            self._record_candidate(collocated_res)

            best_res = ScheduleResult.find_best_schedule(best_res, collocated_res)

            # Pipeline schedule
            for source_gpu_num in range(1, gpu_num - 1):
                sink_gpu_num = gpu_num - source_gpu_num
                source_res: ScheduleResult = self._find_schedule(
                    source_workflow, source_gpu_num
                )
                sink_res: ScheduleResult = self._find_schedule(
                    sink_workflow, sink_gpu_num
                )

                disaggregated_res = ScheduleResult.merger_schedule_results(
                    gpu_num, source_res, sink_res, is_collocated=False
                )
                self._record_candidate(disaggregated_res)

                best_res = ScheduleResult.find_best_schedule(
                    best_res, disaggregated_res
                )

        self._result_cache[key] = best_res
        return best_res

    def run(self) -> ScheduleResult:
        self._result_cache: dict[tuple[Workflow, int], ScheduleResult] = {}
        return self._find_schedule(self.workflow, self.config.total_gpus)

    def _record_candidate(self, res: Optional[ScheduleResult]) -> None:
        if res is None:
            return
        component_costs = get_component_predicted_costs(res)
        if self.config.task_type == "embodied":
            generate_cost = component_costs.get("env", 0.0) + component_costs.get(
                "env_rollout", 0.0
            )
            actor_cost = component_costs.get("actor", 0.0)
        else:
            generate_cost = 0.0
            actor_cost = component_costs.get("actor", 0.0)

        stage_info = (
            _get_embodied_stage_cost_info(res)
            if self.config.task_type == "embodied"
            else {}
        )
        _CANDIDATE_RESULTS.append(
            {
                "mode": res.placement_strategy.value,
                "total_cost": float(res.total_cost),
                "generate_cost": float(generate_cost),
                "actor_cost": float(actor_cost),
                "placement_str": res.placement_str.strip(),
                **stage_info,
            }
        )


def get_component_predicted_costs(
    schedule_result: ScheduleResult,
) -> dict[str, float]:
    """Estimate per-component predicted cost from final placement."""
    component_costs: dict[str, float] = {}
    config = get_global_config()
    for node, gpu_range in schedule_result.placement.items():
        gpu_num = len(gpu_range)
        if config.task_type == "embodied" and node.role == "actor":
            gpu_num = schedule_result.total_gpu_num
        cost = node.profile(gpu_num)
        if cost is None:
            continue
        component_costs[node.role] = cost
    return component_costs


def _get_embodied_stage_cost_info(
    schedule_result: ScheduleResult,
) -> dict[str, float | int | None]:
    """Return per-stage env counts and fitted costs for embodied candidates.

    This mirrors the stage-aware model used in `placement.py` for env<->env_rollout cuts,
    but is used purely for logging/debugging candidates.
    """
    config = get_global_config()
    # In collocated mode (all components share the same GPUs), there is no meaningful
    # env<->rollout pipeline separation in this cost model. Default stage_num to 1
    # for clearer logging.
    if schedule_result.placement_strategy == PlacementStrategy.COLLOCATED:
        stage_num = 1
    else:
        stage_num = int(getattr(config, "pipeline_stage_num", 2))
        if stage_num <= 0:
            stage_num = 2
    total_envs = int(getattr(config, "env_num", 0))
    if total_envs <= 0:
        return {
            "pipeline_stage_num": stage_num,
            "env_num_per_gpu_per_stage": None,
            "rollout_num_per_gpu_per_stage": None,
            "env_stage_cost": None,
            "rollout_stage_cost": None,
        }

    env_node = None
    rollout_node = None
    env_gpus = 0
    rollout_gpus = 0
    for node, gpu_range in schedule_result.placement.items():
        if node.role == "env":
            env_node = node
            env_gpus = len(gpu_range)
        elif node.role == "env_rollout":
            rollout_node = node
            rollout_gpus = len(gpu_range)

    if env_node is None or rollout_node is None or env_gpus <= 0 or rollout_gpus <= 0:
        return {
            "pipeline_stage_num": stage_num,
            "env_num_per_gpu_per_stage": None,
            "rollout_num_per_gpu_per_stage": None,
            "env_stage_cost": None,
            "rollout_stage_cost": None,
        }

    denom_env = env_gpus * stage_num
    denom_rollout = rollout_gpus * stage_num
    if (total_envs % denom_env != 0) or (total_envs % denom_rollout != 0):
        return {
            "pipeline_stage_num": stage_num,
            "env_num_per_gpu_per_stage": None,
            "rollout_num_per_gpu_per_stage": None,
            "env_stage_cost": None,
            "rollout_stage_cost": None,
        }

    env_num_per_gpu_per_stage = total_envs // denom_env
    rollout_num_per_gpu_per_stage = total_envs // denom_rollout

    # EnvNode / EnvRolloutNode both carry an EnvProfiler instance.
    env_prof = getattr(env_node, "profiler", None)
    rollout_prof = getattr(rollout_node, "profiler", None)
    env_stage_cost = (
        float(env_prof.cost_for_envs_per_instance(env_num_per_gpu_per_stage))
        if env_prof is not None
        else None
    )
    rollout_stage_cost = (
        float(rollout_prof.cost_for_envs_per_instance(rollout_num_per_gpu_per_stage))
        if rollout_prof is not None
        else None
    )

    return {
        "pipeline_stage_num": stage_num,
        "env_num_per_gpu_per_stage": int(env_num_per_gpu_per_stage),
        "rollout_num_per_gpu_per_stage": int(rollout_num_per_gpu_per_stage),
        "env_stage_cost": env_stage_cost,
        "rollout_stage_cost": rollout_stage_cost,
    }


def get_workflow_graph(cfg) -> dict[str, list[str]]:
    if cfg.runner.task_type == "reasoning":
        if cfg.algorithm.recompute_logprobs:
            return {
                "rollout": ["inference"],
                "inference": ["actor"],
                "actor": [],
            }
        else:
            return {
                "rollout": ["actor"],
                "actor": [],
            }
    elif cfg.runner.task_type == "embodied":
        return {
            "env": ["env_rollout"],
            "env_rollout": ["actor"],
            "actor": [],
        }
    else:
        raise ValueError(f"{cfg.runner.task_type=} is not supported")


@hydra.main(version_base="1.1")
def main(cfg):
    cluster = Cluster(cfg.cluster.num_nodes)
    if cfg.runner.task_type == "reasoning":
        component_placement = ModelParallelComponentPlacement(cfg, cluster)
    else:  # embodiment task
        component_placement = HybridComponentPlacement(cfg, cluster)
    init_global_config(cfg, component_placement, cluster)

    workflow_graph: dict[str, list[str]] = get_workflow_graph(cfg)
    auto_placement_worker = AutoPlacementWorker(
        cfg, component_placement, workflow_graph
    )

    schedule_result: ScheduleResult = auto_placement_worker.run()

    if schedule_result is None:
        logging.error("=" * 50)
        logging.error(
            "Error: Auto scheduler could not find any valid placement strategy."
        )
        logging.error("Possible reasons:")
        logging.error("1. Missing profile data for certain GPU scales.")
        logging.error(
            "2. The hardware rank provided by component_placement configure is not compatible with that ray cluster detect."
        )
        return None

    res = (
        ", ".join(
            [
                node.role
                for node in schedule_result.placement
                if node.role != "inference"
            ]
        )
        + " : all"
    )

    logging.info("=" * 50)
    logging.info("Best placement for this task is:\n%s", res)
    logging.info(
        "Predicted step time (from profile): %.3f s",
        schedule_result.total_cost,
    )

    if _CANDIDATE_RESULTS:
        top_k = int(cfg.get("auto_placement_top_k", 10))
        uniq: dict[tuple[str, str], dict] = {}
        for item in _CANDIDATE_RESULTS:
            key = (item["mode"], item["placement_str"])
            if key not in uniq or item["total_cost"] < uniq[key]["total_cost"]:
                uniq[key] = item
        rows = sorted(uniq.values(), key=lambda x: x["total_cost"])[:top_k]
        logging.info("=" * 50)
        logging.info("Candidate placements (top %d by predicted cost):", top_k)
        for i, row in enumerate(rows):
            stage_str = ""
            if cfg.runner.task_type == "embodied":
                stage_str = (
                    "\n"
                    f"stage_num={row.get('pipeline_stage_num')} "
                    f"env_num_per_gpu_per_stage={row.get('env_num_per_gpu_per_stage')} "
                    f"rollout_num_per_gpu_per_stage={row.get('rollout_num_per_gpu_per_stage')}\n"
                    f"EnvNode[{row.get('env_num_per_gpu_per_stage')}]={row.get('env_stage_cost')} "
                    f"EnvRolloutNode[{row.get('rollout_num_per_gpu_per_stage')}]={row.get('rollout_stage_cost')}"
                )
            logging.info(
                "[%02d] mode=%s temp_step_time=%.3fs%s \n%s",
                i,
                row["mode"],
                row["total_cost"],
                stage_str,
                row["placement_str"] or "(empty)",
            )


if __name__ == "__main__":
    main()
