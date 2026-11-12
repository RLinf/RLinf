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

"""RLinf embodied control and weight reception for SGLang Diffusion servers."""

from fastapi import HTTPException
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler as _Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

from rlinf.hybrid_engines.sglang.common.http_routes import add_rlinf_routes
from rlinf.hybrid_engines.sglang.common.io_struct import (
    SyncHFWeightInput,
    TaskMethodInput,
    TaskMethodOutput,
)
from rlinf.hybrid_engines.sglang.common.weight_target import receive_hf_weights
from rlinf.scheduler import Worker


class Scheduler(_Scheduler):
    """Receive actor buckets in the process that owns the serving parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rlinf_worker = None
        self._weights_ready = True
        self._sync_failed = False
        self.request_handlers.update(
            {
                TaskMethodInput: self.run_task_method,
                SyncHFWeightInput: self.sync_hf_weight,
            }
        )

    def run_task_method(self, requests: list[TaskMethodInput]) -> OutputBatch:
        req = requests[0]
        result = getattr(self, req.method_name)(*req.args, **req.kwargs)
        return OutputBatch(output=TaskMethodOutput(req.method_name, result))

    def init_rlinf_worker(
        self,
        parent_address,
        weight_reload="sync",
        placement=None,
        config=None,
        model_instance_id=None,
    ) -> None:
        """Initialize evaluation weights or register the actor weight receiver."""
        only_eval = config.runner.get("only_eval", False)
        checkpoint = config.runner.get("ckpt_path")
        if hasattr(self, "weight_target") or hasattr(self, "_loaded_checkpoint"):
            if (
                only_eval
                and self._rlinf_worker is None
                and self._weights_ready
                and self._loaded_checkpoint == checkpoint
            ):
                return
            raise RuntimeError("The server's RLinf state is already initialized")
        if only_eval and not checkpoint:
            self._loaded_checkpoint = checkpoint
            self._weights_ready = True
            return
        if self.server_args.dp_size != 1:
            raise ValueError("Use separate embodied server instances for DP replicas")
        from rlinf.models.embodiment.sglang_adapter import get_sglang_adapter_cls

        self._weights_ready = False
        adapter_cls = get_sglang_adapter_cls(config.rollout.model.model_type)
        if adapter_cls is None:
            raise ValueError(f"No SGLang adapter for {config.rollout.model.model_type}")
        adapter = adapter_cls(config, 0)
        if not only_eval:
            adapter.validate_training(config)
        self.weight_target = adapter.create_weight_target(
            self._load_weights, self.server_args.model_path
        )
        if only_eval:
            self.weight_target.load_checkpoint(checkpoint)
            self._loaded_checkpoint = checkpoint
            self._weights_ready = True
            return
        from rlinf.workers.rollout.utils import DisaggRankMapper

        ranks_per_instance = placement.rollout_gpus_per_process
        if ranks_per_instance != self.server_args.num_gpus:
            raise ValueError("Server GPU count does not match rollout placement")
        rank = self.worker.rank
        self._rlinf_worker = Worker(
            parent_address=parent_address, world_size=ranks_per_instance, rank=rank
        )
        rank_map = DisaggRankMapper.get_rollout_rank_to_actor_rank_map(
            actor_tp_size=1,
            actor_pp_size=1,
            actor_world_size=placement.get_world_size("actor"),
            rollout_tp_size=placement.rollout_tp_size,
            rollout_world_size=placement.rollout_world_size,
            rollout_sp_degree=placement.rollout_sp_degree,
        )
        self.actor_weight_rank = rank_map[(model_instance_id, rank)]
        self._actor_group_name = config.actor.group_name
        self._weights_ready = False

    def _load_weights(self, module: str, named_tensors: list) -> None:
        from sglang.multimodal_gen.runtime.post_training.weights_updater import (
            WeightsUpdater,
        )

        success, message = WeightsUpdater(
            self.worker.pipeline
        ).update_weights_from_tensor(named_tensors, target_modules=[module])
        if not success:
            raise RuntimeError(message)

    def sync_hf_weight(self, requests: list[SyncHFWeightInput]) -> OutputBatch:
        if self._rlinf_worker is None:
            raise RuntimeError(
                "Initialize the server's training weight receiver before sync"
            )
        if self._sync_failed:
            raise RuntimeError(
                "Restart the server after a failed weight synchronization"
            )
        self._weights_ready = False
        import torch.distributed as dist
        from sglang.multimodal_gen.runtime.distributed import get_replica_group

        version = None
        error = None
        try:
            self.weight_target.begin_sync(full=True)
            version = receive_hf_weights(
                self._rlinf_worker,
                self._actor_group_name,
                self.actor_weight_rank,
                self.weight_target.load_state_dict,
            )
            self.weight_target.finish_sync()
            if version is None:
                raise ValueError(
                    "Embodied actor weight buckets must include weight_version"
                )
        except Exception as exc:
            error = f"rank {self.worker.rank}: {type(exc).__name__}: {exc}"
        replica = get_replica_group()
        results = [None] * replica.world_size
        try:
            dist.all_gather_object(results, (version, error), group=replica.cpu_group)
        except Exception:
            self._sync_failed = True
            raise
        errors = [message for _, message in results if message is not None]
        if errors or len({v for v, _ in results}) != 1:
            self._sync_failed = True
            raise RuntimeError(
                f"Embodied replica weight sync failed: {errors or results}; restart server"
            )
        self._weights_ready = True
        return OutputBatch(output={"version": version})

    def _handle_generation(self, reqs, **kwargs):
        if not self._weights_ready:
            raise RuntimeError(
                "Serving weights are unavailable until a complete actor sync succeeds"
            )
        return super()._handle_generation(reqs, **kwargs)


class _SchedulerManager:
    """Forward the shared HTTP control API to the embodied scheduler."""

    async def _forward(self, obj):
        from sglang.multimodal_gen.runtime.scheduler_client import (
            async_scheduler_client,
        )

        result = await async_scheduler_client.forward(obj)
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        return result.output

    async def run_task_method(self, obj, request):
        return await self._forward(obj)

    async def sync_hf_weight(self, obj, request):
        return await self._forward(obj)

    async def abort_generation(self, obj, request):
        raise HTTPException(
            status_code=400, detail="Embodied rollout does not support abort"
        )


def create_app(server_args):
    from sglang.multimodal_gen.runtime.entrypoints.http_server import (
        create_app as native_create_app,
    )

    app = native_create_app(server_args)
    add_rlinf_routes(app, manager=_SchedulerManager())
    return app


def run_scheduler_process(*args, **kwargs):
    from sglang.multimodal_gen.runtime.managers.gpu_worker import (
        run_scheduler_process as native_run,
    )

    native_run(*args, **kwargs, scheduler_cls=Scheduler)
