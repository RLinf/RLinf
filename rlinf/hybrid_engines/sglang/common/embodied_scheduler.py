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

"""RLinf embodied control and weight reception for SGLang Diffusion servers."""

import asyncio
import gc
import itertools
from collections.abc import Mapping

import torch
from fastapi import HTTPException
from sglang.multimodal_gen.runtime.managers.scheduler import Scheduler as _Scheduler
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import OutputBatch

from rlinf.hybrid_engines.sglang.common.http_routes import add_rlinf_routes
from rlinf.hybrid_engines.sglang.common.io_struct import (
    SyncHFWeightInput,
    TaskMethodInput,
    TaskMethodOutput,
)
from rlinf.hybrid_engines.sglang.common.peer_whitelist import (
    EMBODIED_WHITELISTED_PATHS,
    PeerWhitelistMiddleware,
)
from rlinf.hybrid_engines.sglang.common.weight_target import receive_hf_weights
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.scheduler import Worker
from rlinf.utils.pinned_offload import pinned_offload_context


class Scheduler(_Scheduler):
    """Receive actor buckets in the process that owns the serving parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rlinf_worker = None
        self._weights_ready = True
        self._sync_failed = False
        self._offloaded_modules: dict[str, torch.device] = {}
        self._is_offloaded = False
        self.weight_syncer = None
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
        only_eval = bool(
            weight_reload is None
            or config.runner.get("only_eval", False)
            or config.runner.get("task_type") == "embodied_eval"
        )
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
            actor_world_size=placement.get_world_size("actor"),
            rollout_tp_size=placement.rollout_tp_size,
            rollout_world_size=placement.rollout_world_size,
            rollout_sp_degree=placement.rollout_sp_degree,
            rollout_dp_size=placement.rollout_dp_size,
        )
        self.actor_weight_rank = rank_map[(model_instance_id, rank)]
        self._actor_group_name = config.actor.group_name
        if config.get("weight_syncer") is not None:
            self.weight_target.bind_patch_model(
                self.worker.pipeline.get_module(self.weight_target.target_module)
            )
            self.weight_syncer = WeightSyncer.create(config.weight_syncer)
            self._actor_world_size = placement.get_world_size("actor")
            self._patch_metadata_sender = model_instance_id == 0 and rank == 0
        self._weights_ready = False

    async def _sync_patch_weights(self) -> int:
        """Apply the actor's patch inside the process owning serving storage."""
        options = self.weight_syncer.comm_options

        async def recv():
            return await self._rlinf_worker.recv(
                src_group_name=self._actor_group_name,
                src_rank=0,
                async_op=True,
                options=options,
            ).async_wait()

        async def send(metadata):
            if self._patch_metadata_sender:
                for rank in range(self._actor_world_size):
                    await self._rlinf_worker.send(
                        metadata,
                        dst_group_name=self._actor_group_name,
                        dst_rank=rank,
                        async_op=True,
                        options=options,
                    ).async_wait()

        with torch.inference_mode():
            if not self.weight_syncer.receiver_initialized():
                await self.weight_syncer.init_receiver(
                    self.weight_target.state_dict(), recv=recv, send=send
                )
            version = await self.weight_syncer.apply(self.weight_target, recv)
        Worker.torch_platform.current_stream().synchronize()
        return version

    def _load_weights(self, module: str, named_tensors: list) -> None:
        from sglang.multimodal_gen.runtime.post_training.weights_updater import (
            WeightsUpdater,
        )

        success, message = WeightsUpdater(
            self.worker.pipeline
        ).update_weights_from_tensor(named_tensors, target_modules=[module])
        if not success:
            raise RuntimeError(message)

    def _resolve_pipeline_modules(
        self, tags: list[str] | None = None
    ) -> list[tuple[str, torch.nn.Module]]:
        modules = getattr(self.worker.pipeline, "modules", None)
        if not isinstance(modules, Mapping):
            return []
        if tags is None or "weights" in tags:
            return [
                (name, module)
                for name, module in modules.items()
                if isinstance(module, torch.nn.Module)
            ]
        selected: list[tuple[str, torch.nn.Module]] = []
        available = [
            name for name, mod in modules.items() if isinstance(mod, torch.nn.Module)
        ]
        for tag in tags:
            module = modules.get(tag)
            if not isinstance(module, torch.nn.Module):
                raise ValueError(
                    f"Unknown embodied offload tag {tag!r}; expected one of {available}"
                )
            selected.append((tag, module))
        return selected

    def release_memory_occupation(
        self, tags: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Offload serving pipeline modules to pinned CPU memory."""
        moved = False
        with pinned_offload_context():
            for name, module in self._resolve_pipeline_modules(tags):
                if name in self._offloaded_modules:
                    continue
                device = next(
                    (
                        tensor.device
                        for tensor in itertools.chain(
                            module.parameters(), module.buffers()
                        )
                        if tensor.device.type != "cpu"
                    ),
                    None,
                )
                if device is None:
                    continue
                self._offloaded_modules[name] = device
                module.to("cpu")
                moved = True
        if moved:
            Worker.torch_platform.current_stream().synchronize()
            gc.collect()
            Worker.torch_platform.empty_cache()
        self._is_offloaded = bool(self._offloaded_modules)
        return {"offloaded": sorted(self._offloaded_modules)}

    def resume_memory_occupation(
        self, tags: list[str] | None = None
    ) -> dict[str, list[str]]:
        """Restore offloaded serving pipeline modules back to their accelerators."""
        if not self._offloaded_modules:
            self._is_offloaded = False
            return {"resumed": [], "offloaded": []}
        if tags is None or "weights" in tags:
            target_names = list(self._offloaded_modules)
        else:
            target_names = [tag for tag in tags if tag in self._offloaded_modules]
        modules = getattr(self.worker.pipeline, "modules", {})
        resumed: list[str] = []
        for name in target_names:
            device = self._offloaded_modules.pop(name)
            module = modules.get(name) if isinstance(modules, Mapping) else None
            if isinstance(module, torch.nn.Module):
                module.to(device)
                resumed.append(name)
        if resumed:
            Worker.torch_platform.current_stream().synchronize()
        self._is_offloaded = bool(self._offloaded_modules)
        return {
            "resumed": sorted(resumed),
            "offloaded": sorted(self._offloaded_modules),
        }

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
        from sglang.multimodal_gen.runtime.distributed import get_world_group

        version = None
        error = None
        try:
            if self._is_offloaded:
                self.resume_memory_occupation()
            if self.weight_syncer is not None:
                version = asyncio.run(self._sync_patch_weights())
            else:
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
        # All DP replicas must commit the same version before serving resumes.
        world = get_world_group()
        results = [None] * world.world_size
        try:
            dist.all_gather_object(results, (version, error), group=world.cpu_group)
        except Exception:
            self._sync_failed = True
            raise
        errors = [message for _, message in results if message is not None]
        if errors or len({v for v, _ in results}) != 1:
            self._sync_failed = True
            raise RuntimeError(
                f"Embodied server weight sync failed: {errors or results}; restart server"
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

        if async_scheduler_client.server_args.dp_size == 1:
            try:
                results = [await async_scheduler_client.forward(obj)]
            except Exception as exc:
                results = [exc]
        else:
            # Native forward() treats RLinf's custom requests as generation.
            # Send concurrently: initialization and sync use world collectives.
            results = await asyncio.gather(
                *(
                    async_scheduler_client._forward_one(endpoint, obj, None)
                    for endpoint in async_scheduler_client.server_args.scheduler_endpoints
                ),
                return_exceptions=True,
            )
        errors = [
            str(result) if isinstance(result, BaseException) else result.error
            for result in results
            if isinstance(result, BaseException) or result.error
        ]
        if errors:
            raise HTTPException(status_code=500, detail="; ".join(errors))
        return results[0].output

    async def run_task_method(self, obj, request):
        output = await self._forward(obj)
        return output.result if isinstance(output, TaskMethodOutput) else output

    async def sync_hf_weight(self, obj, request):
        return await self._forward(obj)

    async def abort_generation(self, obj, request):
        raise HTTPException(
            status_code=400, detail="Embodied rollout does not support abort"
        )


def create_app(server_args, *, control_token: str):
    from sglang.multimodal_gen.runtime.entrypoints.http_server import (
        create_app as native_create_app,
    )

    app = native_create_app(server_args)
    app.add_middleware(
        PeerWhitelistMiddleware, allowed_paths=EMBODIED_WHITELISTED_PATHS
    )
    add_rlinf_routes(app, control_token=control_token, manager=_SchedulerManager())
    return app


def run_scheduler_process(*args, **kwargs):
    from sglang.multimodal_gen.runtime.managers.gpu_worker import (
        run_scheduler_process as native_run,
    )

    native_run(*args, **kwargs, scheduler_cls=Scheduler)
