# Copyright 2026 The RLinf Authors.
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

"""Native SRT rollout using registered model adapters and embodied routing."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.sglang_adapter import get_sglang_adapter_cls
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class SGLangEngineRolloutWorker(MultiStepRolloutWorker):
    """Own an SRT engine; adapters supply requests and mapped CPU weights."""

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self._engine = None
        self._engine_thread = None

    def init_worker(self) -> None:
        """Load the registered adapter and start its native SRT engine."""
        from sglang import Engine

        adapter_cls = get_sglang_adapter_cls(
            self.model_cfg.model_type, backend="engine"
        )
        if adapter_cls is None:
            raise ValueError(
                f"No SGLang engine adapter registered for {self.model_cfg.model_type!r}"
            )
        self.sglang_adapter = adapter_cls(self.cfg)
        self.hf_model = self.sglang_adapter.model
        if self.cfg.runner.get("ckpt_path"):
            self.hf_model.load_state_dict(
                torch.load(self.cfg.runner.ckpt_path, map_location="cpu")
            )
        options = OmegaConf.to_container(
            self.cfg.rollout.get("sglang", {}), resolve=True
        )
        options.update(self.sglang_adapter.get_engine_args())
        options.update(
            tp_size=1,
            base_gpu_id=self.device.index
            if isinstance(self.device, torch.device)
            else int(self.device),
            enable_memory_saver=self.enable_offload,
            enable_weights_cpu_backup=self.enable_offload,
        )
        # Keep SRT's event loop separate from Ray.
        self._engine_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sglang-rollout"
        )
        self._engine = None
        try:
            self._engine = self._engine_thread.submit(
                lambda: Engine(**options)
            ).result()
            self._suspended = False
            self._weights_dirty = bool(self.cfg.runner.get("ckpt_path"))
            self._first_weight_update = True
            self.setup_sample_params()
            if self.enable_offload:
                self.offload_model()
        except BaseException:
            self.shutdown()
            raise

    def _engine_call(self, method, *args, **kwargs):
        return self._engine_thread.submit(
            getattr(self._engine, method), *args, **kwargs
        ).result()

    async def sync_model_from_actor(self):
        # Receive CPU weights while the actor owns the GPU.
        await super().sync_model_from_actor()
        self._weights_dirty = True

    def _apply_pending_weights(self):
        if not self._weights_dirty:
            return
        self._weight_call("begin_weight_update", selector="target")
        bucket, size = [], 0
        limit = 256 * 1024**2
        for name, tensor in self.sglang_adapter.named_weights(
            trainable_only=not self._first_weight_update
        ):
            size += tensor.numel() * tensor.element_size()
            bucket.append((name, tensor))
            if size >= limit:
                self._weight_call(
                    "update_weights_from_tensor", bucket, flush_cache=False
                )
                bucket, size = [], 0
        if bucket:
            self._weight_call("update_weights_from_tensor", bucket, flush_cache=False)
        self._weight_call("end_weight_update")
        if not self._engine_call("flush_cache"):
            raise RuntimeError(
                "SRT could not flush stale KV after the actor weight update"
            )
        self._weights_dirty = False
        self._first_weight_update = False
        self.log_info(f"Applied actor weights to SRT at version {self.version}")

    def _weight_call(self, method, *args, **kwargs):
        success, message = self._engine_call(method, *args, **kwargs)
        if not success:
            raise RuntimeError(f"SRT {method} failed: {message}")

    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "train"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Generate native SRT responses and let the adapter pack trajectories."""
        if self._suspended:
            raise RuntimeError("Resume the SRT engine before generating actions")
        self._apply_pending_weights()
        params = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )
        request, state = self.sglang_adapter.build_request(
            env_obs, sampling_params=params
        )
        outputs = self._engine_call("generate", **request)
        return self.sglang_adapter.parse_response(outputs, state)

    def offload_model(self):
        if not self._suspended:
            self._engine_call("release_memory_occupation")
            self._suspended = True

    def reload_model(self):
        if self._suspended:
            self._engine_call("resume_memory_occupation")
            self._suspended = False
        # The runner waits for actor offload before resuming SRT.
        self._apply_pending_weights()

    def shutdown(self):
        """Release only the subprocesses and thread owned by this worker."""
        try:
            if self._engine is not None:
                self._engine_call("shutdown")
                self._engine = None
        finally:
            if getattr(self, "_engine_thread", None) is not None:
                self._engine_thread.shutdown(wait=True)
