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

"""Native SRT generation with RLinf's embodied routing and weight transport."""

from concurrent.futures import ThreadPoolExecutor

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.activevln import get_model
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class ActiveVLNRolloutWorker(MultiStepRolloutWorker):
    """Native SRT rollout with a CPU mirror for actor weight patches."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._engine = None
        self._engine_thread = None

    def init_worker(self):
        from sglang import Engine

        model_cfg = OmegaConf.merge(self.model_cfg, self.cfg.rollout.model)
        self.hf_model = get_model(model_cfg).cpu().eval()
        if self.cfg.runner.get("ckpt_path"):
            self.hf_model.load_state_dict(
                torch.load(self.cfg.runner.ckpt_path, map_location="cpu")
            )
        options = OmegaConf.to_container(
            self.cfg.rollout.get("sglang", {}), resolve=True
        )
        options.update(
            model_path=model_cfg.model_path,
            dtype=str(next(self.hf_model.parameters()).dtype).removeprefix("torch."),
            context_length=self.hf_model.max_context,
            tp_size=1,
            base_gpu_id=self.device.index
            if isinstance(self.device, torch.device)
            else int(self.device),
            enable_memory_saver=self.enable_offload,
            enable_weights_cpu_backup=self.enable_offload,
        )
        # Keep SRT's event loop separate from Ray.
        self._engine_thread = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="activevln-srt"
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
        for name, tensor in self.hf_model.named_srt_weights(
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
        self.log_info(
            f"Applied ActiveVLN actor weights to SRT at version {self.version}"
        )

    def _weight_call(self, method, *args, **kwargs):
        success, message = self._engine_call(method, *args, **kwargs)
        if not success:
            raise RuntimeError(f"SRT {method} failed: {message}")

    def predict(self, env_obs, mode="train"):
        if self._suspended:
            raise RuntimeError(
                "Resume the SRT engine before generating ActiveVLN actions"
            )
        self._apply_pending_weights()
        params = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )
        temperature = (
            float(params.get("temperature", 1.0))
            if params.get("do_sample", True)
            else 0.0
        )
        top_p = float(params.get("top_p", 1.0)) if temperature > 0 else 1.0
        repetition_penalty = float(params.get("repetition_penalty", 1.0))
        prompts = [
            self.hf_model.prepare_observation(messages, images)
            for messages, images in zip(
                env_obs["messages"], env_obs["images"], strict=True
            )
        ]
        # Share image preprocessing with actor replay.
        outputs = self._engine_call(
            "generate",
            input_ids=[ids[0].tolist() for ids, _, _ in prompts],
            image_data=[
                [dict(encoded, format="processor_output")] for _, _, encoded in prompts
            ],
            sampling_params={
                "temperature": temperature,
                "top_k": -1,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": self.hf_model.max_new_tokens,
            },
            return_logprob=True,
        )
        responses, logprobs = [], []
        for (ids, _, _), output in zip(prompts, outputs, strict=True):
            meta = output["meta_info"]
            tokens = [item[1] for item in meta["output_token_logprobs"]]
            if meta["prompt_tokens"] != ids.shape[1]:
                raise ValueError(
                    "SRT prompt length differs from the actor's image/token layout"
                )
            if "output_ids" in output and output["output_ids"] != tokens:
                raise ValueError("SRT output tokens and logprob tokens differ")
            responses.append(torch.tensor(tokens, dtype=torch.long))
            logprobs.append(
                torch.tensor([item[0] for item in meta["output_token_logprobs"]])
            )
        return self.hf_model.pack_rollout(
            prompts,
            responses,
            logprobs,
            sampling=(
                temperature or 1.0,
                top_p,
                repetition_penalty,
                float(temperature > 0),
            ),
        )

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
