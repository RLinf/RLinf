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

"""PhyAI-backed embodied rollout worker."""

from __future__ import annotations

import gc
import os
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlinf.config import torch_dtype_from_precision
from rlinf.data.schema.embodied_types import (
    EnvPart,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    split_batch_value,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.models.embodiment.openpi import _resolve_action_horizon_and_chunk
from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.transforms.apply import (
    apply_input_transform,
    apply_output_transform,
)
from rlinf.models.embodiment.openpi.transforms.env import repack_env_obs
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    build_openpi_transforms,
)
from rlinf.models.embodiment.prefix.config import is_prefix_off_policy_loss
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import (
    new_to_old_state_dict,
)
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size
from rlinf.utils.placement import HybridComponentPlacement

if TYPE_CHECKING:
    from phyai.engine import Engine
    from phyai.models.pi05.configuration_pi05 import PI05Config
    from phyai.models.pi05.scheduler_pi05 import PI05Request, PI05RolloutRequest

_VISION_INPUT_DTYPE = torch.float32


@dataclass(frozen=True)
class _PI05SamplingPlan:
    """RLinf-owned policy decisions plus the resulting backend request."""

    request: PI05RolloutRequest
    denoise_inds: torch.Tensor
    action_chunk: int
    action_dim: int


class _PhyAIWeightTarget(torch.nn.Module):
    """Receive bucket-synced actor weights on behalf of a PhyAI engine.

    ``BucketWeightSyncer.apply`` loads each bucket through ``load_state_dict``;
    this target converts the bucket to PhyAI parameter names and hands it to
    the engine's open weight update.
    """

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> None:
        """Convert one bucket of actor weights and apply it to the engine.

        Args:
            state_dict: Actor parameter names and tensors for one bucket.
            strict: Ignored; a bucket holds only part of the model.
            assign: Ignored; the engine copies into its own buffers.
        """
        del strict, assign
        normalized = {
            PhyAIRolloutWorker._actor_weight_name(name): tensor
            for name, tensor in state_dict.items()
        }
        converted = new_to_old_state_dict(normalized)
        converted.update(
            (name, tensor)
            for name, tensor in normalized.items()
            if name.startswith("value_head.")
        )
        self.engine.update_weights(converted)


class PhyAIRolloutWorker(Worker):
    """Embodied rollout worker that runs PI0.5 through an in-process PhyAI engine.

    Each worker owns one PhyAI ``Engine`` on one GPU. Evaluation asks
    ``Engine.step`` for actions only. Training builds the denoising plan in
    RLinf, runs it with ``Engine.rollout_step``, and returns the replay data
    the OpenPI actor reads: ``prev_logprobs``, ``prev_values``, and
    ``forward_inputs``.

    Args:
        cfg: Experiment config. ``rollout.phyai`` holds the engine options.
            Training reads the model from ``actor.model``, with
            ``rollout.model.model_path`` and ``rollout.model.precision`` as
            overrides; evaluation reads ``rollout.model``.
    """

    def __init__(self, cfg: DictConfig) -> None:
        Worker.__init__(self)

        self._phyai_cfg = cfg.rollout.get("phyai", {})
        self._phyai_plugin = self._phyai_cfg.get("plugin", "pi05")
        if self._phyai_plugin != "pi05":
            raise NotImplementedError(
                "PhyAIRolloutWorker supports the 'pi05' engine plugin; "
                f"got {self._phyai_plugin!r}."
            )

        self.cfg = cfg
        self.only_eval = bool(cfg.runner.get("only_eval", False))
        self.model_cfg = cfg.rollout.model if self.only_eval else cfg.actor.model
        self.actor_group_name = (
            cfg.actor.get("group_name", None)
            if cfg.get("actor", None) is not None
            else None
        )
        self.num_pipeline_stages = int(cfg.rollout.pipeline_stage_num)
        self.enable_offload = bool(cfg.rollout.get("enable_offload", False))
        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0

        train_env_cfg = cfg.env.get("train", None)
        eval_env_cfg = cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        self.rollout_epoch = (
            int(train_env_cfg.rollout_epoch) if train_env_cfg is not None else 1
        )
        self.eval_rollout_epoch = (
            int(eval_env_cfg.rollout_epoch) if self.enable_eval else 1
        )
        self.total_num_train_envs = (
            int(train_env_cfg.total_num_envs) if self.enable_train else 0
        )
        self.total_num_eval_envs = (
            int(eval_env_cfg.total_num_envs) if self.enable_eval else 0
        )
        self.train_batch_size = self.total_num_train_envs // self.num_pipeline_stages
        self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages
        self.per_node_train_batch_size = (
            self.train_batch_size // self._world_size if self.enable_train else 0
        )
        self.per_node_eval_batch_size = (
            self.eval_batch_size // self._world_size if self.enable_eval else 0
        )
        self.n_train_chunk_steps = (
            int(train_env_cfg.max_steps_per_rollout_epoch)
            // int(self.model_cfg.num_action_chunks)
            if self.enable_train
            else 0
        )
        self.n_eval_chunk_steps = (
            int(eval_env_cfg.max_steps_per_rollout_epoch)
            // int(self.model_cfg.num_action_chunks)
            if self.enable_eval
            else 0
        )
        self.collect_transitions = bool(cfg.rollout.get("collect_transitions", False))
        self.algorithm_cfg = cfg.get("algorithm", {})
        self.enable_prefix = is_prefix_off_policy_loss(
            self.algorithm_cfg.get("loss_type")
        )
        self.collect_prev_infos = bool(cfg.rollout.get("collect_prev_infos", True))
        self.version = 0
        self.weight_syncer = None
        self._sync_weight_comm_options = None
        if not self.only_eval:
            weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
            if weight_syncer_cfg is None:
                raise ValueError("rollout.weight_syncer config must be provided")
            self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
            self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._engine = None
        self._input_transform_fn = None
        self._output_transform_fn = None
        self._openpi_config_name = ""
        self._state_indices: list[int] | None = None
        self._weight_target = None
        self._engine_device: torch.device | None = None
        self._num_images = 0
        self._model_action_horizon = 0
        self._model_action_dim = 0
        self._num_steps = 0

        if self.enable_offload:
            raise NotImplementedError(
                "PhyAIRolloutWorker does not support rollout.enable_offload yet."
            )
        if cfg.runner.get("enable_decoupled_mode", False):
            raise NotImplementedError(
                "PhyAIRolloutWorker does not support runner.enable_decoupled_mode yet."
            )
        if len(self.global_accelerator_ids) != 1:
            raise ValueError(
                "Expected exactly one GPU per PhyAI rollout worker, got "
                f"accelerators {self.global_accelerator_ids}."
            )

    @property
    def collect_raw_transition_obs(self) -> bool:
        """Whether trajectory parts carry raw observations for the actor."""
        return self.collect_transitions and not self.enable_prefix

    def init_worker(self) -> None:
        """Build the PhyAI engine and the OpenPI transforms on this worker's GPU.

        The engine loads the checkpoint at the model's ``model_path``. When
        ``runner.ckpt_path`` is set, that state dict is applied on top, so
        evaluation and the first training rollout start from the same weights
        as the actor.

        Raises:
            RuntimeError: If the engine is already initialized.
        """
        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import (
            DeviceConfig,
            EngineConfig,
            KernelConfig,
            RuntimeConfig,
        )
        from phyai.models.pi05.main_pi05 import PI05Args

        if self._engine is not None:
            raise RuntimeError("PhyAI engine is already initialized.")
        if not self.only_eval:
            rollout_model_overrides = OmegaConf.masked_copy(
                self.cfg.rollout.model, ["model_path", "precision"]
            )
            self.model_cfg = OmegaConf.merge(
                self.model_cfg,
                rollout_model_overrides,
            )

        checkpoint_dir = str(self.model_cfg.model_path)
        plugin_cfg = self._build_plugin_config(self.model_cfg)
        engine_precision = self._phyai_cfg.get("params_dtype", self.model_cfg.precision)
        if engine_precision is None:
            engine_precision = "bf16"
        dtype = torch_dtype_from_precision(engine_precision)
        if dtype is None:
            raise ValueError(
                f"Unsupported PhyAI model precision: {engine_precision!r}."
            )
        self._engine_device = torch.device(self.torch_device_type)

        runtime_cfg = self._phyai_cfg.get("runtime", {})
        runtime_values = (
            OmegaConf.to_container(runtime_cfg, resolve=True)
            if OmegaConf.is_config(runtime_cfg)
            else runtime_cfg
        )
        runtime_values = dict(runtime_values or {})
        requested_cuda_graph = bool(
            self._phyai_cfg.get("use_cuda_graph", self.only_eval)
        )
        runtime_values.setdefault("use_cuda_graph", requested_cuda_graph)
        kernel_config = self._phyai_cfg.get("kernel_config")
        if isinstance(kernel_config, Mapping):
            policy_config = (
                OmegaConf.to_container(kernel_config, resolve=True)
                if OmegaConf.is_config(kernel_config)
                else dict(kernel_config)
            )
            kernel_cfg = KernelConfig(policy_config=policy_config)
        else:
            kernel_cfg = KernelConfig(
                config_path=(
                    str(Path(kernel_config).expanduser()) if kernel_config else None
                )
            )

        configured_max_batch_size = self._phyai_cfg.get("max_batch_size", None)
        max_batch_size = int(
            configured_max_batch_size
            if configured_max_batch_size is not None
            else max(self.per_node_train_batch_size, self.per_node_eval_batch_size, 1)
        )
        openpi_cfg = self.model_cfg.get("openpi", {})
        num_images = int(openpi_cfg.get("num_images_in_input", 3))
        config_name = str(openpi_cfg.get("config_name", ""))
        data_kwargs = OmegaConf.select(self.model_cfg, "openpi_data", default=None)
        if data_kwargs is not None:
            data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
        input_transforms, output_transforms = build_openpi_transforms(
            self.model_cfg.model_path,
            config_name,
            data_kwargs=data_kwargs,
        )
        from openpi.transforms import compose

        self._input_transform_fn = compose(input_transforms)
        self._output_transform_fn = compose(output_transforms)
        self._openpi_config_name = config_name
        state_indices = openpi_cfg.get("state_indices")
        self._state_indices = list(state_indices) if state_indices else None
        vision_dtype = self._optional_dtype(
            self._phyai_cfg.get("vision_params_dtype", None)
        )
        inputs_image_shape = [
            [
                plugin_cfg.vision.image_size,
                plugin_cfg.vision.image_size,
                plugin_cfg.vision.num_channels,
            ]
            for _ in range(num_images)
        ]

        self.log_info(
            "Launching PhyAI engine: "
            f"plugin={self._phyai_plugin}, weight_checkpoint={checkpoint_dir}, "
            "model_config_source=rlinf, "
            f"max_batch_size={max_batch_size}, num_images={num_images}, "
            f"device={self._engine_device}, dtype={dtype}, "
            f"kernel_config={'inline' if isinstance(kernel_config, Mapping) else kernel_cfg.config_path}."
        )
        # RLinf's worker-group launcher exports its own WORLD_SIZE/RANK. The
        # single-GPU PhyAI engine must resolve as an independent replica.
        launcher_keys = ["WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT"]
        if isinstance(kernel_config, Mapping):
            # Explicit Hydra rules take precedence over a kernel file from the environment.
            launcher_keys.append("PHYAI_KERNEL_CONFIG")
        launcher_values = {key: os.environ.pop(key, None) for key in launcher_keys}
        try:
            engine = Engine(
                EngineArgs(
                    plugin=self._phyai_plugin,
                    plugin_args=PI05Args(
                        checkpoint_dir=checkpoint_dir,
                        max_batch_size=max_batch_size,
                        config=plugin_cfg,
                        weight_remap=self._actor_weight_name,
                        vision_params_dtype=vision_dtype,
                        inputs_image_shape=inputs_image_shape,
                        capture_rollout=not self.only_eval and requested_cuda_graph,
                    ),
                    config=EngineConfig(
                        device=DeviceConfig(
                            target=str(self._engine_device),
                            params_dtype=dtype,
                        ),
                        kernel=kernel_cfg,
                        runtime=RuntimeConfig(**runtime_values),
                    ),
                )
            )
        finally:
            for key, value in launcher_values.items():
                if value is not None:
                    os.environ[key] = value

        self._engine = engine
        self._num_images = num_images
        self._model_action_horizon = int(plugin_cfg.chunk_size)
        self._model_action_dim = int(plugin_cfg.max_action_dim)
        self._num_steps = int(plugin_cfg.num_inference_steps)
        self._weight_target = _PhyAIWeightTarget(engine)

        checkpoint_path = self.cfg.runner.get("ckpt_path", None)
        if checkpoint_path:
            checkpoint_state = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            engine.begin_weight_update()
            try:
                self._weight_target.load_state_dict(checkpoint_state)
                engine.finish_weight_update()
            except Exception:
                engine.abort_weight_update()
                raise
            self.log_info(
                f"Loaded runner.ckpt_path into PhyAI engine: {checkpoint_path}."
            )

    @staticmethod
    def _build_plugin_config(model_cfg: Any) -> PI05Config:
        """Map RLinf's PI0.5 model semantics to the PhyAI plugin config."""
        from phyai.models.pi05.configuration_pi05 import PI05Config

        defaults = PI05Config()
        openpi_cfg = model_cfg.get("openpi", {})
        return PI05Config(
            chunk_size=_resolve_action_horizon_and_chunk(model_cfg, openpi_cfg)[0],
            max_action_dim=int(
                openpi_cfg.get("model_action_dim", defaults.max_action_dim)
            ),
            num_inference_steps=int(model_cfg.num_steps),
            tokenizer_max_length=int(
                openpi_cfg.get("max_token_len", defaults.tokenizer_max_length)
            ),
            add_value_head=bool(model_cfg.get("add_value_head", False)),
            value_after_vlm=bool(
                openpi_cfg.get("value_after_vlm", defaults.value_after_vlm)
            ),
        )

    @staticmethod
    def _optional_dtype(value: Any) -> torch.dtype | None:
        return None if value is None else torch_dtype_from_precision(value)

    @staticmethod
    def _actor_weight_name(name: str) -> str:
        """Map the native wrapper prefix to PhyAI HF parameter names."""
        return name[len("model.") :] if name.startswith("model.") else name

    def _select_configured_state(self, states: Any) -> Any:
        indices = self._state_indices
        if not indices:
            return states
        state_dim = states.shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )
        if torch.is_tensor(states):
            index = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index)
        return np.asarray(states)[..., indices]

    def _observation_to_device(self, transformed: dict[str, Any]) -> Observation:
        assert self._engine_device is not None
        observation = Observation.from_dict(transformed)

        def move(value: Any, *, dtype: torch.dtype | None = None) -> Any:
            if not torch.is_tensor(value):
                return value
            return value.to(device=self._engine_device, dtype=dtype)

        return Observation(
            images={key: move(value) for key, value in observation.images.items()},
            image_masks={
                key: move(value) for key, value in observation.image_masks.items()
            },
            state=move(observation.state, dtype=torch.float32),
            tokenized_prompt=move(observation.tokenized_prompt),
            tokenized_prompt_mask=move(observation.tokenized_prompt_mask),
            token_ar_mask=move(observation.token_ar_mask),
            token_loss_mask=move(observation.token_loss_mask),
            pcd_xyz=move(observation.pcd_xyz),
        )

    def _build_request(
        self, env_obs: dict[str, Any]
    ) -> tuple[PI05Request, Observation]:
        from phyai.models.pi05.scheduler_pi05 import PI05Request

        assert self._engine_device is not None
        assert self._input_transform_fn is not None
        repacked = repack_env_obs(
            self._openpi_config_name,
            env_obs,
            select_state=self._select_configured_state,
        )
        transformed = apply_input_transform(self._input_transform_fn, repacked)
        image_keys = IMAGE_KEYS[: self._num_images]
        for key in image_keys:
            if not bool(transformed["image_mask"][key].all()):
                raise ValueError(
                    f"PhyAI cannot omit masked image tokens for input {key!r}."
                )
        token_mask = transformed.get("tokenized_prompt_mask")
        if token_mask is None or transformed.get("tokenized_prompt") is None:
            raise RuntimeError("OpenPI preprocessing did not produce prompt tokens.")
        lang_lens = token_mask.sum(dim=-1, dtype=torch.int64)
        expected_mask = torch.arange(token_mask.shape[-1])[None, :] < lang_lens[:, None]
        if not torch.equal(token_mask, expected_mask):
            raise ValueError("PhyAI requires right-padded OpenPI prompt tokens.")

        processed = self._observation_to_device(transformed)
        processed = preprocess_observation(processed, train=False)
        pixel_values = torch.stack(
            [processed.images[key].permute(0, 3, 1, 2) for key in image_keys],
            dim=1,
        ).to(device=self._engine_device, dtype=_VISION_INPUT_DTYPE)

        input_ids = processed.tokenized_prompt
        input_ids = input_ids.to(device=self._engine_device)

        request = PI05Request(
            pixel_values=pixel_values.contiguous(),
            input_ids=input_ids.contiguous(),
            lang_lens=lang_lens.to(device=self._engine_device).contiguous(),
        )
        return request, processed

    def _build_rollout_plan(self, request: PI05Request) -> _PI05SamplingPlan:
        """Resolve RLinf policy choices into a sigma execution plan."""
        from phyai.models.pi05.scheduler_pi05 import PI05RolloutRequest

        assert self._engine_device is not None
        openpi_cfg = self.model_cfg.get("openpi", {})
        batch_size = int(request.pixel_values.shape[0])
        num_steps = self._num_steps

        action_horizon = self._model_action_horizon
        model_action_dim = self._model_action_dim
        action_chunk = int(self.model_cfg.num_action_chunks)
        action_dim = int(self.model_cfg.action_dim)
        if not 1 <= action_chunk <= action_horizon:
            raise ValueError(
                f"num_action_chunks must be in [1, {action_horizon}], got "
                f"{action_chunk}."
            )
        if not 1 <= action_dim <= model_action_dim:
            raise ValueError(
                f"action_dim must be in [1, {model_action_dim}], got {action_dim}."
            )

        ignore_last = bool(openpi_cfg.get("ignore_last", False))
        if ignore_last and num_steps < 2:
            raise ValueError(
                "Expected at least two denoise steps with ignore_last=True, "
                f"got num_steps={num_steps}."
            )
        last = num_steps - 2 if ignore_last else num_steps - 1
        selected = random.randint(0, last)
        denoise_inds = torch.full(
            (batch_size, num_steps),
            selected,
            dtype=torch.int64,
            device=self._engine_device,
        )

        noise_method = str(openpi_cfg.get("noise_method", "flow_ode"))
        noise_level = float(openpi_cfg.get("noise_level", 0.0))
        sigmas = torch.zeros(
            batch_size,
            num_steps,
            dtype=torch.float32,
            device=self._engine_device,
        )
        if noise_method == "flow_sde" and noise_level != 0:
            timesteps = torch.linspace(
                1.0,
                1.0 / num_steps,
                num_steps,
                dtype=torch.float32,
                device=self._engine_device,
            )
            timesteps = torch.cat(
                [
                    timesteps,
                    torch.zeros(1, dtype=torch.float32, device=self._engine_device),
                ]
            )
            denominator = torch.where(timesteps == 1, timesteps[1], timesteps)
            schedule = noise_level * torch.sqrt(timesteps / (1.0 - denominator))[:-1]
            step_ids = torch.arange(num_steps, device=self._engine_device)[None]
            sigmas.copy_(
                torch.where(step_ids == denoise_inds[:, :1], schedule[None], 0.0)
            )

        execution_request = PI05RolloutRequest(
            pixel_values=request.pixel_values,
            input_ids=request.input_ids,
            lang_lens=request.lang_lens,
            sigmas=sigmas.contiguous(),
            compute_values=bool(self.model_cfg.get("add_value_head", False)),
        )
        return _PI05SamplingPlan(
            request=execution_request,
            denoise_inds=denoise_inds,
            action_chunk=action_chunk,
            action_dim=action_dim,
        )

    @staticmethod
    def _reduce_rollout_outputs(
        plan: _PI05SamplingPlan,
        rollout_result: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply RLinf's PPO log-probability and value reductions."""
        chains = rollout_result.chains
        if chains.ndim != 4:
            raise ValueError(
                f"PhyAI chains must have shape (B, N+1, H, D); got {chains.shape}."
            )
        batch_size, chain_steps, action_horizon, model_action_dim = chains.shape
        num_steps = int(plan.denoise_inds.shape[1])
        if chain_steps != num_steps + 1:
            raise ValueError(
                f"PhyAI chains contain {chain_steps} states for {num_steps} transitions."
            )
        expected_logprob_shape = (
            batch_size,
            num_steps,
            action_horizon,
            model_action_dim,
        )
        if rollout_result.transition_logprobs.shape != expected_logprob_shape:
            raise ValueError(
                "PhyAI transition_logprobs shape "
                f"{tuple(rollout_result.transition_logprobs.shape)} != "
                f"{expected_logprob_shape}."
            )
        cropped = rollout_result.transition_logprobs[
            :, :, : plan.action_chunk, : plan.action_dim
        ]
        selected = plan.denoise_inds[:, 0].to(device=cropped.device)
        batch_indices = torch.arange(batch_size, device=cropped.device)
        prev_logprobs = cropped[batch_indices, selected]
        prev_logprobs = prev_logprobs.to(torch.float32).contiguous()

        raw_values = rollout_result.raw_values
        if not plan.request.compute_values:
            return prev_logprobs, None
        if raw_values is None:
            raise ValueError("PhyAI did not return requested rollout values.")
        if raw_values.ndim != 2 or raw_values.shape[0] != batch_size:
            raise ValueError(
                "PhyAI values must have shape (B, value_steps); got "
                f"{tuple(raw_values.shape)}."
            )
        prev_values = raw_values.to(torch.float32).mean(dim=1, keepdim=True)
        return prev_logprobs, prev_values.contiguous()

    def _build_openpi_forward_inputs(
        self,
        processed: Observation,
        rollout_result: Any,
        denoise_inds: torch.Tensor,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the preprocessed replay contract consumed by openpi."""
        if processed.state is None:
            raise RuntimeError("OpenPI preprocessing did not produce normalized state.")
        if (
            processed.tokenized_prompt is None
            or processed.tokenized_prompt_mask is None
        ):
            raise RuntimeError("OpenPI preprocessing did not produce prompt tokens.")

        batch_size = int(processed.state.shape[0])
        token_ids = processed.tokenized_prompt.to(device=self._engine_device)
        token_mask = processed.tokenized_prompt_mask.to(device=self._engine_device)

        state = processed.state.to(device=self._engine_device, dtype=torch.float32)
        model_action_dim = int(
            self.model_cfg.get("openpi", {}).get(
                "model_action_dim", rollout_result.actions.shape[-1]
            )
        )
        if state.shape[-1] > model_action_dim:
            raise ValueError(
                f"Normalized state dim {state.shape[-1]} exceeds "
                f"openpi model_action_dim={model_action_dim}."
            )
        state = torch.nn.functional.pad(state, (0, model_action_dim - state.shape[-1]))

        forward_inputs = {
            "chains": rollout_result.chains,
            "denoise_inds": denoise_inds,
            "obs_state": state,
            "tokenized_prompt": token_ids,
            "tokenized_prompt_mask": token_mask,
            "action": actions.to(device=self._engine_device).reshape(batch_size, -1),
            "model_action": rollout_result.actions.reshape(batch_size, -1),
        }
        for key in IMAGE_KEYS:
            forward_inputs[f"obs_image__{key}"] = processed.images[key].to(
                device=self._engine_device
            )
            forward_inputs[f"obs_image_mask__{key}"] = processed.image_masks[key].to(
                device=self._engine_device
            )
        return {key: value.contiguous() for key, value in forward_inputs.items()}

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict action chunks for a batch of environment observations.

        Args:
            env_obs: Batched environment observations.
            mode: ``"eval"`` returns actions only. ``"train"`` also samples
                the PPO denoising step and returns the replay data.

        Returns:
            Actions of shape ``[batch, num_action_chunks, action_dim]`` and a
            dict with ``prev_logprobs``, ``prev_values`` (both ``None`` in
            eval mode), ``forward_inputs`` for the actor, and
            ``expert_label_flag``.

        Raises:
            RuntimeError: If ``init_worker`` has not run, or train mode is
                requested on an evaluation-only worker.
        """
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported PhyAI rollout mode: {mode!r}.")
        if mode == "train" and self.only_eval:
            raise RuntimeError("An evaluation-only PhyAI worker cannot run train mode.")
        if self._engine is None or self._input_transform_fn is None:
            raise RuntimeError("init_worker() must be called before PhyAI inference.")

        request, processed = self._build_request(env_obs)
        rollout_plan = None
        rollout_result = None
        if mode == "train":
            rollout_plan = self._build_rollout_plan(request)
            rollout_result = self._engine.rollout_step(rollout_plan.request)
            model_actions = rollout_result.actions
        else:
            model_actions = self._engine.step(request)
        assert self._output_transform_fn is not None
        actions = apply_output_transform(
            self._output_transform_fn,
            {"actions": model_actions, "state": processed.state},
        )["actions"]
        requested_chunks = int(self.model_cfg.num_action_chunks)
        if actions.shape[1] < requested_chunks:
            raise ValueError(
                "PhyAI returned fewer action chunks than RLinf requested: "
                f"returned={actions.shape[1]}, requested={requested_chunks}."
            )
        actions = actions[:, :requested_chunks]
        actions = actions.to(dtype=torch.float32).contiguous()

        batch_size = actions.shape[0]
        if rollout_result is None:
            result = {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {
                    "action": actions.reshape(batch_size, -1),
                    "model_action": model_actions.reshape(batch_size, -1),
                },
                "expert_label_flag": False,
            }
        else:
            assert rollout_plan is not None
            prev_logprobs, prev_values = self._reduce_rollout_outputs(
                rollout_plan, rollout_result
            )
            result = {
                "prev_logprobs": prev_logprobs,
                "prev_values": prev_values,
                "forward_inputs": self._build_openpi_forward_inputs(
                    processed,
                    rollout_result,
                    rollout_plan.denoise_inds,
                    actions,
                ),
                "expert_label_flag": False,
            }
        return actions, result

    @staticmethod
    def _infer_policy_input_batch_size(policy_input: PolicyInput) -> int:
        return infer_obs_batch_size(policy_input.obs)

    @staticmethod
    def _decompress_policy_input(policy_input: PolicyInput) -> PolicyInput:
        policy_input.obs = decompress_obs(policy_input.obs)
        for env_part in policy_input.env_parts:
            if env_part is not None and env_part.next_obs is not None:
                env_part.next_obs = decompress_obs(env_part.next_obs)
        return policy_input

    @classmethod
    def _merge_policy_inputs(cls, policy_inputs: list[PolicyInput]) -> PolicyInput:
        return PolicyInput.merge(
            [cls._decompress_policy_input(item) for item in policy_inputs]
        )

    async def _receive_policy_input(
        self,
        channel: Channel,
        tag: str,
        stage_id: int,
    ) -> PolicyInput:
        return await self.recv_from(
            group_name=self.cfg.env.group_name,
            channel=channel,
            tag=f"train_{tag}",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            merge_fn=self._merge_policy_inputs,
            infer_batch_size_fn=self._infer_policy_input_batch_size,
        ).async_wait()

    def _build_policy_output(self, result: dict[str, Any]) -> PolicyOutput:
        return PolicyOutput(
            prev_logprobs=result["prev_logprobs"] if self.collect_prev_infos else None,
            prev_values=result["prev_values"] if self.collect_prev_infos else None,
            intervene_flags=None,
            forward_inputs=result["forward_inputs"],
            versions=torch.full_like(
                result["prev_logprobs"], float(self.version), dtype=torch.float32
            )
            if self.collect_prev_infos
            else None,
        )

    def _publish_env_part(
        self,
        env_part: EnvPart,
        policy_obs: dict[str, Any] | None,
        actor_channel: Channel,
    ) -> None:
        next_obs = env_part.next_obs if env_part.next_obs is not None else policy_obs
        final_prev_values = None
        if env_part.requires_inference:
            if next_obs is None:
                raise ValueError(
                    "Terminal inference requires a post-action observation."
                )
            _, result = self.predict(next_obs, mode="train")
            values = result.get("prev_values")
            if values is not None:
                final_prev_values = values.cpu().contiguous()
        actor_channel.put(
            env_part.complete(
                next_obs=(next_obs if self.collect_raw_transition_obs else None),
                next_rlt_obs=None,
                final_prev_values=final_prev_values,
            ),
            async_op=True,
        )

    def _publish_env_parts(
        self,
        policy_input: PolicyInput,
        actor_channel: Channel,
    ) -> None:
        if not policy_input.env_parts:
            return
        if len(policy_input.env_parts) != len(policy_input.request_sizes):
            raise ValueError("Environment parts and request sizes must align.")

        needs_policy_obs = any(
            part is not None
            and (
                self.collect_raw_transition_obs
                or (part.requires_inference and part.next_obs is None)
            )
            for part in policy_input.env_parts
        )
        policy_observations = (
            split_batch_value(policy_input.obs, policy_input.request_sizes)
            if needs_policy_obs
            else [None] * len(policy_input.env_parts)
        )
        for env_part, policy_obs in zip(
            policy_input.env_parts,
            policy_observations,
        ):
            if env_part is not None:
                self._publish_env_part(env_part, policy_obs, actor_channel)

    @staticmethod
    def _split_actions(actions: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        return [shard.contiguous() for shard in torch.split(actions, sizes, dim=0)]

    def _send_actions(
        self,
        channel: Channel,
        actions: torch.Tensor,
        stage_id: int,
    ) -> None:
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=channel,
            data=actions.detach().cpu().contiguous(),
            tag="policy",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            split_fn=self._split_actions,
        )

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Serve one training rollout epoch.

        Args:
            input_channel: Channel the env workers send policy inputs on.
            output_channel: Channel to send actions back to the env workers.
            actor_channel: Channel that collects trajectory parts for the actor.
        """
        for _ in range(self.n_train_chunk_steps):
            for stage_id in range(self.num_pipeline_stages):
                policy_input = await self._receive_policy_input(
                    input_channel, "policy", stage_id
                )
                if policy_input.is_last:
                    raise ValueError("Received a final input in the policy stream.")
                if not policy_input.requires_inference:
                    actions = policy_input.external_actions
                    policy_output = None
                else:
                    actions, result = self.predict(policy_input.obs, mode="train")
                    policy_output = self._build_policy_output(result)
                self._send_actions(output_channel, actions.contiguous(), stage_id)
                transition_obs = (
                    policy_input.obs if self.collect_raw_transition_obs else {}
                )
                if policy_input.requires_inference:
                    trajectory_part = PolicyPart(
                        sources=policy_input.sources,
                        obs=transition_obs,
                        output=policy_output,
                    )
                else:
                    trajectory_part = PolicyPart(
                        sources=policy_input.sources,
                        obs=transition_obs,
                        external_actions=actions,
                    )
                actor_channel.put(trajectory_part, async_op=True)
                self._publish_env_parts(policy_input, actor_channel)

        for stage_id in range(self.num_pipeline_stages):
            policy_input = await self._receive_policy_input(
                input_channel, "policy_final", stage_id
            )
            if not policy_input.is_last:
                raise ValueError("Expected a final policy input at epoch end.")
            self._publish_env_parts(policy_input, actor_channel)

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Serve ``env.train.rollout_epoch`` training rollout epochs.

        Args:
            input_channel: Channel the env workers send policy inputs on.
            output_channel: Channel to send actions back to the env workers.
            actor_channel: Channel that collects trajectory parts for the actor.
        """
        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(
                input_channel,
                output_channel,
                actor_channel,
            )

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_batches = [decompress_obs(obs_batch) for obs_batch in obs_batches]
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        merged: dict[str, Any] = {}
        for key in obs_dicts[0]:
            values = [obs_dict[key] for obs_dict in obs_dicts]
            first = next((value for value in values if value is not None), None)
            if first is None:
                merged[key] = None
            elif isinstance(first, torch.Tensor):
                merged[key] = torch.cat(values, dim=0)
            elif isinstance(first, list):
                merged[key] = [item for value in values for item in value]
            else:
                merged[key] = values
        return {"obs": merged}

    @Worker.timer("evaluate")
    async def evaluate(self, input_channel: Channel, output_channel: Channel) -> None:
        """Serve evaluation epochs with action-only inference.

        Args:
            input_channel: Channel the env workers send observations on.
            output_channel: Channel to send actions back to the env workers.
        """
        for _ in tqdm(
            range(self.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for stage_id in range(self.num_pipeline_stages):
                    env_output = await self.recv_from(
                        group_name=self.cfg.env.group_name,
                        channel=input_channel,
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                        merge_fn=self._merge_obs_batches,
                        infer_batch_size_fn=infer_obs_batch_size,
                    ).async_wait()
                    actions, _ = self.predict(env_output["obs"], mode="eval")
                    self.send_to(
                        group_name=self.cfg.env.group_name,
                        channel=output_channel,
                        data=actions.detach().cpu().contiguous(),
                        tag="eval_rollout_results",
                        route_key=stage_id,
                        async_op=True,
                        batch_size=self.eval_batch_size,
                    )

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self) -> None:
        """Receive the actor's weights and load them into the PhyAI engine.

        The update is opened before the first bucket arrives and finished after
        the last one. If receiving or applying any bucket fails, the update is
        aborted, the error propagates to the runner, and ``self.version`` keeps
        its previous value.

        Raises:
            RuntimeError: If ``init_worker`` has not run or no weight syncer is
                configured.
            NotImplementedError: If the weight syncer is not a bucket syncer.
        """
        if self._engine is None or self._weight_target is None:
            raise RuntimeError("init_worker() must be called before weight sync.")
        if self.weight_syncer is None:
            raise RuntimeError("PhyAI weight sync requires weight_syncer config.")
        if not isinstance(self.weight_syncer, BucketWeightSyncer):
            # Patch sync assumes the same state-dict layout on both sides, but
            # PhyAI stores fused weights under different names.
            raise NotImplementedError(
                "Expected a BucketWeightSyncer for PhyAI, got "
                f"{type(self.weight_syncer).__name__}."
            )

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=None,
                recv=recv_func,
                send=send_func,
            )

        self._engine.begin_weight_update()
        try:
            applied_version = await self.weight_syncer.apply(
                self._weight_target,
                recv_func,
            )
            report = self._engine.finish_weight_update(version=applied_version)
        except Exception:
            self._engine.abort_weight_update()
            raise
        self.version = applied_version
        self.log_info(
            "PhyAI hot weight update applied: "
            f"version={self.version}, loaded={len(report.loaded)}."
        )
        gc.collect()
        if self.torch_platform is not None:
            self.torch_platform.empty_cache()

    def set_global_step(self, global_step: int) -> None:
        """Ignore the step; PhyAI sampling has no step-dependent schedule.

        Args:
            global_step: Current training step, unused.
        """
        del global_step
