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

import gc
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, KernelConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_pi05 import (
    PI05Request,
    PI05RolloutRequest,
)
from torch.utils._pytree import tree_map

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.models.embodiment.openpi import _resolve_action_horizon_and_chunk
from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
    preprocess_observation,
)
from rlinf.models.embodiment.openpi.transforms.pipeline import (
    build_openpi_transforms,
)
from rlinf.scheduler import Worker
from rlinf.utils.ckpt_convertor.openpi.openpi_to_openpi_pytorch import (
    new_to_old_state_dict,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

_VISION_INPUT_DTYPE = torch.float32


@dataclass(frozen=True)
class _PI05SamplingPlan:
    """RLinf-owned policy decisions plus the resulting backend request."""

    request: PI05RolloutRequest
    denoise_inds: torch.Tensor
    action_chunk: int
    action_dim: int
    joint_logprob: bool


class _PhyAIWeightTarget(torch.nn.Module):
    """Adapt bucket syncer's ``load_state_dict`` calls to a PhyAI Engine."""

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        del strict, assign
        normalized = {
            PhyAIWorker._actor_weight_name(name): tensor
            for name, tensor in state_dict.items()
        }
        converted = new_to_old_state_dict(normalized)
        converted.update(
            (name, tensor)
            for name, tensor in normalized.items()
            if name.startswith("value_head.")
        )
        self.engine.update_weights(converted)
        return None


class PhyAIWorker(MultiStepRolloutWorker):
    """Run the reusable multi-step rollout loop with a PhyAI engine.

    The RLinf worker remains the Ray actor and owns one in-process PhyAI
    ``Engine``. Channel communication, batch routing, and the evaluation loop
    are inherited from :class:`MultiStepRolloutWorker`; model construction and
    action inference are replaced with the PhyAI engine path.

    Evaluation keeps the original action-only ``Engine.step`` path. For
    training, RLinf constructs the sampling plan, asks ``Engine.rollout_step``
    to execute it, and reduces the raw trajectory into the OpenPI replay
    contract consumed by the Actor.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self._phyai_cfg = cfg.rollout.get("phyai", {})
        self._phyai_plugin = str(self._phyai_cfg.get("plugin", "pi05")).lower()
        self._engine = None
        self._input_transform_fn = None
        self._output_transform_fn = None
        self._openpi_config_name = ""
        self._state_indices: list[int] | None = None
        self._weight_target = None
        self._engine_device: torch.device | None = None
        self._engine_dtype: torch.dtype | None = None
        self._num_images = 0
        self._model_action_horizon = 0
        self._model_action_dim = 0
        self._num_steps = 0

        if self._phyai_plugin != "pi05":
            raise NotImplementedError(
                "PhyAIWorker supports the 'pi05' engine plugin; "
                f"got {self._phyai_plugin!r}."
            )
        if self.enable_offload:
            raise NotImplementedError(
                "PhyAIWorker does not support rollout.enable_offload yet."
            )
        if len(self.global_accelerator_ids) != 1:
            raise ValueError(
                "The initial PhyAI integration requires exactly one GPU per "
                "rollout Ray actor (PhyAI world_size=1); got "
                f"{self.global_accelerator_ids}."
            )

    # Configuration and engine setup.

    def init_worker(self) -> None:
        """Construct one PhyAI engine and the shared OpenPI transforms."""
        if self._engine is not None:
            raise RuntimeError("PhyAI engine is already initialized.")
        if str(self.model_cfg.get("model_type", "")) != "openpi":
            raise NotImplementedError(
                "PhyAIWorker requires actor.model.model_type='openpi'."
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
        self._engine_dtype = dtype
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
        kernel_config_path = self._phyai_cfg.get("kernel_config")
        kernel_cfg = KernelConfig(
            config_path=(
                str(Path(kernel_config_path).expanduser())
                if kernel_config_path
                else None
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
        if not 1 <= num_images <= len(IMAGE_KEYS):
            raise ValueError(
                f"openpi.num_images_in_input must be in [1, {len(IMAGE_KEYS)}]; "
                f"got {num_images}."
            )
        if not self.only_eval and num_images != 2:
            raise ValueError(
                "Native OpenPI pi0.5 PPO parity currently requires exactly two "
                f"real cameras; got num_images={num_images}."
            )
        config_name = str(openpi_cfg.get("config_name", ""))
        if not config_name:
            raise ValueError(
                "actor.model.openpi.config_name is required for PhyAI preprocessing."
            )
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
            f"kernel_config={kernel_cfg.config_path}."
        )
        # RLinf's worker-group launcher exports its own WORLD_SIZE/RANK. The
        # single-GPU PhyAI engine must resolve as an independent replica.
        launcher_values = {
            key: os.environ.pop(key, None)
            for key in ("WORLD_SIZE", "RANK", "MASTER_ADDR", "MASTER_PORT")
        }
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
                            params_dtype=self._engine_dtype,
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

    @staticmethod
    def _build_plugin_config(model_cfg: Any) -> PI05Config:
        """Map RLinf's PI0.5 model semantics to the PhyAI plugin config."""
        defaults = PI05Config()
        openpi_cfg = model_cfg.get("openpi", {})
        if not bool(model_cfg.get("pi05", True)):
            raise ValueError("The PhyAI pi05 plugin requires actor.model.pi05=True.")

        expected_variants = {
            "paligemma_variant": "gemma_2b",
            "action_expert_variant": "gemma_300m",
        }
        for field, expected in expected_variants.items():
            configured = str(openpi_cfg.get(field, expected))
            if configured != expected:
                raise ValueError(
                    f"The PhyAI pi05 plugin requires {field}={expected!r}; "
                    f"got {configured!r}."
                )

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
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        precision = {
            "bfloat16": "bf16",
            "float16": "fp16",
            "float32": "fp32",
        }.get(str(value).lower(), str(value))
        dtype = torch_dtype_from_precision(precision)
        if dtype is None:
            raise ValueError(f"Unsupported PhyAI vision_params_dtype: {value!r}.")
        return dtype

    @staticmethod
    def _actor_weight_name(name: str) -> str:
        """Map the native wrapper prefix to PhyAI HF parameter names."""
        return name[len("model.") :] if name.startswith("model.") else name

    # RLinf/OpenPI preprocessing remains local to this backend adapter.

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

    def _repack_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        states = self._select_configured_state(env_obs["states"])
        repacked = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self._openpi_config_name:
            repacked["observation/state_ee_pos"] = states[:, :3]
            repacked["observation/state_ee_rot"] = states[:, 3:6]
            repacked["observation/state_gripper"] = states[:, 6:7]
        else:
            repacked["observation/state"] = states
        if env_obs.get("wrist_images") is not None:
            repacked["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs.get("extra_view_images") is not None:
            repacked["observation/extra_view_image"] = env_obs["extra_view_images"]
        return repacked

    def _apply_input_transforms(self, observation: dict[str, Any]) -> dict[str, Any]:
        assert self._input_transform_fn is not None
        prompts = observation["prompt"]
        inputs = tree_map(
            lambda value: (
                np.asarray(value.detach().cpu()) if torch.is_tensor(value) else value
            ),
            {key: value for key, value in observation.items() if key != "prompt"},
        )
        batch_size = next(iter(inputs.values())).shape[0]
        if isinstance(prompts, np.ndarray):
            prompts = prompts.tolist()
        samples = []
        for index in range(batch_size):
            sample = tree_map(lambda value: value[index], inputs)
            sample["prompt"] = prompts[index]
            samples.append(sample)
        with ThreadPoolExecutor(max_workers=min(batch_size, 8)) as executor:
            transformed = list(executor.map(self._input_transform_fn, samples))
        return tree_map(
            lambda *values: torch.from_numpy(np.asarray(values).copy()),
            *transformed,
        )

    def _apply_output_transforms(
        self, actions: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        assert self._output_transform_fn is not None
        transformed = []
        for index in range(actions.shape[0]):
            transformed.append(
                self._output_transform_fn(
                    {
                        "actions": np.asarray(actions[index].detach().cpu()),
                        "state": np.asarray(state[index].detach().cpu()),
                    }
                )["actions"]
            )
        return torch.from_numpy(np.asarray(transformed).copy())

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
        assert self._engine_device is not None
        transformed = self._apply_input_transforms(self._repack_env_obs(env_obs))
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

    # RLinf decides the transition distribution; PhyAI executes it.

    def _build_rollout_plan(self, request: PI05Request) -> _PI05SamplingPlan:
        """Resolve RLinf policy choices into a sigma execution plan."""
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

        joint_logprob = bool(openpi_cfg.get("joint_logprob", False))
        ignore_last = bool(openpi_cfg.get("ignore_last", False))
        if joint_logprob:
            denoise_inds = (
                torch.arange(num_steps, dtype=torch.int64, device=self._engine_device)[
                    None
                ]
                .expand(batch_size, -1)
                .clone()
            )
        else:
            if ignore_last and num_steps < 2:
                raise ValueError(
                    "ignore_last=True requires at least two denoise steps."
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
        if noise_method not in ("flow_ode", "flow_sde"):
            raise ValueError(
                f"noise_method must be 'flow_ode' or 'flow_sde', got {noise_method!r}."
            )
        noise_level = float(openpi_cfg.get("noise_level", 0.0))
        if not math.isfinite(noise_level) or noise_level < 0:
            raise ValueError(
                f"noise_level must be finite and non-negative, got {noise_level}."
            )
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
            if joint_logprob:
                sigmas.copy_(schedule[None])
            else:
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
            joint_logprob=joint_logprob,
        )

    @staticmethod
    def _initial_logprob(sample: torch.Tensor) -> torch.Tensor:
        """Return RLinf's initial-noise contribution for joint policies."""
        return (
            -0.5 * torch.log(torch.full_like(sample, 2.0 * torch.pi))
            - 0.5 * sample.square()
        )

    # Raw PhyAI execution results are reduced into the Actor replay contract.

    @classmethod
    def _reduce_rollout_outputs(
        cls,
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
        if plan.joint_logprob:
            initial = cls._initial_logprob(
                chains[:, 0, : plan.action_chunk, : plan.action_dim]
            )
            prev_logprobs = torch.cat([initial[:, None], cropped], dim=1).mean(dim=1)
        else:
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

    # Evaluation uses action-only inference; training returns replay state.

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run inference or collect the real behavior-policy PPO state."""
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
        actions = self._apply_output_transforms(model_actions, processed.state)
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

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        """Compute the final-observation value for native GAE."""
        if final_obs is None or self.only_eval:
            return None
        with torch.no_grad():
            _, result = self.predict(final_obs, mode="train")
        values = result["prev_values"]
        if values is None:
            raise RuntimeError(
                "PhyAI training rollout did not return bootstrap values."
            )
        return values[:, :1].cpu().contiguous()

    # Actor weights are committed transactionally before their version advances.

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self) -> None:
        """Receive actor buckets and hot-update the in-process PhyAI engine."""
        if self._engine is None or self._weight_target is None:
            raise RuntimeError("init_worker() must be called before weight sync.")
        if self.weight_syncer is None:
            raise RuntimeError("PhyAI weight sync requires weight_syncer config.")
        if not isinstance(self.weight_syncer, BucketWeightSyncer):
            raise NotImplementedError(
                "PhyAI currently supports only bucket weight synchronization. "
                "Patch synchronization assumes identical sender/receiver state "
                "dict layouts, which is incompatible with PhyAI fused weights."
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
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        self.log_info(
            "PhyAI hot weight update applied: "
            f"version={self.version}, loaded={len(report.loaded)}."
        )
        gc.collect()
        self.torch_platform.empty_cache()

    def set_global_step(self, global_step: int) -> None:
        """PhyAI sampling has no global-step-dependent schedule."""
        del global_step

    def shutdown(self) -> None:
        """Release the PhyAI entry and its process-local distributed state."""
        if self._engine is None:
            return
        self.log_info(f"Shutting down PhyAI engine on rollout rank {self._rank}.")
        self._engine.close()
        self._engine = None
        self._weight_target = None
        self._input_transform_fn = None
        self._output_transform_fn = None
