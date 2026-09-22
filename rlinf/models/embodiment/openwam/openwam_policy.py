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

"""RLinf policy wrapper around OpenWAM's native deployment engine."""

from __future__ import annotations

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class OpenWAMPolicy(nn.Module, BasePolicy):
    """RLinf adapter for OpenWAM inference and native SFT training."""

    def __init__(
        self,
        engine: Any,
        *,
        num_frames: int,
        height: int,
        width: int,
        denoise_steps: int,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        inference_horizon: int | None = None,
    ):
        super().__init__()
        self.engine = engine
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.denoise_steps = denoise_steps
        self.lambda_video = float(lambda_video)
        self.lambda_action = float(lambda_action)
        # Receding-horizon eval: execute only the first ``inference_horizon``
        # actions of each generated chunk, as OpenWAM's deploy executor does.
        self.inference_horizon = (
            None if inference_horizon is None else int(inference_horizon)
        )
        if self.inference_horizon is not None and not (
            0 < self.inference_horizon <= max(1, num_frames - 1)
        ):
            raise ValueError(
                "OpenWAM inference_horizon must be in [1, num_frames - 1], got "
                f"{inference_horizon} for num_frames={num_frames}"
            )
        self.architecture = engine.architecture
        self._frozen_module_names: tuple[str, ...] = ()
        engine_cfg = getattr(engine, "cfg", None)
        dataloader_cfg = getattr(engine_cfg, "dataloader", None)
        self._multiview = bool(getattr(dataloader_cfg, "multiview", False))
        # RoboTwin-style readers wrap every instruction in a fixed sentence at
        # training time; LIBERO does not. Match the checkpoint's reader.
        self._prompt_template = _prompt_template_for_dataset(
            getattr(dataloader_cfg, "type", None)
        )
        # No per-block FSDP wrapping of the DiT / action blocks: OpenWAM's
        # joint denoising driver reads their weights directly
        # (pre_attn_at_layer) outside the blocks' forward, where a wrapped
        # block would still be sharded; they stay in the root unit. The
        # architecture itself cannot be an FSDP2 unit either (ABC-first
        # inheritance defeats fully_shard's class swap), so the SFT recipe
        # names the frozen VAE residual blocks as FSDP2 sub-units instead.
        # Checkpoints without an explicit layout (e.g. the LIBERO multiview
        # readers) use OpenWAM's three-slot canvas: head camera on top, two
        # wrist cameras below; missing slots stay black, as in the reader.
        layout = getattr(dataloader_cfg, "camera_layout", None)
        self._camera_layout = (
            list(layout)
            if layout is not None
            else ["head_camera", "left_camera", "right_camera"]
        )

    @classmethod
    def from_checkpoint(
        cls,
        model_path: str,
        *,
        ckpt_name: str | None,
        device: str,
        torch_dtype: torch.dtype | None,
        num_frames: int,
        height: int,
        width: int,
        denoise_steps: int,
        lambda_video: float = 1.0,
        lambda_action: float = 1.0,
        encoder_model_path: str | None = None,
        inference_horizon: int | None = None,
        runtime_dtype: torch.dtype | None = None,
    ) -> "OpenWAMPolicy":
        """Build OpenWAM's checkpoint loader and joint inference engine."""
        from omegaconf import OmegaConf
        from openwam.deploy import JointInferenceEngine, load_from_checkpoint_dir

        # Checkpoints produced on the training host may retain an absolute
        # ``model.video_backbone.encoder.model_path``.  Alternate encoders
        # (DINOv3, Flux VAE, V-JEPA) are not always copied into the deploy
        # bundle, so allow the eval config to repoint that one dependency while
        # keeping the checkpoint itself immutable.
        with _checkpoint_with_encoder_override(
            model_path, encoder_model_path
        ) as load_dir:
            cfg, architecture = load_from_checkpoint_dir(
                load_dir, device=device, ckpt_name=ckpt_name
            )
        if OmegaConf.select(cfg, "inference", default=None) is None:
            cfg.inference = {}
        cfg.inference.num_frames = num_frames
        cfg.inference.denoise_steps = denoise_steps
        cfg.inference.height = height
        cfg.inference.width = width
        # ``num_frames`` is the raw action/state window.  Wan sees only every
        # ``video_stride``-th frame, so preserve the native video contract when
        # the checkpoint does not already record an explicit inference value.
        if OmegaConf.select(cfg, "inference.video_num_frames", default=None) is None:
            dataloader_cfg = OmegaConf.select(cfg, "dataloader", default=None)
            if dataloader_cfg is not None:
                raw_frames = int(
                    OmegaConf.select(dataloader_cfg, "num_frames", default=num_frames)
                )
                video_stride = max(
                    1, int(OmegaConf.select(dataloader_cfg, "video_stride", default=1))
                )
                cfg.inference.video_num_frames = (raw_frames - 1) // video_stride + 1
        # Evaluation only consumes actions: skip the VAE decode of the generated
        # video. JointInferenceEngine reads this switch from the top-level
        # cfg.optimization at construction, not from the per-call condition.
        if OmegaConf.select(cfg, "optimization", default=None) is None:
            cfg.optimization = {}
        cfg.optimization.decode_video = False
        freeze_names = OmegaConf.select(cfg, "model.freeze", default=[]) or []
        frozen_module_names = architecture.freeze_modules(list(freeze_names)) or []
        training_cfg = OmegaConf.select(cfg, "training", default=OmegaConf.create({}))
        architecture.init_training_schedulers(1000)
        training_runtime = {
            "use_gradient_checkpointing": bool(
                OmegaConf.select(
                    training_cfg, "use_gradient_checkpointing", default=False
                )
            ),
            "use_gradient_checkpointing_offload": bool(
                OmegaConf.select(
                    training_cfg, "use_gradient_checkpointing_offload", default=False
                )
            ),
            "max_timestep_boundary": float(
                OmegaConf.select(training_cfg, "max_timestep_boundary", default=1.0)
            ),
            "min_timestep_boundary": float(
                OmegaConf.select(training_cfg, "min_timestep_boundary", default=0.0)
            ),
        }
        architecture.set_training_runtime(**training_runtime)
        model = cls(
            JointInferenceEngine(cfg=cfg, architecture=architecture),
            num_frames=num_frames,
            height=height,
            width=width,
            denoise_steps=denoise_steps,
            lambda_video=lambda_video,
            lambda_action=lambda_action,
            inference_horizon=inference_horizon,
        )
        param_dtype = torch_dtype or next(architecture.parameters()).dtype
        model.to(device=device, dtype=param_dtype)
        model._training_runtime = training_runtime
        model._frozen_module_names = tuple(frozen_module_names)
        # ``load_from_checkpoint_dir`` caches the checkpoint's training dtype as
        # the dtype OpenWAM builds noise, timesteps and proprio tensors in. Once
        # the parameters are recast that cache must follow them, unless FSDP
        # mixed precision names a separate compute dtype.
        model.set_runtime_dtype(runtime_dtype or param_dtype)
        return model

    def set_runtime_dtype(self, dtype: torch.dtype | None) -> None:
        """Set OpenWAM's input/runtime dtype without changing parameter storage.

        FSDP keeps fp32 master parameters while casting the forward parameters
        to its configured mixed-precision dtype. OpenWAM creates action and
        proprio tensors from its cached runtime dtype, so that cache must follow
        FSDP's compute dtype independently of the optimizer/master dtype.
        """
        if dtype is None:
            return
        owners = [self.architecture]
        backbones = getattr(self.architecture, "backbones", None)
        if backbones is not None:
            owners.extend(backbones.values())
        for owner in owners:
            if hasattr(owner, "_dtype"):
                owner._dtype = dtype

    def _restore_frozen_eval(self) -> None:
        """Keep native frozen subtrees in eval mode after a trainer ``train()``."""
        get_submodule = getattr(self.architecture, "get_submodule", None)
        if get_submodule is None:
            return
        for name in self._frozen_module_names:
            try:
                module = get_submodule(name)
            except (AttributeError, KeyError):
                continue
            for submodule in module.modules():
                submodule.training = False

    def train(self, mode: bool = True):
        """Set mode while preserving OpenWAM's frozen-module eval contract."""
        super().train(mode)
        if mode:
            self._restore_frozen_eval()
        return self

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """Route ``fsdp_config.gradient_checkpointing`` to OpenWAM's runtime flags.

        OpenWAM checkpoints its DiT blocks itself when
        ``use_gradient_checkpointing`` is set on the architecture; RLinf's
        ``use_reentrant`` choice does not apply and is ignored.
        """
        del gradient_checkpointing_kwargs
        runtime = dict(getattr(self, "_training_runtime", None) or {})
        runtime["use_gradient_checkpointing"] = True
        self.architecture.set_training_runtime(**runtime)
        self._training_runtime = runtime

    def retarget_runtime_device(self, device: torch.device | str) -> None:
        """Point OpenWAM's cached input device at ``device`` without moving weights.

        OpenWAM's architecture and video backbones remember the device they
        prepare inputs on (text token ids, frame tensors, noise). The FSDP SFT
        path builds the policy on the CPU and lets FSDP move parameters and
        buffers, so that cache is retargeted here; ``set_dtype_device`` is not
        used because it would move the whole model.
        """
        device = torch.device(device)
        owners = [self.architecture]
        owners.extend((getattr(self.architecture, "backbones", None) or {}).values())
        for owner in owners:
            if hasattr(owner, "_device"):
                owner._device = device

    def forward(self, forward_type: ForwardType = ForwardType.DEFAULT, **kwargs):
        if forward_type in (ForwardType.SFT, ForwardType.DEFAULT):
            return self.sft_forward(**kwargs)
        raise NotImplementedError(
            f"OpenWAMPolicy supports SFT and evaluation only, got {forward_type!r}"
        )

    def default_forward(self, **kwargs):
        # SFT is the only training path of this adapter.
        return self.sft_forward(**kwargs)

    def sft_forward(self, data: Any = None, **kwargs) -> dict[str, torch.Tensor]:
        """Compute OpenWAM's native joint video/action SFT loss."""
        if data is None:
            data = kwargs.get("batch")
        if data is None:
            raise ValueError(
                "OpenWAM sft_forward requires `data` from the SFT dataloader."
            )
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, (list, tuple)):
            raise TypeError(
                f"OpenWAM SFT data must be a sample list, got {type(data)!r}"
            )
        inputs = self.architecture.prepare_inputs(list(data))
        result = self.architecture.compute_loss(
            **inputs,
            lambda_video=float(kwargs.get("lambda_video", self.lambda_video)),
            lambda_action=float(kwargs.get("lambda_action", self.lambda_action)),
        )
        return {
            "loss": result["loss"],
            "loss_video": result.get("loss_video", result["loss"].detach()),
            "loss_action": result.get("loss_action", result["loss"].detach()),
        }

    def _format_prompt(self, prompt: Any) -> str:
        """Apply the checkpoint reader's instruction template, if it has one."""
        text = "" if prompt is None else str(prompt)
        if self._prompt_template is None:
            return text
        return self._prompt_template + text

    def predict_action_batch(
        self, env_obs: dict[str, Any], mode: str = "eval", **kwargs
    ):
        """Generate one action chunk per observation in ``env_obs``.

        OpenWAM's deploy engine generates one condition at a time, so this
        runs ``len(env_obs)`` sequential generations; evaluation cost grows
        linearly with ``total_num_envs``.
        """
        if mode == "train":
            raise NotImplementedError(
                "OpenWAMPolicy only supports evaluation rollouts (mode='eval'); "
                "RL rollouts are not part of this adapter."
            )
        if mode != "eval":
            raise ValueError(f"Unknown OpenWAM predict mode: {mode!r}")
        batch_size = _infer_batch_size(env_obs)
        actions, results = [], []
        for index in range(batch_size):
            main_image = _to_pil(
                _batch_value(env_obs, ("main_images", "image", "images"), index)
            )
            wrist_image = _batch_value(
                env_obs, ("wrist_images", "wrist_image"), index, None
            )
            image = _compose_observation_image(
                main_image,
                wrist_image,
                multiview=self._multiview,
                camera_layout=self._camera_layout,
                height=self.height,
                width=self.width,
            )
            condition = {
                "prompt": self._format_prompt(
                    _batch_value(
                        env_obs,
                        ("task_descriptions", "task_description", "language"),
                        index,
                        "",
                    )
                ),
                "first_frame_image": [image],
                "proprio": _observation_proprio(env_obs, index),
                "num_frames": self.num_frames,
                "height": self.height,
                "width": self.width,
                "denoise_steps": self.denoise_steps,
            }
            video_num_frames = getattr(
                getattr(getattr(self.engine, "cfg", None), "inference", None),
                "video_num_frames",
                None,
            )
            if video_num_frames is not None:
                # Omitting the key preserves the engine's native fallback;
                # passing ``None`` would reach ``int(None)`` in JointInferenceEngine.
                condition["video_num_frames"] = video_num_frames
            result = self.engine.generate(condition)
            actions.append(np.asarray(result["actions"]))
            results.append(result)
        stacked = torch.as_tensor(np.stack(actions), dtype=torch.float32)
        if self.inference_horizon is not None:
            stacked = stacked[:, : self.inference_horizon]
        return stacked, {"results": results}


@contextmanager
def _checkpoint_with_encoder_override(
    model_path: str, encoder_model_path: str | None
) -> Iterator[str]:
    """Yield a checkpoint directory with deploy-time encoder fixes applied.

    OpenWAM checkpoints can retain training-host paths and legacy encoder names
    (``vjepa2_1``, ``flux_vae``, ``wan_vae``). A temporary staging directory
    rewrites only those config fields without editing the checkpoint or source
    checkout. All other files are symlinked, so this adds no model-storage cost.
    """
    checkpoint_dir = Path(model_path).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(
            f"OpenWAM checkpoint directory not found: {checkpoint_dir}"
        )

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    encoder_cfg = OmegaConf.select(cfg, "model.video_backbone.encoder", default=None)
    encoder_name = (
        str(getattr(encoder_cfg, "name", "")) if encoder_cfg is not None else ""
    )
    name_aliases = {
        "vjepa2_1": "vjepa21",
        "flux_vae": "flux2_vae",
        "wan_vae": "wan22_vae",
    }
    canonical_name = name_aliases.get(encoder_name, encoder_name)
    needs_staging = encoder_model_path is not None or canonical_name != encoder_name
    if not needs_staging:
        yield model_path
        return
    if encoder_cfg is None:
        raise ValueError(
            "OpenWAM encoder override/alias normalization requires "
            "model.video_backbone.encoder in the checkpoint config"
        )
    if encoder_model_path is not None:
        encoder_dir = Path(encoder_model_path).expanduser().resolve()
        if not encoder_dir.is_dir():
            raise FileNotFoundError(
                f"OpenWAM encoder_model_path is not a directory: {encoder_dir}"
            )
        encoder_cfg.model_path = str(encoder_dir)
    if canonical_name != encoder_name:
        encoder_cfg.name = canonical_name

    with tempfile.TemporaryDirectory(prefix="rlinf-openwam-") as tmp:
        staged = Path(tmp) / checkpoint_dir.name
        staged.mkdir()
        for entry in checkpoint_dir.iterdir():
            if entry.name != "config.yaml":
                os.symlink(
                    entry, staged / entry.name, target_is_directory=entry.is_dir()
                )
        OmegaConf.save(cfg, staged / "config.yaml")
        yield str(staged)


def _infer_batch_size(obs: dict[str, Any]) -> int:
    for key in ("states", "state", "proprio", "main_images", "image", "images"):
        if key in obs:
            value = obs[key]
            if isinstance(value, (torch.Tensor, np.ndarray)):
                return int(value.shape[0]) if value.ndim > 1 else 1
            if isinstance(value, (list, tuple)):
                return len(value)
    return 1


def _batch_value(
    obs: dict[str, Any], keys: tuple[str, ...], index: int, default: Any = None
) -> Any:
    value = next((obs[key] for key in keys if key in obs), default)
    if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 1:
        return value[index]
    if isinstance(value, (list, tuple)) and value and len(value) > index:
        return value[index]
    return value


def _to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, (list, tuple)):
        value = value[0]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    if array.dtype != np.uint8:
        array = np.clip(array * 255 if array.max() <= 1.0 else array, 0, 255).astype(
            np.uint8
        )
    return Image.fromarray(array)


def _axis_angle_to_rotation_6d(value: Any) -> np.ndarray:
    """Convert a LIBERO axis-angle state to OpenWAM's 6-D representation."""
    from scipy.spatial.transform import Rotation

    matrix = Rotation.from_rotvec(
        np.asarray(value, dtype=np.float64).reshape(3)
    ).as_matrix()
    return np.concatenate([matrix[:, 0], matrix[:, 1]]).astype(np.float32)


def _libero_state_to_eef10(value: Any) -> np.ndarray | None:
    """Convert RLinf's 8-D LIBERO state to OpenWAM's achieved EEF10 state."""
    if value is None:
        return None
    state = np.asarray(value, dtype=np.float32).reshape(-1)
    if state.shape[0] == 10:
        return state
    if state.shape[0] != 8:
        raise ValueError(
            "OpenWAM LIBERO rollout expects an 8-D state "
            f"(xyz, axis-angle, gripper qpos), got {state.shape[0]}"
        )
    gripper_width = float(state[6] - state[7])
    gripper_open_scale = np.clip(2.0 * gripper_width / 0.08 - 1.0, -1.0, 1.0)
    return np.concatenate(
        [state[:3], _axis_angle_to_rotation_6d(state[3:6]), [gripper_open_scale]]
    ).astype(np.float32)


ROBOTWIN_PROMPT_PREFIX = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: "
)
_TEMPLATED_DATASET_TYPES = ("robotwin", "robodojo", "ebench")


def _prompt_template_for_dataset(dataset_type: Any) -> str | None:
    """Return the training-time instruction prefix of an OpenWAM dataset reader."""
    if dataset_type is None or str(dataset_type) not in _TEMPLATED_DATASET_TYPES:
        return None
    try:
        from openwam.dataloader.transforms.multiview import (
            format_prompt_for_inference,
        )

        return format_prompt_for_inference("")
    except ImportError:
        return ROBOTWIN_PROMPT_PREFIX


def _wrist_frames(value: Any) -> list[Image.Image]:
    """Split a wrist observation into per-camera PIL frames, in layout order.

    RLinf environments hand back either one wrist image (``[H, W, 3]``) or a
    stack of them (``[n, H, W, 3]``; RoboTwin stacks left then right).
    """
    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)):
        return [_to_pil(item) for item in value if item is not None]
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        return [_to_pil(frame) for frame in array]
    return [_to_pil(array)]


def _compose_observation_image(
    main_image: Image.Image,
    wrist_image: Any,
    *,
    multiview: bool,
    camera_layout: list[str],
    height: int,
    width: int,
) -> Image.Image:
    """Match OpenWAM's single-view or three-camera training layout.

    Single-view checkpoints see the head camera center-cropped and resized to
    the training canvas, as ``openwam.deploy.obs_preprocess`` does. Multiview
    checkpoints get the head camera in slot 0 and the wrist cameras in the
    following slots of ``camera_layout``; missing cameras stay black, exactly
    like the dataset reader pads them.
    """
    from openwam.dataloader.transforms.multiview import assemble_multiview_layout

    if not multiview:
        if main_image.size == (width, height):
            return main_image
        # Native single-view readers decode directly to (width, height); do not
        # center-crop a square environment frame and discard its vertical view.
        return main_image.resize((width, height), Image.Resampling.LANCZOS)

    frames = {camera_layout[0]: main_image}
    for slot, frame in zip(camera_layout[1:], _wrist_frames(wrist_image)):
        frames[slot] = frame
    return assemble_multiview_layout(
        frames,
        camera_layout=camera_layout,
        out_h=height,
        out_w=width,
    )


def _observation_proprio(env_obs: dict[str, Any], index: int) -> np.ndarray | None:
    """Use explicit native physical units, or convert LIBERO's achieved pose."""
    value = _batch_value(env_obs, ("native_proprio",), index, None)
    if value is not None:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim != 1 or not np.isfinite(value).all():
            raise ValueError(
                "native_proprio must be a finite vector in checkpoint physical units"
            )
        return value
    return _libero_state_to_eef10(
        _batch_value(env_obs, ("states", "state", "proprio"), index, None)
    )
