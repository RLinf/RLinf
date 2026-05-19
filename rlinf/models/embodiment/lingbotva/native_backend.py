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

"""In-process official LingBot-VA runtime backend."""

from __future__ import annotations

import atexit
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from rlinf.utils.logging import get_logger

logger = get_logger()

_TRANSFORMER_STATE_PREFIXES = (
    "_sft_core.transformer.",
    "transformer.",
)


def _extend_import_path(repo_path: Path) -> None:
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def _extract_transformer_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    for prefix in _TRANSFORMER_STATE_PREFIXES:
        extracted = {
            key[len(prefix) :]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if extracted:
            return extracted
    return state_dict


def _resolve_cuda_local_rank() -> int:
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None:
        try:
            return int(local_rank)
        except ValueError as exc:
            raise ValueError(f"Invalid LOCAL_RANK value: {local_rank!r}") from exc
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    return 0


def _resolve_runtime_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    repo_root = os.environ.get("REPO_PATH")
    if repo_root:
        return (Path(repo_root) / path).resolve()
    return path.resolve()


class LingbotVANativeBackend:
    """Single-process backend using the official LingBot-VA modules directly."""

    _BATCH_CACHE_NAME = "rlinf_batch"

    def __init__(self, cfg: Any, torch_dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.torch_dtype = torch_dtype
        self.config_name = getattr(cfg.lingbotva, "config_name", "robotwin")
        self.repo_path = Path(getattr(cfg.lingbotva, "repo_path"))
        self.model_path = Path(cfg.model_path)
        transformer_state_dict_path = getattr(
            cfg.lingbotva, "transformer_state_dict_path", None
        )
        self.transformer_state_dict_path = (
            Path(transformer_state_dict_path)
            if transformer_state_dict_path is not None
            else None
        )
        self.save_root = _resolve_runtime_path(
            getattr(cfg.lingbotva, "save_root", "./runtime/lingbotva")
        )
        self._get_mesh_id = None
        self._data_seq_to_patch = None
        self._validate_runtime_paths()
        self._validate_attn_mode(self.model_path)
        self._server = self._build_server()
        atexit.register(self.close)

    def _validate_runtime_paths(self) -> None:
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA repo path does not exist: {self.repo_path}"
            )
        if not self.repo_path.is_dir():
            raise NotADirectoryError(
                f"LingBot-VA repo path is not a directory: {self.repo_path}"
            )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA model path does not exist: {self.model_path}"
            )
        if not self.model_path.is_dir():
            raise NotADirectoryError(
                f"LingBot-VA model path is not a directory: {self.model_path}"
            )

    @staticmethod
    def _validate_attn_mode(model_path: Path) -> None:
        config_path = model_path / "transformer" / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"LingBot-VA transformer config not found at {config_path}."
            )
        config_data = json.loads(config_path.read_text(encoding="utf-8"))
        attn_mode = config_data.get("attn_mode")
        if attn_mode not in {"torch", "flashattn"}:
            raise ValueError(
                "LingBot-VA inference requires transformer/config.json attn_mode "
                f'to be "torch" or "flashattn", but got {attn_mode!r}.'
            )

    def _build_server(self):
        _extend_import_path(self.repo_path)

        from wan_va.configs import VA_CONFIGS
        from wan_va.utils import data_seq_to_patch, get_mesh_id
        from wan_va.wan_va_server import VA_Server

        job_config = copy.deepcopy(VA_CONFIGS[self.config_name])
        job_config.wan22_pretrained_model_name_or_path = str(self.model_path)
        job_config.param_dtype = self.torch_dtype
        job_config.enable_offload = bool(
            getattr(self.cfg.lingbotva, "enable_offload", False)
        )
        job_config.save_root = str(self.save_root)

        current_device = _resolve_cuda_local_rank()
        if torch.cuda.is_available():
            torch.cuda.set_device(current_device)
        job_config.rank = 0
        job_config.local_rank = current_device
        job_config.world_size = 1
        server = VA_Server(job_config)
        self._data_seq_to_patch = data_seq_to_patch
        self._get_mesh_id = get_mesh_id
        if self.transformer_state_dict_path is not None:
            self._load_transformer_state_dict(server.transformer)
        logger.info(
            "Initialized in-process LingBot-VA runtime on device %s "
            "(CUDA_VISIBLE_DEVICES=%s, LOCAL_ACCELERATOR_RANK=%s).",
            current_device,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            os.environ.get("LOCAL_ACCELERATOR_RANK"),
        )
        return server

    def _load_transformer_state_dict(self, transformer: torch.nn.Module) -> None:
        if self.transformer_state_dict_path is None:
            return
        if not self.transformer_state_dict_path.exists():
            raise FileNotFoundError(
                "LingBot-VA transformer state dict path does not exist: "
                f"{self.transformer_state_dict_path}"
            )

        checkpoint_path = self.transformer_state_dict_path
        if checkpoint_path.is_dir():
            transformer_dir = (
                checkpoint_path / "transformer"
                if (checkpoint_path / "transformer" / "config.json").exists()
                else checkpoint_path
            )
            state_path = transformer_dir / "diffusion_pytorch_model.safetensors"
            if not state_path.exists():
                raise FileNotFoundError(
                    "LingBot-VA transformer checkpoint directory must contain "
                    f"diffusion_pytorch_model.safetensors: {transformer_dir}"
                )
            transformer_state = load_file(str(state_path), device="cpu")
        else:
            raw_state = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
            if not isinstance(raw_state, dict):
                raise TypeError(
                    "LingBot-VA transformer state dict must deserialize to a dict, got "
                    f"{type(raw_state)!r}."
                )
            transformer_state = _extract_transformer_state_dict(raw_state)
        missing_keys, unexpected_keys = transformer.load_state_dict(
            transformer_state, strict=False
        )
        if missing_keys or unexpected_keys:
            preview_missing = ", ".join(missing_keys[:5])
            preview_unexpected = ", ".join(unexpected_keys[:5])
            raise RuntimeError(
                "LingBot-VA transformer checkpoint does not match the runtime "
                f"model (missing={len(missing_keys)}, unexpected={len(unexpected_keys)}). "
                f"Missing preview: [{preview_missing}] "
                f"Unexpected preview: [{preview_unexpected}]"
            )
        logger.info(
            "Loaded LingBot-VA transformer checkpoint from %s (missing=%d, unexpected=%d).",
            self.transformer_state_dict_path,
            len(missing_keys),
            len(unexpected_keys),
        )

    def _cfg_batch_size(self, batch_size: int) -> int:
        return batch_size * (2 if self._server.use_cfg else 1)

    def _ensure_observation_runtime_shape(self) -> None:
        server = self._server
        if all(
            hasattr(server, attr)
            for attr in ("height", "width", "latent_height", "latent_width")
        ):
            return

        server.action_per_frame = server.job_config.action_per_frame
        server.height, server.width = server.job_config.height, server.job_config.width
        if server.env_type == "robotwin_tshape":
            server.latent_height, server.latent_width = (
                ((server.height // 16) * 3) // 2,
                server.width // 16,
            )
        else:
            server.latent_height, server.latent_width = (
                server.height // 16,
                server.width // 16 * len(server.job_config.obs_cam_keys),
            )

    def _reset_batch_runtime(self, prompts: list[str]) -> None:
        if not prompts:
            raise ValueError("LingBot-VA batch runtime requires at least one prompt.")

        server = self._server
        batch_size = len(prompts)
        server.cache_name = self._BATCH_CACHE_NAME
        server.use_cfg = (server.job_config.guidance_scale > 1) or (
            server.job_config.action_guidance_scale > 1
        )
        server.frame_st_id = 0
        server.init_latent = None
        server.transformer.clear_cache(server.cache_name)
        server.streaming_vae.clear_cache()

        self._ensure_observation_runtime_shape()

        if server.env_type == "robotwin_tshape":
            server.streaming_vae_half.clear_cache()

        patch_size = server.job_config.patch_size
        latent_token_per_chunk = (
            server.job_config.frame_chunk_size
            * server.latent_height
            * server.latent_width
        ) // (patch_size[0] * patch_size[1] * patch_size[2])
        action_token_per_chunk = (
            server.job_config.frame_chunk_size * server.action_per_frame
        )
        server.transformer.create_empty_cache(
            server.cache_name,
            server.job_config.attn_window,
            latent_token_per_chunk,
            action_token_per_chunk,
            dtype=server.dtype,
            device=server.device,
            batch_size=self._cfg_batch_size(batch_size),
        )

        server.action_mask = torch.zeros([server.job_config.action_dim]).bool()
        server.action_mask[server.job_config.used_action_channel_ids] = True
        server.actions_q01 = torch.tensor(
            server.job_config.norm_stat["q01"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        server.actions_q99 = torch.tensor(
            server.job_config.norm_stat["q99"], dtype=torch.float32
        ).reshape(-1, 1, 1)
        server.action_norm_method = server.job_config.action_norm_method
        server.prompt_embeds, server.negative_prompt_embeds = server.encode_prompt(
            prompt=prompts,
            negative_prompt=None,
            do_classifier_free_guidance=server.job_config.guidance_scale > 1,
            num_videos_per_prompt=1,
            prompt_embeds=None,
            negative_prompt_embeds=None,
            max_sequence_length=512,
            device=server.device,
            dtype=server.dtype,
        )
        server.exp_name = "rlinf_batch"
        server.exp_save_root = str(self.save_root / "real")
        os.makedirs(server.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    @staticmethod
    def _normalize_obs_sequences(
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> list[list[dict[str, Any]]]:
        if not obs_batch:
            raise ValueError("LingBot-VA batch infer requires non-empty observations.")
        if isinstance(obs_batch[0], dict):
            return [[obs] for obs in obs_batch]
        return obs_batch  # type: ignore[return-value]

    @staticmethod
    def _validate_uniform_sequence_lengths(
        obs_sequences: list[list[dict[str, Any]]],
        *,
        context: str,
    ) -> None:
        lengths = [len(sequence) for sequence in obs_sequences]
        if not lengths:
            raise ValueError(f"{context} requires non-empty observation sequences.")
        if len(set(lengths)) != 1:
            raise ValueError(
                f"{context} requires uniform sequence lengths across the batch, got {lengths}."
            )

    def _encode_obs_batch(
        self,
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
    ) -> torch.Tensor | None:
        server = self._server
        self._ensure_observation_runtime_shape()
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        if not obs_sequences:
            return None
        self._validate_uniform_sequence_lengths(
            obs_sequences,
            context="LingBot-VA batched observation encoding",
        )

        videos = []
        for camera_index, camera_key in enumerate(server.job_config.obs_cam_keys):
            if server.env_type == "robotwin_tshape":
                if camera_index == 0:
                    height_i, width_i = server.height, server.width
                else:
                    height_i, width_i = server.height // 2, server.width // 2
            else:
                height_i, width_i = server.height, server.width

            history_video = torch.stack(
                [
                    torch.from_numpy(np.stack([frame[camera_key] for frame in seq]))
                    .float()
                    .permute(3, 0, 1, 2)
                    for seq in obs_sequences
                ],
                dim=0,
            )
            history_video = F.interpolate(
                history_video.flatten(0, 1),
                size=(height_i, width_i),
                mode="bilinear",
                align_corners=False,
            ).unflatten(0, (history_video.shape[0], history_video.shape[1]))
            videos.append(history_video)

        if server.env_type == "robotwin_tshape":
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:], dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(server.streaming_vae.vae.parameters()).device
            enc_out_high = server.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(server.dtype)
            )
            enc_out_left_and_right = server.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(server.dtype)
            )
            left_enc, right_enc = enc_out_left_and_right.split(
                videos_high.shape[0], dim=0
            )
            enc_out = torch.cat(
                [
                    torch.cat([left_enc, right_enc], dim=-1),
                    enc_out_high,
                ],
                dim=-2,
            )
        else:
            videos_tensor = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(server.streaming_vae.vae.parameters()).device
            enc_out = server.streaming_vae.encode_chunk(
                videos_tensor.to(vae_device).to(server.dtype)
            )

        mu, _logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(server.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(server.vae.config.latents_std).to(mu.device)
        mu_norm = server.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        if server.env_type == "robotwin_tshape":
            video_latent = mu_norm.to(server.device)
        else:
            video_latent = torch.cat(
                mu_norm.split(len(obs_sequences), dim=0), dim=-1
            ).to(server.device)
        return video_latent

    def _preprocess_action_batch(self, state_batch: np.ndarray) -> torch.Tensor:
        server = self._server
        tensors = [
            server.preprocess_action(np.asarray(state, dtype=np.float32))
            for state in state_batch
        ]
        return torch.cat(tensors, dim=0)

    def _prepare_batch_input(
        self,
        *,
        latent_model_input: torch.Tensor | None,
        action_model_input: torch.Tensor | None,
        latent_t: float = 0,
        action_t: float = 0,
        latent_cond: torch.Tensor | None = None,
        action_cond: torch.Tensor | None = None,
        frame_st_id: int = 0,
    ) -> dict[str, dict[str, torch.Tensor]]:
        server = self._server
        batch_size = (
            latent_model_input.shape[0]
            if latent_model_input is not None
            else action_model_input.shape[0]
        )
        input_dict: dict[str, dict[str, torch.Tensor]] = {}

        if latent_model_input is not None:
            timesteps = (
                torch.ones(
                    [latent_model_input.shape[2]],
                    dtype=torch.float32,
                    device=server.device,
                )
                * latent_t
            )
            if self._get_mesh_id is None:
                raise RuntimeError(
                    "LingBot-VA batch runtime missing get_mesh_id helper."
                )
            grid_id = self._get_mesh_id(
                latent_model_input.shape[-3] // server.job_config.patch_size[0],
                latent_model_input.shape[-2] // server.job_config.patch_size[1],
                latent_model_input.shape[-1] // server.job_config.patch_size[2],
                0,
                1,
                frame_st_id,
            ).to(server.device)
            noisy_latents = latent_model_input.clone()
            if latent_cond is not None:
                noisy_latents[:, :, 0:1] = latent_cond[:, :, 0:1]
                timesteps[0:1] *= 0
            input_dict["latent_res_lst"] = {
                "noisy_latents": noisy_latents,
                "timesteps": timesteps,
                "grid_id": grid_id,
                "text_emb": server.prompt_embeds.to(server.dtype).clone(),
            }

        if action_model_input is not None:
            timesteps = (
                torch.ones(
                    [action_model_input.shape[2]],
                    dtype=torch.float32,
                    device=server.device,
                )
                * action_t
            )
            if self._get_mesh_id is None:
                raise RuntimeError(
                    "LingBot-VA batch runtime missing get_mesh_id helper."
                )
            grid_id = self._get_mesh_id(
                action_model_input.shape[-3],
                action_model_input.shape[-2],
                action_model_input.shape[-1],
                1,
                1,
                frame_st_id,
                action=True,
            ).to(server.device)
            noisy_actions = action_model_input.clone()
            if action_cond is not None:
                noisy_actions[:, :, 0:1] = action_cond[:, :, 0:1]
                timesteps[0:1] *= 0
            noisy_actions[:, ~server.action_mask] *= 0
            input_dict["action_res_lst"] = {
                "noisy_latents": noisy_actions,
                "timesteps": timesteps,
                "grid_id": grid_id,
                "text_emb": server.prompt_embeds.to(server.dtype).clone(),
            }

        for input_value in input_dict.values():
            if server.use_cfg:
                input_value["noisy_latents"] = input_value["noisy_latents"].repeat(
                    2, 1, 1, 1, 1
                )
                input_value["text_emb"] = torch.cat(
                    [
                        server.prompt_embeds.to(server.dtype).clone(),
                        server.negative_prompt_embeds.to(server.dtype).clone(),
                    ],
                    dim=0,
                )
                input_value["grid_id"] = input_value["grid_id"][None].repeat(
                    self._cfg_batch_size(batch_size), 1, 1
                )
                input_value["timesteps"] = input_value["timesteps"][None].repeat(
                    self._cfg_batch_size(batch_size), 1
                )
            else:
                input_value["grid_id"] = input_value["grid_id"][None].repeat(
                    batch_size, 1, 1
                )
                input_value["timesteps"] = input_value["timesteps"][None].repeat(
                    batch_size, 1
                )
        return input_dict

    def _postprocess_action_batch(self, action: torch.Tensor) -> list[np.ndarray]:
        server = self._server
        action = action.detach().cpu()[..., 0]
        if server.action_norm_method == "quantiles":
            action = (action + 1) / 2 * (
                server.actions_q99 - server.actions_q01 + 1e-6
            ) + server.actions_q01
        else:
            raise NotImplementedError
        action_np = action.numpy()
        used = action_np[:, server.job_config.used_action_channel_ids]
        return [used[idx].astype(np.float32) for idx in range(used.shape[0])]

    def _infer_batch_impl(
        self,
        obs_batch: list[dict[str, Any]] | list[list[dict[str, Any]]],
        *,
        frame_st_id: int = 0,
    ) -> list[np.ndarray]:
        server = self._server
        obs_sequences = self._normalize_obs_sequences(obs_batch)
        batch_size = len(obs_sequences)

        if frame_st_id == 0 and server.init_latent is None:
            server.init_latent = self._encode_obs_batch(obs_sequences)

        latents = torch.randn(
            batch_size,
            48,
            server.job_config.frame_chunk_size,
            server.latent_height,
            server.latent_width,
            device=server.device,
            dtype=server.dtype,
        )
        actions = torch.randn(
            batch_size,
            server.job_config.action_dim,
            server.job_config.frame_chunk_size,
            server.action_per_frame,
            1,
            device=server.device,
            dtype=server.dtype,
        )

        server.scheduler.set_timesteps(server.job_config.num_inference_steps)
        server.action_scheduler.set_timesteps(
            server.job_config.action_num_inference_steps
        )
        timesteps = torch.nn.functional.pad(
            server.scheduler.timesteps, (0, 1), mode="constant", value=0
        )
        if server.job_config.video_exec_step != -1:
            timesteps = timesteps[: server.job_config.video_exec_step]
        action_timesteps = torch.nn.functional.pad(
            server.action_scheduler.timesteps, (0, 1), mode="constant", value=0
        )

        with torch.no_grad():
            for step_idx, timestep in enumerate(timesteps):
                last_step = step_idx == len(timesteps) - 1
                latent_cond = (
                    server.init_latent[:, :, 0:1] if frame_st_id == 0 else None
                )
                input_dict = self._prepare_batch_input(
                    latent_model_input=latents,
                    action_model_input=None,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=latent_cond,
                    action_cond=None,
                    frame_st_id=frame_st_id,
                )
                video_noise_pred = server.transformer(
                    input_dict["latent_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=False,
                )
                if not last_step or server.job_config.video_exec_step != -1:
                    if self._data_seq_to_patch is None:
                        raise RuntimeError(
                            "LingBot-VA batch runtime missing data_seq_to_patch helper."
                        )
                    video_noise_pred = self._data_seq_to_patch(
                        server.job_config.patch_size,
                        video_noise_pred,
                        server.job_config.frame_chunk_size,
                        server.latent_height,
                        server.latent_width,
                        batch_size=self._cfg_batch_size(batch_size),
                    )
                    if server.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[
                            batch_size:
                        ] + server.job_config.guidance_scale * (
                            video_noise_pred[:batch_size]
                            - video_noise_pred[batch_size:]
                        )
                    else:
                        video_noise_pred = video_noise_pred[:batch_size]
                    latents = server.scheduler.step(
                        video_noise_pred, timestep, latents, return_dict=False
                    )
                if latent_cond is not None:
                    latents[:, :, 0:1] = latent_cond

            for step_idx, timestep in enumerate(action_timesteps):
                last_step = step_idx == len(action_timesteps) - 1
                action_cond = (
                    torch.zeros(
                        [
                            batch_size,
                            server.job_config.action_dim,
                            1,
                            server.action_per_frame,
                            1,
                        ],
                        device=server.device,
                        dtype=server.dtype,
                    )
                    if frame_st_id == 0
                    else None
                )
                input_dict = self._prepare_batch_input(
                    latent_model_input=None,
                    action_model_input=actions,
                    latent_t=float(timestep),
                    action_t=float(timestep),
                    latent_cond=None,
                    action_cond=action_cond,
                    frame_st_id=frame_st_id,
                )
                action_noise_pred = server.transformer(
                    input_dict["action_res_lst"],
                    update_cache=1 if last_step else 0,
                    cache_name=server.cache_name,
                    action_mode=True,
                )
                if not last_step:
                    action_noise_pred = (
                        action_noise_pred.unflatten(
                            1,
                            (
                                server.job_config.frame_chunk_size,
                                server.action_per_frame,
                            ),
                        )
                        .permute(0, 3, 1, 2)
                        .unsqueeze(-1)
                    )
                    if server.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[
                            batch_size:
                        ] + server.job_config.action_guidance_scale * (
                            action_noise_pred[:batch_size]
                            - action_noise_pred[batch_size:]
                        )
                    else:
                        action_noise_pred = action_noise_pred[:batch_size]
                    actions = server.action_scheduler.step(
                        action_noise_pred, timestep, actions, return_dict=False
                    )
                if action_cond is not None:
                    actions[:, :, 0:1] = action_cond

        actions[:, ~server.action_mask] *= 0
        torch.cuda.empty_cache()
        return self._postprocess_action_batch(actions)

    def _compute_kv_cache_batch_impl(
        self,
        *,
        obs_batch: list[list[dict[str, Any]]],
        state_batch: np.ndarray,
        frame_st_id: int,
    ) -> int:
        server = self._server
        server.transformer.clear_pred_cache(server.cache_name)
        latent_model_input = self._encode_obs_batch(obs_batch)
        if frame_st_id == 0:
            latent_model_input = (
                torch.cat([server.init_latent, latent_model_input], dim=2)
                if latent_model_input is not None
                else server.init_latent
            )

        action_model_input = self._preprocess_action_batch(state_batch).to(
            latent_model_input
        )
        input_dict = self._prepare_batch_input(
            latent_model_input=latent_model_input,
            action_model_input=action_model_input,
            frame_st_id=frame_st_id,
        )

        with torch.no_grad():
            server.transformer(
                input_dict["latent_res_lst"],
                update_cache=2,
                cache_name=server.cache_name,
                action_mode=False,
            )
            server.transformer(
                input_dict["action_res_lst"],
                update_cache=2,
                cache_name=server.cache_name,
                action_mode=True,
            )
        torch.cuda.empty_cache()
        return frame_st_id + int(latent_model_input.shape[2])

    def infer_batch(
        self,
        obs_batch: list[dict[str, Any]],
        prompts: list[str],
        *,
        kv_cache_histories: list[list[tuple[list[dict[str, Any]], np.ndarray]]]
        | None = None,
    ) -> list[np.ndarray]:
        if len(obs_batch) != len(prompts):
            raise ValueError(
                f"LingBot-VA batch infer expects equal numbers of obs and prompts, got {len(obs_batch)} and {len(prompts)}."
            )
        self._reset_batch_runtime(prompts)
        frame_st_id = 0
        if kv_cache_histories:
            self._server.init_latent = self._encode_obs_batch(obs_batch)
            history_len = len(kv_cache_histories[0])
            if not all(len(history) == history_len for history in kv_cache_histories):
                raise ValueError(
                    "LingBot-VA batched follow-up infer requires groups with equal kv history lengths."
                )
            for history_idx in range(history_len):
                obs_sequences = [
                    history[history_idx][0] for history in kv_cache_histories
                ]
                self._validate_uniform_sequence_lengths(
                    obs_sequences,
                    context=(
                        f"LingBot-VA batched follow-up infer replay step {history_idx}"
                    ),
                )
                state_batch = np.stack(
                    [history[history_idx][1] for history in kv_cache_histories], axis=0
                )
                try:
                    frame_st_id = self._compute_kv_cache_batch_impl(
                        obs_batch=obs_sequences,
                        state_batch=state_batch,
                        frame_st_id=frame_st_id,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "LingBot-VA batched follow-up kv-cache replay failed at "
                        f"history_idx={history_idx}, "
                        f"obs_lengths={[len(seq) for seq in obs_sequences]}, "
                        f"state_batch_shape={tuple(state_batch.shape)}, "
                        f"frame_st_id={frame_st_id}."
                    ) from exc
        else:
            return self._infer_batch_impl(obs_batch, frame_st_id=frame_st_id)
        return self._infer_batch_impl(obs_batch, frame_st_id=frame_st_id)

    def close(self) -> None:
        if getattr(self, "_server", None) is None:
            return
        transformer = getattr(self._server, "transformer", None)
        if transformer is not None:
            try:
                transformer.clear_cache(self._BATCH_CACHE_NAME)
            except Exception:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
