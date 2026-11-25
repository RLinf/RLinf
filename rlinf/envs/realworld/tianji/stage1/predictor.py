# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen Stage1 GPU inference; no Stage2 policy, optimizer or Ray cluster."""

import os
import time
from pathlib import Path

import numpy as np

from ..tianji_contract import DEFAULT_CONTRACT, TASK_PROMPT, identity_state
from ..tianji_images import fit_frame
from .contract import Stage1Contract, Stage1Request, validate_response
from .. import tianji_joint


class GPUStage1Predictor:
    def __init__(
        self,
        contract=Stage1Contract(),
        *,
        model_path=None,
        norm_stats_path=None,
        device="cuda:0",
    ):
        import torch
        from omegaconf import OmegaConf

        from rlinf.models.embodiment.openpi_rlinf import get_model

        self.contract = contract
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("Stage1 GPU inference requires a CUDA device")
        torch.cuda.set_device(self.device)
        torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
        root = Path(__file__).resolve().parents[5]
        # Use the exact architecture already checked against this checkpoint.
        cfg = OmegaConf.load(
            root / "examples/embodiment/config/tianji_rlt_stage2_async.yaml"
        ).rollout.rlt_feature_model
        # Joint fork: the 16D joint checkpoint (pi05_tianji_joint_30hz).
        self.joint = tianji_joint.joint_mode()
        if self.joint:
            joint_config = os.environ.get("TIANJI_OPENPI_CONFIG", "").strip()
            if not joint_config:
                joint_config = (
                    "tianji_stage1_joint_h60"
                    if int(self.contract.action_chunk_length) == 60
                    else "tianji_stage1_joint"
                )
            if joint_config not in ("tianji_stage1_joint", "tianji_stage1_joint_h60"):
                raise ValueError(
                    f"Unsupported joint OpenPI config {joint_config!r}; "
                    "expected 'tianji_stage1_joint' or 'tianji_stage1_joint_h60'"
                )
            cfg.action_dim = tianji_joint.JOINT_STATE_DIM
            cfg.openpi.action_env_dim = tianji_joint.JOINT_STATE_DIM
            cfg.openpi.config_name = joint_config
            cfg.openpi_data.repo_id = joint_config
            self.fk = tianji_joint.NativeTcpFK()
            self.cache = tianji_joint.CACHE
        if model_path is not None:
            cfg.model_path = model_path
        if norm_stats_path is not None:
            cfg.openpi_data.norm_stats_path = norm_stats_path
        if not cfg.model_path or not cfg.openpi_data.norm_stats_path:
            raise ValueError(
                "Set Stage1 checkpoint and norm stats paths before GPU inference"
            )
        self.model_path = str(cfg.model_path)
        self.norm_stats_path = str(cfg.openpi_data.norm_stats_path)
        self.model = get_model(cfg).eval().requires_grad_(False).to(self.device)
        self.last_inference_s = None

    def _env_obs(self, state, frames, prompt):
        import torch

        if self.joint:
            state = np.asarray(state, dtype=np.float32)
            if state.shape != (tianji_joint.JOINT_STATE_DIM,) or not np.isfinite(state).all():
                raise ValueError("joint Stage1 state must be finite float32[16]")
        else:
            state = DEFAULT_CONTRACT.validate_state(state)
        # The global prompt or the current subtask's prompt (tianji_subtasks).
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Stage1 inference needs a non-empty prompt")
        images = {
            role: fit_frame(frames[role], role) for role in self.contract.camera_roles
        }
        return {
            "states": torch.from_numpy(state[None].copy()),
            "main_images": torch.from_numpy(images["head"][None]),
            "extra_view_images": torch.from_numpy(
                np.stack([images["wrist_A"], images["wrist_B"]])[None]
            ),
            "task_descriptions": [prompt],
        }

    def _physical(self, encoded):
        # The eval wrapper returns the environment's affine action frame.
        # Restore physical targets for the standalone Stage1 v2 contract.
        physical = (
            encoded * DEFAULT_CONTRACT.action_scale + DEFAULT_CONTRACT.action_bias
        )
        return validate_response(physical, self.contract)

    def predict_frames(self, state, frames, prompt=TASK_PROMPT, observation=None):
        """Return physical absolute [H,20], with the same transforms as Stage2."""
        import torch

        if self.joint:
            return self.predict_frames_rtc(state, frames, prompt, None, observation)[0]

        obs = self._env_obs(state, frames, prompt)
        started = time.monotonic()
        with torch.cuda.device(self.device), torch.inference_mode():
            encoded, _ = self.model.predict_action_batch(obs, mode="eval")
            encoded = encoded[0].float().cpu().numpy()
        self.last_inference_s = time.monotonic() - started
        return self._physical(encoded)

    def predict_frames_rtc(self, state, frames, prompt=TASK_PROMPT, rtc=None, observation=None):
        """Same transforms as ``predict_frames``, sampled with RLinf RTC guidance.

        ``rtc`` carries the previous chunk's physical absolute targets. Model
        actions are relative to the inference state, so they are mapped into
        this request's model space by the training input transforms (subtract
        this state, quantile-normalize, pad). Returns physical [H,20] and a
        guidance summary; ``rtc=None`` is unguided.
        """
        import torch

        from rlinf.models.embodiment.openpi.rtc_guidance import RTCGuidanceContext

        from .rtc_sampler import sample_actions_rtc

        wrapper = self.model
        if self.joint:
            if observation is None:
                raise ValueError("joint Stage1 inference needs its source observation")
            state = tianji_joint.joint_state(observation.raw, state)
        repacked = wrapper._repack_env_obs(self._env_obs(state, frames, prompt))
        horizon = self.contract.action_chunk_length
        if rtc is not None:
            previous = np.asarray(rtc.prev_physical, dtype=np.float32)
            if previous.shape != (horizon, 20):
                raise ValueError(
                    f"RTC guidance needs the previous [{horizon},20] physical chunk"
                )
            if self.joint:
                # The model's RTC space is joints: guide with the joint chunk
                # that produced these TCP rows.
                previous = self.cache.joints_for_physical_chunk(previous).astype(np.float32)
            repacked["actions"] = previous[None]
        started = time.monotonic()
        with torch.cuda.device(self.device), torch.inference_mode():
            processed = wrapper.input_transform(repacked, transpose=False)
            context = None
            if rtc is not None:
                context = RTCGuidanceContext(
                    prev_model_actions=processed.pop("actions").to(
                        self.device, torch.float32
                    ),
                    executed_horizon=rtc.executed_horizon,
                    delay_steps=rtc.delay_steps,
                )
            observation = wrapper._observation_dict_to_device(processed)
            model_actions, summary = sample_actions_rtc(
                wrapper,
                observation,
                context,
                hard_only=bool(rtc is not None and getattr(rtc, "hard_only", False)),
            )
            outputs = wrapper.output_transform(
                {"actions": model_actions, "state": observation.state}
            )
            if self.joint:
                joints = outputs["actions"][0].double().cpu().numpy()
            else:
                encoded = wrapper._to_rl_frame(outputs["actions"])
                encoded = encoded[0].float().cpu().numpy()
        if self.joint:
            physical = validate_response(self.fk.tcp_rows(joints), self.contract)
            rows = DEFAULT_CONTRACT.encode_action(physical)
            self.cache.add(physical, [DEFAULT_CONTRACT.validate_action(a) for a in rows], joints)
            self.last_inference_s = time.monotonic() - started
            summary["joint_first_row"] = joints[0].round(5).tolist()
            return physical, summary
        self.last_inference_s = time.monotonic() - started
        return self._physical(encoded), summary

    def __call__(self, request: Stage1Request):
        from ..tianji_backend import _decode_image

        return self.predict_frames(
            request.state,
            {k: _decode_image(v) for k, v in request.images.items()},
            request.prompt,
        )

    def warmup(self):
        """Perform cold CUDA initialization before acquiring a robot observation."""
        if self.joint:
            return self._warmup_joint()
        state = identity_state()
        state[:3] = [0.55, 0.3, 0.15]
        state[10:13] = [0.55, -0.3, 0.15]
        frames = {
            role: np.zeros((224, 224, 3), dtype=np.uint8)
            for role in self.contract.camera_roles
        }
        return self.predict_frames(state, frames)


    def _warmup_joint(self):
        """Cold CUDA pass through the joint model; nothing enters the target cache."""
        import torch

        from .rtc_sampler import sample_actions_rtc

        frames = {
            role: np.zeros((224, 224, 3), dtype=np.uint8)
            for role in self.contract.camera_roles
        }
        obs = self._env_obs(np.zeros(tianji_joint.JOINT_STATE_DIM, np.float32), frames, TASK_PROMPT)
        with torch.cuda.device(self.device), torch.inference_mode():
            processed = self.model.input_transform(self.model._repack_env_obs(obs), transpose=False)
            observation = self.model._observation_dict_to_device(processed)
            actions, _ = sample_actions_rtc(self.model, observation, None)
            joints = self.model.output_transform({"actions": actions, "state": observation.state})["actions"]
        if tuple(joints.shape) != (1, self.contract.action_chunk_length, tianji_joint.JOINT_STATE_DIM):
            raise RuntimeError(f"joint model output {tuple(joints.shape)}, expected [1,H,16]")
        return joints[0].cpu().numpy()


def create_predictor(contract):
    predictor = GPUStage1Predictor(contract)
    predictor.warmup()
    return predictor
