"""RLT Stage 2 policy integrated into RLinf.

This policy keeps the original Stage 2 structure:
- frozen OpenPI VLA
- frozen RL token encoder
- trainable direct Gaussian actor
- trainable twin-Q critic

The policy exposes RLinf-compatible interfaces so the existing rollout/env
pipeline can be reused. Training itself is handled by a dedicated actor worker.
"""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .components import DirectGaussianActor, TwinQCritic, compute_td_target
from .rl_token import RLTokenModel
from .vla_wrapper import Stage2VLAWrapper


class RLTStage2Policy(torch.nn.Module, BasePolicy):
    ROLLOUT_SYNC_PREFIXES = ("actor.",)

    def __init__(
        self,
        cfg: DictConfig,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)

        stage2_cfg = cfg.rlt_stage2
        self.chunk_length = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        self.action_chunk_dim = self.chunk_length * self.action_dim
        self.proprio_dim = int(stage2_cfg.get("proprio_dim", self.action_dim))
        self.act_as_vla_reference = bool(stage2_cfg.get("act_as_vla_reference", False))
        self.load_feature_backbones = bool(
            stage2_cfg.get("load_feature_backbones", True)
        )
        self.load_rl_token_model = bool(
            stage2_cfg.get("load_rl_token_model", self.load_feature_backbones)
        )

        self.vla = None
        self.rl_token_model = None
        if self.load_feature_backbones:
            self.vla = Stage2VLAWrapper(
                model_path=cfg.model_path,
                config_name=stage2_cfg.config_name,
                norm_stats_path=stage2_cfg.get("norm_stats_path", None),
                num_images_in_input=int(stage2_cfg.get("num_images_in_input", 1)),
                num_action_chunks=self.chunk_length,
                action_dim=self.action_dim,
                num_steps=int(stage2_cfg.get("num_steps", cfg.get("num_steps", 5))),
                device=self.device,
            )

        if self.load_rl_token_model:
            self.rl_token_model = RLTokenModel(
                embedding_dim=int(stage2_cfg.get("embedding_dim", 2048)),
                encoder_layers=int(stage2_cfg.get("encoder_layers", 2)),
                encoder_heads=int(stage2_cfg.get("encoder_heads", 8)),
                decoder_layers=int(stage2_cfg.get("decoder_layers", 2)),
                decoder_heads=int(stage2_cfg.get("decoder_heads", 8)),
            ).to(self.device)
            rl_token_ckpt = torch.load(stage2_cfg.rl_token_path, map_location="cpu")
            if "model_state_dict" in rl_token_ckpt:
                rl_token_ckpt = rl_token_ckpt["model_state_dict"]
            self.rl_token_model.load_state_dict(rl_token_ckpt, strict=False)
            self.rl_token_model.eval()
            for param in self.rl_token_model.parameters():
                param.requires_grad_(False)

        embedding_dim = int(stage2_cfg.get("embedding_dim", 2048))
        self.state_dim = embedding_dim + self.proprio_dim

        self.actor = DirectGaussianActor(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=int(stage2_cfg.get("mlp_hidden_dim", 256)),
            num_hidden_layers=int(stage2_cfg.get("mlp_num_hidden_layers", 2)),
            sigma=float(stage2_cfg.get("actor_noise_sigma", 0.1)),
            ref_dropout=float(stage2_cfg.get("ref_action_dropout", 0.0)),
        ).to(self.device)
        self.target_actor = copy.deepcopy(self.actor)
        for param in self.target_actor.parameters():
            param.requires_grad_(False)

        self.critic = TwinQCritic(
            state_dim=self.state_dim,
            action_chunk_dim=self.action_chunk_dim,
            hidden_dim=int(stage2_cfg.get("mlp_hidden_dim", 256)),
            num_hidden_layers=int(stage2_cfg.get("mlp_num_hidden_layers", 2)),
        ).to(self.device)

    @staticmethod
    def _shape_str(tensor: torch.Tensor | None) -> str:
        return "None" if tensor is None else str(tuple(getattr(tensor, "shape", ())))

    @staticmethod
    def _normalize_state_dict_key(key: str) -> str:
        for prefix in ("_fsdp_wrapped_module.", "module."):
            if key.startswith(prefix):
                return key[len(prefix) :]
        return key

    @classmethod
    def _is_rollout_sync_key(cls, key: str) -> bool:
        normalized_key = cls._normalize_state_dict_key(key)
        return any(
            normalized_key.startswith(prefix) for prefix in cls.ROLLOUT_SYNC_PREFIXES
        )

    @classmethod
    def filter_rollout_state_dict(cls, state_dict: dict[str, Any]) -> dict[str, Any]:
        filtered: dict[str, Any] = {}
        for key, value in state_dict.items():
            normalized_key = cls._normalize_state_dict_key(key)
            if not cls._is_rollout_sync_key(normalized_key):
                continue
            if normalized_key in filtered:
                raise ValueError(
                    "Duplicate RLT Stage2 rollout sync key after normalization: "
                    f"{normalized_key}"
                )
            filtered[normalized_key] = value
        if not filtered:
            raise ValueError(
                "RLT Stage2 rollout sync state_dict is empty. Expected actor.* "
                "parameters for direct actor weight sync."
            )
        return filtered

    def rollout_state_dict(self) -> dict[str, Any]:
        return self.filter_rollout_state_dict(self.state_dict())

    def _require_feature_backbones(self, caller: str) -> None:
        if self.vla is None or self.rl_token_model is None:
            raise RuntimeError(
                f"RLT Stage2 {caller} requires VLA/RL-token feature backbones, "
                "but this policy was initialized with "
                "rlt_stage2.load_feature_backbones=False. This mode is only "
                "valid for actor/critic training on cached rollout features."
            )

    def _validate_action_chunk(self, tensor: torch.Tensor | None, *, name: str) -> None:
        expected_tail = (self.chunk_length, self.action_dim)
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 3
            or tuple(tensor.shape[1:]) != expected_tail
        ):
            raise ValueError(
                f"RLT Stage2 {name} shape mismatch: expected [B, "
                f"{self.chunk_length}, {self.action_dim}], got "
                f"{self._shape_str(tensor)}. Check num_action_chunks, "
                "action_dim, OpenPI action_horizon/action_env_dim, and dataset "
                "action shape."
            )

    def _validate_flat_action(self, tensor: torch.Tensor | None, *, name: str) -> None:
        if (
            not isinstance(tensor, torch.Tensor)
            or tensor.ndim != 2
            or tensor.shape[1] != self.action_chunk_dim
        ):
            raise ValueError(
                f"RLT Stage2 {name} shape mismatch: expected [B, "
                f"{self.action_chunk_dim}] from "
                f"{self.chunk_length}x{self.action_dim}, got "
                f"{self._shape_str(tensor)}. Refuse to continue with an "
                "ambiguous action chunk layout."
            )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type for RLT Stage 2: {forward_type}")

    def _encode_state_and_reference(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x, a_tilde_flat, _ = self._prepare_features(env_obs)
        return x, a_tilde_flat

    def _prepare_features(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._require_feature_backbones("_prepare_features")
        observation, processed_obs = self.vla.prepare_obs(env_obs)
        embeddings, pad_mask = self.vla.extract_embeddings(observation)
        z_rl = self.rl_token_model.encode(embeddings, pad_mask)
        a_tilde = self.vla.get_rl_chunk_reference(observation, self.chunk_length)
        self._validate_action_chunk(a_tilde, name="a_tilde")
        a_tilde_flat = a_tilde.reshape(a_tilde.shape[0], -1)
        self._validate_flat_action(a_tilde_flat, name="a_tilde_flat")
        state = self.vla.extract_proprio(observation, self.proprio_dim)
        x = torch.cat([z_rl.to(torch.float32), state], dim=-1)
        return x, a_tilde_flat, processed_obs

    def default_forward(self, **kwargs):
        raise NotImplementedError(
            "RLT Stage 2 does not use RLinf PPO-style default_forward."
        )

    def actor_forward(
        self,
        x: torch.Tensor,
        a_tilde: torch.Tensor,
        *,
        deterministic: bool = False,
        apply_ref_dropout: bool | None = None,
        apply_action_noise: bool | None = None,
    ) -> torch.Tensor:
        self._validate_flat_action(a_tilde, name="actor input a_tilde")
        action = self.actor(
            x,
            a_tilde,
            deterministic=deterministic,
            apply_ref_dropout=apply_ref_dropout,
            apply_action_noise=apply_action_noise,
        )
        self._validate_flat_action(action, name="actor output action_flat")
        return action

    def critic_forward(
        self,
        x: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_flat_action(actions, name="critic input actions")
        return self.critic(x, actions)

    def critic_min(self, x: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        self._validate_flat_action(actions, name="critic_min input actions")
        return self.critic.q_min(x, actions)

    @torch.no_grad()
    def compute_td_target_batch(
        self,
        *,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_x: torch.Tensor,
        next_a_tilde: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_flat_action(
            next_a_tilde,
            name="compute_td_target_batch next_a_tilde",
        )
        stage2_cfg = self.cfg.rlt_stage2
        return compute_td_target(
            rewards=rewards,
            dones=dones,
            next_x=next_x,
            next_a_tilde=next_a_tilde,
            target_actor=self.target_actor,
            critic=self.critic,
            gamma=float(stage2_cfg.get("gamma", self.cfg.get("gamma", 0.99))),
            chunk_length=self.chunk_length,
        )

    @torch.no_grad()
    def update_target_networks(self, tau: float) -> None:
        for online_param, target_param in zip(
            self.actor.parameters(),
            self.target_actor.parameters(),
            strict=True,
        ):
            target_param.data.lerp_(online_param.data, tau)
        self.critic.update_targets(tau)

    def set_online_critic_requires_grad(self, requires_grad: bool) -> None:
        for module in (self.critic.q1, self.critic.q2):
            for param in module.parameters():
                param.requires_grad_(requires_grad)

    @torch.no_grad()
    def encode_obs(
        self,
        env_obs: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._encode_state_and_reference(env_obs)

    @torch.no_grad()
    def predict_vla_reference_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "eval",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del mode, kwargs
        if self.vla is None:
            raise RuntimeError(
                "RLT Stage2 VLA reference prediction requires the VLA backbone. "
                "Do not call this method on actor-only training policies."
            )
        observation, processed_obs = self.vla.prepare_obs(env_obs)
        a_tilde = self.vla.get_rl_chunk_reference(observation, self.chunk_length)
        self._validate_action_chunk(a_tilde, name="expert vla_reference actions")
        action_flat = a_tilde.reshape(a_tilde.shape[0], -1)
        self._validate_flat_action(action_flat, name="expert vla_reference action_flat")
        zeros = torch.zeros(
            action_flat.shape[0], 1, device=action_flat.device, dtype=torch.float32
        )
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {
                "action": action_flat.detach(),
                "a_tilde": action_flat.detach(),
                "tokenized_prompt": processed_obs["tokenized_prompt"].detach(),
                "tokenized_prompt_mask": processed_obs[
                    "tokenized_prompt_mask"
                ].detach(),
            },
        }
        return a_tilde, result

    def sac_forward(
        self,
        obs=None,
        data=None,
        deterministic: bool = False,
        **kwargs,
    ):
        if obs is None:
            obs = data if data is not None else kwargs.get("obs")
        x, a_tilde = self._encode_state_and_reference(obs)
        action = self.actor_forward(x, a_tilde, deterministic=deterministic)
        logprob = torch.zeros(
            action.shape[0], 1, device=action.device, dtype=torch.float32
        )
        return action, logprob, {"x": x, "a_tilde": a_tilde}

    def sac_q_forward(
        self,
        obs=None,
        data=None,
        actions=None,
        state_info=None,
        **kwargs,
    ):
        if state_info is None:
            if obs is None:
                obs = data if data is not None else kwargs.get("obs")
            x, _ = self._encode_state_and_reference(obs)
        else:
            x = state_info["x"]
        q1, q2 = self.critic_forward(x, actions)
        return torch.cat([q1, q2], dim=-1)

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        x, a_tilde, processed_obs = self._prepare_features(env_obs)
        deterministic = mode == "eval"
        if deterministic or self.act_as_vla_reference:
            self.actor.eval()
        if self.act_as_vla_reference:
            action_flat = a_tilde
        else:
            action_flat = self.actor_forward(
                x,
                a_tilde,
                deterministic=deterministic,
            )
        actions = action_flat.reshape(
            action_flat.shape[0],
            self.chunk_length,
            self.action_dim,
        )
        self._validate_action_chunk(actions, name="predict_action_batch actions")
        zeros = torch.zeros(
            action_flat.shape[0], 1, device=action_flat.device, dtype=torch.float32
        )
        result = {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": {
                "action": action_flat.detach(),
                "x": x.detach(),
                "a_tilde": a_tilde.detach(),
                "tokenized_prompt": processed_obs["tokenized_prompt"].detach(),
                "tokenized_prompt_mask": processed_obs["tokenized_prompt_mask"].detach(),
            },
        }
        return actions, result
