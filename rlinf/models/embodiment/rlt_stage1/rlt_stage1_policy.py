"""Native RLinf Stage 1 RL-token training policy."""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType

from .rl_token import RLTokenModel
from .vla_wrapper import Stage1VLAWrapper


class RLTStage1Policy(torch.nn.Module, BasePolicy):
    def __init__(
        self,
        cfg: DictConfig,
        *,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(device)
        self._latest_stage1_metrics: dict[str, torch.Tensor] = {}

        stage1_cfg = cfg.rlt_stage1
        self.alpha = float(stage1_cfg.get("vla_finetune_alpha", 0.0))

        self.vla = Stage1VLAWrapper(
            model_path=cfg.model_path,
            config_name=stage1_cfg.config_name,
            norm_stats_path=stage1_cfg.get("norm_stats_path", None),
            num_images_in_input=int(stage1_cfg.get("num_images_in_input", 2)),
            num_action_chunks=int(cfg.num_action_chunks),
            action_dim=int(cfg.action_dim),
            num_steps=int(stage1_cfg.get("num_steps", 5)),
            device=self.device,
        )
        if self.alpha <= 0:
            self.vla.model.eval()
            for param in self.vla.model.parameters():
                param.requires_grad_(False)
        else:
            self.vla.unfreeze()
            if bool(stage1_cfg.get("gradient_checkpointing", True)) and hasattr(
                self.vla.model, "gradient_checkpointing_enable"
            ):
                self.vla.model.gradient_checkpointing_enable()

        self.rl_token_model = RLTokenModel(
            embedding_dim=int(stage1_cfg.get("embedding_dim", 2048)),
            encoder_layers=int(stage1_cfg.get("encoder_layers", 2)),
            encoder_heads=int(stage1_cfg.get("encoder_heads", 8)),
            decoder_layers=int(stage1_cfg.get("decoder_layers", 2)),
            decoder_heads=int(stage1_cfg.get("decoder_heads", 8)),
        ).to(self.device)

    @property
    def trainable_vla_parameters(self):
        return self.vla.trainable_parameters()

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SFT:
            return self.sft_forward(**kwargs)
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(f"Unsupported forward_type for RLT Stage 1: {forward_type}")

    def sft_forward(self, data: dict[str, Any], **kwargs) -> dict[str, torch.Tensor]:
        observation = data["observation"]
        actions = data["actions"].to(self.device, dtype=torch.float32)

        if self.alpha > 0:
            z, pad_mask, l_vla = self.vla.compute_vla_loss_with_embeddings(
                observation, actions
            )
            l_ro, z_rl, z_hat = self.rl_token_model(
                z.to(self.device), pad_mask.to(self.device)
            )
            loss = l_ro + self.alpha * l_vla
            metrics = {
                "loss": loss,
                "l_ro": l_ro.detach(),
                "l_vla": l_vla.detach(),
                "z_rl": z_rl.detach(),
                "z_hat": z_hat.detach(),
            }
            self._latest_stage1_metrics = {
                key: value for key, value in metrics.items() if key != "loss"
            }
            return metrics

        with torch.no_grad():
            z, pad_mask = self.vla.extract_embeddings(observation)
        l_ro, z_rl, z_hat = self.rl_token_model(
            z.to(self.device), pad_mask.to(self.device)
        )
        metrics = {
            "loss": l_ro,
            "l_ro": l_ro.detach(),
            "z_rl": z_rl.detach(),
            "z_hat": z_hat.detach(),
        }
        self._latest_stage1_metrics = {
            key: value for key, value in metrics.items() if key != "loss"
        }
        return metrics

    def default_forward(self, **kwargs):
        raise NotImplementedError("RLT Stage 1 does not use default_forward.")

    def predict_action_batch(self, **kwargs):
        raise NotImplementedError("RLT Stage 1 does not expose rollout actions.")
