from __future__ import annotations

import os
from typing import Any

import torch
from transformers.feature_extraction_utils import BatchFeature

from rlinf.models.embodiment.dreamzero.wan_flow_matching_runtime import (
    WANDiffusionRuntime,
    WANEncoderRuntime,
)

_PATCHED = False
_LOG_PREFIX = "[RLINF-WAN-REFACTOR]"


def _clone_runtime_state_value(self, value):
    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, list):
        return [self._clone_runtime_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(self._clone_runtime_state_value(v) for v in value)
    return value


def _capture_lazy_runtime_state(self) -> dict[str, Any]:
    return {
        "current_start_frame": self.current_start_frame,
        "language": self._clone_runtime_state_value(self.language),
        "clip_feas": self._clone_runtime_state_value(self.clip_feas),
        "ys": self._clone_runtime_state_value(self.ys),
        "kv_cache1": self._clone_runtime_state_value(self.kv_cache1),
        "kv_cache_neg": self._clone_runtime_state_value(self.kv_cache_neg),
        "crossattn_cache": self._clone_runtime_state_value(self.crossattn_cache),
        "crossattn_cache_neg": self._clone_runtime_state_value(self.crossattn_cache_neg),
        "skip_countdown": getattr(self, "skip_countdown", 0),
    }


def _restore_lazy_runtime_state(self, state: dict[str, Any]) -> None:
    self.current_start_frame = state["current_start_frame"]
    self.language = self._clone_runtime_state_value(state["language"])
    self.clip_feas = self._clone_runtime_state_value(state["clip_feas"])
    self.ys = self._clone_runtime_state_value(state["ys"])
    self.kv_cache1 = self._clone_runtime_state_value(state["kv_cache1"])
    self.kv_cache_neg = self._clone_runtime_state_value(state["kv_cache_neg"])
    self.crossattn_cache = self._clone_runtime_state_value(state["crossattn_cache"])
    self.crossattn_cache_neg = self._clone_runtime_state_value(state["crossattn_cache_neg"])
    self.skip_countdown = state["skip_countdown"]


def _lazy_joint_video_action_refactored(
    self,
    action_input: BatchFeature,
    latent_video: torch.Tensor | None = None,
) -> BatchFeature:
    encoded = self.encoder_runtime.run(action_input=action_input, latent_video=latent_video)
    return self.diffusion_runtime.run(encoded)


def _verify_lazy_joint_video_action_equivalence(
    self,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    latent_video: torch.Tensor | None = None,
    atol: float = 1e-6,
    rtol: float = 1e-6,
) -> BatchFeature:
    initial_state = self._capture_lazy_runtime_state()
    legacy_output = self._lazy_joint_video_action_legacy(
        backbone_output=backbone_output,
        action_input=action_input,
        latent_video=latent_video,
    )
    self._restore_lazy_runtime_state(initial_state)
    ref_output = self._lazy_joint_video_action_refactored(
        action_input=action_input,
        latent_video=latent_video,
    )

    action_close = torch.allclose(
        legacy_output["action_pred"],
        ref_output["action_pred"],
        atol=atol,
        rtol=rtol,
    )
    video_close = torch.allclose(
        legacy_output["video_pred"],
        ref_output["video_pred"],
        atol=atol,
        rtol=rtol,
    )
    action_max_abs = (legacy_output["action_pred"] - ref_output["action_pred"]).abs().max().item()
    video_max_abs = (legacy_output["video_pred"] - ref_output["video_pred"]).abs().max().item()
    if not action_close or not video_close:
        raise AssertionError(
            "lazy_joint_video_action refactor mismatch: "
            f"action_close={action_close}, video_close={video_close}, "
            f"action_max_abs={action_max_abs:.6e}, video_max_abs={video_max_abs:.6e}"
        )
    if self.ip_rank == 0:
        print(
            "[VERIFY] lazy_joint_video_action refactor matched: "
            f"action_max_abs={action_max_abs:.6e}, video_max_abs={video_max_abs:.6e}"
        )
    return ref_output


def lazy_joint_video_action(
    self,
    backbone_output: BatchFeature,
    action_input: BatchFeature,
    latent_video: torch.Tensor | None = None,
) -> BatchFeature:
    self.set_frozen_modules_to_eval_mode()
    verify_refactor = os.getenv("VERIFY_WAN_REFACTOR", "False").lower() == "true"
    if not getattr(self, "_rlinf_refactor_entry_logged", False):
        mode = "verify(double-run)" if verify_refactor else "refactored(single-run)"
        print(f"{_LOG_PREFIX} lazy_joint_video_action entry mode={mode}")
        self._rlinf_refactor_entry_logged = True
    if verify_refactor:
        return self._verify_lazy_joint_video_action_equivalence(
            backbone_output=backbone_output,
            action_input=action_input,
            latent_video=latent_video,
        )
    return self._lazy_joint_video_action_refactored(
        action_input=action_input,
        latent_video=latent_video,
    )


def apply_wan_refactor_patch() -> None:
    global _PATCHED
    if _PATCHED:
        return

    from groot.vla.model.dreamzero.action_head.wan_flow_matching_action_tf import WANPolicyHead

    if not hasattr(WANPolicyHead, "_lazy_joint_video_action_legacy"):
        WANPolicyHead._lazy_joint_video_action_legacy = WANPolicyHead.lazy_joint_video_action

    old_init = WANPolicyHead.__init__

    def patched_init(self, *args, **kwargs):
        old_init(self, *args, **kwargs)
        self.encoder_runtime = WANEncoderRuntime(self)
        self.diffusion_runtime = WANDiffusionRuntime(self)
        self._rlinf_refactor_entry_logged = False
        if getattr(self, "ip_rank", 0) == 0:
            print(
                f"{_LOG_PREFIX} runtime attached "
                "(encoder=WANEncoderRuntime, diffusion=WANDiffusionRuntime)"
            )

    WANPolicyHead.__init__ = patched_init
    WANPolicyHead._clone_runtime_state_value = _clone_runtime_state_value
    WANPolicyHead._capture_lazy_runtime_state = _capture_lazy_runtime_state
    WANPolicyHead._restore_lazy_runtime_state = _restore_lazy_runtime_state
    WANPolicyHead._lazy_joint_video_action_refactored = _lazy_joint_video_action_refactored
    WANPolicyHead._verify_lazy_joint_video_action_equivalence = _verify_lazy_joint_video_action_equivalence
    WANPolicyHead.lazy_joint_video_action = lazy_joint_video_action

    print(f"{_LOG_PREFIX} patch applied to WANPolicyHead")
    _PATCHED = True
