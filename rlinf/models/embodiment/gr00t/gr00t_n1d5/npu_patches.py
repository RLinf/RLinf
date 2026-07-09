# Copyright 2025 The RLinf Authors.
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

import torch

try:
    import torch_npu
except ImportError:
    # Only the fused kernels below use torch_npu, and only on Ascend. Keep the
    # module importable elsewhere so apply_npu_patches can no-op.
    torch_npu = None


def npu_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """``Qwen3RMSNorm.forward`` via the fused ``npu_rms_norm`` kernel."""
    return torch_npu.npu_rms_norm(
        hidden_states, self.weight, epsilon=self.variance_epsilon
    )[0]


def npu_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Qwen3 ``apply_rotary_pos_emb`` via the fused ``npu_rotary_mul`` kernel.

    ``position_ids`` is accepted for signature compatibility but unused, matching
    upstream where cos/sin are already gathered by position.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


def get_radio_compatible_cuda_capability_on_npu(*_args, **_kwargs) -> tuple[int, int]:
    """RADIO's minimum accepted CUDA capability (Ampere 8.0), as a sentinel.

    Isaac-GR00T N1.5's radio_model calls ``torch.cuda.get_device_capability()``
    unconditionally at import. Ascend has none, but RADIO uses a separate NPU
    attention path, so the value only has to clear the check — it does not
    describe the device.
    """
    return (8, 0)


# Patcher references replacement objects by string path.
_MODULE = "rlinf.models.embodiment.gr00t.gr00t_n1d5.npu_patches"


def _is_npu() -> bool:
    """Whether this worker runs on an Ascend NPU, per the Worker device API."""
    from rlinf.scheduler import AcceleratorType, Worker

    return Worker.accelerator_type == AcceleratorType.NPU


def apply_npu_patches(patcher) -> dict | None:
    """Register the Ascend patches for building GR00T N1.5; return restore state.

    No-op returning ``None`` off NPU. Call before ``patcher.apply()``; pass the
    result to :func:`restore_npu_patches` after model construction. Installs:

    * fused NPU kernels for Qwen3 RMSNorm / rotary embedding (kept for the
      model's lifetime, not restored);
    * a ``get_device_capability`` sentinel so RADIO's import-time check passes;
    * a ``flash_attn`` stub so import sites succeed, cleared once the
      eagle_2_5_vl config loads so the absent package reads as unavailable;
    * a ``AutoConfig.from_pretrained`` wrapper that triggers that clearing.

    The last two are patched directly, not through ``patcher``: it matches by
    ``id`` and a classmethod resolves to an ephemeral bound method, so a Patcher
    entry for ``from_pretrained`` would never take effect.
    """
    if not _is_npu():
        return None

    patcher.add_patch(
        "transformers.models.qwen3.modeling_qwen3.apply_rotary_pos_emb",
        f"{_MODULE}.npu_apply_rotary_pos_emb",
    )
    patcher.add_patch(
        "transformers.models.qwen3.modeling_qwen3.Qwen3RMSNorm.forward",
        f"{_MODULE}.npu_rmsnorm_forward",
    )
    # Capture before patcher.apply() replaces it, so restore can put it back.
    original_get_device_capability = torch.cuda.get_device_capability
    patcher.add_patch(
        "torch.cuda.get_device_capability",
        f"{_MODULE}.get_radio_compatible_cuda_capability_on_npu",
    )
    patcher.skip_import("flash_attn")

    from transformers import AutoConfig

    original_from_pretrained_descriptor = AutoConfig.__dict__["from_pretrained"]
    original_from_pretrained = AutoConfig.from_pretrained

    def _from_pretrained_and_clear_flash_attn_stub(cls, *args, **kwargs):
        config = original_from_pretrained(*args, **kwargs)
        if getattr(config, "model_type", None) == "eagle_2_5_vl":
            patcher.clear_stub_import("flash_attn")
        return config

    AutoConfig.from_pretrained = classmethod(_from_pretrained_and_clear_flash_attn_stub)

    return {
        "get_device_capability": original_get_device_capability,
        "autoconfig_from_pretrained": original_from_pretrained_descriptor,
    }


def restore_npu_patches(patcher, state: dict | None) -> None:
    """Undo the process-global patches from :func:`apply_npu_patches`.

    ``state`` is that call's return value; ``None`` (off NPU) is a no-op. The
    fused RMSNorm / rotary patches are intentionally left in place.
    """
    if state is None:
        return

    from transformers import AutoConfig

    AutoConfig.from_pretrained = state["autoconfig_from_pretrained"]
    torch.cuda.get_device_capability = state["get_device_capability"]
    patcher.clear_stub_import("flash_attn")
