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

"""Ascend NPU building blocks shared by the GR00T model versions.

Every GR00T version needs the same three things on Ascend: fused kernels for the
Qwen RMSNorm and rotary embedding, a way to keep an unloadable CUDA ``torchcodec``
from escaping GR00T's optional-import handling, and flash-attention entry points
for backbone code that assumes the ``flash_attn`` package. The per-version
``npu_patches`` modules register these through :class:`rlinf.utils.patcher`.
"""

from __future__ import annotations

import contextlib
import importlib.abc
import importlib.util
import inspect
import sys
from typing import Callable, Iterable

import torch

try:
    import torch_npu
except ImportError:
    # Only the fused kernels use torch_npu, and only on Ascend. Keep the module
    # importable elsewhere so the apply_npu_patches callers can no-op.
    torch_npu = None


def is_npu() -> bool:
    """Whether this worker runs on an Ascend NPU, per the Worker device API."""
    from rlinf.scheduler import AcceleratorType, Worker

    return Worker.accelerator_type == AcceleratorType.NPU


def npu_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """``RMSNorm.forward`` via the fused ``npu_rms_norm`` kernel.

    Applies to any RMSNorm module holding ``weight`` and ``variance_epsilon``,
    which covers Qwen3 (``Qwen3RMSNorm``) and Qwen3-VL (``Qwen3VLTextRMSNorm``).
    """
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
    """``apply_rotary_pos_emb`` via the fused ``npu_rotary_mul`` kernel.

    ``position_ids`` is accepted for signature compatibility but unused, matching
    upstream where cos/sin are already gathered by position.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    return q_embed, k_embed


class _TorchcodecUnavailableFinder:
    """Raise ``ImportError`` for ``torchcodec`` so GR00T treats it as optional.

    GR00T imports ``torchcodec`` and handles only ``ImportError`` /
    ``RuntimeError``. The PyPI wheel is CUDA-only and ``dlopen``s
    ``libnvrtc.so``, which raises ``OSError`` on Ascend. Intercepting the import
    converts that failure into the exception GR00T already handles.
    """

    def find_spec(self, fullname, _path, _target=None):
        if fullname == "torchcodec" or fullname.startswith("torchcodec."):
            raise ImportError(
                "torchcodec is unavailable on this platform (CUDA wheel "
                "cannot load without NVIDIA libraries)."
            )
        return None


def hide_unimportable_torchcodec() -> None:
    """If torchcodec is installed but cannot load, make later imports ImportError."""
    if "torchcodec" in sys.modules:
        return
    try:
        import torchcodec  # noqa: F401
    except OSError:
        for name in list(sys.modules):
            if name == "torchcodec" or name.startswith("torchcodec."):
                sys.modules.pop(name, None)
        sys.meta_path.insert(0, _TorchcodecUnavailableFinder())
    except (ImportError, RuntimeError):
        pass


@contextlib.contextmanager
def npu_broadcast_safe_sdpa():
    """Give the NPU attention kernel a mask it accepts, for the duration.

    ``scaled_dot_product_attention`` on CUDA broadcasts an attention mask whose
    query dimension is 1 over the real query length. Ascend's
    ``FlashAttentionScore`` does not: it rejects the shape outright, for example
    ``get unsupported atten_mask shape, the shape is [2, 32, 1, 570] ... Sq=[51]``
    for GR00T's action-head DiT, which builds one mask row and relies on the
    broadcast. Materializing that dimension keeps the mask's meaning and costs a
    few megabytes.
    """
    original = torch.nn.functional.scaled_dot_product_attention

    def scaled_dot_product_attention(
        query, key, value, attn_mask=None, *args, **kwargs
    ):
        if (
            attn_mask is not None
            and attn_mask.dim() == 4
            and attn_mask.shape[-2] == 1
            and query.shape[-2] != 1
        ):
            attn_mask = attn_mask.expand(
                *attn_mask.shape[:-2], query.shape[-2], attn_mask.shape[-1]
            ).contiguous()
        return original(query, key, value, attn_mask, *args, **kwargs)

    torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def npu_sdpa_context():
    """Replacement for GR00T's ``_sdpa_context``, which is a no-op off CUDA.

    GR00T wraps its DiT attention in that hook to force a safe SDPA path on
    platforms where the fused kernel misbehaves. It selects the backend through
    ``torch.backends.cuda``, which does nothing on Ascend, so this substitutes
    the fix Ascend actually needs.
    """
    return npu_broadcast_safe_sdpa()


def _npu_flash_attention_symbols() -> dict[str, object]:
    """The names backbone code expects from ``flash_attn``, backed by NPU kernels.

    Transformers ships Ascend implementations of the flash-attention entry points
    in ``transformers.integrations.npu_flash_attention`` and selects them itself
    for its own models. Backbone code vendored from elsewhere imports the names
    from ``flash_attn`` instead, under a ``is_flash_attn_2_available()`` guard
    that is False on Ascend, so those names stay undefined and the forward pass
    fails with ``NameError``. This returns the same names bound to the NPU
    kernels, for :func:`bind_npu_flash_attention` to install.

    Which names exist depends on the Transformers release: 4.51 exports the
    padding helpers, 4.57 no longer does. Only what is present is returned.
    """
    from transformers.integrations import npu_flash_attention as npu_fa

    symbols: dict[str, object] = {}
    for name, attr in (
        ("flash_attn_func", "npu_flash_attn_func"),
        ("flash_attn_varlen_func", "npu_flash_attn_varlen_func"),
        ("index_first_axis", "index_first_axis"),
        ("index_put_first_axis", "index_put_first_axis"),
        ("pad_input", "pad_input"),
        ("unpad_input", "unpad_input"),
    ):
        value = getattr(npu_fa, attr, None)
        if value is not None:
            symbols[name] = value

    varlen = symbols.get("flash_attn_varlen_func")
    if varlen is not None:
        # Callers gate a `window_size` kwarg on this. The NPU entry points take
        # no such parameter, so report it as unsupported rather than passing one.
        symbols["_flash_supports_window_size"] = (
            "window_size" in inspect.signature(varlen).parameters
        )
    return symbols


def bind_npu_flash_attention(module) -> None:
    """Bind the NPU flash-attention entry points into ``module``'s namespace.

    Only names the module left undefined are filled in, so a module that found
    real kernels keeps them.
    """
    for name, value in _npu_flash_attention_symbols().items():
        if getattr(module, name, None) is None:
            setattr(module, name, value)


class _NpuFlashAttentionBinder(importlib.abc.MetaPathFinder):
    """Bind NPU flash-attention into named modules as they are imported.

    Backbone code loaded through ``trust_remote_code`` lands in a
    ``transformers_modules.*`` package whose name is not known in advance and
    which does not exist until the model config is read, so it cannot be patched
    up front. Matching on the final component and wrapping the loader binds the
    names the moment the module finishes executing, before any forward pass.
    """

    def __init__(self, basenames: Iterable[str]):
        self._basenames = set(basenames)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.rsplit(".", 1)[-1] not in self._basenames:
            return None
        # Re-entrant: drop out of the path while resolving the real spec.
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        except (ImportError, AttributeError, ValueError):
            return None
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec

        original_exec_module = spec.loader.exec_module

        def exec_module(module, _original=original_exec_module):
            _original(module)
            bind_npu_flash_attention(module)

        spec.loader.exec_module = exec_module
        return spec


def install_npu_flash_attention(*basenames: str) -> Callable[[], None]:
    """Bind NPU flash-attention into the named modules, now and on later import.

    ``basenames`` are final module name components, e.g. ``"modeling_siglip2"``.
    Modules already imported are bound immediately. Returns a callable that stops
    binding future imports; bindings already made stay, because the model built
    under them keeps calling those names.
    """
    for name, module in list(sys.modules.items()):
        if module is not None and name.rsplit(".", 1)[-1] in basenames:
            bind_npu_flash_attention(module)

    binder = _NpuFlashAttentionBinder(basenames)
    sys.meta_path.insert(0, binder)

    def uninstall() -> None:
        if binder in sys.meta_path:
            sys.meta_path.remove(binder)

    return uninstall
