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

import os
import warnings
from contextlib import contextmanager
from math import sqrt
from typing import ContextManager, Generator, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForVision2Seq

from rlinf.config import SupportedModel, get_supported_model, torch_dtype_from_precision
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.hybrid_engines.fsdp import (
    FSDP,
    FSDPModule,
)
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.hybrid_engines.fsdp.utils import (
    create_device_mesh,
    get_lr_scheduler,
)
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import warmup_optimizer_state

warnings.filterwarnings(
    "ignore",
    message=".*NO_SHARD.*full_state_dict.*",
    category=UserWarning,
)


def _is_musa_available() -> bool:
    return hasattr(torch, "musa") and torch.musa.is_available()


def _is_cuda_backend_available() -> bool:
    return torch.cuda.is_available() and hasattr(torch._C, "_cuda_setDevice")


_ORIGINAL_SDPA = None
_ORIGINAL_SDPA_C = None
_LAST_SDPA_INPUTS = None
_LAST_SDPA_TENSORS = None
_SDPA_DEBUG_CALL_COUNT = 0
_IN_SDPA_DEBUG_WRAPPER = False
_SDPA_CAPTURE_TENSORS = False
_SDPA_CAPTURE_MAX_NUMEL = 0
_MUSA_EAGER_SDPA_FALLBACK_INSTALLED = False


def _musa_sdpa_eager(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """
    Eager SDPA fallback for MUSA.

    This implementation avoids the backend flash/mem-efficient SDPA kernels that
    can produce NaNs on unsupported head dimensions.
    """
    if enable_gqa:
        query_heads = query.size(-3)
        kv_heads = key.size(-3)
        if query_heads % kv_heads != 0:
            raise ValueError(
                f"enable_gqa=True requires query_heads % kv_heads == 0, "
                f"got {query_heads} and {kv_heads}."
            )
        repeat = query_heads // kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=-3)
            value = value.repeat_interleave(repeat, dim=-3)

    compute_dtype = (
        torch.float32 if query.dtype in (torch.float16, torch.bfloat16) else query.dtype
    )
    q = query.to(compute_dtype)
    k = key.to(compute_dtype)
    v = value.to(compute_dtype)

    scale_factor = (1.0 / sqrt(q.size(-1))) if scale is None else scale
    attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

    if is_causal:
        tgt_len, src_len = q.size(-2), k.size(-2)
        causal_mask = torch.ones(
            (tgt_len, src_len),
            device=q.device,
            dtype=torch.bool,
        ).tril()
        attn_weight = attn_weight.masked_fill(
            ~causal_mask, torch.finfo(attn_weight.dtype).min
        )

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(
                ~attn_mask, torch.finfo(attn_weight.dtype).min
            )
        else:
            attn_weight = attn_weight + attn_mask.to(attn_weight.dtype)

    attn_weight = torch.softmax(attn_weight, dim=-1, dtype=torch.float32).to(
        attn_weight.dtype
    )
    if dropout_p > 0.0:
        attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    output = torch.matmul(attn_weight, v)
    return output.to(query.dtype)


def install_musa_eager_sdpa_fallback(logger, rank: int) -> None:
    """Force torch SDPA to use eager math implementation on MUSA."""
    global _ORIGINAL_SDPA
    global _ORIGINAL_SDPA_C
    global _MUSA_EAGER_SDPA_FALLBACK_INSTALLED

    if _MUSA_EAGER_SDPA_FALLBACK_INSTALLED:
        return

    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = _musa_sdpa_eager

    sdpa_c = getattr(getattr(torch, "_C", None), "_nn", None)
    sdpa_c_fn = (
        getattr(sdpa_c, "scaled_dot_product_attention", None)
        if sdpa_c is not None
        else None
    )
    if _ORIGINAL_SDPA_C is None and callable(sdpa_c_fn):
        _ORIGINAL_SDPA_C = sdpa_c_fn
    if callable(sdpa_c_fn):
        try:
            sdpa_c.scaled_dot_product_attention = _musa_sdpa_eager
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[MUSA-SDPA] Failed to replace torch._C._nn.scaled_dot_product_attention: "
                f"{exc}"
            )

    _MUSA_EAGER_SDPA_FALLBACK_INSTALLED = True
    logger.warning(
        f"[MUSA-SDPA] rank={rank} forced scaled_dot_product_attention to eager math fallback."
    )


def _tensor_meta(tensor: torch.Tensor | None) -> dict | None:
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return {"type": type(tensor).__name__}
    return {
        "shape": tuple(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
    }


def get_last_sdpa_input_shapes() -> dict | None:
    """Return the latest recorded SDPA inputs metadata for debug."""
    return _LAST_SDPA_INPUTS


def _clone_tensor_for_debug(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None or not isinstance(tensor, torch.Tensor):
        return None
    return tensor.detach().to("cpu").clone()


def get_last_sdpa_input_tensors() -> dict | None:
    """Return latest recorded SDPA tensors (CPU) for debug dump."""
    return _LAST_SDPA_TENSORS


def dump_last_sdpa_inputs(
    save_dir: str,
    rank: int,
    step: int,
    logger=None,
) -> str | None:
    """
    Persist latest SDPA debug snapshot to disk.

    The saved pt includes:
      - meta: shape/dtype/device information
      - tensors: q/k/v/mask CPU tensors (if capture enabled and within size limit)
    """
    snapshot = {
        "meta": get_last_sdpa_input_shapes(),
        "tensors": get_last_sdpa_input_tensors(),
        "rank": rank,
        "step": step,
    }
    if snapshot["meta"] is None and snapshot["tensors"] is None:
        return None

    os.makedirs(save_dir, exist_ok=True)
    filename = f"sdpa_input_rank{rank}_step{step}.pt"
    save_path = os.path.join(save_dir, filename)
    torch.save(snapshot, save_path)
    if logger is not None:
        logger.warning(f"[SDPA-DEBUG] dumped latest SDPA inputs to: {save_path}")
    return save_path


def install_sdpa_input_shape_debug(
    logger,
    rank: int,
    max_print: int = 8,
    capture_tensors: bool = False,
    capture_max_numel: int = 0,
) -> None:
    """Install a debug wrapper to record scaled_dot_product_attention input shapes."""
    global _ORIGINAL_SDPA
    global _ORIGINAL_SDPA_C
    global _LAST_SDPA_TENSORS
    global _SDPA_DEBUG_CALL_COUNT
    global _IN_SDPA_DEBUG_WRAPPER
    global _SDPA_CAPTURE_TENSORS
    global _SDPA_CAPTURE_MAX_NUMEL
    installed_targets = []
    _SDPA_CAPTURE_TENSORS = capture_tensors
    _SDPA_CAPTURE_MAX_NUMEL = capture_max_numel

    def _record_and_maybe_log(args, kwargs, source: str) -> None:
        global _LAST_SDPA_INPUTS
        global _SDPA_DEBUG_CALL_COUNT

        query = args[0] if len(args) > 0 else kwargs.get("query", None)
        key = args[1] if len(args) > 1 else kwargs.get("key", None)
        value = args[2] if len(args) > 2 else kwargs.get("value", None)
        attn_mask = (
            args[3]
            if len(args) > 3
            else kwargs.get("attn_mask", kwargs.get("mask", None))
        )
        dropout_p = (
            args[4] if len(args) > 4 else kwargs.get("dropout_p", None)
        )
        is_causal = (
            args[5] if len(args) > 5 else kwargs.get("is_causal", None)
        )

        _LAST_SDPA_INPUTS = {
            "rank": rank,
            "source": source,
            "query": _tensor_meta(query),
            "key": _tensor_meta(key),
            "value": _tensor_meta(value),
            "attn_mask": _tensor_meta(attn_mask),
            "dropout_p": dropout_p,
            "is_causal": is_causal,
        }
        if _SDPA_CAPTURE_TENSORS:
            tensors = {
                "query": _clone_tensor_for_debug(query),
                "key": _clone_tensor_for_debug(key),
                "value": _clone_tensor_for_debug(value),
                "attn_mask": _clone_tensor_for_debug(attn_mask),
            }
            if _SDPA_CAPTURE_MAX_NUMEL > 0:
                for name, tensor in tensors.items():
                    if tensor is not None and tensor.numel() > _SDPA_CAPTURE_MAX_NUMEL:
                        tensors[name] = None
                        _LAST_SDPA_INPUTS[f"{name}_capture_skipped_numel"] = int(
                            tensor.numel()
                        )
            _LAST_SDPA_TENSORS = tensors
        else:
            _LAST_SDPA_TENSORS = None
        _SDPA_DEBUG_CALL_COUNT += 1
        if _SDPA_DEBUG_CALL_COUNT <= max_print:
            logger.warning(
                "[SDPA-DEBUG] "
                f"rank={rank}, call={_SDPA_DEBUG_CALL_COUNT}, "
                f"q={_LAST_SDPA_INPUTS['query']}, "
                f"k={_LAST_SDPA_INPUTS['key']}, "
                f"v={_LAST_SDPA_INPUTS['value']}, "
                f"mask={_LAST_SDPA_INPUTS['attn_mask']}, "
                f"dropout_p={dropout_p}, is_causal={is_causal}"
            )

    def _wrap_sdpa(fn, source: str):
        def _wrapped(*args, **kwargs):
            global _IN_SDPA_DEBUG_WRAPPER
            if _IN_SDPA_DEBUG_WRAPPER:
                return fn(*args, **kwargs)
            _IN_SDPA_DEBUG_WRAPPER = True
            try:
                _record_and_maybe_log(args, kwargs, source=source)
                return fn(*args, **kwargs)
            finally:
                _IN_SDPA_DEBUG_WRAPPER = False

        return _wrapped

    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = _wrap_sdpa(_ORIGINAL_SDPA, source="torch.nn.functional")
        installed_targets.append("torch.nn.functional")

    sdpa_c = getattr(getattr(torch, "_C", None), "_nn", None)
    sdpa_c_fn = (
        getattr(sdpa_c, "scaled_dot_product_attention", None) if sdpa_c is not None else None
    )
    if _ORIGINAL_SDPA_C is None and callable(sdpa_c_fn):
        _ORIGINAL_SDPA_C = sdpa_c_fn
        try:
            sdpa_c.scaled_dot_product_attention = _wrap_sdpa(
                _ORIGINAL_SDPA_C, source="torch._C._nn"
            )
            installed_targets.append("torch._C._nn")
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "[SDPA-DEBUG] Failed to wrap torch._C._nn.scaled_dot_product_attention: "
                f"{exc}"
            )

    if installed_targets:
        logger.warning(
            "[SDPA-DEBUG] installed hooks on: "
            f"{installed_targets}, rank={rank}, max_print={max_print}, "
            f"capture_tensors={_SDPA_CAPTURE_TENSORS}, "
            f"capture_max_numel={_SDPA_CAPTURE_MAX_NUMEL}"
        )


class FSDPModelManager:
    """
    FSDP Model Manager for RL training
    """

    def __init__(self, cfg: DictConfig, world_size: int, rank: int) -> None:
        """
        Initialize FSDP Model Manager.

        Args:
            cfg: actor config in yaml file.
            world_size: total number of FSDP actor processes.
        """
        self._cfg = cfg
        self._logger = get_logger()
        self._rank = rank
        self.torch_dtype = torch_dtype_from_precision(self._cfg.model.precision)

        self.optimizer_steps = 0
        self.critic_warmup_steps = 0
        if self._cfg.get("optim", {}).get(
            "critic_warmup_steps", None
        ) and self._cfg.model.get("add_value_head", False):
            self.critic_warmup_steps = self._cfg.optim.critic_warmup_steps
        self.store_requires_grad_param_name = []

        if cfg.get("tokenizer", {}).get("tokenizer_model", None) is not None:
            self.tokenizer = hf_tokenizer(cfg.tokenizer.tokenizer_model)

        self._device_mesh = create_device_mesh(
            world_size, self._cfg.fsdp_config.get("fsdp_size", -1)
        )
        self._dp_group = (
            self._device_mesh["ddp"].get_group()
            if "ddp" in self._device_mesh.mesh_dim_names
            else None
        )

        self._strategy = FSDPStrategyBase.create(
            self._cfg, world_size, self._dp_group, self._logger
        )
        self.accelerator_type = self._select_accelerator_type()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if self.accelerator_type == "musa":
            torch.musa.set_device(local_rank)
            self.device = torch.musa.current_device()
        elif self.accelerator_type == "cuda":
            torch.cuda.set_device(local_rank)
            self.device = torch.cuda.current_device()
        else:
            raise RuntimeError(
                "No available accelerator backend found for FSDP. "
                "Expected one of cuda/musa."
            )

        force_musa_eager_sdpa = bool(
            self._cfg.model.get("force_musa_eager_sdpa", True)
        )
        if self.accelerator_type == "musa" and force_musa_eager_sdpa:
            install_musa_eager_sdpa_fallback(logger=self._logger, rank=self._rank)

        enable_sdpa_debug = bool(
            self._cfg.get("debug_sdpa_input_shapes", False)
            or self._cfg.get("debug_detect_anomaly", False)
            or self._cfg.get("debug_nan_checks", False)
        )
        capture_sdpa_tensors = bool(
            self._cfg.get("debug_sdpa_save_inputs", enable_sdpa_debug)
        )
        if enable_sdpa_debug:
            install_sdpa_input_shape_debug(
                logger=self._logger,
                rank=self._rank,
                max_print=int(self._cfg.get("debug_sdpa_print_max", 8)),
                capture_tensors=capture_sdpa_tensors,
                capture_max_numel=int(self._cfg.get("debug_sdpa_capture_max_numel", 0)),
            )
        self.amp_context = self._create_amp_context()

        self.is_weight_offloaded = False
        self.is_optimizer_offloaded = False

    def _create_amp_context(self) -> ContextManager:
        """
        Create AMP context manager based on configuration.

        Returns:
            A context manager for automatic mixed precision (AMP) if enabled,
            otherwise a null context manager.
        """
        from contextlib import nullcontext

        if not self._cfg.fsdp_config.amp.enabled:
            self._logger.info("[FSDP] AMP is disabled.")
            return nullcontext()

        precision = torch_dtype_from_precision(self._cfg.fsdp_config.amp.precision)

        self._logger.info(f"[FSDP] AMP is enabled with precision: {precision}.")
        return torch.amp.autocast(device_type=self.accelerator_type, dtype=precision)

    def _select_accelerator_type(self) -> str:
        # Prefer MUSA on environments where CUDA APIs may be partially visible
        # but unsupported.
        if _is_musa_available():
            return "musa"
        if _is_cuda_backend_available():
            return "cuda"
        return "cpu"
    def model_provider_func(self) -> torch.nn.Module:
        """
        Initialize model used by FSDP actor

        Returns:
            model: the initialized model.
        """
        cfg = self._cfg
        use_gptq = cfg.model.get("gptq_model", False)
        load_in_8bit = cfg.model.get("load_in_8bit", False)

        use_triton = cfg.get("use_triton", True)

        if self.accelerator_type == "cuda":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"cuda:{local_rank}")
        elif self.accelerator_type == "musa":
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            device = torch.device(f"musa:{local_rank}")
        else:
            raise ValueError("Device(Cuda,Musa) not available")

        model_config = AutoConfig.from_pretrained(
            cfg.model.model_path,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        if use_gptq:
            from auto_gptq import AutoGPTQForCausalLM  # type: ignore[import-not-found]

            model_wrapper = AutoGPTQForCausalLM.from_quantized(
                cfg.model.model_path,
                device=device,
                use_triton=use_triton,
            )
            model = model_wrapper.model
        elif load_in_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                cfg.model.model_path,
                config=model_config,
                load_in_8bit=True,
            )
        else:
            if type(model_config) in AutoModelForVision2Seq._model_mapping.keys():
                auto_model_class = AutoModelForVision2Seq
            else:
                auto_model_class = AutoModelForCausalLM

            model = auto_model_class.from_pretrained(
                cfg.model.model_path,
                torch_dtype=self.torch_dtype,
                config=model_config,
                trust_remote_code=True,
            )

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if cfg.fsdp_config.use_liger_kernel:
            self._optimize_with_liger_kernel(model)

        return model

    def _optimize_with_liger_kernel(self, model: torch.nn.Module) -> None:
        """
        Replace model modules with liger-kernel optimized modules.

        Args:
            model: the model to be optimized.
        """
        if self._cfg.model.get("gptq_model", False) or self._cfg.model.get(
            "load_in_8bit", False
        ):
            self._logger.info(
                "[FSDP] Skip using liger-kernel optimized modules for GPTQ/8bit models."
            )
            return
        try:
            from liger_kernel.transformers import (
                apply_liger_kernel_to_qwen2,
                apply_liger_kernel_to_qwen2_5_vl,
            )

            MODEL_LIGER_KERNEL_APPLY_FUNC = {
                SupportedModel.QWEN2_5: (
                    apply_liger_kernel_to_qwen2,
                    {
                        "rope": True,
                        "rms_norm": True,
                        "swiglu": True,
                        "fused_linear_cross_entropy": True,
                    },
                ),
                SupportedModel.QWEN2_5_VL: (
                    apply_liger_kernel_to_qwen2_5_vl,
                    {
                        "rope": True,
                        "rms_norm": True,
                        "swiglu": True,
                        "fused_linear_cross_entropy": True,
                    },
                ),
            }
            model_type = get_supported_model(
                self._cfg.model.get("model_type", "").lower()
            )
            if model_type in MODEL_LIGER_KERNEL_APPLY_FUNC:
                apply_func, apply_kwargs = MODEL_LIGER_KERNEL_APPLY_FUNC[model_type]
                apply_func(
                    model=model,
                    **apply_kwargs,
                )
                self._logger.info(
                    f"[FSDP] Applied liger-kernel optimizations for model_type: {model_type.value}, used kwargs: {apply_kwargs}"
                )
            else:
                self._logger.info(
                    f"[FSDP] No liger-kernel optimizations applied for model_type: {model_type.value}"
                )
                return
        except Exception as e:
            self._logger.warning(f"[FSDP] Liger kernels not applied: {e}")

    def setup_model_and_optimizer(self) -> None:
        """
        Setup model, lr_scheduler, optimizer and grad_scaler.
        """
        module = self.model_provider_func()

        # Enable gradient checkpointing if configured
        if self._cfg.fsdp_config.get("gradient_checkpointing", False):
            self._logger.info("[FSDP] Enabling gradient checkpointing")
            module.gradient_checkpointing_enable()
        else:
            self._logger.info("[FSDP] Gradient checkpointing is disabled")

        # build model, optimizer, lr_scheduler, grad_scaler
        self.model = self._strategy.wrap_model(
            model=module, device_mesh=self._device_mesh
        )
        self.optimizer = self.build_optimizer(
            model=self.model, enable_critic_warmup=self.critic_warmup_steps > 0
        )

        self.lr_scheduler = self.build_lr_scheduler(
            optimizer=self.optimizer, optim_config=self._cfg.optim
        )
        self.grad_scaler = self.build_grad_scaler(
            self._cfg.fsdp_config.amp.use_grad_scaler
        )

    def get_model_state_dict(self, cpu_offload: bool, full_state_dict: bool) -> dict:
        """
        Get the model state dict according to the specified options.

        Args:
            - cpu_offload (bool): Whether returned state_dict's value will be offloaded to CPU
                If true, will be copied to CPU memory, or just keep a reference to the original GPU tensor.
            - full_state_dict (bool): Whether to get the full state dict.

        Returns:
            - dict: The state dict of the FSDP wrapped model according to the specified options
        """
        state_dict = self._strategy.get_model_state_dict(
            self.model, cpu_offload, full_state_dict
        )
        return state_dict

    @contextmanager
    def _checkpoint_io_context(self) -> Generator[None, None, None]:
        """
        Temporarily onload checkpoint-related state to the active accelerator.

        DCP save/load calls may query the current model state dict before the actual
        serialization/deserialization happens. If the actor uses RLinf's manual
        offload path, FSDP flat parameters can be on CPU at that moment, which
        violates FSDP's device assumptions. This context ensures checkpoint IO
        always runs with parameters/optimizer on the expected device, then
        restores the previous offload state.
        """
        should_restore_weight_offload = self.is_weight_offloaded
        should_restore_optimizer_offload = self.is_optimizer_offloaded

        if should_restore_weight_offload:
            self.load_param_and_grad(self.device)
        if should_restore_optimizer_offload:
            self.load_optimizer(self.device)

        try:
            yield
        finally:
            if should_restore_weight_offload:
                self.offload_param_and_grad()
            if should_restore_optimizer_offload:
                self.offload_optimizer()

    def load_checkpoint(self, load_path: str) -> None:
        """
        Load checkpoint from local path.

        Args:
            load_path: the directory to load checkpoint.
        """
        with self._checkpoint_io_context():
            self._strategy.load_checkpoint(
                self.model, self.optimizer, self.lr_scheduler, load_path
            )

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """
        Save checkpoint to local path.
        Every rank will save its own model and optim shard.

        Args:
            save_path: the directory to save checkpoint.
        """
        with self._checkpoint_io_context():
            self._strategy.save_checkpoint(
                self.model,
                self.optimizer,
                self.lr_scheduler,
                save_path,
            )

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        """
        Offload FSDP parameters and gradients(options) to CPU.

        Args:
            offload_grad: whether to offload gradients.
        """
        self._strategy.offload_param_and_grad(self.model, offload_grad)
        self.is_weight_offloaded = True

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        """
        Load FSDP parameters and gradients(options) to the specified device.

        Args:
            device_id: the target device id to load parameters and gradients.
            load_grad: whether to load gradients.
        """
        self._strategy.onload_param_and_grad(self.model, device_id, load_grad)
        self.is_weight_offloaded = False

    def offload_optimizer(self) -> None:
        """
        Offload optimizer states to CPU.
        """
        self._strategy.offload_optimizer(self.optimizer)
        self.is_optimizer_offloaded = True

    def load_optimizer(self, device_id: int) -> None:
        """
        Load optimizer states to the specified device.

        Args:
            device_id: the target device id to load optimizer states.
        """
        self._strategy.onload_optimizer(self.optimizer, device_id)
        self.is_optimizer_offloaded = False

    def optimizer_step(self) -> tuple[float, list[float]]:
        """
        Perform optimizer step using its optimizer, lr_scheduler and grad_scaler.

        Returns:
            A tuple of (grad_norm, lr_list), lr_list contains learning rates for all param groups.
        """
        self.optimizer_steps += 1
        self.grad_scaler.unscale_(self.optimizer)
        grad_norm = self._strategy.clip_grad_norm_(
            model=self.model,
        )

        if not torch.isfinite(torch.as_tensor(grad_norm)):
            self._logger.warning(
                f"[FSDP] Non-finite grad norm {grad_norm} detected. Skipping optimizer step."
            )
        else:
            self.grad_scaler.step(optimizer=self.optimizer)

        self.grad_scaler.update()

        if self.critic_warmup_steps > 0:
            lr_list = [0.0 for _ in self.optimizer.param_groups]
            if self.optimizer_steps >= self.critic_warmup_steps:
                self.optimizer = self.build_optimizer(model=self.model)
                self.critic_warmup_steps = 0
        else:
            lr_list = [group["lr"] for group in self.optimizer.param_groups]

        return grad_norm, lr_list

    def build_lr_scheduler(
        self, optimizer: Optimizer, optim_config: DictConfig
    ) -> LRScheduler:
        """
        Build the learning rate scheduler based on the configuration.
        Currently only support LambdaLR scheduler with various warmup styles.

        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            optim_config (DictConfig): The optimizer config.

        Returns:
            LRScheduler: The learning rate scheduler.
        """
        total_steps = optim_config.get("total_training_steps", 0)
        num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
        lr_scheduler = optim_config.get("lr_scheduler", "constant")
        num_cycles = optim_config.get("num_cycles", 0.5)
        min_lr = optim_config.get("min_lr", 0.0)
        min_lr_rate = optim_config.get("min_lr_rate", None)
        if num_warmup_steps < 0:
            num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        return get_lr_scheduler(
            lr_scheduler=lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
            min_lr=min_lr,
            min_lr_rate=min_lr_rate,
        )

    def build_optimizer(
        self,
        model: Union[nn.Module, FSDPModule, FSDP],
        enable_critic_warmup: bool = False,
    ) -> Optimizer:
        """
        Build the optimizer based on the configuration, currently only support Adam optimizer.

        Args:
            model: The model to optimize, can be nn.Module, FSDPModule (used in FSDP2) or FSDP.
            enable_critic_warmup: Whether to enable critic warmup used for value network.

        Returns:
            Optimizer: The constructed optimizer.
        """
        betas = (self._cfg.optim.adam_beta1, self._cfg.optim.adam_beta2)
        adam_eps = self._cfg.optim.get("adam_eps", 1e-8)
        weight_decay = self._cfg.optim.get("weight_decay", 1e-2)

        params_actor = []
        params_critic = []

        if enable_critic_warmup:
            self._logger.info("[FSDP] Enable critic warmup for value head.")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.store_requires_grad_param_name.append(name)
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                        continue
                    param.requires_grad = False

        else:
            for name, param in model.named_parameters():
                if name in self.store_requires_grad_param_name:
                    param.requires_grad = True
                if param.requires_grad:
                    if "value_head" in name or "model.value_head" in name:
                        params_critic.append(param)
                    else:
                        params_actor.append(param)

        param_groups = []
        if len(params_actor) > 0:
            param_groups.append(
                {
                    "params": params_actor,
                    "lr": self._cfg.optim.lr,
                    "betas": betas,
                }
            )
        if len(params_critic) > 0:
            param_groups.append(
                {
                    "params": params_critic,
                    "lr": self._cfg.optim.value_lr,
                    "betas": betas,
                }
            )
        optimizer = torch.optim.AdamW(
            param_groups,
            eps=adam_eps,
            weight_decay=weight_decay,
        )

        # run optimizer empty step to initialize optimizer.state
        # to avoid KeyError during get_state_dict/set_state_dict
        # in save/load_checkpoint calls
        warmup_optimizer_state(optimizer)
        return optimizer

    def build_optimizers(
        self,
        model: Union[nn.Module, FSDPModule, FSDP],
        main_optim_config: DictConfig,
        param_filters: dict[str, list[str]],
        filtered_optim_config: dict[str, DictConfig],
    ):
        main_betas = (
            main_optim_config.get("adam_beta1", 0.9),
            main_optim_config.get("adam_beta2", 0.999),
        )
        main_adam_eps = main_optim_config.get("adam_eps", 1e-8)
        main_lr = main_optim_config.lr

        filtered_optim_batas_map = {}
        filtered_optim_adam_eps_map = {}
        filtered_optim_lr_map = {}
        for key, config in filtered_optim_config.items():
            filtered_optim_batas_map[key] = (
                config.get("adam_beta1", 0.9),
                config.get("adam_beta2", 0.999),
            )
            filtered_optim_adam_eps_map[key] = config.get("adam_eps", 1e-8)
            filtered_optim_lr_map[key] = config.lr

        main_optim_params = []

        filtered_params_dict = {}
        for key in param_filters.keys():
            filtered_params_dict[key] = []

        # ISSUE: currently the net weight still bind with the actor.
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            is_matched = False
            for key, filters in param_filters.items():
                for f in filters:
                    if f in name:
                        filtered_params_dict[key].append(param)
                        is_matched = True
                        break
                if is_matched:
                    break

            if is_matched:
                continue

            main_optim_params.append(param)

        assert len(main_optim_params) > 0
        main_optimizer = torch.optim.Adam(
            [
                {
                    "params": main_optim_params,
                    "lr": main_lr,
                    "betas": main_betas,
                    "eps": main_adam_eps,
                },
            ]
        )
        optimizers = [main_optimizer]

        for key, params in filtered_params_dict.items():
            assert len(params) > 0, (
                f"optimer {key=} is not match any params, with {param_filters(key)=}"
            )
        for key, params in filtered_params_dict.items():
            optimizers.append(
                torch.optim.Adam(
                    [
                        {
                            "params": params,
                            "lr": filtered_optim_lr_map[key],
                            "betas": filtered_optim_batas_map[key],
                            "eps": filtered_optim_adam_eps_map[key],
                        },
                    ]
                )
            )
        return optimizers

    def build_grad_scaler(self, enabled: bool) -> ShardedGradScaler:
        """
        Build the gradient scaler based on the configuration.

        Args:
            enabled (bool): Whether to enable gradient scaling.

        Returns:
            ShardedGradScaler: The gradient scaler.
        """
        return ShardedGradScaler(enabled=enabled)

    def before_micro_batch(
        self, model: Union[FSDP, FSDPModule], is_last_micro_batch: bool
    ) -> ContextManager:
        """
            Setup context manager before processing a micro-batch.
            This is used to control gradient synchronization behavior.
            Depending on the specific FSDP strategy being used, if using
            FSDP, it will return model.no_sync() for non-last micro-batches to
            avoid gradient synchronization, and nullcontext() for the last
            micro-batch to ensure gradients are synchronized and updated.
            If using FSDP2, it will set requires_gradient_sync flag
            on the model accordingly.

        Args:
            model: The FSDP or FSDPModule model.
            is_last_micro_batch: A boolean indicating if this is the last micro-batch.

        Returns:
            A context manager for the micro-batch processing.
        """
        return self._strategy.before_micro_batch(
            model=model, is_last_micro_batch=is_last_micro_batch
        )
