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

import gc
import itertools
from typing import TYPE_CHECKING, Iterator, Optional

import torch
from omegaconf import DictConfig

from rlinf.config import build_config, build_transformer_config
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.utils.flops import FLOPSCalculator, ModelConfig
from rlinf.utils.initialize import initialize_megatron, set_megatron_args
from rlinf.utils.logging import get_logger
from rlinf.utils.profiler import PyTorchProfiler, PyTorchProfilerFunc
from rlinf.utils.utils import clear_memory

from .utils import (
    postprocess_packed_seqs,
    preprocess_packed_seqs,
    recover_left_padding,
    remove_left_padding,
)

try:
    from megatron.core import parallel_state
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.enums import ModelType
    from megatron.core.model_parallel_config import ModelParallelConfig
    from megatron.core.models.gpt import GPTModel as MCoreGPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_decoder_block_spec,
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.optimizer import ChainedOptimizer
    from megatron.core.transformer.module import Float16Module as MCoreFloat16Module

    HAVE_MEGATRON_CORE = True

except (ImportError, ModuleNotFoundError):
    HAVE_MEGATRON_CORE = False
    raise "import error"
try:
    from megatron.legacy.model import Float16Module
except ImportError:
    from megatron.core.transformer.module import Float16Module
except ImportError:
    raise "Could not import Float16Module from megatron"
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.training import (
    get_args,
    preprocess_common_state_dict,
    setup_model_and_optimizer,
    unwrap_model,
)

try:
    import transformer_engine
    from transformer_engine.pytorch import module as te_module

    HAVE_TE = True
    HAVE_TE_MODULE = True
except ImportError:
    transformer_engine = None
    te_module = None
    HAVE_TE = False
    HAVE_TE_MODULE = False

HAVE_TE = HAVE_TE and HAVE_TE_MODULE

if TYPE_CHECKING:
    pass


def get_specs(spec_name, transformer_config=None, use_te=False):
    if use_te and spec_name == "":
        spec_name = "te_gpt"

    num_experts = transformer_config.num_moe_experts if transformer_config else None
    moe_grouped_gemm = (
        transformer_config.moe_grouped_gemm if transformer_config else False
    )

    name_spec_dict = {
        "decoder_gpt": get_gpt_decoder_block_spec(transformer_config, use_te),
        "local_gpt": get_gpt_layer_local_spec(num_experts, moe_grouped_gemm),
        "te_gpt": get_gpt_layer_with_transformer_engine_spec(
            num_experts, moe_grouped_gemm, qk_layernorm=transformer_config.qk_layernorm
        ),
    }
    if spec_name not in name_spec_dict:
        raise ValueError(f"Spec name '{spec_name}' is not recognized.")
    return name_spec_dict[spec_name]


class MegatronModelManager:
    """
    Megatron Model Manager for RL training
    """

    def __init__(self, cfg: DictConfig):
        if not HAVE_MEGATRON_CORE:
            raise "Megatron-core was not found. Please see the RLinf README for installation instructions."

        self.tokenizer = hf_tokenizer(cfg.tokenizer.tokenizer_model)

        initialize_megatron(cfg)

        self.transformer_config = build_transformer_config(cfg.model)

        self._cfg = cfg
        self.mcore_gpt = cfg.mcore_gpt
        self.spec_name = cfg.spec_name
        self.distributed_adam_offload_manager = None
        self._logger = get_logger()

        if torch.distributed.get_rank() == 0:
            self._logger.info(f"{self.transformer_config}")

        self.checkpoint_context = self._get_checkpoint_context()

        if self._cfg.megatron.use_hf_ckpt:
            self._cfg.megatron.load = self._cfg.megatron.ckpt_convertor.save_path

        config = build_config(ModelConfig, cfg.model)
        self.flops_calculator = FLOPSCalculator(config)

        # In AUTO mode, the actor will occupy all GPUs for initialization, but not all Megatron processes will be in the running state.
        self.is_running = True

    def setup_model_and_optimizer(self, model_type=ModelType.encoder_or_decoder):
        """Setup model and optimizer."""
        set_megatron_args(self._cfg)

        self.model, self.optimizer, self.lr_scheduler = setup_model_and_optimizer(
            model_provider_func=self.model_provider_func,
            model_type=model_type,
            checkpointing_context=self.checkpoint_context,
        )

    def model_provider_func(self, pre_process, post_process):
        """Model depends on pipeline paralellism."""
        use_te = HAVE_TE

        if self.mcore_gpt:
            model = MCoreGPTModel(
                config=self.transformer_config,
                transformer_layer_spec=get_specs(
                    self.spec_name,
                    self.transformer_config,
                    use_te,
                ),
                vocab_size=self._cfg.model.padded_vocab_size,
                max_sequence_length=self._cfg.model.max_position_embeddings,
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=True,
                share_embeddings_and_output_weights=self._cfg.model.share_embeddings_and_output_weights,
                position_embedding_type=self._cfg.model.position_embedding_type,
                rotary_percent=self._cfg.model.rotary_percentage,
                seq_len_interpolation_factor=self._cfg.model.seq_len_interpolation_factor,
                rotary_base=self._cfg.model.rotary_base,
            )

        else:
            from megatron.legacy.model.gpt_model import GPTModel

            config = build_config(ModelParallelConfig, self._cfg.model)
            setattr(config, "hidden_size", self._cfg.model.hidden_size)

            model = GPTModel(
                config=config,
                num_tokentypes=0,
                parallel_output=True,
                pre_process=pre_process,
                post_process=post_process,
            )
        return model

    def optimizer_step(self, increment):
        success, grad_norm, num_zeros_in_grad = self.optimizer.step()

        self.lr_scheduler.step(increment=increment)

        lr = self.optimizer.param_groups[0]["lr"]

        return success, grad_norm, num_zeros_in_grad, lr

    def padding_to_max(self, chain_iterator):
        microbatches = list(chain_iterator)
        max_batch_seqlen = 0

        for batch in microbatches:
            current_seqlen = 0
            if isinstance(batch, dict):
                seqlens_in_batch = batch["attention_mask"].sum(
                    dim=-1, dtype=torch.int32
                )
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                cp_size = parallel_state.get_context_parallel_world_size()
                align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
                pad_size = (align_size - seqlens_in_batch % align_size) % align_size
                seqlens_in_batch_padded = seqlens_in_batch + pad_size
                current_seqlen = seqlens_in_batch_padded.sum()
            elif isinstance(batch, list):
                item = batch[1]
                seqlens_in_batch = item.sum(dim=-1, dtype=torch.int32)
                tp_size = parallel_state.get_tensor_model_parallel_world_size()
                cp_size = parallel_state.get_context_parallel_world_size()
                align_size = tp_size * cp_size * 2 if cp_size > 1 else tp_size
                pad_size = (align_size - seqlens_in_batch % align_size) % align_size
                seqlens_in_batch_padded = seqlens_in_batch + pad_size
                current_seqlen = seqlens_in_batch_padded.sum()
            max_batch_seqlen = max(max_batch_seqlen, current_seqlen)

        for batch in microbatches:
            if isinstance(batch, dict):
                batch["max_batch_seqlen"] = max_batch_seqlen
            elif isinstance(batch, list):
                batch.append(max_batch_seqlen)

        return itertools.chain(microbatches)

    def make_data_iterator_list(
        self, data_iterator: Iterator, padding: bool = False, vpp_size: int = 1
    ) -> list[Iterator]:
        """
        Convert the data iterator into the format expected by Megatron.
        With interleaved pipeline parallelism, Megatron expects a
        list of one data iterator per model chunk.
        """
        if padding:
            data_iterator = self.padding_to_max(data_iterator)
        import copy

        if vpp_size > 1:
            batch_generator = batch_generator = [
                copy.deepcopy(data_iterator) for _ in range(vpp_size)
            ]  # number of vpp chunks
            batch_generator = [iter(b) for b in batch_generator]
        else:
            # no vpp
            batch_generator = iter(data_iterator)
        return batch_generator

    def _get_checkpoint_context(self):
        if self._cfg.megatron.non_persistent_ckpt_type == "local":
            try:
                from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import (  # type: ignore
                    LocalCheckpointManager,
                )
                from nvidia_resiliency_ext.checkpointing.local.replication.strategies import (  # type: ignore
                    CliqueReplicationStrategy,
                )
            except ModuleNotFoundError:
                raise RuntimeError(
                    "The 'nvidia_resiliency_ext' module is required for local "
                    "checkpointing but was not found. Please ensure it is installed."
                )

            if self._cfg.megatron.replication:
                repl_strategy = CliqueReplicationStrategy.from_replication_params(
                    self._cfg.megatron.replication_jump,
                    self._cfg.megatron.replication_factor,
                )
            else:
                repl_strategy = None

            checkpointing_context = {
                "local_checkpoint_manager": LocalCheckpointManager(
                    self._cfg.megatron.non_persistent_local_ckpt_dir,
                    repl_strategy=repl_strategy,
                )
            }
        else:
            checkpointing_context = {}
        return checkpointing_context

    def save_checkpoint(
        self,
        save_path: str,
        step: int,
        num_floating_point_operations_so_far: int = 0,
    ) -> None:
        if not self.is_running:
            return
        args = get_args()
        args.save = save_path
        save_checkpoint(
            iteration=step,
            model=self.model,
            optimizer=self.optimizer,
            opt_param_scheduler=self.lr_scheduler,
            num_floating_point_operations_so_far=num_floating_point_operations_so_far,
            checkpointing_context=self.checkpoint_context,
            preprocess_common_state_dict_fn=preprocess_common_state_dict,
        )

    def load_checkpoint(self, load_path):
        args = get_args()
        args.load = load_path
        load_checkpoint(
            self.model,
            self.optimizer,
            self.lr_scheduler,
            checkpointing_context=self.checkpoint_context,
        )

    def load_state_dict(self, state_dict, strict=True):
        if len(self.model) == 1:
            self.model[0].load_state_dict(state_dict, strict=strict)
        else:
            for i in range(len(self.model)):
                parallel_state.set_virtual_pipeline_model_parallel_rank(i)
                self.model[i].load_state_dict(state_dict["model%d" % i], strict=strict)

    def get_model_module_list(self):
        def extract_module(model):
            if isinstance(model, (DDP, MCoreFloat16Module, Float16Module)):
                return extract_module(model.module)
            else:
                return model

        if isinstance(self.model, list):
            return list(map(extract_module, self.model))
        else:
            return [extract_module(self.model)]

    @staticmethod
    def custom_forward(
        model,
        input_ids,
        attention_mask,
        position_ids,
        sequence_parallel,
        value_model=False,
        pack_seqs=True,
        logits_processor=None,
        logits_processor_args: Optional[dict] = None,
        temperature: float = 1.0,
        max_batch_seqlen: int = 4096,
    ):
        """Default forward pass for GPT models with optional sequence packing."""
        pre_process = unwrap_model(model).pre_process
        post_process = unwrap_model(model).post_process
        if pack_seqs:
            batch_size, seq_len = attention_mask.shape[:2]
            input_ids_rmpad, packed_seq_params = preprocess_packed_seqs(
                input_ids, attention_mask, pre_process=pre_process
            )
            input_ids_rmpad = input_ids_rmpad.contiguous()
            output_orig = model(
                input_ids=input_ids_rmpad,
                attention_mask=None,
                position_ids=position_ids,
                packed_seq_params=packed_seq_params,
            )
            output_orig /= temperature
            if post_process and logits_processor is not None:
                args = {
                    k: preprocess_packed_seqs(v, attention_mask, pre_process=True)[0]
                    for k, v in logits_processor_args.items()
                }
                output_dict = logits_processor(output_orig, **args)
                output = {
                    k: postprocess_packed_seqs(
                        v,
                        packed_seq_params,
                        attention_mask,
                        batch_size,
                        seq_len,
                        post_process=post_process,
                    )
                    for k, v in output_dict.items()
                }
            else:
                output = postprocess_packed_seqs(
                    output_orig,
                    packed_seq_params,
                    attention_mask,
                    batch_size,
                    seq_len,
                    post_process=post_process,
                )
        else:
            assert logits_processor is None, (
                "logits_processor is not supported for non-packed sequence"
            )
            batch_size, sequence_length = attention_mask.shape
            new_input_ids, new_attention_mask, new_position_ids = remove_left_padding(
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel,
                pre_process=pre_process,
            )
            output = model(
                input_ids=new_input_ids,
                attention_mask=new_attention_mask,
                position_ids=new_position_ids,
            )
            output = recover_left_padding(
                output,
                new_attention_mask,
                attention_mask,
                sequence_length,
                post_process=post_process,
            )
        if value_model and post_process:
            output = output[..., 0]
        return output

    def offload_model_weights_and_grad(self, offload_grad=True, offload_weight=True):
        for model_idx, model_chunk in enumerate(self.model):
            if isinstance(model_chunk, DDP):
                for buffer_idx, buffer in enumerate(model_chunk.buffers):
                    if (
                        offload_weight
                        and buffer.param_data.untyped_storage().size() > 0
                    ):
                        param_size = buffer.param_data.untyped_storage().size()

                        buffer.param_data.cpu_data = (
                            buffer.param_data.data.cpu().pin_memory()
                        )
                        buffer.param_data_size = param_size

                        buffer.param_data.untyped_storage().resize_(0)

                        assert (
                            buffer.param_data_size
                            == buffer.param_data.cpu_data.untyped_storage().size()
                        )

                    if offload_grad and buffer.grad_data.untyped_storage().size() > 0:
                        grad_size = buffer.grad_data.untyped_storage().size()
                        buffer.grad_data_size = grad_size
                        buffer.grad_data.untyped_storage().resize_(0)

            else:
                for param_name, param in model_chunk.named_parameters():
                    if offload_weight and param.data is not None:
                        param.data = param.data.to("cpu", non_blocking=True)

                    if offload_grad and param.grad is not None:
                        param.grad = param.grad.to("cpu", non_blocking=True)

        # clear memory
        clear_memory()

    def onload_model_weights_and_grad(self, load_grad=True):
        gc.collect()
        torch.cuda.empty_cache()
        for model_chunk in self.model:
            if isinstance(model_chunk, DDP):
                for buffer in model_chunk.buffers:
                    # sometimes, we don't want to load grad for pure inference
                    if load_grad and hasattr(buffer, "grad_data_size"):
                        buffer.grad_data.untyped_storage().resize_(
                            buffer.grad_data_size
                        )
                        buffer.grad_data.zero_()

                    if buffer.param_data.untyped_storage().size() == 0:
                        buffer.param_data.untyped_storage().resize_(
                            buffer.param_data_size
                        )
                        # copy data from cpu to cuda
                        buffer.param_data.copy_(
                            buffer.param_data.cpu_data, non_blocking=True
                        )
            else:
                device_id = torch.cuda.current_device()
                for _, param in model_chunk.named_parameters():
                    param.data = param.data.to(device_id, non_blocking=True)
                    if load_grad and param.grad is not None:
                        param.grad = param.grad.to(device_id, non_blocking=True)
        clear_memory()

    def offload_megatron_copy_params(self, optimizers):
        """
        Offload optimizer parameters to CPU. Supports both Megatron optimizers
        and `ChainedOptimizer`, which wraps a list of underlying optimizers.

        Args:
            optimizers: The optimizer or ChainedOptimizer instance.
        """

        def _iter_opts(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        def offload_tensor_to_cpu(tensor):
            if tensor is None:
                return
            tensor.data = tensor.data.to("cpu", non_blocking=True)

        def offload_group_to_cpu(group):
            if group is None:
                return

            if isinstance(group, list):
                for param_group in group:
                    if isinstance(param_group, list):
                        for param in param_group:
                            offload_tensor_to_cpu(param)
                    else:
                        offload_tensor_to_cpu(param_group)
            else:
                offload_tensor_to_cpu(group)

        # Offload all parameter groups to CPU for each underlying optimizer

        for _opt in _iter_opts(optimizers):
            if hasattr(_opt, "shard_fp32_from_float16_groups"):
                offload_group_to_cpu(_opt.shard_fp32_from_float16_groups)

    def load_megatron_copy_params(self, optimizers):
        """
        Load optimizer parameters back to GPU. Handles ChainedOptimizer.

        Args:
            optimizers: Optimizer or ChainedOptimizer instance.
        """

        def _iter_opts(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        def load_tensor_to_gpu(tensor):
            if tensor is None:
                return
            device_id = torch.cuda.current_device()
            tensor.data = tensor.data.to(device_id, non_blocking=True)

        def load_group_to_gpu(group):
            if group is None:
                return

            if isinstance(group, list):
                for param_group in group:
                    if isinstance(param_group, list):
                        for param in param_group:
                            load_tensor_to_gpu(param)
                    else:
                        load_tensor_to_gpu(param_group)
            else:
                load_tensor_to_gpu(group)

        # Load all parameter groups to GPU for each underlying optimizer

        for _opt in _iter_opts(optimizers):
            if hasattr(_opt, "shard_fp32_from_float16_groups"):
                load_group_to_gpu(_opt.shard_fp32_from_float16_groups)

    def offload_megatron_optimizer(self):
        def _iter_opts(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        for _opt in _iter_opts(self.optimizer):
            self.offload_megatron_copy_params(_opt)
            for v in _opt.optimizer.state.values():
                # Offloading through resetting the storage size can ensure that the tensor can be offloaded correctly even when it has tensor views.
                if "exp_avg" in v and v["exp_avg"].is_cuda:
                    buffer = v["exp_avg"]
                    buffer.cpu_data = buffer.data.cpu().pin_memory()
                    buffer.storage().resize_(0)
                if "exp_avg_sq" in v and v["exp_avg_sq"].is_cuda:
                    buffer = v["exp_avg_sq"]
                    buffer.cpu_data = buffer.data.cpu().pin_memory()
                    buffer.storage().resize_(0)
        clear_memory()

    def onload_megatron_optimizer(self):
        def _iter_opts(opt):
            if isinstance(opt, ChainedOptimizer):
                return opt.chained_optimizers
            return [opt]

        for _opt in _iter_opts(self.optimizer):
            self.load_megatron_copy_params(_opt)
            for v in _opt.optimizer.state.values():
                if "exp_avg" in v:
                    v["exp_avg"].data = v["exp_avg"].cpu_data.to(
                        torch.cuda.current_device(), non_blocking=True
                    )
                if "exp_avg_sq" in v:
                    v["exp_avg_sq"].data = v["exp_avg_sq"].cpu_data.to(
                        torch.cuda.current_device(), non_blocking=True
                    )
        clear_memory()

    def init_profiler(self):
        # here we should validate profiler's schedule info
        assert (
            self._cfg.megatron.profiler.schedule_warmup is not None
            and self._cfg.megatron.profiler.schedule_warmup >= 0
        ), "<schedule_warmup> must be set and greater than 0 when using profiler."
        assert (
            self._cfg.megatron.profiler.schedule_active is not None
            and self._cfg.megatron.profiler.schedule_active > 0
        ), "<schedule_active> must be set and greater than 0 when using profiler."

        self.profiler = PyTorchProfiler.from_config(self._cfg.megatron.profiler)

        self.forward_only_record = PyTorchProfilerFunc("forward_only")
        self.dynamic_batch_processing_record = PyTorchProfilerFunc(
            "dynamic_batch_processing"
        )
        self.static_batch_processing_record = PyTorchProfilerFunc(
            "static_batch_processing"
        )
        self.broadcast_outputs_record = PyTorchProfilerFunc("broadcast_outputs")
        self.megatron_forward_backward_record = PyTorchProfilerFunc(
            "megatron_forward_backward"
        )
