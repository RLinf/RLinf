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
import time
import subprocess
from contextlib import contextmanager
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, RolloutResult
from rlinf.data.lerobot_paths import resolve_lerobot_repo_id
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.data_iter_utils import (
    get_iterator_k_split,
    get_reverse_idx,
    get_seqlen_balanced_partitions,
    split_dynamic_batch_size,
)
from rlinf.utils.distributed import (
    RolloutDataBalance,
    all_reduce_dict,
    all_reduce_int,
    masked_normalization,
)
from rlinf.utils.distributed import (
    compute_rollout_metrics as compute_math_rollout_metrics,
)
from rlinf.utils.metric_utils import (
    append_to_dict,
    compute_loss_mask,
    compute_rollout_metrics,
    compute_split_num,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
    ModelParallelComponentPlacement,
)
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
)
from rlinf.workers.rollout.utils import RankMapper


def process_nested_dict_for_adv(nested_dict, rollout_epoch):
    """
    original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
    target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
    """
    ret_dict = {}
    for key, value in nested_dict.items():
        if isinstance(value, torch.Tensor):
            new_value = value.reshape(
                rollout_epoch, -1, *value.shape[1:]
            )  # [rollout_epoch, n_chunk_step, bsz, ...]
            new_value = new_value.transpose(
                0, 1
            )  # [n_chunk_step, rollout_epoch, bsz, ...]
            new_value = new_value.reshape(new_value.shape[0], -1, *new_value.shape[3:])
            ret_dict[key] = new_value
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_adv(value, rollout_epoch)
    return ret_dict


def process_nested_dict_for_train(nested_dict, shuffle_id):
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            value = value[:-1]
        if "env_info" in key:
            raise NotImplementedError
        if value is None:
            ret_dict[key] = None
        if isinstance(value, torch.Tensor):
            ret_dict[key] = value.reshape(-1, *value.shape[2:])[shuffle_id]
        elif isinstance(value, dict):
            ret_dict[key] = process_nested_dict_for_train(value, shuffle_id)
    return ret_dict


def compute_rollout_train_kl(
    m_batch: dict, loss_mask: torch.Tensor
) -> Optional[torch.Tensor]:
    """
    Compute the masked mean of absolute difference between rollout and training logprobs.

    Args:
        m_batch: Dictionary containing 'rollout_logprobs' and 'recomputed_logprobs'.
        loss_mask: Mask tensor for computing weighted mean.

    Returns:
        Masked mean of abs(recomputed_logprobs - rollout_logprobs), or None if keys are missing.
    """
    if "rollout_logprobs" not in m_batch or "recomputed_logprobs" not in m_batch:
        return None
    rollout_logprobs = m_batch["rollout_logprobs"]
    recomputed_logprobs = m_batch["recomputed_logprobs"]
    kl = torch.abs(recomputed_logprobs - rollout_logprobs)
    return masked_mean(kl, loss_mask)


def _get_device_type():
    if hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _get_device_api(device_type):
    if device_type == "npu":
        return torch.npu
    if device_type == "cuda":
        return torch.cuda
    return None


def _gib(x):
    return x / 1024**3


def _sync_device(device_type):
    api = _get_device_api(device_type)
    if api is not None:
        try:
            api.synchronize()
        except Exception:
            pass


def _reset_peak_mem(device_type):
    api = _get_device_api(device_type)
    if api is not None:
        try:
            api.reset_peak_memory_stats()
        except Exception:
            pass


def _mem_report(tag, rank=None, log_all_ranks=False, sync=True):
    device_type = _get_device_type()
    api = _get_device_api(device_type)

    if rank is None:
        rank = int(os.environ.get("RANK", "0"))

    if not log_all_ranks and rank != 0:
        return

    if sync:
        _sync_device(device_type)

    if api is None:
        print(f"[BWD_MEM][rank={rank}] {tag}: no cuda/npu memory api", flush=True)
        return

    def safe_call(name):
        try:
            return getattr(api, name)()
        except Exception:
            return -1

    alloc = safe_call("memory_allocated")
    reserved = safe_call("memory_reserved")
    max_alloc = safe_call("max_memory_allocated")
    max_reserved = safe_call("max_memory_reserved")

    print(
        f"[BWD_MEM][rank={rank}] {tag}: "
        f"alloc={_gib(alloc):.3f} GiB, "
        f"reserved={_gib(reserved):.3f} GiB, "
        f"max_alloc={_gib(max_alloc):.3f} GiB, "
        f"max_reserved={_gib(max_reserved):.3f} GiB",
        flush=True,
    )


def _get_submodule_by_path(model, path):
    cur = model
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _find_transformer_layers(model):
    """
    尽量兼容：
      - AutoModel: model.layers
      - AutoModelForCausalLM: model.model.layers
      - Qwen/LLaMA 类结构
      - 某些 wrapper: module.model.layers / base_model.model.layers
    """
    candidate_paths = [
        "layers",
        "h",
        "blocks",
        "model.layers",
        "model.h",
        "model.blocks",
        "model.model.layers",
        "base_model.layers",
        "base_model.model.layers",
        "module.layers",
        "module.model.layers",
        "transformer.h",
        "transformer.blocks",
    ]

    for path in candidate_paths:
        obj = _get_submodule_by_path(model, path)
        if obj is not None and isinstance(obj, torch.nn.ModuleList):
            return path, obj

    # fallback：找名字像 decoder layer 的模块
    rows = []
    for name, mod in model.named_modules():
        cls = mod.__class__.__name__.lower()
        if (
            "decoderlayer" in cls
            or "transformerlayer" in cls
            or "qwen" in cls and "layer" in cls
            or "llama" in cls and "layer" in cls
        ):
            rows.append((name, mod))

    if rows:
        return "fallback_named_modules", rows

    raise RuntimeError("Cannot find transformer layers in self.model")


def install_backward_memory_hooks(
    model,
    *,
    hook_every=1,
    log_all_ranks=False,
    reset_peak_before_backward=True,
    include_param_grad_hooks=False,
    max_param_hooks_per_layer=2,
):
    """
    返回 handles，训练结束后需要 remove。

    hook_every:
      每隔多少层打一次 hook。想精确定位就设 1。
      层很多、日志太多时可以设 4/8。

    include_param_grad_hooks:
      是否额外给每层前几个参数挂 Tensor grad hook。
      这个日志会更多，但可以看到 grad 具体在哪个参数附近 ready。
    """
    rank = int(os.environ.get("RANK", "0"))
    handles = []

    if reset_peak_before_backward:
        _reset_peak_mem(_get_device_type())

    layer_path, layers = _find_transformer_layers(model)

    if rank == 0 or log_all_ranks:
        if isinstance(layers, torch.nn.ModuleList):
            print(
                f"[BWD_HOOK][rank={rank}] found layers at {layer_path}, "
                f"num_layers={len(layers)}, hook_every={hook_every}",
                flush=True,
            )
        else:
            print(
                f"[BWD_HOOK][rank={rank}] found fallback layers, "
                f"num_layers={len(layers)}, hook_every={hook_every}",
                flush=True,
            )

    def need_hook(i):
        return hook_every > 0 and (i % hook_every == 0)

    def make_bwd_pre(name):
        def hook(module, grad_output):
            _mem_report(f"BWD_PRE  {name}", rank=rank, log_all_ranks=log_all_ranks)
        return hook

    def make_bwd_post(name):
        def hook(module, grad_input, grad_output):
            _mem_report(f"BWD_POST {name}", rank=rank, log_all_ranks=log_all_ranks)
        return hook

    def make_param_grad_hook(name):
        def hook(grad):
            _mem_report(
                f"PARAM_GRAD_READY {name} "
                f"grad_shape={tuple(grad.shape)} grad_dtype={grad.dtype} grad_device={grad.device}",
                rank=rank,
                log_all_ranks=log_all_ranks,
            )
            return grad
        return hook

    if isinstance(layers, torch.nn.ModuleList):
        iterable = [(f"layer.{i}", layer) for i, layer in enumerate(layers)]
    else:
        iterable = [(name, mod) for name, mod in layers]

    for idx, (name, layer) in enumerate(iterable):
        if not need_hook(idx):
            continue

        try:
            handles.append(layer.register_full_backward_pre_hook(make_bwd_pre(name)))
        except Exception as e:
            if rank == 0 or log_all_ranks:
                print(f"[BWD_HOOK][rank={rank}] failed pre hook {name}: {e}", flush=True)

        try:
            handles.append(layer.register_full_backward_hook(make_bwd_post(name)))
        except Exception as e:
            if rank == 0 or log_all_ranks:
                print(f"[BWD_HOOK][rank={rank}] failed post hook {name}: {e}", flush=True)

        if include_param_grad_hooks:
            n = 0
            for pname, p in layer.named_parameters(recurse=True):
                if not p.requires_grad:
                    continue
                if n >= max_param_hooks_per_layer:
                    break
                try:
                    handles.append(
                        p.register_hook(make_param_grad_hook(f"{name}.{pname}"))
                    )
                    n += 1
                except Exception as e:
                    if rank == 0 or log_all_ranks:
                        print(
                            f"[BWD_HOOK][rank={rank}] failed param grad hook "
                            f"{name}.{pname}: {e}",
                            flush=True,
                        )

    if rank == 0 or log_all_ranks:
        print(
            f"[BWD_HOOK][rank={rank}] installed num_handles={len(handles)}",
            flush=True,
        )

    return handles


def remove_hooks(handles):
    for h in handles:
        try:
            h.remove()
        except Exception:
            pass


class FSDPActor(FSDPModelManager, Worker):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        """
        FSDPActor worker used to train the model with data from rollout workers.

        Args:
            cfg (DictConfig): The global yaml configuration.
            placement (ModelParallelComponentPlacement): The accelerator placement for actor worker.
        """
        if cfg_fsdp is None:
            cfg_fsdp = cfg.actor
        Worker.__init__(self)
        super().__init__(cfg_fsdp, self._world_size, self._rank)

        self.cfg = cfg

        self.response_len = (
            cfg.actor.model.encoder_seq_length - cfg.data.max_prompt_length
        )
        self.calculate_entropy = cfg.algorithm.calculate_entropy
        self.calculate_entropy_loss = (
            cfg.algorithm.entropy_bonus > 0 and self.calculate_entropy
        )
        self.kl_beta = cfg.algorithm.kl_beta
        self.kl_penalty_type = cfg.algorithm.kl_penalty_type
        self.reinpp_kl_beta = cfg.algorithm.get("reinpp_kl_beta", 0.0)
        self.combine_reference_model = cfg.actor.get("combine_reference_model", True)

        self.total_batch_size_per_dp = (
            cfg.data.rollout_batch_size * cfg.algorithm.group_size // self._world_size
        )

        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = placement
        self.is_pipeline = self._component_placement.is_disaggregated
        self.ref_policy_state_dict = None
        if self.is_pipeline:
            self._inference_group_name = cfg.inference.group_name
            self._inference_world_size = self._component_placement.get_world_size(
                "inference"
            )
            self._inference_dst_map: dict[int, list[str]] = {}
        else:
            self._inference_group_name = None
            self._inference_world_size = 0
            self._inference_dst_map = None
        self.loss_agg_func = get_loss_agg_func(cfg.algorithm.loss_agg_func)
        self.enable_offload = not self.is_pipeline and cfg.actor.get(
            "enable_offload", False
        )
        self.micro_batch_size = cfg.actor.micro_batch_size
        self.n_mini_batches = cfg.algorithm.n_minibatches
        self.task_type = cfg.runner.task_type
        self.entropy_op_type = cfg.algorithm.get("entropy_op_type", "flash_attn")
        self.enable_dp_load_balance = cfg.actor.get("enable_dp_load_balance", False)
        self.lr_sched_sync_with_optim = cfg.actor.get("lr_sched_sync_with_optim", True)
        self.enable_dynamic_batch_size = cfg.runner.get(
            "enable_dynamic_batch_size", False
        )
        if self.is_pipeline:
            assert not self.enable_dp_load_balance, (
                "DP load balance is not supported in pipeline mode."
            )
            assert not self.enable_dynamic_batch_size, (
                "Dynamic batch size is not supported in pipeline mode."
            )
        self.max_tokens_per_mbs = cfg.runner.get("max_tokens_per_mbs", 2048)
        self.variable_seq_lengths = self.cfg.actor.model.get(
            "variable_seq_lengths", False
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend
        (FSDP/FSDP2) to wrap it. If needed, offload model parameters and optimizer states to CPU.
        If kl_beta > 0, retrieve the reference policy model state dict to CPU.
        If mode is disaggregated, setup which inference ranks it needs to sync weights to by
        doing a handshake with inference workers.
        """
        self.setup_model_and_optimizer()
        if (
            self.kl_beta > 0 or self.reinpp_kl_beta > 0
        ) and self.combine_reference_model:
            self.ref_policy_state_dict = self.get_model_state_dict(
                cpu_offload=True,
                full_state_dict=True,
            )
            self.offload_model_buffer = {}

        if self.enable_offload and not self.is_pipeline:
            self.offload_param_and_grad()
            self.offload_optimizer()
        self._setup_rollout_weight_dst_ranks()

    def _setup_rollout_weight_dst_ranks(self) -> None:
        """Setup destination ranks for token and weight communication."""
        rank_map = RankMapper.get_actor_rank_to_rollout_rank_map(
            self._component_placement
        )
        self._weight_dst_rank_in_rollout = rank_map[self._rank]
        self.log_info(
            f"Actor rank {self._rank} will send weights to {self._weight_dst_rank_in_rollout}"
        )

    def del_reshard_state_dict(self) -> None:
        """Just for interface compatibility with MegatronActor."""
        pass

    def sync_model_to_inference(self) -> None:
        """
        Sync the model's full state dict to the inference worker.
        The model state_dict is the reference of actor's model
        parameters(by setting cpu_offload=False).
        """
        if not self._inference_dst_map:
            self._strategy.setup_actor_sync_inference_ranks(self)

        if self.enable_offload and not self.is_optimizer_offloaded:
            self.offload_optimizer()

        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device, False)

        inference_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        # NOTE: we have already know which inference rank needs which params
        # by calling _strategy.setup_actor_sync_inference_ranks() to do handshake
        # with each inference rank. just send them accordingly.
        for rank, needed_params in self._inference_dst_map.items():
            sended_params = {}
            for name in needed_params:
                if name in inference_state_dict:
                    # mentioned again, no ShardedTensor here.
                    sended_params[name] = (
                        inference_state_dict[name].to_local()
                        if isinstance(inference_state_dict[name], DTensor)
                        else inference_state_dict[name]
                    )
            self.send(
                object=sended_params,
                dst_group_name=self._inference_group_name,
                dst_rank=rank,
                async_op=True,
            )

        if self.enable_offload and not self.is_weight_offloaded:
            self.offload_param_and_grad()

        torch.distributed.barrier()

    @Worker.timer("actor/sync_model_to_rollout")
    def sync_model_to_rollout(self):
        """
        Sync the model's full state dict to the rollout worker.
        """
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        rollout_dtype = None
        if self._cfg.get("sync_precision", None) is not None:
            rollout_dtype = torch_dtype_from_precision(self._cfg.sync_precision)

        rollout_state_dict = self.get_model_state_dict(
            cpu_offload=False, full_state_dict=False
        )
        has_visual = any("visual." in k for k in rollout_state_dict.keys())
        model_bucket_list = self.divide_model_to_bucket(rollout_state_dict, has_visual)
        del rollout_state_dict
        send_handles = []
        buffer = {}
        for bucket_idx, model_bucket in enumerate(model_bucket_list):
            for k, v in model_bucket.items():
                if isinstance(v, DTensor):
                    v = v.full_tensor()
                if rollout_dtype is not None:
                    v = v.to(rollout_dtype)
                if not self.is_pipeline:
                    # TODO: nv support
                    v = v.detach().to('cpu')
                    v = reduce_tensor(v)
                buffer[k] = v
            if bucket_idx == 0:
                buffer["bucket_length"] = len(model_bucket_list)

            for send_handle in send_handles:
                send_handle.wait()
            send_handles = []

            if not self.is_pipeline:
                send_handle = self.send(
                    buffer,
                    self._rollout_group_name,
                    self._weight_dst_rank_in_rollout,
                    async_op=True,
                )
                send_handles.append(send_handle)
            else:
                for rank in self._weight_dst_rank_in_rollout:
                    send_handle = self.send(
                        buffer,
                        self._rollout_group_name,
                        rank,
                        async_op=True,
                    )
                    send_handles.append(send_handle)
            buffer = {}

        for send_handle in send_handles:
            send_handle.wait()

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad()

        clear_memory(sync=False)

    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], RolloutResult]:
        result: RolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.data.max_prompt_length,
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def get_dynamic_batch_as_much(
        self,
        input_channel: Channel,
        min_result_len: int,
        max_result_len: int,
        cliped_results=[],
        unfinished_result=None,
    ):
        assert not input_channel.is_local
        rollout_results = cliped_results
        # get min_result_len
        while len(rollout_results) < min_result_len:
            if unfinished_result is not None:
                rollout_result: RolloutResult = unfinished_result.wait()
                unfinished_result = None
            else:
                rollout_result: RolloutResult = input_channel.get()
            rollout_results.append(rollout_result)

        # try to get result as much
        # get result in every 0.1s and do all reduce to get the min result between dp (result_len)
        # stop at: the min result between dp (result_len) is same as the last min result
        last_result_len = 0
        result_len = len(rollout_results)
        time_until = time.time() + 0.1
        while last_result_len < result_len:
            if len(rollout_results) < max_result_len:
                if unfinished_result is None:
                    unfinished_result = input_channel.get(async_op=True)
                else:
                    time.sleep(0.001)
                if unfinished_result.done():
                    rollout_results.append(unfinished_result.wait())
                    unfinished_result = None
                if time.time() >= time_until:
                    last_result_len = result_len
                    result_len = all_reduce_int(len(rollout_results))
                    if last_result_len < result_len:
                        time_until = time.time() + 0.1
            else:
                last_result_len = result_len
                result_len = all_reduce_int(len(rollout_results))

        cliped_results = list(rollout_results[result_len:])
        rollout_results = rollout_results[:result_len]

        batches = []
        for rollout_result in rollout_results:
            batch = rollout_result.to_actor_batch(
                self.cfg.data.max_prompt_length,
                self.cfg.actor.model.encoder_seq_length,
                self.tokenizer.eos_token_id,
            )
            batches.append(batch)

        batch = RolloutResult.merge_batches(batches)
        rollout_result = RolloutResult.merge_result_list(rollout_results)
        return batch, rollout_result, result_len, cliped_results, unfinished_result

    @staticmethod
    def _split_to_micro_batch(
        batch,
        enable_dynamic_batch_size: bool,
        *,
        max_tokens_per_mbs: Optional[int] = None,
        split_num,
    ):
        if enable_dynamic_batch_size:
            (
                micro_batches_iter,
                _,
                micro_batch_cnt,
                dbs_indices,
            ) = split_dynamic_batch_size(
                batch=batch,
                cp_world_size=1,
                vpp_world_size=1,
                max_tokens_per_mbs=max_tokens_per_mbs,
                microbatch_group_size_per_vp_stage=1,
            )
        else:
            micro_batch_cnt = split_num
            micro_batches_iter = get_iterator_k_split(batch, micro_batch_cnt)
            dbs_indices = None
        return micro_batches_iter, micro_batch_cnt, dbs_indices

    def _load_weight_and_optimizer(self) -> None:
        # Acquire the GPUs to ensure that no one is using them before loading models
        # Otherwise, it may lead to OOM
        with self.device_lock:
            if not self.enable_offload:
                return
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

    @contextmanager
    def _swap_to_ref_policy(self):
        """Temporarily swap the actor weights to the reference-policy weights.

        FSDP/FSDP2 models need full-state-dict semantics here; local/sharded state dicts
        cannot be round-tripped through ``load_state_dict`` for reference-logprob passes.
        """
        assert self.ref_policy_state_dict is not None, (
            "Reference policy state dict is None but reference swap is requested"
        )

        current_policy_state_dict = self.get_model_state_dict(
            cpu_offload=True,
            full_state_dict=True,
        )
        self._strategy.load_model_with_state_dict(
            self.model,
            self.ref_policy_state_dict,
            cpu_offload=False,
            full_state_dict=True,
        )

        try:
            yield
        finally:
            self._strategy.load_model_with_state_dict(
                self.model,
                current_policy_state_dict,
                cpu_offload=False,
                full_state_dict=True,
            )

    def compute_logprobs(self, logits, target):
        return compute_logprobs_from_logits(
            logits,
            target,
            op_type=self.entropy_op_type,
        )

    def forward_batch(
        self, m_batch: dict[str, torch.Tensor], calculate_entropy: bool = False
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch.keys():
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = self.cfg.actor.model.encoder_seq_length
            max_prompt_len = self.cfg.data.max_prompt_length
            max_response_len = max_seq_len_unpack - max_prompt_len
            idx_starts, idx_ends = prepare_pack_fsdp(m_batch, max_prompt_len)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_to_fixed_len=not self.variable_seq_lengths,
            )

        with self.amp_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
                **multi_modal_inputs,
            )

        logits: torch.Tensor = outputs.logits

        logits.div_(self.cfg.algorithm.sampling_params.temperature)

        if self.enable_dynamic_batch_size or self.variable_seq_lengths:
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
            logprobs = logprobs[:, -max_response_len:]
        else:
            # (bsz, response_length, vocab_size)
            logits = logits[:, -self.response_len - 1 : -1, :]
            responses = input_ids[:, -self.response_len :]
            logprobs = self.compute_logprobs(logits, responses)

        if calculate_entropy:
            entropy = compute_entropy_from_logits(logits)

            if self.enable_dynamic_batch_size or self.variable_seq_lengths:
                entropy = unpack_sequences(
                    entropy, idx_starts, idx_ends, max_seq_len_unpack, pad_val=0
                )[:, -self.response_len :]

            return logprobs, entropy

        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        num_sequences: int,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=num_sequences
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        recomputed_logprobs, ref_logprobs = None, None

        # Recompute logprobs
        recomputed_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == recomputed_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {recomputed_logprobs.size(0)}"
            )
            recomputed_logprobs = recomputed_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with self._swap_to_ref_policy():
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return recomputed_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
        do_offload=False,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.

        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
            do_offload: Whether offload weights after inference is done
        """
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        inference_split = self.cfg.actor.get("inference_split", None)
        if inference_split is None:
            if not self.is_pipeline:
                inference_split = 1
            else:
                inference_split = self.cfg.algorithm.n_minibatches
        assert self.total_batch_size_per_dp % inference_split == 0, (
            f"FSDPActor: total_batch_size_per_dp[{self.total_batch_size_per_dp}] should be divisible by inference_split[{inference_split}]"
        )

        min_result_len = 1
        max_result_len = (
            self.cfg.data.rollout_batch_size // self._world_size // inference_split
        )
        if not self.is_pipeline:
            min_result_len = max_result_len
            coll_rollout_results = []
        total_result_len = 0
        total_result_len_per_dp = self.cfg.data.rollout_batch_size // self._world_size
        cliped_results, unfinished_result = [], None
        while total_result_len < total_result_len_per_dp:
            batch, rollout_result, result_len, cliped_results, unfinished_result = (
                self.get_dynamic_batch_as_much(
                    input_channel,
                    min(min_result_len, total_result_len_per_dp - total_result_len),
                    min(max_result_len, total_result_len_per_dp - total_result_len),
                    cliped_results,
                    unfinished_result,
                )
            )
            total_result_len += result_len
            self.log_debug(
                f"[dynamic inference rank-{self._rank}] inference result_len={result_len}, total_result_len={total_result_len}/{total_result_len_per_dp}"
            )
            self._load_weight_and_optimizer()
            self.model.eval()

            with self.worker_timer():
                with torch.no_grad():
                    recomputed_logprobs, ref_logprobs = self.inference_step(
                        batch, rollout_result.num_sequence, compute_ref_logprobs
                    )

                rollout_result.recomputed_logprobs = recomputed_logprobs

                # Ref logprobs
                if compute_ref_logprobs:
                    rollout_result.ref_logprobs = ref_logprobs

            if self.is_pipeline:
                # for pipeline mode, send after inference to reduce latency.
                # should do split to ensure actor won't get too much batches.
                split_results = RolloutResult.split_results(rollout_result, result_len)
                for split_result in split_results:
                    output_channel.put(split_result, async_op=True)
            else:
                coll_rollout_results.append(rollout_result)

        if not self.is_pipeline:
            # for coll mode, merge results to reduce send time.
            rollout_result = RolloutResult.merge_result_list(coll_rollout_results)
            split_results = RolloutResult.split_results(
                rollout_result,
                min(total_result_len, self.cfg.algorithm.n_minibatches),
            )
            for split_result in split_results:
                output_channel.put(split_result)
        assert total_result_len == total_result_len_per_dp, (
            f"Expected {total_result_len_per_dp} sequences from channel, but got {total_result_len}"
        )
        
    def _print_optimizer_state_summary(self, tag):
        if not self._should_print_mem():
            return

        rank = self._debug_mem_rank()
        total_numel = 0
        total_bytes = 0
        dtype_count = {}

        try:
            states = self.optimizer.state
        except Exception as e:
            print(f"[OPT_STATE][rank={rank}][{tag}] cannot access optimizer.state: {e}", flush=True)
            return

        n_state_tensors = 0

        for _, state in states.items():
            if not isinstance(state, dict):
                continue
            for k, v in state.items():
                if torch.is_tensor(v):
                    n_state_tensors += 1
                    numel = v.numel()
                    elem_size = v.element_size()
                    total_numel += numel
                    total_bytes += numel * elem_size
                    key = str(v.dtype)
                    dtype_count[key] = dtype_count.get(key, 0) + numel

        print(
            f"[OPT_STATE][rank={rank}][{tag}] "
            f"num_state_entries={len(states)}, "
            f"n_state_tensors={n_state_tensors}, "
            f"total_numel={total_numel:,}, "
            f"total_bytes={total_bytes / 1024**3:.2f} GiB, "
            f"dtype_numel={dtype_count}",
            flush=True,
        )
        
    def _print_batch_summary(self, batch, tag):
        if not self._should_print_mem():
            return

        rank = self._debug_mem_rank()

        def walk(x, prefix=""):
            if torch.is_tensor(x):
                print(
                    f"[BATCH][rank={rank}][{tag}] "
                    f"{prefix}: shape={tuple(x.shape)}, "
                    f"dtype={x.dtype}, device={x.device}, "
                    f"bytes={x.numel() * x.element_size() / 1024**2:.2f} MiB",
                    flush=True,
                )
            elif isinstance(x, dict):
                for k, v in x.items():
                    walk(v, f"{prefix}.{k}" if prefix else str(k))
            elif isinstance(x, (list, tuple)):
                for i, v in enumerate(x):
                    walk(v, f"{prefix}[{i}]")

        walk(batch)

    def _tensor_local_numel_and_bytes(self, p):
        """
        兼容普通 Tensor / DTensor。
        返回:
        logical_numel, local_numel, logical_bytes, local_bytes, dtype_str, device_str
        """
        try:
            logical_numel = p.numel()
        except Exception:
            logical_numel = 0

        try:
            elem_size = p.element_size()
        except Exception:
            elem_size = 0

        local_numel = logical_numel
        local_elem_size = elem_size

        # DTensor 通常有 to_local()
        try:
            if hasattr(p, "to_local"):
                lp = p.to_local()
                local_numel = lp.numel()
                local_elem_size = lp.element_size()
        except Exception:
            pass

        logical_bytes = logical_numel * elem_size
        local_bytes = local_numel * local_elem_size

        try:
            dtype_str = str(p.dtype)
        except Exception:
            dtype_str = "unknown"

        try:
            device_str = str(p.device)
        except Exception:
            device_str = "unknown"

        return (
            logical_numel,
            local_numel,
            logical_bytes,
            local_bytes,
            dtype_str,
            device_str,
        )

    def _print_fsdp_unit_sizes(self, max_units=200):
        if not self._should_print_mem():
            return

        rank = self._debug_mem_rank()

        rows = []

        for name, module in self.model.named_modules():
            cls_name = module.__class__.__name__

            # 适配你现在看到的 FSDPQwen2DecoderLayer / FSDPEmbedding 等
            if "FSDP" not in cls_name and "FullySharded" not in cls_name:
                continue

            logical_numel = 0
            local_numel = 0
            logical_bytes = 0
            local_bytes = 0
            dtype_counter = {}
            device_counter = {}
            n_params = 0

            # root module 会递归统计全模型；也可以保留，但解释时要注意
            for p_name, p in module.named_parameters(recurse=True):
                n_params += 1
                (
                    ln,
                    locn,
                    lb,
                    locb,
                    dtype_str,
                    device_str,
                ) = self._tensor_local_numel_and_bytes(p)

                logical_numel += ln
                local_numel += locn
                logical_bytes += lb
                local_bytes += locb
                dtype_counter[dtype_str] = dtype_counter.get(dtype_str, 0) + ln
                device_counter[device_str] = device_counter.get(device_str, 0) + ln

            rows.append(
                (
                    name,
                    cls_name,
                    n_params,
                    logical_numel,
                    local_numel,
                    logical_bytes,
                    local_bytes,
                    dtype_counter,
                    device_counter,
                )
            )

        print(f"[FSDP_UNITS_V2][rank={rank}] num_units={len(rows)}", flush=True)

        for (
            name,
            cls_name,
            n_params,
            logical_numel,
            local_numel,
            logical_bytes,
            local_bytes,
            dtype_counter,
            device_counter,
        ) in rows[:max_units]:
            print(
                f"[FSDP_UNITS_V2][rank={rank}] "
                f"name={name}, cls={cls_name}, n_params={n_params}, "
                f"logical_numel={logical_numel:,}, "
                f"local_numel={local_numel:,}, "
                f"logical_bytes={logical_bytes / 1024**3:.2f} GiB, "
                f"local_bytes={local_bytes / 1024**3:.2f} GiB, "
                f"dtypes={dtype_counter}, "
                f"devices={device_counter}",
                flush=True,
            )

    def training_step(
        self, batch: dict[str, torch.Tensor] | BatchResizingIterator
    ) -> tuple[dict[str, torch.Tensor], float, list[float]]:
        if isinstance(batch, dict):
            global_batch_size = batch["input_ids"].shape[0]
            assert global_batch_size % self.micro_batch_size == 0, (
                f"global batch size {global_batch_size} can not divide micro_batch_size {self.micro_batch_size}"
            )
            micro_batches_iter, micro_batch_cnt, _ = self._split_to_micro_batch(
                batch,
                self.enable_dynamic_batch_size,
                max_tokens_per_mbs=self.max_tokens_per_mbs,
                split_num=global_batch_size // self.micro_batch_size,
            )
            self.gradient_accumulation = micro_batch_cnt
        else:
            global_batch_size = self.total_batch_size_per_dp // self.n_mini_batches
            micro_batch_cnt = global_batch_size // self.micro_batch_size
            self.gradient_accumulation = micro_batch_cnt

            def iterator_wrapper():
                for _ in range(micro_batch_cnt):
                    yield next(batch)

            micro_batches_iter = iterator_wrapper()
        self.optimizer.zero_grad()
        mbs_metrics_list = {}
        for idx, m_batch in enumerate(micro_batches_iter):
            backward_ctx = self.before_micro_batch(
                self.model,
                is_last_micro_batch=(idx + 1) == micro_batch_cnt,
            )
            for k, v in m_batch.items():
                m_batch[k] = (
                    v.to(Worker.torch_device_type) if isinstance(v, torch.Tensor) else v
                )

            # batch for forward
            logprobs, entropy = self.forward_batch(m_batch, True)

            # batch for backward
            # Prefer recomputed_logprobs (from actor inference), fallback to rollout_logprobs
            old_logprobs = m_batch.get("recomputed_logprobs")
            if old_logprobs is None:
                old_logprobs = m_batch["rollout_logprobs"]
            advantages = m_batch["advantages"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"][:, -self.response_len :]

            clip_ratio = self.cfg.algorithm.ratio_clip_eps
            clip_ratio_low = self.cfg.algorithm.get("clip_ratio_low", None)
            clip_ratio_high = self.cfg.algorithm.get("clip_ratio_high", None)
            clip_ratio_low = (
                clip_ratio_low if clip_ratio_low is not None else clip_ratio
            )
            clip_ratio_high = (
                clip_ratio_high if clip_ratio_high is not None else clip_ratio
            )
            clip_ratio_c = self.cfg.algorithm.get("clip_ratio_c", 3.0)

            if self.cfg.algorithm.get("importance_sampling_fix", False):
                if (
                    "rollout_logprobs" not in m_batch
                    or "recomputed_logprobs" not in m_batch
                ):
                    raise ValueError(
                        "importance_sampling_fix requires both rollout_logprobs and recomputed_logprobs"
                    )
                rollout_logprobs = m_batch["rollout_logprobs"]
                recomputed_logprobs = m_batch["recomputed_logprobs"]
                advantages = advantages * torch.clamp(
                    (recomputed_logprobs - rollout_logprobs).exp(),
                    max=self.cfg.algorithm.importance_sampling_clip,
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=old_logprobs,
                advantages=advantages,
                clip_ratio_c=clip_ratio_c,
                clip_ratio_low=clip_ratio_low,
                clip_ratio_high=clip_ratio_high,
                loss_mask=loss_mask,
                clip_log_ratio_min=self.cfg.algorithm.get("clip_log_ratio_min", None),
                clip_log_ratio_max=self.cfg.algorithm.get("clip_log_ratio_max", None),
                fast_path_zero_loss_mask=True,
            )

            entropy_loss = torch.tensor(
                0.0, device=Worker.torch_platform.current_device()
            )
            if self.calculate_entropy:
                entropy_loss = self.loss_agg_func(entropy, mask=loss_mask)
                if self.calculate_entropy_loss:
                    loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

            kl_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
            if self.kl_beta > 0 and ref_logprobs is not None:
                kld = kl_penalty(ref_logprobs, logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld, loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation

            self._print_fsdp_unit_sizes()

            self._print_mem(
                f"training_step: mb={idx}: before backward",
                reset_peak=True,
            )

            # with backward_ctx:
            #     self.grad_scaler.scale(loss).backward()
                
            rank = int(os.environ.get("RANK", "0"))

            bwd_hook_handles = install_backward_memory_hooks(
                self.model,
                hook_every=1,                 # 精确定位时设 1
                log_all_ranks=False,          # 先只看 rank0；如果 OOM rank 不确定，再改 True
                reset_peak_before_backward=True,
                include_param_grad_hooks=False,
            )

            try:
                _mem_report(
                    "BEFORE grad_scaler.scale(loss).backward()",
                    rank=rank,
                    log_all_ranks=False,
                )

                from contextlib import nullcontext
                with nullcontext():
                    self.grad_scaler.scale(loss).backward()

                _mem_report(
                    "AFTER grad_scaler.scale(loss).backward()",
                    rank=rank,
                    log_all_ranks=False,
                )

            finally:
                remove_hooks(bwd_hook_handles)

            self._print_mem(f"training_step: mb={idx}: after backward")

            # --------------------------------------------------------
            # metrics update
            # --------------------------------------------------------
            self._print_mem(f"training_step: mb={idx}: before metrics update")

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

            self._print_mem(f"training_step: mb={idx}: after metrics update")

            # 可选：释放一些明显不再需要的局部引用，帮助判断是否有中间张量被保留。
            # 注意这不会破坏 autograd，因为 backward 已经完成。
            self._print_mem(f"training_step: mb={idx}: before delete local tensors")

            try:
                del logprobs
                del entropy
                del loss
                del final_loss_metric
                del entropy_loss
                del kl_loss
                if "kld" in locals():
                    del kld
            except Exception:
                pass

            self._print_mem(f"training_step: mb={idx}: after delete local tensors")

        # ------------------------------------------------------------
        # 4. optimizer step
        # 如果日志停在 before optimizer_step，强烈怀疑 Adam state lazy init、
        # grad norm、unscale、clip_grad 或 optimizer.step 内部峰值。
        # ------------------------------------------------------------
        self._print_mem(
            "training_step: before optimizer_step",
            reset_peak=True,
        )

        if hasattr(self, "_print_optimizer_state_summary"):
            self._print_optimizer_state_summary("training_step: before optimizer_step")

        grad_norm, lr_list = self.optimizer_step()

        self._print_mem("training_step: after optimizer_step")

        if hasattr(self, "_print_optimizer_state_summary"):
            self._print_optimizer_state_summary("training_step: after optimizer_step")

        # ------------------------------------------------------------
        # 5. lr scheduler
        # ------------------------------------------------------------
        if self.lr_sched_sync_with_optim:
            self._print_mem("training_step: before lr_scheduler.step")
            self.lr_scheduler.step()

        # display the degree of mismatch between training and rollout
        rollout_train_kl = compute_rollout_train_kl(m_batch, loss_mask)

        # aggregate metrics across micro-batches
        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }
        if rollout_train_kl is not None:
            mean_metric_dict["actor/rollout_train_kl"] = rollout_train_kl


        self._print_mem("training_step: after local aggregate metrics before all_reduce")

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        self._print_mem("training_step: after all_reduce_dict")

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]

        self._print_mem("training_step: before return")

        return mean_metric_dict

    def run_training_pipeline(self, input_channel: Channel) -> tuple[dict, list]:
        self.model.train()
        train_batch_iterator = BatchResizingIterator(
            cfg=self.cfg,
            get_batch_fn=partial(self.get_batch, input_channel),
            micro_batch_size=self.micro_batch_size,
            total_batch_size=self.total_batch_size_per_dp,
            num_global_batches=self.n_mini_batches,
            forward_only=False,
        )
        train_batch_iterator.register_get_batch_handler(
            self.compute_advantages_and_returns
        )

        if self.cfg.algorithm.normalize_advantages:

            def normalize_advantages(batch: dict[str, torch.Tensor]):
                mask = batch["response_mask"][:, -self.response_len :]
                batch["advantages"] = masked_normalization(batch["advantages"], mask)
                return batch

            train_batch_iterator.register_global_batch_handler(normalize_advantages)

        self._load_weight_and_optimizer()
        training_metrics_list = []
        with self.worker_timer("run_training"):
            for _ in range(self.n_mini_batches):
                mean_metric_dict = self.training_step(batch=train_batch_iterator)
                training_metrics_list.append(mean_metric_dict)
            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        # Rollout metrics
        batch = train_batch_iterator.get_all_batches()
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            batch, self.cfg.data.max_prompt_length, self.response_len
        )

        return rollout_metrics, training_metrics_list

    def _dp_load_balance(self, batch: dict[str, torch.Tensor]):
        batch_size = batch["input_ids"].shape[0]
        assert batch_size == self.total_batch_size_per_dp, (
            f"DP Load balance is only available when a single batch contains all data, e.g., in collocated mode. But got {batch_size=} and {self.total_batch_size_per_dp=}."
        )
        batch = RolloutDataBalance.from_rollout_batches(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            partitioning_tool=get_seqlen_balanced_partitions,
        )
        return batch
    
    def _debug_mem_rank(self):
        """
        尽量获取全局 rank。优先使用 torch.distributed，其次用环境变量，
        最后尝试 self 上常见的 rank 属性。
        """
        try:
            if dist is not None and dist.is_available() and dist.is_initialized():
                return dist.get_rank()
        except Exception:
            pass

        for env_name in ("RANK", "WORLD_RANK", "GLOBAL_RANK"):
            v = os.environ.get(env_name)
            if v is not None:
                try:
                    return int(v)
                except ValueError:
                    pass

        for attr_name in (
            "global_rank",
            "rank",
            "_rank",
            "worker_rank",
            "actor_rank",
            "dp_rank",
        ):
            if hasattr(self, attr_name):
                try:
                    return int(getattr(self, attr_name))
                except Exception:
                    pass

        return 0

    def _should_print_mem(self):
        """
        默认只让 rank0 打印。
        如果想所有 rank 都打印，可以设置：
            export DEBUG_MEM_ALL_RANKS=1
        """
        if os.environ.get("DEBUG_MEM_ALL_RANKS", "0") == "1":
            return True
        return self._debug_mem_rank() == 0

    def _fmt_bytes(self, x):
        if x is None:
            return "N/A"
        return f"{x / 1024**3:.2f} GiB"

    def _safe_torch_mem_call(self, backend, name, device=None):
        fn = getattr(backend, name, None)
        if fn is None:
            return None
        try:
            return fn(device)
        except TypeError:
            return fn()
        except Exception:
            return None

    def _print_mem(self, tag, *, reset_peak=False, use_smi=False):
        """
        打印当前进程视角下的 torch.npu 显存。
        - allocated: 当前 PyTorch tensor 实际占用
        - reserved: PyTorch/CANN allocator 当前保留
        - max_*: 进程启动以来，或 reset_peak 后的峰值
        """
        if not self._should_print_mem():
            return

        rank = self._debug_mem_rank()
        pid = os.getpid()

        if not hasattr(torch, "npu") or not torch.npu.is_available():
            print(
                f"[MEM][rank={rank}][pid={pid}][{tag}] torch.npu unavailable",
                flush=True,
            )
            return

        backend = torch.npu

        try:
            device = backend.current_device()
        except Exception:
            device = None

        try:
            if device is not None:
                backend.synchronize(device)
            else:
                backend.synchronize()
        except Exception:
            pass

        if reset_peak:
            for name in ("reset_peak_memory_stats", "reset_max_memory_allocated"):
                fn = getattr(backend, name, None)
                if fn is not None:
                    try:
                        fn(device)
                    except TypeError:
                        fn()
                    except Exception:
                        pass

        allocated = self._safe_torch_mem_call(backend, "memory_allocated", device)
        reserved = self._safe_torch_mem_call(backend, "memory_reserved", device)
        max_allocated = self._safe_torch_mem_call(backend, "max_memory_allocated", device)
        max_reserved = self._safe_torch_mem_call(backend, "max_memory_reserved", device)

        free_mem = None
        total_mem = None
        mem_get_info = getattr(backend, "mem_get_info", None)
        if mem_get_info is not None:
            try:
                free_mem, total_mem = mem_get_info(device)
            except TypeError:
                try:
                    free_mem, total_mem = mem_get_info()
                except Exception:
                    pass
            except Exception:
                pass

        try:
            device_name = backend.get_device_name(device)
        except Exception:
            device_name = "NPU"

        print(
            "[MEM]"
            f"[rank={rank}]"
            f"[pid={pid}]"
            f"[device={device}:{device_name}]"
            f"[{tag}] "
            f"allocated={self._fmt_bytes(allocated)}, "
            f"reserved={self._fmt_bytes(reserved)}, "
            f"max_allocated={self._fmt_bytes(max_allocated)}, "
            f"max_reserved={self._fmt_bytes(max_reserved)}, "
            f"free={self._fmt_bytes(free_mem)}, "
            f"total={self._fmt_bytes(total_mem)}",
            flush=True,
        )

        if use_smi or os.environ.get("DEBUG_NPU_SMI", "0") == "1":
            try:
                r = subprocess.run(
                    ["bash", "-lc", "npu-smi info"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10,
                )
                print(
                    f"[MEM][rank={rank}][pid={pid}][{tag}] npu-smi info:\n{r.stdout}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[MEM][rank={rank}][pid={pid}][{tag}] npu-smi failed: {repr(e)}",
                    flush=True,
                )

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        # Get all batches for this DP
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )

        if self.is_pipeline:
            return self.run_training_pipeline(input_channel)

        batches = []
        recv_batch_size = 0
        while recv_batch_size < self.total_batch_size_per_dp:
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            recv_batch_size += rollout_result.num_sequence
        assert recv_batch_size == self.total_batch_size_per_dp, (
            f"Expected {self.total_batch_size_per_dp} sequences from channel, but got {recv_batch_size}"
        )
        global_batch = RolloutResult.merge_batches(batches)

        assert (
            "recomputed_logprobs" in global_batch or "rollout_logprobs" in global_batch
        )

        # Compute advantages and returns
        global_batch = self.compute_advantages_and_returns(global_batch)

        if self.enable_dp_load_balance:
            global_batch = self._dp_load_balance(global_batch)

        if self.cfg.algorithm.normalize_advantages:
            mask = global_batch["response_mask"][:, -self.response_len :]
            global_batch["advantages"] = masked_normalization(
                global_batch["advantages"], mask
            )

        # Must be called after batch is retrieved, which is when rollout has stopped
        # Otherwise, loading model might cause OOM
        self._load_weight_and_optimizer()

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )

        training_metrics_list = []
        # Global batch iterations
        with self.worker_timer():
            for mb_idx, mini_batch in enumerate(mini_batches):
                self._print_mem(
                    f"run_training: before training_step mini_batch={mb_idx}",
                    reset_peak=True,
                )

                mean_metric_dict = self.training_step(batch=mini_batch)

                self._print_mem(
                    f"run_training: after training_step mini_batch={mb_idx}",
                )

                training_metrics_list.append(mean_metric_dict)

            if not self.lr_sched_sync_with_optim:
                self._print_mem("run_training: before lr_scheduler.step")
                self.lr_scheduler.step()
                self._print_mem("run_training: after lr_scheduler.step")

        self._print_mem("run_training: after training loop")

        # Rollout metrics
        self._print_mem("run_training: before compute_math_rollout_metrics")
        rollout_metrics, _, _ = compute_math_rollout_metrics(
            global_batch, self.cfg.data.max_prompt_length, self.response_len
        )
        self._print_mem("run_training: after compute_math_rollout_metrics")

        self._print_mem("run_training: before return")

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"][:, -self.response_len :]
                logprob = batch.get("recomputed_logprobs")
                if logprob is None:
                    logprob = batch.get("rollout_logprobs")
                logprob = logprob.to(Worker.torch_device_type)

                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    group_size=self.cfg.algorithm.group_size,
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=logprob,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages
        return batch


class EmbodiedFSDPActor(FSDPModelManager, Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)
        self.cfg = cfg
        self._env_group_name = cfg.env.group_name
        self._rollout_group_name = cfg.rollout.group_name
        self._component_placement = HybridComponentPlacement(cfg, Cluster())

        # stage_num: default to 2, use for pipeline rollout process
        self.stage_num = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.actor.get("enable_offload", False)
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")

        self.enable_sft_co_train = cfg.actor.get("enable_sft_co_train", False)
        self.version = 0
        if self.enable_sft_co_train:
            self._build_sft_data_loader()

        # create weight syncer
        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer")
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        self.update_epoch = self.cfg.algorithm.get("update_epoch", 1)

        self._sync_weight_comm_options = self.weight_syncer.comm_options

        self._is_weight_sender = self._rank == 0
        self._actor_world_size = self._world_size
        self._rollout_all_ranks = list(
            range(self._component_placement.get_world_size("rollout"))
        )

    def init_worker(self) -> None:
        """
        Initialize the actor worker. build the model and use corresponding training backend,
        if needed, offload model parameters and optimizer states to CPU.
        """
        self.setup_model_and_optimizer()

        if self.enable_offload:
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> nn.Module:
        model = get_model(self.cfg.actor.model)
        if model is None:
            model = super().model_provider_func()

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            model.load_state_dict(model_dict)

        return model

    def get_rollout_state_dict(self) -> dict:
        return self.get_model_state_dict(cpu_offload=False, full_state_dict=False)

    @Worker.timer("actor/sync_model_to_rollout")
    async def sync_model_to_rollout(self) -> None:
        if self.enable_offload:
            if not self.is_optimizer_offloaded:
                self.offload_optimizer()

            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device, False)

        state_dict = self.get_rollout_state_dict()

        async def send_func(data):
            if not self._is_weight_sender:
                return
            await self.broadcast(
                data,
                groups=[
                    (self._group_name, 0),
                    (self._rollout_group_name, self._rollout_all_ranks),
                ],
                src=(self._group_name, 0),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def recv_func():
            return await self.recv(
                src_group_name=self._rollout_group_name,
                src_rank=0,
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        if not self.weight_syncer.sender_initialized():
            await self.weight_syncer.init_sender(
                state_dict=state_dict,
                send=send_func,
                recv=recv_func,
                param_names_need_sync=self.param_names_need_sync,
                is_sender=self._is_weight_sender,
            )

        version = (
            self.get_rollout_sync_version()
            if hasattr(self, "get_rollout_sync_version")
            else self.version
        )
        await self.weight_syncer.sync(state_dict, send_func, version=version)

        if self.enable_offload:
            assert not self.is_weight_offloaded, (
                "weight should be offloaded in sync_model_to_rollout"
            )
            self.offload_param_and_grad(True)

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel: Channel) -> None:
        """
        Receive rollout trajectories from rollout workers.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        self.rollout_batch = convert_trajectories_to_batch(recv_list)

        self.rollout_batch = self._process_received_rollout_batch(self.rollout_batch)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """
        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]
        """
        rollout_epoch = self.cfg.env.train.rollout_epoch
        rollout_batch = process_nested_dict_for_adv(rollout_batch, rollout_epoch)

        if (
            not self.cfg.env.train.auto_reset
            and not self.cfg.env.train.ignore_terminations
        ):
            dones = rollout_batch[
                "dones"
            ]  # [n_chunk_step, rollout_epoch x bsz, num_action_chunks]
            loss_mask, loss_mask_sum = compute_loss_mask(dones)

            if self.cfg.algorithm.reward_type == "chunk_level":
                loss_mask = loss_mask.any(dim=-1, keepdim=True)
                loss_mask_sum = loss_mask_sum[..., -1:]

            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        # filter data by rewards
        if self.cfg.algorithm.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = self.cfg.algorithm.group_size
            assert batch_size % group_size == 0, (
                f"batch {batch_size} not divisible by group_size {group_size}"
            )
            n_prompts = batch_size // group_size

            # calculate rewards by prompt
            rewards = rewards.transpose(
                0, 1
            )  # [batch, n_chunk_step, num_action_chunks]
            rewards = rewards.reshape(rewards.shape[0], -1)  # [batch, n_step]
            reward_matrix = rewards.reshape(
                n_prompts, group_size, rewards.shape[-1]
            )  # [n_prompts, group_size, n_step]
            reward_matrix = reward_matrix.sum(dim=-1)  # [n_prompts, group_size]
            mean_reward_in_group = reward_matrix.mean(dim=1)  # [n_prompts]

            # mask
            reward_filter_mask = (
                mean_reward_in_group >= self.cfg.algorithm.rewards_lower_bound
            ) & (
                mean_reward_in_group <= self.cfg.algorithm.rewards_upper_bound
            )  # [n_prompts]

            # extend mask dimension
            reward_filter_mask = reward_filter_mask.repeat_interleave(
                group_size
            )  # [batch]
            reward_filter_mask = (
                reward_filter_mask.unsqueeze(0).expand(n_chunk_step, -1).unsqueeze(-1)
            )  # [n_chunk_step, batch, 1]

            # update loss_mask
            if rollout_batch.get("loss_mask", None) is not None:
                rollout_batch["loss_mask"] = (
                    reward_filter_mask & rollout_batch["loss_mask"]
                )
            else:
                rollout_batch["loss_mask"] = reward_filter_mask

        return rollout_batch

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """
        Compute the advantages and returns.
        """
        kwargs = {
            "task_type": self.cfg.runner.task_type,
            "adv_type": self.cfg.algorithm.adv_type,
            "rewards": self.rollout_batch["rewards"],
            "dones": self.rollout_batch["dones"],
            "values": self.rollout_batch.get("prev_values", None),
            "gamma": self.cfg.algorithm.get("gamma", 1),
            "gae_lambda": self.cfg.algorithm.get("gae_lambda", 1),
            "group_size": self.cfg.algorithm.get("group_size", 8),
            "reward_type": self.cfg.algorithm.reward_type,
            "loss_mask": self.rollout_batch.get("loss_mask", None),
            "loss_mask_sum": self.rollout_batch.get("loss_mask_sum", None),
        }

        advantages_and_returns = calculate_adv_and_returns(**kwargs)

        self.rollout_batch.update(advantages_and_returns)
        if kwargs["loss_mask"] is not None:
            self.rollout_batch.update({"loss_mask": kwargs["loss_mask"]})
        if kwargs["loss_mask_sum"] is not None:
            self.rollout_batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        rollout_metrics = compute_rollout_metrics(self.rollout_batch)
        return rollout_metrics

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            if "config_name" not in self.cfg.actor:
                raise ValueError(
                    "config_name is required when enable_sft_co_train=True"
                )
            training_config_name = self.cfg.actor.config_name
            data_loader_config = get_openpi_config(
                training_config_name,
                model_path=self.cfg.actor.model.model_path,
                repo_id=repo_id,
                data_kwargs=getattr(self.cfg.actor.model, "openpi_data", None),
            )
            self.data_loader = _data.create_data_loader(
                data_loader_config, framework="pytorch", shuffle=True
            )
            self.sft_iterator = iter(self.data_loader)
            self.train_epoch = 0
            self.sft_loss_weight = self.cfg.actor.get("sft_loss_weight", 0.1)
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def _train_sft_epoch(
        self, metrics_data: dict[str, torch.Tensor], loss: torch.Tensor
    ) -> torch.Tensor:
        """
        Train one epoch of SFT.
        """
        metrics_data["ppo_loss"] = loss.clone().detach().item()

        # Get next data batch
        try:
            observation, actions = next(self.sft_iterator)
        except StopIteration:
            self.train_epoch += 1
            self.data_loader.set_epoch(self.train_epoch)
            self.sft_iterator = iter(self.data_loader)
            observation, actions = next(self.sft_iterator)

        sft_loss = self.model(
            data=(observation, actions),
            forward_type=ForwardType.SFT,
        )
        metrics_data["sft_loss"] = sft_loss.detach().item()
        total_loss = loss + self.sft_loss_weight * sft_loss
        loss = total_loss

        metrics_data["loss_ratio"] = (
            np.abs(metrics_data["sft_loss"]) / np.abs(metrics_data["ppo_loss"])
            if np.abs(metrics_data["ppo_loss"]) > 0
            else float("inf")
        )
        if metrics_data["loss_ratio"] > 1e5:
            self.logger.warning(
                "SFT/PPO loss imbalance detected: "
                f"ratio={metrics_data['loss_ratio']:.3e}, "
                f"sft_loss={metrics_data['sft_loss']:.6f}, "
                f"ppo_loss={metrics_data['ppo_loss']:.6f}, "
                f"sft_loss_weight={self.sft_loss_weight:.6f}"
            )
        return loss

    @Worker.timer("run_training")
    def run_training(self) -> None:
        """
        Run the training process using the received rollout batch.
        """
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                # split batch into micro_batches
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == self.gradient_accumulation,
                    )
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
    ) -> None:
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_sum = micro_batch.get("loss_mask_sum", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs["temperature"] = self.cfg.rollout.sampling_params.temperature_train
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=self.cfg.algorithm.entropy_bonus > 0,
                compute_values=compute_values,
                use_cache=False,
                **kwargs,
            )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            prev_logprobs = output_dict["prev_logprobs"]

        loss_kwargs = {
            "loss_type": self.cfg.algorithm.loss_type,
            "logprob_type": self.cfg.algorithm.logprob_type,
            "reward_type": self.cfg.algorithm.reward_type,
            "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
            "logprobs": output_dict["logprobs"],
            "values": output_dict.get("values", None),
            "old_logprobs": prev_logprobs,
            "advantages": advantages,
            "returns": returns,
            "prev_values": prev_values,
            "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
            "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
            "value_clip": self.cfg.algorithm.get("value_clip", None),
            "huber_delta": self.cfg.algorithm.get("huber_delta", None),
            "loss_mask": loss_mask,
            "loss_mask_sum": loss_mask_sum,
            "max_episode_steps": self.cfg.env.train.max_episode_steps,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
        }

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
        ]:
            loss_kwargs["clip_ratio_c"] = self.cfg.algorithm.get("clip_ratio_c", 3.0)
            if self.cfg.algorithm.get("clip_log_ratio_min") is not None:
                loss_kwargs["clip_log_ratio_min"] = (
                    self.cfg.algorithm.clip_log_ratio_min
                )
            if self.cfg.algorithm.get("clip_log_ratio_max") is not None:
                loss_kwargs["clip_log_ratio_max"] = (
                    self.cfg.algorithm.clip_log_ratio_max
                )

        loss, metrics_data = policy_loss(**loss_kwargs)
        entropy_loss = torch.tensor(0.0, device=Worker.torch_platform.current_device())
        if self.cfg.algorithm.entropy_bonus > 0 and not loss_kwargs["critic_warmup"]:
            entropy = output_dict["entropy"]
            entropy = reshape_entropy(
                entropy,
                entropy_type=self.cfg.algorithm.entropy_type,
                action_dim=self.cfg.actor.model.get("action_dim", 7),
                batch_size=output_dict["logprobs"].shape[0],
            )
            entropy_loss = masked_mean(entropy, mask=loss_mask)
            loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        loss /= self.gradient_accumulation
        with backward_ctx:
            self.grad_scaler.scale(loss).backward()

        metrics_data["actor/total_loss"] = loss.detach().item()
        append_to_dict(metrics, metrics_data)

    def set_global_step(self, global_step: int) -> None:
        """
        Set the global step for the model, if needed.
        """
        self.version = global_step
        if hasattr(self.model, "set_global_step"):
            self.model.set_global_step(global_step)

    def finish_global_batch(self, metrics: dict[str, list[float]]) -> None:
        self.torch_platform.empty_cache()
        grad_norm, lr_list = self.optimizer_step()
        self.optimizer.zero_grad()
        metric_data = {
            "actor/grad_norm": grad_norm,
            "actor/lr": lr_list[0],
        }
        if len(lr_list) > 1:
            metric_data["critic/lr"] = lr_list[1]
        append_to_dict(metrics, metric_data)
