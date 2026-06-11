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
from functools import partial
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import reduce_tensor
from torch.utils import _pytree

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import (
    calculate_adv_and_returns,
    get_loss_scales,
    policy_loss,
)
from rlinf.algorithms.utils import (
    kl_penalty,
)
from rlinf.config import SupportedModel, torch_dtype_from_precision
from rlinf.data.embodied_io_struct import Trajectory, convert_trajectories_to_batch
from rlinf.data.io_struct import BatchResizingIterator, DynamicRolloutResult
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import (
    FSDPModelManager,
)
from rlinf.hybrid_engines.fsdp.utils import (
    pack_fsdp_input,
    prepare_pack_fsdp,
    unpack_fsdp_logprobs,
    unpack_sequences,
)
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, CollectiveGroupOptions, Worker
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
    compute_rollout_metrics_dynamic,
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
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.utils.utils import (
    clear_memory,
    compute_entropy_from_logits,
    compute_logprobs_from_logits,
    cpu_dict,
    cpu_weight_swap,
    get_loss_agg_func,
    masked_mean,
    reshape_entropy,
    retrieve_model_state_dict_in_cpu,
)
from rlinf.workers.rollout.utils import RankMapper

from rlinf.workers.actor.fsdp_actor_worker import (
    FSDPActor,
)


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


class MAFSDPActor(FSDPActor):
    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
        cfg_fsdp: Optional[DictConfig] = None,
    ) -> None:
        super().__init__(cfg, placement, cfg_fsdp)
        self.is_dynamic_rollout_batch = self.cfg.agentloop.is_dynamic_rollout_batch
        assert self.is_dynamic_rollout_batch
        assert self.enable_dp_load_balance, (
            "enable_dp_load_balance must be True when is_dynamic_rollout_batch is True"
        )
        self.placement = placement
        assert self.placement.is_collocated, (
            "Only collocated placement is supported for multi-agent actor"
        )
        loss_scales = self.cfg.algorithm.get("loss_scales", [])
        self.loss_scale_fns = get_loss_scales(loss_scales)
        self.pack_traj = self.cfg.actor.get("pack_traj", True)
    
    def get_batch(
        self, channel: Channel
    ) -> tuple[dict[str, torch.Tensor], DynamicRolloutResult]:
        result: DynamicRolloutResult = channel.get()

        batch = result.to_actor_batch(
            self.cfg.actor.model.encoder_seq_length,
            self.tokenizer.eos_token_id,
        )
        return batch, result

    def forward_batch(
        self,
        m_batch: dict[str, torch.Tensor],
        calculate_entropy: bool = False,
    ) -> torch.Tensor:
        input_ids = m_batch["input_ids"]
        attention_mask = m_batch["attention_mask"]
        position_ids = m_batch["position_ids"]

        seq_length = input_ids.shape[1]

        multi_modal_inputs = {}
        if "multi_modal_inputs" in m_batch:
            for key in m_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat(
                    [inputs[key] for inputs in m_batch["multi_modal_inputs"]],
                    dim=0,
                ).to(Worker.torch_device_type)

        if self.enable_dynamic_batch_size:
            max_seq_len_pack = self.max_tokens_per_mbs
            max_seq_len_unpack = seq_length

            if "prompt_lengths" in m_batch and "response_lengths" in m_batch:
                seq_lens = (
                    m_batch["prompt_lengths"].to(input_ids.device)
                    + m_batch["response_lengths"].to(input_ids.device)
                )
            else:
                # Fallback: DynamicRolloutResult attention_mask is True for
                # all valid prompt+response tokens.
                seq_lens = attention_mask.to(torch.long).sum(dim=1)

            idx_starts = torch.zeros_like(seq_lens, dtype=torch.long)
            idx_ends = seq_lens.to(torch.long)

            input_ids, position_ids, attention_mask = pack_fsdp_input(
                input_ids,
                position_ids,
                attention_mask,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_pack=max_seq_len_pack,
                eos_token_id=self.tokenizer.eos_token_id,
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

        if self.enable_dynamic_batch_size:
            # unpack_fsdp_logprobs is expected to return logprobs unpacked to
            # [B, max_seq_len_unpack], aligned with token positions.
            logprobs = unpack_fsdp_logprobs(
                logits,
                input_ids,
                idx_starts=idx_starts,
                idx_ends=idx_ends,
                max_seq_len_unpack=max_seq_len_unpack,
                eos_token_id=self.tokenizer.eos_token_id,
                compute_logprobs_fn=self.compute_logprobs,
            )
        else:
            if input_ids.shape[1] <= 1:
                logprobs = torch.zeros(
                    input_ids.shape,
                    dtype=logits.dtype,
                    device=logits.device,
                )
            else:
                # token_logprobs[:, t] corresponds to input_ids[:, t + 1]
                token_logprobs = self.compute_logprobs(
                    logits[:, :-1, :],
                    input_ids[:, 1:],
                )

                logprobs = torch.zeros(
                    input_ids.shape,
                    dtype=token_logprobs.dtype,
                    device=token_logprobs.device,
                )
                logprobs[:, 1:] = token_logprobs

        if calculate_entropy:
            if self.enable_dynamic_batch_size:
                pos_entropy = compute_entropy_from_logits(logits)

                pos_entropy = unpack_sequences(
                    pos_entropy,
                    idx_starts,
                    idx_ends,
                    max_seq_len_unpack,
                    pad_val=0,
                )

                entropy = torch.zeros(
                    pos_entropy.shape,
                    dtype=pos_entropy.dtype,
                    device=pos_entropy.device,
                )
                if pos_entropy.shape[1] > 1:
                    entropy[:, 1:] = pos_entropy[:, :-1]
            else:
                if input_ids.shape[1] <= 1:
                    entropy = torch.zeros(
                        input_ids.shape,
                        dtype=logits.dtype,
                        device=logits.device,
                    )
                else:
                    token_entropy = compute_entropy_from_logits(logits[:, :-1, :])

                    entropy = torch.zeros(
                        input_ids.shape,
                        dtype=token_entropy.dtype,
                        device=token_entropy.device,
                    )
                    entropy[:, 1:] = token_entropy

            return logprobs, entropy

        return logprobs

    def inference_step(
        self,
        batch: dict[str, torch.Tensor],
        rollout_result: DynamicRolloutResult,
        compute_ref_logprobs: bool,
    ):
        micro_batches_iter, _, dbs_indices = self._split_to_micro_batch(
            batch,
            self.enable_dynamic_batch_size,
            max_tokens_per_mbs=self.max_tokens_per_mbs,
            split_num=rollout_result.num_sequence
            // self.cfg.algorithm.logprob_forward_micro_batch_size,
        )
        if self.enable_dynamic_batch_size:
            indices = sum(dbs_indices, [])
            revert_indices = torch.tensor(
                get_reverse_idx(indices),
                dtype=torch.long,
            )
        micro_batches = list(micro_batches_iter)

        prev_logprobs, ref_logprobs = None, None

        # Prev logprobs
        prev_logprobs = torch.cat(
            [self.forward_batch(batch) for batch in micro_batches]
        ).cpu()

        if self.enable_dynamic_batch_size:
            assert len(indices) == prev_logprobs.size(0), (
                f"Dynamic batch size indices length {len(indices)} does not equal "
                f"output length {prev_logprobs.size(0)}"
            )
            prev_logprobs = prev_logprobs[revert_indices]

        # Ref logprobs
        if compute_ref_logprobs:
            assert self.ref_policy_state_dict is not None, (
                "Reference policy state dict is None but compute_ref_logprobs is True"
            )
            with cpu_weight_swap(
                self.model,
                self.ref_policy_state_dict,
                self.offload_model_buffer,
            ):
                ref_logprobs = torch.cat(
                    [self.forward_batch(batch) for batch in micro_batches]
                ).cpu()

                if self.enable_dynamic_batch_size:
                    assert len(indices) == ref_logprobs.size(0), (
                        f"Dynamic batch size indices length {len(indices)} does not equal "
                        f"output length {ref_logprobs.size(0)}"
                    )
                    ref_logprobs = ref_logprobs[revert_indices]

        return prev_logprobs, ref_logprobs

    def run_inference(
        self,
        input_channel: Channel,
        output_channel: Channel,
        compute_ref_logprobs: bool,
    ):
        """
        Compute prev/ref logprobs using the actor Model's forward.
        Override to handle DynamicRolloutResult which has different structure.
        Args:
            input_channel: The input channel to read from.
            output_channel: The output channel to send results to.
            compute_ref_logprobs: Whether to compute reference logprobs.
        """
        assert not self.is_pipeline, (
            "MAFSDPActor currently only supports collocated inference"
        )

        batches = []
        rollout_results = []
        total_result_size_per_dp = (
            self.cfg.data.rollout_batch_size // torch.distributed.get_world_size()
        )

        for _ in range(total_result_size_per_dp):
            batch, rollout_result = self.get_batch(input_channel)
            batches.append(batch)
            rollout_results.append(rollout_result)

        merged_batch, num_sequence_per_group = DynamicRolloutResult.merge_batches(
            batches, adjust_traj_indices=False, return_num_sequence_per_group=True
        )
        rollout_result = DynamicRolloutResult.merge_result_list(rollout_results)

        self._load_weight_and_optimizer()
        self.model.eval()
        with self.worker_timer():
            with torch.no_grad():
                prev_logprobs, ref_logprobs = self.inference_step(
                    merged_batch,
                    rollout_result,
                    compute_ref_logprobs,
                )

            if rollout_result.rollout_logprobs is not None:
                rollout_result.recompute_prev_logprobs = prev_logprobs
            else:
                rollout_result.prev_logprobs = prev_logprobs

            if compute_ref_logprobs:
                rollout_result.ref_logprobs = ref_logprobs

        rollout_result_per_group = DynamicRolloutResult.split_results(
            rollout_result, num_sequence_per_group
        )
        for rollout_result in rollout_result_per_group:
            output_channel.put(rollout_result)

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
            prev_logprobs = m_batch["prev_logprobs"]
            advantages = m_batch["advantages"] * m_batch["loss_scales"]
            ref_logprobs = None
            if "ref_logprobs" in m_batch:
                ref_logprobs = m_batch["ref_logprobs"]

            loss_mask = m_batch["response_mask"]

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
                raise AssertionError(
                    "importance_sampling_fix is not supported for dynamic rollout batch"
                )

            loss, mbs_metrics_data = policy_loss(
                task_type=self.task_type,
                loss_type=self.cfg.algorithm.loss_type,
                loss_agg_func=self.loss_agg_func,
                logprobs=logprobs,
                old_logprobs=prev_logprobs,
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
                kld = kl_penalty(logprobs, ref_logprobs, self.kl_penalty_type)
                kl_loss = self.loss_agg_func(kld * m_batch["loss_scales"], loss_mask)
                loss = loss + kl_loss * self.kl_beta

            # add to log
            # scale loss for gradient accumulation and backprop
            final_loss_metric = loss.detach()
            loss = loss / self.gradient_accumulation

            with backward_ctx:
                self.grad_scaler.scale(loss).backward()

            mbs_metrics_data.update(
                {
                    "actor/final_loss": final_loss_metric,
                    "actor/entropy_loss": entropy_loss.detach(),
                    "actor/kl_loss": kl_loss.detach(),
                }
            )

            append_to_dict(mbs_metrics_list, mbs_metrics_data)

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

        grad_norm, lr_list = self.optimizer_step()

        # ------------------------------------------------------------
        # 5. lr scheduler
        # ------------------------------------------------------------
        if self.lr_sched_sync_with_optim:
            self.lr_scheduler.step()

        # ------------------------------------------------------------
        # 6. aggregate metrics
        # ------------------------------------------------------------

        mean_metric_dict = {
            key: torch.mean(torch.stack(value))
            for key, value in mbs_metrics_list.items()
        }

        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )

        mean_metric_dict["actor/grad_norm"] = float(grad_norm)
        mean_metric_dict["actor/lr"] = lr_list[0]

        return mean_metric_dict

    def _dp_load_balance_dynamic(
        self,
        batch: dict[str, torch.Tensor],
        batch_pad: dict[str, torch.Tensor],
        split_fix_chunk: int,
    ):
        return RolloutDataBalance.from_rollout_batches_dynamic(
            rollout_batches=batch,
            dp_world_size=torch.distributed.get_world_size(),
            dp_rank=torch.distributed.get_rank(),
            dp_group=torch.distributed.group.WORLD,
            rollout_batch_pad=batch_pad,
            split_fix_chunk=split_fix_chunk,
            partitioning_tool=get_seqlen_balanced_partitions,
        )

    def _compute_rollout_metrics(self, batch: dict[str, torch.Tensor]):
        rollout_metrics, total_prompt_lengths, total_decode_lengths = (
            compute_rollout_metrics_dynamic(
                batch,
                self.cfg.data.max_prompt_length,
                self.response_len,
                torch.distributed.group.WORLD,
            )
        )
        rollout_metrics = cpu_dict(rollout_metrics)

        if self.cfg.actor.get("calculate_flops", False):
            rollout_tflops = self.flops_calculator.flops_generate(
                total_prompt_lengths, total_decode_lengths
            )
            rollout_tflops = rollout_tflops.float().sum().item() / 1e12
            inference_tflops = self.flops_calculator.flops_inference(
                total_prompt_lengths + total_decode_lengths
            )
            inference_tflops = inference_tflops.float().sum().item() / 1e12
            rollout_metrics.update(
                {
                    "rollout_tflops": rollout_tflops,
                    "inference_tflops": inference_tflops,
                    "training_tflops": inference_tflops * 3,
                }
            )
        return rollout_metrics

    def run_training(
        self, input_channel: Channel, do_offload=False
    ) -> tuple[dict, list]:
        assert not do_offload, (
            "do_offload argument of run_inference/run_training is not supported in FSDP for now"
        )
        assert not self.is_pipeline, (
            "MAFSDPActor currently only supports collocated training"
        )

        batches = []
        total_result_size_per_dp = (
            self.cfg.data.rollout_batch_size // torch.distributed.get_world_size()
        )
        for _ in range(total_result_size_per_dp):
            batch, _ = self.get_batch(input_channel)
            batches.append(batch)

        global_batch = DynamicRolloutResult.merge_batches(
            batches,
            self.cfg.algorithm.group_size,
        )
        assert "prev_logprobs" in global_batch

        global_batch = self.compute_advantages_and_returns(global_batch)
        global_batch["loss_scales"] = torch.ones_like(
            global_batch["advantages"]
        ).masked_fill(~global_batch["response_mask"], 0)

        if self.cfg.algorithm.normalize_advantages:
            raise AssertionError("normalize_advantages is not implemented in multi-agent")

        rollout_metrics = self._compute_rollout_metrics(global_batch)

        scale_context = {
            "folding_scale": [],
            "enable_scale_of_group": False,
            "actor_global_batch_size": (
                self.cfg.data.rollout_batch_size
                * self.cfg.algorithm.get("group_size", 1)
                / self.cfg.algorithm.n_minibatches
            ),
            "data_parallel_world_size": torch.distributed.get_world_size(),
        }
        for loss_scale_fn in self.loss_scale_fns:
            global_batch = loss_scale_fn(scale_context, global_batch)
        if self.pack_traj:
            global_batch = DynamicRolloutResult.pack_traj_batch(
                scale_context, global_batch
            )
        for key in list(global_batch.keys()):
            if key == "idx_to_traj" or key.startswith("extra:"):
                global_batch.pop(key, None)

        self._load_weight_and_optimizer()

        if self.enable_dp_load_balance:
            batch_pad = DynamicRolloutResult.get_batch_pad(
                self.cfg.actor.model.encoder_seq_length,
                list(global_batch.keys()),
            )
            global_batch = self._dp_load_balance_dynamic(
                global_batch,
                batch_pad,
                self.cfg.actor.micro_batch_size,
            )

        mini_batches = get_iterator_k_split(
            global_batch,
            num_splits=self.cfg.algorithm.n_minibatches,
            shuffle=self.cfg.algorithm.get("shuffle_rollout", True),
            shuffle_seed=self.cfg.actor.seed,
        )

        self.model.train()
        training_metrics_list = []
        with self.worker_timer():
            for mini_batch in mini_batches:
                if mini_batch["input_ids"].shape == torch.Size([0]):
                    continue
                mean_metric_dict = self.training_step(batch=mini_batch)
                training_metrics_list.append(mean_metric_dict)

            if not self.lr_sched_sync_with_optim:
                self.lr_scheduler.step()

        return rollout_metrics, training_metrics_list

    # Advantages and returns
    def compute_advantages_and_returns(self, batch: dict[str, torch.Tensor]):
        """Compute the advantages and returns.

        Args:
            batch (Dict[str, torch.Tensor]): The rollout batch.
        """
        with self.worker_timer():
            if batch.get("advantages", None) is None:
                mask = batch["response_mask"]
                advantages, _ = calculate_adv_and_returns(
                    task_type=self.task_type,
                    adv_type=self.cfg.algorithm.adv_type,
                    advantage_mode=self.cfg.algorithm.advantage_mode,
                    rewards=batch["rewards"].to(Worker.torch_device_type),
                    loss_mask=mask.to(Worker.torch_device_type),
                    num_sequence=len(batch["input_ids"]),
                    group_size=self.cfg.algorithm.group_size,
                    idx_to_traj=batch["idx_to_traj"],
                    kl_beta=self.reinpp_kl_beta,
                    kl_penalty_type=self.kl_penalty_type,
                    logprob=batch["prev_logprobs"].to(Worker.torch_device_type)
                    if "prev_logprobs" in batch
                    else None,
                    ref_logprob=batch["ref_logprobs"].to(Worker.torch_device_type)
                    if "ref_logprobs" in batch
                    else None,
                    use_reinpp_baseline=self.cfg.algorithm.get(
                        "use_reinpp_baseline", False
                    ),
                )
                batch["advantages"] = advantages

        return batch
