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

import asyncio
import os
import queue
import threading
from typing import Any, Optional

import numpy as np
import torch

from rlinf.algorithms.registry import policy_loss
from rlinf.algorithms.utils import compute_entropy_loss
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import PriorityStore
from rlinf.scheduler import Worker
from rlinf.utils.distributed import all_reduce_dict, masked_normalization
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import put_tensor_device, split_dict_to_chunk
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


def flatten_rollout_batch_for_train(
    nested_dict: dict, shuffle_id: Optional[torch.Tensor]
) -> dict:
    """Flatten [T, B, ...] rollout tensors to [T*B, ...] for actor training."""
    ret_dict = {}
    for key, value in nested_dict.items():
        if key in ["dones", "terminations", "truncations", "prev_values"]:
            if isinstance(value, torch.Tensor):
                value = value[:-1]

        if "env_info" in key:
            raise NotImplementedError("env_info nested dict is not supported here")

        if value is None:
            ret_dict[key] = None
            continue

        if isinstance(value, torch.Tensor):
            flat = value.reshape(-1, *value.shape[2:])
            ret_dict[key] = flat[shuffle_id] if shuffle_id is not None else flat
        elif isinstance(value, dict):
            ret_dict[key] = flatten_rollout_batch_for_train(value, shuffle_id)
        else:
            raise NotImplementedError(
                f"Unsupported value type in rollout batch: key={key}, type={type(value)}"
            )

    return ret_dict


class AsyncPPOEmbodiedFSDPActor(EmbodiedFSDPActor):
    """Embodied FSDP actor worker for async PPO / decoupled actor-critic training."""

    should_stop = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_store_size = self.cfg.algorithm.get(
            "rollout_store_size_per_rank", 1
        )
        # Each env gets the share of the store that it sends this actor per step,
        # so the training mix follows the env counts whatever each env's speed.
        env_shards = self._rollout_geometry.env_shards_per_actor()
        total_shards = sum(env_shards.values())
        self.rollout_store_quotas = {}
        for name, shards in env_shards.items():
            assert self.rollout_store_size * shards % total_shards == 0, (
                "algorithm.rollout_store_size_per_rank must split into the envs' "
                f"shares {env_shards}, got {self.rollout_store_size}."
            )
            self.rollout_store_quotas[name] = (
                self.rollout_store_size * shards // total_shards
            )
        self.rollout_stores = {
            name: PriorityStore(maxsize=quota)
            for name, quota in self.rollout_store_quotas.items()
        }

    async def recv_rollout_trajectories(self, input_channel):
        # drain channel
        if getattr(self, "_recv_queue", None) is None:
            self._recv_queue = queue.Queue()
        if (
            getattr(self, "_recv_rollout_thread", None) is None
            or not self._recv_rollout_thread.is_alive()
        ):
            self._recv_rollout_thread = threading.Thread(
                target=self._recv_rollout_thread_main,
                args=(input_channel,),
                daemon=True,
            )
            self._recv_rollout_thread.start()

    def _recv_rollout_thread_main(self, input_channel):
        while not self.should_stop:
            trajectory: Trajectory = input_channel.get()
            self.log_info(
                f"recv trajectory versions.shape={trajectory.versions.shape} "
                "from trajectory channel"
            )
            if trajectory.versions.min() < self.version - self.cfg.algorithm.get(
                "staleness_threshold", None
            ):
                continue
            self._recv_queue.put(trajectory)

    @Worker.timer("drain_received_trajectories")
    def _drain_received_trajectories(self):
        while True:
            try:
                traj: Trajectory = self._recv_queue.get_nowait()
                self.log_info(
                    f"drain traj versions.shape={traj.versions.shape} "
                    f"versions.min={traj.versions.min()} version={self.version} "
                    f"recv_queue.size={self._recv_queue.qsize()}"
                )
                if traj.versions.min() < self.version - self.cfg.algorithm.get(
                    "staleness_threshold", None
                ):
                    continue
                min_v = float(traj.versions.min().item())
                mean_v = float(traj.versions.float().mean().item())
                self.rollout_stores[traj.env_name].add((min_v, mean_v), traj)
                self.log_info(
                    "rollout_store sizes="
                    f"{ {name: len(store) for name, store in self.rollout_stores.items()} }"
                )
            except queue.Empty:
                break

    @Worker.timer("wait_for_rollout_store_ready")
    async def _wait_for_rollout_store_ready(self):
        while getattr(self, "_recv_queue", None) is None:
            await asyncio.sleep(1)

        on_policy_min_ratio = self.cfg.algorithm.get("on_policy_min_ratio", 0.0)
        while True:
            self._drain_received_trajectories()
            with self.worker_timer("remove_below"):
                for store in self.rollout_stores.values():
                    store.remove_below(
                        self.version
                        - self.cfg.algorithm.get("staleness_threshold", None)
                    )
            if all(
                len(self.rollout_stores[name]) >= quota
                for name, quota in self.rollout_store_quotas.items()
            ):
                if on_policy_min_ratio <= 0.0:
                    break
                on_policy_ratios = {
                    name: store.get_metric()
                    .get(int(self.version), {})
                    .get("ratio", 0.0)
                    for name, store in self.rollout_stores.items()
                }
                self.log_info(
                    f"rollout store on_policy_ratios={on_policy_ratios} "
                    f"on_policy_min_ratio={on_policy_min_ratio}"
                )
                if min(on_policy_ratios.values()) >= on_policy_min_ratio:
                    break
            await asyncio.sleep(1)

    @Worker.timer("construct_rollout_batch")
    async def construct_rollout_batch(self, max_trajectories: int | None = None):
        """Take each env's quota of the freshest trajectories as its batch.

        Returns:
            Staleness metrics, under ``<name>/`` for each env of a heterogeneous
            run.
        """
        await self._wait_for_rollout_store_ready()
        torch.distributed.barrier()

        staleness_metrics: dict = {}
        self.env_batches = {}
        self._group_success_metrics = {}
        for spec in self._rollout_geometry.layout.specs:
            store = self.rollout_stores[spec.name]
            trajectories = store.topn(self.rollout_store_quotas[spec.name])
            version_metrics = store.get_metric()
            self.log_info(
                f"rollout store {spec.name} version metrics={version_metrics}"
            )
            env_metrics = {}
            for version_val, stats in version_metrics.items():
                if version_val == "discarded_unused":
                    env_metrics["discarded_unused_trajs"] = stats
                    continue
                diff = int(self.version) - int(version_val)
                env_metrics[f"data_staleness_{diff}/ratio"] = stats["ratio"]
            staleness_metrics.update(spec.scoped(env_metrics) or env_metrics)

            batch, group_metrics = self._process_received_rollout_batch(
                Trajectory.to_batch(trajectories), spec
            )
            self.env_batches[spec.name] = batch
            self._group_success_metrics.update(
                spec.scoped(group_metrics) or group_metrics
            )
        self.log_info(f"staleness metrics={staleness_metrics}")
        return staleness_metrics

    @torch.inference_mode()
    def compute_proximal_logprobs(self) -> None:
        assert not self.is_weight_offloaded, (
            "Weight offloading is not supported when recomputing proximal logprobs."
        )

        t_dim = self.rollout_batch["prev_logprobs"].shape[0]
        b_dim = self.rollout_batch["prev_logprobs"].shape[1]

        flat = flatten_rollout_batch_for_train(self.rollout_batch, shuffle_id=None)
        total = flat["prev_logprobs"].shape[0]
        micro_batch_size = self.cfg.actor.micro_batch_size
        num_splits = (total + micro_batch_size - 1) // micro_batch_size

        iterator = split_dict_to_chunk(flat, num_splits)

        self.model.eval()
        proximal_logprobs_list = []

        for micro_batch in iterator:
            micro_batch = put_tensor_device(micro_batch, self.device)
            forward_inputs = micro_batch.get("forward_inputs", None)
            if forward_inputs is None:
                raise ValueError(
                    "Missing forward_inputs in compute_proximal_logprobs. "
                    "This usually means batch splitting dropped nested dict fields."
                )

            model_kwargs = {}
            if SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.OPENVLA,
                SupportedModel.OPENVLA_OFT,
            ]:
                model_kwargs["temperature"] = (
                    self.cfg.rollout.sampling_params.temperature_train
                )
                model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
            elif SupportedModel(self.cfg.actor.model.model_type) in [
                SupportedModel.GR00T,
                SupportedModel.ABOT_M0,
            ]:
                model_kwargs["prev_logprobs"] = micro_batch["prev_logprobs"]

            out = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=False,
                compute_values=False,
                use_cache=False,
                **model_kwargs,
            )
            proximal_logprobs_list.append(out["logprobs"].cpu())

        proximal_logprobs = torch.cat(proximal_logprobs_list, dim=0).view(
            t_dim,
            b_dim,
            *self.rollout_batch["prev_logprobs"].shape[2:],
        )
        self.rollout_batch["proximal_logprobs"] = proximal_logprobs

    def run_training(self) -> dict[str, Any]:
        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        self._order_training_batch()
        if self.cfg.algorithm.normalize_advantages:
            for pool in self.rollout_batches:
                pool["advantages"] = masked_normalization(
                    pool["advantages"], pool.get("loss_mask", None)
                )

        self.model.train()

        metrics: dict[str, list] = {}
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))

        for _ in range(update_epoch):
            for micro_batch_iter in self._global_batches():
                global_batch_rows = sum(
                    batch["prev_logprobs"].shape[0] for batch in micro_batch_iter
                )
                self.optimizer.zero_grad()

                for mb_idx, data in enumerate(micro_batch_iter):
                    micro_batch_rows = data["prev_logprobs"].shape[0]
                    data = put_tensor_device(
                        data,
                        f"cuda:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(mb_idx + 1) == len(micro_batch_iter),
                    )

                    advantages = data["advantages"]
                    old_logprobs = data["prev_logprobs"]
                    returns = data.get("returns", None)
                    prev_values = data.get("prev_values", None)
                    loss_mask = data.get("loss_mask", None)
                    loss_mask_ratio = data.get("loss_mask_ratio", None)

                    versions = data.get("versions", None)
                    proximal_logprobs = data.get("proximal_logprobs", None)
                    proximal_values = data.get("proximal_values", None)
                    current_version = int(self.version) + 1

                    forward_inputs = data.get("forward_inputs", None)
                    if forward_inputs is None:
                        raise ValueError(
                            "Missing forward_inputs in run_training. "
                            "This usually means batch splitting dropped nested dict fields."
                        )

                    model_kwargs = {}
                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.OPENVLA,
                        SupportedModel.OPENVLA_OFT,
                    ]:
                        model_kwargs["temperature"] = (
                            self.cfg.rollout.sampling_params.temperature_train
                        )
                        model_kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
                    elif SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        model_kwargs["prev_logprobs"] = old_logprobs

                    compute_values = self.cfg.algorithm.adv_type == "gae"

                    with self.amp_context:
                        out = self.model(
                            forward_inputs=forward_inputs,
                            compute_logprobs=True,
                            compute_entropy=(self.cfg.algorithm.entropy_bonus > 0),
                            compute_values=compute_values,
                            use_cache=False,
                            **model_kwargs,
                        )

                    if SupportedModel(self.cfg.actor.model.model_type) in [
                        SupportedModel.GR00T,
                        SupportedModel.ABOT_M0,
                    ]:
                        old_logprobs = out["prev_logprobs"]

                    loss_kwargs = {
                        "loss_type": self.cfg.algorithm.loss_type,
                        "logprob_type": self.cfg.algorithm.logprob_type,
                        "reward_type": self.cfg.algorithm.reward_type,
                        "single_action_dim": self.cfg.actor.model.get("action_dim", 7),
                        "logprobs": out["logprobs"],
                        "values": out.get("values", None),
                        "old_logprobs": old_logprobs,
                        "advantages": advantages,
                        "returns": returns,
                        "prev_values": proximal_values
                        if proximal_values is not None
                        else prev_values,
                        "proximal_logprobs": proximal_logprobs,
                        "versions": versions,
                        "current_version": current_version,
                        "behave_weight_threshold": self.cfg.algorithm.get(
                            "behave_weight_threshold", None
                        ),
                        "clip_ratio_c": self.cfg.algorithm.get("clip_ratio_c", 3.0),
                        "clip_ratio_high": self.cfg.algorithm.clip_ratio_high,
                        "clip_ratio_low": self.cfg.algorithm.clip_ratio_low,
                        "value_clip": self.cfg.algorithm.get("value_clip", None),
                        "huber_delta": self.cfg.algorithm.get("huber_delta", None),
                        "loss_mask": loss_mask,
                        "loss_mask_ratio": loss_mask_ratio,
                        "task_type": self.cfg.runner.task_type,
                        "critic_warmup": self.optimizer_steps
                        < self.critic_warmup_steps,
                    }

                    loss, metrics_data = policy_loss(**loss_kwargs)

                    entropy_loss = torch.tensor(0.0, device=torch.cuda.current_device())
                    if (
                        self.cfg.algorithm.entropy_bonus > 0
                        and not loss_kwargs["critic_warmup"]
                    ):
                        entropy_loss = compute_entropy_loss(
                            out["entropy"],
                            entropy_type=self.cfg.algorithm.entropy_type,
                            loss_mask=loss_mask,
                            action_dim=self.cfg.actor.model.get("action_dim", 7),
                            batch_size=out["logprobs"].shape[0],
                        )
                        loss = loss - self.cfg.algorithm.entropy_bonus * entropy_loss

                    loss = loss / (global_batch_rows / micro_batch_rows)
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/entropy_loss"] = float(
                        entropy_loss.detach().item()
                    )
                    metrics_data["actor/total_loss"] = float(loss.detach().item())
                    append_to_dict(metrics, metrics_data)

                torch.cuda.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                extra_metrics = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    extra_metrics["critic/lr"] = lr_list[1]
                append_to_dict(metrics, extra_metrics)

        self.lr_scheduler.step()
        self.optimizer.zero_grad()
        clear_memory()

        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {k: float(np.mean(v)) for k, v in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict,
            op=torch.distributed.ReduceOp.AVG,
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats,
                op=torch.distributed.ReduceOp.SUM,
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )
        return mean_metric_dict
