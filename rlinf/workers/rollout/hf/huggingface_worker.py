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
import copy
import gc
import time
from dataclasses import replace
from typing import Any, Callable, Collection, Hashable, Iterable, Literal, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.prefix_off_policy.config import is_prefix_off_policy_loss
from rlinf.algorithms.prefix_off_policy.rollout import PrefixRolloutRuntime
from rlinf.algorithms.prefix_off_policy.transition import (
    extract_prefix_obs_from_forward_inputs,
)
from rlinf.algorithms.rlt import build_rlt_route, predict_rlt_actions
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_types import (
    ActionPayload,
    EnvPart,
    PolicyInput,
    PolicyOutput,
    PolicyPart,
    TrajectoryKey,
    TrajectorySource,
    batch_format,
    merge_batch_values,
    split_batch_value,
)
from rlinf.envs.layout import EnvLayout, EnvSpec
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.scheduler import (
    Channel,
    Cluster,
    CommMapper,
    Worker,
    split_channel_message,
)
from rlinf.utils.obs_compression import decompress_obs, infer_obs_batch_size
from rlinf.utils.pinned_offload import pinned_offload_context
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.utils.utils import seed_everything


class MultiStepRolloutWorker(Worker):
    """Hugging Face rollout worker for multi-step embodied interaction.

    Owns the inference copy of an embodied policy and exchanges observations
    and action chunks with env workers over pipelined channels. Synchronizes
    updated weights from the actor and emits trajectory segments for training.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.only_eval = cfg.runner.get("only_eval", False)
        self.algorithm_cfg = cfg.get("algorithm", {})
        self.model_cfg = cfg.rollout.model if self.only_eval else cfg.actor.model
        self.actor_group_name = (
            cfg.actor.get("group_name", None)
            if cfg.get("actor", None) is not None
            else None
        )
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0
        train_env_cfg = cfg.env.get("train", None)
        eval_env_cfg = cfg.env.get("eval", None)
        self.enable_train = not self.only_eval and train_env_cfg is not None
        self.enable_eval = (
            cfg.runner.get("val_check_interval", -1) > 0 or self.only_eval
        )
        env_world_size = self.placement.get_world_size("env")
        self.train_layout = (
            EnvLayout(train_env_cfg, env_world_size, "env.train")
            if self.enable_train
            else None
        )
        self.eval_layout = (
            EnvLayout(eval_env_cfg, env_world_size, "env.eval")
            if self.enable_eval
            else None
        )
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.enable_dagger = self.algorithm_cfg.get("loss_type") == "embodied_dagger"
        self.enable_opd = self.algorithm_cfg.get("adv_type") == "opd"
        self.enable_rlt = self.algorithm_cfg.get("loss_type") in {"rlt_ac", "rlt_td3"}
        self.enable_prefix = is_prefix_off_policy_loss(
            self.algorithm_cfg.get("loss_type")
        )
        self.batch_bootstrap_inference = bool(
            self.cfg.rollout.get("batch_bootstrap_inference", False)
        )
        self.value_only_bootstrap = bool(
            self.cfg.rollout.get("value_only_bootstrap", False)
        )
        self.expert_model = None
        self.prefix_runtime = None
        self.rlt_feature_model = None
        self.rlt_route = None

        self.total_num_train_envs = (
            cfg.env.train.total_num_envs if self.enable_train else 0
        )
        # Trajectories one training step produces over all env configurations.
        self.train_trajectories_per_step = (
            sum(
                spec.cfg.total_num_envs * spec.cfg.rollout_epoch
                for spec in self.train_layout.specs
            )
            if self.enable_train
            else 0
        )
        self.total_num_eval_envs = (
            cfg.env.eval.total_num_envs if self.enable_eval else 0
        )
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = self.total_num_train_envs // self.num_pipeline_stages
        self.eval_batch_size = self.total_num_eval_envs // self.num_pipeline_stages

        self.per_node_train_batch_size = (
            self.train_batch_size // self._world_size if self.enable_train else 0
        )
        self.per_node_eval_batch_size = (
            self.eval_batch_size // self._world_size if self.enable_eval else 0
        )

        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)

        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.collect_final_values = self.cfg.rollout.get("collect_final_values", True)
        self.version = 0
        self.global_step = 0
        self.finished_episodes = None

        self.weight_syncer = None
        self._sync_weight_comm_options = None
        if not self.only_eval:
            weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
            assert weight_syncer_cfg is not None, (
                "rollout.weight_syncer config must be provided"
            )
            self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
            self._sync_weight_comm_options = self.weight_syncer.comm_options

        self.env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if self.env_decoupled_mode and self.enable_train:
            # Every source sends one action request per chunk step plus one
            # epoch-closing final message on route_key=step_id.
            requests_per_step = sum(
                len(spec.ranks)
                * self.num_pipeline_stages
                * spec.cfg.rollout_epoch
                * (spec.chunk_steps(self.model_cfg.num_action_chunks) + 1)
                for spec in self.train_layout.specs
            )
            self.decoupled_requests_per_step = self._decoupled_share(
                requests_per_step, self._world_size, self._rank
            )

        if self.env_decoupled_mode:
            # save the run-time imformation in communicate channel for decoupled mode
            # The batch_router is a dictionary that maps the tag to the list of batch_index.
            self.batch_router = {
                "policy": [],
                "rollout_results": [],
            }
        self._decoupled_step = 0
        self.rollout_queue_size = self.cfg.rollout.get("rollout_queue_size", 0)

    @property
    def collect_raw_transition_obs(self) -> bool:
        """Return whether Actor training requires raw transition observations."""
        return self.collect_transitions and not (self.enable_prefix or self.enable_rlt)

    def init_worker(self):
        seed = self.cfg.rollout.get("seed")
        if self.enable_train and seed is not None:
            seed_everything(seed + self._rank)
        if self.enable_prefix:
            rollout_model_config = OmegaConf.merge(
                OmegaConf.to_container(self.model_cfg, resolve=True),
                OmegaConf.to_container(self.cfg.rollout.model, resolve=True),
            )
        else:
            rollout_model_config = OmegaConf.merge(
                self.model_cfg, self.cfg.rollout.model
            )

        self.hf_model: BasePolicy = get_model(rollout_model_config)
        if self.value_only_bootstrap and getattr(
            type(self.hf_model), "predict_value_batch", None
        ) in (None, BasePolicy.predict_value_batch):
            raise ValueError(
                "rollout.value_only_bootstrap=True requires a policy implementing "
                f"predict_value_batch, got {type(self.hf_model).__name__}."
            )

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        self.prefix_runtime = PrefixRolloutRuntime.build(
            self.cfg, self.hf_model, device=f"{self.torch_device_type}:{self.device}"
        )

        rlt_feature_model_config = OmegaConf.select(
            self.cfg, "rollout.rlt_feature_model", default=None
        )
        if rlt_feature_model_config is not None:
            self.rlt_feature_model = get_model(copy.deepcopy(rlt_feature_model_config))
            self.rlt_feature_model.eval()
            self.rlt_feature_model.requires_grad_(False)
            self.rlt_route = build_rlt_route(self.cfg)

        if (
            self.cfg.rollout.get("expert_model", None)
            and not self.enable_opd
            and self.prefix_runtime is None
        ):
            expert_model_config = build_expert_model_config(
                self.cfg,
                self.model_cfg,
                rlt_feature_model_config=rlt_feature_model_config,
            )
            self.expert_model = get_model(expert_model_config)

            if self.cfg.runner.get("expert_ckpt_path", None):
                expert_model_dict = torch.load(self.cfg.runner.expert_ckpt_path)
                self.expert_model.load_state_dict(expert_model_dict)

        self.hf_model.eval()
        if self.expert_model is not None:
            self.expert_model.eval()

        if self.cfg.rollout.get("enable_torch_compile", False):
            mode = self.cfg.rollout.get(
                "torch_compile_mode", "max-autotune-no-cudagraphs"
            )
            self.hf_model.enable_torch_compile(mode=mode)
        if self.enable_cuda_graph and not self.enable_offload:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def setup_sample_params(self):
        # sampling parameters for rollout
        sampling_params = self.cfg.rollout.get("sampling_params", None)
        if sampling_params is not None:
            sampling_params = OmegaConf.to_container(sampling_params, resolve=True)
            self._train_sampling_params = {
                "do_sample": sampling_params["do_sample"],
                "temperature": sampling_params["temperature_train"]
                if sampling_params["do_sample"]
                else 1.0,
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
            self._eval_sampling_params = {
                "do_sample": True
                if sampling_params.get("temperature_eval", -1) > 0
                else False,
                "temperature": sampling_params["temperature_eval"],
                "top_k": sampling_params["top_k"],
                "top_p": sampling_params["top_p"],
                "max_new_tokens": sampling_params["max_new_tokens"],
            }
        else:
            self._train_sampling_params = {}
            self._eval_sampling_params = {}

        if self.expert_model is not None and self.enable_dagger:
            init_beta = self.algorithm_cfg.get("dagger", {}).get("init_beta", 0.5)
            self._dagger_sampling_params = {
                "init_beta": init_beta,
                "beta": init_beta,
                "beta_schedule": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.algorithm_cfg.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.algorithm_cfg.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    async def _take_decoupled_shard(
        self,
        channel: Any,
        key: Any,
        timeout: float | None,
    ) -> Any | None:
        """Take one routed shard, or ``None`` when a finite wait expires.

        A finite wait that expires leaves its ``channel.get`` in flight. The next
        gather consumes that same get. A second get on this key would wait for a
        later observation, while the env workers that produced the first one stay
        blocked until its actions are sent.
        """
        inflight = getattr(self, "decoupled_inflight_get_tasks", None)
        if inflight is None:
            inflight = {}
            self.decoupled_inflight_get_tasks = inflight
        task = inflight.get(key)
        if task is None:
            work = channel.get(key=key, async_op=True)
            task = asyncio.create_task(work.async_wait())
            inflight[key] = task

        if timeout is None:
            done, _ = await asyncio.wait({task})
        else:
            done, _ = await asyncio.wait({task}, timeout=timeout)
        if task not in done:
            return None
        inflight.pop(key, None)
        return task.result()

    async def recv_from_and_record_batch_routes_with_timeout(
        self,
        group_name: str,
        channel: Any | None,
        *,
        route_key: Any = None,
        tag: str | None = None,
        batch_size: int | None = None,
        merge_fn: Optional[Callable[[list[Any]], Any]] = None,
        infer_batch_size_fn: Optional[Callable[[Any], int]] = None,
        timeout_time: float = 0.02,
        recv_queue_size: int = 0,
        max_items: int | None = None,
    ):
        """Receive routed batch shards and record their return routes.

        Used in env-decoupled mode. The first shard waits until it arrives.
        Further shards are collected for at most ``timeout_time`` seconds after
        that. Shards already in hand are then returned so their actions can be
        sent. A receive still in flight at the deadline is kept and becomes the
        first shard of a later call.

        Each received channel item must be a dict with ``batch_index`` (route
        metadata) and ``batch`` (payload shard).

        The ``batch_index`` values are stored in ``self.batch_router[tag]`` so a
        later send call can split the response and send each shard back to the original
        source rank. The received payload shards are validated, then merged with
        ``merge_fn`` or the default ``merge_batches``.

        Args:
            group_name: Source worker group name.
            channel: Channel used to receive routed batch shards.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag used to build receive keys and index recorded routes.
            batch_size: Expected batch size for each planned receive entry.
            merge_fn: Optional custom function for merging received shards.
            infer_batch_size_fn: Optional function used to infer shard batch size during
                validation.
            timeout_time: Seconds to keep collecting shards after the first one
                arrives. The first shard is not limited by this deadline.
            recv_queue_size: Number of receive queue entries used when building the
                receive plan.
            max_items: Optional upper bound on the shards pulled in this call,
                capped by the receive plan size.

        Returns:

            A merged payload and its split sizes. If only one shard is received, the
            current implementation returns that shard directly.
        """
        from rlinf.scheduler import (
            decoupled_build_recv_plan,
            get_batch_size,
            get_group_world_size,
            merge_batches,
        )

        world_size = get_group_world_size(self._manager_proxy, group_name)
        plan = decoupled_build_recv_plan(
            src_group_name=group_name,
            dst_group_name=self.worker_address.root_group_name,
            recv_rank=None,
            src_world_size=self._world_size,
            dst_world_size=world_size,
            tag=tag,
            route_key=route_key,
            batch_size=batch_size,
            recv_queue_size=recv_queue_size,
        )

        def _finalize(received_items: list[Any]):
            if not received_items:
                assert False, "received_items is empty"

            # get the tag from the received_items
            _, _, _, tag = split_channel_message(received_items[0]["batch_index"])

            assert tag in self.batch_router, (
                f"{tag=} need to be already in the batch_router"
            )
            # Save the batch_index to the batch_router.
            list_received_items = []
            for item in received_items:
                batch_index = item["batch_index"]
                received_item = item["batch"]
                list_received_items.append(received_item)
                # Save the batch_index to the batch_router.
                self.batch_router[tag].append(batch_index)
            received_items = list_received_items
            split_sizes = [
                get_batch_size(item, infer_batch_size_fn) for item in received_items
            ]

            if merge_fn is not None:
                return merge_fn(received_items), split_sizes
            if len(received_items) == 1:
                return received_items[0]
            return merge_batches(received_items), split_sizes

        key = plan.entries[0].key
        max_item_num = len(plan.entries)
        if max_items is not None:
            max_item_num = min(max_item_num, max_items)
        received_items: list[Any] = []
        deadline: float | None = None
        while len(received_items) < max_item_num:
            if deadline is not None and time.monotonic() >= deadline:
                break
            wait = None if deadline is None else deadline - time.monotonic()
            item = await self._take_decoupled_shard(channel, key, wait)
            if item is None:
                break
            received_items.append(item)
            if deadline is None:
                deadline = time.monotonic() + timeout_time

        return _finalize(received_items)

    def send_to_recorded_batch_routes(
        self,
        group_name: str,
        channel: Any | None,
        data: Any,
        *,
        route_key: Any = None,
        tag: str | None = None,
        split_fn: Optional[Callable[[Any, list[int]], list[Any]]] = None,
        split_sizes: list[int],
    ):
        """Send split batch results back using recorded batch routes.

        This method is used after a previous receive call has populated
        ``self.batch_router[tag]`` with batch indices from incoming messages.
        The outgoing ``data`` is split according to ``split_sizes`` and each shard is
        sent to the rank encoded in the corresponding recorded batch index.

        Each outgoing channel item is a dict with ``batch_index`` (recorded batch
        index) and ``batch`` (payload shard).

        After all shards are queued, the recorded batch indices for ``tag`` are cleared
        to avoid reusing stale routes.

        Args:
            group_name: Destination worker group name.
            channel: Channel used to send the split payloads.
            data: Payload to split and send.
            route_key: Optional key used to separate independent routed streams.
            tag: Routing tag whose recorded batch indices should be consumed.
            split_fn: Optional custom splitter. If omitted, ``split_batch`` is used.
            split_sizes: Batch sizes used to split ``data``. Must have the same length
                as ``self.batch_router[tag]``.

        Returns:

            AsyncRouteWork wrapping the async channel put operations.
        """
        from rlinf.scheduler import build_send_key, split_batch
        from rlinf.scheduler.collective import AsyncRouteWork

        assert tag in self.batch_router, (
            f"{tag=} need to be already in the batch_router"
        )

        assert len(self.batch_router[tag]) > 0, f"{self.batch_router[tag]=} is empty"
        assert len(self.batch_router[tag]) == len(split_sizes), (
            f"{self.batch_router[tag]=} length should equal {split_sizes=} length"
        )

        payloads = (
            split_fn(data, split_sizes)
            if split_fn is not None
            else split_batch(data, split_sizes)
        )

        works = []
        for i, payload in enumerate(payloads):
            batch_index = self.batch_router[tag][i]
            send_rank, _, mode, _ = split_channel_message(batch_index)
            # After enabling env_decoupled_mode, the data sending format is as follows:
            # {
            #     "batch_index": batch_index,
            #     "batch": batch,
            # }
            # The batch_index is the index of the batch in the data.
            # The batch is the data to send.
            # batch_index: {send_rank}_{batch_idx}_{mode}_{tag}
            # The send_rank is the rank of the worker that originally sent the data.
            # The batch_idx is the index of the batch in the data.
            # The tag is the tag of the data.
            senditem = {
                "batch_index": batch_index,
                "batch": payload,
            }
            key = build_send_key(
                src_group_name=self.worker_address.root_group_name,
                dst_group_name=group_name,
                src_rank=None,
                dst_rank=send_rank,
                tag=tag if mode is None else f"{mode}_{tag}",
                route_key=route_key,
            )
            work = channel.put(
                item=senditem,
                key=key,
                async_op=True,
            )
            works.append(work)

        # clear the batch_router for the next send
        self.batch_router[tag] = []
        return AsyncRouteWork(works, lambda _: None)

    def update_dagger_beta(self):
        if self.expert_model is None or not self.enable_dagger:
            return

        if self._dagger_sampling_params["beta_schedule"] == "exponential":
            self._dagger_sampling_params["beta"] = max(
                self._dagger_sampling_params["beta_min"],
                self._dagger_sampling_params["beta"]
                * self._dagger_sampling_params["beta_decay"],
            )
        else:
            raise NotImplementedError(
                f"Beta schedule {self._dagger_sampling_params['beta_schedule']} is not implemented"
            )

    def _dagger_beta_for_source(self, source: Any) -> float | None:
        """Compute DAgger expert-sampling beta from the source's own epoch."""
        if self.expert_model is None or not self.enable_dagger:
            return None
        schedule = self._dagger_sampling_params["beta_schedule"]
        if schedule != "exponential":
            raise NotImplementedError(f"Beta schedule {schedule} is not implemented")
        spec = self.train_layout.spec_of_rank(source.key.env_rank)
        epoch_index = (
            source.key.step_id * spec.cfg.rollout_epoch + source.key.epoch_id + 1
        )
        init_beta = self._dagger_sampling_params["init_beta"]
        return max(
            self._dagger_sampling_params["beta_min"],
            init_beta * (self._dagger_sampling_params["beta_decay"] ** epoch_index),
        )

    @Worker.timer("predict")
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        policy_kwargs: dict[str, Any] | None = None,
        dagger_beta: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Act on one batch of observations.

        Args:
            env_obs: Observations of envs that share their policy fields.
            mode: Whether the batch comes from training or evaluation envs.
            policy_kwargs: The envs' :attr:`EnvSpec.POLICY_FIELDS`, passed to
                the policy's ``predict_action_batch``.
        """
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        model_type = SupportedModel(self.model_cfg.model_type)
        if model_type in [
            SupportedModel.OPENPI,
            SupportedModel.PI0_FAST,
            SupportedModel.GO2_POLICY,
            SupportedModel.EVO1,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
            SupportedModel.MOLMOACT2,
            SupportedModel.PSI0,
            SupportedModel.LINGBOTVLAV2,
        ]:
            if self.enable_dagger:
                kwargs = {"mode": "eval"}
            elif model_type in (SupportedModel.PI0_FAST, SupportedModel.GO2_POLICY):
                kwargs = dict(kwargs)
                kwargs["mode"] = mode
            else:
                kwargs = {"mode": mode}
        if policy_kwargs:
            kwargs = {**kwargs, **policy_kwargs}

        if model_type in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.algorithm_cfg.get("dagger", {}).get(
            "only_save_expert", True
        )

        if mode == "train" and self.expert_model is not None and self.enable_dagger:
            # training with expert model. Beta-probability acting.
            beta = (
                self._dagger_sampling_params["beta"]
                if dagger_beta is None
                else dagger_beta
            )
            use_expert = torch.rand(1).item() < beta
        else:
            use_expert = False

        with torch.no_grad():
            expert_label_flag = False
            # Decide which model to act via use_expert
            if use_expert:
                actions, result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_label_flag = True
            else:
                actions, result = self.hf_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )

            # Decide re-label or not
            if (
                not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self.expert_model is not None  # only re-label if expert exists
                and self.enable_dagger  # only re-label in DAgger mode
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self.expert_model.predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs["model_action"]
                expert_action = expert_forward_inputs["action"]
                if expert_target is not None:
                    result["forward_inputs"]["action"] = expert_action
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def _predict_rollout_actions(
        self,
        request: PolicyInput,
        mode: Literal["train", "eval"] = "train",
        stage_id: int = 0,
        policy_kwargs: dict[str, Any] | None = None,
        dagger_beta: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.prefix_runtime is not None:
            if mode == "train" and self.prefix_runtime.gate is not None:
                resets = [
                    source.key.chunk_id == 0
                    and not self.train_layout.spec_of_rank(source.key.env_rank).cfg.get(
                        "auto_reset", False
                    )
                    for source in request.sources
                ]
                if any(resets):
                    batch_size = self._infer_policy_input_batch_size(request)
                    reset_mask = torch.zeros(batch_size, dtype=torch.bool)
                    if request.dones is not None:
                        reset_mask |= (
                            request.dones.reshape(batch_size, -1).any(dim=1).cpu()
                        )
                    offset = 0
                    for source, reset in zip(request.sources, resets):
                        if reset:
                            reset_mask[offset : offset + source.size] = True
                        offset += source.size
                    request = replace(request, dones=reset_mask)
            return self.prefix_runtime.predict(
                request, mode=mode, version=self.version, stage_id=stage_id
            )
        if self.rlt_feature_model is not None:
            return predict_rlt_actions(
                policy_model=self.hf_model,
                feature_model=self.rlt_feature_model,
                rlt_route=self.rlt_route,
                env_obs=request.obs,
                final_obs=request.final_obs,
                mode=mode,
                version=self.version,
                rlt_switch_flags=request.rlt_switch_flags,
                intervene_requested=request.intervene_flags,
                expert_model=self.expert_model,
            )
        return self.predict(
            request.obs,
            mode=mode,
            policy_kwargs=policy_kwargs,
            dagger_beta=dagger_beta,
        )

    def _build_policy_output(
        self, actions: torch.Tensor, result: dict[str, Any]
    ) -> PolicyOutput:
        """Keep inference metadata locally until its environment result arrives."""
        intervene_flags = result.get("intervene_flags")
        if (
            intervene_flags is None
            and self.enable_dagger
            and result.get("expert_label_flag", False)
        ):
            intervene_flags = torch.ones(
                (actions.shape[0], self.model_cfg.num_action_chunks),
                dtype=torch.bool,
                device=actions.device,
            )
        return PolicyOutput(
            prev_logprobs=result["prev_logprobs"] if self.collect_prev_infos else None,
            prev_values=result["prev_values"] if self.collect_prev_infos else None,
            intervene_flags=intervene_flags,
            forward_inputs=result["forward_inputs"],
            trajectory_infos=result.get("trajectory_infos", {}),
            versions=torch.full_like(
                result["prev_logprobs"], float(self.version), dtype=torch.float32
            )
            if self.collect_prev_infos
            else None,
        )

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self):
        """Sync model parameters from the actor worker."""

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=self.hf_model.state_dict(),
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = self.version * self.train_trajectories_per_step
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @classmethod
    def _infer_policy_input_batch_size(cls, policy_input: PolicyInput) -> int:
        """Infer the batch size of a routed policy input."""
        return cls._infer_env_batch_size(policy_input.obs)

    @staticmethod
    def _decompress_policy_input(policy_input: PolicyInput) -> PolicyInput:
        """Restore compressed current and transition observations in place."""
        policy_input.obs = decompress_obs(policy_input.obs)
        for env_part in policy_input.env_parts:
            if env_part is not None and env_part.next_obs is not None:
                env_part.next_obs = decompress_obs(env_part.next_obs)
        return policy_input

    @classmethod
    def _decompress_policy_inputs(
        cls, policy_inputs: list[PolicyInput]
    ) -> list[PolicyInput]:
        """Decompress routed requests, keeping them apart for format grouping."""
        return [cls._decompress_policy_input(item) for item in policy_inputs]

    async def _receive_decoupled_policy_inputs(
        self,
        channel: Channel,
        *,
        route_key: Any = None,
        max_items: int | None = None,
    ) -> tuple[list[PolicyInput], list[int]]:
        """Receive one round of decompressed decoupled requests, in route order."""
        return await self.recv_from_and_record_batch_routes_with_timeout(
            group_name=self.cfg.env.group_name,
            channel=channel,
            route_key=route_key,
            tag="policy",
            batch_size=self.train_batch_size,
            merge_fn=self._decompress_policy_inputs,
            infer_batch_size_fn=self._infer_policy_input_batch_size,
            timeout_time=0.02,
            recv_queue_size=self.rollout_queue_size,
            max_items=max_items,
        )

    def _can_predict_value_only(self) -> bool:
        if isinstance(self.hf_model, BasePolicy):
            return self.hf_model.supports_value_only_bootstrap()
        return callable(getattr(self.hf_model, "predict_value_batch", None))

    def _predict_bootstrap(
        self, env_obs: dict[str, Any], policy_kwargs: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if self.model_cfg.model_type == SupportedModel.PSI0.value or (
            self.value_only_bootstrap
            and not (self.enable_prefix or self.enable_rlt)
            and not self.enable_dagger
            and self.prefix_runtime is None
            and self.rlt_feature_model is None
            and self._can_predict_value_only()
        ):
            return {"prev_values": self.hf_model.predict_value_batch(env_obs)}
        _, result = self._predict_rollout_actions(
            PolicyInput(obs=env_obs), policy_kwargs=policy_kwargs
        )
        return result

    def _publish_env_part(
        self,
        env_part: EnvPart,
        policy_obs: dict[str, Any] | None,
        next_rlt_obs: dict[str, Any] | None,
        actor_channel: Channel,
        bootstrap_result: dict[str, Any] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        next_obs = env_part.next_obs if env_part.next_obs is not None else policy_obs
        final_prev_values = None
        # Terminal inference contributes values or transition features only.
        if self.prefix_runtime is not None:
            next_rlt_obs = self.prefix_runtime.transition_obs(
                env_part, policy_obs, next_rlt_obs
            )
        elif env_part.requires_inference:
            if next_obs is None:
                raise ValueError(
                    "Terminal inference requires a post-action observation."
                )
            result = (
                bootstrap_result
                if bootstrap_result is not None
                else self._predict_bootstrap(next_obs, policy_kwargs)
            )
            values = result.get("prev_values")
            if (
                self.enable_rlt
                and self.model_cfg.model_type != SupportedModel.PSI0.value
            ):
                next_rlt_obs = extract_rlt_obs_from_forward_inputs(
                    result.get("forward_inputs"), transition=True
                )
            if values is not None:
                final_prev_values = values.cpu().contiguous()
        # Republish environment-owned data here to keep one Actor-channel writer.
        actor_channel.put(
            env_part.complete(
                next_obs=(next_obs if self.collect_raw_transition_obs else None),
                next_rlt_obs=next_rlt_obs
                if (self.enable_prefix or self.enable_rlt)
                else None,
                final_prev_values=final_prev_values,
            ),
            async_op=True,
        )

    def _publish_env_parts(
        self,
        policy_input: PolicyInput,
        forward_inputs: dict[str, Any] | None,
        actor_channel: Channel,
        policy_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """Complete attached environment parts without duplicating request data.

        ``policy_kwargs`` are the policy fields of the envs that sent
        ``policy_input``, for their terminal inference.
        """
        if not policy_input.env_parts:
            return
        if len(policy_input.env_parts) != len(policy_input.request_sizes):
            raise ValueError("Environment parts and request sizes must align.")

        if self.prefix_runtime is not None and policy_input.is_last:
            self.prefix_runtime.prepare_transitions(policy_input)
        part_count = len(policy_input.env_parts)
        next_rlt_obs = (
            extract_prefix_obs_from_forward_inputs(forward_inputs, transition=True)
            if self.enable_prefix and forward_inputs is not None
            else None
        )
        if self.enable_rlt and forward_inputs is not None:
            next_rlt_obs = extract_rlt_obs_from_forward_inputs(
                forward_inputs, transition=True
            )
        next_rlt_observations = (
            split_batch_value(next_rlt_obs, policy_input.request_sizes)
            if next_rlt_obs is not None
            else [None] * part_count
        )
        needs_policy_obs = any(
            part is not None
            and (
                self.collect_raw_transition_obs
                or (part.requires_inference and part.next_obs is None)
            )
            for part in policy_input.env_parts
        )
        policy_observations = (
            split_batch_value(policy_input.obs, policy_input.request_sizes)
            if needs_policy_obs
            else [None] * part_count
        )
        bootstrap_results = [None] * part_count
        if (
            self.batch_bootstrap_inference
            and not (self.enable_prefix or self.enable_rlt)
            and not self.enable_dagger
            and self.prefix_runtime is None
        ):
            indices = [
                index
                for index, part in enumerate(policy_input.env_parts)
                if part is not None and part.requires_inference
            ]
            if len(indices) > 1:
                observations = [
                    policy_input.env_parts[index].next_obs
                    if policy_input.env_parts[index].next_obs is not None
                    else policy_observations[index]
                    for index in indices
                ]
                if any(obs is None for obs in observations):
                    raise ValueError(
                        "Terminal inference requires a post-action observation."
                    )
                # The parts share policy_kwargs, so their format decides a pass.
                obs_groups = self._group_indices(
                    batch_format(obs) for obs in observations
                )
                for positions in obs_groups.values():
                    if len(positions) <= 1:
                        continue
                    group_indices = [indices[position] for position in positions]
                    result = self._predict_bootstrap(
                        merge_batch_values([observations[p] for p in positions]),
                        policy_kwargs,
                    )
                    results = split_batch_value(
                        {"prev_values": result.get("prev_values")},
                        [policy_input.request_sizes[index] for index in group_indices],
                    )
                    for index, split_result in zip(group_indices, results):
                        bootstrap_results[index] = split_result
        for env_part, policy_obs, next_features, bootstrap_result in zip(
            policy_input.env_parts,
            policy_observations,
            next_rlt_observations,
            bootstrap_results,
        ):
            if env_part is not None:
                self._publish_env_part(
                    env_part,
                    policy_obs,
                    next_features,
                    actor_channel,
                    bootstrap_result,
                    policy_kwargs,
                )

    def _send_actions(
        self,
        channel: Channel,
        actions: torch.Tensor | ActionPayload,
        stage_id: int,
        split_sizes: list[int] | None,
        diagnostics: dict[str, Any] | None = None,
        *,
        peer_ranks: Collection[int] | None = None,
    ) -> None:
        # Send by value on CPU: a device-resident tensor reaches a same-device env
        # as a CUDA IPC handle aliasing the buffer the next forward pass reuses.
        payload = (
            actions
            if isinstance(actions, ActionPayload)
            else self._build_action_payload(actions, diagnostics)
        )
        split_fn = self._split_actions
        if isinstance(payload, ActionPayload):
            split_fn = self._split_action_payload
        if self.env_decoupled_mode:
            if split_sizes is None:
                raise ValueError("Decoupled policy output requires recorded routes.")
            self.send_to_recorded_batch_routes(
                group_name=self.cfg.env.group_name,
                channel=channel,
                data=payload,
                tag="policy",
                split_fn=split_fn,
                split_sizes=split_sizes,
            )
            return
        self.send_to(
            group_name=self.cfg.env.group_name,
            channel=channel,
            data=payload,
            tag="policy",
            route_key=stage_id,
            async_op=True,
            batch_size=self.train_batch_size,
            split_fn=split_fn,
            peer_ranks=peer_ranks,
        )

    def _build_action_payload(
        self,
        actions: torch.Tensor,
        diagnostics: dict[str, Any] | None,
    ) -> torch.Tensor | ActionPayload:
        """Build optional runtime feedback without changing plain action transport."""
        if self.prefix_runtime is not None:
            return self.prefix_runtime.action_payload(actions, diagnostics)
        return actions.detach().cpu().contiguous()

    def _serve_policy_inputs(
        self,
        requests: list[PolicyInput],
        output_channel: Channel,
        actor_channel: Channel,
        stage_id: int,
        split_sizes: list[int] | None = None,
        peer_ranks: list[int] | None = None,
    ) -> None:
        """Answer one round of policy requests and publish their trajectory parts.

        Requests share one forward pass when their observations share a format
        and their envs share policy fields, so the model never sees two of
        either in one batch. The inferred actions go back in request order; a
        request that supplied its own actions gets no reply, since its env runs
        them without waiting.

        Args:
            requests: Decompressed requests that each need or supply actions,
                in route order.
            output_channel: Channel the actions go back on.
            actor_channel: Channel the trajectory parts go to.
            stage_id: Pipeline stage of the requests.
            split_sizes: Recorded batch routes in decoupled mode.
            peer_ranks: Env ranks that sent the requests, with fixed routes.
        """
        groups = self._group_indices(
            (
                request.requires_inference,
                *self._forward_key(
                    self.train_layout,
                    {source.key.env_rank for source in request.sources},
                    request.obs,
                ),
                self._dagger_beta_for_source(request.sources[0]),
            )
            for request in requests
        )

        # The env runs the actions it supplied, so nothing goes back.
        replied = [
            index
            for index, request in enumerate(requests)
            if request.requires_inference
        ]
        single_reply_group = len(groups) == 1 and len(replied) == len(requests)
        request_actions: list[torch.Tensor | ActionPayload | None] | None = (
            None if single_reply_group else [None] * len(requests)
        )
        direct_actions: torch.Tensor | ActionPayload | None = None
        for (_, _, policy_fields, dagger_beta), indices in groups.items():
            policy_input = (
                requests[indices[0]]
                if len(indices) == 1
                else PolicyInput.merge([requests[index] for index in indices])
            )
            if not policy_input.requires_inference:
                actions = policy_input.external_actions
                forward_inputs = None
                policy_output = None
            else:
                actions, result = self._predict_rollout_actions(
                    policy_input,
                    stage_id=stage_id,
                    policy_kwargs=dict(policy_fields),
                    dagger_beta=dagger_beta,
                )
                forward_inputs = result.get("forward_inputs")
                policy_output = self._build_policy_output(actions, result)
            payload = self._build_action_payload(
                actions,
                result.get("rollout_infos")
                if policy_input.requires_inference
                else None,
            )
            if single_reply_group:
                direct_actions = payload
            else:
                sizes = [
                    self._infer_policy_input_batch_size(requests[index])
                    for index in indices
                ]
                for index, shard in zip(indices, split_batch_value(payload, sizes)):
                    request_actions[index] = shard
            # Omit raw observations when the collector does not consume them.
            transition_obs = policy_input.obs if self.collect_raw_transition_obs else {}
            if not policy_input.requires_inference:
                trajectory_part = PolicyPart(
                    sources=policy_input.sources,
                    obs=transition_obs,
                    external_actions=actions,
                )
            else:
                trajectory_part = PolicyPart(
                    sources=policy_input.sources,
                    obs=transition_obs,
                    output=policy_output,
                )
            # Publish model-owned data immediately; its environment half
            # arrives later through an EnvPart with the same key.
            actor_channel.put(trajectory_part, async_op=True)
            # Attached parts close previously executed chunks.
            self._publish_env_parts(
                policy_input, forward_inputs, actor_channel, dict(policy_fields)
            )

        if len(replied) < len(requests):
            if split_sizes is not None:
                split_sizes = [split_sizes[index] for index in replied]
            if peer_ranks is not None:
                peer_ranks = [peer_ranks[index] for index in replied]
        if replied:
            actions_to_send = (
                direct_actions
                if single_reply_group
                else merge_batch_values([request_actions[index] for index in replied])
            )
            self._send_actions(
                output_channel,
                actions_to_send,
                stage_id,
                split_sizes,
                peer_ranks=peer_ranks,
            )

    def _publish_final_inputs(
        self,
        finals: list[PolicyInput],
        final_ranks: list[int],
        actor_channel: Channel,
    ) -> None:
        """Publish epoch-closing final inputs without sending action replies."""
        if not finals:
            return
        final_keys = [
            self._forward_key(self.train_layout, [env_rank], item.obs)
            for env_rank, item in zip(final_ranks, finals)
        ]
        if self.batch_bootstrap_inference and len(finals) > 1:
            groups = self._group_indices(final_keys)
            for (_, policy_fields), indices in groups.items():
                group = [finals[index] for index in indices]
                merged = PolicyInput.merge(group) if len(group) > 1 else group[0]
                self._publish_env_parts(
                    merged, None, actor_channel, dict(policy_fields)
                )
        else:
            for item, (_, policy_fields) in zip(finals, final_keys):
                self._publish_env_parts(item, None, actor_channel, dict(policy_fields))

    @Worker.timer("serve_decoupled_step")
    async def _serve_decoupled_step(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Serve this worker's share of one training step's decoupled messages.

        Each training step routes on ``route_key=step_id``, so rollout workers
        only pull messages for the active step and finish every step's final
        transitions before returning.
        """
        served = 0
        while served < self.decoupled_requests_per_step:
            route_start = len(self.batch_router["policy"])
            requests, split_sizes = await self._receive_decoupled_policy_inputs(
                input_channel,
                route_key=self._decoupled_step,
                max_items=self.decoupled_requests_per_step - served,
            )
            routes = self.batch_router["policy"][route_start:]
            del self.batch_router["policy"][route_start:]

            active_requests: list[PolicyInput] = []
            active_sizes: list[int] = []
            active_routes: list[Any] = []
            finals: list[PolicyInput] = []
            final_ranks: list[int] = []
            for request, size, route in zip(requests, split_sizes, routes):
                if request.is_last:
                    finals.append(request)
                    final_ranks.append(split_channel_message(route)[0])
                else:
                    active_requests.append(request)
                    active_sizes.append(size)
                    active_routes.append(route)

            if active_requests:
                self.batch_router["policy"].extend(active_routes)
                self._serve_policy_inputs(
                    active_requests,
                    output_channel,
                    actor_channel,
                    0,
                    split_sizes=active_sizes,
                )
            self._publish_final_inputs(finals, final_ranks, actor_channel)
            served += len(requests)
        self._decoupled_step += 1

    def _source_budgets(
        self, layout: EnvLayout, batch_size: int, count: Callable[[EnvSpec], int]
    ) -> dict[int, dict[int, int]]:
        """Return, per pipeline stage, how many messages each routed env rank owes.

        Args:
            layout: Env layout of the mode being served.
            batch_size: Stage batch size the env-rollout routes are planned for.
            count: Messages one source of an env spec sends in one call.

        Returns:
            ``{stage_id: {env_rank: count}}`` for the env ranks routed here.
        """
        env_ranks = [
            rank
            for rank, _ in CommMapper.get_src_ranks(
                batch_size=batch_size,
                src_world_size=self.placement.get_world_size("env"),
                dst_world_size=self._world_size,
                dst_rank=self._rank,
            )
        ]
        return {
            stage_id: {rank: count(layout.spec_of_rank(rank)) for rank in env_ranks}
            for stage_id in range(self.num_pipeline_stages)
        }

    @staticmethod
    def _env_policy_fields(
        layout: EnvLayout, env_ranks: Iterable[int]
    ) -> tuple[tuple[str, Any], ...]:
        """Return the policy fields shared by the envs of ``env_ranks``.

        Args:
            layout: Env layout of the mode being served.
            env_ranks: Env worker ranks whose observations form one batch.

        Returns:
            The envs' :attr:`EnvSpec.policy_kwargs` as sorted items, which a
            batch can be grouped by.

        Raises:
            ValueError: If the envs set different policy fields.
        """
        fields = {
            tuple(sorted(layout.spec_of_rank(rank).policy_kwargs.items()))
            for rank in env_ranks
        }
        if len(fields) > 1:
            raise ValueError(
                "Expected the envs of one batch to set the same policy fields, "
                f"got {sorted(fields)} from env ranks {sorted(env_ranks)}."
            )
        return next(iter(fields), ())

    @classmethod
    def _forward_key(
        cls, layout: EnvLayout, env_ranks: Iterable[int], obs: Any
    ) -> tuple[Any, tuple[tuple[str, Any], ...]]:
        """Return what observations must share to run in one forward pass.

        Observations share a pass when they have one format and their envs set
        the same policy fields, so the policy never sees two of either.

        Args:
            layout: Env layout of the mode being served.
            env_ranks: Env worker ranks whose observations ``obs`` holds.
            obs: The observations.

        Returns:
            ``(observation format, policy fields)``.
        """
        return batch_format(obs), cls._env_policy_fields(layout, env_ranks)

    @staticmethod
    def _decoupled_share(requests: int, world_size: int, rank: int) -> int:
        """Return how many of a step's ``requests`` decoupled ``rank`` serves.

        The shares add up to ``requests``, and the first ranks take the
        remainder, so any env:rollout ratio splits a step.
        """
        share, remainder = divmod(requests, world_size)
        return share + int(rank < remainder)

    @staticmethod
    def _group_indices(keys: Iterable[Hashable]) -> dict[Hashable, list[int]]:
        """Group the positions of ``keys`` by key, in first-seen order."""
        groups: dict[Hashable, list[int]] = {}
        for index, key in enumerate(keys):
            groups.setdefault(key, []).append(index)
        return groups

    def _group_env_ranks(
        self, layout: EnvLayout, env_ranks: Iterable[int]
    ) -> dict[tuple[tuple[str, Any], ...], list[int]]:
        """Group ``env_ranks`` by their envs' policy fields, keeping their order."""
        env_ranks = list(env_ranks)
        groups = self._group_indices(
            self._env_policy_fields(layout, [rank]) for rank in env_ranks
        )
        return {
            fields: [env_ranks[index] for index in indices]
            for fields, indices in groups.items()
        }

    @Worker.timer("serve_step")
    async def _serve_step(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Serve fixed-route env sources until each has finished its training step.

        Every round takes one message from each source still in the step. The
        requests among them share one forward pass, and a final message closes
        its source's epoch. A source leaves the step after the final message of
        its last epoch, so each source runs its own epochs and chunk steps.
        """
        finals_left = self._source_budgets(
            self.train_layout,
            self.train_batch_size,
            lambda spec: spec.cfg.rollout_epoch,
        )
        while any(any(left.values()) for left in finals_left.values()):
            for stage_id, left in finals_left.items():
                # Received items follow the plan's peer order, which is ascending.
                peers = sorted(rank for rank, count in left.items() if count > 0)
                if not peers:
                    continue
                received: list[PolicyInput] = await self.recv_from(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="train_policy",
                    route_key=stage_id,
                    async_op=True,
                    batch_size=self.train_batch_size,
                    merge_fn=self._decompress_policy_inputs,
                    infer_batch_size_fn=self._infer_policy_input_batch_size,
                    peer_ranks=peers,
                ).async_wait()
                finals, final_ranks, requests, request_ranks = [], [], [], []
                for env_rank, item in zip(peers, received):
                    if not item.is_last:
                        requests.append(item)
                        request_ranks.append(env_rank)
                        continue
                    # The final message closes the last chunk without requesting actions.
                    finals.append(item)
                    final_ranks.append(env_rank)
                    left[env_rank] -= 1
                self._publish_final_inputs(finals, final_ranks, actor_channel)
                if not requests:
                    continue
                self._serve_policy_inputs(
                    requests,
                    output_channel,
                    actor_channel,
                    stage_id,
                    peer_ranks=request_ranks,
                )

    async def generate_one_step(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ) -> None:
        """Serve the policy requests of one training step."""
        if self.env_decoupled_mode:
            await self._serve_decoupled_step(
                input_channel, output_channel, actor_channel
            )
        else:
            await self._serve_step(input_channel, output_channel, actor_channel)

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
        actor_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()
        await self.generate_one_step(input_channel, output_channel, actor_channel)
        if self.enable_offload:
            self.offload_model()

    @Worker.timer("evaluate")
    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        if self.prefix_runtime is not None:
            self.prefix_runtime.reset("eval")
        if self.env_decoupled_mode:
            while True:
                (
                    shards,
                    split_sizes,
                ) = await self.recv_from_and_record_batch_routes_with_timeout(
                    group_name=self.cfg.env.group_name,
                    channel=input_channel,
                    tag="rollout_results",
                    batch_size=self.eval_batch_size,
                    merge_fn=list,
                    infer_batch_size_fn=self._infer_env_batch_size,
                    timeout_time=0.02,
                    recv_queue_size=self.rollout_queue_size,
                )
                shards = [decompress_obs(shard) for shard in shards]
                # The recorded routes are this call's shards, in order.
                senders = [
                    split_channel_message(route)[0]
                    for route in self.batch_router["rollout_results"]
                ]
                groups = self._group_indices(
                    self._forward_key(
                        self.eval_layout, [sender], shard.get("obs", shard)
                    )
                    for sender, shard in zip(senders, shards)
                )
                shard_actions: list[torch.Tensor | ActionPayload | None] = [None] * len(
                    shards
                )
                for (_, policy_fields), indices in groups.items():
                    actions = self._predict_eval_shards(
                        [shards[index] for index in indices],
                        policy_fields,
                        peer_ranks=[senders[index] for index in indices],
                    )
                    sizes = [split_sizes[index] for index in indices]
                    for index, shard in zip(indices, split_batch_value(actions, sizes)):
                        shard_actions[index] = shard
                payload = merge_batch_values(shard_actions)
                self.send_to_recorded_batch_routes(
                    group_name=self.cfg.env.group_name,
                    channel=output_channel,
                    data=payload,
                    tag="rollout_results",
                    split_sizes=split_sizes,
                    split_fn=(
                        self._split_action_payload
                        if isinstance(payload, ActionPayload)
                        else self._split_actions
                    ),
                )
        else:
            requests_left = self._source_budgets(
                self.eval_layout,
                self.eval_batch_size,
                lambda spec: (
                    spec.cfg.rollout_epoch
                    * spec.chunk_steps(self.model_cfg.num_action_chunks)
                ),
            )
            while any(any(left.values()) for left in requests_left.values()):
                for stage_id, left in requests_left.items():
                    peers = [rank for rank, count in left.items() if count > 0]
                    groups = self._group_env_ranks(self.eval_layout, peers)
                    for policy_fields, group in groups.items():
                        shards = await self.recv_from(
                            group_name=self.cfg.env.group_name,
                            channel=input_channel,
                            tag="eval_rollout_results",
                            route_key=stage_id,
                            async_op=True,
                            batch_size=self.eval_batch_size,
                            merge_fn=list,
                            infer_batch_size_fn=self._infer_env_batch_size,
                            peer_ranks=group,
                        ).async_wait()
                        if any(shard.get("eval_epoch_done") for shard in shards):
                            # The env workers ended this epoch early and await no
                            # actions; they owe only their remaining whole epochs.
                            for rank in group:
                                epoch = self.eval_layout.spec_of_rank(rank).chunk_steps(
                                    self.model_cfg.num_action_chunks
                                )
                                left[rank] = (left[rank] - 1) // epoch * epoch
                            continue
                        if self.prefix_runtime is not None:
                            for rank, shard in zip(group, shards):
                                spec = self.eval_layout.spec_of_rank(rank)
                                epoch_steps = spec.chunk_steps(
                                    self.model_cfg.num_action_chunks
                                )
                                if (
                                    not spec.cfg.get("auto_reset", False)
                                    and left[rank] % epoch_steps == 0
                                ):
                                    shard["dones"] = torch.ones(
                                        self._infer_env_batch_size(shard),
                                        dtype=torch.bool,
                                    )
                        payload = self._predict_eval_shards(
                            shards, policy_fields, peer_ranks=group, stage_id=stage_id
                        )
                        self.send_to(
                            group_name=self.cfg.env.group_name,
                            channel=output_channel,
                            data=payload,
                            split_fn=(
                                self._split_action_payload
                                if isinstance(payload, ActionPayload)
                                else self._split_actions
                            ),
                            tag="eval_rollout_results",
                            route_key=stage_id,
                            async_op=True,
                            batch_size=self.eval_batch_size,
                            peer_ranks=group,
                        )
                        for rank in group:
                            left[rank] -= 1

            if self.enable_offload:
                self.offload_model()

    def _predict_eval_shards(
        self,
        shards: list[dict[str, Any]],
        policy_fields: tuple[tuple[str, Any], ...],
        *,
        peer_ranks: list[int] | None = None,
        stage_id: int = 0,
    ) -> torch.Tensor | ActionPayload:
        """Predict compatible eval shards while retaining their source identities."""
        shards = [decompress_obs(shard) for shard in shards]
        formats = self._group_indices(
            batch_format(shard.get("obs", shard)) for shard in shards
        )
        shard_actions: list[torch.Tensor | ActionPayload | None] = [None] * len(shards)
        for indices in formats.values():
            sizes = [self._infer_env_batch_size(shards[index]) for index in indices]
            sources = []
            if peer_ranks is not None:
                for index, size in zip(indices, sizes):
                    sources.append(
                        TrajectorySource(
                            key=TrajectoryKey(0, 0, peer_ranks[index], stage_id, 0),
                            size=size,
                        )
                    )
            actions = self._predict_eval_actions(
                self._merge_obs_batches([shards[index] for index in indices]),
                policy_fields,
                sources=sources,
                stage_id=stage_id,
            )
            for index, shard in zip(indices, split_batch_value(actions, sizes)):
                shard_actions[index] = shard
        return merge_batch_values(shard_actions)

    def _predict_eval_actions(
        self,
        env_output: dict[str, Any],
        policy_fields: tuple[tuple[str, Any], ...] = (),
        *,
        sources: list[TrajectorySource] | None = None,
        stage_id: int = 0,
    ) -> torch.Tensor | ActionPayload:
        """Return executable eval actions and optional environment feedback."""
        actions, result = self._predict_rollout_actions(
            PolicyInput(
                obs=env_output["obs"],
                rlt_switch_flags=env_output.get("rlt_switch_flags"),
                intervene_flags=env_output.get("intervene_flags"),
                dones=env_output.get("dones"),
                sources=sources or [],
            ),
            mode="eval",
            stage_id=stage_id,
            policy_kwargs=dict(policy_fields),
        )
        return self._build_action_payload(actions, result.get("rollout_infos"))

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        with pinned_offload_context():
            self.hf_model.to("cpu")
            if self.prefix_runtime is not None:
                self.prefix_runtime.to("cpu")
            if self.expert_model is not None:
                self.expert_model.to("cpu")
            if self.rlt_feature_model is not None:
                self.rlt_feature_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.prefix_runtime is not None:
            self.prefix_runtime.to(self.device)
        if self.expert_model is not None:
            self.expert_model.to(self.device)
        if self.rlt_feature_model is not None:
            self.rlt_feature_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.per_node_train_batch_size,
                eval_batch_size=self.per_node_eval_batch_size,
            )

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        # Delegates to the shared helper, which also understands compressed
        # image markers (inference runs before decompression on the recv path).
        return infer_obs_batch_size(obs_batch)

    def _merge_optional_flag_tensors(
        self,
        obs_dicts: list[dict[str, Any]],
        flags_list: list[torch.Tensor | None],
    ) -> torch.Tensor | None:
        if not any(flags is not None for flags in flags_list):
            return None
        ref_flags = next(flags for flags in flags_list if flags is not None)
        filled_flags = []
        for obs_dict, flags in zip(obs_dicts, flags_list):
            if flags is None:
                batch_size = self._infer_env_batch_size(obs_dict)
                fill_shape = (batch_size, *ref_flags.shape[1:])
                filled_flags.append(torch.zeros(fill_shape, dtype=ref_flags.dtype))
            else:
                filled_flags.append(flags)
        return torch.cat(filled_flags, dim=0)

    def _merge_obs_batches(self, obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        # Reconstruct any image tensors compressed by the env workers. This is a
        # no-op when `env.obs_compression` is disabled (no compression markers).
        obs_batches = [decompress_obs(obs_batch) for obs_batch in obs_batches]
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]
        rlt_switch_flags_list = [
            obs_batch.get("rlt_switch_flags", None) for obs_batch in obs_batches
        ]
        intervene_flags_list = [
            obs_batch.get("intervene_flags", None) for obs_batch in obs_batches
        ]
        dones_list = [
            dones.reshape(self._infer_env_batch_size(obs_batch), -1).any(dim=1)
            if (dones := obs_batch.get("dones")) is not None
            else None
            for obs_batch in obs_batches
        ]

        def _merge_obs_dicts(dicts: list[dict[str, Any]]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for key in dicts[0].keys():
                values = [obs_dict[key] for obs_dict in dicts]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged[key] = torch.cat(values, dim=0)
                elif isinstance(first_non_none, list):
                    merged[key] = [item for sublist in values for item in sublist]
                elif isinstance(first_non_none, dict):
                    merged[key] = _merge_obs_dicts(values)
                else:
                    merged[key] = values
            return merged

        merged_obs = _merge_obs_dicts(obs_dicts)
        merged_final_obs = None
        if any(final_obs is not None for final_obs in final_obs_list):
            final_obs_or_obs = [
                final_obs if final_obs is not None else obs_dict
                for obs_dict, final_obs in zip(obs_dicts, final_obs_list)
            ]
            merged_final_obs = _merge_obs_dicts(final_obs_or_obs)

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "rlt_switch_flags": self._merge_optional_flag_tensors(
                obs_dicts, rlt_switch_flags_list
            ),
            "intervene_flags": self._merge_optional_flag_tensors(
                obs_dicts, intervene_flags_list
            ),
            "dones": self._merge_optional_flag_tensors(obs_dicts, dones_list),
            "eval_epoch_done": any(
                bool(obs_batch.get("eval_epoch_done")) for obs_batch in obs_batches
            ),
        }

    @staticmethod
    def _split_actions(actions: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
        """Split policy actions for the environment workers that requested them."""
        return [shard.contiguous() for shard in torch.split(actions, sizes, dim=0)]

    @staticmethod
    def _split_action_payload(
        payload: ActionPayload, sizes: list[int]
    ) -> list[ActionPayload]:
        """Split typed action feedback on the routed batch dimension."""
        return split_batch_value(payload, sizes)

    def set_global_step(self, global_step: int):
        self.global_step = global_step
        self._decoupled_step = global_step
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
