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

import math
from collections import defaultdict
from typing import Any, Iterator

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.expert import build_expert_model_config
from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss
from rlinf.algorithms.utils import compute_entropy_loss
from rlinf.config import SupportedModel
from rlinf.data.schema.embodied_trajectory import RolloutGeometry
from rlinf.data.schema.embodied_types import Trajectory, batch_format
from rlinf.data.storage.lerobot import resolve_lerobot_repo_id
from rlinf.envs.layout import EnvSpec
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.distributed import (
    all_reduce_dict,
)
from rlinf.utils.metric_utils import (
    CRITIC_EXPLAINED_VARIANCE_KEY,
    append_to_dict,
    compute_critic_explained_variance_from_stats,
    compute_group_success_metrics,
    compute_rollout_metrics,
    pop_critic_explained_variance_stats,
)
from rlinf.utils.nested_dict_process import (
    cat_list_of_dict_tensor,
    flatten_nested_tensor_time_batch,
    process_nested_dict_for_adv,
    process_nested_dict_for_train,
    put_tensor_device,
    select_nested_rows,
    split_dict_to_chunk,
    trim_nested_tensor_time_dim,
)
from rlinf.utils.placement import (
    HybridComponentPlacement,
)
from rlinf.utils.utils import (
    add_loss_weights,
    clear_memory,
    compute_trajectory_loss_mask,
    get_loss_agg_func,
    masked_mean,
)


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
        self._opd_teacher_model = None
        self.entropy_op_type = self.cfg.algorithm.get("entropy_op_type", "torch")
        self.loss_agg_func = get_loss_agg_func(
            self.cfg.algorithm.get("loss_agg_func", "token-mean")
        )
        self._group_success_metrics: dict[str, float] = {}

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
        self._rollout_geometry = RolloutGeometry.from_cfg(cfg)
        self._env_algorithm_cfgs = {
            spec.name: spec.algorithm_cfg(cfg.algorithm)
            for spec in self._rollout_geometry.layout.specs
        }

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
        """Receive this actor's trajectory shards and preprocess each env's batch.

        Args:
            input_channel: The input channel to read from.
        """
        clear_memory(sync=False)

        trajectories: dict[str | None, list[Trajectory]] = defaultdict(list)
        for _ in range(self._rollout_geometry.shards_per_actor()):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            trajectories[trajectory.env_name].append(trajectory)

        self.env_batches: dict[str | None, dict[str, Any]] = {}
        self._group_success_metrics = {}
        for spec in self._rollout_geometry.layout.specs:
            assert trajectories[spec.name], (
                "Expected trajectories of every env on each actor, but env "
                f"{spec.name!r} sent none; received envs {sorted(trajectories)}."
            )
            batch, metrics = self._process_received_rollout_batch(
                Trajectory.to_batch(trajectories[spec.name]), spec
            )
            self.env_batches[spec.name] = batch
            self._group_success_metrics.update(spec.scoped(metrics) or metrics)

    def _process_received_rollout_batch(
        self, rollout_batch: dict[str, torch.Tensor], spec: EnvSpec
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """Reshape one env's batch, then build its loss mask and reward filter.

        original shape: [rollout_epoch x n_chunk_steps, bsz, num_action_chunks, ...]
        target shape: [n_chunk_steps, rollout_epoch x bsz, num_action_chunks, ...]

        Args:
            rollout_batch: The env's stacked trajectories.
            spec: The env's spec.

        Returns:
            The processed batch and its group success metrics.
        """
        algorithm_cfg = self._env_algorithm_cfgs[spec.name]
        rollout_batch = process_nested_dict_for_adv(
            rollout_batch, spec.cfg.rollout_epoch
        )

        if spec.has_loss_mask:
            loss_mask, loss_mask_sum = compute_trajectory_loss_mask(
                rollout_batch["dones"], algorithm_cfg.reward_type
            )
            rollout_batch["loss_mask"] = loss_mask
            rollout_batch["loss_mask_sum"] = loss_mask_sum

        group_success_metrics = {}
        if algorithm_cfg.get("log_group_success_metrics", False):
            group_success_metrics = compute_group_success_metrics(
                rollout_batch["rewards"],
                group_size=algorithm_cfg.group_size,
                loss_mask=rollout_batch.get("loss_mask", None),
                rewards_lower_bound=algorithm_cfg.get("rewards_lower_bound", None),
                rewards_upper_bound=algorithm_cfg.get("rewards_upper_bound", None),
            )

        # filter data by rewards
        if algorithm_cfg.get("filter_rewards", False):
            rewards = rollout_batch[
                "rewards"
            ]  # [n_chunk_step, batch, num_action_chunks]
            if rollout_batch.get("loss_mask", None) is not None:
                rewards = rewards * rollout_batch["loss_mask"]
            n_chunk_step, batch_size, num_action_chunks = rewards.shape

            group_size = algorithm_cfg.group_size
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
                mean_reward_in_group >= algorithm_cfg.rewards_lower_bound
            ) & (
                mean_reward_in_group <= algorithm_cfg.rewards_upper_bound
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

        return rollout_batch, group_success_metrics

    @Worker.timer("actor/compute_adv")
    def compute_advantages_and_returns(self) -> dict[str, torch.Tensor]:
        """Compute each env's advantages and returns, then merge the envs.

        Returns:
            Rollout metrics over all envs, plus a ``<name>/`` copy for each env
            of a heterogeneous run.
        """
        normalize_jointly = self.cfg.algorithm.get("adv_norm_scope", "env") == "all"
        for spec in self._rollout_geometry.layout.specs:
            batch = self.env_batches[spec.name]
            algorithm_cfg = self._env_algorithm_cfgs[spec.name]
            if algorithm_cfg.adv_type == "opd":
                self.compute_opd_teacher_logprobs(batch)

            kwargs = {
                "task_type": self.cfg.runner.task_type,
                "adv_type": algorithm_cfg.adv_type,
                "rewards": batch["rewards"],
                "dones": batch["dones"],
                "values": batch.get("prev_values", None),
                "prev_logprobs": batch.get("prev_logprobs", None),
                "teacher_logprobs": batch.get("teacher_logprobs", None),
                "num_action_chunks": self.cfg.actor.model.num_action_chunks,
                "gamma": algorithm_cfg.get("gamma", 1),
                "gae_lambda": algorithm_cfg.get("gae_lambda", 1),
                "group_size": algorithm_cfg.get("group_size", 8),
                "reward_type": algorithm_cfg.reward_type,
                "loss_mask": batch.get("loss_mask", None),
                "loss_mask_sum": batch.get("loss_mask_sum", None),
                "advantage_mode": algorithm_cfg.get("advantage_mode", None),
            }
            if normalize_jointly:
                kwargs["normalize_advantages"] = False

            batch.update(calculate_adv_and_returns(**kwargs))
            if kwargs["loss_mask"] is not None:
                batch.update({"loss_mask": kwargs["loss_mask"]})
            if kwargs["loss_mask_sum"] is not None:
                batch.update({"loss_mask_sum": kwargs["loss_mask_sum"]})

        if normalize_jointly:
            self._normalize_advantages_jointly()

        rollout_metrics = dict(self._group_success_metrics)
        for spec in self._rollout_geometry.layout.specs:
            rollout_metrics.update(
                spec.scoped(compute_rollout_metrics(self.env_batches[spec.name]))
            )
        self.rollout_batches = self._merge_env_batches()
        rollout_metrics.update(
            compute_rollout_metrics(
                cat_list_of_dict_tensor(
                    [
                        {
                            key: pool[key]
                            for key in ("rewards", "advantages", "returns", "loss_mask")
                            if pool.get(key) is not None
                        }
                        for pool in self.rollout_batches
                    ]
                )
            )
        )
        return rollout_metrics

    def _normalize_advantages_jointly(self) -> None:
        """Normalize advantages over the valid steps of every env together.

        This applies GAE's own normalization to the union of the envs' batches.
        """
        valid = []
        for batch in self.env_batches.values():
            advantages, loss_mask = batch["advantages"], batch.get("loss_mask")
            if loss_mask is None:
                valid.append(advantages.reshape(-1))
            else:
                valid.append(advantages[loss_mask.expand_as(advantages).bool()])
        valid = torch.cat(valid)
        mean, std = valid.mean(), valid.std()
        for batch in self.env_batches.values():
            batch["advantages"] = (batch["advantages"] - mean) / (std + 1e-5)

    def _merge_env_batches(self) -> list[dict[str, Any]]:
        """Weight each env's steps, flatten them to transitions, and pool them.

        Every step gets a ``loss_mask`` and a ``loss_mask_ratio`` (see
        :func:`add_loss_weights`), so the loss treats all envs with one rule.
        Envs whose transitions share a format, the same keys and trailing
        shapes, are concatenated into one pool; envs whose policy inputs differ
        get separate pools. Rows keep env order; :meth:`_order_training_batch`
        shuffles them.

        Returns:
            One ``[N, ...]`` batch per format, usually a single one.
        """
        reference_lengths = self._rollout_geometry.layout.reference_lengths(
            self.cfg.algorithm.get("env_weighting", "trajectory")
        )
        pools: dict[Any, list[dict[str, Any]]] = {}
        # Env name -> (pool index, first row in the pool, rows).
        self._env_rows: dict[str | None, tuple[int, int, int]] = {}
        for spec in self._rollout_geometry.layout.specs:
            batch = self.env_batches[spec.name]
            algorithm_cfg = self._env_algorithm_cfgs[spec.name]
            if algorithm_cfg.loss_type == "opd":
                target_steps = int(batch["advantages"].shape[0])
                for key in [
                    "prev_logprobs",
                    "forward_inputs",
                    "loss_mask",
                    "loss_mask_sum",
                ]:
                    assert key in batch, f"OPD training requires {key}."
                    batch[key] = trim_nested_tensor_time_dim(
                        batch[key], target_steps, (key,)
                    )
            add_loss_weights(
                batch,
                reward_type=algorithm_cfg.reward_type,
                reference_length=reference_lengths[spec.name],
                loss_weight=algorithm_cfg.get("loss_weight", 1.0),
            )
            rows = batch["prev_logprobs"].shape[0] * batch["prev_logprobs"].shape[1]
            flat = process_nested_dict_for_train(batch, torch.arange(rows))
            pool = pools.setdefault(batch_format(flat), [])
            self._env_rows[spec.name] = (
                list(pools).index(batch_format(flat)),
                sum(entry["prev_logprobs"].shape[0] for entry in pool),
                rows,
            )
            pool.append(flat)
        self.env_batches = {}
        return [cat_list_of_dict_tensor(entries) for entries in pools.values()]

    def _order_training_batch(self) -> None:
        """Order each pool's rows and plan the rows every global batch takes.

        Rows are ordered so each pool hands them out front to back, and
        ``self._global_batch_plan`` lists, per global batch, the rows it takes
        from each pool. With ``actor.mix_envs`` every global batch takes the same
        share of each pool; otherwise each takes the rows of one env, in a
        random order every actor rank shares.
        """
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        self._training_round = getattr(self, "_training_round", 0) + 1
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        pool_rows = [pool["prev_logprobs"].shape[0] for pool in self.rollout_batches]
        assert sum(pool_rows) % batch_size_per_rank == 0, (
            f"Expected the {sum(pool_rows)} samples on each actor to fill whole "
            f"global batches of {batch_size_per_rank}."
        )
        if self.cfg.actor.get("mix_envs", True):
            orders = [torch.randperm(rows, generator=g) for rows in pool_rows]
            num_global_batches = sum(pool_rows) // batch_size_per_rank
            for rows in pool_rows:
                assert rows % num_global_batches == 0, (
                    "Every global batch takes an equal share of each observation "
                    f"format, so a format's {rows} samples per actor must split "
                    f"into {num_global_batches} global batches."
                )
            plan = [
                [rows // num_global_batches for rows in pool_rows]
                for _ in range(num_global_batches)
            ]
        else:
            env_parts = []
            for name, (pool_index, offset, rows) in self._env_rows.items():
                assert rows % batch_size_per_rank == 0, (
                    "actor.mix_envs=False needs each env's samples per actor to "
                    f"fill whole global batches of {batch_size_per_rank}, but env "
                    f"{name!r} has {rows}."
                )
                shuffled = offset + torch.randperm(rows, generator=g)
                env_parts.extend(
                    (pool_index, part) for part in shuffled.split(batch_size_per_rank)
                )
            shared = torch.Generator()
            shared.manual_seed(self.cfg.actor.seed + self._training_round)
            env_parts = [
                env_parts[index]
                for index in torch.randperm(len(env_parts), generator=shared)
            ]
            orders = [
                torch.cat([part for index, part in env_parts if index == pool_index])
                for pool_index in range(len(pool_rows))
            ]
            plan = [
                [
                    batch_size_per_rank if index == pool_index else 0
                    for pool_index in range(len(pool_rows))
                ]
                for index, _ in env_parts
            ]
        with torch.no_grad():
            self.rollout_batches = [
                select_nested_rows(pool, order)
                for pool, order in zip(self.rollout_batches, orders)
            ]
        self._global_batch_plan = plan

    def _global_batches(self) -> Iterator[list[dict[str, Any]]]:
        """Yield each planned global batch as its micro-batches.

        A micro-batch holds rows of one pool, so it never mixes observation
        formats.
        """
        micro_batch_size = self.cfg.actor.micro_batch_size
        offsets = [0] * len(self.rollout_batches)
        for takes in self._global_batch_plan:
            micro_batches = []
            for pool_index, rows in enumerate(takes):
                if rows == 0:
                    continue
                start = offsets[pool_index]
                part = select_nested_rows(
                    self.rollout_batches[pool_index], slice(start, start + rows)
                )
                offsets[pool_index] += rows
                micro_batches.extend(
                    split_dict_to_chunk(part, math.ceil(rows / micro_batch_size))
                )
            yield micro_batches

    @Worker.timer("actor/compute_opd_teacher_logprobs")
    def compute_opd_teacher_logprobs(self, batch: dict[str, Any]) -> None:
        assert batch.get("teacher_logprobs", None) is None, (
            "OPD teacher_logprobs must be computed after rollout on actor workers."
        )
        assert self.cfg.rollout.get("expert_model", None) is not None, (
            "OPD requires rollout.expert_model as teacher model config."
        )
        assert "forward_inputs" in batch, (
            "OPD teacher logprob computation requires rollout forward_inputs."
        )
        assert "prev_logprobs" in batch, (
            "OPD teacher logprob computation requires student prev_logprobs."
        )
        assert SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ], "OPD teacher logprob computation currently supports OpenVLA models."

        prev_logprobs = batch["prev_logprobs"]
        time_dim, batch_dim = prev_logprobs.shape[:2]
        flat_batch_size = time_dim * batch_dim

        assert self.enable_offload and self.is_weight_offloaded, (
            "OPD teacher logprob computation expects actor weights to be "
            "offloaded before moving the teacher model to GPU."
        )
        teacher_model = self._get_opd_teacher_model()
        teacher_model.to(self.device)

        flat_forward_inputs = flatten_nested_tensor_time_batch(
            batch["forward_inputs"], ("forward_inputs",)
        )
        num_chunks = (
            flat_batch_size + self.cfg.actor.micro_batch_size - 1
        ) // self.cfg.actor.micro_batch_size
        teacher_logprobs = []
        kwargs = {
            "temperature": self.cfg.rollout.sampling_params.temperature_train,
            "top_k": self.cfg.rollout.sampling_params.top_k,
        }
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(flat_forward_inputs, num_chunks):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    teacher_output = teacher_model(
                        forward_inputs=micro_batch,
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                teacher_logprobs.append(teacher_output["logprobs"].detach().cpu())

        teacher_logprobs = torch.cat(teacher_logprobs, dim=0)
        expected_shape = (flat_batch_size, *prev_logprobs.shape[2:])
        assert teacher_logprobs.shape == expected_shape, (
            f"teacher_logprobs shape {teacher_logprobs.shape} must match "
            f"flattened student logprobs shape {expected_shape}."
        )
        batch["teacher_logprobs"] = teacher_logprobs.reshape(
            time_dim, batch_dim, *teacher_logprobs.shape[1:]
        )

        teacher_model.to("cpu")
        clear_memory()

    @Worker.timer("actor/recompute_logprobs")
    def recompute_prev_logprobs(self, batch: dict[str, Any]) -> dict[str, float]:
        """
        Recompute ``prev_logprobs`` with the actor's own forward, so both ends of
        the PPO ratio come from the same path. Runs after the shuffle and scores
        the pool front to back in micro-batches of ``actor.micro_batch_size``,
        the same micro-batches the update loop trains on when a global batch's
        share of the pool is a multiple of that size.

        Args:
            batch: One ordered pool of training transitions; its
                ``prev_logprobs`` are replaced.

        Returns:
            Dict with ``actor/rollout_train_logprob_gap`` (mean absolute logprob
            difference) and ``actor/rollout_train_joint_logprob_gap_max`` (largest
            absolute difference of a sample's summed log-probability).

        Raises:
            ValueError: If ``algorithm.max_recompute_logprob_gap`` is set and the
                joint gap exceeds it or is not finite.
        """
        assert "forward_inputs" in batch, (
            "Recomputing logprobs requires rollout forward_inputs."
        )

        rollout_logprobs = batch["prev_logprobs"]
        rollout_size = rollout_logprobs.shape[0]
        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
        ]:
            kwargs = {
                "temperature": self.cfg.rollout.sampling_params.temperature_train,
                "top_k": self.cfg.rollout.sampling_params.top_k,
            }

        recomputed_logprobs = []
        with torch.no_grad():
            for micro_batch in split_dict_to_chunk(
                batch, math.ceil(rollout_size / self.cfg.actor.micro_batch_size)
            ):
                micro_batch = put_tensor_device(micro_batch, self.device)
                with self.amp_context:
                    output = self.model(
                        forward_inputs=micro_batch["forward_inputs"],
                        compute_logprobs=True,
                        compute_entropy=False,
                        compute_values=False,
                        use_cache=False,
                        **kwargs,
                    )
                recomputed_logprobs.append(output["logprobs"].detach().cpu())

        recomputed_logprobs = torch.cat(recomputed_logprobs, dim=0).to(
            rollout_logprobs.dtype
        )
        assert recomputed_logprobs.shape == rollout_logprobs.shape, (
            f"recomputed logprobs shape {recomputed_logprobs.shape} must match "
            f"rollout logprobs shape {rollout_logprobs.shape}."
        )
        clear_memory()

        # Logprobs are per action token, the mask and the advantages per action.
        action_dim = self.cfg.actor.model.get("action_dim", 7)
        log_ratio = (recomputed_logprobs - rollout_logprobs).reshape(
            *rollout_logprobs.shape[:-1], -1, action_dim
        )
        loss_mask = batch.get("loss_mask", None)
        mask = (
            None
            if loss_mask is None
            else loss_mask.bool().unsqueeze(-1).expand_as(log_ratio)
        )
        joint_log_ratio = (
            log_ratio if mask is None else log_ratio.masked_fill(~mask, 0)
        ).sum(dim=(-2, -1))
        joint_gap = joint_log_ratio.abs().max().item()
        metrics = {
            "actor/rollout_train_logprob_gap": masked_mean(
                log_ratio.abs(), mask=mask
            ).item(),
            "actor/rollout_train_joint_logprob_gap_max": joint_gap,
        }
        max_gap = self.cfg.algorithm.get("max_recompute_logprob_gap", None)
        # Written so that a NaN gap also fails.
        if max_gap is not None and not joint_gap <= max_gap:
            raise ValueError(
                "Expected the rollout/actor log-probability of every sample to "
                f"differ by at most algorithm.max_recompute_logprob_gap={max_gap}, "
                f"found {joint_gap}."
            )

        batch["prev_logprobs"] = recomputed_logprobs
        return metrics

    def _get_opd_teacher_model(self):
        if self._opd_teacher_model is None:
            teacher_model_config = build_expert_model_config(
                self.cfg, self.cfg.actor.model
            )
            teacher_model = get_model(teacher_model_config)
            if self.cfg.runner.get("expert_ckpt_path", None):
                teacher_model_dict = torch.load(
                    self.cfg.runner.expert_ckpt_path, map_location="cpu"
                )
                teacher_model.load_state_dict(teacher_model_dict)
            teacher_model.eval()
            teacher_model.requires_grad_(False)
            teacher_model.to("cpu")
            self._opd_teacher_model = teacher_model
        return self._opd_teacher_model

    def _build_sft_data_loader(self):
        if SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI:
            repo_id = resolve_lerobot_repo_id(self.cfg.actor.get("sft_data_path"))
            if repo_id is None:
                raise ValueError(
                    "actor.sft_data_path must be set to a local dataset path or "
                    "LeRobot repo id when enable_sft_co_train=True."
                )

            import openpi.training.data_loader as _data

            from rlinf.models.embodiment.openpi.dataconfig import (
                get_openpi_config,
            )

            training_config_name = OmegaConf.select(
                self.cfg.actor.model, "openpi.config_name", default=None
            )
            if not training_config_name:
                raise ValueError(
                    "enable_sft_co_train=True requires actor.model.openpi.config_name."
                )
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
        self._order_training_batch()

        metrics = {}
        if self.cfg.algorithm.get("recompute_logprobs", False):
            for pool in self.rollout_batches:
                append_to_dict(metrics, self.recompute_prev_logprobs(pool))
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            # See PPO paper for details. https://arxiv.org/abs/1707.06347
            for micro_batches in self._global_batches():
                global_batch_rows = sum(
                    batch["prev_logprobs"].shape[0] for batch in micro_batches
                )
                self.optimizer.zero_grad()
                for idx, batch in enumerate(micro_batches):
                    self.train_micro_batch(
                        micro_batch=batch,
                        metrics=metrics,
                        is_last=(idx + 1) == len(micro_batches),
                        global_batch_rows=global_batch_rows,
                    )
                    # avoid gpu memory leak
                    micro_batches[idx] = None
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
        explained_variance_stats = pop_critic_explained_variance_stats(metrics)
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        if explained_variance_stats:
            reduced_stats = all_reduce_dict(
                explained_variance_stats, op=torch.distributed.ReduceOp.SUM
            )
            mean_metric_dict[CRITIC_EXPLAINED_VARIANCE_KEY] = (
                compute_critic_explained_variance_from_stats(reduced_stats).item()
            )

        return mean_metric_dict

    def train_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        metrics: dict[str, list[float]],
        *,
        is_last: bool,
        global_batch_rows: int | None = None,
    ) -> None:
        """Accumulate the gradient of one micro-batch.

        Args:
            micro_batch: Transitions of one observation format.
            metrics: Metrics to append to.
            is_last: Whether this micro-batch ends its global batch.
            global_batch_rows: Rows in the global batch; defaults to
                ``gradient_accumulation`` micro-batches of this size.
        """
        micro_batch_rows = micro_batch["prev_logprobs"].shape[0]
        if global_batch_rows is None:
            global_batch_rows = micro_batch_rows * self.gradient_accumulation
        micro_batch = put_tensor_device(micro_batch, self.device)
        backward_ctx = self.before_micro_batch(self.model, is_last_micro_batch=is_last)
        advantages = micro_batch["advantages"]
        prev_logprobs = micro_batch["prev_logprobs"]
        returns = micro_batch.get("returns", None)
        prev_values = micro_batch.get("prev_values", None)
        loss_mask = micro_batch.get("loss_mask", None)
        loss_mask_ratio = micro_batch.get("loss_mask_ratio", None)
        forward_inputs = micro_batch.get("forward_inputs", None)

        kwargs = {}
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENVLA,
            SupportedModel.OPENVLA_OFT,
            SupportedModel.PI0_FAST,
        ]:
            temperature = self.cfg.rollout.sampling_params.temperature_train
            if (
                SupportedModel(self.cfg.actor.model.model_type)
                == SupportedModel.PI0_FAST
                and not self.cfg.rollout.sampling_params.do_sample
            ):
                temperature = 1.0
            kwargs["temperature"] = temperature
            kwargs["top_k"] = self.cfg.rollout.sampling_params.top_k
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
        ]:
            kwargs["prev_logprobs"] = prev_logprobs

        compute_values = self.cfg.algorithm.adv_type == "gae"
        compute_entropy = (
            self.cfg.algorithm.entropy_bonus > 0
            or self.cfg.algorithm.get("log_entropy_metrics", False)
        )
        with self.amp_context:
            output_dict = self.model(
                forward_inputs=forward_inputs,
                compute_logprobs=True,
                compute_entropy=compute_entropy,
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
            "loss_mask_ratio": loss_mask_ratio,
            "task_type": self.cfg.runner.task_type,
            "critic_warmup": self.optimizer_steps < self.critic_warmup_steps,
            "logprob_mask": output_dict.get("logprob_mask", None),
            "loss_agg_func": self.loss_agg_func,
            "log_logprob_diagnostics": self.cfg.algorithm.get(
                "log_logprob_diagnostics", False
            ),
        }

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.PI0_FAST,
            SupportedModel.LINGBOTVLAV2,
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
        if compute_entropy and not loss_kwargs["critic_warmup"]:
            entropy = output_dict.get("entropy", None)
            if entropy is not None and self.cfg.algorithm.entropy_bonus <= 0:
                entropy = entropy.detach()
            if entropy is not None:
                entropy_mask = loss_mask
                logprob_mask = output_dict.get("logprob_mask", None)
                if logprob_mask is not None:
                    entropy_mask = logprob_mask.to(dtype=torch.bool)
                    if loss_mask is not None:
                        env_mask = loss_mask.to(
                            device=entropy_mask.device, dtype=torch.bool
                        )
                        while env_mask.ndim > 1 and env_mask.shape[-1] == 1:
                            env_mask = env_mask.squeeze(-1)
                        while env_mask.ndim < entropy_mask.ndim:
                            env_mask = env_mask.unsqueeze(-1)
                        entropy_mask = entropy_mask & env_mask
                entropy_loss = compute_entropy_loss(
                    entropy,
                    entropy_type=self.cfg.algorithm.entropy_type,
                    loss_mask=entropy_mask,
                    action_dim=self.cfg.actor.model.get("action_dim", 7),
                    batch_size=output_dict["logprobs"].shape[0],
                )
                if self.cfg.algorithm.entropy_bonus > 0:
                    loss -= self.cfg.algorithm.entropy_bonus * entropy_loss
        metrics_data["actor/entropy_loss"] = entropy_loss.detach().item()
        if compute_entropy:
            metrics_data["actor/entropy"] = entropy_loss.detach().item()
            metrics_data["actor/entropy_bonus_term"] = (
                self.cfg.algorithm.entropy_bonus * entropy_loss.detach().item()
            )

        if self.enable_sft_co_train:
            loss = self._train_sft_epoch(metrics_data, loss)

        # The loss is this micro-batch's mean; dividing by its share of the global
        # batch makes the accumulated gradient the global batch's mean.
        loss /= global_batch_rows / micro_batch_rows
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
