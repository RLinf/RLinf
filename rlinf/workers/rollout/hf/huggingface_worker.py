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

import copy
import gc
import os
from typing import Any, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

from rlinf.config import SupportedModel
from rlinf.data.embodied_io_struct import (
    RolloutResult,
)
from rlinf.hybrid_engines.weight_syncer import WeightSyncer
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.rlt_stage2.rollout_adapter import RLTStage2RolloutAdapter
from rlinf.scheduler import Channel, Cluster, Worker
from rlinf.utils.comm_mapping import CommMapper
from rlinf.utils.placement import HybridComponentPlacement


class MultiStepRolloutWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        self.should_stop = False

        self.actor_group_name = cfg.actor.group_name
        self.device = self.torch_platform.current_device()

        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num
        self.enable_offload = self.cfg.rollout.get("enable_offload", False)

        self.placement = HybridComponentPlacement(cfg, Cluster())

        rollout_world_size = self.placement.get_world_size("rollout")
        self.actor_weight_src_rank = 0
        self._weight_sync_rollout_ranks = list(range(rollout_world_size))
        self._weight_sync_is_sender = self._rank == 0
        self.rollout_epoch = cfg.algorithm.get("rollout_epoch", 1)
        self.collect_transitions = self.cfg.rollout.get("collect_transitions", False)
        self.expert_model = None
        self._expert_model_config = None
        self._has_expert_model_config = (
            self.cfg.rollout.get("expert_model", None) is not None
        )

        self.total_num_train_envs = cfg.env.train.total_num_envs
        self.total_num_eval_envs = cfg.env.eval.total_num_envs
        self.num_pipeline_stages = cfg.rollout.pipeline_stage_num

        self.train_batch_size = (
            self.total_num_train_envs // self._world_size // self.num_pipeline_stages
        )
        self.eval_batch_size = (
            self.total_num_eval_envs // self._world_size // self.num_pipeline_stages
        )
        self.enable_cuda_graph = cfg.rollout.get("enable_cuda_graph", False)
        self.enable_eval = cfg.runner.val_check_interval > 0 or cfg.runner.only_eval
        self.eval_policy_mode = str(cfg.env.eval.get("policy_mode", "eval"))
        if self.eval_policy_mode not in {"train", "eval"}:
            raise ValueError(
                "env.eval.policy_mode must be either 'train' or 'eval', got "
                f"{self.eval_policy_mode!r}"
            )
        self.eval_action_exec_chunks = int(
            cfg.env.eval.get("action_exec_chunks", cfg.actor.model.num_action_chunks)
        )
        if self.eval_action_exec_chunks <= 0:
            raise ValueError(
                "env.eval.action_exec_chunks must be positive, got "
                f"{self.eval_action_exec_chunks}"
            )
        if cfg.env.eval.max_steps_per_rollout_epoch % self.eval_action_exec_chunks != 0:
            raise ValueError(
                "env.eval.max_steps_per_rollout_epoch must be divisible by "
                "env.eval.action_exec_chunks, got "
                f"{cfg.env.eval.max_steps_per_rollout_epoch} and "
                f"{self.eval_action_exec_chunks}"
            )

        self.n_train_chunk_steps = (
            cfg.env.train.max_steps_per_rollout_epoch
            // cfg.actor.model.num_action_chunks
        )
        self.n_eval_chunk_steps = (
            cfg.env.eval.max_steps_per_rollout_epoch // self.eval_action_exec_chunks
        )
        self.collect_prev_infos = self.cfg.rollout.get("collect_prev_infos", True)
        self.version = 0
        self.finished_episodes = None

        weight_syncer_cfg = OmegaConf.select(cfg, "weight_syncer", default=None)
        assert weight_syncer_cfg is not None, (
            "rollout.weight_syncer config must be provided"
        )
        self.weight_syncer = WeightSyncer.create(weight_syncer_cfg)
        self._sync_weight_comm_options = self.weight_syncer.comm_options

    def _is_rlt_stage2_td3(self) -> bool:
        return (
            self.cfg.algorithm.get("loss_type", None) == "rlt_td3"
            and SupportedModel(self.cfg.actor.model.model_type)
            == SupportedModel.RLT_STAGE2
        )

    def init_worker(self):
        rollout_model_config = copy.deepcopy(self.cfg.actor.model)
        with open_dict(rollout_model_config):
            rollout_model_config.precision = self.cfg.rollout.model.precision
            rollout_model_config.model_path = self.cfg.rollout.model.model_path

        self.hf_model: BasePolicy = get_model(rollout_model_config)

        if self.cfg.runner.get("ckpt_path", None):
            model_dict = torch.load(self.cfg.runner.ckpt_path)
            self.hf_model.load_state_dict(model_dict)

        if self._has_expert_model_config:
            self._expert_model_config = self._build_expert_model_config()
            if not self._is_rlt_stage2_td3():
                self._ensure_expert_model_loaded()

        self.rlt_rollout_adapter = RLTStage2RolloutAdapter(
            cfg=self.cfg,
            student_model=self.hf_model,
            expert_model_getter=self._ensure_expert_model_loaded,
            has_expert_model_config=self._has_expert_model_config,
        )

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
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

        self.dst_ranks = {}
        self.src_ranks = {}
        if not self.cfg.runner.only_eval:
            self.dst_ranks = {
                "train": self._setup_dst_ranks(
                    self.total_num_train_envs // self.num_pipeline_stages
                ),
            }
            self.src_ranks = {
                "train": self._setup_src_ranks(
                    self.total_num_train_envs // self.num_pipeline_stages
                ),
            }
        if self.enable_eval:
            self.dst_ranks["eval"] = self._setup_dst_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )
            self.src_ranks["eval"] = self._setup_src_ranks(
                self.total_num_eval_envs // self.num_pipeline_stages
            )

        self.log_info(f"Rollout worker initialized with dst_ranks: {self.dst_ranks}")
        self.log_info(f"Rollout worker initialized with src_ranks: {self.src_ranks}")
        self.setup_sample_params()
        if self.enable_offload:
            self.offload_model()

    def _build_expert_model_config(self):
        expert_model_config = copy.deepcopy(self.cfg.actor.model)
        expert_ckpt_path = self.cfg.runner.get("expert_ckpt_path", None)
        expert_model_path = self.cfg.rollout.expert_model.model_path
        if (
            self._is_rlt_stage2_td3()
            and expert_ckpt_path
            and os.path.isdir(str(expert_ckpt_path))
        ):
            expert_model_path = expert_ckpt_path
        with open_dict(expert_model_config):
            expert_model_config.precision = self.cfg.rollout.expert_model.precision
            expert_model_config.model_path = expert_model_path
            if expert_model_config.get("rlt_stage2", None) is not None:
                act_as_vla_reference = self.cfg.rollout.expert_model.get(
                    "act_as_vla_reference", self._is_rlt_stage2_td3()
                )
                expert_model_config.rlt_stage2.act_as_vla_reference = (
                    act_as_vla_reference
                )
                if act_as_vla_reference:
                    expert_model_config.rlt_stage2.load_feature_backbones = True
                    expert_model_config.rlt_stage2.load_rl_token_model = False
        return expert_model_config

    def _ensure_expert_model_loaded(self):
        if self.expert_model is not None:
            return self.expert_model
        if self._expert_model_config is None:
            raise RuntimeError(
                "Expert intervention was requested, but rollout.expert_model is not configured."
            )

        self.expert_model = get_model(self._expert_model_config)
        expert_ckpt_path = self.cfg.runner.get("expert_ckpt_path", None)
        if expert_ckpt_path and not os.path.isdir(str(expert_ckpt_path)):
            expert_model_dict = torch.load(expert_ckpt_path)
            self.expert_model.load_state_dict(expert_model_dict)
        self.expert_model.eval()
        return self.expert_model

    def setup_sample_params(self):
        # length parameters for rollout
        self._length_params = OmegaConf.to_container(
            self.cfg.algorithm.length_params, resolve=True
        )
        # sampling parameters for rollout
        self._sampling_params = OmegaConf.to_container(
            self.cfg.algorithm.sampling_params, resolve=True
        )
        self._train_sampling_params = {
            "do_sample": self._sampling_params["do_sample"],
            "temperature": self._sampling_params["temperature_train"]
            if self._sampling_params["do_sample"]
            else 1.0,
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        self._eval_sampling_params = {
            "do_sample": True
            if self._sampling_params.get("temperature_eval", -1) > 0
            else False,
            "temperature": self._sampling_params["temperature_eval"],
            "top_k": self._sampling_params["top_k"],
            "top_p": self._sampling_params["top_p"],
            "max_new_tokens": self._length_params["max_new_token"],
        }

        if self._has_expert_model_config:
            intervention_cfg = self.cfg.algorithm.get("intervention", {})
            default_beta = intervention_cfg.get(
                "probability",
                self.cfg.algorithm.get("dagger", {}).get("init_beta", 0.5),
            )
            self._dagger_sampling_params = {
                "beta": default_beta,
                "beta_schedule": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_schedule", "exponential"
                ),
                "beta_min": self.cfg.algorithm.get("dagger", {}).get("beta_min", 0.05),
                "beta_decay": self.cfg.algorithm.get("dagger", {}).get(
                    "beta_decay", 0.99
                ),
            }

    def update_dagger_beta(self):
        if not self._has_expert_model_config:
            return
        if self._is_rlt_stage2_td3():
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

    def _setup_dst_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env peer ranks for this rollout worker.

        This mapping supports both one-to-many and many-to-one env/rollout layouts.
        The returned ranks are used as communication counterparts for receiving env
        outputs and sending action chunks.

        Args:
            batch_size: Total env batch size per pipeline stage across all workers.

        Returns:
            Ordered ``(env_rank, batch_size)`` tuples this rollout worker should
            send action chunks to.
        """
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_dst_ranks(
            batch_size=batch_size,
            src_world_size=rollout_world_size,
            dst_world_size=env_world_size,
            src_rank=self._rank,
        )

    def _setup_src_ranks(self, batch_size: int) -> list[tuple[int, int]]:
        """Compute env source ranks and sizes for receiving env outputs."""
        env_world_size = self.placement.get_world_size("env")
        rollout_world_size = self.placement.get_world_size("rollout")
        return CommMapper.get_src_ranks(
            batch_size=batch_size,
            src_world_size=env_world_size,
            dst_world_size=rollout_world_size,
            dst_rank=self._rank,
        )

    @Worker.timer("predict")
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        *,
        allow_expert: bool = True,
        policy_info: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        kwargs = (
            self._train_sampling_params
            if mode == "train"
            else self._eval_sampling_params
        )

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.OPENPI,
            SupportedModel.MLP_POLICY,
            SupportedModel.GR00T,
            SupportedModel.GR00T_N1D6,
            SupportedModel.GR00T_N1D7,
            SupportedModel.ABOT_M0,
            SupportedModel.DREAMZERO,
            SupportedModel.CNN_POLICY,
            SupportedModel.CFG_MODEL,
            SupportedModel.RLT_STAGE2,
        ]:
            if self.cfg.algorithm.loss_type == "embodied_dagger":
                kwargs = {"mode": "eval"}
            else:
                kwargs = {"mode": mode}

        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.CNN_POLICY,
            SupportedModel.FLOW_POLICY,
            SupportedModel.MLP_POLICY,
        ]:
            kwargs["return_obs"] = not hasattr(self.hf_model, "q_head")

        only_save_expert = self.cfg.algorithm.get("dagger", {}).get(
            "only_save_expert", True
        )
        is_rlt_stage2_td3 = self._is_rlt_stage2_td3()

        use_expert = (
            not is_rlt_stage2_td3
            and mode == "train"
            and allow_expert
            and self._has_expert_model_config
            and torch.rand(1).item() < self._dagger_sampling_params["beta"]
        )

        with torch.no_grad():
            expert_label_flag = False
            if is_rlt_stage2_td3:
                route_result = self.rlt_rollout_adapter.predict(
                    env_obs=env_obs,
                    policy_info=policy_info,
                    model_kwargs=kwargs,
                    mode=mode,
                    allow_expert=allow_expert,
                    update_version=self.version,
                )
                actions = route_result.actions
                result = route_result.result
                expert_label_flag = route_result.expert_label_flag
            elif use_expert:
                actions, result = self._ensure_expert_model_loaded().predict_action_batch(
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
                not is_rlt_stage2_td3
                and not only_save_expert  # only re-label in classic dagger mode
                and not use_expert  # only re-label if not using expert
                and self._has_expert_model_config  # only re-label if expert exists
                and mode == "train"  # only re-label in train mode
            ):
                _, expert_result = self._ensure_expert_model_loaded().predict_action_batch(
                    env_obs=env_obs,
                    **kwargs,
                )
                expert_forward_inputs = expert_result["forward_inputs"]
                expert_target = expert_forward_inputs.get(
                    "model_action", expert_forward_inputs.get("action")
                )
                if expert_target is not None:
                    result["forward_inputs"]["model_action"] = expert_target
                expert_label_flag = True

        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        result["expert_label_flag"] = bool(expert_label_flag)
        return actions, result

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        if final_obs is None:
            return None
        if self._is_rlt_stage2_td3():
            return None
        if not (
            hasattr(self.hf_model, "value_head") or hasattr(self.hf_model, "q_head")
        ):
            return None
        with torch.no_grad():
            actions, result = self.predict(final_obs)
            if "prev_values" in result and result["prev_values"] is not None:
                final_values = result["prev_values"]
            else:
                final_values = torch.zeros_like(actions[:, :1], dtype=torch.float32)
        return final_values[:, :1].cpu().contiguous()

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
            receiver_state_dict = (
                self.rlt_rollout_adapter.rollout_state_dict()
                if self._is_rlt_stage2_td3()
                else self.hf_model.state_dict()
            )
            await self.weight_syncer.init_receiver(
                state_dict=receiver_state_dict,
                recv=recv_func,
                send=send_func,
            )

        applied_version = await self.weight_syncer.apply(self.hf_model, recv_func)
        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(applied_version)

        gc.collect()
        self.torch_platform.empty_cache()

    @Worker.timer("generate_one_epoch")
    async def generate_one_epoch(self, input_channel: Channel, output_channel: Channel):
        self.update_dagger_beta()
        for _ in range(self.n_train_chunk_steps):
            for _ in range(self.num_pipeline_stages):
                env_output = await self.recv_env_output(input_channel)
                actions, result = self.predict(
                    env_output["obs"],
                    policy_info=env_output.get("policy_info", None),
                )
                rlt_step_trace = self.rlt_rollout_adapter.encode_step_trace(
                    env_output.get("step_obs", None)
                )

                save_flags = result.get("forward_inputs", {}).get(
                    "intervention_flags", None
                )
                if save_flags is None and result.get("expert_label_flag", False):
                    save_flags = torch.full(
                        (actions.shape[0], self.cfg.actor.model.num_action_chunks),
                        True,
                        dtype=torch.bool,
                        device=actions.device,
                    )
                rollout_result = RolloutResult(
                    actions=actions,
                    prev_logprobs=result["prev_logprobs"]
                    if self.collect_prev_infos
                    else None,
                    prev_values=result["prev_values"]
                    if self.collect_prev_infos
                    else None,
                    bootstrap_values=self.get_bootstrap_values(
                        env_output.get("final_obs", None)
                    ),
                    save_flags=save_flags,
                    rlt_step_trace=rlt_step_trace,
                    forward_inputs=result["forward_inputs"],
                    versions=torch.full_like(
                        result["prev_logprobs"],
                        float(self.version),
                        dtype=torch.float32,
                    ),
                )
                self.send_rollout_result(output_channel, rollout_result, mode="train")
        for _ in range(self.num_pipeline_stages):
            env_output = await self.recv_env_output(input_channel)
            actions, result = self.predict(
                env_output["obs"],
                allow_expert=False,
                policy_info=env_output.get("policy_info", None),
            )
            rlt_step_trace = self.rlt_rollout_adapter.encode_step_trace(
                env_output.get("step_obs", None)
            )

            forward_inputs = result["forward_inputs"] if self._is_rlt_stage2_td3() else {}
            rollout_result = RolloutResult(
                actions=actions,
                prev_values=result["prev_values"] if self.collect_prev_infos else None,
                bootstrap_values=self.get_bootstrap_values(
                    env_output.get("final_obs", None)
                ),
                rlt_step_trace=rlt_step_trace,
                forward_inputs=forward_inputs,
            )
            self.send_rollout_result(output_channel, rollout_result, mode="train")

    @Worker.timer("rollout/generate")
    async def generate(
        self,
        input_channel: Channel,
        output_channel: Channel,
    ):
        if self.enable_offload:
            self.reload_model()

        for _ in tqdm(
            range(self.rollout_epoch),
            desc="Generating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            await self.generate_one_epoch(input_channel, output_channel)

        if self.enable_offload:
            self.offload_model()

    async def evaluate(self, input_channel: Channel, output_channel: Channel):
        if self.enable_offload:
            self.reload_model()
        for _ in tqdm(
            range(self.cfg.algorithm.eval_rollout_epoch),
            desc="Evaluating Rollout Epochs",
            disable=(self._rank != 0),
        ):
            for _ in range(self.n_eval_chunk_steps):
                for _ in range(self.num_pipeline_stages):
                    env_output = await self.recv_env_output(input_channel, mode="eval")
                    actions, _ = self.predict(
                        env_output["obs"],
                        mode=self.eval_policy_mode,
                        allow_expert=False,
                        policy_info=env_output.get("policy_info", None),
                    )
                    self.send_chunk_actions(output_channel, actions, mode="eval")

        if self.enable_offload:
            self.offload_model()

    def offload_model(self):
        if self.enable_cuda_graph:
            self.hf_model.release_cuda_graph()
        self.hf_model.to("cpu")
        if self.expert_model is not None:
            self.expert_model.to("cpu")
        self.torch_platform.empty_cache()

    def reload_model(self):
        self.hf_model.to(self.device)
        if self.expert_model is not None:
            self.expert_model.to(self.device)
        if self.enable_cuda_graph:
            self.hf_model.capture_cuda_graph(
                train_batch_size=self.train_batch_size,
                eval_batch_size=self.eval_batch_size,
            )

    @Worker.timer("rollout/recv_obs")
    async def recv_env_output(
        self, input_channel: Channel, mode: Literal["train", "eval"] = "train"
    ) -> dict[str, Any]:
        """Receive env outputs from mapped env ranks and merge if needed.

        Args:
            input_channel: Channel carrying env->rollout outputs.
            mode: Rollout mode, either ``"train"`` or ``"eval"``.

        Returns:
            A single env output dict. When multiple env ranks are mapped to this
            rollout worker, outputs are merged on batch dimension.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        src_ranks_and_sizes = self.src_ranks[mode]
        obs_batches = []
        for src_rank, expected_size in src_ranks_and_sizes:
            obs_batch = await input_channel.get(
                key=CommMapper.build_channel_key(
                    src_rank, self._rank, extra=f"{mode}_obs"
                ),
                async_op=True,
            ).async_wait()
            actual_size = self._infer_env_batch_size(obs_batch)
            assert actual_size == expected_size, (
                f"Expected env output batch size {expected_size} from env rank {src_rank}, "
                f"got {actual_size}."
            )
            obs_batches.append(obs_batch)
        return self._merge_obs_batches(obs_batches)

    def _split_actions(
        self, actions: torch.Tensor | np.ndarray, sizes: list[int]
    ) -> list[torch.Tensor | np.ndarray]:
        """Split rollout actions into size-specified shards along dim-0.

        Args:
            actions: Model-predicted action chunk batch (tensor or ndarray).
            sizes: Batch sizes for each destination env rank.

        Returns:
            A list of action shards aligned with destination rank order.
        """
        assert sum(sizes) == actions.shape[0], (
            f"Number of actions ({actions.shape[0]}) must equal split sizes sum ({sum(sizes)})."
        )
        if isinstance(actions, np.ndarray):
            split_indices = np.cumsum(sizes[:-1]).tolist()
            return list(np.split(actions, split_indices, axis=0))
        return list(torch.split(actions, sizes, dim=0))

    @staticmethod
    def _infer_env_batch_size(obs_batch: dict[str, Any]) -> int:
        obs = obs_batch["obs"] if "obs" in obs_batch else obs_batch
        for key in ("states", "main_images", "task_descriptions"):
            value = obs.get(key)
            if isinstance(value, torch.Tensor):
                return value.shape[0]
            if isinstance(value, list):
                return len(value)
        raise ValueError("Cannot infer batch size from env obs.")

    @staticmethod
    def _merge_obs_batches(obs_batches: list[dict[str, Any]]) -> dict[str, Any]:
        if not obs_batches:
            return {}
        obs_dicts = [
            obs_batch["obs"] if "obs" in obs_batch else obs_batch
            for obs_batch in obs_batches
        ]
        final_obs_list = [obs_batch.get("final_obs", None) for obs_batch in obs_batches]
        step_obs_list = [obs_batch.get("step_obs", None) for obs_batch in obs_batches]
        policy_info_list = [obs_batch.get("policy_info", None) for obs_batch in obs_batches]

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

        merged_step_obs = None
        if any(step_obs is not None for step_obs in step_obs_list):
            if any(step_obs is None for step_obs in step_obs_list):
                raise ValueError(
                    "Inconsistent RLT step_obs: some env shards are None while others are present."
                )
            assert step_obs_list[0] is not None
            merged_step_obs = {}
            for key in step_obs_list[0].keys():
                values = [step_obs[key] for step_obs in step_obs_list]
                first_non_none = next(
                    (value for value in values if value is not None), None
                )
                if first_non_none is None:
                    merged_step_obs[key] = None
                elif isinstance(first_non_none, torch.Tensor):
                    merged_step_obs[key] = torch.cat(values, dim=1)
                elif isinstance(first_non_none, list):
                    merged_step_obs[key] = [
                        [item for value in values for item in value[t]]
                        for t in range(len(first_non_none))
                    ]
                else:
                    merged_step_obs[key] = values

        merged_policy_info = None
        if any(policy_info is not None for policy_info in policy_info_list):
            if any(policy_info is None for policy_info in policy_info_list):
                raise ValueError(
                    "Inconsistent policy_info: some env shards are None while "
                    "others are present."
                )
            policy_info_dicts = [
                policy_info
                for policy_info in policy_info_list
                if policy_info is not None
            ]
            expected_keys = set(policy_info_dicts[0].keys())
            for idx, policy_info in enumerate(policy_info_dicts[1:], start=1):
                keys = set(policy_info.keys())
                if keys != expected_keys:
                    raise ValueError(
                        "Inconsistent policy_info keys across env shards: "
                        f"expected {sorted(expected_keys)}, got {sorted(keys)} "
                        f"at shard {idx}."
                    )
            merged_policy_info = {}
            for key in sorted(expected_keys):
                merged_policy_info[key] = torch.cat(
                    [policy_info[key] for policy_info in policy_info_dicts],
                    dim=0,
                )

        return {
            "obs": merged_obs,
            "final_obs": merged_final_obs,
            "step_obs": merged_step_obs,
            "policy_info": merged_policy_info,
        }

    @Worker.timer("rollout/send_actions")
    def send_chunk_actions(
        self,
        output_channel: Channel,
        chunk_actions: torch.Tensor | np.ndarray,
        mode: Literal["train", "eval"] = "train",
    ):
        """Send action shards to mapped env ranks.

        Args:
            output_channel: Channel carrying rollout->env action chunks.
            chunk_actions: Predicted action chunk batch (tensor or ndarray).
            mode: Rollout mode, either ``"train"`` or ``"eval"``.
        """
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        chunk_actions_split = self._split_actions(chunk_actions, split_sizes)
        for (dst_rank, _), chunk_action_i in zip(
            dst_ranks_and_sizes, chunk_actions_split
        ):
            if isinstance(chunk_action_i, torch.Tensor):
                chunk_action_i = (
                    chunk_action_i.detach().cpu().contiguous()
                )  # for evaluation
            output_channel.put(
                chunk_action_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_actions"
                ),
                async_op=True,
            )

    def _split_rollout_result(
        self, rollout_result: RolloutResult, sizes: list[int]
    ) -> list[RolloutResult]:
        def _split_optional_tensor(
            tensor: torch.Tensor | None,
        ) -> tuple[torch.Tensor | None, ...]:
            if tensor is None:
                return tuple(None for _ in sizes)
            return tuple(torch.split(tensor, sizes, dim=0))

        split_actions = _split_optional_tensor(rollout_result.actions)
        split_prev_logprobs = _split_optional_tensor(rollout_result.prev_logprobs)
        split_prev_values = _split_optional_tensor(rollout_result.prev_values)
        split_bootstrap_values = _split_optional_tensor(rollout_result.bootstrap_values)
        split_save_flags = _split_optional_tensor(rollout_result.save_flags)
        split_versions = _split_optional_tensor(rollout_result.versions)
        split_forward_inputs = (
            [{} for _ in sizes]
            if not rollout_result.forward_inputs
            else [
                {
                    key: torch.split(value, sizes, dim=0)[idx]
                    for key, value in rollout_result.forward_inputs.items()
                }
                for idx in range(len(sizes))
            ]
        )
        split_rlt_step_trace = (
            [{} for _ in sizes]
            if not rollout_result.rlt_step_trace
            else [
                {
                    key: torch.split(value, sizes, dim=1)[idx]
                    for key, value in rollout_result.rlt_step_trace.items()
                }
                for idx in range(len(sizes))
            ]
        )

        return [
            RolloutResult(
                actions=split_actions[idx],
                prev_logprobs=split_prev_logprobs[idx],
                prev_values=split_prev_values[idx],
                bootstrap_values=split_bootstrap_values[idx],
                save_flags=split_save_flags[idx],
                forward_inputs=split_forward_inputs[idx],
                rlt_step_trace=split_rlt_step_trace[idx],
                versions=split_versions[idx],
            )
            for idx in range(len(sizes))
        ]

    @Worker.timer("rollout/send_traj")
    def send_rollout_result(
        self,
        output_channel: Channel,
        rollout_result: RolloutResult,
        mode: Literal["train", "eval"] = "train",
    ):
        assert mode in ["train", "eval"], f"{mode=} is not supported"
        dst_ranks_and_sizes = self.dst_ranks[mode]
        split_sizes = [size for _, size in dst_ranks_and_sizes]
        split_rollout_results = self._split_rollout_result(rollout_result, split_sizes)
        for (dst_rank, _), rollout_result_i in zip(
            dst_ranks_and_sizes, split_rollout_results
        ):
            output_channel.put(
                rollout_result_i,
                key=CommMapper.build_channel_key(
                    self._rank, dst_rank, extra=f"{mode}_rollout_results"
                ),
                async_op=True,
            )

    def set_global_step(self, global_step: int):
        if hasattr(self.hf_model, "set_global_step"):
            self.hf_model.set_global_step(global_step)
