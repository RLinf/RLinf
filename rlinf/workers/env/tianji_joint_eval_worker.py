# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Joint Actor deployment using training's stage routing and owner queue."""

from collections import defaultdict

import torch

from rlinf.data.schema.embodied_types import EnvOutput
from rlinf.envs.action_utils import prepare_actions
from rlinf.envs.utils import get_env_attr, to_tensor
from rlinf.workers.env.env_worker import EnvWorker


class TianjiJointEvalWorker(EnvWorker):
    """Inference only; no replay, optimizer or change to the training worker."""

    def _rlt_stage_now(self, stage_id):
        vals = self.eval_env_list[stage_id].env.call("get_wrapper_attr", "rlt_switch_flags")
        return [int(v or 0) for v in vals]

    def _poll_early_eval_recv(self, input_channel, stage_id):
        super()._poll_early_eval_recv(input_channel, stage_id)
        state = self._early_eval_recv[stage_id]
        if state["output"] is None or state.get("offered"):
            return
        output = state["output"]
        acts = output.actions.clone()
        fi = output.forward_inputs
        stages = self._rlt_stage_now(stage_id)
        for b_i, _, _, new in self._stage_head_switches_of(fi, fi["actions_by_stage"], stages):
            acts[b_i] = new.to(acts).reshape(acts[b_i].shape)
        offer = getattr(self.eval_env_list[stage_id], "offer_next_chunk", None)
        if offer is not None:
            offer(prepare_actions(
                raw_chunk_actions=acts, env_type=self.cfg.env.eval.env_type,
                model_type=self.model_cfg.model_type,
                num_action_chunks=self.model_cfg.num_action_chunks,
                action_dim=self.model_cfg.action_dim,
                policy=self.model_cfg.get("policy_setup", None),
                wm_env_type=self.cfg.env.eval.get("wm_env_type", None),
                env_cfg=self.cfg.env.eval,
            ))
        state["offered"] = True

    def _reject_stale_policy(self, policy_output, stage_id):
        return False

    def evaluate(self, input_channel, rollout_channel):
        metrics = defaultdict(list)
        outputs, stale = {}, set()

        def send(stage_id):
            output = outputs[stage_id]
            stages = self._rlt_stage_now(stage_id)
            output.rlt_switch_flags = to_tensor(stages) if any(stages) else None
            self.send_to(
                group_name=self.cfg.rollout.group_name, channel=rollout_channel,
                data=self._build_rollout_input_data(output.to_dict()),
                split_fn=self._obs_split_fn, mode="eval", tag="rollout_results",
                route_key=stage_id, decoupled_mode=False,
            )

        if self.env_decoupled_mode or self.eval_rollout_epoch != 1:
            raise ValueError("Joint deployment requires one coupled evaluation epoch")
        for stage_id in range(self.stage_num):
            self.eval_prev_done[stage_id] = torch.zeros(self.eval_num_envs_per_stage, dtype=torch.bool)
            obs, infos = self.eval_env_list[stage_id].reset()
            outputs[stage_id] = EnvOutput(obs=obs, env_infos=infos)
            send(stage_id)
        for step in range(self.n_eval_chunk_steps):
            last = step == self.n_eval_chunk_steps - 1
            for stage_id in range(self.stage_num):
                policy = self._take_eval_policy_output(input_channel, stage_id)
                if stage_id in stale:
                    # Drain exactly one old reply, preserving channel pairing.
                    stale.remove(stage_id)
                    if not last:
                        send(stage_id)
                    continue
                if self._reject_stale_policy(policy, stage_id):
                    if not last:
                        env = self.eval_env_list[stage_id]
                        outputs[stage_id] = EnvOutput(obs=env.refresh_observation())
                        send(stage_id)
                    continue
                self._select_stage_head(policy, self._rlt_stage_now(stage_id))
                actions = policy.actions.detach().cpu().numpy()
                output, info, sent = self._evaluate_step_prefetch(
                    actions, stage_id, rollout_channel,
                    int(self.cfg.rollout.eval_prefetch_at),
                    send=not last, input_channel=input_channel,
                )
                outputs[stage_id] = output
                for key, value in info.items():
                    metrics[key].append(value)
                if output.final_obs is not None and sent:
                    # Auto-reset returned a new episode observation. The in-flight
                    # result still belongs to the episode that just ended.
                    stale.add(stage_id)
                if not last and not sent:
                    send(stage_id)
        self.finish_rollout(mode="eval")
        for env in self.eval_env_list:
            if self.eval_enable_offload:
                get_env_attr(env, "offload")()
        return {key: torch.cat(value, dim=0).contiguous().cpu() for key, value in metrics.items()}


class TianjiVLAEnvWorker(TianjiJointEvalWorker):
    """Same joint execution/RTC/HIL queue, with no Actor stage-head selection."""

    def _reject_stale_policy(self, policy_output, stage_id):
        import os

        if os.environ.get("TIANJI_CAR_1118") != "1":
            return False
        metadata = policy_output.forward_inputs.get("car_stage_meta")
        if metadata is None:
            raise RuntimeError("1118v1 policy reply has no control revision")
        env = self.eval_env_list[stage_id].env
        live = env.call("get_wrapper_attr", "car_control_revision")
        return any(int(meta[4]) != int(revision) for meta, revision in zip(metadata, live, strict=True))

    def _select_stage_head(self, policy_output, stages):
        pass

    @staticmethod
    def _stage_head_switches_of(fi, by_stage, stages):
        return []
