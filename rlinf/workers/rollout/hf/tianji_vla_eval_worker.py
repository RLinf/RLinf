# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Frozen joint VLA deployment; no Actor, Critic or TCP target cache."""
import copy
import os

import torch

from rlinf.models import get_model
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def joint_reference_actions(observation, chunk, offset):
    """Select absolute joint targets in the contract's RL frame, without FK/IK."""
    reference = observation["ref_chunk"]
    if reference.ndim != 3 or reference.shape[-1] != 16:
        raise ValueError("Joint Stage1 needs a [B,H,16] VLA reference")
    if offset < 0 or offset + chunk > reference.shape[1]:
        raise ValueError("Joint Stage1 reference slice exceeds VLA horizon")
    if not torch.isfinite(reference).all():
        raise ValueError("Joint VLA returned non-finite actions")
    if bool((reference.abs() > 1.00001).any()):
        raise ValueError("Joint VLA returned actions outside its contract frame")
    return reference[:, offset:offset + chunk].clamp(-0.999, 0.999).contiguous()


class TianjiVLAEvalWorker(MultiStepRolloutWorker):
    """Reuse async eval channels with only the frozen joint VLA loaded."""

    def init_worker(self):
        if self.cfg.runner.get("ckpt_path"):
            raise ValueError("Stage1 loads the base VLA; an Actor checkpoint is not accepted")
        feature = copy.deepcopy(self.cfg.rollout.rlt_feature_model)
        if feature.openpi.config_name != "tianji_stage1_joint" or feature.action_dim != 16:
            raise ValueError("Stage1 direct-joint deployment requires the joint VLA")
        self.hf_model = get_model(feature).eval().requires_grad_(False)
        self.rlt_feature_model = None
        self.car_monitor = None
        if os.environ.get("TIANJI_CAR_1118") == "1":
            from rlinf.envs.realworld.tianji.car_monitor_tap import MonitorTap

            self.car_monitor = MonitorTap(os.environ.get("TIANJI_CAR_MONITOR_SOCKET"))
        self.setup_sample_params()

    def _predict_rollout_actions(self, env_obs, mode="eval", final_obs=None,
                                 rlt_switch_flags=None, intervene_requested=None):
        del mode, final_obs, rlt_switch_flags, intervene_requested
        with torch.no_grad():
            observation = self.hf_model.extract_rlt_obs(env_obs)
            actions = joint_reference_actions(
                observation, int(self.model_cfg.num_action_chunks),
                int(os.environ.get("RLT_REF_OFFSET", "0")),
            )
        # The shared queue accepts the same PolicyOutput envelope as Stage2.
        # Stage1 has one VLA output, not per-stage Actor candidates.
        forward_inputs = {
            "ref_chunk": observation["ref_chunk"],
            "action": actions.reshape(actions.shape[0], -1),
            "actions_by_stage": actions.reshape(actions.shape[0], 1, -1),
        }
        if self.car_monitor is not None:
            meta = env_obs["car_stage_meta"]
            forward_inputs["car_stage_meta"] = meta
            self.car_monitor.publish(meta, observation["z_rl"], observation["proprio"])
        return actions, {"forward_inputs": forward_inputs}
