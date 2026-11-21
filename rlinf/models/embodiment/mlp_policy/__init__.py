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

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import IQLMLPPolicy
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy

    iql_config = cfg.get("iql_config", None)
    if cfg.model_type == "rlt_mlp_policy":
        model = RLTMLPPolicy(
            z_dim=cfg.z_dim,
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            ref_num_action_chunks=cfg.get(
                "ref_num_action_chunks", cfg.num_action_chunks
            ),
            residual_ref=bool(cfg.get("residual_ref", False)),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            fixed_std=cfg.get("fixed_std", 0.002),
            num_stage_actors=int(cfg.get("num_stage_actors", 0)),
            stage_onehot_dim=int(cfg.get("stage_onehot_dim", 8)),
            indep_dropout_head=bool(cfg.get("indep_dropout_head", False)),
            residual_limit_rad=float(cfg.get("residual_limit_rad", 0.0)),
            action_scales=list(cfg.get("action_scales")) if cfg.get("action_scales", None) is not None else None,
            exclude_gripper_residual=bool(cfg.get("exclude_gripper_residual", False)),
            gripper_dims=tuple(cfg.get("gripper_dims", (6, 13))),
            rollout_noise=bool(cfg.get("rollout_noise", True)),
            residual_smooth_sigma=float(cfg.get("residual_smooth_sigma", 0.0)),
            residual_ramp_steps=int(cfg.get("residual_ramp_steps", 0)),
            residual_limits=list(cfg.get("residual_limits")) if cfg.get("residual_limits", None) is not None else None,
            residual_in_cap_units=bool(cfg.get("residual_in_cap_units", False)),
            critic_proprio_dim=cfg.get("critic_proprio_dim", None),
            critic_action_dim=cfg.get("critic_action_dim", None),
            critic_block_encoders=bool(cfg.get("critic_block_encoders", False)),
            critic_relative_action=bool(cfg.get("critic_relative_action", False)),
            critic_action_scale=list(cfg.get("critic_action_scale")) if cfg.get("critic_action_scale", None) is not None else None,
            critic_action_space=str(cfg.get("critic_action_space", "critic")),
            critic_value_head=bool(cfg.get("critic_value_head", False)),
            action_bias=list(cfg.get("action_bias")) if cfg.get("action_bias", None) is not None else None,
            critic_block_widths=dict(cfg.get("critic_block_widths")) if cfg.get("critic_block_widths", None) is not None else None,
            critic_hidden_dims=list(cfg.get("critic_hidden_dims")) if cfg.get("critic_hidden_dims", None) is not None else None,
            critic_dropout=float(cfg.get("critic_dropout", 0.0)),
            critic_penalty_head=bool(cfg.get("critic_penalty_head", False)),
            critic_penalty_q_form=str(cfg.get("critic_penalty_q_form", "prob")),
            critic_penalty_kappa=float(cfg.get("critic_penalty_kappa", 0.1) or 0.1),
            critic_penalty_temp=float(cfg.get("critic_penalty_temp", 3.0) or 3.0),
            critic_penalty_only=bool(cfg.get("critic_penalty_only", False)),
            tactile_enabled=bool(cfg.get("tactile_enabled", False)),
            tactile_embed_dim=int(cfg.get("tactile_embed_dim", 64)),
            actor_ref_full=bool(cfg.get("actor_ref_full", False)),
            executor_action_grid=bool(cfg.get("executor_action_grid", False)),
            action_grid_offset=cfg.get("action_grid_offset", None),
            action_grid_scale=float(cfg.get("action_grid_scale", 1.0)),
            action_grid_dense=bool(cfg.get("action_grid_dense", False)),
        )
    elif cfg.model_type == "rlt_td3_mlp_policy":
        model = RLTTD3MLPPolicy(
            z_dim=cfg.z_dim,
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            ref_num_action_chunks=cfg.get(
                "ref_num_action_chunks", cfg.num_action_chunks
            ),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            mlp_hidden_dim=cfg.get("mlp_hidden_dim", 256),
            mlp_num_hidden_layers=cfg.get("mlp_num_hidden_layers", 2),
            actor_noise_sigma=cfg.get("actor_noise_sigma", 0.1),
            ref_action_dropout=cfg.get("ref_action_dropout", 0.0),
        )
    elif iql_config is not None:
        model = IQLMLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
        )
        model.configure_iql(iql_config)
    else:
        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
        )

    return model
