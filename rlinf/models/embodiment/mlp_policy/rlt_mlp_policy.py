# Copyright 2026 The RLinf Authors.
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
from rlinf.algorithms.rlt.action_grid import ActionGrid
from rlinf.algorithms.rlt.human_bc import ref_tilde, residual_limit_vector, soft_limit
import torch.nn.functional as F
from torch.distributions.normal import Normal

import torch.nn as nn

from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.mlp_policy.tactile_encoder import (
    PACKED_DIM, TactileEncoder, pack_tactile,
)
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.utils import get_act_func, layer_init, make_mlp


# 参考动作块的起始偏移(步)。预取时,动作块按"请求那一刻"的观测算出,
# 但要到下一个块边界才开始执行,中间隔了 (块长 - 请求步) 步。
# actor 的 20 个动作对应"观测之后第 OFF..OFF+19 步",参考块也从同一位置截取,
# 这样新块从第 0 个动作执行时正好对上当前时刻,不会往回拽。0 = 不偏移(原行为)。
_RLT_REF_OFFSET = int(__import__("os").environ.get("RLT_REF_OFFSET", "0"))


def gaussian_smoother(steps: int, sigma: float) -> torch.Tensor:
    """[steps, steps] row-normalized Gaussian average over ``sigma`` steps.

    Every output step is a convex combination of the input steps: it never
    leaves their range (a capped residual stays capped), has no end effects,
    and removes a step-to-step sawtooth. Measured with a 30 mm cap at sigma 6:
    a -c..+c flip anywhere changes at most 4 mm per step, a +-c alternation
    keeps 1.9 mm (a least-squares cubic spline overshoots 2.3x at the ends and
    let random +-c steps through at 22 mm per step).
    """
    if sigma <= 0:
        raise ValueError("residual_smooth_sigma must be positive")
    t = torch.arange(steps, dtype=torch.float64)
    kernel = torch.exp(-((t[:, None] - t[None, :]) ** 2) / (2.0 * float(sigma) ** 2))
    return (kernel / kernel.sum(dim=1, keepdim=True)).float()


def residual_time_operator(steps: int, sigma: float = 0.0, ramp_steps: int = 0) -> torch.Tensor | None:
    """[steps, steps] linear map applied to the executed (capped) residual along the chunk.

    ``sigma`` > 0 averages it over time (see gaussian_smoother);
    ``ramp_steps`` > 0 then fades it in from zero at the chunk's first step
    (smoothstep, full weight from step ``ramp_steps``). Stream-mode RTC pins
    the next reference to the executed actions, so a residual that starts at
    zero continues the in-flight trajectory instead of jumping by its own size
    at every chunk boundary. Zero stays zero. None when both are off.
    """
    if not sigma and not ramp_steps:
        return None
    operator = gaussian_smoother(steps, sigma) if sigma else torch.eye(steps)
    if ramp_steps:
        if not 1 <= ramp_steps < steps:
            raise ValueError(f"residual_ramp_steps must be in 1..{steps - 1}")
        u = (torch.arange(steps, dtype=torch.float32) / float(ramp_steps)).clamp(0.0, 1.0)
        operator = (u * u * (3.0 - 2.0 * u))[:, None] * operator
    return operator


class BlockQHead(MultiQHead):
    """Twin Q with block-projected inputs and one MLP head per stage.

    Each Q network has its own Linear + LayerNorm per input block (z_rl,
    critic proprio, VLA plan, action), shared by its stages, and one
    LayerNorm-tanh MLP per stage on the concatenated projections. With
    everything concatenated raw, the 2048D z took 82% of the first layer's
    variance and Q followed z (10-28 probe). The projections learn from every
    stage (10-28 takeover chunks per stage: 7 / 64 / 62 / 1); each stage's
    judgement stays in its own head. Returns [B, num_q_heads * num_stages];
    the caller keeps each sample's stage, so only that head is trained.
    Subclasses MultiQHead only so the FSDP wrap policy treats it the same.

    ``value_blocks``: also a state-only V with the same layout (its own
    projections, one head per stage) on those blocks, for a V-bootstrapped Q
    target. It lives under q_head, so the critic optimizer owns it.
    ``penalty_prior``: one more state-only net on those blocks, the logit of
    the state's average penalty probability (critic_penalty_head).
    """

    def __init__(self, block_dims: dict, block_widths: dict, hidden_dims, num_q_heads=2, num_stages=1,
                 value_blocks=None, dropout: float = 0.0, penalty_prior: bool = False):
        nn.Module.__init__(self)
        self.num_q_heads = num_q_heads
        self.block_names = tuple(block_dims)
        self.qs = nn.ModuleList([
            self._net(block_dims, block_widths, hidden_dims, num_stages, dropout) for _ in range(num_q_heads)
        ])
        self.value_names = tuple(value_blocks or ())
        if self.value_names:
            self.v = self._net({k: block_dims[k] for k in self.value_names}, block_widths, hidden_dims, num_stages, dropout)
        if penalty_prior:
            self.pbar = self._net({k: block_dims[k] for k in self.value_names}, block_widths, hidden_dims, num_stages, dropout)

    @staticmethod
    def _net(block_dims: dict, block_widths: dict, hidden_dims, num_stages: int, dropout: float = 0.0) -> nn.Module:
        gain = nn.init.calculate_gain("tanh")
        net = nn.Module()
        net.enc = nn.ModuleDict({
            name: (TactileEncoder(block_widths[name]) if name == "tactile" else
                   nn.Sequential(nn.Linear(dim, block_widths[name]), nn.LayerNorm(block_widths[name])))
            for name, dim in block_dims.items()
        })
        net.heads = nn.ModuleList()
        for _ in range(num_stages):
            layers, width = [], sum(block_widths[name] for name in block_dims)
            for h in hidden_dims:                 # Linear-LayerNorm-tanh(-dropout), as make_mlp plus dropout
                layers += [nn.Linear(width, h), nn.LayerNorm(h), nn.Tanh()] + ([nn.Dropout(dropout)] if dropout > 0 else [])
                width = h
            head = nn.Sequential(*layers, nn.Linear(width, 1))
            linears = [m for m in head if isinstance(m, nn.Linear)]
            for m in linears[:-1]:            # as QHead: Xavier for tanh, small last layer
                nn.init.xavier_uniform_(m.weight, gain=gain); nn.init.zeros_(m.bias)
            nn.init.normal_(linears[-1].weight, std=0.02); nn.init.zeros_(linears[-1].bias)
            net.heads.append(head)
        return net

    @staticmethod
    def _run(net: nn.Module, blocks: dict, names) -> torch.Tensor:
        feat = torch.cat([net.enc[name](blocks[name]) for name in names], dim=-1)
        return torch.cat([head(feat) for head in net.heads], dim=-1)

    def forward(self, blocks: dict, head: str = "q") -> torch.Tensor:
        """Twin Q [B, num_q_heads * num_stages]; ``head="value"``: [B, num_stages]
        state values; ``head="prior"``: [B, num_stages] logits of the state's
        penalty probability.

        V and pbar go through forward as well: FSDP wraps this module as its
        own unit and refreshes its parameter views only in forward, so a
        direct method call after an optimizer step reads stale views (11-02:
        the first real training round failed on its second update)."""
        if head == "value":
            return self._run(self.v, blocks, self.value_names)
        if head == "prior":
            return self._run(self.pbar, blocks, self.value_names)
        if head != "q":
            raise ValueError(f"unknown BlockQHead head {head!r}")
        return torch.cat([self._run(q, blocks, self.block_names) for q in self.qs], dim=-1)


class RLTMLPPolicy(MLPPolicy):
    """MLP actor-critic policy for RLT Stage 2 heads.

    Actor input follows RLT: reference action chunk, RL token feature, and
    proprioceptive state. Critic input follows RLT: action chunk, RL token
    feature, and proprioceptive state.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        fixed_std: float = 0.002,
        residual_ref: bool = False,
        num_stage_actors: int = 0,
        stage_onehot_dim: int = 8,
        indep_dropout_head: bool = False,
        residual_limit_rad: float = 0.0,
        action_scales=None,
        exclude_gripper_residual: bool = False,
        gripper_dims=(6, 13),
        rollout_noise: bool = True,
        residual_smooth_sigma: float = 0.0,
        residual_ramp_steps: int = 0,
        residual_limits=None,
        residual_in_cap_units: bool = False,
        critic_proprio_dim: int | None = None,
        critic_action_dim: int | None = None,
        critic_block_encoders: bool = False,
        critic_relative_action: bool = False,
        critic_action_scale=None,
        critic_action_space: str = "critic",
        critic_value_head: bool = False,
        action_bias=None,
        critic_block_widths=None,
        critic_hidden_dims=None,
        critic_dropout: float = 0.0,
        critic_penalty_head: bool = False,
        critic_penalty_q_form: str = "prob",
        critic_penalty_kappa: float = 0.1,
        critic_penalty_temp: float = 3.0,
        critic_penalty_only: bool = False,
        tactile_enabled: bool = False,
        tactile_embed_dim: int = 64,
        actor_ref_full: bool = False,
        executor_action_grid: bool = False,
        action_grid_offset: int | None = None,
        action_grid_scale: float = 1.0,
        action_grid_dense: bool = False,
    ):
        if not add_q_head:
            raise ValueError(
                "RLTMLPPolicy requires add_q_head=True for actor-critic training."
            )
        z_dim = int(z_dim)
        self.tactile_enabled = bool(tactile_enabled)
        self.tactile_embed_dim = int(tactile_embed_dim)
        if self.tactile_enabled and (
            not critic_block_encoders or int(num_stage_actors) <= 0 or q_head_type != "default"
            or self.tactile_embed_dim <= 0
        ):
            raise ValueError("tactile requires per-stage default block Q and a positive embedding width")
        proprio_dim = int(proprio_dim)
        step_action_dim = int(action_dim)
        chunk_len = int(num_action_chunks)
        ref_chunk_len = (
            chunk_len if ref_num_action_chunks is None else int(ref_num_action_chunks)
        )
        if ref_chunk_len < chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{ref_chunk_len} < {chunk_len}."
            )
        self.action_grid = ActionGrid(
            env_steps=chunk_len, reference_steps=ref_chunk_len,
            offset=_RLT_REF_OFFSET if action_grid_offset is None else int(action_grid_offset),
            row_scale=float(action_grid_scale), dense=bool(action_grid_dense),
            aligned=bool(executor_action_grid),
        )
        self.env_chunk_len = chunk_len
        chunk_len = self.action_grid.target_steps
        flat_action_dim = chunk_len * step_action_dim
        # rtj joint Stage2: the actor plans 16D joints from the joint state; the
        # critic reads the TCP state (critic_proprio) and 20D TCP actions, the
        # actor's through the differentiable native FK (the actor worker's
        # _critic_actions).
        critic_step_action_dim = step_action_dim if critic_action_dim is None else int(critic_action_dim)
        if critic_step_action_dim != step_action_dim and (step_action_dim, critic_step_action_dim) != (16, 20):
            raise ValueError("a separate critic action space is Tianji's joint16 -> tcp20 only")
        critic_flat_action_dim = chunk_len * critic_step_action_dim

        # 11-06: actor_ref_full -- the actor reads all ref_num_action_chunks reference rows
        # (the VLA's 60), not only the executed ones from RLT_REF_OFFSET. The rows before
        # the offset are the ones the previous chunk executes while this one is inferred
        # (RTC pins them to it): where the arm will be, and how fast, when this chunk
        # starts. It still outputs the residual on the executed rows only. Offline
        # (11-06, 5 folds by episode, with the 14-row fade-in): BC gain on held-out
        # human corrections 0.155 -> 0.182, on the first 14 executed rows 0.099 -> 0.130.
        actor_ref_rows = ref_chunk_len if actor_ref_full else chunk_len
        actor_obs_dim = z_dim + proprio_dim + actor_ref_rows * step_action_dim
        if self.tactile_enabled:
            actor_obs_dim += self.tactile_embed_dim
        critic_obs_dim = z_dim + (proprio_dim if critic_proprio_dim is None else int(critic_proprio_dim))

        super().__init__(
            obs_dim=actor_obs_dim,
            action_dim=flat_action_dim,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=add_q_head,
            q_head_type=q_head_type,
            critic_obs_dim=critic_obs_dim,
        )
        self.z_dim = z_dim
        if self.action_grid.aligned:
            self.register_buffer("_action_grid_contract", self.action_grid.metadata(1))
        self.actor_ref_full = bool(actor_ref_full)
        self.actor_ref_dim = actor_ref_rows * step_action_dim
        if self.tactile_enabled:
            # Base SAC routes names containing "encoder" to the critic optimizer.
            # This actor-only network must stay outside that name filter.
            self.tactile_actor_net = TactileEncoder(self.tactile_embed_dim)
        self.proprio_dim = proprio_dim
        self.step_action_dim = step_action_dim
        self.chunk_len = chunk_len
        self.ref_chunk_len = ref_chunk_len
        self.flat_action_dim = flat_action_dim
        self.critic_proprio_dim = None if critic_proprio_dim is None else int(critic_proprio_dim)
        self.critic_step_action_dim = critic_step_action_dim
        self.critic_flat_action_dim = critic_flat_action_dim
        # 10-28: the critic reads its action as the deviation from the VLA plan
        # (obs["critic_plan"], the reference rows in the critic's space, given
        # by the actor worker) in units of critic_action_scale, and the plan
        # itself as state. Same information, but Q can no longer value an
        # action chunk as "where the arm goes" (a second state description);
        # its action dependence has to explain returns by the deviation.
        self.critic_block_encoders = bool(critic_block_encoders)
        self.critic_relative_action = bool(critic_relative_action)
        if self.critic_relative_action:
            if not self.critic_block_encoders or critic_action_scale is None:
                raise ValueError("critic_relative_action needs critic_block_encoders and critic_action_scale")
            if len(critic_action_scale) != critic_step_action_dim:
                raise ValueError(f"critic_action_scale needs {critic_step_action_dim} values")
            self.register_buffer(
                "_critic_action_scale",
                torch.as_tensor(list(critic_action_scale), dtype=torch.float32).repeat(chunk_len),
                persistent=False,
            )
        if self.critic_block_encoders and (q_head_type != "default" or int(num_stage_actors) == 0):
            raise ValueError("critic_block_encoders is built for the per-stage default twin Q")
        # 10-28 evening, "joint_rel": the critic reads the actor's own joint plan
        # as its displacement from the current joints (physical units, rows
        # minus proprio[:16], divided by critic_action_scale), not the deviation
        # from the VLA plan: every -1 sat at zero deviation, so Q learned
        # "deviating is good". The state block adds the joint state to the TCP
        # one. An offline classifier on these inputs spotted the chunks that
        # got taken over (AUC 0.93) and ranked human corrections < VLA plan <
        # the opposite change in risk (93-97% of takeover chunks).
        self.critic_action_space = str(critic_action_space)
        if self.critic_action_space not in ("critic", "joint_rel"):
            raise ValueError(f"critic_action_space={critic_action_space!r}; use 'critic' or 'joint_rel'")
        self.critic_value_head = bool(critic_value_head)
        if self.critic_action_space == "joint_rel":
            if not self.critic_block_encoders or self.critic_relative_action:
                raise ValueError("critic_action_space=joint_rel needs critic_block_encoders and no critic_relative_action")
            if critic_action_scale is None or len(critic_action_scale) != step_action_dim \
                    or action_scales is None or action_bias is None:
                raise ValueError(f"joint_rel needs {step_action_dim} critic_action_scale values, action_scales and action_bias")
            self.critic_flat_action_dim = critic_flat_action_dim = flat_action_dim
            self.register_buffer("_critic_rel_scale", torch.as_tensor(list(critic_action_scale), dtype=torch.float32), persistent=False)
            self.register_buffer("_joint_scale", torch.as_tensor(list(action_scales), dtype=torch.float32), persistent=False)
            self.register_buffer("_joint_bias", torch.as_tensor(list(action_bias), dtype=torch.float32), persistent=False)
        if self.critic_value_head and not self.critic_block_encoders:
            raise ValueError("critic_value_head needs critic_block_encoders")
        # 10-28 night: Q(s, a) = V(s) + pbar(s) - phat(s, a). The twin Q nets
        # output phat's logits, the probability that chunk a gets a "当前步失败"
        # or a takeover here; pbar is the state's average, so V(s) keeps the
        # return (time cost and penalties included) and phat - pbar is how much
        # worse this plan is than the usual one. Offline the TD-regressed Q lost
        # the takeovers (AUC 0.5-0.6); phat trained as a classifier kept them
        # (0.87-0.92).
        self.critic_penalty_head = bool(critic_penalty_head)
        # 10-30: how phat / pbar enter Q. "prob": V + sigmoid(pbar) - sigmoid(phat) (10-28). The
        # classifier saturates (0 or 1 almost everywhere), so Q is flat in the action except at
        # the few states it already calls risky, and the actor gets no gradient. "temp" softens
        # the sigmoid by critic_penalty_temp; "logit" uses kappa * (pbar_logit - phat_logit).
        self.critic_penalty_q_form = str(critic_penalty_q_form)
        self.critic_penalty_kappa = float(critic_penalty_kappa)
        self.critic_penalty_temp = float(critic_penalty_temp)
        # 10-30: Q = -phat only (no V, no pbar). The TD-regressed V could not honour the
        # takeover truncation (the -1 leaked through neighbouring states and summed over a
        # 48-chunk horizon to -3 ... -5) and never reached the actor's gradient anyway;
        # the critic is a short-horizon one, "a penalty within critic_penalty_window chunks".
        self.critic_penalty_only = bool(critic_penalty_only)
        if self.critic_penalty_head and not self.critic_value_head:
            raise ValueError("critic_penalty_head needs critic_value_head")
        if critic_flat_action_dim != flat_action_dim and int(num_stage_actors) == 0:
            # Per-stage heads build their own Q head below.
            from rlinf.models.embodiment.modules.q_head import MultiCrossQHead, MultiQHead
            q_head_cls = MultiQHead if q_head_type == "default" else MultiCrossQHead
            self.q_head = q_head_cls(
                hidden_size=critic_obs_dim,
                hidden_dims=[256, 256, 256],
                num_q_heads=2,
                output_dim=1,
                action_feature_dim=critic_flat_action_dim,
            )
        self.fixed_std = float(fixed_std)
        # False: training rollouts execute the actor mean (no per-step sample).
        # fixed_std still shapes the actor loss. Tianji's absolute TCP frame
        # makes 0.002 about 4 mm per axis per 33 ms step, i.e. visible jitter.
        self.rollout_noise = bool(rollout_noise)
        # >0: the executed residual is averaged over that many steps along the
        # chunk (training and execution alike). Per-dimension critic gradients
        # otherwise make it a step-by-step sawtooth that smooth_weight only
        # damps (A1: 0.001-0.004, i.e. 2-8 mm per step in Tianji's TCP frame).
        # residual_ramp_steps > 0 also fades it in from zero at the chunk start.
        smoother = residual_time_operator(
            chunk_len, float(residual_smooth_sigma) * self.action_grid.rate_ratio,
            float(residual_ramp_steps) * self.action_grid.rate_ratio,
        )
        if smoother is not None:
            self.register_buffer("_residual_smoother", smoother, persistent=False)
        else:
            self._residual_smoother = None
        # 残差参数化:action_mean = atanh(ref_chunk) + actor_mean(feat),且 actor_mean 零初始化。
        # 目的只有一个 —— 让"从零开始"的 actor 在第 0 步就精确输出基座 VLA 的参考动作,
        # 而不是随机权重的输出直接开真机(实测会乱动)。损失函数不变、收敛后与完整 actor 等价;
        # BC 项会把修正量拴在 0 附近,Q 项再推它偏离。ref 已在 RL 帧 [-1,1],与 tanh 输出同帧。
        self.residual_ref = bool(residual_ref)
        if self._residual_smoother is not None and not self.residual_ref:
            raise ValueError("residual_smooth_sigma / residual_ramp_steps shape a residual: set residual_ref")
        if self.residual_ref:
            torch.nn.init.zeros_(self.actor_mean.weight)
            torch.nn.init.zeros_(self.actor_mean.bias)
        if self.fixed_std <= 0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")
        # 残差软限幅(训练/执行共用):π = r̃ + c·tanh((tanh(atanh(r̃)+Δ) − r̃)/c),c 按弧度给、按各关节量程折算;
        # 夹爪可退出残差(直接 clamp(ref)):人的夹爪纠正是开/闭阶跃,残差参数化表达不了,且 VLA 夹爪值常越界。
        self.gripper_dims = tuple(int(g) for g in (gripper_dims or ()))
        self.exclude_gripper_residual = bool(exclude_gripper_residual)
        # residual_limits: one physical limit per dimension instead (Tianji's
        # Cartesian frame: metres, rot6 units, closure; <= 0 unlimited).
        self._residual_limit = None
        if action_scales is not None and (
            residual_limits is not None or (residual_limit_rad and float(residual_limit_rad) > 0)
        ):
            self._residual_limit = residual_limit_vector(
                action_scales, float(residual_limit_rad or 0.0), self.gripper_dims, limits=residual_limits
            )
        # residual_in_cap_units: the residual heads output in units of the cap
        # (x min(c, 1) per dimension). Adam moves every output weight by about
        # lr per update whatever the gradient, i.e. the residual by ~1e-3 RL
        # units: ~2 mm in Tianji's 2 m position frame. Against a 30 mm cap
        # (0.015 units; A1 0.17-0.32) that saturated 93% of a sim-trained
        # residual within 80 updates, where the soft limit's gradient vanishes
        # and BC can no longer pull it back. Zero stays zero.
        if residual_in_cap_units:
            if self._residual_limit is None or not self.residual_ref:
                raise ValueError("residual_in_cap_units needs residual_ref and a residual limit")
            self.register_buffer(
                "_residual_scale",
                self._residual_limit.clamp(max=1.0).repeat(chunk_len),
                persistent=False,
            )
        else:
            self._residual_scale = None

        # 每个阶段一个独立 actor(按阶段号路由),critic 每个 Q 网络的主干各阶段共享、输出端每阶段一个头。
        # 阶段号取自 proprio 末尾的 stage_onehot_dim 维 one-hot(全零 = 阶段 0,按阶段 1 处理,
        # 该情形只出现在尚未按 1 的门控期,那时执行的是 VLA 参考、不记录转移)。
        self.num_stage_actors = int(num_stage_actors)
        self.stage_onehot_dim = int(stage_onehot_dim)
        if self.num_stage_actors > 0:
            import torch.nn as nn
            from rlinf.models.embodiment.modules.q_head import MultiQHead
            act = get_act_func("tanh")
            self.stage_backbones = nn.ModuleList([
                nn.Sequential(
                    layer_init(nn.Linear(actor_obs_dim, 256)), act(),
                    layer_init(nn.Linear(256, 256)), act(),
                    layer_init(nn.Linear(256, 256)), act(),
                )
                for _ in range(self.num_stage_actors)
            ])
            self.stage_actor_means = nn.ModuleList(
                [nn.Linear(256, flat_action_dim) for _ in range(self.num_stage_actors)]
            )
            for head in self.stage_actor_means:
                if self.residual_ref:
                    torch.nn.init.zeros_(head.weight); torch.nn.init.zeros_(head.bias)
                else:
                    layer_init(head, std=0.01)
            # 残差 + 参考 dropout 的冲突:参考被置零时残差要独自生成整个动作,与"参考在时残差≈0"相矛盾,
            # 训练初期网络分不清两种样本,残差被学成"整个动作的一半"—— 实测前 8 次更新右臂 J1 就
            # 朝相机偏 +0.5~0.9 rad(两次撞相机)。论文里 dropout 的本意是"保留一条独立的动作生成通路",
            # 这里就让参考被置零的样本走独立输出头,残差头只由参考在场的样本训练,开局仍精确等于 VLA。
            self.indep_dropout_head = bool(indep_dropout_head) and self.residual_ref
            if self.indep_dropout_head:
                self.stage_indep_heads = nn.ModuleList(
                    [layer_init(nn.Linear(256, flat_action_dim), std=0.01) for _ in range(self.num_stage_actors)]
                )
            if self.critic_block_encoders:
                critic_state_dim = critic_obs_dim - z_dim
                if self.critic_action_space == "joint_rel":   # + joints and joint velocities (proprio without its one-hot)
                    critic_state_dim += proprio_dim - self.stage_onehot_dim
                blocks = {"z": z_dim, "proprio": critic_state_dim}
                widths = {"z": 256, "proprio": 64, "plan": 256, "action": 256}
                if self.tactile_enabled:
                    blocks["tactile"] = PACKED_DIM
                    widths["tactile"] = self.tactile_embed_dim
                # 10-28 evening: the offline Q at these widths (~2.4M weights per
                # Q, 6k transitions) overfit and lost the takeover signal that a
                # 32/16-dim classifier with dropout kept (AUC 0.93).
                widths.update(dict(critic_block_widths or {}))
                if self.critic_relative_action:
                    blocks["plan"] = critic_flat_action_dim
                blocks["action"] = critic_flat_action_dim
                widths = {k: int(widths[k]) for k in blocks}
                self.q_head = BlockQHead(blocks, widths, list(critic_hidden_dims or [256, 256, 256]), num_q_heads=2,
                                         num_stages=self.num_stage_actors,
                                         value_blocks=(("z", "proprio") + (("tactile",) if self.tactile_enabled else ())) if self.critic_value_head else None,
                                         dropout=float(critic_dropout), penalty_prior=self.critic_penalty_head)
            else:
                self.q_head = MultiQHead(
                    hidden_size=critic_obs_dim,
                    hidden_dims=[256, 256, 256],
                    num_q_heads=2,
                    output_dim=self.num_stage_actors,
                    action_feature_dim=critic_flat_action_dim,
                )
            # 共享的单 actor 不再使用,删掉以免进优化器、进检查点造成误解
            del self.backbone, self.actor_mean, self.actor_logstd

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        key = prefix + "_action_grid_contract"
        if self.action_grid.aligned and key in state_dict:
            expected = self.action_grid.metadata(1, device=state_dict[key].device)
            if state_dict[key].shape != expected.shape or not torch.allclose(
                state_dict[key].to(torch.float64), expected, rtol=0, atol=1e-9
            ):
                error_msgs.append("checkpoint action grid differs from the executor target time grid")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed = {}
        for key, value in env_obs.items():
            processed[key] = value.to(device) if torch.is_tensor(value) else value
        return processed

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["z_rl"])

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["proprio"])

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        if self.action_grid.aligned:
            return self.action_grid.reference_targets(ref_chunk).reshape(ref_chunk.shape[0], -1)
        _o = _RLT_REF_OFFSET
        if _o + self.chunk_len > ref_chunk.shape[1]:
            raise ValueError(
                "RLT_REF_OFFSET=%d + chunk_len=%d 超出参考块长度 %d" % (_o, self.chunk_len, ref_chunk.shape[1])
            )
        ref_chunk = ref_chunk[:, _o : _o + self.chunk_len]
        return ref_chunk.reshape(ref_chunk.shape[0], -1)

    def _actor_ref_input(self, obs: dict) -> torch.Tensor:
        """The actor's reference block: the executed rows, or all rows (actor_ref_full)."""
        if not getattr(self, "actor_ref_full", False):
            return self._get_ref_chunk(obs)
        ref = self._flatten_batch(obs["ref_chunk"])
        if ref.shape[-1] != self.actor_ref_dim:
            raise ValueError(
                f"actor_ref_full reads {self.actor_ref_dim // self.step_action_dim} reference rows, "
                f"got {ref.shape[-1] // self.step_action_dim}"
            )
        return ref

    def _executed_ref_of(self, actor_state: torch.Tensor) -> torch.Tensor:
        """The executed reference rows inside the actor input (after any dropout)."""
        block = actor_state[:, : self.actor_ref_dim]
        if not getattr(self, "actor_ref_full", False):
            return block
        if self.action_grid.aligned:
            return self.action_grid.reference_targets(
                block.reshape(block.shape[0], self.ref_chunk_len, self.step_action_dim)
            ).reshape(block.shape[0], -1)
        start = _RLT_REF_OFFSET * self.step_action_dim
        return block[:, start : start + self.flat_action_dim]

    def _maybe_drop_reference(
        self,
        ref_chunk: torch.Tensor,
        reference_dropout_prob: float,
    ) -> torch.Tensor:
        if reference_dropout_prob <= 0:
            return ref_chunk
        keep_prob = 1.0 - float(reference_dropout_prob)
        keep_mask = (
            torch.rand((ref_chunk.shape[0], 1), device=ref_chunk.device) < keep_prob
        )
        return ref_chunk * keep_mask.to(dtype=ref_chunk.dtype)

    def _actor_state(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
    ) -> torch.Tensor:
        ref_chunk = self._actor_ref_input(obs)
        if apply_reference_dropout:
            ref_chunk = self._maybe_drop_reference(ref_chunk, reference_dropout_prob)
        blocks = [ref_chunk, self._get_z(obs), self._get_proprio(obs)]
        if self.tactile_enabled:
            blocks.append(self.tactile_actor_net(pack_tactile(
                obs, device=ref_chunk.device, dtype=ref_chunk.dtype)))
        return torch.cat(blocks, dim=-1)

    def _stage_index(self, obs: dict) -> torch.Tensor:
        oh = self._get_proprio(obs)[..., -self.stage_onehot_dim:]
        idx = oh.argmax(dim=-1)                       # 阶段 k → k-1;全零 → 0
        return idx.clamp_(0, self.num_stage_actors - 1)

    def _critic_state(self, obs: dict) -> torch.Tensor:
        if self.critic_proprio_dim is not None:
            return torch.cat([self._get_z(obs), self._flatten_batch(obs["critic_proprio"])], dim=-1)
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def _residual_head(self, output: torch.Tensor) -> torch.Tensor:
        """A residual head's output in RL units (see residual_in_cap_units)."""
        return output if self._residual_scale is None else output * self._residual_scale.to(output)

    def _shape_residual(self, action: torch.Tensor, ref3: torch.Tensor, dropped) -> torch.Tensor:
        """Smooth and fade in the executed residual, after its per-step cap.

        Capping first means a large raw residual can neither jump to the cap
        right after the ramp nor flip from -c to +c where it crosses zero; the
        average then bounds its change per step and stays inside the cap.
        Samples whose reference was dropped (their own head, never executed)
        keep their action.
        """
        if self._residual_smoother is None or action.shape[0] == 0:
            return action
        B = action.shape[0]
        rt = ref_tilde(ref3)
        residual = action.reshape(B, self.chunk_len, self.step_action_dim) - rt
        residual = torch.einsum("ts,nsd->ntd", self._residual_smoother.to(residual.dtype), residual)
        # The reference moves along the chunk, so ref + averaged residual can
        # still leave the RL frame by a little.
        shaped = (rt + residual).clamp(-1.0, 1.0).reshape(B, -1)
        return shaped if dropped is None else torch.where(dropped[:, None], action, shaped)

    def sac_forward(
        self,
        obs,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
        **kwargs,
    ):
        actor_state = self._actor_state(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        ref_used = self._executed_ref_of(actor_state)   # 执行行的(可能被 dropout 的)ref
        dropped = (ref_used.abs().sum(dim=-1) == 0) if getattr(self, "indep_dropout_head", False) else None
        if self.num_stage_actors > 0:
            idx = self._stage_index(obs)
            action_mean = actor_state.new_zeros(actor_state.shape[0], self.flat_action_dim)
            for k in range(self.num_stage_actors):   # 每个样本只经过本阶段的 actor,梯度只流向它
                m = idx == k
                if not m.any():
                    continue
                feat = self.stage_backbones[k](actor_state[m])
                if dropped is None:
                    action_mean[m] = self._residual_head(self.stage_actor_means[k](feat))
                else:                                  # 参考在场 → 残差头;参考被置零 → 独立头
                    dm = dropped[m]
                    out = actor_state.new_zeros(int(m.sum()), self.flat_action_dim)
                    if (~dm).any():
                        out[~dm] = torch.atanh(ref_used[m][~dm].clamp(-0.999, 0.999)) + self._residual_head(
                            self.stage_actor_means[k](feat[~dm]))
                    if dm.any():
                        out[dm] = self.stage_indep_heads[k](feat[dm])
                    action_mean[m] = out
        else:
            feat = self.backbone(actor_state)
            action_mean = self.actor_mean(feat)
            if self.residual_ref:
                action_mean = self._residual_head(action_mean)
        if self.residual_ref and dropped is None:
            action_mean = torch.atanh(ref_used.clamp(-0.999, 0.999)) + action_mean
        action_std = torch.full_like(action_mean, self.fixed_std)
        probs = Normal(action_mean, action_std)
        action = action_mean if deterministic else probs.rsample()
        chunk_logprobs = probs.log_prob(action)
        action = torch.tanh(action)
        limit = self._residual_limit is not None or self.exclude_gripper_residual
        if self.residual_ref and (limit or self._residual_smoother is not None):
            B = action.shape[0]
            ref3 = self._get_ref_chunk(obs).reshape(B, self.chunk_len, self.step_action_dim)   # 原始参考(未 dropout)
            if limit:
                a3 = action.reshape(B, self.chunk_len, self.step_action_dim)
                limited = soft_limit(ref_tilde(ref3), a3, self._residual_limit, self.gripper_dims,
                                     self.exclude_gripper_residual, ref_raw=ref3).reshape(B, -1)
                action = limited if dropped is None else torch.where(dropped[:, None], action, limited)
            action = self._shape_residual(action, ref3, dropped)
        return action, chunk_logprobs, dropped

    def _critic_blocks(self, obs: dict, actions: torch.Tensor | None) -> dict:
        proprio = (self._flatten_batch(obs["critic_proprio"]) if self.critic_proprio_dim is not None
                   else self._get_proprio(obs))
        joint = self._get_proprio(obs)
        if self.critic_action_space == "joint_rel":
            proprio = torch.cat([proprio, joint[..., : joint.shape[-1] - self.stage_onehot_dim]], dim=-1)
        blocks = {"z": self._get_z(obs), "proprio": proprio}
        if self.tactile_enabled:
            blocks["tactile"] = pack_tactile(obs, device=proprio.device, dtype=proprio.dtype)
        if actions is None:                          # state-only blocks for V
            return blocks
        if self.critic_relative_action:
            plan = self._flatten_batch(obs["critic_plan"])
            blocks["plan"] = plan
            actions = (actions - plan) / self._critic_action_scale.to(actions)
        if self.critic_action_space == "joint_rel":   # RL-frame joint rows -> physical, minus the current joints
            B, D = actions.shape[0], self.step_action_dim
            rows = actions.reshape(B, -1, D) * self._joint_scale.to(actions) + self._joint_bias.to(actions)
            actions = ((rows - joint[:, None, :D].to(actions)) / self._critic_rel_scale.to(actions)).reshape(B, -1)
        blocks["action"] = actions
        return blocks

    def _stage_value(self, v: torch.Tensor, obs: dict) -> torch.Tensor:
        return v.gather(1, self._stage_index(obs).unsqueeze(-1))

    def sac_q_forward(self, obs, actions=None, shared_feature=None, detach_encoder=False, value=False,
                      penalty_logits=False, penalty_prior=False):
        """Twin Q [B, 2] of ``actions``; with ``value=True`` the state value V [B, 1].

        With critic_penalty_head: ``penalty_logits=True`` returns phat's twin
        logits [B, 2], ``penalty_prior=True`` pbar's logit [B, 1], and the
        default V + pbar - phat."""
        del shared_feature
        if value or penalty_prior:
            if not self.critic_value_head or (penalty_prior and not self.critic_penalty_head):
                raise ValueError("value=True needs critic_value_head, penalty_prior=True critic_penalty_head")
            blocks = self._critic_blocks(obs, None)
            return self._stage_value(self.q_head(blocks, head="prior" if penalty_prior else "value"), obs)
        if self.critic_block_encoders:
            blocks = self._critic_blocks(obs, self._flatten_batch(actions))
            if detach_encoder:
                blocks = {k: v if k == "action" else v.detach() for k, v in blocks.items()}
            q = self.q_head(blocks)
        else:
            critic_state = self._critic_state(obs)
            if detach_encoder:
                critic_state = critic_state.detach()
            q = self.q_head(critic_state, self._flatten_batch(actions))
        if self.num_stage_actors > 0:                 # [B, 2S] → 按阶段取出 [B, 2](Q1, Q2)
            S = self.num_stage_actors; idx = self._stage_index(obs).unsqueeze(-1)
            q = torch.cat([q[:, :S].gather(1, idx), q[:, S:].gather(1, idx)], dim=-1)
        if self.critic_penalty_head and not penalty_logits:
            if self.critic_penalty_only:
                return -self.penalty_to_q(q)
            v = self._stage_value(self.q_head(blocks, head="value"), obs)
            prior = self._stage_value(self.q_head(blocks, head="prior"), obs)
            q = v + self.penalty_to_q(prior) - self.penalty_to_q(q)
        return q

    def penalty_to_q(self, logit: torch.Tensor) -> torch.Tensor:
        """The penalty head's contribution to Q from its logit (see critic_penalty_q_form)."""
        if self.critic_penalty_q_form == "logit":
            return self.critic_penalty_kappa * logit
        if self.critic_penalty_q_form == "temp":
            return torch.sigmoid(logit / self.critic_penalty_temp)
        return torch.sigmoid(logit)

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        del shared_feature
        critic_state = self._critic_state(obs)
        next_critic_state = (
            self._critic_state(next_obs) if next_obs is not None else None
        )
        if detach_encoder:
            critic_state = critic_state.detach()
            if next_critic_state is not None:
                next_critic_state = next_critic_state.detach()
        return self.q_head(
            critic_state,
            self._flatten_batch(actions),
            next_state_features=next_critic_state,
            next_action_features=(
                self._flatten_batch(next_actions) if next_actions is not None else None
            ),
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def sft_forward(self, data, **kwargs):
        obs = data["obs"] if "obs" in data else data
        target_actions = self._flatten_batch(
            data["action"] if "action" in data else data["actions"]
        )
        if self.num_stage_actors > 0:                 # 按阶段路由的确定性输出(已过 tanh)
            pred_actions = self.sac_forward(obs, deterministic=True)[0]
        else:
            actor_state = self._actor_state(obs)
            pred_actions = self.actor_mean(self.backbone(actor_state))
        return F.mse_loss(pred_actions, target_actions, reduction="none")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values
        obs = self.preprocess_env_obs(env_obs=env_obs)
        action, chunk_logprobs, _ = self.sac_forward(
            obs, deterministic=(mode == "eval" or not self.rollout_noise)
        )
        target_actions = self._format_chunk_actions(action)
        chunk_actions = self.action_grid.to_env(target_actions) if self.action_grid.aligned else target_actions
        env_action = chunk_actions.reshape(chunk_actions.shape[0], -1)

        forward_inputs = {"action": env_action, "model_action": env_action}
        if self.action_grid.aligned:
            forward_inputs.update(
                target_model_action=action,
                target_policy_action=action.clone(),
                target_action_grid=self.action_grid.metadata(action.shape[0], device=action.device),
            )
        if return_obs:
            forward_inputs.update(obs)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(chunk_logprobs[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
