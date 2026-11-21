"""残差 actor 学习人工介入的公共函数(训练 worker、模型前向、离线验证共用同一份实现)。

背景(2026-09-28 四轮审查定稿):
  * 论文/RLinf 在介入步把参考块改写成人的动作。对直接输出的 actor 无害;对残差 actor(π = ref + Δ)则让
    Δ 的 BC 目标恒为 0,人的纠正从未进入执行时使用的残差头。这里改为:参考保持 VLA 原始输出,
    介入步的 BC 目标 = 参考 + (人 − 参考)的块内二次拟合并裁剪。
  * 介入标记(intervene_flag)包含暂停/回锁时的原地保持与进入介入的渐变,它们不是人的纠正。
    forward_inputs.iv_kind 把每步分成 0 自主 / 1 保持 / 2 渐变 / 3 主动,只有主动步做人类 BC。
  * 残差软限幅 π = r̃ + c·tanh(u/c),训练与执行共用 soft_limit();夹爪可退出残差(直接跟 VLA)。
    c 由 residual_limit_vector() 给出(A1 每关节同一弧度;天机每维一个笛卡尔上限)。
"""
from __future__ import annotations

import torch

KIND_AUTO, KIND_HOLD, KIND_RAMP, KIND_ACTIVE = 0, 1, 2, 3
REF_CLAMP = 0.999


def ref_tilde(ref: torch.Tensor) -> torch.Tensor:
    return ref.clamp(-REF_CLAMP, REF_CLAMP)


def soft_limit(rt: torch.Tensor, a_raw: torch.Tensor, c: torch.Tensor | None,
               gripper_dims=(), exclude_gripper: bool = False, ref_raw: torch.Tensor | None = None) -> torch.Tensor:
    """rt = clamp(ref) [B,C,D];a_raw = tanh(atanh(rt)+Δ) [B,C,D];c [D] 每维残差上限(归一化),None 表示不限。
    关节:π = rt + c·tanh((a_raw − rt)/c),落在 [-1,1] 内。夹爪(exclude_gripper):π = clamp(ref_raw, ±1)。"""
    if c is None:
        out = a_raw
    else:
        cc = c.to(a_raw.dtype).to(a_raw.device).reshape(1, 1, -1)
        u = a_raw - rt
        out = rt + cc * torch.tanh(u / cc)
    if exclude_gripper and len(gripper_dims) > 0:
        g = list(gripper_dims)
        src = ref_raw if ref_raw is not None else rt
        out = out.clone()
        out[..., g] = src[..., g].clamp(-1.0, 1.0)
    return out


def residual_limit_vector(action_scales, limit_rad: float, gripper_dims=(), limits=None) -> torch.Tensor | None:
    """按弧度给的残差上限 → 各维归一化上限;夹爪维给一个大数(不限)。limit_rad<=0 → None(不限)。
    limits:改为每维一个物理上限(天机笛卡尔帧:位置 m、rot6 原值、夹爪闭合度),<=0 的维不限,夹爪也按给定值。"""
    if limits is not None:
        limits = [float(x) for x in limits]
        if len(limits) != len(action_scales):
            raise ValueError(f"residual_limits has {len(limits)} values for {len(action_scales)} action dims")
        return torch.tensor([x / float(s) if x > 0 else 10.0 for x, s in zip(limits, action_scales)],
                            dtype=torch.float32)
    if limit_rad is None or float(limit_rad) <= 0:
        return None
    c = torch.tensor([float(limit_rad) / float(s) for s in action_scales], dtype=torch.float32)
    for g in gripper_dims:
        c[g] = 10.0
    return c


def switch_steps_from_flags(flags: torch.Tensor) -> torch.Tensor:
    """逐步阶段号 [B,C](切换后为新阶段号,latch)→ 块内切阶段之后的步 [B,C] int8。"""
    st = flags.reshape(flags.shape[0], -1).long()
    changed = (st != st[:, :1]) & (st > 0)
    return torch.cummax(changed.to(torch.int8), dim=1).values


def _quadratic_fit_runs(res: torch.Tensor, active: torch.Tensor, min_run: int):
    """res [B,C,D] 残差,active [B,C] bool。对每个样本内连续的 active 段做逐维二次最小二乘拟合。
    返回 fitted [B,C,D](非拟合处为 0)与 fit_ok [B,C] bool(段长 ≥ min_run 的 active 步)。"""
    B, C, D = res.shape
    dev = res.device
    a = active.bool()
    start = a & ~torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=dev), a[:, :-1]], 1)
    run_id = torch.cumsum(start.long(), dim=1) * a.long()          # 0 = 非 active
    t = torch.arange(C, device=dev, dtype=res.dtype)
    X = torch.stack([torch.ones_like(t), t / C, (t / C) ** 2], -1)   # [C,3],归一化时间避免病态
    fitted = torch.zeros_like(res)
    fit_ok = torch.zeros(B, C, dtype=torch.bool, device=dev)
    max_runs = int(run_id.max().item()) if a.any() else 0
    eye = torch.eye(3, device=dev, dtype=res.dtype) * 1e-6
    for r in range(1, max_runs + 1):
        m = (run_id == r)                                            # [B,C]
        n = m.sum(1)                                                 # [B]
        sel = n >= min_run
        if not sel.any():
            continue
        w = m.to(res.dtype)                                          # [B,C]
        Xw = X.unsqueeze(0) * w.unsqueeze(-1)                        # [B,C,3]
        A = Xw.transpose(1, 2) @ X.unsqueeze(0).expand(B, -1, -1) + eye   # [B,3,3]
        bvec = Xw.transpose(1, 2) @ res                              # [B,3,D]
        coef = torch.linalg.solve(A, bvec)                           # [B,3,D]
        f = X.unsqueeze(0) @ coef                                    # [B,C,D]
        mm = (m & sel.unsqueeze(1))
        fitted = torch.where(mm.unsqueeze(-1), f, fitted)
        fit_ok |= mm
    return fitted, fit_ok


def _near_boundary(mask: torch.Tensor, steps: int, before: bool) -> torch.Tensor:
    """mask [B,C] bool。before=True:返回在某个 mask 段开始之前 steps 步以内的位置;False:段结束之后 steps 步以内。"""
    B, C = mask.shape
    out = torch.zeros_like(mask)
    m = mask.long()
    for k in range(1, steps + 1):
        if before:
            shifted = torch.cat([m[:, k:], torch.zeros(B, k, dtype=m.dtype, device=m.device)], 1)
        else:
            shifted = torch.cat([torch.zeros(B, k, dtype=m.dtype, device=m.device), m[:, :-k]], 1)
        out |= shifted.bool()
    return out & ~mask


def still_edge_mask(actions: torch.Tensor, active: torch.Tensor, eps: float, edge_min: int) -> torch.Tensor:
    """主动介入段首/段尾"人停着不动"的步 [B,C] bool(2026-10-06 用户确认剔除:它们会教 actor"纠正完就停住")。
    静止 = 两臂关节与夹爪相邻两步变化都 < eps(块首步看它与下一步)。块内一段连续的"主动且静止":
      * 在块内紧挨着非主动步(保持/渐变/自主)→ 是段首或段尾,剔除;
      * 只碰到块的边缘(看不到相邻块,分不清是否段首尾)→ 连续 ≥ edge_min 步才剔除;
      * 段中间的短暂停顿、只动夹爪的步保留。"""
    B, C, _ = actions.shape
    d = (actions[:, 1:] - actions[:, :-1]).abs().amax(-1)               # [B,C-1]
    d = torch.cat([d[:, :1], d], 1)                                      # 块首步用它与下一步的变化
    s = active & (d < eps)
    out = torch.zeros_like(s)
    for b in torch.nonzero(s.any(1)).flatten().tolist():
        row = s[b].tolist(); act = active[b].tolist(); i = 0
        while i < C:
            if not row[i]:
                i += 1; continue
            j = i
            while j < C and row[j]:
                j += 1
            touches_iv_edge = (i > 0 and not act[i - 1]) or (j < C and not act[j])
            touches_block_edge = i == 0 or j == C
            if touches_iv_edge or (touches_block_edge and j - i >= edge_min):
                out[b, i:j] = True
            i = j
    return out


def build_bc_targets(
    ref_raw: torch.Tensor,          # [B,C,D] VLA 原始参考(actor 读的那一段)
    actions: torch.Tensor,          # [B,C,D] 实际执行动作(介入步 = 人的动作)
    iv_kind: torch.Tensor,          # [B,C] int
    switch_step: torch.Tensor | None,   # [B,C] int,块内切阶段之后为 1;None/全 0 → 用 stage_changed
    stage_changed: torch.Tensor | None, # [B] bool,块首与下块块首阶段不同(旧数据退化用)
    c: torch.Tensor | None,         # [D] 残差上限(归一化)
    *,
    human_weight: float,
    clip_frac: float = 0.8,
    min_run: int = 5,
    guard_steps: int = 5,
    trim_still: bool = False,
    still_eps: float = 1e-3,
    still_edge_min: int = 8,
):
    """返回 target [B,C,D],step_weight [B,C](自主步 1、主动步 human_weight、其余 0),info dict。
    trim_still:主动介入段首/段尾人停着不动的步不算 BC,且不参与二次拟合(见 still_edge_mask)。"""
    B, C, D = ref_raw.shape
    rt = ref_tilde(ref_raw)
    kind = iv_kind.reshape(B, C).long()
    active = kind == KIND_ACTIVE
    anyiv = kind != KIND_AUTO
    still = still_edge_mask(actions, active, still_eps, still_edge_min) if trim_still else torch.zeros_like(active)
    active = active & ~still
    target = rt.clone()
    step_w = torch.ones(B, C, dtype=ref_raw.dtype, device=ref_raw.device)
    step_w[anyiv] = 0.0
    if active.any():
        res = actions - rt
        fitted, fit_ok = _quadratic_fit_runs(res, active, min_run)
        if c is not None:
            cc = c.to(ref_raw.dtype).to(ref_raw.device).reshape(1, 1, -1)
            # 可达上界是 c·tanh((±1−rt)/c),再乘 clip_frac
            hi = clip_frac * cc * torch.tanh((1.0 - rt) / cc)
            lo = clip_frac * cc * torch.tanh((-1.0 - rt) / cc)
            fitted = torch.maximum(torch.minimum(fitted, hi), lo)
        target = torch.where(fit_ok.unsqueeze(-1), rt + fitted, target)
        step_w = torch.where(fit_ok, torch.full_like(step_w, float(human_weight)), step_w)
    # 介入段前后各 guard_steps 步的自主步不算 BC(目标处有拐点)
    guard = _near_boundary(anyiv, guard_steps, before=True) | _near_boundary(anyiv, guard_steps, before=False)
    step_w = torch.where(guard & (kind == KIND_AUTO), torch.zeros_like(step_w), step_w)
    # 块内切阶段:之后的主动步不算(旧数据没有逐步标记时,整块主动步不算)
    if switch_step is not None and bool((switch_step != 0).any()):
        sw = switch_step.reshape(B, C) != 0
        step_w = torch.where(sw & active, torch.zeros_like(step_w), step_w)
    elif stage_changed is not None:
        sc = stage_changed.reshape(B, 1).bool() & active
        step_w = torch.where(sc, torch.zeros_like(step_w), step_w)
    info = {
        "active_ratio": active.float().mean().item(),
        "hold_ratio": (kind == KIND_HOLD).float().mean().item(),
        "ramp_ratio": (kind == KIND_RAMP).float().mean().item(),
        "still_trim_ratio": still.float().mean().item(),
        "human_bc_steps": int((step_w > 1.0 + 1e-6).sum().item()) if human_weight > 1 else int((active & (step_w > 0)).sum().item()),
    }
    return target, step_w, info
