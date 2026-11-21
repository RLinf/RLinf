"""A1 单臂正运动学(取自官方 URDF A1_URDF_0607_0028.urdf)与复位用的"指尖竖直抬起"数值逆解。

坐标系:手臂基座,z 向上。指尖 = 两指根中点沿腕→法兰方向外延 TIP_LEN(估计值,只用于相对高度)。
"""
from __future__ import annotations

import numpy as np

# (xyz, rpy),依次为 arm_joint1..6,全部绕本地 z 轴转
_CHAIN = [
    ((-0.0011147, 0.0, 0.0892), (0.0, 0.0, 3.1416)),
    ((0.0, -4e-05, 0.0615), (1.5708, 0.0, 0.0)),
    ((0.34928, 0.02, 0.0), (0.0, 0.0, 1.5708)),
    ((0.07, -0.00395, -4e-05), (-1.5708, 1.5708, 0.0)),
    ((0.0, 0.0, 0.2776), (-1.5708, 0.0, 3.1416)),
    ((0.0, -0.1575, -0.00023266), (1.5708, 0.0, 0.0)),
]
_FINGER_ROOTS = (np.array([0.014, -0.011914, 0.003125]), np.array([-0.014, 0.011914, 0.003125]))
TIP_LEN = 0.13


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


_FIXED = []
for _xyz, _r in _CHAIN:
    _T = np.eye(4); _T[:3, :3] = _rpy(*_r); _T[:3, 3] = _xyz
    _FIXED.append(_T)


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    T = np.eye(4); T[:2, :2] = [[c, -s], [s, c]]
    return T


def fingertip(q) -> np.ndarray:
    """6 维关节角 → 指尖位置 (3,)。"""
    T = np.eye(4)
    wrist = None
    for k in range(6):
        T = T @ _FIXED[k] @ _rz(float(q[k]))
        if k == 4:
            wrist = T[:3, 3].copy()
    flange = T[:3, 3]
    roots = [(T @ np.append(r, 1.0))[:3] for r in _FINGER_ROOTS]
    d = flange - wrist
    d /= max(np.linalg.norm(d), 1e-9)
    return (roots[0] + roots[1]) / 2 + d * TIP_LEN


def lift_path(q0, lo, hi, dz=0.12, z_cap=0.45, max_step=0.03, wrist_w=0.2, iters=80):
    """指尖竖直上抬 dz(不超过 z_cap),水平位置保持,腕部(J4-J6)少动。
    返回关节路径(含起点),每步各关节变化 ≤ max_step,始终在 [lo, hi] 内。到不了就停在能到的最高处。"""
    q = np.clip(np.asarray(q0, np.float64).copy(), lo, hi)
    path = [q.copy()]
    p0 = fingertip(q)
    z_goal = min(p0[2] + dz, max(z_cap, p0[2]))
    if z_goal - p0[2] < 0.01:
        return path
    goal = np.array([p0[0], p0[1], z_goal])
    w = np.array([1, 1, 1, wrist_w, wrist_w, wrist_w])
    for _ in range(iters):
        p = fingertip(q)
        err = goal - p
        if np.linalg.norm(err) < 0.005:
            break
        J = np.zeros((3, 6))
        for i in range(6):
            dq = np.zeros(6); dq[i] = 1e-4
            J[:, i] = (fingertip(q + dq) - p) / 1e-4
        Jw = J * w
        step = w * (Jw.T @ np.linalg.solve(Jw @ Jw.T + 1e-3 * np.eye(3), 0.5 * err))
        q_new = np.clip(q + np.clip(step, -max_step, max_step), lo, hi)
        if np.abs(q_new - q).max() < 1e-5:      # 顶到限位,走不动了
            break
        q = q_new
        path.append(q.copy())
    return path
