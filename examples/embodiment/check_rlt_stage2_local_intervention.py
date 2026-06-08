"""Smoke-check RLT Stage2 local intervention state transitions.

Run from repo root:
    PYTHONPATH=aaapush/RLinf python3 aaapush/RLinf/examples/embodiment/check_rlt_stage2_local_intervention.py
"""

from __future__ import annotations

import torch
from omegaconf import OmegaConf

from rlinf.workers.env.env_worker import EnvWorker


class _DummyUnwrapped:
    def __init__(self):
        self.box_hole_radii = torch.tensor([0.035], dtype=torch.float32)


class _DummyEnv:
    def __init__(self):
        self.unwrapped = _DummyUnwrapped()


def _make_worker() -> EnvWorker:
    worker = object.__new__(EnvWorker)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "loss_type": "rlt_td3",
                "intervention": {
                    "enable": True,
                    "deviation_patience": 2,
                    "takeover_chunks": 5,
                    "takeover_max_chunks": 10,
                    "safe_yz_margin": 1.25,
                    "progress_eps": 0.002,
                    "yz_error_eps": 0.002,
                    "near_hole_x_min": -0.05,
                },
            },
            "actor": {"model": {"model_type": "rlt_stage2"}},
        }
    )
    worker.train_num_envs_per_stage = 1
    worker.env_list = [_DummyEnv()]
    worker.rlt_local_policy_state = []
    worker._init_rlt_local_policy_state(0)
    return worker


def _infos(
    *,
    hole_x: float,
    yz_error: float = 0.02,
    grasp: bool = True,
    success: bool = False,
    abs_y: float = 0.01,
    abs_z: float = 0.01,
) -> dict[str, torch.Tensor]:
    return {
        "consecutive_grasp_current": torch.tensor([grasp]),
        "prealigned_current": torch.tensor([yz_error < 0.01]),
        "partial_insert_current": torch.tensor([False]),
        "success_current": torch.tensor([success]),
        "peg_head_goal_yz_dist": torch.tensor([yz_error], dtype=torch.float32),
        "peg_body_goal_yz_dist": torch.tensor([yz_error], dtype=torch.float32),
        "peg_head_hole_x": torch.tensor([hole_x], dtype=torch.float32),
        "peg_head_hole_abs_y": torch.tensor([abs_y], dtype=torch.float32),
        "peg_head_hole_abs_z": torch.tensor([abs_z], dtype=torch.float32),
    }


def _update(
    worker: EnvWorker,
    *,
    hole_x: float,
    yz_error: float = 0.02,
    done: bool = False,
) -> dict[str, torch.Tensor]:
    chunk_dones = torch.full((1, 10), done, dtype=torch.bool)
    return worker._update_rlt_local_policy_state(
        _infos(hole_x=hole_x, yz_error=yz_error),
        chunk_dones,
        0,
    )


def _flag(policy_info: dict[str, torch.Tensor], key: str) -> bool:
    return bool(policy_info[key].reshape(-1)[0].item())


def _scalar(policy_info: dict[str, torch.Tensor], key: str) -> float:
    return float(policy_info[key].reshape(-1)[0].item())


def main() -> None:
    worker = _make_worker()

    policy_info = _update(worker, hole_x=-0.20)
    assert not _flag(policy_info, "deviation")
    assert not _flag(policy_info, "expert_takeover")

    policy_info = _update(worker, hole_x=-0.04)
    assert not _flag(policy_info, "deviation")
    assert _scalar(policy_info, "deviation_count") == 0.0
    assert not _flag(policy_info, "expert_takeover")

    policy_info = _update(worker, hole_x=-0.04)
    assert _flag(policy_info, "deviation")
    assert _scalar(policy_info, "deviation_count") == 1.0
    assert not _flag(policy_info, "expert_takeover")

    policy_info = _update(worker, hole_x=-0.04)
    assert _flag(policy_info, "expert_takeover")
    assert _scalar(policy_info, "takeover_left") == 5.0

    for step in range(4):
        policy_info = _update(worker, hole_x=-0.02 + step * 0.01)
        assert _flag(policy_info, "expert_takeover")

    policy_info = _update(worker, hole_x=0.03)
    assert not _flag(policy_info, "expert_takeover")
    assert _scalar(policy_info, "takeover_used") == 0.0

    policy_info = _update(worker, hole_x=-0.04, done=True)
    assert not _flag(policy_info, "deviation")
    assert not _flag(policy_info, "expert_takeover")

    print("RLT Stage2 local intervention smoke check passed.")


if __name__ == "__main__":
    main()
