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

"""In-process smoke of collect / fit / imagine / checkpoint.

Uses MuJoCo Warp and the G1 asset pose as the reference. That pose is not
a LAFAN1 retarget. Three updates check the chain and the numerics, not a
tracking score. The official entry is ``train_supermimic.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("mjlab")
if not torch.cuda.is_available():
    pytest.skip("MuJoCo Warp smoke needs CUDA", allow_module_level=True)

from omegaconf import OmegaConf

from rlinf.workers.supermimic.trainer import SuperMimicTrainer
from rlinf.workers.supermimic.validate import validate_supermimic_cfg

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples" / "embodiment" / "config"


def _write_asset_pose_clip(directory: Path, frames: int = 200) -> None:
    """Repeat the G1 asset keyframe. This is not a LAFAN1 retarget."""
    from rlinf.envs.sim.g1_mimic.constants import G1_JOINT_NAMES
    from rlinf.envs.sim.g1_mimic.g1_fk import asset_default_state

    state = asset_default_state()
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "fps": 50.0,
            "joint_names": list(G1_JOINT_NAMES),
            "q0": state[7:36].clone(),
            "state": state.unsqueeze(0).repeat(frames, 1),
        },
        directory / "clip_000.pt",
    )


def test_smoke_three_updates_and_checkpoint(tmp_path):
    cfg = OmegaConf.merge(
        OmegaConf.load(CONFIG_DIR / "supermimic_base.yaml"),
        OmegaConf.load(CONFIG_DIR / "supermimic_lafan1.yaml"),
        OmegaConf.load(CONFIG_DIR / "supermimic_smoke.yaml"),
    )
    cfg.pop("defaults", None)
    cfg.pop("hydra", None)
    _write_asset_pose_clip(tmp_path / "pose")
    cfg.env.motion_path = str(tmp_path / "pose")
    cfg.env.num_clips = 1
    cfg = validate_supermimic_cfg(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = SuperMimicTrainer(cfg, device=device)
    metrics = trainer.run_until_updates(3)

    assert int(metrics["train/global_update"]) == 3
    assert metrics["buffer/valid_windows"] > 0
    assert math.isfinite(metrics["wm/state_loss"])
    assert math.isfinite(metrics["actor/objective"])
    assert math.isfinite(metrics["critic/loss"])
    assert math.isfinite(metrics["wm/grad_norm"]) and metrics["wm/grad_norm"] > 0
    assert math.isfinite(metrics["actor/grad_norm"]) and metrics["actor/grad_norm"] > 0
    assert 0.0 <= metrics["imagination/survival"] <= 1.0
    assert math.isfinite(metrics["sim/terminal_rate"])
    assert "memory/peak_allocated" in metrics
    assert metrics["time/replay_sample_s"] > 0.0
    assert metrics["time/world_model_s"] > 0.0
    assert metrics["time/bptt_s"] > 0.0
    assert metrics["time/critic_s"] > 0.0
    assert metrics["time/sim_s"] > 0.0
    assert metrics["wm_optimizer_steps"] >= 10
    assert metrics["actor_optimizer_steps"] >= 3

    ckpt = tmp_path / "global_step_3"
    trainer.save_checkpoint(str(ckpt))
    assert (ckpt / "supermimic.pt").is_file()
    trainer.load_checkpoint(str(ckpt))
    assert trainer.global_update == 3
    assert trainer.replay_warm is False
    assert torch.allclose(
        trainer.reset_sampler.bin_return,
        torch.full_like(trainer.reset_sampler.bin_return, trainer.G_max),
    )
    steps_before = trainer.sim_vector_steps
    acc_before = trainer.step_acc
    trainer.run_iteration()
    # Warmup refills the new buffer and must not add to the UTD accumulator.
    # Only the one post-warmup collection step does.
    accounted = trainer.step_acc + (trainer.global_update - 3)
    assert trainer.replay_warm is True
    assert trainer.sim_vector_steps == steps_before + trainer.warmup + 1
    assert math.isclose(accounted, acc_before + trainer.updates_per_env_step, abs_tol=1e-6)

    eval_metrics = trainer.evaluate()
    for key in (
        "eval/normalized_return",
        "eval/torso_pos_err",
        "eval/free_root_pos_err",
        "eval/free_root_xy_drift",
        "eval/free_root_yaw_drift",
        "eval/fail_rate",
        "eval/fail_time_s",
        "eval/episode_length_s",
        "eval/clip_end_rate",
    ):
        assert math.isfinite(eval_metrics[key])
    assert eval_metrics["eval/num_clips"] == pytest.approx(1.0)

    # A new trainer must warm the new buffer. The restored step count is not
    # evidence that the empty replay is ready.
    del trainer
    torch.cuda.empty_cache()
    resumed = SuperMimicTrainer(cfg, device=device)
    resumed.load_checkpoint(str(ckpt))
    assert resumed.replay_warm is False
    restored_steps = resumed.sim_vector_steps
    resumed.run_iteration()
    assert resumed.replay_warm is True
    assert resumed.sim_vector_steps == restored_steps + resumed.warmup + 1
