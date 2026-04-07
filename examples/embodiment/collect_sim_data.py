#!/usr/bin/env python3
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
#
# Collect human demonstrations in simulation using SpaceMouse teleoperation.
#
# The script wraps a GenieSimEnv with SpacemouseSimIntervention (which injects
# SpaceMouse input for env_0) and CollectEpisode (which saves completed demos
# to disk).  Only the right arm end-effector is controlled; the left arm holds
# its reset pose.
#
# Usage — real SpaceMouse:
#   cd RLinf
#   GENIESIM_ROOT=.. python examples/embodiment/collect_sim_data.py \
#       --config examples/embodiment/config/env/geniesim_place_workpiece.yaml \
#       --save-dir /tmp/sim_demos \
#       --num-demos 10
#       --num-demos 10
#
# SpaceMouse controls — place_workpiece task:
#   Translate device     → move right arm EEF (x/y/z)
#   Rotate device        → rotate right arm EEF (roll/pitch/yaw)
#   Left  button (press) → end trajectory; save demo; env resets
#   Right button (press) → end trajectory; discard demo; env resets
#   (No gripper button — place_workpiece collection does not use gripper-close on SpaceMouse.)

import argparse
import atexit
import os
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict


# ---------------------------------------------------------------------------
# Resolve the RLinf repo root so that `rlinf.*` is importable when the script
# is run from any working directory (not just the repo root).
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_RLINF_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))
if _RLINF_ROOT not in sys.path:
    sys.path.insert(0, _RLINF_ROOT)

# Ensure GENIESIM_ROOT is set so that GenieSimBaseEnv can find main/source.
if "GENIESIM_ROOT" not in os.environ:
    _default_gs_root = os.path.abspath(os.path.join(_RLINF_ROOT, ".."))
    os.environ["GENIESIM_ROOT"] = _default_gs_root
    print(f"[collect_sim_data] GENIESIM_ROOT not set; defaulting to {_default_gs_root}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Collect sim demonstrations with SpaceMouse teleoperation."
    )
    p.add_argument(
        "--config",
        default="examples/embodiment/config/env/geniesim_place_workpiece.yaml",
        help="Path to the GenieSimEnv OmegaConf config YAML.",
    )
    p.add_argument(
        "--save-dir",
        default="/tmp/sim_demos",
        help="Directory where collected episodes are written.",
    )
    p.add_argument(
        "--num-demos",
        type=int,
        default=5,
        help="Number of successful demos to collect before exiting.",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of parallel simulation instances (SpaceMouse controls env_0).",
    )
    p.add_argument(
        "--export-format",
        choices=["pickle", "lerobot"],
        default="pickle",
        help="Format for saved episode data.",
    )
    p.add_argument(
        "--action-scale",
        type=float,
        default=0.01,
        help="SpaceMouse translational output → EEF position delta scale (m/unit).",
    )
    p.add_argument(
        "--rotation-scale",
        type=float,
        default=0.04,
        help="SpaceMouse rotational output → EEF orientation delta scale (rad/unit).",
    )
    p.add_argument(
        "--step-hz",
        type=float,
        default=10.0,
        help="Target control frequency (Hz); 0 disables rate limiting.",
    )
    return p


# ---------------------------------------------------------------------------
# Main collection loop
# ---------------------------------------------------------------------------

def main():
    args = _build_parser().parse_args()

    # ---- 1. Load config -------------------------------------------------- #
    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(_RLINF_ROOT, cfg_path)
    if not os.path.exists(cfg_path):
        sys.exit(f"[collect_sim_data] Config not found: {cfg_path}")

    cfg = OmegaConf.load(cfg_path)
    with open_dict(cfg):
        cfg.init_params.num_envs = args.num_envs
        cfg.init_params.headless = False

    action_dim: int = cfg.init_params.get("action_dim", 14)
    model_action_dim: int = cfg.init_params.get("model_action_dim", action_dim)

    # ---- 2. Instantiate GenieSimEnv -------------------------------------- #
    # Import triggers @register_geniesim_env decorators
    from rlinf.envs.geniesim import REGISTER_GENIESIM_ENVS
    from rlinf.envs.geniesim.tasks import PlaceWorkpieceEnv  # noqa: F401

    task_id = cfg.init_params.id
    if task_id not in REGISTER_GENIESIM_ENVS:
        sys.exit(
            f"[collect_sim_data] Unknown task_id={task_id!r}. "
            f"Registered: {list(REGISTER_GENIESIM_ENVS.keys())}"
        )

    EnvCls = REGISTER_GENIESIM_ENVS[task_id]
    print(f"[collect_sim_data] Creating {EnvCls.__name__} "
          f"(num_envs={args.num_envs}, task={task_id})")

    base_env = EnvCls(
        cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )
    base_env._collecting = True
    atexit.register(base_env.close)

    # ---- 3. Build SpaceMouse expert -------------------------------------- #
    from rlinf.envs.wrappers.spacemouse_sim_intervention import (
        SpacemouseSimIntervention,
    )

    try:
        from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import (
            SpaceMouseExpert,
        )
    except ModuleNotFoundError as e:
        sys.exit(
            f"\n[collect_sim_data] ERROR: SpaceMouse driver not installed ({e}).\n"
            f"  Install it with:  pip install pyspacemouse easyhid\n"
        )
    expert = SpaceMouseExpert()
    print("[collect_sim_data] Using real SpaceMouse device.")
    print()
    print("  Controls (place_workpiece):")
    print("    Move device     → translate right arm EEF")
    print("    Twist device    → rotate right arm EEF")
    print("    Left  button    → end episode & save demo")
    print("    Right button    → end episode & discard demo")
    print()

    # ---- 4. Stack wrappers: SpaceMouse → CollectEpisode ------------------ #
    sm_kwargs = getattr(base_env, "spacemouse_wrapper_kwargs", {})
    sm_env = SpacemouseSimIntervention(
        base_env,
        expert=expert,
        action_scale=args.action_scale,
        rotation_scale=args.rotation_scale,
        intervention_env_id=0,
        button_mode="place_workpiece",
        **sm_kwargs,
    )

    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    os.makedirs(args.save_dir, exist_ok=True)
    env = CollectEpisode(
        sm_env,
        save_dir=args.save_dir,
        rank=0,
        num_envs=args.num_envs,
        export_format=args.export_format,
        only_success=True,   # only save episodes ended via left-button (place_workpiece)
    )
    atexit.register(env.close)

    # ---- 5. Collection loop ---------------------------------------------- #
    print(f"[collect_sim_data] Collecting {args.num_demos} demos → {args.save_dir}")
    step_dt = 1.0 / args.step_hz if args.step_hz > 0 else 0.0

    demos_collected = 0
    total_steps = 0

    obs, _ = env.reset()
    episode_steps = 0
    t_last = time.monotonic()

    while demos_collected < args.num_demos:
        # Zero action — SpacemouseSimIntervention overwrites env_0's slice
        actions = torch.zeros(args.num_envs, model_action_dim, dtype=torch.float32)

        obs, reward, terminated, truncated, info = env.step(actions)
        episode_steps += 1
        total_steps += 1

        # Rate limiting
        if step_dt > 0:
            elapsed = time.monotonic() - t_last
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)
        t_last = time.monotonic()

        # Check whether env_0 finished
        done_0 = bool(terminated[0]) or bool(truncated[0])
        if done_0:
            is_success = info.get("success", False)
            status = "SUCCESS" if is_success else "truncated"
            print(
                f"[collect_sim_data] Episode ended ({status}) "
                f"after {episode_steps} steps  —  "
                f"demos saved: {demos_collected + (1 if is_success else 0)}"
                f"/{args.num_demos}"
            )
            if is_success:
                demos_collected += 1

            if demos_collected < args.num_demos:
                obs, _ = env.reset()
                episode_steps = 0

    print(
        f"\n[collect_sim_data] Done. "
        f"{demos_collected} demos saved to {args.save_dir}  "
        f"({total_steps} total steps)."
    )
    env.close()


if __name__ == "__main__":
    main()
