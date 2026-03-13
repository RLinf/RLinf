#!/usr/bin/env python3
"""
Test script for DROID environment (RLinf).

Prerequisites:
  - sim_evals and isaaclab installed (see docs/env_droid_install_and_test.md)
  - DROID assets downloaded to sim-evals/assets (run from sim-evals root:
      uvx hf download owhan/DROID-sim-environments --repo-type dataset --local-dir assets)

Usage:
  # From RLinf repo root, with env that has sim_evals + isaaclab:
  python scripts/test_droid_env.py

  # Import-only check (no subprocess):
  python scripts/test_droid_env.py --import-only
"""

import argparse
import sys


def check_imports():
    """Check that sim_evals and isaaclab are importable."""
    missing = []
    try:
        from isaaclab.app import AppLauncher  # noqa: F401
    except ImportError as e:
        missing.append(f"isaaclab: {e}")
    try:
        import sim_evals.environments  # noqa: F401
    except ImportError as e:
        missing.append(f"sim_evals: {e}")
    try:
        from isaaclab_tasks.utils import parse_env_cfg  # noqa: F401
    except ImportError as e:
        missing.append(f"isaaclab_tasks: {e}")
    if missing:
        print("Missing dependencies:", missing)
        print("See docs/env_droid_install_and_test.md for installation.")
        return False
    print("Imports OK (isaaclab, sim_evals, isaaclab_tasks)")
    return True


def run_full_test():
    """Create RLinf DroidEnv in subprocess, reset + one step + close."""
    import numpy as np
    from omegaconf import OmegaConf

    from rlinf.envs.droid import DroidEnv

    cfg = OmegaConf.create({
        "seed": 0,
        "group_size": 1,
        "ignore_terminations": False,
        "auto_reset": False,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "max_episode_steps": 256,
        "video_cfg": {},
        "init_params": {
            "scene": 1,
            "device": "cuda:0",
            "task_description": "put the cube in the bowl",
            "headless": True,
            "warmup_reset": True,
        },
    })

    print("Creating DroidEnv (subprocess will start Isaac Sim)...")
    env = DroidEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
    )
    print("Reset...")
    obs, info = env.reset()
    print("Obs keys:", list(obs.keys()))
    if "main_images" in obs:
        print("  main_images shape:", obs["main_images"].shape)
    if "states" in obs:
        print("  states shape:", obs["states"].shape)
    print("Step (zero action)...")
    action = np.zeros((1, 8), dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action, auto_reset=False)
    print("Close...")
    env.close()
    print("DROID env test: OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test DROID environment for RLinf")
    parser.add_argument(
        "--import-only",
        action="store_true",
        help="Only check imports, do not create env (no subprocess).",
    )
    args = parser.parse_args()

    if not check_imports():
        sys.exit(1)
    if args.import_only:
        sys.exit(0)

    try:
        run_full_test()
    except Exception as e:
        print("Test failed:", e)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
