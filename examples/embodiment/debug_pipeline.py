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

"""Standalone pipeline debug tool — test model inference without a robot.

Loads the OpenPI model, feeds it synthetic (or saved) observations, runs
``predict_action_batch``, and prints the output actions. No robot, no gRPC,
no Ray required.

Usage::

    # Basic smoke test (random dummy observations):
    python examples/embodiment/debug_pipeline.py \
        --model-path thomas0829/folding_towel_pi05

    # With a saved observation snapshot (for reproducibility):
    python examples/embodiment/debug_pipeline.py \
        --model-path thomas0829/folding_towel_pi05 \
        --load-obs snapshot.pt

    # Save current dummy observation for later comparison:
    python examples/embodiment/debug_pipeline.py \
        --model-path thomas0829/folding_towel_pi05 \
        --save-obs snapshot.pt

    # Compare actions against a saved reference:
    python examples/embodiment/debug_pipeline.py \
        --model-path thomas0829/folding_towel_pi05 \
        --load-obs snapshot.pt \
        --reference-actions reference.pt

    # Run N iterations to check for consistency / warmup:
    python examples/embodiment/debug_pipeline.py \
        --model-path thomas0829/folding_towel_pi05 \
        --iterations 5

"""

import argparse
import time

import numpy as np
import torch

from rlinf.utils.logging import get_logger

logger = get_logger()


def make_dummy_obs(
    config_name: str,
    img_h: int,
    img_w: int,
    state_dim: int,
    seed: int | None = None,
) -> dict:
    """Create synthetic observations matching the expected pipeline input."""
    rng = np.random.RandomState(seed)

    # Random uint8 images (HWC, batch dim added)
    main_image = rng.randint(0, 256, (1, img_h, img_w, 3), dtype=np.uint8)
    states = rng.randn(1, state_dim).astype(np.float32) * 0.1

    obs = {
        "main_images": torch.from_numpy(main_image),
        "states": torch.from_numpy(states),
        "task_descriptions": ["debug pipeline test"],
    }

    if config_name == "pi05_yam_follower":
        # YAM PI0.5 needs 3 cameras: left, right packed in wrist_images,
        # plus the main (top) camera.
        left_img = rng.randint(0, 256, (1, img_h, img_w, 3), dtype=np.uint8)
        right_img = rng.randint(0, 256, (1, img_h, img_w, 3), dtype=np.uint8)
        # wrist_images shape: (batch, num_wrist_cams, H, W, C)
        wrist_images = np.stack([left_img[:, None], right_img[:, None]], axis=1)
        obs["wrist_images"] = torch.from_numpy(wrist_images.squeeze(2))
    elif "pi05" in config_name:
        # Other PI0.5 configs typically expect a wrist image
        wrist_image = rng.randint(0, 256, (1, img_h, img_w, 3), dtype=np.uint8)
        obs["wrist_images"] = torch.from_numpy(wrist_image[:, None])

    return obs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug the OpenPI inference pipeline without a robot"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="thomas0829/folding_towel_pi05",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_yam_follower",
        help="OpenPI config name",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=14,
        help="Action dimension (14 for YAM bimanual)",
    )
    parser.add_argument(
        "--action-chunk",
        type=int,
        default=30,
        help="Action chunk size",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Denoising steps",
    )
    parser.add_argument(
        "--img-height",
        type=int,
        default=360,
        help="Image height",
    )
    parser.add_argument(
        "--img-width",
        type=int,
        default=640,
        help="Image width",
    )
    parser.add_argument(
        "--state-dim",
        type=int,
        default=14,
        help="State dimension (14 for YAM bimanual)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dummy observations",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of inference iterations to run",
    )
    parser.add_argument(
        "--save-obs",
        type=str,
        default=None,
        help="Save the dummy observation to a .pt file",
    )
    parser.add_argument(
        "--load-obs",
        type=str,
        default=None,
        help="Load observation from a .pt file instead of generating dummy obs",
    )
    parser.add_argument(
        "--save-actions",
        type=str,
        default=None,
        help="Save the output actions to a .pt file (for future reference comparison)",
    )
    parser.add_argument(
        "--reference-actions",
        type=str,
        default=None,
        help="Compare output against saved reference actions (.pt file)",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="debug pipeline test",
        help="Task description prompt",
    )
    return parser


@torch.no_grad()
def main():
    args = build_arg_parser().parse_args()

    # --- Load model ---
    from examples.embodiment.infer_embodied_agent import (
        load_model,
    )

    logger.info("Loading model...")
    model = load_model(
        args.model_path,
        args.config_name,
        args.action_dim,
        args.action_chunk,
        args.num_steps,
    )
    device = next(model.parameters()).device
    logger.info(f"Model loaded on {device}, dtype={next(model.parameters()).dtype}")

    # --- Prepare observation ---
    if args.load_obs:
        logger.info(f"Loading observation from {args.load_obs}")
        obs = torch.load(args.load_obs, map_location="cpu", weights_only=False)
    else:
        logger.info("Generating dummy observation (seed=%d)", args.seed)
        obs = make_dummy_obs(
            config_name=args.config_name,
            img_h=args.img_height,
            img_w=args.img_width,
            state_dim=args.state_dim,
            seed=args.seed,
        )

    obs["task_descriptions"] = [args.task_description]

    if args.save_obs:
        torch.save(obs, args.save_obs)
        logger.info(f"Saved observation to {args.save_obs}")

    # Move tensors to device
    env_obs = {}
    for key, value in obs.items():
        if torch.is_tensor(value):
            env_obs[key] = value.to(device)
        else:
            env_obs[key] = value

    # --- Print observation summary ---
    logger.info("--- Observation Summary ---")
    for key, value in env_obs.items():
        if torch.is_tensor(value):
            logger.info(f"  {key}: shape={tuple(value.shape)}, dtype={value.dtype}")
        else:
            logger.info(f"  {key}: {value}")

    # --- Run inference ---
    all_actions = []
    for i in range(args.iterations):
        t0 = time.perf_counter()
        actions, result = model.predict_action_batch(
            env_obs, mode="eval", compute_values=False
        )
        dt = time.perf_counter() - t0
        actions_np = actions.detach().cpu().numpy()
        all_actions.append(actions_np)

        logger.info(f"--- Iteration {i + 1}/{args.iterations} ({dt:.3f}s) ---")
        logger.info(f"  Actions shape: {actions_np.shape}")
        logger.info(
            f"  Actions range: [{actions_np.min():.6f}, {actions_np.max():.6f}]"
        )
        logger.info(f"  Actions mean:  {actions_np.mean():.6f}")
        logger.info(f"  Actions std:   {actions_np.std():.6f}")
        with np.printoptions(precision=6, suppress=True):
            logger.info(f"  First action:  {actions_np[0, 0]}")
            if actions_np.shape[1] > 1:
                logger.info(f"  Last action:   {actions_np[0, -1]}")

    # --- Save actions ---
    if args.save_actions:
        torch.save(torch.from_numpy(all_actions[-1]), args.save_actions)
        logger.info(f"Saved actions to {args.save_actions}")

    # --- Compare against reference ---
    if args.reference_actions:
        ref = torch.load(args.reference_actions, map_location="cpu", weights_only=False)
        ref_np = ref.numpy() if torch.is_tensor(ref) else np.asarray(ref)
        current = all_actions[-1]

        abs_diff = np.abs(current - ref_np)
        max_diff = abs_diff.max()
        mean_diff = abs_diff.mean()
        logger.info("--- Reference Comparison ---")
        logger.info(f"  Max absolute diff:  {max_diff:.8f}")
        logger.info(f"  Mean absolute diff: {mean_diff:.8f}")
        if max_diff < 1e-4:
            logger.info("  PASS: Actions match reference (max diff < 1e-4)")
        elif max_diff < 1e-2:
            logger.info("  WARN: Actions close to reference (max diff < 1e-2)")
        else:
            logger.info("  FAIL: Actions differ significantly from reference")

    # --- Check consistency across iterations ---
    if args.iterations > 1:
        logger.info("--- Iteration Consistency ---")
        for i in range(1, len(all_actions)):
            diff = np.abs(all_actions[i] - all_actions[0]).max()
            logger.info(f"  Iter {i + 1} vs Iter 1 max diff: {diff:.8f}")

    logger.info("Pipeline debug complete.")


if __name__ == "__main__":
    main()
