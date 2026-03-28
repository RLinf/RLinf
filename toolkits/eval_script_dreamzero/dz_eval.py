#! /usr/bin/env python
"""Test client for DreamZero policy server using rlinf interface.

Sends real video frames from debug_image/ directory instead of zero dummy images.

Frame schedule (matching debug_inference.py):
  - Step 0 (initial): send frame [0]             (1 frame, H W 3)
  - Step 1: send frames [0, 7, 15, 23]           (4 frames, 4 H W 3)
  - Step 2: send frames [24, 31, 39, 47]         (4 frames)
  - Step 3: send frames [48, 55, 63, 71]         (4 frames)
  - ...

Expected server configuration:
    - image_resolution: (180, 320)
    - n_external_cameras: 2
    - needs_wrist_camera: True
    - action_space: "joint_position"

Usage:

    # Run this test:
    python dz_eval.py 

    # Use zero images instead of real video (old behavior):
    python dz_eval.py --use-zero-images
"""
import argparse
import logging
import os
import time
import uuid
import dataclasses

import cv2
import numpy as np

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")
VIDEO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "video_output")
if not os.path.exists(VIDEO_OUTPUT_DIR):
    os.makedirs(VIDEO_OUTPUT_DIR)

# roboarena key -> video filename
CAMERA_FILES = {
    "observation/exterior_image_0_left": "exterior_image_1_left.mp4",
    "observation/exterior_image_1_left": "exterior_image_2_left.mp4",
    "observation/wrist_image_left": "wrist_image_left.mp4",
}

# Frame schedule constants (matching debug_inference.py)
RELATIVE_OFFSETS = [-23, -16, -8, 0]
ACTION_HORIZON = 24


@dataclasses.dataclass
class PolicyImageConfig:
    # Resolution that images get resized to client-side, None means no resizing.
    # It's beneficial to resize images to the desired resolution client-side for faster communication.
    image_resolution: tuple[int, int] | None = (224, 224)
    # Whether or not wrist camera image(s) should be sent.
    needs_wrist_camera: bool = True
    # Number of external cameras to send.
    n_external_cameras: int = 1  # can be in [0, 1, 2]
    # Whether or not stereo camera image(s) should be sent.
    needs_stereo_camera: bool = False
    # Whether or not the unique eval session id should be sent (e.g. for policies that want to keep track of history).
    needs_session_id: bool = False
    # Which action space to use.
    action_space: str = "joint_position"  # can be in ["joint_position", "joint_velocity", "cartesian_position", "cartesian_velocity"]

def _make_zero_observation(
    image_config: PolicyImageConfig,
    prompt: str = "pick up the object",
    session_id: str | None = None,
) -> dict:
    """Create a dummy observation matching AR_droid expectations.
    
    AR_droid expects:
        - 2 external cameras (exterior_image_0_left, exterior_image_1_left)
        - 1 wrist camera (wrist_image_left)
        - Image resolution: 180x320 (H x W)
        - joint_position: 7 DoF
        - gripper_position: 1 DoF
    """
    obs = {}
    if image_config.image_resolution is not None:
        h, w = image_config.image_resolution
    else:
        h, w = 180, 320
    # external cameras (0 - index)
    for i in range(image_config.n_external_cameras):
        obs[f"observation/exterior_image_{i}_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if image_config.needs_stereo_camera:
            obs[f"observation/exterior_image_{i}_right"] = np.zeros((h, w, 3), dtype=np.uint8)

    # wrist camera
    if image_config.needs_wrist_camera:
        obs["observation/wrist_image_left"] = np.zeros((h, w, 3), dtype=np.uint8)
        if image_config.needs_stereo_camera:
            obs["observation/wrist_image_right"] = np.zeros((h, w, 3), dtype=np.uint8)

    # session id
    if image_config.needs_session_id:
        import uuid
        obs["session_id"] = session_id if session_id else str(uuid.uuid4())
    
    # state observations (7 DOF arm + 1 gripper)
    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)

    # prompt
    obs["prompt"] = prompt

    return obs

def _make_obs_from_video(
    camera_frames: dict[str, np.ndarray],
    frame_indices: list[int],
    prompt: str,
    session_id: str,
) -> dict:
    """Build an observation dict from real video frames.

    For 1 frame: each image key is (H, W, 3).
    For 4 frames: each image key is (4, H, W, 3).
    """
    obs: dict = {}
    for cam_key, all_frames in camera_frames.items():
        selected = all_frames[frame_indices]  # (T, H, W, 3)
        if len(frame_indices) == 1:
            selected = selected[0]  # (H, W, 3)
        obs[cam_key] = selected

    obs["observation/joint_position"] = np.zeros(7, dtype=np.float32)
    obs["observation/cartesian_position"] = np.zeros(6, dtype=np.float32)
    obs["observation/gripper_position"] = np.zeros(1, dtype=np.float32)
    obs["prompt"] = prompt
    obs["session_id"] = session_id
    return obs

def load_all_frames(video_path: str) -> np.ndarray:
    """Load all frames from a video file. Returns (N, H, W, 3) uint8 array (RGB)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames loaded from {video_path}")
    return np.stack(frames, axis=0)

def load_camera_frames() -> dict[str, np.ndarray]:
    """Load all video frames for each camera from the debug_image/ directory.

    Returns:
        Dict mapping roboarena camera keys to (N, H, W, 3) uint8 arrays.
    """
    camera_frames: dict[str, np.ndarray] = {}
    for cam_key, fname in CAMERA_FILES.items():
        path = os.path.join(VIDEO_DIR, fname)
        camera_frames[cam_key] = load_all_frames(path)
        logging.info(f"Loaded {cam_key}: {camera_frames[cam_key].shape}")
    return camera_frames

def build_frame_schedule(total_frames: int, num_chunks: int) -> list[list[int]]:
    """Build the frame index schedule for multi-frame chunks.

    Returns a list of frame-index lists. Each inner list has 4 indices.
    """
    chunks: list[list[int]] = []
    current_frame = 23  # first anchor frame
    for _ in range(num_chunks):
        indices = [max(current_frame + off, 0) for off in RELATIVE_OFFSETS]
        if indices[-1] >= total_frames:
            logging.info(
                f"Frame {indices[-1]} >= {total_frames}, stopping at {len(chunks)} chunks"
            )
            break
        chunks.append(indices)
        current_frame += ACTION_HORIZON
    return chunks

def test_policy(
    num_chunks: int, 
    prompt: str, 
    use_zero_images: bool,
    video_output_dir: str
):
    """Test the DreamZero policy server with rlinf interface.

    When use_zero_images is False (default), loads real video frames from
    ./debug_image/ 
    """
    from toolkits.eval_script_dreamzero import setup_policy
    policy = setup_policy("/mnt/project_rlinf/liyanghao/data/models/DreamZero-DROID", video_output_dir=video_output_dir)

    # prepare obs
    image_config = PolicyImageConfig(
        image_resolution=(180, 320),  # DreamZero expects 180x320 images
        needs_wrist_camera=True,
        n_external_cameras=2,
        needs_stereo_camera=False,
        needs_session_id=True,  # Track session to reset state for new clients
        action_space="joint_position",
    )
    logging.info(f"Image config: {image_config}")

    assert image_config.n_external_cameras == 2, f"Expected 2 external cameras, got {image_config.n_external_cameras}"
    assert image_config.needs_wrist_camera, "Expected wrist camera to be enabled"
    assert image_config.action_space == "joint_position", f"Expected joint_position action space, got {image_config.action_space}"

    import uuid
    session_id = str(uuid.uuid4())
    logging.info(f"Starting new session with ID: {session_id}")

    if use_zero_images:
        logging.info("Using ZERO images")
        for i in range(num_chunks):
            obs = _make_zero_observation(image_config, prompt, session_id)
            #logging.info(f"Observation {i}: {obs}")
            logging.info(f"Inference {i + 1}/{num_chunks}:prompt={prompt}")
            t0 = time.time()
            action = policy.infer(obs)
            dt = time.time() - t0
            _log_action(action, dt)
        return
    
    # real video frames
    logging.info("Using real video frames")
    camera_frames = load_camera_frames()

    total_frames = min(v.shape[0] for v in camera_frames.values())
    logging.info(f"Total frames available: {total_frames}")

    # Build frame schedule
    chunks = build_frame_schedule(total_frames, num_chunks)

    logging.info(f"Frame schedule:")
    logging.info("Initial : [0]")
    for i, indices in enumerate(chunks):
        logging.info(f"  Chunk {i}: {indices}")
    
    # Step 0: initial single frame
    logging.info("=== Initial: frame [0] ===")
    obs = _make_obs_from_video(camera_frames, [0], prompt, session_id)
    t0 = time.time()
    actions = policy.infer(obs)
    dt = time.time() - t0
    _log_action(actions, dt)

    # Subsequent chunks: send 4 frames at a time
    for chunk_idx, frame_indices in enumerate(chunks):
        logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} ===")
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
        t0 = time.time()
        actions = policy.infer(obs)
        dt = time.time() - t0
        _log_action(actions, dt)
    policy._reset_state(save_video=video_output_dir is not None)
    logging.info("Done.")

            

def _log_action(actions: np.ndarray, dt: float) -> None:
    """Pretty-print action shape, range, and timing."""
    assert isinstance(actions, np.ndarray), f"Expected numpy array, got {type(actions)}"
    assert actions.ndim == 2, f"Expected 2D array, got shape {actions.shape}"
    assert actions.shape[-1] == 8, (
        f"Expected 8 action dims (7 joints + 1 gripper), got {actions.shape[-1]}"
    )
    logging.info(
        f"  Action shape: {actions.shape}, "
        f"range: [{actions.min():.4f}, {actions.max():.4f}], "
        f"time: {dt:.2f}s"
    )
def main():
    parser = argparse.ArgumentParser(
        description="Test DreamZero policy server with real video frames from ./debug_image/"
    )
    parser.add_argument(
        "--num-chunks",
        type=int,
        default=15,
        help="Number of 4-frame chunks to send after the initial frame (default: 15)",
    )
    parser.add_argument(
        "--prompt",
        default="Move the pan forward and use the brush in the middle of the plates to brush the inside of the pan",
        help="Language prompt for the policy",
    )
    parser.add_argument(
        "--use-zero-images",
        action="store_true",
        help="Use zero dummy images instead of real video frames (legacy mode)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(f"Starting DreamZero policy server evaluation with args: {args}")

    test_policy(
        num_chunks=args.num_chunks,
        prompt=args.prompt,
        use_zero_images=args.use_zero_images,
        video_output_dir=VIDEO_OUTPUT_DIR
    )

if __name__ == "__main__":
    main()