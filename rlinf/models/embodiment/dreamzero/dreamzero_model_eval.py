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
from tianshou.data import Batch

import cv2
import numpy as np

from dataclasses import dataclass, field
from typing import Tuple

from hydra.utils import instantiate
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

BACKBONE_FEATURE_KEY = "backbone_features"
ACTION_KEY = "action_pred"
LOSS_KEY = "loss"
ERROR_MSG = "Error: unexpected input/output"
N_COLOR_CHANNELS = 3

VIDEO_DIR = os.path.join(os.path.dirname(__file__), "debug_image")
VIDEO_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "video_output")
MODEL_PATH = "/mnt/project_rlinf/liyanghao/data/models/DreamZero-DROID"

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

class DreamZeroModel:
    FRAMES_PER_CHUNK = 4

    def __init__(
        self,
        model_path: str,
        embodiment_tag: str = "oxe_droid",
        device: str | int = "cuda",
        num_action_chunks: int = 24,
        action_dim: int = 8,
        add_value_head: bool = False,
        train_backbone: bool = False,
        video_output_dir: str = VIDEO_OUTPUT_DIR,
    ):
        from groot.vla.data.schema import EmbodimentTag
        from rlinf.models.embodiment.dreamzero.sim_policy import GrootSimPolicy

        self.model_path = model_path
        self.embodiment_tag = EmbodimentTag(embodiment_tag)
        self.device = device
        self.num_action_chunks = num_action_chunks
        self.action_dim = action_dim
        self.add_value_head = add_value_head
        self.train_backbone = train_backbone
        self.video_output_dir = video_output_dir
        self._frame_buffers: dict[str, list[np.ndarray]] = {
            "video.exterior_image_1_left": [],
            "video.exterior_image_2_left": [],
            "video.wrist_image_left": []
        }
        self._is_first_call = True
        self._call_count = 0
        self.video_across_time = []
        self._groot_policy = GrootSimPolicy(
            embodiment_tag=self.embodiment_tag,
            model_path=model_path,
            device=device,
            lazy_load=False,
            enable_grad=train_backbone,
        )
        self._groot_policy.eval()

    def _convert_observation(self, obs: dict) -> dict:
        """Convert roboarena observation format to AR_droid format.
        
        Roboarena format:
            - observation/exterior_image_0_left: (H, W, 3) single frame
            - observation/exterior_image_1_left: (H, W, 3) single frame
            - observation/wrist_image_left: (H, W, 3) single frame
            - observation/joint_position: (7,)
            - observation/gripper_position: (1,)
            - prompt: str
        
        AR_droid format:
            - video.exterior_image_1_left: (T, H, W, 3) multi-frame
            - video.exterior_image_2_left: (T, H, W, 3) multi-frame
            - video.wrist_image_left: (T, H, W, 3) multi-frame
            - state.joint_position: (1, 7)
            - state.gripper_position: (1, 1)
            - annotation.language.action_text: str
        """
        converted = {}
        
        # Map image keys (roboarena uses 0-indexed, AR_droid uses 1-indexed)
        image_key_mapping = {
            "observation/exterior_image_0_left": "video.exterior_image_1_left",
            "observation/exterior_image_1_left": "video.exterior_image_2_left",
            "observation/wrist_image_left": "video.wrist_image_left",
        }
        
        # Accumulate frames for each camera view
        for roboarena_key, droid_key in image_key_mapping.items():
            if roboarena_key in obs:
                data = obs[roboarena_key]
                if isinstance(data, np.ndarray):
                    if data.ndim == 4:
                        # Multiple frames (T, H, W, 3)
                        self._frame_buffers[droid_key].extend(list(data))
                    else:
                        # Single frame (H, W, 3)
                        self._frame_buffers[droid_key].append(data)

        # Determine how many frames to use
        if self._is_first_call:
            # First call: use only 1 frame
            num_frames = 1
        else:
            # Subsequent calls: use exactly FRAMES_PER_CHUNK frames
            num_frames = self.FRAMES_PER_CHUNK
        
        # Build video tensors from accumulated frames
        for droid_key, buffer in self._frame_buffers.items():
            if len(buffer) > 0:
                if len(buffer) >= num_frames:
                    # Take the last num_frames frames
                    frames_to_use = buffer[-num_frames:]
                else:
                    # Pad by repeating the first frame to reach num_frames
                    frames_to_use = buffer.copy()
                    while len(frames_to_use) < num_frames:
                        # Prepend the first frame to pad
                        frames_to_use.insert(0, buffer[0])
                # Stack to (T, H, W, C)
                video = np.stack(frames_to_use, axis=0)
                converted[droid_key] = video
        
        # Convert state observations
        if "observation/joint_position" in obs:
            joint_pos = obs["observation/joint_position"]
            # Reshape to (1, 7) if needed
            if joint_pos.ndim == 1:
                joint_pos = joint_pos.reshape(1, -1)
            converted["state.joint_position"] = joint_pos.astype(np.float64)
        else:
            converted["state.joint_position"] = np.zeros((1, 7), dtype=np.float64)
        
        if "observation/gripper_position" in obs:
            gripper_pos = obs["observation/gripper_position"]
            # Reshape to (1, 1) if needed
            if gripper_pos.ndim == 1:
                gripper_pos = gripper_pos.reshape(1, -1)
            converted["state.gripper_position"] = gripper_pos.astype(np.float64)
        else:
            converted["state.gripper_position"] = np.zeros((1, 1), dtype=np.float64)
        
        # Convert prompt
        if "prompt" in obs:
            converted["annotation.language.action_text"] = obs["prompt"]
        else:
            converted["annotation.language.action_text"] = ""
        
        return converted

    def _convert_action(self, action_dict: dict) -> np.ndarray:
        """Convert AR_droid action dict to roboarena action array.
        
        AR_droid format:
            - action.joint_position: (N, 7)
            - action.gripper_position: (N,) or (N, 1)
        
        Roboarena format:
            - action: (N, 8) - 7 joint positions + 1 gripper
        """
        joint_action = None
        gripper_action = None
        
        # Extract actions from dict
        for key, value in action_dict.items():
            if "joint_position" in key:
                joint_action = value
            elif "gripper_position" in key or "gripper" in key:
                gripper_action = value
        
        if joint_action is None:
            # Fallback: return zeros
            return np.zeros((1, 8), dtype=np.float32)
        
        # Convert to numpy if tensor
        if isinstance(joint_action, torch.Tensor):
            joint_action = joint_action.cpu().numpy()
        
        # Ensure 2D shape (N, 7)
        if joint_action.ndim == 1:
            joint_action = joint_action.reshape(1, -1)
        
        N = joint_action.shape[0]
        
        # Handle gripper action
        if gripper_action is not None:
            if isinstance(gripper_action, torch.Tensor):
                gripper_action = gripper_action.cpu().numpy()
            # Reshape to (N, 1) if needed
            if gripper_action.ndim == 1:
                gripper_action = gripper_action.reshape(-1, 1)
            elif gripper_action.ndim == 0:
                gripper_action = gripper_action.reshape(1, 1)
        else:
            gripper_action = np.zeros((N, 1), dtype=np.float32)
        
        # Concatenate: (N, 7) + (N, 1) -> (N, 8)
        action = np.concatenate([joint_action, gripper_action], axis=-1).astype(np.float32)
        
        return action

    def save_video(self, save_video: bool = False) -> None:
            """Internal method to save video.
            
            Args:
                save_video: Whether to save accumulated video before reset.
            """
            # Optionally save accumulated video before reset
            if save_video and len(self.video_across_time) > 0 and self.video_output_dir:
                try:
                    frame_list = []
                    video_across_time_cat = torch.cat(self.video_across_time, dim=2)
                    frames = self._groot_policy.trained_model.action_head.vae.decode(
                        video_across_time_cat,
                        tiled=self._groot_policy.trained_model.action_head.tiled,
                        tile_size=(self._groot_policy.trained_model.action_head.tile_size_height, self._groot_policy.trained_model.action_head.tile_size_width),
                        tile_stride=(self._groot_policy.trained_model.action_head.tile_stride_height, self._groot_policy.trained_model.action_head.tile_stride_width),
                    )
                    frames = rearrange(frames, "B C T H W -> B T H W C")
                    frames = frames[0]
                    frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
                    for frame in frames:
                        frame_list.append(frame)
                    
                    if len(frame_list) > 0:
                        sample_frame = frame_list[0]
                        if len(sample_frame.shape) == 3 and sample_frame.shape[2] in [1, 3, 4]:
                            save_dir = self.video_output_dir
                            os.makedirs(save_dir, exist_ok=True)
                            all_mp4_files = [f for f in os.listdir(save_dir) if f.endswith(".mp4")]
                            timestamp = datetime.datetime.now().strftime("%m_%d_%H_%M_%S")
                            num_frames = len(frame_list)
                            n = (num_frames - 1) // 8
                            output_path = os.path.join(save_dir, f'{len(all_mp4_files):06}_{timestamp}_n{n}.mp4')
                            imageio.mimsave(output_path, frame_list, fps=5, codec='libx264')
                            print(f"Saved video on reset to: {output_path}")
                except Exception as e:
                    print(f"Failed to save video on reset: {e}")
            
            # Clear frame buffers
            for key in self._frame_buffers:
                self._frame_buffers[key] = []
            
            self._call_count = 0
            self._is_first_call = True
            self.video_across_time = []

    def eval(
        self,
        obs: dict,
    ) -> np.ndarray:
        converted_obs = self._convert_observation(obs)
        batch = Batch(obs=converted_obs)
        with torch.no_grad():
            result_batch, video_pred = self._groot_policy.lazy_joint_forward_causal(batch)
        self.video_across_time.append(video_pred)
        action_chunk_dict = result_batch.act
        action_dict = {}
        for k in dir(action_chunk_dict):
            if k.startswith("action."):
                action_dict[k] = getattr(action_chunk_dict, k)
        action = self._convert_action(action_dict)
        return action


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
        description="Test DreamZero model with real video frames from ./debug_image/"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=MODEL_PATH,
        help="Path to the DreamZero model",
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
    use_zero_images = args.use_zero_images
    prompt = args.prompt
    num_chunks = args.num_chunks
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logging.info(f"Starting DreamZero policy server evaluation with args: {args}")

    dreamzero_model = DreamZeroModel(
        model_path=args.model_path
    )

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
            t0 = time.time()
            action = dreamzero_model.eval(obs)
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
    actions = dreamzero_model.eval(obs)
    dt = time.time() - t0
    _log_action(actions, dt)

    # Subsequent chunks: send 4 frames at a time
    for chunk_idx, frame_indices in enumerate(chunks):
        logging.info(f"=== Chunk {chunk_idx}: frames {frame_indices} ===")
        obs = _make_obs_from_video(camera_frames, frame_indices, prompt, session_id)
        t0 = time.time()
        actions = dreamzero_model.eval(obs)
        dt = time.time() - t0
        _log_action(actions, dt)
    dreamzero_model.save_video(save_video=True)
    logging.info("Done.")
        


if __name__ == "__main__":
    main()