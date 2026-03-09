"""
Video Data Saver for YAM robot teleoperation.
Saves camera feeds as video files without robot state data.
"""

import logging
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class VideoDataSaver:
    """
    Saves camera feeds as video files.
    Only records camera videos, no robot state or action data.
    
    Directory structure:
        output/
            {task_name}/
                {timestamp}/
                    left_camera.mp4
                    front_camera.mp4
    
    Usage:
        saver = VideoDataSaver(
            task_name="pick_and_place",
            save_dir="./output",
            fps=30,
            camera_names=["left_camera", "front_camera"]
        )
        
        # During data collection
        saver.add_frame(observation)
        
        # When episode ends
        saver.save_episode()
    """
    
    def __init__(
        self,
        task_name: str,
        save_dir: str = "./output",
        fps: int = 30,
        camera_names: Optional[list[str]] = None,
    ):
        """
        Initialize VideoDataSaver.
        
        Args:
            task_name: Task name (used as directory name)
            save_dir: Base output directory
            fps: Video frame rate
            camera_names: List of camera names to save
        """
        self.task_name = task_name
        self.save_dir = Path(save_dir)
        self.fps = fps
        self.camera_names = camera_names or []
        
        # Episode state
        self.episode_started = False
        self.video_writers: Dict[str, cv2.VideoWriter] = {}
        self.episode_dir: Optional[Path] = None
        self.frame_count = 0
        
        logger.info(f"VideoDataSaver initialized:")
        logger.info(f"  Task: {task_name}")
        logger.info(f"  Output: {save_dir}")
        logger.info(f"  FPS: {fps}")
        logger.info(f"  Cameras: {camera_names}")
    
    def _start_episode(self) -> None:
        """Start a new episode - create directory."""
        # Create episode directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.episode_dir = self.save_dir / self.task_name / timestamp
        self.episode_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting new episode: {self.episode_dir}")
        self.episode_started = True
        self.frame_count = 0
    
    def _ensure_video_writers(self, observation: Dict) -> None:
        """Create video writers based on first frame size."""
        if self.video_writers:
            return  # Already created
        
        for camera_name in self.camera_names:
            camera_key = f"{camera_name}_image"
            if camera_key in observation:
                image = observation[camera_key]
                
                if isinstance(image, np.ndarray):
                    height, width = image.shape[:2]
                    
                    # Create video writer
                    output_path = self.episode_dir / f"{camera_name}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(
                        str(output_path),
                        fourcc,
                        self.fps,
                        (width, height)
                    )
                    
                    if writer.isOpened():
                        self.video_writers[camera_name] = writer
                        logger.info(f"  Created video writer: {camera_name} ({width}x{height})")
                    else:
                        logger.error(f"  Failed to create video writer: {camera_name}")
    
    def add_frame(self, observation: Dict) -> None:
        """
        Add a frame to the current episode.
        
        Args:
            observation: Dictionary containing camera observations
        """
        # Start episode on first frame
        if not self.episode_started:
            self._start_episode()
        
        # Ensure video writers are created
        self._ensure_video_writers(observation)
        
        # Write camera frames
        for camera_name in self.camera_names:
            camera_key = f"{camera_name}_image"
            if camera_key in observation:
                image = observation[camera_key]
                
                # Convert to BGR if needed (OpenCV uses BGR)
                if isinstance(image, np.ndarray):
                    if len(image.shape) == 3 and image.shape[2] == 3:
                        # Assume RGB, convert to BGR
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    else:
                        image_bgr = image
                    
                    # Write frame
                    if camera_name in self.video_writers:
                        self.video_writers[camera_name].write(image_bgr)
        
        self.frame_count += 1
    
    def save_episode(self) -> bool:
        """
        Finalize and save the current episode.
        
        Returns:
            True if episode was saved successfully
        """
        if not self.episode_started:
            logger.warning("No episode in progress")
            return False
        
        if self.frame_count == 0:
            logger.warning("No frames recorded, skipping save")
            self._cleanup_episode()
            return False
        
        # Release video writers
        logger.info(f"Saving episode with {self.frame_count} frames...")
        for camera_name, writer in self.video_writers.items():
            writer.release()
            output_path = self.episode_dir / f"{camera_name}.mp4"
            logger.info(f"  Saved {camera_name}: {output_path}")
        
        self._cleanup_episode()
        logger.info("Episode saved successfully!")
        return True
    
    def _cleanup_episode(self) -> None:
        """Reset episode state."""
        self.video_writers.clear()
        self.episode_dir = None
        self.episode_started = False
        self.frame_count = 0
    
    def finalize(self) -> None:
        """Clean up any remaining resources."""
        if self.episode_started:
            logger.warning("Finalizing with episode in progress - saving...")
            self.save_episode()
        
        logger.info("VideoDataSaver finalized")
