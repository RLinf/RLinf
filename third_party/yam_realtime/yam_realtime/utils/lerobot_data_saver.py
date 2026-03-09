"""
LeRobot Data Saver for YAM robot teleoperation data collection.
Saves data directly in LeRobot v3.0 dataset format (Parquet + MP4).
"""

import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from src.lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)


GITATTRIBUTES_TEMPLATE = """*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ckpt filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.lz4 filter=lfs diff=lfs merge=lfs -text
*.mlmodel filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
*.safetensors filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tar filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
# Audio files - uncompressed
*.pcm filter=lfs diff=lfs merge=lfs -text
*.sam filter=lfs diff=lfs merge=lfs -text
*.raw filter=lfs diff=lfs merge=lfs -text
# Audio files - compressed
*.aac filter=lfs diff=lfs merge=lfs -text
*.flac filter=lfs diff=lfs merge=lfs -text
*.mp3 filter=lfs diff=lfs merge=lfs -text
*.ogg filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
# Image files - uncompressed
*.bmp filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.tiff filter=lfs diff=lfs merge=lfs -text
# Image files - compressed
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.webp filter=lfs diff=lfs merge=lfs -text
# Video files - compressed
*.mp4 filter=lfs diff=lfs merge=lfs -text
*.webm filter=lfs diff=lfs merge=lfs -text
"""


def create_readme_with_info(repo_id: str, info_json: str) -> str:
    """Create README.md template with embedded info.json."""
    return f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{info_json}
```

## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```
"""


def create_readme_with_info(repo_id: str, info_json: str) -> str:
    """Create README.md template with embedded info.json."""
    return f"""---
license: apache-2.0
task_categories:
- robotics
tags:
- LeRobot
configs:
- config_name: default
  data_files: data/*/*.parquet
---

This dataset was created using [LeRobot](https://github.com/huggingface/lerobot).

## Dataset Description

- **Homepage:** [More Information Needed]
- **Paper:** [More Information Needed]
- **License:** apache-2.0

## Dataset Structure

[meta/info.json](meta/info.json):
```json
{info_json}
```

## Citation

**BibTeX:**

```bibtex
[More Information Needed]
```
"""


class LeRobotDataSaver:
    """
    Saves YAM teleoperation data directly in LeRobot v3.0 dataset format.
    
    Usage:
        saver = LeRobotDataSaver(
            repo_id="yam_dataset",
            root="./data/lerobot_datasets",
            fps=30,
            task_name="pick_and_place"
        )
        
        # During data collection
        saver.add_frame(observation, action)
        
        # When episode ends
        saver.save_episode()
        
        # When collection session ends
        saver.finalize()
    """
    
    def __init__(
        self,
        repo_id: str,
        root: str,
        fps: int = 30,
        task_name: str = "default_task",
        robot_type: str = "yam",
        camera_names: list[str] = None,
        camera_width: int = 640,
        camera_height: int = 480,
        use_videos: bool = True,
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
        hf_user: Optional[str] = None,
        auto_upload: bool = True,
        batch_encoding_size: int = None,
    ):
        """
        Initialize LeRobot data saver.
        
        Args:
            repo_id: Dataset identifier (e.g., "yam_pick_place" or "lerobot/task_name")
            root: Root directory for saving datasets. If None, uses HF_LEROBOT_HOME
            fps: Frames per second for data collection
            task_name: Task description
            robot_type: Type of robot (default: "yam")
            camera_names: List of camera names (e.g., ["camera_0", "camera_1"])
            use_videos: Whether to save videos (MP4) or individual images
            image_writer_processes: Number of processes for async image writing
            image_writer_threads: Number of threads for image writing
            hf_user: Hugging Face username (e.g., "username"). If None, uses local repo_id
            auto_upload: Whether to automatically upload to Hugging Face on finalize
            batch_encoding_size: Number of episodes to collect before encoding videos.
                If None, videos are encoded immediately after each episode (slower but immediate).
                If > 1, videos are encoded in batches (faster, no blocking during collection).
        """
        self.repo_id = repo_id
        
        # Handle root=None case: use HF_LEROBOT_HOME or default
        if root is None:
            from lerobot.utils.constants import HF_LEROBOT_HOME
            self.root = HF_LEROBOT_HOME
        else:
            self.root = Path(root)
        
        self.fps = fps
        self.task_name = task_name
        self.robot_type = robot_type
        self.camera_names = camera_names or []
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.hf_user = hf_user
        self.auto_upload = auto_upload
        
        # Full repo ID for Hugging Face (username/dataset_name)
        self.hf_repo_id = f"{hf_user}/{repo_id}" if hf_user else repo_id
        
        # Define features for YAM bimanual robot
        # observation.state: 14 DoF (7 per arm: 6 joints + 1 gripper)
        # action: 14 DoF (7 per arm: 6 joints + 1 gripper)
        features = self._create_features()
        
        # Check if local dataset exists with complete structure
        dataset_path = self.root / repo_id
        local_exists = (
            dataset_path.exists() and
            (dataset_path / "meta").exists() and
            (dataset_path / "data").exists()
        )
        
        # Create or load dataset
        if local_exists:
            try:
                logger.info(f"Loading existing local dataset: {repo_id}")
                # Load from local path without querying Hugging Face
                self.dataset = LeRobotDataset(repo_id, root=str(self.root), local_files_only=True)
                
                # CRITICAL: Set batch_encoding_size even for loaded datasets!
                # Otherwise it defaults to 1 and encodes immediately
                if batch_encoding_size and batch_encoding_size > 1:
                    self.dataset.batch_encoding_size = batch_encoding_size
                    logger.info(f"Set batch_encoding_size to {batch_encoding_size} for existing dataset")
                
                logger.info(f"Loaded existing dataset with {self.dataset.num_episodes} episodes")
            except Exception as e:
                logger.warning(f"Failed to load existing dataset: {e}")
                logger.info("Will create new dataset instead")
                local_exists = False
        
        if not local_exists:
            logger.info(f"Creating new LeRobot dataset: {repo_id}")
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=fps,
                root=str(self.root),
                robot_type=robot_type,
                features=features,
                use_videos=use_videos,
                image_writer_processes=image_writer_processes,
                image_writer_threads=image_writer_threads,
            )
            
            # CRITICAL: Set batch_encoding_size AFTER creating dataset
            # LeRobotDataset.create() doesn't accept this parameter
            if batch_encoding_size and batch_encoding_size > 1:
                self.dataset.batch_encoding_size = batch_encoding_size
                logger.info(f"Set batch_encoding_size to {batch_encoding_size}")
                logger.info(f"Video encoding will happen in batches of {batch_encoding_size} episodes")
            
            logger.info(f"LeRobot dataset created successfully")
        
        self.episode_started = False
        
    def _create_features(self) -> dict:
        """
        Create features dictionary for LeRobot dataset.
        
        Returns:
            Features dictionary defining data schema
        """
        features = {
            # Robot joint state (observation)
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),  # 7 joints per arm (left + right)
                "names": {
                    "motors": [
                        "left_joint_0", "left_joint_1", "left_joint_2", 
                        "left_joint_3", "left_joint_4", "left_joint_5", "left_gripper",
                        "right_joint_0", "right_joint_1", "right_joint_2",
                        "right_joint_3", "right_joint_4", "right_joint_5", "right_gripper"
                    ]
                },
            },
            # Robot action (target joint positions)
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": {
                    "motors": [
                        "left_joint_0", "left_joint_1", "left_joint_2",
                        "left_joint_3", "left_joint_4", "left_joint_5", "left_gripper",
                        "right_joint_0", "right_joint_1", "right_joint_2",
                        "right_joint_3", "right_joint_4", "right_joint_5", "right_gripper"
                    ]
                },
            },
        }
        
        # Add camera features
        for camera_name in self.camera_names:
            features[f"observation.images.{camera_name}"] = {
                "dtype": "video",
                "shape": (self.camera_height, self.camera_width, 3),  # Height, width, channels
                "names": ["height", "width", "channel"],
            }
        
        return features
    
    def add_frame(
        self,
        observation: Dict[str, Any],
        action: Dict[str, Any],
        timestamp: Optional[float] = None,
    ) -> None:
        """
        Add a single frame to the current episode buffer.
        
        Args:
            observation: Dictionary containing:
                - "left": {"joint": array(7,), "gripper_pos": float}
                - "right": {"joint": array(7,), "gripper_pos": float}
                - "images": {"camera_name": array(H, W, 3)}
            action: Dictionary containing:
                - "left": {"pos": array(7,)}  # target joint positions
                - "right": {"pos": array(7,)}
            timestamp: Optional timestamp (auto-generated if None)
        """
        # Concatenate left and right joint states (14,)
        # observation structure: {"left": {"joint_pos": (6,), "gripper_pos": (1,)}, "right": {...}}
        left_joint = observation["left"]["joint_pos"]  # (6,)
        left_gripper = observation["left"]["gripper_pos"]  # (1,)
        left_state = np.concatenate([left_joint, left_gripper]).astype(np.float32)  # (7,)
        
        right_joint = observation["right"]["joint_pos"]  # (6,)
        right_gripper = observation["right"]["gripper_pos"]  # (1,)
        right_state = np.concatenate([right_joint, right_gripper]).astype(np.float32)  # (7,)
        
        obs_state = np.concatenate([left_state, right_state]).astype(np.float32)  # (14,)
        
        # Concatenate left and right actions (14,)
        left_action = action["left"]["pos"]  # (7,)
        right_action = action["right"]["pos"]  # (7,)
        action_array = np.concatenate([left_action, right_action]).astype(np.float32)
        
        # Prepare frame data
        frame_data = {
            "observation.state": obs_state,
            "action": action_array,
            "task": self.task_name,
        }
        
        # Add camera images from observation
        for camera_name in self.camera_names:
            if camera_name in observation:
                camera_data = observation[camera_name]
                # Handle different camera data formats
                if isinstance(camera_data, dict):
                    # Camera data structure: {'images': {...}, 'timestamp': float}
                    if 'images' in camera_data:
                        images_data = camera_data['images']
                        if isinstance(images_data, dict):
                            # Try common keys
                            if 'color' in images_data:
                                image = images_data['color']
                            elif 'rgb' in images_data:
                                image = images_data['rgb']
                            elif 'image' in images_data:
                                image = images_data['image']
                            else:
                                logger.error(f"Camera {camera_name} 'images' dict has no recognized key: {list(images_data.keys())}")
                                continue
                        else:
                            image = images_data
                    elif 'image' in camera_data:
                        image = camera_data['image']
                    elif 'color' in camera_data:
                        image = camera_data['color']
                    elif 'rgb' in camera_data:
                        image = camera_data['rgb']
                    else:
                        logger.error(f"Camera {camera_name} dict has no recognized image key: {list(camera_data.keys())}")
                        continue
                elif isinstance(camera_data, np.ndarray):
                    image = camera_data
                else:
                    logger.warning(f"Unknown camera data format for {camera_name}: {type(camera_data)}")
                    continue
                frame_data[f"observation.images.{camera_name}"] = image
            else:
                logger.error(f"Camera {camera_name} not found in observation!")
                logger.error(f"Available keys: {list(observation.keys())}")
                raise KeyError(f"Camera {camera_name} not found in observation")
        
        # Add frame to episode buffer
        # Note: LeRobotDataset.add_frame() will auto-create episode_buffer if it's None
        self.dataset.add_frame(frame_data)
        self.episode_started = True
    
    def get_episode_buffer_copy(self):
        """
        Get a deep copy of the current episode buffer for background saving.
        
        Returns:
            A deep copy of the episode buffer, or None if no episode started.
        """
        import copy
        if not self.episode_started:
            return None
        
        if self.dataset.episode_buffer is None:
            return None
            
        # Deep copy the episode buffer to avoid conflicts
        episode_buffer_copy = {}
        for key, value in self.dataset.episode_buffer.items():
            if isinstance(value, list):
                # Deep copy lists
                episode_buffer_copy[key] = copy.deepcopy(value)
            else:
                # Copy other values (like episode_index, size, etc.)
                episode_buffer_copy[key] = value
        
        logger.info(f"Episode buffer copied successfully (episode_index={episode_buffer_copy.get('episode_index')}, size={episode_buffer_copy.get('size')})")
        return episode_buffer_copy
    
    def save_episode(self) -> None:
        """
        Save the current episode to disk.
        This encodes videos (if use_videos=True) and writes parquet files.
        """
        if not self.episode_started:
            logger.warning("No frames in episode buffer, skipping save_episode")
            return
        
        logger.info(f"Saving episode {self.dataset.num_episodes}...")
        self.dataset.save_episode()
        self.episode_started = False
        logger.info(f"Episode saved. Total episodes: {self.dataset.num_episodes}")
    
    def finalize(self) -> None:
        """
        Finalize the dataset by writing metadata files (info.json, stats.json, etc.).
        Creates .gitattributes and README.md, then uploads to Hugging Face if enabled.
        """
        print("[finalize] Starting finalization...")
        logger.info("Finalizing LeRobot dataset...")
        
        try:
            # Encode any remaining episodes if using batch encoding
            if hasattr(self.dataset, 'episodes_since_last_encoding') and self.dataset.episodes_since_last_encoding > 0:
                print(f"[finalize] Encoding remaining {self.dataset.episodes_since_last_encoding} episodes...")
                logger.info(f"Encoding remaining {self.dataset.episodes_since_last_encoding} episodes...")
                start_ep = self.dataset.num_episodes - self.dataset.episodes_since_last_encoding
                end_ep = self.dataset.num_episodes
                print(f"[finalize] Batch encoding episodes {start_ep} to {end_ep - 1}")
                logger.info(f"Batch encoding episodes {start_ep} to {end_ep - 1}")
                self.dataset._batch_save_episode_video(start_ep, end_ep)
                print("[finalize] Batch encoding done")
            
            print("[finalize] Stopping image writer...")
            logger.info("Stopping image writer...")
            self.dataset.stop_image_writer()
            print("[finalize] Image writer stopped")
            
            print("[finalize] Calling dataset.finalize()...")
            logger.info("Calling dataset.finalize()...")
            self.dataset.finalize()
            print("[finalize] Dataset finalize() completed")
            logger.info("Dataset finalize() completed")
            
            # Clean up empty images directory if it exists
            import shutil
            from pathlib import Path
            logger.info("Cleaning up images directory...")
            img_dir = Path(self.dataset.root) / "images"
            if img_dir.exists():
                # Check for any remaining PNG files
                png_files = list(img_dir.rglob("*.png"))
                if len(png_files) == 0:
                    logger.info("Removing empty images directory...")
                    shutil.rmtree(img_dir)
                else:
                    logger.warning(f"Images directory contains {len(png_files)} PNG files, not removing")
        except Exception as e:
            logger.error(f"Error during dataset finalization: {e}")
            import traceback
            traceback.print_exc()
        
        # Create .gitattributes and README.md in dataset root
        # Use dataset.root which is the actual dataset directory
        dataset_root = Path(self.dataset.root)
        try:
            logger.info("Creating dataset files (.gitattributes, README.md)...")
            self._create_dataset_files(dataset_root)
            logger.info("Dataset files created successfully")
        except Exception as e:
            logger.warning(f"Error creating dataset files (.gitattributes, README.md): {e}")
        except Exception as e:
            logger.warning(f"Error creating dataset files (.gitattributes, README.md): {e}")
        
        logger.info(f"Dataset finalized:")
        logger.info(f"  Total episodes: {self.dataset.num_episodes}")
        logger.info(f"  Total frames: {len(self.dataset)}")
        logger.info(f"  Location: {dataset_root}")
        
        # Upload to Hugging Face if enabled
        if self.auto_upload:
            logger.info("Auto-upload is enabled, uploading to Hugging Face...")
            self._upload_to_hub(dataset_root)
        else:
            logger.info("Auto-upload is disabled, skipping Hugging Face upload")
    
    def _create_dataset_files(self, dataset_root: Path) -> None:
        """Create .gitattributes and README.md files."""
        # Create .gitattributes
        gitattributes_path = dataset_root / ".gitattributes"
        with open(gitattributes_path, 'w') as f:
            f.write(GITATTRIBUTES_TEMPLATE)
        
        # Read info.json to embed in README
        import json
        info_json_path = dataset_root / "meta" / "info.json"
        info_json_str = ""
        if info_json_path.exists():
            with open(info_json_path, 'r') as f:
                info_data = json.load(f)
                info_json_str = json.dumps(info_data, indent=4)
        
        # Create README.md with embedded info.json
        readme_content = create_readme_with_info(
            repo_id=self.hf_repo_id,
            info_json=info_json_str,
        )
        readme_path = dataset_root / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
    
    def _upload_to_hub(self, dataset_root: Path) -> None:
        """Upload dataset to Hugging Face Hub."""
        api = None
        try:
            from huggingface_hub import HfApi, create_repo
            import urllib3
            
            # Ensure dataset_root is absolute path and exists
            dataset_root = dataset_root.resolve()
            if not dataset_root.exists():
                logger.error(f"Dataset directory does not exist: {dataset_root}")
                return
            
            logger.info(f"Uploading dataset to Hugging Face: {self.hf_repo_id}")
            
            # Create repository if it doesn't exist
            api = HfApi()
            try:
                create_repo(
                    repo_id=self.hf_repo_id,
                    repo_type="dataset",
                    exist_ok=True,
                    private=False,
                )
            except Exception:
                pass  # Repository may already exist
            
            # Upload all files
            logger.info("Uploading files (this may take a while for large datasets)...")
            api.upload_folder(
                folder_path=str(dataset_root),
                repo_id=self.hf_repo_id,
                repo_type="dataset",
                commit_message=f"Upload {self.dataset.num_episodes} episodes ({len(self.dataset)} frames)",
            )
            
            logger.info(f"✓ Dataset uploaded successfully!")
            logger.info(f"  View at: https://huggingface.co/datasets/{self.hf_repo_id}")
            
        except ImportError:
            logger.error("huggingface_hub not installed. Install with: pip install huggingface_hub")
        except Exception as e:
            logger.error(f"Failed to upload to Hugging Face: {e}")
            logger.info(f"You can manually upload later using:")
            logger.info(f"  huggingface-cli upload {self.hf_repo_id} {dataset_root} . --repo-type=dataset")
        finally:
            # Aggressively clean up HTTP connections to prevent "Bad file descriptor" on exit
            try:
                # Close the HfApi session if it exists
                if api is not None and hasattr(api, '_session'):
                    api._session.close()
                
                # Disable urllib3 connection pool cleanup to prevent Bad file descriptor
                import urllib3.connectionpool
                import warnings
                
                # Clear all connection pools
                try:
                    urllib3.connectionpool.clear()
                except AttributeError:
                    pass
                
                # Monkey-patch connectionpool to prevent cleanup on exit
                # This prevents "Bad file descriptor" errors when Python exits
                original_close = urllib3.connectionpool.HTTPConnectionPool.close
                def silent_close(self):
                    try:
                        original_close(self)
                    except OSError:
                        pass  # Silently ignore "Bad file descriptor" errors
                urllib3.connectionpool.HTTPConnectionPool.close = silent_close
                
                # Also suppress SSL socket close errors
                import ssl
                original_ssl_close = ssl.SSLSocket._real_close
                def silent_ssl_close(self):
                    try:
                        original_ssl_close(self)
                    except OSError:
                        pass
                ssl.SSLSocket._real_close = silent_ssl_close
                
                # Force garbage collection
                import gc
                gc.collect()
                
            except Exception:
                pass
    
    def get_num_episodes(self) -> int:
        """Get number of episodes saved."""
        return self.dataset.num_episodes
    
    def get_num_frames(self) -> int:
        """Get total number of frames saved."""
        return len(self.dataset) if hasattr(self.dataset, '__len__') else 0
