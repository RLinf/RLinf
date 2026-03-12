"""Visualize replay buffer trajectories with interactive navigation.

Usage:
    python toolkits/replay_buffer/visualize.py --replay_dir <path_to_replay_buffer_dir>

Example:
    python toolkits/replay_buffer/visualize.py --replay_dir logs/20260312-16:05:30-frankasim_sac_cnn_async/replay_buffer/rank_0
"""

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.backend_bases import KeyEvent
from matplotlib.widgets import TextBox

from rlinf.data.replay_buffer import TrajectoryReplayBuffer


class MultiTrajectoryVisualizer:
    """Interactive visualizer for replay buffer trajectories with lazy loading."""

    def __init__(self, replay_buffer_dir: str, camera_key: str = "main_images", 
                 save_image: bool = False, output_image_path: Optional[str] = None):
        """
        Initialize visualizer with lazy loading via TrajectoryReplayBuffer.

        Args:
            replay_buffer_dir: Directory containing replay buffer (metadata.json, trajectory_index.json, trajectory_*.pt)
            camera_key: Which camera to visualize ("main_images", "wrist_images", "extra_view_images")
            save_image: If True, save current view to image file after each update
            output_image_path: Path to save the image (default: replay_buffer_viz.png in current dir)
        """
        self.replay_dir = Path(replay_buffer_dir)
        self.camera_key = camera_key
        self.save_image = save_image
        self.output_image_path = output_image_path or "replay_buffer_viz.png"
        
        if not self.replay_dir.exists():
            raise ValueError(f"Replay buffer directory not found: {replay_buffer_dir}")
        
        self.buffer = TrajectoryReplayBuffer(
            auto_save=True,
            auto_save_path=str(self.replay_dir),
            enable_cache=True,
            cache_size=5,
        )
        self.buffer.load_checkpoint(str(self.replay_dir))
        
        self.traj_idx = 0
        self.step_idx = 0
        self.batch_idx = 0
        
        self.current_trajectory = None
        self.current_traj_id = None
        self.current_traj_shape = None
        
        self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.textbox_ax = plt.axes([0.35, 0.02, 0.3, 0.04])
        self.textbox = TextBox(self.textbox_ax, 'Jump to traj ID:', initial='')
        self.textbox.on_submit(self.jump_to_trajectory)
        
        print(f"\nLoaded replay buffer from {replay_buffer_dir}")
        print(f"Total trajectories: {self.buffer.size}")
        print(f"Total samples: {self.buffer.total_samples}")
        print(f"Camera view: {camera_key}")
        print(f"Trajectory IDs: {self.buffer._trajectory_id_list[:10]}{'...' if self.buffer.size > 10 else ''}")
        
        if self.save_image:
            print(f"Auto-saving visualization to: {self.output_image_path}")
        print()
        
        self._load_current_trajectory()
        self.update_display()
    
    def _load_current_trajectory(self):
        """Lazily load the current trajectory from disk."""
        traj_id = self.buffer._trajectory_id_list[self.traj_idx]
        traj_info = self.buffer._trajectory_index[traj_id]
        model_weights_id = traj_info["model_weights_id"]
        
        self.current_trajectory = self.buffer._load_trajectory(traj_id, model_weights_id)
        self.current_traj_id = traj_id
        self.current_traj_shape = traj_info["shape"]
        
        T, B = self.current_traj_shape[:2]
        print(f"Loaded trajectory {traj_id}: shape={self.current_traj_shape} (T={T}, B={B}, samples={T*B})")
    
    def _get_max_step_idx(self) -> int:
        """Get the maximum step index for current trajectory."""
        if self.current_traj_shape is None:
            return 0
        T, B = self.current_traj_shape[:2]
        return T - 1
    
    def on_key(self, event: KeyEvent):
        """Handle keyboard events."""
        if event.key == 'right' or event.key == 'n':
            self._next_step()
        elif event.key == 'left' or event.key == 'p':
            self._prev_step()
        elif event.key == 'up':
            self._next_trajectory()
        elif event.key == 'down':
            self._prev_trajectory()
        elif event.key == 'b':
            self._next_batch()
        elif event.key == 'v':
            self._prev_batch()
        elif event.key == 's':
            self._save_current_view()
            return
        elif event.key == 'q' or event.key == 'escape':
            plt.close()
            return
        elif event.key == 'home':
            self.step_idx = 0
            self.update_display()
        elif event.key == 'end':
            self.step_idx = self._get_max_step_idx()
            self.update_display()
        else:
            return
        
        self.update_display()
    
    def _next_step(self):
        """Go to next step, auto-advance to next trajectory if at end."""
        max_step = self._get_max_step_idx()
        
        if self.step_idx < max_step:
            self.step_idx += 1
        elif self.traj_idx < self.buffer.size - 1:
            self.traj_idx += 1
            self.step_idx = 0
            self.batch_idx = 0
            self._load_current_trajectory()
            next_traj_id = self.buffer._trajectory_id_list[self.traj_idx]
            print(f"→ Auto-switched to trajectory {next_traj_id}")
    
    def _prev_step(self):
        """Go to previous step, auto-rewind to previous trajectory if at start."""
        if self.step_idx > 0:
            self.step_idx -= 1
        elif self.traj_idx > 0:
            self.traj_idx -= 1
            self._load_current_trajectory()
            self.step_idx = self._get_max_step_idx()
            T, B = self.current_traj_shape[:2]
            self.batch_idx = B - 1
            prev_traj_id = self.buffer._trajectory_id_list[self.traj_idx]
            print(f"← Auto-switched to trajectory {prev_traj_id}")
    
    def _next_trajectory(self):
        """Jump to first step of next trajectory."""
        if self.traj_idx < self.buffer.size - 1:
            self.traj_idx += 1
            self.step_idx = 0
            self.batch_idx = 0
            self._load_current_trajectory()
            next_traj_id = self.buffer._trajectory_id_list[self.traj_idx]
            print(f"↑ Jumped to trajectory {next_traj_id}")
    
    def _prev_trajectory(self):
        """Jump to first step of previous trajectory."""
        if self.traj_idx > 0:
            self.traj_idx -= 1
            self.step_idx = 0
            self.batch_idx = 0
            self._load_current_trajectory()
            prev_traj_id = self.buffer._trajectory_id_list[self.traj_idx]
            print(f"↓ Jumped to trajectory {prev_traj_id}")
    
    def _next_batch(self):
        """Switch to next batch index within current trajectory."""
        if self.current_traj_shape is None:
            return
        T, B = self.current_traj_shape[:2]
        if self.batch_idx < B - 1:
            self.batch_idx += 1
            print(f"→ Batch {self.batch_idx}/{B-1}")
    
    def _prev_batch(self):
        """Switch to previous batch index within current trajectory."""
        if self.batch_idx > 0:
            self.batch_idx -= 1
            T, B = self.current_traj_shape[:2]
            print(f"← Batch {self.batch_idx}/{B-1}")
    
    def jump_to_trajectory(self, text: str):
        """Jump to trajectory by ID."""
        try:
            target_id = int(text)
            
            if target_id in self.buffer._trajectory_id_list:
                self.traj_idx = self.buffer._trajectory_id_list.index(target_id)
                self.step_idx = 0
                self.batch_idx = 0
                self._load_current_trajectory()
                print(f"Jumped to trajectory ID {target_id}")
                self.update_display()
            else:
                print(f"Trajectory ID {target_id} not found. Valid IDs: {self.buffer._trajectory_id_list[:10]}{'...' if self.buffer.size > 10 else ''}")
        except ValueError:
            print(f"Invalid input: '{text}'. Please enter a numeric trajectory ID.")
    
    def _save_current_view(self, filepath: Optional[str] = None):
        """Save the current visualization to an image file."""
        save_path = filepath or self.output_image_path
        self.fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved current view to: {save_path}")
    
    def _extract_image_from_obs(self, obs_dict: dict, step_idx: int, batch_idx: int) -> Optional[np.ndarray]:
        """Extract and format image from observation dict at given step and batch index."""
        if obs_dict is None:
            return None
        
        img = obs_dict.get(self.camera_key)
        if img is None:
            for fallback_key in ["main_images", "wrist_images", "extra_view_images"]:
                img = obs_dict.get(fallback_key)
                if img is not None:
                    if fallback_key != self.camera_key:
                        print(f"Warning: {self.camera_key} not found, using {fallback_key}")
                        self.camera_key = fallback_key
                    break
        
        if img is None:
            return None
        
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        
        if img.ndim == 5:
            img = img[step_idx, batch_idx]
        elif img.ndim == 4:
            img = img[step_idx]
        elif img.ndim == 3:
            pass
        else:
            print(f"Warning: Unexpected image shape: {img.shape}")
            return None
        
        if img.ndim == 3 and img.shape[0] in [1, 3, 4]:
            img = img.transpose(1, 2, 0)
        
        if img.dtype == np.float32 or img.dtype == np.float64:
            if img.max() <= 1.0 and img.max() > 0:
                img = (img * 255).astype(np.uint8)
            elif img.max() > 1.0:
                img = img.astype(np.uint8)
        
        if img.ndim == 3 and img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        return img
    
    def update_display(self):
        """Update the visualization with current trajectory step."""
        if self.current_trajectory is None:
            return
        
        T, B = self.current_traj_shape[:2]
        
        curr_obs = self.current_trajectory.curr_obs
        next_obs = self.current_trajectory.next_obs
        
        curr_img = self._extract_image_from_obs(curr_obs, self.step_idx, self.batch_idx)
        next_img = self._extract_image_from_obs(next_obs, self.step_idx, self.batch_idx)
        
        if curr_img is None or next_img is None:
            print("Warning: No image data found in observations")
            return
        
        self.axes[0].clear()
        self.axes[0].imshow(curr_img)
        self.axes[0].set_title('Current Observation')
        self.axes[0].axis('off')
        
        self.axes[1].clear()
        self.axes[1].imshow(next_img)
        self.axes[1].set_title('Next Observation')
        self.axes[1].axis('off')
        
        info_text = []
        if self.current_trajectory.actions is not None:
            action = self.current_trajectory.actions[self.step_idx, self.batch_idx]
            if torch.is_tensor(action):
                action = action.cpu().numpy()
            action_str = np.array2string(action, precision=3, suppress_small=True, separator=',')
            info_text.append(f"Action: {action_str}")
        
        if self.current_trajectory.rewards is not None:
            reward = self.current_trajectory.rewards[self.step_idx, self.batch_idx]
            if torch.is_tensor(reward):
                reward = reward.item()
            info_text.append(f"Reward: {reward:.4f}")
        
        if self.current_trajectory.dones is not None:
            done = self.current_trajectory.dones[self.step_idx, self.batch_idx]
            if torch.is_tensor(done):
                done = done.item()
            info_text.append(f"Done: {done}")
        
        title_lines = [
            f'Trajectory ID {self.current_traj_id} (#{self.traj_idx}/{self.buffer.size-1}) | '
            f'Step {self.step_idx}/{T-1} | Batch {self.batch_idx}/{B-1}',
            f'{" | ".join(info_text)}' if info_text else '',
            'Keys: ←→/n/p=step | ↑↓=traj | b/v=batch | s=save | Home/End | q=quit | Type traj ID in box'
        ]
        
        self.fig.suptitle('\n'.join(title_lines), fontsize=9)
        self.fig.canvas.draw()
        
        if self.save_image:
            self._save_current_view()
    
    def show(self):
        """Display the visualizer."""
        plt.show()


def main():
    """Main entry point for trajectory visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize replay buffer trajectories interactively",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Keyboard Controls:
  ← → (or p/n)  : Navigate to previous/next step (auto-switches trajectories at boundaries)
  ↑ ↓           : Jump to previous/next trajectory (resets to step 0)
  b / v         : Switch between batch indices (if B > 1)
  s             : Save current view to image file
  Home / End    : Jump to first/last step of current trajectory
  q / Esc       : Quit
  Type in box   : Enter trajectory ID to jump directly to that trajectory

Examples:
  # Interactive mode (requires X11/display)
  python toolkits/replay_buffer/visualize.py --replay_dir logs/my_run/replay_buffer/rank_0
  
  # SSH/headless mode: auto-save image on each navigation
  python toolkits/replay_buffer/visualize.py --replay_dir logs/my_run/replay_buffer/rank_0 --save_image --output viz.png
  
  # With specific camera
  python toolkits/replay_buffer/visualize.py --replay_dir logs/my_run/replay_buffer/rank_0 --camera wrist_images
        """
    )
    parser.add_argument(
        "--replay_dir",
        type=str,
        required=True,
        help="Path to replay buffer directory (containing metadata.json and trajectory files)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="main_images",
        choices=["main_images", "wrist_images", "extra_view_images"],
        help="Which camera view to visualize (default: main_images)",
    )
    parser.add_argument(
        "--save_image",
        action="store_true",
        help="Auto-save visualization to image file after each update (useful for SSH/headless)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="replay_buffer_viz.png",
        help="Output image path when --save_image is enabled (default: replay_buffer_viz.png)",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Don't show interactive window, only save images (requires --save_image)",
    )
    
    args = parser.parse_args()
    
    if args.no_display and not args.save_image:
        parser.error("--no_display requires --save_image to be enabled")
    
    if args.no_display:
        import matplotlib
        matplotlib.use('Agg')
    
    viz = MultiTrajectoryVisualizer(
        args.replay_dir, 
        camera_key=args.camera,
        save_image=args.save_image,
        output_image_path=args.output
    )
    
    if not args.no_display:
        viz.show()
    else:
        print(f"\nRunning in headless mode. Image saved to: {args.output}")
        print("To navigate, you can modify the script or use interactive mode on a machine with display.")


if __name__ == '__main__':
    main()