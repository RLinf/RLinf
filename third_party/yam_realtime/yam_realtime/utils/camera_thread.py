import threading
import queue
import logging

from yam_realtime.utils.data_saver import DataSaver

logger = logging.getLogger(__name__)


class EpisodeSaverThread(threading.Thread):
    """
    A background thread that listens for completed episodes and saves them asynchronously.
    """
    def __init__(self, data_saver: DataSaver):
        super().__init__()
        self.data_saver = data_saver
        self.episode_queue = queue.Queue()  # Queue to hold episodes to save
        self.daemon = True  # Ensure the thread exits when the main program ends

    def run(self):
        """
        Continuously listens for episodes to save.
        """
        while True:
            try:
                # Wait for a new episode to save
                episode_data = self.episode_queue.get(timeout=5)  # Wait for 5 seconds
                if episode_data is None:  # Exit signal
                    break
                self.data_saver.save_episode_json(episode_data, pickle_only=False)
                self.episode_queue.task_done()  # Mark the task as done
            except queue.Empty:
                continue

    def save_episode(self, episode_data):
        """
        Put episode data in the queue for background saving.
        """
        self.episode_queue.put(episode_data)

    def stop(self):
        """Signal to stop the background thread."""
        self.episode_queue.put(None)


class LeRobotSaverThread(threading.Thread):
    """
    A background thread for saving LeRobot episodes asynchronously.
    """
    def __init__(self, data_saver):
        super().__init__()
        self.data_saver = data_saver
        self.save_queue = queue.Queue()  # Queue to hold episode buffers to save
        self.daemon = True  # Ensure the thread exits when the main program ends
        self.episode_count = 0

    def run(self):
        """
        Continuously listens for episodes to save.
        """
        while True:
            try:
                # Wait for an episode buffer
                episode_buffer = self.save_queue.get(timeout=5)  # Wait for 5 seconds
                if episode_buffer is None:  # Exit signal
                    logger.info("Received stop signal, exiting saver thread")
                    break
                
                # Save the episode using the copied buffer
                self.episode_count += 1
                episode_idx = episode_buffer.get('episode_index', '?')
                logger.info(f"[Background Thread] Saving episode {self.episode_count} (episode_index={episode_idx})...")
                try:
                    # Save episode with the copied data
                    # This will increment dataset.meta.total_episodes and encode videos
                    self.data_saver.dataset.save_episode(episode_data=episode_buffer)
                    
                    # After successful save, delete the temporary images for this episode
                    # (they've been encoded into MP4 already)
                    import shutil
                    from pathlib import Path
                    images_dir = Path(self.data_saver.dataset.images_path)
                    for camera_key in self.data_saver.dataset.meta.video_keys:
                        episode_images_dir = images_dir / camera_key / f"episode-{episode_idx:06d}"
                        if episode_images_dir.exists():
                            try:
                                shutil.rmtree(episode_images_dir)
                                logger.debug(f"[Background Thread] Deleted {episode_images_dir}")
                            except Exception as e:
                                logger.warning(f"[Background Thread] Failed to delete {episode_images_dir}: {e}")
                    
                    logger.info(f"[Background Thread] Episode {self.episode_count} saved successfully")
                except Exception as e:
                    logger.error(f"[Background Thread] Error saving episode {self.episode_count}: {e}", exc_info=True)
                
                self.save_queue.task_done()  # Mark the task as done
            except queue.Empty:
                continue
        
        logger.info(f"[Background Thread] Exiting after saving {self.episode_count} episodes")

    def save_episode(self, episode_buffer):
        """
        Put episode buffer in the queue for background saving.
        Args:
            episode_buffer: The copied episode buffer to save
        """
        self.save_queue.put(episode_buffer)

    def stop(self):
        """Signal to stop the background thread."""
        self.save_queue.put(None)
