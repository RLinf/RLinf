import concurrent
import json
import logging
import os
from re import L
import shutil
import time
from PIL import Image
import numpy as np
import warnings

# Suppress numpy warnings for empty slices
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered')

logger = logging.getLogger("data_saver")
logger.setLevel(logging.INFO)


class DataSaver:
    def __init__(self, save_dir="/home/sean/Desktop/YAM/yam_realtime/yam_realtime/scripts/delta_trajectory/", task_directory="Testing_dir", language_instruction="Test"):
        self.save_dir = os.path.join(
            save_dir,
            task_directory
        )

        # self.task = task
        self.traj_count = 1 # number of actions saved
        self.buffer = [] # buffer for a single action
        self.instruction = language_instruction

        if os.path.exists(self.save_dir):
            remove_dir = input(f"The directory {self.save_dir} already exists. Do you want to remove it? (y/n): ")
            if remove_dir == "y":
                shutil.rmtree(self.save_dir)
                logger.info(f"Removed existing directory: {self.save_dir}.")
            else:
                raise FileExistsError(f"The directory {self.save_dir} already exists.")

        os.makedirs(self.save_dir, exist_ok=True)
    
    def reset_buffer(self):
        old_size = len(self.buffer)
        self.buffer = []
        logger.info(f"Reset buffer: {old_size} observations cleared.")
        
    def add_observation(self, obs, action):
        obs['action'] = {'left': action['left']['pos'], 'right': action['right']['pos']}
        obs['instruction'] = self.instruction
        obs['delta'] = {'left': np.concatenate([action['left']['delta'], [action['left']['pos'][-1]]]), 'right': np.concatenate([action['right']['delta'], [action['right']['pos'][-1]]])}
        
        # Dynamically add camera data based on what's available in obs
        for key in list(obs.keys()):  # Convert to list to avoid RuntimeError
            if key.endswith('_camera') and isinstance(obs[key], dict) and 'images' in obs[key]:
                camera_name = key.replace('_camera', '_rgb')
                obs[camera_name] = obs[key]['images']['rgb']
        
        obs['joint'] = {'left': np.concatenate([obs['left']['joint_pos'], obs['left']['gripper_pos']]), 'right': np.concatenate([obs['right']['joint_pos'], obs['right']['gripper_pos']])}
        self.buffer.append(obs.copy())

    def save_episode_json(self, buffer, pickle_only=False):
        if not buffer:
            logger.warning("Empty buffer, no observations to save.")

        logger.info(f"Saving episode {self.traj_count} to {self.save_dir} with {len(buffer)} observations.")
        
        def get_buffer_dict(buffer):
            logger.info(f"Converting buffer to dictionary with {len(buffer)} observations.")
            buffer_dict = {}
            if buffer == []:
                return buffer_dict
            for key in buffer[0].keys():
                buffer_dict[key] = np.stack([obs[key] for obs in buffer])
            return buffer_dict
        
        buffer_dict = get_buffer_dict(buffer)
        if buffer_dict == {}:
            logger.warning("Empty buffer, no observations to save.")
            return
        
        img_paths = {}
        task_name = self.instruction
        actions = buffer_dict['action']
        delta = buffer_dict['delta']
        joint = buffer_dict['joint']

        if not pickle_only:
            # save rgb from camera
            rgb_keys = [key for key in buffer_dict.keys() if "rgb" in key]
            rgb_keys.sort()  # Sort from smallest to largest
            # logger.info(f"Found {len(rgb_keys)} RGB cameras: {rgb_keys}")
            
            for key in rgb_keys:
                save_dir = os.path.join(self.save_dir, f'{self.traj_count:06d}')
                os.makedirs(save_dir, exist_ok=True)
                
                save_dir = os.path.join(save_dir, key)
                
                os.makedirs(save_dir, exist_ok=True)
                # logger.info(f"Created directory for camera {key}: {save_dir}")

                paths = []
                tasks = []

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    for i, img in enumerate(buffer_dict[key]):
                        img_path = os.path.join(save_dir, f'{i:06d}.png')
                        tasks.append(executor.submit(self.save_image, img, img_path))
                        paths.append(img_path)

                    # Wait for all parallel tasks to complete
                    concurrent.futures.wait(tasks)
                    img_paths.setdefault(key, []).extend(paths)
                    # logger.info(f"Saved {len(paths)} images for camera {key}")

            # add action and delta to json
            # note that raw action is the joint returned by the viser ik, while the other joint is from robot obs
            json_data = []
            for i in range(len(actions)):
                json_data_obs = {
                    'language_instruction': task_name,
                    'left_raw_action': str(actions[i]['left'].tolist()),
                    'left_delta_action': str(delta[i]['left'].tolist()),
                    'right_raw_action': str(actions[i]['right'].tolist()),
                    'right_delta_action' : str(delta[i]['right'].tolist()),
                    'left_joint': str(joint[i]['left'].tolist()),
                    'right_joint': str(joint[i]['right'].tolist())
                }

                # add image paths to json
                for j, rgb_key in enumerate(rgb_keys):
                    json_data_obs[f"image_{rgb_key}"] = img_paths[rgb_key][i]
                
                json_data.append(json_data_obs)

            json_save_path = os.path.join(self.save_dir, f'{self.traj_count:06d}', f'{self.traj_count:06d}.json')
            os.makedirs(os.path.dirname(json_save_path), exist_ok=True)

            if not os.path.exists(json_save_path):
                with open(json_save_path, 'w') as f:
                    json.dump(json_data, f, indent=4)
                # logger.info(f"Created new JSON file: with {len(json_data)} observations.")
            else:
                with open(json_save_path, 'r') as f:
                    existing_data = json.load(f)
                existing_data.extend(json_data)
                with open(json_save_path, 'w') as f:
                    json.dump(existing_data, f, indent=4)
                logger.info(f"Added {len(json_data)} observations to {json_save_path}")
        # Just to let user know that the episode is saved
        logger.info(f"Complete!!!! Saved episode {self.traj_count} to {self.save_dir} with {len(buffer)} observations.")
        self.traj_count += 1
    
    def save_image(self, image, path):
        Image.fromarray(image).save(path)
