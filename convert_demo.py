import os
import pickle as pkl
import numpy as np
import torch
from tqdm import tqdm

demo_path = "franka_lift_cube_image_20_trajs.pkl"
tgt_path = f"torch_franka_lift_cube_image_20_trajs.pkl"
expand_state = True

demo_path = "peg_insert_20_demos_2025-10-23_18-21-43.pkl"
tgt_path = "torch_peg_insert_20_demos_2025-10-23_18-21-43.pkl"
expand_state = False

if not os.path.exists(demo_path):
    raise FileNotFoundError(f"File {demo_path} not found")

with open(demo_path, "rb") as f:
    trajs = pkl.load(f)

obs_key_map = {
    "front": "images/base_camera", 
    "wrist": "images/wrist_camera", 
    "wrist_1": "images/wrist_1", 
    "state": "states"
}

def convert_data():
    torch_trajs = []
    for traj in tqdm(trajs):
        torch_traj = dict()

        torch_traj["transitions"] = {
            "obs": {}, "next_obs": {}
        }
        # observations
        for key, value in traj["observations"].items():
            tgt_key = obs_key_map[key]
            tgt_value = torch.from_numpy(value)
            if "images" in tgt_key:
                tgt_value = tgt_value[0].permute(2, 0, 1).float() / 255
            if tgt_key == "states":
                if len(tgt_value.shape) > 1:
                    tgt_value = tgt_value.flatten()
                if expand_state:
                    tgt_value_0 = torch.zeros(29)
                    tgt_value_0[:7] = tgt_value
                    tgt_value = tgt_value_0

            torch_traj["transitions"]["obs"][tgt_key] = tgt_value

        # next observations
        for key, value in traj["next_observations"].items():
            tgt_key = obs_key_map[key]
            tgt_value = torch.from_numpy(value)
            if "images" in tgt_key:
                tgt_value = tgt_value[0].permute(2, 0, 1).float() / 255
            if tgt_key == "states":
                if len(tgt_value.shape) > 1:
                    tgt_value = tgt_value.flatten()
                if expand_state:
                    tgt_value_0 = torch.zeros(29)
                    tgt_value_0[:7] = tgt_value
                    tgt_value = tgt_value_0
            torch_traj["transitions"]["next_obs"][tgt_key] = tgt_value
        
        torch_traj["action"] = torch.from_numpy(traj["actions"].flatten())
        
        for key in ["rewards", "dones"]:
            value = traj[key]
            if isinstance(value, np.ndarray):
                if len(value.shape) == 0:
                    value = np.array([value, ])
                torch_traj[key] = torch.from_numpy(value)
            else:
                torch_traj[key] = torch.tensor([value, ])

        torch_trajs.append(torch_traj)
        

    with open(tgt_path, "wb") as f:
        pkl.dump(torch_trajs, f)

def show_data_info():
    traj = trajs[0]
    print("keys:", traj.keys()) # dict_keys(['observations', 'actions', 'next_observations', 'rewards', 'masks', 'dones'])
    obs = traj["observations"]
    print(obs.keys()) # state, wrist_1
    print(obs["wrist_1"].shape)
    print(obs["state"].shape, type(obs["state"])) # [1, 19]
    print(obs["wrist_1"].shape) # [1, 128, 128, 3]

    print(traj["actions"].shape) # [6, ]; 
    for key in ["rewards", "masks", "dones"]:
        print(traj[key], type(traj[key]))

if __name__ == "__main__":
    show_data_info()
    convert_data()