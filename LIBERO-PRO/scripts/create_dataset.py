import argparse
import os
from pathlib import Path
import h5py
import numpy as np
import json
import robosuite
import robosuite.utils.transform_utils as T
import robosuite.macros as macros

import init_path
import liberopro.liberopro.utils.utils as libero_utils
import cv2
from PIL import Image
from robosuite.utils import camera_utils

from liberopro.liberopro.envs import *
from liberopro.liberopro import get_libero_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo-file", default="demo.hdf5")

    parser.add_argument(
        "--use-actions",
        action="store_true",
    )
    parser.add_argument("--use-camera-obs", action="store_true")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="datasets/",
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default="training_set",
    )

    parser.add_argument("--no-proprio", action="store_true")

    parser.add_argument(
        "--use-depth",
        action="store_true",
    )

    args = parser.parse_args()

    hdf5_path = args.demo_file
    f = h5py.File(hdf5_path, "r")
    env_name = f["data"].attrs["env"]

    env_args = f["data"].attrs["env_info"]
    env_kwargs = json.loads(f["data"].attrs["env_info"])

    problem_info = json.loads(f["data"].attrs["problem_info"])
    problem_info["domain_name"]
    problem_name = problem_info["problem_name"]
    language_instruction = problem_info["language_instruction"]

    # list of all demonstrations episodes
    demos = list(f["data"].keys())

    bddl_file_name = f["data"].attrs["bddl_file_name"]

    bddl_file_dir = os.path.dirname(bddl_file_name)
    replace_bddl_prefix = "/".join(bddl_file_dir.split("bddl_files/")[:-1] + "bddl_files")

    hdf5_path = os.path.join(get_libero_path("datasets"), bddl_file_dir.split("bddl_files/")[-1].replace(".bddl", "_demo.hdf5"))

    output_parent_dir = Path(hdf5_path).parent
    output_parent_dir.mkdir(parents=True, exist_ok=True)

    h5py_f = h5py.File(hdf5_path, "w")

    grp = h5py_f.create_group("data")

    grp.attrs["env_name"] = env_name
    grp.attrs["problem_info"] = f["data"].attrs["problem_info"]
    grp.attrs["macros_image_convention"] = macros.IMAGE_CONVENTION

    libero_utils.update_env_kwargs(
        env_kwargs,
        bddl_file_name=bddl_file_name,
        has_renderer=not args.use_camera_obs,
        has_offscreen_renderer=args.use_camera_obs,
        ignore_done=True,
        use_camera_obs=args.use_camera_obs,
        camera_depths=args.use_depth,
        camera_names=[
            "robot0_eye_in_hand",
            "agentview",
        ],
        reward_shaping=True,
        control_freq=20,
        camera_heights=128,
        camera_widths=128,
        camera_segmentations=None,
    )

    grp.attrs["bddl_file_name"] = bddl_file_name
    grp.attrs["bddl_file_content"] = open(bddl_file_name, "r").read()
    print(grp.attrs["bddl_file_content"])

    env = TASK_MAPPING[problem_name](
        **env_kwargs,
    )

    env_args = {
        "type": 1,
        "env_name": env_name,
        "problem_name": problem_name,
        "bddl_file": f["data"].attrs["bddl_file_name"],
        "env_kwargs": env_kwargs,
    }

    grp.attrs["env_args"] = json.dumps(env_args)
    print(grp.attrs["env_args"])
    total_len = 0
    demos = demos

    cap_index = 5

    for (i, ep) in enumerate(demos):
        print("Playing back random episode... (press ESC to quit)")

        # # select an episode randomly
        # read the model xml, using the metadata stored in the attribute for this episode
        model_xml = f["data/{}".format(ep)].attrs["model_file"]
        reset_success = False
        while not reset_success:
            try:
                env.reset()
                reset_success = True
            except:
                continue

        model_xml = libero_utils.postprocess_model_xml(model_xml, {})

        if not args.use_camera_obs:
            env.viewer.set_camera(0)

        # load the flattened mujoco states
        states = f["data/{}/states".format(ep)][()]
        actions = np.array(f["data/{}/actions".format(ep)][()])

        num_actions = actions.shape[0]

        init_idx = 0
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(states[init_idx])
        env.sim.forward()
        model_xml = env.sim.model.get_xml()

        ee_states = []
        gripper_states = []
        joint_states = []
        robot_states = []

        agentview_images = []
        eye_in_hand_images = []

        agentview_depths = []
        eye_in_hand_depths = []

        agentview_seg = {0: [], 1: [], 2: [], 3: [], 4: []}

        rewards = []
        dones = []

        valid_index = []

        for j, action in enumerate(actions):

            obs, reward, done, info = env.step(action)

            if j < num_actions - 1:
                # ensure that the actions deterministically lead to the same recorded states
                state_playback = env.sim.get_state().flatten()
                # assert(np.all(np.equal(states[j + 1], state_playback)))
                err = np.linalg.norm(states[j + 1] - state_playback)

                if err > 0.01:
                    print(
                        f"[warning] playback diverged by {err:.2f} for ep {ep} at step {j}"
                    )

            # Skip recording because the force sensor is not stable in
            # the beginning
            if j < cap_index:
                continue

            valid_index.append(j)

            if not args.no_proprio:
                if "robot0_gripper_qpos" in obs:
                    gripper_states.append(obs["robot0_gripper_qpos"])

                joint_states.append(obs["robot0_joint_pos"])

                ee_states.append(
                    np.hstack(
                        (
                            obs["robot0_eef_pos"],
                            T.quat2axisangle(obs["robot0_eef_quat"]),
                        )
                    )
                )

            robot_states.append(env.get_robot_state_vector(obs))

            if args.use_camera_obs:

                if args.use_depth:
                    agentview_depths.append(obs["agentview_depth"])
                    eye_in_hand_depths.append(obs["robot0_eye_in_hand_depth"])

                agentview_images.append(obs["agentview_image"])
                eye_in_hand_images.append(obs["robot0_eye_in_hand_image"])
            else:
                env.render()

        # end of one trajectory
        states = states[valid_index]
        actions = actions[valid_index]
        dones = np.zeros(len(actions)).astype(np.uint8)
        dones[-1] = 1
        rewards = np.zeros(len(actions)).astype(np.uint8)
        rewards[-1] = 1
        print(len(actions), len(agentview_images))
        assert len(actions) == len(agentview_images)
        print(len(actions))

        ep_data_grp = grp.create_group(f"demo_{i}")

        obs_grp = ep_data_grp.create_group("obs")
        if not args.no_proprio:
            obs_grp.create_dataset(
                "gripper_states", data=np.stack(gripper_states, axis=0)
            )
            obs_grp.create_dataset("joint_states", data=np.stack(joint_states, axis=0))
            obs_grp.create_dataset("ee_states", data=np.stack(ee_states, axis=0))
            obs_grp.create_dataset("ee_pos", data=np.stack(ee_states, axis=0)[:, :3])
            obs_grp.create_dataset("ee_ori", data=np.stack(ee_states, axis=0)[:, 3:])

        obs_grp.create_dataset("agentview_rgb", data=np.stack(agentview_images, axis=0))
        obs_grp.create_dataset(
            "eye_in_hand_rgb", data=np.stack(eye_in_hand_images, axis=0)
        )
        if args.use_depth:
            obs_grp.create_dataset(
                "agentview_depth", data=np.stack(agentview_depths, axis=0)
            )
            obs_grp.create_dataset(
                "eye_in_hand_depth", data=np.stack(eye_in_hand_depths, axis=0)
            )

        ep_data_grp.create_dataset("actions", data=actions)
        ep_data_grp.create_dataset("states", data=states)
        ep_data_grp.create_dataset("robot_states", data=np.stack(robot_states, axis=0))
        ep_data_grp.create_dataset("rewards", data=rewards)
        ep_data_grp.create_dataset("dones", data=dones)
        ep_data_grp.attrs["num_samples"] = len(agentview_images)
        ep_data_grp.attrs["model_file"] = model_xml
        ep_data_grp.attrs["init_state"] = states[init_idx]
        total_len += len(agentview_images)

    grp.attrs["num_demos"] = len(demos)
    grp.attrs["total"] = total_len
    env.close()

    h5py_f.close()
    f.close()

    print("The created dataset is saved in the following path: ")
    print(hdf5_path)


if __name__ == "__main__":
    main()
