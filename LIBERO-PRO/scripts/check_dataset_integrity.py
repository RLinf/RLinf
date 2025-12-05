"""A script to check if any demonstration dataset does not have the exact number of demonstration trajectories"""

from pathlib import Path
import h5py
import numpy as np

from liberopro.liberopro import get_libero_path

error_datasets = []
for demo_file_name in Path(get_libero_path("datasets")).rglob("*hdf5"):

    demo_file = h5py.File(demo_file_name)

    count = 0
    for key in demo_file["data"].keys():
        if "demo" in key:
            count += 1

    if count == 50:
        traj_lengths = []
        action_min = np.inf
        action_max = -np.inf
        for demo_name in demo_file["data"].keys():
            traj_lengths.append(demo_file["data/{}/actions".format(demo_name)].shape[0])
        traj_lengths = np.array(traj_lengths)
        print(f"[info] dataset {demo_file_name} is in tact, test passed \u2714")
        print(np.mean(traj_lengths), " +- ", np.std(traj_lengths))
        if demo_file["data"].attrs["tag"] == "libero-v1":
            print("Version correct")

        print("=========================================")

    else:
        print("[error] !!!")
        error_datasets.append(demo_file_name)

if len(error_datasets) > 0:
    print("[error] The following datasets are corrupted:")
    for dataset in error_datasets:
        print(dataset)
