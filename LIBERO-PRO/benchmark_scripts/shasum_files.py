import os
import hashlib
from pathlib import Path


def shasum_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            data = f.read()
        sha_hash = hashlib.sha1(data).hexdigest()
        file_name = file_path.split("/")[-1]
        return {file_name: sha_hash}
    else:
        return None


def shasum_datasets(download_dir="datasets"):
    dataset_shasum = {}
    for dataset_name in [
        "libero_object",
        "libero_goal",
        "libero_spatial",
        "libero_10",
        "libero_90",
    ]:
        dataset_dir = os.path.join(download_dir, dataset_name)
        if os.path.exists(dataset_dir):
            count = 0
            for path in Path(dataset_dir).glob("*.hdf5"):
                count += 1
            if not (
                (count == 10 and dataset_name != "libero_90")
                or (count == 90 and dataset_name == "libero_90")
            ):
                print("file count doesn't match")
        else:
            print("dataset not found")
        for path in Path(dataset_dir).glob("*.hdf5"):
            dataset_shasum.update(shasum_file(str(path)))
        print(dataset_shasum)


# def shalsum_pretrained_models():


# def shalsum_pretrained_policies():


shasum_datasets()
