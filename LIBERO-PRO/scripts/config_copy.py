import os
import shutil
from liberopro.liberopro import get_libero_path


def main():
    target_path = os.path.abspath(os.path.join("./", "configs"))
    print(f"Copying configs to {target_path}")
    if os.path.exists(target_path):
        response = input("The target directory already exists. Overwrite it? (y/n) ")
        if response.lower() != "y":
            return
        shutil.rmtree(target_path)
    shutil.copytree(
        os.path.join(get_libero_path("benchmark_root"), "../configs"), target_path
    )


if __name__ == "__main__":
    main()
