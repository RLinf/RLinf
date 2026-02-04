## Habitat-Sim & Habitat-Lab Installation Guide

```sh
# Prepare habitat env
cd /opt/venv
uv venv habitat --python 3.10
source /opt/venv/habitat/bin/activate
cd /data
git clone https://github.com/RLinf/RLinf.git
cd /data/RLinf
uv sync --active --extra embodied

# Clone Required Repositories
cd /opt
git clone https://github.com/facebookresearch/habitat-sim.git
git clone https://github.com/facebookresearch/habitat-lab.git

cd /opt/habitat-sim
git submodule update --init --recursive
git checkout v0.3,3
# Correct the CMake File
sed -i 's/^cmake_minimum_required.*$/cmake_minimum_required(VERSION 3.5)/' src/deps/zstd/build/cmake/CMakeLists.txt
sed -i 's/^cmake_minimum_required.*$/cmake_minimum_required(VERSION 3.5)/' src/deps/assimp/CMakeLists.txt

# Install System-Level Ninja
apt-get update && apt-get install -y ninja-build
uv pip install ninja
export CMAKE_MAKE_PROGRAM=/usr/bin/ninja
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# Habitat-Sim Installation
uv pip install . --config-settings="--build-option=--headless" --config-settings="--build-option=--with-bullet"
uv pip install build/deps/magnum-bindings/src/python/

# Habitat-lab Installation
cd /opt/habitat-lab
git checkout v0.3.3
uv pip install -e habitat-lab
uv pip install -e habitat-baselines

```

## VLN-CE dataset preparation

Download the scene dataset:
- For **R2R**, **RxR**: Download the MP3D scenes [here](http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip), and put them into `VLN-CE/scene_dataset` folder.

Download the VLN-CE episodes:
 - [r2r](https://drive.google.com/file/d/18DCrNcpxESnps1IbXVjXSbGLDzcSOqzD/view) (Rename `R2R_VLNCE_v1/` -> `r2r/`)
 - [rxr](https://drive.google.com/file/d/145xzLjxBaNTbVgBfQ8e9EsBAV8W-SM0t/view) (Rename `RxR_VLNCE_v0/` -> `rxr/`)
 -  Put them into `VLN-CE/datasets` folder

Download the GT actions:
- For every episodes in eval split, we use `ShortestPathFollower` to generate the gt actions, please download [here](http://kaldir.vc.in.tum.de/matterport/v1/tasks/mp3d_habitat.zip), and put it into `VLN-CE/actions` folder.

 Dataset structure:
 ```sh
 VLN-CE
|-- datasets
|   |-- r2r
|   |-- rxr
`-- scene_dataset
|   |-- mp3d
`-- actions
 ```

## Test habitat env
The multi-environment design takes the Libero environment as a reference. However, in Habitat, the episode order is fixed after `init_env`, and `reset()` does not support specifying a particular episode. Therefore, before launching the environments, I have to evenly distribute the episodes across all environments.

I once tried using Habitat’s built-in `VectorEnv`, but all environments must be reset and stepped synchronously.
```sh
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
python examples/embodiment/debug_habitat_env.py
```
