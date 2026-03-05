# RoboTwin Environment Setup Guide

This guide provides step-by-step instructions for setting up the RoboTwin environment for reinforcement learning experiments.

## Installation Steps

### 1. Clone Required Repositories

Clone the necessary third-party repositories:

```bash
# Clone RoboTwin repository
git clone https://github.com/RoboTwin-Platform/RoboTwin.git third_party/robtowin
```

### 2. Download Assets

Download the required assets for the environment. The specific download process depends on your setup requirements.

```bash
cd third_party/robotwin/assets
bash _download.py
```

### 3. Build and Set Environment Variables

Add the robotwin directory to your Python path:

```bash
cd third_party/robtowin
bash script/_install.sh
export PYTHONPATH="/path/to/robotwin":$PYTHONPATH
```

### 4. Configure Asset Paths (Auto completed, refer to script/_install.sh)

Update the configuration files with the correct paths to your assets:

**File: `assets/embodiments/aloha-agilex/curobo_left.yml`**
```yaml
# Replace with appropriate paths
urdf_path: path/to/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
collision_spheres: path/to/assets/embodiments/aloha-agilex/collision_aloha_left.yml
```

**File: `assets/embodiments/aloha-agilex/curobo_right.yml`**
```yaml
# Replace with appropriate paths
urdf_path: path/to/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf
collision_spheres: path/to/assets/embodiments/aloha-agilex/collision_aloha_right.yml
```

## Run the environment tests

```bash
python robotwin_test.py
```

### Getting Help

If you encounter issues not covered in this guide, please:
1. Check the [CuRobo documentation](https://github.com/NVlabs/curobo)
2. Check the [RoboTwin documentation](https://github.com/RoboTwin-Platform/RoboTwin)
3. Create an issue in RLinf

## License

Please refer to the individual repository licenses for CuRobo and RoboTwin components.
