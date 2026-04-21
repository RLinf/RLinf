# Training Project Configuration

# Project name
from pathlib import PurePosixPath

project_name = "rl_test"

# Task Type: Reinforcement learning training task
task_type = "reinforcement_learning"

#TODO: change to image_name
image_name = "agentic-rlinf0.1-torch2.6.0-openvla-openvlaoft-pi0"

#TODO: Adjust GPU count (load_spec_id)
#TODO: add libero_ppo_openpi.yaml to training_runtime.start_commands
#TODO: Add tensorboard interface (tb_log_dir)
#TODO: Analyze which files users need to modify
#           File format, need a document introducing the purpose of each file, with links to corresponding RLinf docs
#           model custom forward (support similar to custom_function approach)    -- Discuss in Friday meeting

remote_repo_path = "/mnt/public/hzl/RLinf-fork-xusi"
remote_project_path = str(PurePosixPath(remote_repo_path) / project_name)
remote_config_names = [
    "maniskill_ppo_openvla_quickstart, openvla",
    "maniskill_ppo_mlp, openvla",
    "maniskill_ppo_openpi, openpi",
    "maniskill_ppo_openpi_pi05, openpi",
    "maniskill_grpo_openvla, openvla",
    "maniskill_grpo_openvlaoft, openvlaoft",
]
remote_config_name = remote_config_names[0].split(",", 1)[0].strip()

# Runtime Environment Configuration
# Training start commands
training_runtime = {
    "start_commands": [
        'ls -la "$RLINF_REMOTE_PROJECT_PATH"',
        'cd "$RLINF_REMOTE_PROJECT_PATH"',
        'export STEPS="${STEPS:-10}"',
        'export SAVE_INTER="${SAVE_INTER:--1}"',
        'bash "$RLINF_REMOTE_PROJECT_PATH/run_embodiment_batch.sh"',
    ],
    "environment_variables": {
        "RLINF_REMOTE_REPO_PATH": remote_repo_path,
        "RLINF_REMOTE_PROJECT_PATH": remote_project_path,
        "RLINF_CONFIG_NAME": remote_config_name,
        "RLINF_CONFIG_SPECS": "\n".join(remote_config_names),
        "RLINF_CONFIG_NAMES": ",".join(
            config_spec.split(",", 1)[0].strip() for config_spec in remote_config_names
        ),
        "EMBODIED_PATH": remote_project_path,
        "REPO_PATH": remote_repo_path,
        "CONFIG_DIR": remote_project_path,
    },
}

# Compute Resources Configuration
training_compute = {
    "node_count": 1,  # Number of workers
    # TODO: add cluster_name to training_compute, change config_mapping.py
    # "cluster_name": "ningxiaB",
}

# Training Project Configuration
# Only need to modify three items:
# 1. resource_type: "spot" or "reserved"
# 2. resource_region: "beijingD", "ningxiaB" or "zhejiangD"
# 3. resource_gpu_num: GPUs per node, e.g., 1 / 2 / 4 / 8
#    Note: zhejiangD currently only supports Spot instances
# The remaining pool/spec/mount will be auto-matched to the corresponding region and GPU count
training_project = {
    "name": project_name,
    "description": "Reinforcement learning training task example",
    "task_type": task_type,  # Task type, used for auto-config
    "compute": training_compute,
    "runtime": training_runtime,
    "resource_type": "reserved",
    "resource_region": "ningxiaB",
    "resource_gpu_num": 4,
}
