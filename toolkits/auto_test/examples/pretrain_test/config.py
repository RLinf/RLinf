# Training Project Configuration

# Project name
project_name = "pretrain_test"

# Task Type: Pretrain task
task_type = "pretrain"

# Runtime Environment Configuration
# Training start commands
training_runtime = {
    "start_commands": [
        "python model.py train",
    ],
    "environment_variables": {
        "INFINI_API_KEY": "your_infini_api_key",
    },
}

# Compute Resources Configuration
training_compute = {
    "node_count": 1,  # Number of workers
}

# Training Project Configuration
# Note: image_id, framework_id, resource_spec_id, pool_id, volume_id, etc.
# will be automatically obtained from config mapping based on task_type, no need to fill manually
# mount and fault_tolerance configs will also be auto-configured based on task_type, no need to fill manually
training_project = {
    "name": project_name,
    "description": "Pretrain task example",
    "task_type": task_type,  # Task type, used for auto-config
    "compute": training_compute,
    "runtime": training_runtime,
}
