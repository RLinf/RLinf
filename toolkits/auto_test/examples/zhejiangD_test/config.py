# Training Project Configuration

# Project name
project_name = "zhejiangD_test"

# Runtime Environment Configuration
training_runtime = {
    "start_commands": [
        "sleep 600",
    ],
    "environment_variables": {},
}

# Compute Resources Configuration
training_compute = {
    "node_count": 1,
}

# Training Project Configuration
# Minimal connectivity verification task using ZhejiangD Spot resources.
training_project = {
    "name": project_name,
    "description": "zhejiangD spot smoke test",
    "image_id": "im-c7flk4j34cxnbjut",
    "framework_id": "fw-c6q6a7sfyhoeb5xi",
    "compute": training_compute,
    "runtime": training_runtime,
    "resource_type": "spot",
    "resource_region": "zhejiangD",
    "resource_gpu_num": 1,
}
