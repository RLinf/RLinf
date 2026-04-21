# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
