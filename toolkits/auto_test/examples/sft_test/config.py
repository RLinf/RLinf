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
project_name = "sft_test"

# Task Type: Supervised fine-tuning task
task_type = "supervised_learning"

# Runtime Environment Configuration
# Training start commands
training_runtime = {
    "start_commands": [
        "pwd",
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
    "description": "Supervised fine-tuning task example",
    "task_type": task_type,  # Task type, used for auto-config
    "compute": training_compute,
    "runtime": training_runtime,
}
