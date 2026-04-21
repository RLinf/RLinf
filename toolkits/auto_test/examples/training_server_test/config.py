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

# Training Service Idle Resources Configuration

project_name = "training_server_test"
job_api = "train_service_idle_resources"
num_worker = 1
num_gpu_per_worker = 8

training_runtime = {
    "start_commands": [
        "cd /mnt/public/xusi/RLinf-fork-v0.2",
        "bash run_yaml_zhou.sh",
    ],
    "environment_variables": {},
}

training_compute = {
    "node_count": num_worker,
}

training_project = {
    "name": project_name,
    "description": "idle resources smoke test",
    "job_api": job_api,
    "image_id": "im-dcm6egnmqdgurn6e",
    "framework_id": "fw-c6q6a7sfyhoeb5xi",
    "compute": training_compute,
    "runtime": training_runtime,
    "resource_type": "spot",
    "resource_region": "zhejiangD",
    "resource_gpu_num": num_gpu_per_worker,
    "expect_train_complete_time": 360000,
    "shared_mem": 1,
    "rdma_enable": False,
    "mount": [
        {
            "path": "/mnt/public/",
            "volume_id": "vo-dba4je5b5vun473d",
            "rw_setting": "can_write",
        }
    ],
}
