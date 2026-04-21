# Training Service Idle Resources Configuration

project_name = "training_server_test"
job_api = "train_service_idle_resources"
num_worker = 1
num_gpu_per_worker = 1

workspace_dir = "/mnt/public/zhouyiming"

local_pre_commands = [
    "echo 'Preparing...'",
    f"cd {workspace_dir}",
    "mkdir -p RLinf_test",
    "cd RLinf_test",
    "rm -rf RLinf",
    "git clone https://ghfast.top/https://github.com/RLinf/RLinf.git", 
    "cd RLinf",
    "ls",
    # envsetup(optional)
    "python3 -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple uv",
    "bash ./requirements/install.sh embodied --model openvla --env maniskill_libero --use-mirror --venv maniskill_libero_openvla",
    ". ./maniskill_libero_openvla/bin/activate",
    # TODO: "mv required run_yaml.sh and check.py",
    # TODO: where should the source run_yaml.sh and check.py be placed?
    f"mv /mnt/public/zhouyiming/Training_Client/training_client/run_yaml.sh {workspace_dir}/RLinf_test/RLinf",
    f"ls {workspace_dir}/RLinf_test/RLinf", 
]

# xusi
# training_runtime = {
#     "start_commands": [
#         "cd /mnt/public/xusi/RLinf-fork-v0.2",
#         "bash run_yaml.sh",
#     ],
#     "environment_variables": {},
# }

training_runtime = {
    "start_commands": [
        f"cd {workspace_dir}/RLinf_test/RLinf",
        f"ls",
        "sleep 100",
        # "bash run_yaml.sh",
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
    "expect_train_complete_time": 3600,
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
