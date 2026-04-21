# 如何上传文件到存储卷

根据 `document.yaml` API 文档，平台**没有直接的文件上传到存储卷的 API**。以下是几种可行的上传方式：

## 方式1：通过开发机上传（推荐）

1. **创建开发机并挂载存储卷**

```python
from submit_job.cli.api_client import make_api_request

# 创建开发机，挂载存储卷
dev_machine_data = {
    "dev_machine_name": "upload-helper",
    "image_id": "im-xxxxxxxxxxxxx",  # 使用轻量级镜像
    "resource_spec_id": "rs-xxxxxxxxxxxxx",
    "mount": [
        {
            "path": "/mnt/volume",
            "volume_id": "vo-xxxxxxxxxxxxx",  # 你的存储卷 ID
            "rw_setting": "can_write"
        }
    ],
    "resource_type": 1  # Spot
}

result = make_api_request(
    endpoint="/api/platform/open/v1/dev_machine/create",
    json_data=dev_machine_data
)

dev_machine_id = result.get("dev_machine_id")
```

2. **启动开发机**

```python
make_api_request(
    endpoint="/api/platform/open/v1/dev_machine/start",
    json_data={"dev_machine_id": dev_machine_id}
)
```

3. **获取开发机 SSH 信息并上传文件**

```python
# 获取开发机详情（包含 SSH 信息）
detail = make_api_request(
    endpoint="/api/platform/open/v1/dev_machine/detail",
    json_data={"dev_machine_id_list": [dev_machine_id]}
)

# 使用 SSH 或 SCP 上传文件
# scp project.tar.gz user@dev_machine_ip:/mnt/volume/target_path/
```

## 方式2：通过 Web 控制台

1. 登录平台 Web 控制台
2. 进入存储卷管理页面
3. 选择目标存储卷
4. 使用文件管理器上传文件

## 方式3：使用 API Key 和临时任务上传

创建一个临时任务，挂载存储卷，在启动命令中上传文件：

```python
# 创建临时任务上传文件
job_data = {
    "job_type": "train",
    "job_name": "upload-file-temp",
    "image_id": "im-xxxxxxxxxxxxx",
    "framework_id": "fw-xxxxxxxxxxxxx",
    "resource_spec_id": "rs-xxxxxxxxxxxxx",
    "worker_num": 1,
    "entry_point": "python -c \"import urllib.request; urllib.request.urlretrieve('https://your-file-url', '/mnt/volume/target_path/project.tar.gz')\"",
    "resource_type": 1,
    "mount": [
        {
            "path": "/mnt/volume",
            "volume_id": "vo-xxxxxxxxxxxxx",
            "rw_setting": "can_write"
        }
    ]
}

result = make_api_request(
    endpoint="/api/platform/open/v1/job/create",
    json_data=job_data
)
```

## 使用示例

### 使用 Python 上传（通过开发机）

```python
import os
from pathlib import Path
from submit_job.cli.push import push_training_job

# 设置 API Key
os.environ["INFINI_API_KEY"] = "sk-xxxxxxxxxxxxx"
os.environ["INFINI_VOLUME_ID"] = "vo-xxxxxxxxxxxxx"

# 上传项目（如果 API 可用）
try:
    result = push_training_job(
        project_dir=Path("my-project"),
        volume_id="vo-xxxxxxxxxxxxx",
        target_path="my-project"
    )
    print(result)
except ValueError as e:
    # API 不存在，使用替代方案
    print(f"提示: {e}")
    print("请使用开发机或其他方式上传文件")
```

### 使用命令行

```bash
# 设置环境变量
export INFINI_API_KEY="sk-xxxxxxxxxxxxx"
export INFINI_VOLUME_ID="vo-xxxxxxxxxxxxx"

# 尝试上传（如果 API 可用）
submit_job push --volume-id vo-xxxxxxxxxxxxx --target-path my-project
```

## 注意事项

1. **API Key 配置**：确保设置了 `INFINI_API_KEY` 环境变量
2. **存储卷权限**：确保存储卷有写入权限（`rw_setting: can_write`）
3. **文件大小限制**：注意平台的文件大小限制
4. **网络传输**：大文件上传可能需要较长时间，注意超时设置

## 相关 API

- 查询存储卷列表：`/api/platform/open/v1/volume/list`
- 创建开发机：`/api/platform/open/v1/dev_machine/create`
- 创建任务：`/api/platform/open/v1/job/create`

更多详情请参考 `document.yaml` API 文档。
