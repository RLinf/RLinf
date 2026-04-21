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

"""上传项目文件到平台"""

import base64
import os
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Optional

import requests

if __package__ in {None, ""}:
    _cli_dir = Path(__file__).resolve().parent
    if str(_cli_dir) not in sys.path:
        sys.path.insert(0, str(_cli_dir))
    from api_client import get_api_key, get_base_url, make_api_request
    from config_mapping import (
        apply_region_resource_config,
        get_aicoder_upload_preset,
        get_full_config,
        normalize_resource_type,
    )
else:
    from .api_client import get_api_key, get_base_url, make_api_request
    from .config_mapping import (
        apply_region_resource_config,
        get_aicoder_upload_preset,
        get_full_config,
        normalize_resource_type,
    )


DEFAULT_AICODER_MOUNT_PATH = "/mnt/public"
DEFAULT_AICODER_SSH_PROXY_JUMP = "ssh-jumper.cloud.infini-ai.com"
DEFAULT_AICODER_SSH_USER = "root"
AICODER_SSH_DETAIL_MAX_ATTEMPTS = 20
AICODER_SSH_DETAIL_RETRY_INTERVAL_SECONDS = 3


def push_training_job(
    project_dir: Path,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    volume_id: Optional[str] = None,
    target_path: Optional[str] = None,
    image_id: Optional[str] = None,
    framework_id: Optional[str] = None,
    resource_spec_id: Optional[str] = None,
    upload_method: str = "job",
    upload_preset: Optional[str] = None,
    region_id: Optional[str] = None,
    aicoder_mount_path: Optional[str] = None,
    aicoder_mount_rw_setting: str = "can_write",
    ssh_proxy_jump: Optional[str] = None,
    ssh_user: Optional[str] = None,
) -> dict:
    """
    上传项目文件到平台。

    支持两种上传方式：
    1. job: 通过临时任务将 tar.gz 写入挂载卷
    2. aicoder_scp: 创建 AICoder 实例后，通过 scp 递归上传项目目录
    """
    project_dir = Path(project_dir).resolve()
    _validate_project_dir(project_dir)

    upload_method, volume_id, region_id, aicoder_mount_path = _apply_upload_preset(
        upload_preset=upload_preset,
        upload_method=upload_method,
        volume_id=volume_id,
        region_id=region_id,
        aicoder_mount_path=aicoder_mount_path,
    )

    normalized_upload_method = (upload_method or "job").strip().lower()
    if normalized_upload_method in {"job", "direct"}:
        return _push_training_job_via_job(
            project_dir=project_dir,
            api_key=api_key,
            base_url=base_url,
            volume_id=volume_id,
            target_path=target_path,
            image_id=image_id,
            framework_id=framework_id,
            resource_spec_id=resource_spec_id,
        )

    if normalized_upload_method in {"aicoder", "aicoder_scp"}:
        return _push_training_job_via_aicoder_scp(
            project_dir=project_dir,
            api_key=api_key,
            base_url=base_url,
            volume_id=volume_id,
            target_path=target_path,
            region_id=region_id,
            aicoder_mount_path=aicoder_mount_path,
            aicoder_mount_rw_setting=aicoder_mount_rw_setting,
            ssh_proxy_jump=ssh_proxy_jump,
            ssh_user=ssh_user,
        )

    raise ValueError("不支持的 upload_method。支持的值：job、aicoder_scp。")


def _apply_upload_preset(
    upload_preset: Optional[str],
    upload_method: Optional[str],
    volume_id: Optional[str],
    region_id: Optional[str],
    aicoder_mount_path: Optional[str],
):
    """将上传预设展开为具体参数。"""
    if not upload_preset:
        return upload_method, volume_id, region_id, aicoder_mount_path

    preset_config = get_aicoder_upload_preset(upload_preset)
    normalized_upload_method = (upload_method or "").strip().lower()

    if normalized_upload_method in {"", "job"}:
        upload_method = preset_config.get("upload_method", "aicoder_scp")

    return (
        upload_method,
        volume_id or preset_config.get("volume_id"),
        region_id or preset_config.get("region_id"),
        aicoder_mount_path or preset_config.get("aicoder_mount_path"),
    )


def _push_training_job_via_job(
    project_dir: Path,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    volume_id: Optional[str] = None,
    target_path: Optional[str] = None,
    image_id: Optional[str] = None,
    framework_id: Optional[str] = None,
    resource_spec_id: Optional[str] = None,
) -> dict:
    """通过临时任务上传项目 tar.gz 到存储卷。"""
    if api_key is None:
        api_key = get_api_key()

    if base_url is None:
        base_url = get_base_url()

    if target_path is None:
        target_path = project_dir.name

    training_job = _load_training_job_config(project_dir)

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp_file:
        tar_path = Path(tmp_file.name)

    try:
        _create_project_tarball(project_dir, tar_path)

        tar_content = tar_path.read_bytes()
        tar_base64 = base64.b64encode(tar_content).decode("utf-8")

        if image_id is None:
            image_id = training_job.get("image_id") or os.getenv("INFINI_IMAGE_ID")
            if not image_id:
                raise ValueError(
                    "未指定镜像 ID。请选择以下方式之一：\n"
                    "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
                    "2. 在 config.py 中直接设置 image_id\n"
                    "3. 设置环境变量 INFINI_IMAGE_ID\n"
                    "4. 通过 --image-id 参数提供"
                )

        if framework_id is None:
            framework_id = training_job.get("framework_id") or os.getenv(
                "INFINI_FRAMEWORK_ID"
            )
            if not framework_id:
                raise ValueError(
                    "未指定框架 ID。请选择以下方式之一：\n"
                    "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
                    "2. 在 config.py 中直接设置 framework_id\n"
                    "3. 设置环境变量 INFINI_FRAMEWORK_ID\n"
                    "4. 通过 --framework-id 参数提供"
                )

        if resource_spec_id is None:
            resource_spec_id = training_job.get("resource_spec_id") or os.getenv(
                "INFINI_RESOURCE_SPEC_ID"
            )
            if not resource_spec_id:
                raise ValueError(
                    "未指定资源规格 ID。请选择以下方式之一：\n"
                    "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
                    "2. 在 config.py 中直接设置 resource_spec_id\n"
                    "3. 设置环境变量 INFINI_RESOURCE_SPEC_ID\n"
                    "4. 通过 --resource-spec-id 参数提供"
                )

        if volume_id is None:
            volume_id = training_job.get("volume_id") or os.getenv("INFINI_VOLUME_ID")
            if not volume_id:
                raise ValueError(
                    "未指定存储卷 ID。请选择以下方式之一：\n"
                    "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
                    "2. 设置环境变量 INFINI_VOLUME_ID\n"
                    "3. 通过 --volume-id 参数提供"
                )

        resource_type = normalize_resource_type(training_job.get("resource_type", 1))
        pool_id = training_job.get("pool_id")

        mount_path = "/mnt/volume"
        target_file_path = f"{mount_path}/{target_path}/project.tar.gz"

        file_size_mb = len(tar_content) / (1024 * 1024)
        base64_size = len(tar_base64)

        if file_size_mb > 5:
            print(
                f"警告: 文件较大 ({file_size_mb:.2f} MB)，base64 编码后大小: {base64_size / 1024:.2f} KB"
            )
            print("注意: 如果命令过长导致失败，请考虑使用 Web 控制台或其他方式上传。")

        tar_base64_escaped = tar_base64.replace("\\", "\\\\").replace('"', '\\"')
        upload_command = f'''python3 -c "
import base64
import os

target_file = '{target_file_path}'
os.makedirs(os.path.dirname(target_file), exist_ok=True)

file_content_b64 = \\\"{tar_base64_escaped}\\\"
file_content = base64.b64decode(file_content_b64)
with open(target_file, 'wb') as f:
    f.write(file_content)

print(f'文件已上传到: {{target_file}}')
print(f'文件大小: {{len(file_content)}} 字节')
"'''

        job_name = f"upload-{project_dir.name}-{int(time.time())}"
        job_data = {
            "job_type": "train",
            "job_name": job_name,
            "job_description": f"临时任务：上传项目文件 {project_dir.name} 到存储卷",
            "image_id": image_id,
            "framework_id": framework_id,
            "resource_spec_id": resource_spec_id,
            "worker_num": 1,
            "entry_point": upload_command,
            "resource_type": resource_type,
            "mount": [
                {
                    "path": mount_path,
                    "volume_id": volume_id,
                    "rw_setting": "can_write",
                }
            ],
            "fault_tolerance": {
                "auto_restart": {"enable": False},
                "hang_check": {"enable": False},
                "environment_check": {"enable": False},
            },
        }

        if pool_id:
            job_data["pool_id"] = pool_id

        job_data = {k: v for k, v in job_data.items() if v is not None}
        print(f"正在创建临时任务上传文件到存储卷 {volume_id}...")

        try:
            result = make_api_request(
                endpoint="/api/platform/open/v1/job/create",
                api_key=api_key,
                base_url=base_url,
                json_data=job_data,
            )
            job_id = _extract_result_value(result, "job_id")
            if not job_id:
                error_msg = result.get("msg") or result.get("message") or str(result)
                raise ValueError(f"创建上传任务失败: {error_msg}")
        except requests.exceptions.HTTPError as exc:
            error_msg = _extract_error_message(exc)

            if "Spot 规格已受限" in error_msg or (
                "Spot" in error_msg and "受限" in error_msg
            ):
                raise ValueError(
                    f"资源规格受限错误：\n"
                    f"  当前使用的 Spot 规格 ({resource_spec_id}) 已受限，无法使用。\n\n"
                    f"解决方案：\n"
                    f"1. 使用其他 Spot 规格：通过 --resource-spec-id 参数指定其他可用的资源规格 ID\n"
                    f'2. 使用包年包月资源：在 config.py 中设置 resource_type="reserved" 并提供 pool_id\n'
                    f"3. 查询可用资源规格：使用平台 API 查询可用的资源规格列表\n\n"
                    f"当前配置：\n"
                    f"  - 资源类型: {'Spot' if resource_type == 1 else '包年包月'}\n"
                    f"  - 资源规格 ID: {resource_spec_id}\n"
                    f"  - 镜像 ID: {image_id}\n"
                    f"  - 框架 ID: {framework_id}"
                )

            if "resource_spec" in error_msg.lower() or "资源规格" in error_msg:
                raise ValueError(
                    f"资源规格错误：\n"
                    f"  {error_msg}\n\n"
                    f"请检查：\n"
                    f"1. 资源规格 ID ({resource_spec_id}) 是否正确\n"
                    f"2. 资源规格是否可用\n"
                    f"3. 资源类型 (resource_type) 是否匹配\n"
                    f"4. 如果使用包年包月资源，是否提供了 pool_id"
                )

            raise ValueError(f"创建上传任务失败: {error_msg}")
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"创建上传任务时发生错误: {exc}")

        return {
            "ok": True,
            "message": "文件上传任务已创建",
            "upload_method": "job",
            "job_id": job_id,
            "job_name": job_name,
            "volume_id": volume_id,
            "target_path": target_path,
            "target_file": target_file_path,
            "result": result,
        }
    finally:
        if tar_path.exists():
            tar_path.unlink()


def _push_training_job_via_aicoder_scp(
    project_dir: Path,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    volume_id: Optional[str] = None,
    target_path: Optional[str] = None,
    region_id: Optional[str] = None,
    aicoder_mount_path: Optional[str] = None,
    aicoder_mount_rw_setting: str = "can_write",
    ssh_proxy_jump: Optional[str] = None,
    ssh_user: Optional[str] = None,
) -> dict:
    """创建 AICoder 实例并通过 scp 上传项目目录。"""
    if api_key is None:
        api_key = get_api_key()

    if base_url is None:
        base_url = get_base_url()

    training_job = _load_training_job_config(project_dir)

    if volume_id is None:
        volume_id = training_job.get("volume_id") or os.getenv("INFINI_VOLUME_ID")
        if not volume_id:
            raise ValueError(
                "未指定存储卷 ID。请选择以下方式之一：\n"
                "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
                "2. 设置环境变量 INFINI_VOLUME_ID\n"
                "3. 通过 --volume-id 参数提供"
            )

    region_id = region_id or os.getenv("INFINI_AICODER_REGION_ID")
    if not region_id:
        raise ValueError(
            "使用 AICoder 上传时必须提供 region_id。"
            "请通过 --region-id 参数或环境变量 INFINI_AICODER_REGION_ID 设置。"
        )

    aicoder_mount_path = (
        aicoder_mount_path
        or os.getenv("INFINI_AICODER_MOUNT_PATH")
        or DEFAULT_AICODER_MOUNT_PATH
    )
    configured_ssh_proxy_jump = ssh_proxy_jump or os.getenv(
        "INFINI_AICODER_SSH_PROXY_JUMP"
    )
    ssh_user = (
        ssh_user or os.getenv("INFINI_AICODER_SSH_USER") or DEFAULT_AICODER_SSH_USER
    )

    _ensure_local_command("ssh")
    _ensure_local_command("scp")

    create_payload = {
        "region_id": region_id,
        "mount": [
            {
                "path": aicoder_mount_path,
                "volume_id": volume_id,
                "rw_setting": aicoder_mount_rw_setting,
            }
        ],
    }

    print(f"正在创建 AICoder 实例，挂载存储卷 {volume_id}...")
    try:
        create_result = make_api_request(
            endpoint="/api/platform/open/v1/aicoder/create",
            api_key=api_key,
            base_url=base_url,
            json_data=create_payload,
        )
    except requests.exceptions.HTTPError as exc:
        raise ValueError(f"创建 AICoder 实例失败: {_extract_error_message(exc)}")

    ai_coder_id = _extract_result_value(create_result, "ai_coder_id")
    if not ai_coder_id:
        raise ValueError(
            f"创建 AICoder 实例失败，响应中未返回 ai_coder_id: {create_result}"
        )

    ssh_detail_result = _wait_for_aicoder_ssh_detail(
        ai_coder_id=ai_coder_id,
        api_key=api_key,
        base_url=base_url,
    )
    ssh_addr = _extract_result_value(ssh_detail_result, "addr")
    ssh_info = _parse_aicoder_ssh_addr(ssh_addr, default_user=ssh_user)
    ssh_info["proxy_jump"] = (
        configured_ssh_proxy_jump
        or ssh_info.get("proxy_jump")
        or DEFAULT_AICODER_SSH_PROXY_JUMP
    )

    remote_parent_path = _build_remote_parent_path(
        mount_path=aicoder_mount_path,
        target_path=target_path,
    )
    remote_target_path = remote_parent_path / project_dir.name

    print(f"正在准备远端目录: {remote_target_path}")
    _prepare_remote_target(ssh_info=ssh_info, remote_target_path=remote_target_path)

    print(f"正在通过 SCP 上传项目目录到 {remote_target_path}...")
    _run_scp_upload(
        project_dir=project_dir,
        ssh_info=ssh_info,
        remote_parent_path=remote_parent_path,
        remote_target_path=remote_target_path,
    )

    return {
        "ok": True,
        "message": "文件已通过 AICoder SCP 上传",
        "upload_method": "aicoder_scp",
        "ai_coder_id": ai_coder_id,
        "volume_id": volume_id,
        "target_path": str(remote_parent_path),
        "remote_path": str(remote_target_path),
        "ssh_addr": ssh_addr,
        "ssh_host": ssh_info["host_spec"],
        "ssh_proxy_jump": ssh_info.get("proxy_jump"),
        "result": create_result,
    }


def _validate_project_dir(project_dir: Path):
    """验证项目目录结构。"""
    required_files = ["config.py", "model.py"]
    for file_name in required_files:
        if not (project_dir / file_name).exists():
            raise FileNotFoundError(
                f"缺少必需文件: {file_name}。请确保在项目目录中运行此命令。"
            )


def _load_training_job_config(project_dir: Path) -> dict:
    """从 config.py 读取训练配置并合并任务类型映射。"""
    config_file = project_dir / "config.py"
    config_globals: dict[str, Any] = {}
    exec(config_file.read_text(encoding="utf-8"), config_globals)

    training_project = config_globals.get("training_project", {})
    training_job = dict(training_project.get("job", {}))
    task_type = training_job.get("task_type") or config_globals.get("task_type")

    if task_type:
        try:
            mapping_config = get_full_config(task_type)
            for key, value in mapping_config.items():
                if key not in training_job or training_job.get(key) is None:
                    training_job[key] = value
        except ValueError as exc:
            raise ValueError(
                f"无法从任务类型 '{task_type}' 获取配置: {exc}\n"
                f"请检查 config.py 中的 task_type 设置，或直接配置 image_id、framework_id 等字段。"
            )

    apply_region_resource_config(training_job)

    return training_job


def _extract_result_value(result: Any, field_name: str) -> Any:
    """兼容读取顶层字段和 data 字段。"""
    if not isinstance(result, dict):
        return None

    if field_name in result and result[field_name] is not None:
        return result[field_name]

    data = result.get("data")
    if isinstance(data, dict):
        return data.get(field_name)

    return None


def _extract_error_message(error: Exception) -> str:
    """从异常中提取尽可能明确的错误消息。"""
    message = str(error)
    response = getattr(error, "response", None)
    if response is None:
        return message

    try:
        error_data = response.json()
    except ValueError:
        return message

    if isinstance(error_data, dict):
        return error_data.get("msg") or error_data.get("message") or message

    return message


def _wait_for_aicoder_ssh_detail(
    ai_coder_id: str,
    api_key: str,
    base_url: str,
) -> dict:
    """轮询直到 AICoder SSH 地址可用。"""
    last_error: Optional[str] = None

    for attempt in range(1, AICODER_SSH_DETAIL_MAX_ATTEMPTS + 1):
        try:
            detail_result = make_api_request(
                endpoint="/api/platform/open/v1/aicoder/ssh/detail",
                api_key=api_key,
                base_url=base_url,
                json_data={"ai_coder_id": ai_coder_id},
            )
            is_confirm = _extract_result_value(detail_result, "is_confirm")
            ssh_addr = _extract_result_value(detail_result, "addr")

            if is_confirm is False:
                make_api_request(
                    endpoint="/api/platform/open/v1/aicoder/ssh/protocol/confirm",
                    api_key=api_key,
                    base_url=base_url,
                    json_data={},
                )
                last_error = "SSH 协议尚未确认，已自动确认，等待 SSH 地址生效"
            elif ssh_addr:
                return detail_result
            else:
                last_error = "AICoder SSH 地址尚未就绪"
        except requests.exceptions.HTTPError as exc:
            last_error = _extract_error_message(exc)
        except Exception as exc:
            last_error = str(exc)

        if attempt < AICODER_SSH_DETAIL_MAX_ATTEMPTS:
            time.sleep(AICODER_SSH_DETAIL_RETRY_INTERVAL_SECONDS)

    raise ValueError(
        f"等待 AICoder SSH 地址超时 (ai_coder_id={ai_coder_id})。"
        f"最后一次错误: {last_error or '未知错误'}"
    )


def _parse_aicoder_ssh_addr(ssh_addr: Optional[str], default_user: str) -> dict:
    """解析 AICoder 返回的 SSH 地址或 SSH 命令。"""
    if not ssh_addr:
        raise ValueError("AICoder SSH 地址为空，无法执行 SCP 上传。")

    tokens = shlex.split(ssh_addr)
    if tokens and tokens[0] == "ssh":
        tokens = tokens[1:]

    host_spec = None
    user = None
    proxy_jump = None
    port = None

    i = 0
    while i < len(tokens):
        token = tokens[i]

        if token in {"-J", "-oProxyJump"}:
            i += 1
            if i < len(tokens):
                proxy_jump = tokens[i]
        elif token.startswith("-J") and len(token) > 2:
            proxy_jump = token[2:]
        elif token in {"-p", "-P"}:
            i += 1
            if i < len(tokens):
                port = tokens[i]
        elif token == "-l":
            i += 1
            if i < len(tokens):
                user = tokens[i]
        elif token.startswith("-l") and len(token) > 2:
            user = token[2:]
        elif token in {"-i", "-F", "-o", "-S"}:
            i += 1
        elif token.startswith("-"):
            pass
        else:
            host_spec = token

        i += 1

    if host_spec is None:
        if " " not in ssh_addr:
            host_spec = ssh_addr.strip()
        else:
            raise ValueError(f"无法解析 AICoder SSH 地址: {ssh_addr}")

    if "@" in host_spec:
        parsed_user, host = host_spec.split("@", 1)
        user = user or parsed_user
    else:
        host = host_spec
        user = user or default_user
        host_spec = f"{user}@{host}"

    return {
        "host": host,
        "user": user,
        "host_spec": host_spec,
        "proxy_jump": proxy_jump,
        "port": port,
    }


def _build_remote_parent_path(
    mount_path: str,
    target_path: Optional[str],
) -> PurePosixPath:
    """生成远端父目录。"""
    normalized_target = (target_path or "").strip()

    if not normalized_target:
        return PurePosixPath(mount_path)

    if normalized_target.startswith("/"):
        return PurePosixPath(normalized_target)

    return PurePosixPath(mount_path) / normalized_target


def _prepare_remote_target(ssh_info: dict, remote_target_path: PurePosixPath):
    """确保远端目标父目录存在，并覆盖远端同名目录。"""
    remote_parent = shlex.quote(str(remote_target_path.parent))
    remote_target = shlex.quote(str(remote_target_path))
    remote_command = f"mkdir -p {remote_parent} && rm -rf {remote_target}"

    try:
        _run_ssh_command(ssh_info=ssh_info, remote_command=remote_command)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        raise ValueError(f"准备远端目录失败: {stderr or exc}")


def _run_scp_upload(
    project_dir: Path,
    ssh_info: dict,
    remote_parent_path: PurePosixPath,
    remote_target_path: PurePosixPath,
):
    """执行 scp 目录上传。"""
    command = ["scp", "-r"]
    if ssh_info.get("proxy_jump"):
        command.extend(["-J", ssh_info["proxy_jump"]])
    if ssh_info.get("port"):
        command.extend(["-P", str(ssh_info["port"])])

    remote_target = f"{ssh_info['host_spec']}:{remote_parent_path}"
    command.extend([str(project_dir), remote_target])

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        raise ValueError(f"SCP 上传失败: {stderr or stdout or exc}")


def _run_ssh_command(ssh_info: dict, remote_command: str):
    """执行远端 SSH 命令。"""
    command = ["ssh"]
    if ssh_info.get("proxy_jump"):
        command.extend(["-J", ssh_info["proxy_jump"]])
    if ssh_info.get("port"):
        command.extend(["-p", str(ssh_info["port"])])
    command.extend([ssh_info["host_spec"], remote_command])
    return subprocess.run(command, check=True, capture_output=True, text=True)


def _ensure_local_command(command_name: str):
    """检查本地命令是否存在。"""
    if shutil.which(command_name) is None:
        raise ValueError(f"本地未找到 `{command_name}` 命令，无法执行当前上传方式。")


def _create_project_tarball(project_dir: Path, output_path: Path):
    """创建项目 tar.gz 文件。"""
    include_patterns = [
        "config.py",
        "model.py",
        "*.py",
        "*.sh",
        "*.txt",
        "*.yaml",
        "*.yml",
        "data/",
    ]

    with tarfile.open(output_path, "w:gz") as tar:
        for pattern in include_patterns:
            if pattern.endswith("/"):
                dir_path = project_dir / pattern.rstrip("/")
                if dir_path.exists() and dir_path.is_dir():
                    tar.add(dir_path, arcname=pattern.rstrip("/"), recursive=True)
            else:
                for file in project_dir.glob(pattern):
                    if file.is_file():
                        tar.add(file, arcname=file.name)
