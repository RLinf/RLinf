"""使用闲时资源创建训练服务任务。"""

import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    _cli_dir = Path(__file__).resolve().parent
    import sys

    if str(_cli_dir) not in sys.path:
        sys.path.insert(0, str(_cli_dir))
    from api_client import make_api_request
    from config_mapping import apply_region_resource_config
else:
    from .api_client import make_api_request
    from .config_mapping import apply_region_resource_config


def create_idle_resource_train_service(
    project_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    config_overrides: Optional[dict] = None,
) -> dict:
    """根据 config.py 创建闲时资源训练服务任务。"""
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir).resolve()

    config_file = project_dir / "config.py"
    if not config_file.exists():
        raise FileNotFoundError(
            "缺少必需文件: config.py。请确保在项目目录中运行此命令。"
        )

    config_globals: Dict[str, Any] = {}
    exec(config_file.read_text(encoding="utf-8"), config_globals)

    training_project = config_globals.get("training_project", {})
    if config_overrides:
        _deep_update(training_project, config_overrides)

    # 执行本地预命令（在创建任务前执行）
    local_pre_commands = config_globals.get("local_pre_commands", [])
    if local_pre_commands:
        _run_local_pre_commands(local_pre_commands, project_dir)

    apply_region_resource_config(training_project)

    region_id = training_project.get("region_id") or os.getenv("INFINI_REGION_ID")
    resource_spec_id = (
        training_project.get("resource_spec_id")
        or os.getenv("INFINI_RESOURCE_SPEC_ID")
    )
    image_id = training_project.get("image_id") or os.getenv("INFINI_IMAGE_ID")
    framework_id = (
        training_project.get("framework_id") or os.getenv("INFINI_FRAMEWORK_ID")
    )
    worker_num = training_project.get("compute", {}).get("node_count", 1)
    expect_train_complete_time = training_project.get(
        "expect_train_complete_time", 3600
    )
    shared_mem = training_project.get("shared_mem", 1)
    rdma_enable = training_project.get("rdma_enable", False)
    mount = _build_mount_config(training_project)

    if not region_id:
        raise ValueError("未指定 region_id。请在 config.py 中设置 resource_region 或 region_id。")
    if not resource_spec_id:
        raise ValueError(
            "未指定 resource_spec_id。请在 config.py 中设置 resource_spec_id，"
            "或通过 resource_region + resource_gpu_num 自动映射。"
        )
    if not image_id:
        raise ValueError("未指定 image_id。请在 config.py 中设置 image_id。")
    if not framework_id:
        raise ValueError("未指定 framework_id。请在 config.py 中设置 framework_id。")

    plan_payload = {
        "region_id": region_id,
        "resource_spec_id": resource_spec_id,
        "worker_num": worker_num,
        "expect_train_complete_time": expect_train_complete_time,
    }
    plan_result = make_api_request(
        endpoint="/api/platform/open/v1/train_plan/idle_resources/generate",
        api_key=api_key,
        base_url=base_url,
        json_data=plan_payload,
    )

    plan_data = plan_result.get("data", {})
    plan_list = plan_data.get("train_plan_list") or []
    if not plan_list:
        raise ValueError(f"未生成可用训练方案: {plan_result}")

    create_payload = {
        "region_id": region_id,
        "resource_spec_id": resource_spec_id,
        "job_name": training_project.get("name", "training_server_test"),
        "job_description": training_project.get("description", ""),
        "image_id": image_id,
        "framework_id": framework_id,
        "worker_num": worker_num,
        "entry_point": _build_entry_point(training_project),
        "shared_mem": shared_mem,
        "expect_train_complete_time": expect_train_complete_time,
        "train_plan_pre_execution_id": plan_data.get("train_plan_pre_execution_id"),
        "train_plan_id": plan_list[0].get("train_plan_id"),
        "rdma_enable": rdma_enable,
    }
    if mount:
        create_payload["mount"] = mount

    create_result = make_api_request(
        endpoint="/api/platform/open/v1/train_service/idle_resources/create",
        api_key=api_key,
        base_url=base_url,
        json_data=create_payload,
    )

    job_id = _extract_result_value(create_result, "job_id")
    if not job_id:
        raise ValueError(f"创建闲时训练服务任务失败: {create_result}")

    return {
        "ok": True,
        "message": "闲时训练服务任务已创建",
        "job_id": job_id,
        "plan_result": plan_result,
        "result": create_result,
    }


def _deep_update(base_dict: dict, update_dict: dict):
    """深度更新字典。"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def _build_entry_point(training_project: dict) -> str:
    """构建入口点命令。"""
    runtime = training_project.get("runtime", {})
    environment_variables = runtime.get("environment_variables", {})
    start_commands = runtime.get("start_commands", [])

    command_parts = []
    if isinstance(environment_variables, dict):
        for key, value in environment_variables.items():
            command_parts.append(f"export {key}={shlex.quote(str(value))}")

    if start_commands:
        command_parts.extend(start_commands)
        return " && ".join(command_parts)

    return "python model.py train"


def _build_mount_config(training_project: dict) -> list:
    """构建挂载配置。"""
    mount_config = training_project.get("mount")
    if isinstance(mount_config, list):
        return [item.copy() for item in mount_config if isinstance(item, dict)]

    if isinstance(mount_config, dict):
        return [mount_config.copy()]

    volume_id = training_project.get("volume_id")
    if volume_id:
        return [
            {
                "path": "/mnt/public",
                "volume_id": volume_id,
                "rw_setting": "can_write",
            }
        ]

    return []


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


def _run_local_pre_commands(commands: List[str], project_dir: Path) -> None:
    """
    在本地执行预命令（在创建任务前执行）。
    
    所有命令在一个 shell 中串行执行，使用 && 连接，
    这样 cd 命令可以正确影响后续命令的工作目录。
    
    Args:
        commands: 要执行的命令列表
        project_dir: 项目目录路径（作为工作目录）
    
    Raises:
        RuntimeError: 如果命令执行失败
    """
    if not commands:
        return
    
    print(f"\n执行本地预命令 (共 {len(commands)} 条):")
    for i, cmd in enumerate(commands, 1):
        print(f"  [{i}/{len(commands)}] {cmd}")
    print("-" * 50)
    
    # 将所有命令用 && 连接，在一个 shell 中串行执行
    combined_command = " && ".join(commands)
    
    try:
        # 使用 Popen 实时打印输出
        process = subprocess.Popen(
            combined_command,
            shell=True,
            cwd=str(project_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 将 stderr 合并到 stdout
            text=True,
            bufsize=1,  # 行缓冲
            universal_newlines=True
        )
        
        # 实时读取并打印输出
        for line in process.stdout:
            print(line, end='')
        
        # 等待进程完成
        return_code = process.wait()
        
        # 检查返回码
        if return_code != 0:
            print(f"\n✗ 命令执行失败，返回码: {return_code}")
            raise RuntimeError(f"本地预命令执行失败，返回码: {return_code}")
        
        print("-" * 50)
        print("✓ 所有本地预命令执行完成\n")
        
    except subprocess.SubprocessError as e:
        raise RuntimeError(f"本地预命令执行异常: {e}")
