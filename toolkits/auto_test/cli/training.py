"""根据 config.py 启动训练任务"""

import os
import shlex
import sys
from pathlib import Path
from typing import Optional

import requests

if __package__ in {None, ""}:
    _cli_dir = Path(__file__).resolve().parent
    if str(_cli_dir) not in sys.path:
        sys.path.insert(0, str(_cli_dir))
    from api_client import make_api_request
    from config_mapping import apply_region_resource_config, get_full_config, normalize_resource_type
    from train_service_idle_resources import _run_local_pre_commands
else:
    from .api_client import make_api_request
    from .config_mapping import apply_region_resource_config, get_full_config, normalize_resource_type
    from .train_service_idle_resources import _run_local_pre_commands


def train_with_model(
    project_dir: Optional[Path] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    config_overrides: Optional[dict] = None
) -> dict:
    """
    根据 config.py 启动训练任务
    
    Args:
        project_dir: 项目目录路径（默认：当前目录）
        api_key: API Key（如果为 None，则从环境变量获取）
        base_url: API 基础 URL（如果为 None，则从环境变量获取）
        config_overrides: 配置覆盖参数（可选）
    
    Returns:
        包含任务信息的字典
    """
    if project_dir is None:
        project_dir = Path.cwd()
    else:
        project_dir = Path(project_dir).resolve()
    
    # 验证项目结构
    config_file = project_dir / "config.py"
    if not config_file.exists():
        raise FileNotFoundError(
            "缺少必需文件: config.py。请确保在项目目录中运行此命令。"
        )
    
    # 读取并执行 config.py
    config_globals = {}
    exec(config_file.read_text(encoding="utf-8"), config_globals)
    
    # 获取配置
    training_project = config_globals.get("training_project", {})
    
    # 应用配置覆盖
    if config_overrides:
        _deep_update(training_project, config_overrides)
    
    # 执行本地预命令（在创建任务前执行）
    local_pre_commands = config_globals.get("local_pre_commands", [])
    if local_pre_commands:
        _run_local_pre_commands(local_pre_commands, project_dir)
    
    # 检查是否使用任务类型映射
    task_type = training_project.get("task_type") or config_globals.get("task_type")
    
    if task_type:
        # 从任务类型映射中获取配置
        try:
            mapping_config = get_full_config(task_type)
            # 将映射配置合并到 training_project（映射配置优先级较低，可被显式配置覆盖）
            for key, value in mapping_config.items():
                if key not in training_project or training_project.get(key) is None:
                    training_project[key] = value
        except ValueError as e:
            raise ValueError(
                f"无法从任务类型 '{task_type}' 获取配置: {e}\n"
                f"请检查 config.py 中的 task_type 设置，或直接配置 image_id、framework_id 等字段。"
            )

    apply_region_resource_config(training_project)
    
    # 获取必要的配置值（优先级：显式配置 > 映射配置 > 环境变量）
    image_id = training_project.get("image_id") or os.getenv("INFINI_IMAGE_ID")
    framework_id = training_project.get("framework_id") or os.getenv("INFINI_FRAMEWORK_ID")
    resource_type = normalize_resource_type(training_project.get("resource_type", 1))
    
    # 根据 resource_type 获取对应的规格 ID
    # resource_type=0 (包年包月): 使用 load_spec_id
    # resource_type=1 (Spot): 使用 resource_spec_id
    if resource_type == 0:
        # 包年包月资源使用 load_spec_id
        load_spec_id = training_project.get("load_spec_id") or os.getenv("INFINI_LOAD_SPEC_ID")
        spec_id = load_spec_id
        spec_id_name = "load_spec_id"
    else:
        # Spot 资源使用 resource_spec_id
        resource_spec_id = training_project.get("resource_spec_id") or os.getenv("INFINI_RESOURCE_SPEC_ID")
        spec_id = resource_spec_id
        spec_id_name = "resource_spec_id"
    
    # 验证必需字段
    if not image_id:
        raise ValueError(
            "未指定镜像 ID。请选择以下方式之一：\n"
            "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
            "2. 在 config.py 中直接设置 image_id\n"
            "3. 设置环境变量 INFINI_IMAGE_ID"
        )
    if not framework_id:
        raise ValueError(
            "未指定框架 ID。请选择以下方式之一：\n"
            "1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
            "2. 在 config.py 中直接设置 framework_id\n"
            "3. 设置环境变量 INFINI_FRAMEWORK_ID"
        )
    if not spec_id:
        resource_type_name = "包年包月" if resource_type == 0 else "Spot"
        raise ValueError(
            f"未指定{'负载规格 ID (load_spec_id)' if resource_type == 0 else '资源规格 ID (resource_spec_id)'}。\n"
            f"当前资源类型: {resource_type_name} (resource_type={resource_type})\n\n"
            f"请选择以下方式之一：\n"
            f"1. 在 config.py 中设置 task_type（如：'预训练'、'强化学习训练'）\n"
            f"2. 在 config.py 中直接设置 {spec_id_name}\n"
            f"3. 设置环境变量 INFINI_{spec_id_name.upper()}"
        )
    
    # 构建创建任务的请求（参考 curl 指令格式）
    request_data = {
        "job_type": "train",
        "job_name": training_project.get("name", "Training Job"),
        "job_description": training_project.get("description", ""),
        "image_id": image_id,
        "framework_id": framework_id,
        "worker_num": training_project.get("compute", {}).get("node_count", 1),
        "entry_point": _build_entry_point(training_project),
        "resource_type": resource_type,
        "rdma_enable": training_project.get("rdma_enable", False),  # 根据 curl 指令添加
        "fault_tolerance": {
            "auto_restart": {
                "enable": True,
                "conditions": ["job_fail"],
                "max_retry": 1
            },
            "hang_check": {
                "enable": False,
                "timeout": 0
            },
            "environment_check": {
                "enable": True,
                "conditions": ["job_init", "job_fail"]
            }
        },
        "check_train_slow": 1,  # 根据 curl 指令添加
    }
    
    # 根据 resource_type 设置对应的规格 ID
    if resource_type == 0:
        # 包年包月资源使用 load_spec_id
        request_data["load_spec_id"] = spec_id
    else:
        # Spot 资源使用 resource_spec_id
        request_data["resource_spec_id"] = spec_id
    
    # 添加可选字段
    # pool_id 可能是资源池名称（如 "包年包月资源池-北京D"）或实际的 pool_id（如 "po-c7mnppogdw4ctvsj"）
    pool_id = training_project.get("pool_id")
    
    # 如果 pool_id 是资源池名称，尝试从 POOL_MAPPING 中获取实际的 pool_id
    if pool_id and pool_id.startswith("包年包月资源池-"):
        from submit_job.cli.config_mapping import get_pool_config
        try:
            pool_config = get_pool_config(pool_id)
            # 如果配置映射中有实际的 pool_id，使用它
            if "pool_id" in pool_config:
                pool_id = pool_config["pool_id"]
        except ValueError:
            # 如果找不到资源池配置，使用原始的 pool_id
            pass
    
    # 验证：使用包年包月资源时，必须提供 pool_id
    if resource_type == 0 and not pool_id:
        raise ValueError(
            "使用包年包月资源时，必须提供 pool_id。\n"
            f"当前配置：resource_type={resource_type}（reserved=0），但未设置 pool_id。\n\n"
            f"解决方案：\n"
            f"1. 在 config.py 中设置 task_type（会从配置映射自动获取 pool_id）\n"
            f"2. 在 config.py 中直接设置 pool_id\n"
            f"3. 或者使用 Spot 资源：设置 resource_type=\"spot\""
        )
    
    if resource_type == 0 and pool_id:
        request_data["pool_id"] = pool_id
    
    # 自动构建 mount 配置（根据任务类型从配置映射中获取 volume_id）
    mount = _build_mount_config(training_project)
    if mount:
        request_data["mount"] = mount
    
    # 移除 None 值
    request_data = {k: v for k, v in request_data.items() if v is not None}
    
    # 显示 mount 配置信息
    if "mount" in request_data:
        print("挂载配置 (mount):")
        for i, mount_item in enumerate(request_data["mount"], 1):
            print(f"  {i}. 路径: {mount_item.get('path', 'N/A')}")
            print(f"     存储卷 ID: {mount_item.get('volume_id', 'N/A')}")
            print(f"     读写权限: {mount_item.get('rw_setting', 'N/A')}")
        print()
    
    print("request_data: ", request_data)
    try:
        # 调用 API 创建任务
        result = make_api_request(
            endpoint="/api/platform/open/v1/job/create",
            api_key=api_key,
            base_url=base_url,
            json_data=request_data
        )
        
        # API 返回格式: {"code":0,"msg":"Success","data":{"job_id":"jo-xxx"}}
        # 或者: {"job_id":"jo-xxx"}
        job_id = None
        if isinstance(result, dict):
            # 检查是否有 data 字段（新格式）
            if "data" in result and isinstance(result["data"], dict):
                job_id = result["data"].get("job_id")
            else:
                # 直接获取 job_id（旧格式）
                job_id = result.get("job_id")
        
        if not job_id:
            # 检查是否有错误信息
            error_msg = result.get("msg") or result.get("message") or str(result)
            if result.get("code") != 0:
                raise ValueError(f"创建训练任务失败: {error_msg}")
            raise ValueError(f"创建训练任务失败: 未返回 job_id，响应: {result}")
        
        return {
            "ok": True,
            "message": "训练任务已创建",
            "job_id": job_id,
            "result": result
        }
        
    except requests.exceptions.HTTPError as e:
        # 处理 HTTP 错误响应
        error_msg = str(e)
        
        # 尝试从响应中获取详细错误信息
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_data = e.response.json()
                api_error_msg = error_data.get("msg") or error_data.get("message") or ""
                if api_error_msg:
                    error_msg = api_error_msg
            except (ValueError, KeyError):
                pass
        
        # 解析常见的错误情况
        if "Spot 规格已受限" in error_msg or ("Spot" in error_msg and "受限" in error_msg):
            raise ValueError(
                f"资源规格受限错误：\n"
                f"  当前使用的 Spot 规格 ({resource_spec_id}) 已受限，无法使用。\n\n"
                f"解决方案：\n"
                f"1. 使用其他 Spot 规格：在 config.py 中修改 resource_spec_id 为其他可用的资源规格 ID\n"
                f"2. 使用包年包月资源：在 config.py 中设置 resource_type=\"reserved\" 并提供 pool_id\n"
                f"3. 查询可用资源规格：使用平台 API 查询可用的资源规格列表\n\n"
                f"当前配置：\n"
                f"  - 资源类型: {'Spot' if request_data.get('resource_type') == 1 else '包年包月'}\n"
                f"  - 资源规格 ID: {resource_spec_id}\n"
                f"  - 镜像 ID: {image_id}\n"
                f"  - 框架 ID: {framework_id}"
            )
        elif "resource_spec" in error_msg.lower() or "资源规格" in error_msg:
            raise ValueError(
                f"资源规格错误：\n"
                f"  {error_msg}\n\n"
                f"请检查：\n"
                f"1. 资源规格 ID ({resource_spec_id}) 是否正确\n"
                f"2. 资源规格是否可用\n"
                f"3. 资源类型 (resource_type) 是否匹配\n"
                f"4. 如果使用包年包月资源，是否提供了 pool_id"
            )
        else:
            # 其他错误
            raise ValueError(f"创建训练任务失败: {error_msg}")
    except ValueError:
        # 重新抛出 ValueError（通常是我们的错误处理）
        raise
    except Exception as e:
        # 其他未预期的错误
        raise ValueError(f"创建训练任务时发生错误: {e}")


def _deep_update(base_dict: dict, update_dict: dict):
    """深度更新字典"""
    for key, value in update_dict.items():
        if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value


def _build_entry_point(training_project: dict) -> str:
    """构建入口点命令"""
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

    # 默认命令：运行 model.py train
    command_parts.append("python model.py train")
    return " && ".join(command_parts)


def _get_resource_spec_id(training_project: dict) -> str:
    """获取资源规格 ID"""
    # 从配置中获取资源规格 ID
    resource_spec_id = training_project.get("resource_spec_id", "")
    if not resource_spec_id:
        # 如果配置中没有，从环境变量获取
        resource_spec_id = os.getenv("INFINI_RESOURCE_SPEC_ID", "")
    return resource_spec_id


def _build_mount_config(training_project: dict) -> list:
    """
    构建挂载配置
    
    根据任务类型从配置映射中自动获取 volume_id 和 mount 配置。
    如果配置中有 volume_id，则自动创建 mount 配置。
    优先使用配置映射中的 mount 配置（包括 path 和 rw_setting）。
    
    Args:
        training_project: 训练项目配置字典（已包含从配置映射中获取的 volume_id 和 mount）
    
    Returns:
        mount 配置列表，如果无法自动配置则返回空列表
    """
    # 如果配置映射中已经有完整的 mount 配置，直接使用
    mount_config = training_project.get("mount")
    if mount_config and isinstance(mount_config, dict):
        # 如果 mount 是字典格式，转换为列表格式
        volume_id = mount_config.get("volume_id") or training_project.get("volume_id")
        if volume_id:
            return [
                {
                    "path": mount_config.get("path", "/mnt/volume"),
                    "volume_id": volume_id,
                    "rw_setting": mount_config.get("rw_setting", "can_write")
                }
            ]
    
    # 如果配置映射中没有 mount 配置，但从配置映射中获取了 volume_id，则使用默认配置
    volume_id = training_project.get("volume_id")
    if volume_id:
        # 检查是否有 mount 配置中的 path 和 rw_setting
        mount_path = training_project.get("mount", {}).get("path") if isinstance(training_project.get("mount"), dict) else None
        mount_rw_setting = training_project.get("mount", {}).get("rw_setting") if isinstance(training_project.get("mount"), dict) else None
        
        # 自动构建 mount 配置
        return [
            {
                "path": mount_path or "/mnt/volume",  # 使用配置映射中的 path 或默认路径
                "volume_id": volume_id,
                "rw_setting": mount_rw_setting or "can_write"  # 使用配置映射中的 rw_setting 或默认可写
            }
        ]
    
    return []


