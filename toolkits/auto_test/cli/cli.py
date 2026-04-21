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

# !/usr/bin/env python3
"""
Submit Job CLI - 类似 Truss 的训练任务提交工具

usage:
    submit_job push                    # 上传项目文件到存储卷
    submit_job list                    # 列出所有可用的示例任务
    submit_job volumes                 # 列出可用的存储卷列表
    submit_job run <project_dir>        # 根据 config.py 启动任务（训练或推理）
    submit_job close <job_id>           # 关闭/删除任务
    submit_job remove <job_id>          # 删除任务
    submit_job get_task_list           # 查询任务列表
    submit_job get_job_info <job_id>   # 获取任务日志
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

_submit_job_dir = Path(__file__).resolve().parent
_parent_dir = _submit_job_dir.parent

if __package__ in {None, ""}:
    if str(_submit_job_dir) not in sys.path:
        sys.path.insert(0, str(_submit_job_dir))
    from push import push_training_job
    from train_service_idle_resources import create_idle_resource_train_service
    from training import train_with_model
else:
    if str(_parent_dir) not in sys.path:
        sys.path.insert(0, str(_parent_dir))
    from .push import push_training_job
    from .train_service_idle_resources import create_idle_resource_train_service
    from .training import train_with_model


def get_examples_dir() -> Path:
    """获取 examples 目录路径"""
    # 尝试从多个可能的位置找到 examples 目录
    possible_paths = [
        _submit_job_dir.parent / "examples",
        Path(__file__).parent.parent / "examples",
        Path.cwd() / "examples",
    ]

    for path in possible_paths:
        if path.exists() and path.is_dir():
            return path

    # 如果都找不到，返回相对路径
    return _submit_job_dir.parent / "examples"


def list_examples() -> list:
    """列出所有可用的示例任务"""
    examples_dir = get_examples_dir()
    if not examples_dir.exists():
        return []

    examples = []
    for item in examples_dir.iterdir():
        if item.is_dir():
            config_file = item / "config.py"
            if config_file.exists():
                examples.append(item.name)

    return sorted(examples)


def main():
    parser = argparse.ArgumentParser(
        description="Submit Job - 训练任务提交工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # push 命令
    push_parser = subparsers.add_parser("push", help="上传项目文件到平台")
    push_parser.add_argument(
        "project_dir",
        type=str,
        nargs="?",
        default=None,
        help="项目目录路径或示例任务名称（如: pretrain_test）。如果未指定，默认为当前目录",
    )
    push_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    push_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )
    push_parser.add_argument(
        "--volume-id",
        type=str,
        help="存储卷 ID (默认: 从环境变量 INFINI_VOLUME_ID 获取)",
    )
    push_parser.add_argument(
        "--target-path", type=str, help="目标路径 (默认: 项目名称)"
    )
    push_parser.add_argument(
        "--image-id",
        type=str,
        help="镜像 ID (默认: 从 config.py 或环境变量 INFINI_IMAGE_ID 获取)",
    )
    push_parser.add_argument(
        "--framework-id",
        type=str,
        help="框架 ID (默认: 从 config.py 或环境变量 INFINI_FRAMEWORK_ID 获取)",
    )
    push_parser.add_argument(
        "--resource-spec-id",
        type=str,
        help="资源规格 ID (默认: 从 config.py 或环境变量 INFINI_RESOURCE_SPEC_ID 获取)",
    )
    push_parser.add_argument(
        "--upload-method",
        type=str,
        choices=["job", "aicoder_scp"],
        default="job",
        help="上传方式：job=临时任务写卷，aicoder_scp=创建 AICoder 后通过 SCP 上传",
    )
    push_parser.add_argument(
        "--upload-preset",
        type=str,
        help="上传预设（如: beijingD, ningxiaB）。设置后会自动补齐 aicoder_scp、volume-id、region-id、aicoder-mount-path",
    )
    push_parser.add_argument(
        "--region-id",
        type=str,
        help="AICoder 可用区 ID（仅 upload-method=aicoder_scp 时使用）",
    )
    push_parser.add_argument(
        "--aicoder-mount-path",
        type=str,
        help="AICoder 挂载路径（默认: 环境变量 INFINI_AICODER_MOUNT_PATH 或 /mnt/public）",
    )
    push_parser.add_argument(
        "--aicoder-mount-rw-setting",
        type=str,
        default="can_write",
        help="AICoder 挂载读写权限（默认: can_write）",
    )
    push_parser.add_argument(
        "--ssh-proxy-jump",
        type=str,
        help="AICoder SCP 跳板机（默认: 环境变量 INFINI_AICODER_SSH_PROXY_JUMP 或 ssh-jumper.cloud.infini-ai.com）",
    )
    push_parser.add_argument(
        "--ssh-user",
        type=str,
        help="AICoder SSH 用户（默认: 环境变量 INFINI_AICODER_SSH_USER 或 root）",
    )

    # list 命令 - 列出所有可用的示例任务
    subparsers.add_parser("list", help="列出所有可用的示例任务")

    # volumes 命令 - 列出可用的存储卷
    volumes_parser = subparsers.add_parser("volumes", help="列出可用的存储卷列表")
    volumes_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    volumes_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )

    # run 命令 - 根据 config.py 启动任务（训练或推理）
    run_parser = subparsers.add_parser(
        "run", help="根据 config.py 启动任务（根据 task_type 自动判断是训练还是推理）"
    )
    run_parser.add_argument(
        "project_dir",
        type=str,
        nargs="?",
        default=None,
        help="项目目录路径或示例任务名称（如: pretrain_test, sft_test, rl_test）。如果未指定，默认为当前目录",
    )
    run_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    run_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )

    # close 命令 - 关闭/删除任务
    close_parser = subparsers.add_parser("close", help="关闭/删除任务")
    close_parser.add_argument("job_id", type=str, help="任务 ID")
    close_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    close_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )

    # remove 命令 - 删除任务（与 close 功能相同）
    remove_parser = subparsers.add_parser("remove", help="删除任务")
    remove_parser.add_argument("job_id", type=str, help="任务 ID")
    remove_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    remove_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )

    # get_task_list 命令 - 查询任务列表
    get_task_list_parser = subparsers.add_parser("get_task_list", help="查询任务列表")
    get_task_list_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    get_task_list_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )
    get_task_list_parser.add_argument(
        "--offset", type=int, default=0, help="偏移量 (默认: 0)"
    )
    get_task_list_parser.add_argument(
        "--limit", type=int, default=10, help="每页数量 (默认: 10)"
    )
    get_task_list_parser.add_argument("--status", type=int, help="任务状态筛选（可选）")
    get_task_list_parser.add_argument(
        "--return-count", action="store_true", help="返回总数"
    )
    get_task_list_parser.add_argument(
        "--account-id",
        type=str,
        help="账户 ID (默认: 从环境变量 INFINI_ACCOUNT_ID 获取)",
    )

    # get_job_info 命令 - 获取任务日志
    get_job_info_parser = subparsers.add_parser("get_job_info", help="获取任务日志")
    get_job_info_parser.add_argument("job_id", type=str, help="任务 ID")
    get_job_info_parser.add_argument(
        "--api-key", type=str, help="API Key (默认: 从环境变量 INFINI_API_KEY 获取)"
    )
    get_job_info_parser.add_argument(
        "--base-url",
        type=str,
        help="API 基础 URL (默认: 从环境变量 INFINI_BASE_URL 获取或 https://cloud.infini-ai.com)",
    )
    get_job_info_parser.add_argument(
        "--worker-id",
        type=str,
        default="",
        help="Worker ID (默认: 空字符串，表示所有 Worker)",
    )
    get_job_info_parser.add_argument("--search-value", type=str, help="搜索关键词")
    get_job_info_parser.add_argument(
        "--search-type",
        type=str,
        default="fulltext",
        choices=["fulltext", "regex"],
        help="搜索类型: fulltext 或 regex (默认: fulltext)",
    )
    get_job_info_parser.add_argument(
        "--log-num", type=int, default=1000, help="日志数量 (默认: 1000)"
    )
    get_job_info_parser.add_argument(
        "--from", type=int, default=0, dest="from_offset", help="起始位置 (默认: 0)"
    )
    get_job_info_parser.add_argument(
        "--order",
        type=str,
        default="asc",
        choices=["asc", "desc"],
        help="排序方式: asc 或 desc (默认: asc)",
    )
    get_job_info_parser.add_argument(
        "--start-time", type=int, help="开始时间（微秒时间戳）"
    )
    get_job_info_parser.add_argument(
        "--end-time", type=int, help="结束时间（微秒时间戳）"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        # 导入 API 客户端
        try:
            from submit_job.cli.api_client import (
                get_account_id,
                get_api_key,
                get_base_url,
                make_api_request,
            )
        except ImportError:
            from api_client import (
                get_account_id,
                get_api_key,
                get_base_url,
                make_api_request,
            )

        if args.command == "push":
            # 确定项目目录
            project_dir = None
            if args.project_dir:
                # 如果提供了路径，先尝试作为示例名称
                examples_dir = get_examples_dir()
                example_dir = examples_dir / args.project_dir
                if example_dir.exists() and (example_dir / "config.py").exists():
                    project_dir = example_dir
                    print(f"✓ 找到示例任务: {args.project_dir}")
                else:
                    # 尝试作为项目目录路径
                    project_dir = Path(args.project_dir).resolve()
                    if not project_dir.exists():
                        available = list_examples()
                        print(
                            f"✗ 项目目录或示例任务 '{args.project_dir}' 不存在",
                            file=sys.stderr,
                        )
                        if available:
                            print("\n可用的示例任务:", file=sys.stderr)
                            for ex in available:
                                print(f"  - {ex}", file=sys.stderr)
                        sys.exit(1)
            else:
                # 如果未提供，默认为当前目录
                project_dir = Path.cwd()

            print(f"✓ 项目目录: {project_dir}")
            print()

            result = push_training_job(
                project_dir=project_dir,
                api_key=args.api_key,
                base_url=args.base_url,
                volume_id=args.volume_id,
                target_path=args.target_path,
                image_id=args.image_id,
                framework_id=args.framework_id,
                resource_spec_id=args.resource_spec_id,
                upload_method=args.upload_method,
                upload_preset=args.upload_preset,
                region_id=args.region_id,
                aicoder_mount_path=args.aicoder_mount_path,
                aicoder_mount_rw_setting=args.aicoder_mount_rw_setting,
                ssh_proxy_jump=args.ssh_proxy_jump,
                ssh_user=args.ssh_user,
            )
            if result.get("ok"):
                print("✓ 上传完成")
                print(f"  上传方式: {result.get('upload_method', 'N/A')}")
                if result.get("job_id"):
                    print(f"  任务 ID: {result.get('job_id', 'N/A')}")
                if result.get("job_name"):
                    print(f"  任务名称: {result.get('job_name', 'N/A')}")
                if result.get("ai_coder_id"):
                    print(f"  AICoder ID: {result.get('ai_coder_id', 'N/A')}")
                print(f"  存储卷 ID: {result.get('volume_id', 'N/A')}")
                print(f"  目标路径: {result.get('target_path', 'N/A')}")
                if result.get("target_file"):
                    print(f"  目标文件: {result.get('target_file', 'N/A')}")
                if result.get("remote_path"):
                    print(f"  远端目录: {result.get('remote_path', 'N/A')}")
                if result.get("ssh_host"):
                    print(f"  SSH 主机: {result.get('ssh_host', 'N/A')}")
                if result.get("message"):
                    print(f"  提示: {result['message']}")
                if result.get("upload_method") == "job":
                    print("\n注意: 文件上传任务已创建，请等待任务完成。")
                    print(f"您可以通过任务 ID {result.get('job_id')} 查询任务状态。")
                else:
                    print("\n注意: AICoder 实例已创建，当前逻辑不会自动回收该实例。")
            else:
                print(f"✗ 上传失败: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        elif args.command == "list":
            examples = list_examples()
            if not examples:
                print("✗ 未找到可用的示例任务")
                print(f"  请确保 examples 目录存在: {get_examples_dir()}")
                sys.exit(1)

            print("可用的示例任务:")
            print()
            for i, example in enumerate(examples, 1):
                example_dir = get_examples_dir() / example
                config_file = example_dir / "config.py"
                description = example

                # 尝试读取配置获取描述
                try:
                    config_globals = {}
                    exec(config_file.read_text(encoding="utf-8"), config_globals)
                    training_project = config_globals.get("training_project", {})
                    description = training_project.get("description", example)
                except Exception:
                    pass

                print(f"  {i}. {example}")
                print(f"     {description}")
                print()

            print("使用方法:")
            print("  submit_job run <example_name>")
            print("  或")
            print("  submit_job run <project_dir>")
            print()
            print("示例:")
            for example in examples[:3]:  # 只显示前3个
                print(f"  submit_job run {example}")
            print("  submit_job run .  # 当前目录")

        elif args.command == "volumes":
            # 列出可用的存储卷
            try:
                result = make_api_request(
                    endpoint="/api/platform/open/v1/volume/list",
                    method="GET",
                    api_key=args.api_key,
                    base_url=args.base_url,
                )

                # 解析响应
                volumes = []
                if isinstance(result, dict):
                    if "data" in result:
                        volumes = result["data"]
                    elif isinstance(result, list):
                        volumes = result

                if not volumes:
                    print("未找到可用的存储卷")
                    return

                print("可用的存储卷列表:")
                print()
                for i, volume in enumerate(volumes, 1):
                    volume_id = volume.get("volume_id") or volume.get("id") or "N/A"
                    volume_name = (
                        volume.get("volume_name") or volume.get("name") or "N/A"
                    )
                    volume_type = (
                        volume.get("volume_type") or volume.get("type") or "N/A"
                    )
                    size = volume.get("size") or volume.get("capacity") or "N/A"

                    print(f"  {i}. 存储卷 ID: {volume_id}")
                    print(f"     名称: {volume_name}")
                    print(f"     类型: {volume_type}")
                    print(f"     大小: {size}")
                    print()

                print("使用方法:")
                print(
                    "  在 config.py 中设置 task_type，系统会自动从配置映射中获取 volume_id"
                )
                print("  或者手动在 config.py 中设置 volume_id")
                print()
                print("mount 配置格式:")
                print("  mount = [")
                print("      {")
                print('          "path": "/mnt/volume",  # 挂载路径')
                print('          "volume_id": "vo-xxxxxxxxxxxxx",  # 存储卷 ID')
                print('          "rw_setting": "can_write"  # can_write 或 only_read')
                print("      }")
                print("  ]")

            except Exception as e:
                print(f"✗ 获取存储卷列表失败: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "run":
            # 确定项目目录
            project_dir = None
            if args.project_dir:
                # 如果提供了路径，先尝试作为示例名称
                examples_dir = get_examples_dir()
                example_dir = examples_dir / args.project_dir
                if example_dir.exists() and (example_dir / "config.py").exists():
                    project_dir = example_dir
                    print(f"✓ 找到示例任务: {args.project_dir}")
                else:
                    # 尝试作为项目目录路径
                    project_dir = Path(args.project_dir).resolve()
                    if not project_dir.exists():
                        available = list_examples()
                        print(
                            f"✗ 项目目录或示例任务 '{args.project_dir}' 不存在",
                            file=sys.stderr,
                        )
                        if available:
                            print("\n可用的示例任务:", file=sys.stderr)
                            for ex in available:
                                print(f"  - {ex}", file=sys.stderr)
                        sys.exit(1)
            else:
                # 如果未提供，默认为当前目录
                project_dir = Path.cwd()

            # 验证 config.py 存在
            config_file = project_dir / "config.py"
            if not config_file.exists():
                print(f"✗ 缺少 config.py 文件: {config_file}", file=sys.stderr)
                sys.exit(1)

            print(f"✓ 项目目录: {project_dir}")

            # 读取 config.py 获取 task_type
            config_globals = {}
            exec(config_file.read_text(encoding="utf-8"), config_globals)
            training_project = config_globals.get("training_project", {})
            task_type = training_project.get("task_type") or config_globals.get(
                "task_type"
            )
            job_api = training_project.get("job_api") or config_globals.get("job_api")

            if task_type:
                print(f"  任务类型: {task_type}")
            if job_api:
                print(f"  接口类型: {job_api}")
            print()

            if job_api == "train_service_idle_resources":
                result = create_idle_resource_train_service(
                    project_dir=project_dir,
                    api_key=args.api_key,
                    base_url=args.base_url,
                )
            else:
                # 根据 task_type 启动任务（目前所有任务类型都是训练任务）
                # 未来可以根据 task_type 判断是训练还是推理
                result = train_with_model(
                    project_dir=project_dir,
                    api_key=args.api_key,
                    base_url=args.base_url,
                )

            if result.get("ok"):
                print("✓ 训练任务已创建")
                print(f"  任务 ID: {result.get('job_id', 'N/A')}")
                if result.get("message"):
                    print(f"  提示: {result['message']}")
            else:
                print(f"✗ 创建训练任务失败: {result.get('error', 'Unknown error')}")
                sys.exit(1)

        elif args.command == "close":
            # 关闭/删除任务
            api_key = args.api_key or get_api_key()
            base_url = args.base_url or get_base_url()

            print(f"正在关闭任务: {args.job_id}")

            try:
                result = make_api_request(
                    endpoint="/api/platform/open/v1/job/delete",
                    method="POST",
                    api_key=api_key,
                    base_url=base_url,
                    json_data={"job_id": args.job_id},
                )

                if result.get("code") == 0:
                    print(f"✓ 任务 {args.job_id} 已成功关闭")
                else:
                    error_msg = result.get("msg") or result.get("message") or "未知错误"
                    print(f"✗ 关闭任务失败: {error_msg}", file=sys.stderr)
                    sys.exit(1)

            except Exception as e:
                print(f"✗ 关闭任务失败: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "remove":
            # 删除任务（与 close 功能相同）
            api_key = args.api_key or get_api_key()
            base_url = args.base_url or get_base_url()

            print(f"正在删除任务: {args.job_id}")

            try:
                result = make_api_request(
                    endpoint="/api/platform/open/v1/job/delete",
                    method="POST",
                    api_key=api_key,
                    base_url=base_url,
                    json_data={"job_id": args.job_id},
                )

                if result.get("code") == 0:
                    print(f"✓ 任务 {args.job_id} 已成功删除")
                else:
                    error_msg = result.get("msg") or result.get("message") or "未知错误"
                    print(f"✗ 删除任务失败: {error_msg}", file=sys.stderr)
                    sys.exit(1)

            except Exception as e:
                print(f"✗ 删除任务失败: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "get_task_list":
            # 查询任务列表
            api_key = args.api_key or get_api_key()
            base_url = args.base_url or get_base_url()

            # 获取 account_id
            account_id = args.account_id or get_account_id()
            if not account_id:
                print(
                    "✗ 未指定账户 ID。请使用 --account-id 参数或设置环境变量 INFINI_ACCOUNT_ID",
                    file=sys.stderr,
                )
                sys.exit(1)

            print("正在查询任务列表...")

            # 构建请求数据
            request_data = {
                "offset": args.offset,
                "limit": args.limit,
                "account_id_list": [account_id],
                "return_count": args.return_count,
            }

            # 添加可选参数
            if args.status is not None:
                request_data["status"] = args.status

            # 默认排序
            request_data["order_by"] = ["-job.created_at"]

            try:
                result = make_api_request(
                    endpoint="/api/platform/open/v1/job/list",
                    method="POST",
                    api_key=api_key,
                    base_url=base_url,
                    json_data=request_data,
                )

                if result.get("code") == 0:
                    data = result.get("data", {})
                    # API 返回格式: data.job_info_list 而不是 data.items
                    items = data.get("job_info_list", [])
                    total = data.get("result_total", len(items))

                    print(f"✓ 找到 {len(items)} 个任务 (总计: {total})")
                    print()

                    if items:
                        for i, job in enumerate(items, 1):
                            job_id = job.get("job_id") or job.get("id") or "N/A"
                            job_name = job.get("job_name") or job.get("name") or "N/A"
                            job_type = job.get("job_type") or "N/A"
                            status = job.get("status") or "N/A"
                            created_at = job.get("created_at") or "N/A"

                            print(f"  {i}. 任务 ID: {job_id}")
                            print(f"     名称: {job_name}")
                            print(f"     类型: {job_type}")
                            print(f"     状态: {status}")
                            print(f"     创建时间: {created_at}")
                            print()
                    else:
                        print("  未找到任务")
                else:
                    error_msg = result.get("msg") or result.get("message") or "未知错误"
                    print(f"✗ 查询任务列表失败: {error_msg}", file=sys.stderr)
                    sys.exit(1)

            except Exception as e:
                print(f"✗ 查询任务列表失败: {e}", file=sys.stderr)
                sys.exit(1)

        elif args.command == "get_job_info":
            # 获取任务日志
            api_key = args.api_key or get_api_key()
            base_url = args.base_url or get_base_url()

            print(f"正在获取任务日志: {args.job_id}")

            # 如果没有提供 start_time 和 end_time，自动设置
            # EndTime 为当前时间，StartTime 为当前时间向前推7天（确保能获取到所有日志）
            if not args.start_time and not args.end_time:
                now = datetime.now()
                end_time = int(now.timestamp() * 1_000_000)  # 转换为微秒时间戳
                start_time = int(
                    (now - timedelta(days=7)).timestamp() * 1_000_000
                )  # 向前推7天
                args.end_time = end_time
                args.start_time = start_time
                # 显示查询的时间范围
                start_dt = datetime.fromtimestamp(start_time / 1_000_000)
                end_dt = datetime.fromtimestamp(end_time / 1_000_000)
                print(start_time, end_time)
                print(
                    f"查询时间范围: {start_dt.strftime('%Y-%m-%d %H:%M:%S')} 至 {end_dt.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            # 构建请求数据
            request_data = {
                "job_id": args.job_id,
                "worker_id": args.worker_id,
                "log_num": args.log_num,
                "from": args.from_offset,
                "order": args.order,
            }

            # 添加可选参数
            if args.search_value:
                request_data["search_value"] = args.search_value
                request_data["search_type"] = args.search_type

            if args.start_time:
                request_data["start_time"] = args.start_time

            if args.end_time:
                request_data["end_time"] = args.end_time

            try:
                result = make_api_request(
                    endpoint="/api/platform/open/v1/job/log/get",
                    method="POST",
                    api_key=api_key,
                    base_url=base_url,
                    json_data=request_data,
                )

                if result.get("code") == 0:
                    data = result.get("data", {})
                    # API 返回格式: data.log_list 而不是 data.logs
                    logs = data.get("log_list", [])
                    total = data.get("total", len(logs))

                    print(f"✓ 找到 {len(logs)} 条日志 (总计: {total})")
                    print()

                    if logs:
                        for log in logs:
                            # 日志内容在 body 字段中
                            log_content = log.get("body", "")
                            pod_name = log.get("pod_name", "N/A")

                            # 简化格式：worker_id: log_text（单行）
                            print(f"{pod_name}: {log_content}")
                    else:
                        print("  未找到日志")
                else:
                    error_msg = result.get("msg") or result.get("message") or "未知错误"
                    print(f"✗ 获取任务日志失败: {error_msg}", file=sys.stderr)
                    sys.exit(1)

            except Exception as e:
                print(f"✗ 获取任务日志失败: {e}", file=sys.stderr)
                sys.exit(1)

    except ValueError as e:
        # ValueError 通常是用户输入错误或 API 不可用，只显示错误消息
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        # 其他异常显示完整 traceback 用于调试
        print(f"✗ 错误: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
