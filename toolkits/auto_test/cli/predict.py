"""启动模型推理任务"""

import json
import os
from pathlib import Path
from typing import Optional

try:
    from submit_job.cli.api_client import get_api_key, get_base_url, make_api_request
except ImportError:
    from .api_client import get_api_key, get_base_url, make_api_request


def predict_with_model(
    inference_name: str,
    model_path: str,
    input_data: dict,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    image_id: Optional[str] = None,
    resource_spec_id: Optional[str] = None,
    worker_num: int = 1,
    mount: Optional[list] = None,
    port: Optional[list] = None,
    **kwargs
) -> dict:
    """
    启动模型推理任务
    
    Args:
        inference_name: 推理服务名称
        model_path: 模型路径（在存储卷中的路径）
        input_data: 推理输入数据（符合模型输入格式的字典）
        api_key: API Key（如果为 None，则从环境变量获取）
        base_url: API 基础 URL（如果为 None，则从环境变量获取）
        image_id: 镜像 ID（如果为 None，则从环境变量获取）
        resource_spec_id: 资源规格 ID（如果为 None，则从环境变量获取）
        worker_num: Worker 数量（默认：1）
        mount: 挂载配置列表（可选）
        port: 端口配置列表（可选）
        **kwargs: 其他推理服务配置参数
    
    Returns:
        包含推理服务信息的字典
    """
    if api_key is None:
        api_key = get_api_key()
    
    if base_url is None:
        base_url = get_base_url()
    
    if image_id is None:
        image_id = os.getenv("INFINI_IMAGE_ID")
        if not image_id:
            raise ValueError(
                "未指定镜像 ID。请设置环境变量 INFINI_IMAGE_ID，"
                "或通过 --image-id 参数提供。"
            )
    
    if resource_spec_id is None:
        resource_spec_id = os.getenv("INFINI_RESOURCE_SPEC_ID")
        if not resource_spec_id:
            raise ValueError(
                "未指定资源规格 ID。请设置环境变量 INFINI_RESOURCE_SPEC_ID，"
                "或通过 --resource-spec-id 参数提供。"
            )
    
    # 构建入口点命令（根据模型路径启动推理服务）
    entry_point = f"python -m model.serve --model_path {model_path}"
    
    # 构建创建推理服务的请求
    request_data = {
        "inference_name": inference_name,
        "inference_description": f"推理服务: {inference_name}",
        "image_id": image_id,
        "resource_spec_id": resource_spec_id,
        "worker_num": worker_num,
        "entry_point": entry_point,
        "resource_type": kwargs.get("resource_type", 1),  # 1=Spot, 0=Reserved
        "is_preset_image": kwargs.get("is_preset_image", 1),
        "is_external_access": kwargs.get("is_external_access", 1),
        "external_access_type": kwargs.get("external_access_type", 0),
    }
    
    if mount:
        request_data["mount"] = mount
    
    if port:
        request_data["port"] = port
    else:
        # 默认端口配置
        request_data["port"] = [
            {
                "calling_port": 80,
                "listening_port": 8000
            }
        ]
    
    if "pool_id" in kwargs:
        request_data["pool_id"] = kwargs["pool_id"]
    
    # 移除 None 值
    request_data = {k: v for k, v in request_data.items() if v is not None}
    
    # 调用 API 创建推理服务
    result = make_api_request(
        endpoint="/api/platform/open/v1/inference/create",
        api_key=api_key,
        base_url=base_url,
        json_data=request_data
    )
    
    inference_id = result.get("inference_id")
    
    # 如果创建成功，启动推理服务
    if inference_id:
        start_result = make_api_request(
            endpoint="/api/platform/open/v1/inference/start",
            api_key=api_key,
            base_url=base_url,
            json_data={"inference_id": inference_id}
        )
        
        return {
            "ok": True,
            "message": "推理服务已创建并启动",
            "inference_id": inference_id,
            "input_data": input_data,
            "result": {
                "create": result,
                "start": start_result
            }
        }
    
    return {
        "ok": True,
        "message": "推理服务已创建",
        "inference_id": inference_id,
        "input_data": input_data,
        "result": result
    }
