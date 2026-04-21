"""API client tools module"""

import os
from typing import Optional
import requests


def get_api_key() -> str:
    """从环境变量获取 API Key"""
    api_key = os.getenv("INFINI_API_KEY")
    if not api_key:
        raise ValueError(
            "未设置 API Key。请设置环境变量 INFINI_API_KEY，"
            "或通过 --api-key 参数提供。"
        )
    return api_key

def get_account_id() -> str:
    """从环境变量获取账户 ID"""
    account_id = os.getenv("INFINI_ACCOUNT_ID")
    if not account_id:
        account_id = "ac-c6ufbjizmek2plv3"
        # TODO change to raise ValueError
        # raise ValueError(
        #     "未设置账户 ID。请设置环境变量 INFINI_ACCOUNT_ID，"
        #     "或通过 --account-id 参数提供。"
        # )
    return account_id

def get_base_url() -> str:
    """获取 API 基础 URL"""
    return os.getenv("INFINI_BASE_URL", "https://cloud.infini-ai.com")


def make_api_request(
    endpoint: str,
    method: str = "POST",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    json_data: Optional[dict] = None,
    **kwargs
) -> dict:
    """
    发送 API 请求
    
    Args:
        endpoint: API 端点路径（如 "/api/platform/open/v1/job/create"）
        method: HTTP 方法（默认 POST）
        api_key: API Key（如果为 None，则从环境变量获取）
        base_url: 基础 URL（如果为 None，则从环境变量获取）
        json_data: JSON 请求体
        **kwargs: 其他 requests 参数
    
    Returns:
        API 响应 JSON 数据
    """
    if api_key is None:
        api_key = get_api_key()
    
    if base_url is None:
        base_url = get_base_url()
    
    url = f"{base_url.rstrip('/')}{endpoint}"
    
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # 合并用户提供的 headers
    if "headers" in kwargs:
        headers.update(kwargs.pop("headers"))
    
    response = requests.request(
        method=method,
        url=url,
        headers=headers,
        json=json_data,
        **kwargs
    )
    
    # 检查 HTTP 状态码
    response.raise_for_status()
    
    # 解析 JSON 响应
    result = response.json()
    
    # 检查 API 业务错误（code != 0）
    if isinstance(result, dict) and "code" in result:
        code = result.get("code")
        if code != 0:
            # API 返回业务错误
            error_msg = result.get("msg") or result.get("message") or f"API 错误 (code: {code})"
            raise requests.exceptions.HTTPError(
                f"API 业务错误: {error_msg}",
                response=response
            )
    
    return result
