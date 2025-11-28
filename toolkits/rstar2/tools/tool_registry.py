# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import importlib
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from omegaconf import OmegaConf, DictConfig
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ToolType(Enum):
    """工具类型枚举"""
    NATIVE = "native"
    MCP = "mcp"


class OpenAIFunctionToolSchema(BaseModel):
    """OpenAI函数工具模式"""
    type: str = "function"
    function: Dict[str, Any]


def get_tool_class(cls_name: str):
    """
    根据类名动态导入工具类
    
    Args:
        cls_name: 完整的类名，如 "toolkits.tools.code_judge_tool.PythonTool"
        
    Returns:
        工具类
    """
    try:
        module_name, class_name = cls_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        tool_cls = getattr(module, class_name)
        return tool_cls
    except Exception as e:
        logger.error(f"Failed to import tool class {cls_name}: {e}")
        raise


async def initialize_native_tool(tool_cls, tool_config) -> Any:
    """
    初始化原生工具
    
    Args:
        tool_cls: 工具类
        tool_config: 工具配置
        
    Returns:
        工具实例
    """
    config_dict = OmegaConf.to_container(tool_config.config, resolve=True)
    
    # 创建工具模式
    tool_schema = None
    if hasattr(tool_config, 'tool_schema') and tool_config.tool_schema is not None:
        tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
        tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
    
    # 创建工具实例
    tool = tool_cls(config=config_dict, tool_schema=tool_schema)
    
    # 如果工具有请求处理器，启动它
    if hasattr(tool, '_start_request_processor'):
        await tool._start_request_processor()
    
    return tool


async def initialize_tools_from_config(tools_config_file: str) -> Dict[str, Any]:
    """
    从配置文件初始化工具
    
    Args:
        tools_config_file: 工具配置文件路径
        
    Returns:
        Dict[str, Any]: 工具名称到工具实例的映射
    """
    logger.info(f"Loading tools from config file: {tools_config_file}")
    
    try:
        tools_config = OmegaConf.load(tools_config_file)
    except Exception as e:
        logger.error(f"Failed to load tools config file {tools_config_file}: {e}")
        raise
    
    tools_dict = {}
    
    for tool_config in tools_config.tools:
        try:
            cls_name = tool_config.class_name
            tool_type = ToolType(tool_config.config.type)
            tool_cls = get_tool_class(cls_name)
            
            logger.info(f"Initializing tool: {cls_name} (type: {tool_type.value})")
            
            if tool_type == ToolType.NATIVE:
                tool = await initialize_native_tool(tool_cls, tool_config)
                
                # 获取工具名称
                if hasattr(tool_config, 'tool_schema') and tool_config.tool_schema is not None:
                    tool_name = tool_config.tool_schema.function.name
                else:
                    # 如果没有工具模式，使用类名的最后一部分
                    tool_name = cls_name.split('.')[-1].lower()
                
                tools_dict[tool_name] = tool
                logger.info(f"Successfully initialized tool: {tool_name}")
                
            elif tool_type == ToolType.MCP:
                # MCP工具支持（如果需要的话）
                logger.warning(f"MCP tool type not yet supported: {cls_name}")
                continue
            else:
                logger.error(f"Unknown tool type: {tool_type}")
                continue
                
        except Exception as e:
            logger.error(f"Failed to initialize tool {tool_config.get('class_name', 'unknown')}: {e}")
            # 继续处理其他工具，不要因为一个工具失败就停止
            continue
    
    logger.info(f"Successfully initialized {len(tools_dict)} tools: {list(tools_dict.keys())}")
    return tools_dict


def get_tool_schemas_from_config(tools_config_file: str) -> List[Dict[str, Any]]:
    """
    从配置文件获取工具模式列表
    
    Args:
        tools_config_file: 工具配置文件路径
        
    Returns:
        List[Dict[str, Any]]: 工具模式列表
    """
    logger.info(f"Loading tool schemas from config file: {tools_config_file}")
    
    try:
        tools_config = OmegaConf.load(tools_config_file)
    except Exception as e:
        logger.error(f"Failed to load tools config file {tools_config_file}: {e}")
        return []
    
    tool_schemas = []
    
    for tool_config in tools_config.tools:
        try:
            if hasattr(tool_config, 'tool_schema') and tool_config.tool_schema is not None:
                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schemas.append(tool_schema_dict)
        except Exception as e:
            logger.error(f"Failed to process tool schema for {tool_config.get('class_name', 'unknown')}: {e}")
            continue
    
    logger.info(f"Loaded {len(tool_schemas)} tool schemas")
    return tool_schemas


class ToolRegistry:
    """
    工具注册表，用于管理和缓存工具实例
    """
    
    _instance: Optional["ToolRegistry"] = None
    _tools_cache: Dict[str, Dict[str, Any]] = {}
    
    def __init__(self):
        self._tools_cache = {}
    
    @classmethod
    def get_instance(cls) -> "ToolRegistry":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    async def load_tools_from_config(self, config_file: str, force_reload: bool = False) -> Dict[str, Any]:
        """
        从配置文件加载工具
        
        Args:
            config_file: 配置文件路径
            force_reload: 是否强制重新加载
            
        Returns:
            Dict[str, Any]: 工具字典
        """
        if config_file in self._tools_cache and not force_reload:
            logger.info(f"Using cached tools for config: {config_file}")
            return self._tools_cache[config_file]
        
        tools = await initialize_tools_from_config(config_file)
        self._tools_cache[config_file] = tools
        return tools
    
    def get_cached_tools(self, config_file: str) -> Optional[Dict[str, Any]]:
        """获取缓存的工具"""
        return self._tools_cache.get(config_file)
    
    async def cleanup_tools(self, config_file: Optional[str] = None):
        """清理工具资源"""
        if config_file is None:
            # 清理所有工具
            for file_path, tools in self._tools_cache.items():
                await self._cleanup_tools_dict(tools)
            self._tools_cache.clear()
        else:
            # 清理特定配置文件的工具
            if config_file in self._tools_cache:
                await self._cleanup_tools_dict(self._tools_cache[config_file])
                del self._tools_cache[config_file]
    
    async def _cleanup_tools_dict(self, tools: Dict[str, Any]):
        """清理工具字典中的资源"""
        for tool_name, tool in tools.items():
            try:
                # 停止请求处理器
                if hasattr(tool, 'request_processor') and tool.request_processor:
                    await tool.request_processor.stop()
                
                # 关闭会话
                if hasattr(tool, '_session') and tool._session:
                    await tool._session.close()
                    
                logger.info(f"Cleaned up tool: {tool_name}")
                
            except Exception as e:
                logger.error(f"Error cleaning up tool {tool_name}: {e}")


# 便利函数
async def load_tools_from_config(config_file: str) -> Dict[str, Any]:
    """
    从配置文件加载工具的便利函数
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        Dict[str, Any]: 工具字典
    """
    registry = ToolRegistry.get_instance()
    return await registry.load_tools_from_config(config_file)
