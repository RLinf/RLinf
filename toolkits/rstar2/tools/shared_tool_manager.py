# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import asyncio
import logging
from typing import Dict, Any, Optional, Union
from omegaconf.dictconfig import DictConfig

logger = logging.getLogger(__name__)


class SharedToolManager:
    """
    共享工具管理器，用于在多个ToolAgentLoop实例之间共享工具。
    
    这个管理器确保工具只被初始化一次，然后在所有ToolAgentLoop实例之间共享，
    从而避免重复的网络连接和资源消耗。
    """
    
    _instance: Optional["SharedToolManager"] = None
    _initialized: bool = False
    
    def __init__(self):
        self._tools: Dict[str, Any] = {}
        self._tool_configs: Dict[str, Dict] = {}
        self._initialization_lock = asyncio.Lock()
        self._is_initialized = False
        
    @classmethod
    def get_instance(cls) -> "SharedToolManager":
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset_instance(cls):
        """重置单例实例（主要用于测试）"""
        if cls._instance is not None:
            # 清理现有工具
            asyncio.create_task(cls._instance.cleanup_tools())
        cls._instance = None
        cls._initialized = False
    
    async def initialize_tools(self, cfg: Union[DictConfig, str]) -> Dict[str, Any]:
        """
        初始化工具（如果尚未初始化）
        
        Args:
            cfg: 配置对象或工具配置文件路径
            
        Returns:
            Dict[str, Any]: 初始化的工具字典
        """
        async with self._initialization_lock:
            if self._is_initialized:
                logger.info("Tools already initialized, returning existing tools")
                return self._tools.copy()
            
            logger.info("Initializing shared tools...")
            
            if isinstance(cfg, str):
                # 如果是字符串，则作为配置文件路径处理
                self._tools = await self._create_tools_from_config_file(cfg)
            else:
                # 否则作为配置对象处理
                self._tools = await self._create_tools_from_config_object(cfg)
            
            self._is_initialized = True
            logger.info(f"Shared tools initialized: {list(self._tools.keys())}")
            
            return self._tools.copy()
    
    async def _create_tools_from_config_file(self, config_file: str) -> Dict[str, Any]:
        """
        从配置文件创建工具实例
        
        Args:
            config_file: 工具配置文件路径
            
        Returns:
            Dict[str, Any]: 工具字典
        """
        try:
            from .tool_registry import load_tools_from_config
            tools = await load_tools_from_config(config_file)
            logger.info(f"Tools loaded from config file {config_file}: {list(tools.keys())}")
            return tools
        except Exception as e:
            logger.error(f"Failed to load tools from config file {config_file}: {e}")
            raise
    
    async def _create_tools_from_config_object(self, cfg: Union[DictConfig, Dict]) -> Dict[str, Any]:
        """
        创建工具实例
        
        Args:
            cfg: 配置对象（DictConfig或dict）
            
        Returns:
            Dict[str, Any]: 工具字典
        """
        tools = {}
        
        # 统一处理cfg，支持DictConfig和dict
        # 对于DictConfig，使用.get()方法
        # 对于dict，也使用.get()方法
        tools_config = None
        if isinstance(cfg, dict):
            tools_config = cfg.get("tools", {})
        else:
            # DictConfig
            tools_config = cfg.get("tools", {})
        
        # 添加CodeJudge工具（如果配置了）
        code_judge_config = None
        if isinstance(tools_config, dict):
            code_judge_config = tools_config.get("code_judge")
        else:
            # DictConfig
            code_judge_config = tools_config.get("code_judge") if tools_config else None
            
        if code_judge_config:
            try:
                from toolkits.rstar2.tools.code_judge_tool import CodeJudgeTool, PythonTool, SimJupyterTool
                
                # 统一获取配置值的方法
                def get_config_value(config, key, default):
                    if isinstance(config, dict):
                        return config.get(key, default)
                    else:
                        return config.get(key, default)
                
                # 创建PythonTool实例
                python_tool = PythonTool(
                    name="python_code_with_standard_io",
                    host_addr=get_config_value(code_judge_config, "host_addr", "localhost"),
                    host_port=get_config_value(code_judge_config, "host_port", 8000),
                    batch_size=get_config_value(code_judge_config, "batch_size", 4),
                    concurrency=get_config_value(code_judge_config, "concurrency", 2),
                    batch_timeout_seconds=get_config_value(code_judge_config, "batch_timeout_seconds", 30.0),
                )
                
                # 启动工具的请求处理器
                await python_tool._start_request_processor()
                
                tools["python_code_with_standard_io"] = python_tool
                
                # 如果配置了jupyter工具
                if get_config_value(code_judge_config, "enable_jupyter", False):
                    jupyter_tool = SimJupyterTool(
                        name="jupyter_code",
                        host_addr=get_config_value(code_judge_config, "host_addr", "localhost"),
                        host_port=get_config_value(code_judge_config, "host_port", 8000),
                        batch_size=get_config_value(code_judge_config, "batch_size", 4),
                        concurrency=get_config_value(code_judge_config, "concurrency", 2),
                        batch_timeout_seconds=get_config_value(code_judge_config, "batch_timeout_seconds", 30.0),
                    )
                    await jupyter_tool._start_request_processor()
                    tools["jupyter_code"] = jupyter_tool
                
                logger.info(f"CodeJudge tools initialized: {list(tools.keys())}")
                
            except Exception as e:
                logger.error(f"Failed to initialize CodeJudge tools: {e}")
                raise
        
        # 可以在这里添加其他类型的工具
        # if cfg.get("tools", {}).get("search"):
        #     # 添加搜索工具
        #     pass
        
        return tools
    
    # 保持向后兼容性的别名
    async def _create_tools(self, cfg: DictConfig) -> Dict[str, Any]:
        """向后兼容性方法"""
        return await self._create_tools_from_config_object(cfg)
    
    def get_tools(self) -> Dict[str, Any]:
        """
        获取已初始化的工具
        
        Returns:
            Dict[str, Any]: 工具字典的副本
        """
        if not self._is_initialized:
            raise RuntimeError("Tools not initialized. Call initialize_tools() first.")
        
        return self._tools.copy()
    
    async def cleanup_tools(self):
        """清理工具资源"""
        if not self._is_initialized:
            return
            
        logger.info("Cleaning up shared tools...")
        
        for tool_name, tool in self._tools.items():
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
        
        self._tools.clear()
        self._is_initialized = False
        logger.info("All shared tools cleaned up")
    
    def is_initialized(self) -> bool:
        """检查工具是否已初始化"""
        return self._is_initialized
    
    def get_tool_stats(self) -> Dict[str, Dict]:
        """获取工具统计信息"""
        stats = {}
        
        for tool_name, tool in self._tools.items():
            try:
                if hasattr(tool, 'request_processor') and tool.request_processor:
                    stats[tool_name] = tool.request_processor.get_stats()
                else:
                    stats[tool_name] = {"status": "no_stats_available"}
            except Exception as e:
                stats[tool_name] = {"error": str(e)}
        
        return stats
    
    def print_tool_stats(self):
        """打印工具统计信息"""
        logger.info("=== Shared Tool Statistics ===")
        
        for tool_name, tool in self._tools.items():
            try:
                logger.info(f"Tool: {tool_name}")
                if hasattr(tool, 'request_processor') and tool.request_processor:
                    tool.request_processor.print_stats()
                else:
                    logger.info("  No statistics available")
            except Exception as e:
                logger.error(f"  Error getting stats: {e}")
        
        logger.info("=== End Tool Statistics ===")


# 便利函数
async def get_shared_tools(cfg: Union[DictConfig, str]) -> Dict[str, Any]:
    """
    获取共享工具的便利函数
    
    Args:
        cfg: 配置对象或工具配置文件路径
        
    Returns:
        Dict[str, Any]: 工具字典
    """
    manager = SharedToolManager.get_instance()
    return await manager.initialize_tools(cfg)


def get_initialized_tools() -> Dict[str, Any]:
    """
    获取已初始化的工具的便利函数
    
    Returns:
        Dict[str, Any]: 工具字典
    """
    manager = SharedToolManager.get_instance()
    return manager.get_tools()


async def cleanup_shared_tools():
    """清理共享工具的便利函数"""
    manager = SharedToolManager.get_instance()
    await manager.cleanup_tools()
