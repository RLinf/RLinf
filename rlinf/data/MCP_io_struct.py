from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from enum import Enum

class MCPRequestType(Enum):
    """MCP request types."""
    LIST_TOOLS = "list_tools"
    CALL_TOOL = "call_tool"
    LIST_RESOURCES = "list_resources"
    READ_RESOURCE = "read_resource"
    LIST_PROMPTS = "list_prompts"
    GET_PROMPT = "get_prompt"


@dataclass
class MCPRequest:
    """MCP request structure for channel communication."""
    request_id: str
    request_type: MCPRequestType
    tool_name: Optional[str] = None
    tool_arguments: Optional[Dict[str, Any]] = None
    resource_uri: Optional[str] = None
    prompt_name: Optional[str] = None
    prompt_arguments: Optional[Dict[str, Any]] = None
    timeout: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MCPResponse:
    """MCP response structure for channel communication."""
    request_id: str
    success: bool
    result: Optional[Any] = None
    error_message: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class MCPSessionState(Enum):
    """MCP session states."""
    INITIALIZING = "initializing"
    CONNECTED = "connected"
    FAILED = "failed"
    TERMINATED = "terminated"
