import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Callable

from omegaconf import DictConfig

from rlinf.data.swe_io_struct import SWETask, SWERequest, SWEResult
from rlinf.scheduler import Channel, Worker
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.timers import Timer
from rlinf.workers.mcp.filesystem_worker import MCPFilesystemClientWorker, MCPRequest, MCPRequestType, MCPResponse

logging.getLogger().setLevel(logging.INFO)


class MCPTestRunner:
    """MCP Test Runner."""

    def __init__(self, cfg: DictConfig, client_worker: MCPFilesystemClientWorker):
        self.cfg = cfg
        self.client_worker = client_worker

        # Communication channels
        self.input_channel = Channel.create("ClientInput", local=False)
        self.output_channel = Channel.create("ClientOutput", local=False)

    def init_workers(self):
        """Initialize the workers."""
        self.client_worker.init_worker(self.input_channel, self.output_channel)

        logging.info("MCP Test Runner initialized")

    def cleanup(self):
        """Cleanup the workers."""
        self.client_worker.cleanup()

    def run(self):
        """Run the MCP test."""
        session_id1 = str(uuid.uuid4())
        session_id2 = str(uuid.uuid4())
        print(f"session_id1: {session_id1}, session_id2: {session_id2}")
        task = [
            MCPRequest(
            request_id=str(uuid.uuid4()),
            request_type=MCPRequestType.LIST_TOOLS,
            tool_name=None,
            tool_arguments=None,
            resource_uri=None,
            prompt_name=None,
            prompt_arguments=None,
            timeout=30,
            metadata={
                "session_id": session_id1,
                "mount_dir": "/path1"
            }
            ),
            MCPRequest(
            request_id=str(uuid.uuid4()),
            request_type=MCPRequestType.CALL_TOOL,
            tool_name="write_file",
            tool_arguments={
                "path": "/projects/test/mcp_written.txt",
                "content": f"Written by mcp at {time.strftime('%Y-%m-%d %H:%M:%S')}"
            },
            resource_uri=None,
            prompt_name=None,
            prompt_arguments=None,
            timeout=30,
            metadata={
                "session_id": session_id1
            }
            ),
            MCPRequest(
            request_id=str(uuid.uuid4()),
            request_type=MCPRequestType.CALL_TOOL,
            tool_name="read_file",
            tool_arguments={
                "path": "/projects/test/mcp_written.txt"
            },
            resource_uri=None,
            prompt_name=None,
            prompt_arguments=None,
            timeout=30,
            metadata={
                "session_id": session_id1
            }
            ),
            MCPRequest(
            request_id=str(uuid.uuid4()),
            request_type=MCPRequestType.CALL_TOOL,
            tool_name="list_directory",
            tool_arguments={
                "path": "/projects/test"
            },
            resource_uri=None,
            prompt_name=None,
            prompt_arguments=None,
            timeout=30,
            metadata={
                "session_id": session_id1,
            }
            ),
            MCPRequest(
            request_id=str(uuid.uuid4()),
            request_type=MCPRequestType.CALL_TOOL,
            tool_name="list_directory",
            tool_arguments={
                "path": "/projects/test"
            },
            resource_uri=None,
            prompt_name=None,
            prompt_arguments=None,
            timeout=30,
            metadata={
                "session_id": session_id2,
                "mount_dir": "/path2"
            }
            ),
        ]
        # self.input_channel.put(task, async_op=True)
        for i in range(len(task)):
            self.input_channel.put(task[i], async_op=True)

            response: MCPResponse = asyncio.run(self.output_channel.get(async_op=True).async_wait())
            logging.info(
                f"MCP Test Runner success={response.success}, error={getattr(response, 'error_message', None)}, result={response.result}"
            )
