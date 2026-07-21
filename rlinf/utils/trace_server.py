# Copyright 2026 The RLinf Authors.
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

import argparse
import asyncio
import json
import logging
import os
import time
from fastapi import FastAPI, HTTPException, Request
import uvicorn

logger = logging.getLogger("rlinf.trace_server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="RLinf Trace Server")
trace_file_path = "trace_events.jsonl"
file_lock = asyncio.Lock()


@app.get("/sync")
async def sync_time():
    """Returns the current server timestamp in microseconds (UTC+0)."""
    server_time_us = time.time_ns() // 1000
    return {"server_time_us": server_time_us}


@app.get("/health")
@app.get("/status")
async def health():
    """Returns a simple healthcheck status."""
    return {"status": "ok"}


@app.post("/trace")
async def receive_trace(request: Request):
    """Receives trace events and appends them to a file in JSONL format."""
    try:
        events = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {str(e)}")

    if not isinstance(events, list):
        raise HTTPException(status_code=400, detail="Expected a JSON list of trace events")

    async with file_lock:
        try:
            with open(trace_file_path, "a") as f:
                for event in events:
                    f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to file: {e}")
            raise HTTPException(status_code=500, detail="Failed to write trace events to file")

    return {"status": "success", "count": len(events)}


def start_server(host: str, port: int, output_file: str):
    """Starts the FastAPI server using Uvicorn."""
    global trace_file_path
    trace_file_path = os.path.abspath(output_file)
    
    # Ensure directory exists before starting
    dir_name = os.path.dirname(trace_file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    logger.info(f"Starting trace server on {host}:{port}")
    logger.info(f"Writing trace events to: {trace_file_path}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RLinf HTTP Trace Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to")
    parser.add_argument(
        "--file", default="trace_events.jsonl", help="Output JSONL trace filepath"
    )
    args = parser.parse_args()

    start_server(args.host, args.port, args.file)
