#!/usr/bin/env python3
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
import os
import sys

# Add repository root to sys.path to make rlinf module discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rlinf.utils.trace_server import start_server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the standalone RLinf HTTP Trace Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host IP to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to (default: 8888)")
    parser.add_argument(
        "--file", default="trace_events.jsonl", help="Output JSONL trace filepath (default: trace_events.jsonl)"
    )
    args = parser.parse_args()

    start_server(args.host, args.port, args.file)
