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

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from omegaconf import DictConfig
from transformers import AutoTokenizer

from rlinf.data.io_struct import (
    RolloutResult,
)
from rlinf.scheduler import Channel, Worker


class TrainingDataStorage:
    """Storage manager for training data received via HTTP API."""

    def __init__(self, storage_config: Optional[dict[str, Any]] = None):
        """
        Initialize storage manager.

        Args:
            storage_config: Configuration dict with options:
                - enabled: bool, whether to enable storage (default: True)
                - storage_dir: str, directory to store files (default: "./training_data")
                - max_files_per_dir: int, max files per directory (default: 1000)
                - compress: bool, whether to compress files (default: False)
        """
        if storage_config is None:
            storage_config = {}

        self.enabled = storage_config.get("enabled", True)
        self.storage_dir = Path(storage_config.get("storage_dir", "./training_data"))
        self.max_files_per_dir = storage_config.get("max_files_per_dir", 1000)
        self.compress = storage_config.get("compress", False)

        # Create storage directory if enabled
        if self.enabled:
            self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Track current file and entry count
        self._current_file_path = None
        self._entries_in_current_file = 0

    def store_training_data(self, training_data: dict[str, Any]) -> Optional[str]:
        """
        Store training data to file.

        Args:
            training_data: The training data dictionary to store

        Returns:
            Path to the stored file, or None if storage is disabled
        """
        if not self.enabled:
            return None

        # Add metadata
        storage_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "stored_at": time.time(),
            "data": training_data,
        }

        # Get or create file for writing
        file_path = self._get_current_file_path()

        # Write data based on format
        self._write_jsonl_entry(file_path, storage_entry)

        return str(file_path)

    def _get_current_file_path(self) -> Path:
        """Get the current file path for writing, creating new file if needed."""
        # Check if we need a new file
        if (
            self._current_file_path is None
            or self._entries_in_current_file >= self.max_files_per_dir
        ):
            # Create new file path
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[
                :-3
            ]  # microseconds to milliseconds
            filename = f"training_data_{timestamp}.jsonl"
            if self.compress:
                filename += ".gz"

            self._current_file_path = self.storage_dir / filename
            self._entries_in_current_file = 0

        return self._current_file_path

    def _write_jsonl_entry(self, file_path: Path, entry: dict[str, Any]):
        """Write entry to JSONL file (one JSON per line)."""
        # JSONL is more efficient for appending
        with open(file_path, "a", encoding="utf-8") as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

        self._entries_in_current_file += 1

    def get_storage_stats(self) -> dict[str, Any]:
        """Get storage statistics."""
        if not self.enabled:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "storage_dir": str(self.storage_dir),
            "current_file": str(self._current_file_path)
            if self._current_file_path
            else None,
            "entries_in_current_file": self._entries_in_current_file,
            "total_files": 0,
            "total_size_bytes": 0,
        }

        # Count files and calculate total size
        if self.storage_dir.exists():
            for file_path in self.storage_dir.iterdir():
                if file_path.is_file():
                    stats["total_files"] += 1
                    stats["total_size_bytes"] += file_path.stat().st_size

        return stats


class ServerRolloutWorker(Worker):
    """
    ServerRolloutWorker that supports both HTTP API and Channel interfaces.
    It can receive training data from router's feedback_worker via HTTP
    and also work with CodingOnlineRLRunner via Channel interface.

    Key features:
    - Unified data processing for both HTTP and Channel inputs
    - Automatic rollout processing after server startup
    - Compatible with CodingOnlineRLRunner interface
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self._cfg = cfg

        # Initialize tokenizer for text processing
        self._tokenizer = AutoTokenizer.from_pretrained(
            self._cfg.rollout.model.model_path
        )

        # Configuration
        self._server_host = cfg.server.tracking_rollout.get("host", "0.0.0.0")
        self._server_port = cfg.server.tracking_rollout.get("port", 8082)
        self._enable_dummy_data = cfg.server.tracking_rollout.get(
            "enable_dummy_data", False
        )

        # Unified data source for both HTTP and Channel data
        self._data_source = asyncio.Queue()

        # Initialize training data storage
        # storage_config = getattr(self._cfg, 'storage', None)
        storage_config = None
        if storage_config is not None:
            storage_config = dict(storage_config)
        self._storage = TrainingDataStorage(storage_config)

        # Processing configuration
        self._max_new_tokens = getattr(
            self._cfg.algorithm.sampling_params, "max_new_tokens", 512
        )
        self._batch_size = cfg.data.rollout_batch_size * cfg.algorithm.group_size

        # Processing control
        self._track_data_enable = False

        # Output channel for continuous processing

        # Setup FastAPI routes
        self._setup_routes()
        self._server_task = None

    def _setup_routes(self):
        """Setup FastAPI routes."""
        app = FastAPI(title="OnlineRouterWorker", version="1.0.0")
        app.add_route("/api/training/submit", self._handle_track, methods=["POST"])

        # Init the HTTP server
        self._server = uvicorn.Server(
            uvicorn.Config(
                app, host=self._server_host, port=self._server_port, log_level="info"
            )
        )

    def server_start(self):
        """Start service."""
        assert self._server_task is None

        # Start server in background task
        self._server_task = asyncio.create_task(self._server.serve())

        self.log_info(f"service started on {self._server_host}:{self._server_port}")

    async def server_stop(self):
        """Stop service."""
        assert self._server_task is not None

        # Stop the HTTP server
        self._server.should_exit = True

        # Wait the HTTP server to stop
        await self._server_task

        self._server_task = None
        self.log_info("service stopped")

    async def _handle_track(self, request: Request):
        """Handle training data submission from router's feedback_worker."""
        # Parse incoming training data
        training_data = await request.json()

        self.log_debug(
            f"Received training data: {training_data.get('metadata', {}).get('request_id', 'unknown')}"
        )

        training_data["received_at"] = time.time()

        if self._track_data_enable:
            # Store training data to file (async, non-blocking)
            storage_path = self._storage.store_training_data(training_data)
            if storage_path:
                training_data["storage_path"] = storage_path
                self.log_debug(f"Training data stored to: {storage_path}")

            # Put data into unified data source
            await self._data_source.put(training_data)

        # Return response to router
        response_data = {
            "status": "submitted",
            "message": "Training data submitted successfully",
            "queue_position": self._data_source.qsize(),
        }

        return Response(
            content=json.dumps(response_data),
            media_type="application/json",
        )

    def _convert_training_data_to_rollout_result(
        self, training_data: dict[str, Any]
    ) -> RolloutResult:
        """Convert training data from HTTP request into RolloutResult format."""
        # Extract text data
        input_text = training_data.get("prompt", "")
        output_text = training_data.get("completion", "")
        reward_score = training_data.get("accepted", 0.0)
        assert input_text is not None
        assert output_text is not None

        # Tokenize texts
        input_encoding = self._tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._cfg.runner.seq_length - self._max_new_tokens,
        )
        input_ids = input_encoding["input_ids"][0].tolist()

        output_encoding = self._tokenizer(
            text=output_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_new_tokens,
        )
        output_ids = output_encoding["input_ids"][0].tolist()

        # Create RolloutResult with the feedback data
        group_size = getattr(self._cfg.algorithm, "group_size", 1)

        rollout_result = RolloutResult(
            num_sequence=1,
            group_size=group_size,
            prompt_lengths=[len(input_ids)],
            prompt_ids=[input_ids],
            response_lengths=[len(output_ids)],
            response_ids=[output_ids],
            is_end=[True],  # Assume the response is complete
            rewards=torch.tensor([reward_score], dtype=torch.float32).reshape(-1, 1),
            advantages=[0.0],  # Will be computed later in the training pipeline
            prompt_texts=[input_text],
            response_texts=[output_text],
            answers=[output_text],
        )

        self.log_debug(
            f"Created RolloutResult from HTTP data with reward {reward_score}"
        )

        return rollout_result

    async def _process_unified_data_continuously(self, output_channel: Channel):
        """Continuously process data from the unified data source."""
        self.log_info("Starting continuous unified data processing")

        # clear existing data in self._data_source
        while not self._data_source.empty():
            self._data_source.get_nowait()

        # start tracking new data
        self._track_data_enable = True
        if self._enable_dummy_data:
            for i in range(self._batch_size):
                data = {
                    "prompt": "Hello, world!",
                    "completion": "Hello, world!",
                    "accepted": 1.0,
                }
                await self._data_source.put(data)

        for i in range(self._batch_size):
            # Get data from unified source (either HTTP or Channel)
            data = await self._data_source.get()

            # Convert data to RolloutResult based on source type
            rollout_result = self._convert_training_data_to_rollout_result(data)

            # Send result to output channel if available
            await output_channel.put(item=rollout_result, async_op=True).async_wait()
            # log the qsize of the output channel
            self.log_debug(f"Output channel qsize: {output_channel.qsize()}")

            # Mark task as done
            self._data_source.task_done()
        self._track_data_enable = False

        self.log_info("Continuous unified data processing stopped")

    async def rollout(self, output_channel: Channel):
        """Run HTTP server and start automatic data processing."""

        # Start automatic processing
        await self._process_unified_data_continuously(output_channel)

        self.log_info(
            "ServerRolloutWorker is running with HTTP server and auto processing"
        )

    def init_worker(self):
        """Initialize the worker (sync version)."""

        self.log_info("ServerRolloutWorker initialized")

    async def shutdown(self):
        """Shutdown the server and cleanup resources."""
        self.log_info("Shutting down ServerRolloutWorker")

        while not self._data_source.empty():
            self._data_source.get_nowait()

        self.log_info("ServerRolloutWorker shutdown complete")
