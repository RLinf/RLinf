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

"""Distributed training utilities."""

import logging
import os

try:
    import torch.distributed as dist

    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False
    dist = None


_loggers: dict[str, "RankAwareLogger"] = {}


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log messages."""

    # ANSI color codes
    RESET = "\033[0m"
    CYAN = "\033[36m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    BOLD = "\033[1m"

    LEVEL_COLORS = {
        logging.DEBUG: "",
        logging.INFO: "",
        logging.WARNING: YELLOW,
        logging.ERROR: RED + BOLD,
        logging.CRITICAL: RED + BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        # Color the name in cyan
        colored_name = f"{self.CYAN}[{record.name}]{self.RESET}"

        # Color the message based on level
        level_color = self.LEVEL_COLORS.get(record.levelno, "")
        if level_color:
            colored_msg = f"{level_color}{record.getMessage()}{self.RESET}"
        else:
            colored_msg = record.getMessage()

        return f"{colored_name} {colored_msg}"


class RankAwareLogger:
    """Logger that only logs on main process (rank 0)."""

    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.propagate = False  # Prevent duplicate logs from parent handlers
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(ColoredFormatter())
            self.logger.addHandler(handler)

    def _should_log(self) -> bool:
        return get_distributed_info()["is_main_process"]

    def info(self, msg: str, *args, **kwargs):
        if self._should_log():
            self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        if self._should_log():
            self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self.logger.error(msg, *args, **kwargs)  # Errors always logged

    def debug(self, msg: str, *args, **kwargs):
        if self._should_log():
            self.logger.debug(msg, *args, **kwargs)


def get_logger(name: str) -> RankAwareLogger:
    """Get a rank-aware logger that only logs on main process (cached)."""
    if name not in _loggers:
        _loggers[name] = RankAwareLogger(name)
    return _loggers[name]


def get_distributed_info():
    """Get distributed training info from environment and torch.distributed."""
    world_size_env = int(os.environ.get("WORLD_SIZE", 1))
    rank_env = int(os.environ.get("RANK", 0))
    local_rank_env = int(os.environ.get("LOCAL_RANK", 0))

    if world_size_env > 1:
        return {
            "rank": rank_env,
            "world_size": world_size_env,
            "local_rank": local_rank_env,
            "is_distributed": True,
            "is_main_process": rank_env == 0,
        }

    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        return {
            "rank": rank,
            "world_size": world_size,
            "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
            "is_distributed": world_size > 1,
            "is_main_process": rank == 0,
        }

    return {
        "rank": 0,
        "world_size": 1,
        "local_rank": 0,
        "is_distributed": False,
        "is_main_process": True,
    }


def get_distributed_info_tuple():
    """Get distributed info as tuple (local_rank, world_size, rank, is_distributed, is_main_process)."""
    info = get_distributed_info()
    return (
        info["local_rank"],
        info["world_size"],
        info["rank"],
        info["is_distributed"],
        info["is_main_process"],
    )


def print_rank(*args, **kwargs):
    """Print only from main process."""
    if get_distributed_info()["is_main_process"]:
        print(*args, **kwargs)


def print_rank0(*args, **kwargs):
    """Alias for print_rank."""
    print_rank(*args, **kwargs)


def print_rank_all(*args, prefix: str = "", **kwargs):
    """Print from every rank with prefix."""
    info = get_distributed_info()
    header = f"[Rank {info['rank']}]" + (f"[{prefix}]" if prefix else "")
    print(header, *args, **kwargs)


def wait_for_everyone(seconds=36000):
    """Synchronize all processes."""
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        try:
            dist.barrier(timeout=seconds)
        except Exception:
            pass


def is_distributed_available():
    return DISTRIBUTED_AVAILABLE


def is_distributed_initialized():
    return DISTRIBUTED_AVAILABLE and dist.is_initialized()


def get_local_rank():
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return int(os.environ.get("LOCAL_RANK", -1))


def get_world_size():
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def get_rank():
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def is_main_process():
    return get_distributed_info()["is_main_process"]


def cleanup_distributed():
    if DISTRIBUTED_AVAILABLE and dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
