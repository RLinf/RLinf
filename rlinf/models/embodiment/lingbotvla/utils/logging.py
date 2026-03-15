# Copyright 2025 Optuna, HuggingFace Inc. and the LlamaFactory team. and Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Logging utils"""
# Based on: https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/utils/logging.py

import logging
import os
import sys
import threading
from functools import lru_cache
from typing import Optional

_thread_lock = threading.RLock()
_default_handler: Optional["logging.Handler"] = None
_default_log_level: "logging._Level" = logging.INFO


class _Logger(logging.Logger):
    """
    A logger that supports info_rank0.
    """

    def info_rank0(self, msg: str) -> None:
        self.info(msg)

    def warning_rank0(self, msg: str) -> None:
        self.warning(msg)

    def warning_once(self, msg: str) -> None:
        self.warning_once(msg)

    def debug_rank0(self, msg: str) -> None:
        self.debug(msg)


def _get_default_logging_level() -> "logging._Level":
    global _default_log_level

    env_lever_str = os.getenv("VEOMNI_VERBOSITY", None)
    if env_lever_str:
        if env_lever_str.upper() in logging._nameToLevel:
            return logging._nameToLevel[env_lever_str.upper()]
        else:
            raise ValueError(f"Unknown verbosity: {env_lever_str}")

    return _default_log_level


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> "logging.Logger":
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    """
    Configures root logger using a stdout stream handler with an explicit format.
    """
    global _default_handler

    with _thread_lock:
        if _default_handler:
            return

        formatter = logging.Formatter(
            fmt="[%(levelname)s][%(name)s:%(lineno)s] %(asctime)s >> %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
        )
        _default_handler = logging.StreamHandler(sys.stdout)
        _default_handler.setFormatter(formatter)
        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        library_root_logger.propagate = False


def get_logger(name: Optional[str] = None) -> "_Logger":
    """
    Returns a logger with the specified name. It is not supposed to be accessed by external scripts.
    """
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)


def set_verbosity_info() -> None:
    """
    Sets the verbosity to the `INFO` level.
    """
    _configure_library_root_logger()
    _get_library_root_logger().setLevel(logging.INFO)


def info_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.info(*args, **kwargs)


logging.Logger.info_rank0 = info_rank0


def debug_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.debug(*args, **kwargs)


logging.Logger.debug_rank0 = debug_rank0


def warning_rank0(self: "logging.Logger", *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning(*args, **kwargs)


logging.Logger.warning_rank0 = warning_rank0


@lru_cache(None)
def warning_once(self, *args, **kwargs) -> None:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        self.warning_rank0(*args, **kwargs)


logging.Logger.warning_once = warning_once
