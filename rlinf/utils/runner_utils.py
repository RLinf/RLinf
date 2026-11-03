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

import math
import os
import tempfile
from typing import Any, Union

from rlinf.utils.logging import get_logger

logger = get_logger()


def safe_is_divisible(a, b):
    """a safe divisible check to allow b to be 0"""
    if a == 0 or b == 0:
        return False
    return a % b == 0


def check_progress(
    step: int,
    max_steps: int,
    val_check_interval: int,
    save_interval: int,
    limit_val_batches: Union[int, float, None],
    run_time_exceeded: bool = False,
):
    is_validation_enabled = limit_val_batches != 0 and val_check_interval > 0
    is_save_enabled = save_interval > 0
    is_train_end = step == max_steps

    if is_validation_enabled:
        assert save_interval < 0 or save_interval % val_check_interval == 0, (
            f"{save_interval=} must be divisible by {val_check_interval=}"
        )

    # run validation on the last step
    # or when we hit the val check interval
    run_val = (
        safe_is_divisible(step, val_check_interval) or is_train_end or run_time_exceeded
    )
    run_val &= is_validation_enabled

    # save model at save intervals or last step
    save_model = (
        safe_is_divisible(step, save_interval) or is_train_end or run_time_exceeded
    )
    # sometimes the user will provide a validation metric
    # to save against, so we need to run val when we save
    save_model &= is_save_enabled

    return run_val, save_model, is_train_end


def local_mkdir_safe(path):
    from filelock import FileLock

    if not os.path.isabs(path):
        working_dir = os.getcwd()
        path = os.path.join(working_dir, path)

    # Using hash value of path as lock file name to avoid long file name
    lock_filename = f"ckpt_{hash(path) & 0xFFFFFFFF:08x}.lock"
    lock_path = os.path.join(tempfile.gettempdir(), lock_filename)

    try:
        with FileLock(lock_path, timeout=60):  # Add timeout
            # make a new dir
            os.makedirs(path, exist_ok=True)
    except Exception as e:
        print(f"Warning: Failed to acquire lock for {path}: {e}")
        # Even if the lock is not acquired, try to create the directory
        os.makedirs(path, exist_ok=True)

    return path


class EarlyStopController:
    """Select the best checkpoint and stop after consecutive failed checks."""

    def __init__(self, cfg: Any) -> None:
        self.enabled = cfg.get("enabled", False)
        self.patience = cfg.get("patience", 5)
        self.min_delta = cfg.get("min_delta", 0.001)
        self.monitor = cfg.get("monitor", "val_loss")
        self.mode = cfg.get("mode", "max" if self.monitor == "val_accuracy" else "min")
        if self.mode not in ("min", "max"):
            raise ValueError(
                f"early_stop.mode must be 'min' or 'max', got {self.mode!r}."
            )
        if self.patience < 1:
            raise ValueError(
                f"early_stop.patience must be positive, got {self.patience}."
            )
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError(
                f"early_stop.min_delta must be finite and non-negative, got {self.min_delta}."
            )

        self.counter = 0
        self.best_value: float | None = None

    @property
    def should_stop(self) -> bool:
        """Whether the configured patience has been exhausted."""
        return self.enabled and self.counter >= self.patience

    def update(self, metrics: dict[str, float]) -> tuple[bool, bool]:
        """Return (should_stop, improved) for the configured metric.

        Non-finite values count as failed checks and never select a checkpoint.
        Best-model selection remains active when early stopping is disabled.
        """
        if self.monitor not in metrics:
            raise KeyError(
                f"Monitored metric {self.monitor!r} is missing; "
                f"available metrics: {sorted(metrics)}."
            )
        value = float(metrics[self.monitor])
        improved = math.isfinite(value) and (
            self.best_value is None
            or (self.mode == "min" and value < self.best_value - self.min_delta)
            or (self.mode == "max" and value > self.best_value + self.min_delta)
        )
        if improved:
            self.best_value = value
            self.counter = 0
        elif self.enabled:
            self.counter += 1
            logger.info(f"Early stop counter: {self.counter}/{self.patience}")

        return self.should_stop, improved

    def state_dict(self) -> dict[str, Any]:
        """Return the metric definition, best value, and patience counter."""
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "min_delta": self.min_delta,
            "best_value": self.best_value,
            "counter": self.counter,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore progress with the same metric definition."""
        for key in ("monitor", "mode", "min_delta"):
            if state[key] != getattr(self, key):
                raise ValueError(
                    f"early_stop.{key} differs from the checkpoint: "
                    f"configured {getattr(self, key)!r}, saved {state[key]!r}."
                )
        best_value = state["best_value"]
        counter = state["counter"]
        if best_value is not None and not math.isfinite(best_value):
            raise ValueError(
                f"Saved early-stop best_value must be finite, got {best_value}."
            )
        if not isinstance(counter, int) or counter < 0:
            raise ValueError(
                f"Saved early-stop counter must be non-negative, got {counter}."
            )
        self.best_value = best_value
        self.counter = counter
