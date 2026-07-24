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


from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from omegaconf import DictConfig


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _validate_num_samples(num_samples: int) -> None:
    if not isinstance(num_samples, int):
        raise TypeError(f"num_samples must be an int, got {type(num_samples).__name__}")
    if num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")


def _validate_non_negative_float(value: Any, name: str) -> float:
    value = float(value)
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _validate_positive_float(value: Any, name: str) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


class DelaySampler(ABC):
    """Sample env delays in seconds."""

    @abstractmethod
    def sample(self, num_samples: int) -> list[float]:
        """Return ``num_samples`` delays in seconds."""

    def sample_one(self) -> float:
        """Return one delay sample in seconds."""
        return self.sample(1)[0]

    @classmethod
    def create(cls, delay_config: DictConfig | None) -> "DelaySampler | None":
        """Build a sampler from a Hydra config block."""
        if delay_config is None:
            return None
        delay_type = str(delay_config.type).lower()
        seed = _cfg_get(delay_config, "seed", None)

        if delay_type == "constant":
            return ConstantDelaySampler(
                delay=delay_config.delay,
                seed=seed,
            )
        if delay_type == "uniform":
            return UniformDelaySampler(
                min_delay=delay_config.min_delay,
                max_delay=delay_config.max_delay,
                seed=seed,
            )
        if delay_type == "exponential":
            return ExponentialDelaySampler(
                rate=delay_config.rate,
                seed=seed,
            )
        if delay_type == "gaussian":
            return GaussianDelaySampler(
                mean=delay_config.mean,
                stddev=delay_config.stddev,
                seed=seed,
            )
        raise ValueError(f"Unknown delay type: {delay_type}")


class ConstantDelaySampler(DelaySampler):
    def __init__(self, delay: float, *, seed: int | None = None):
        del seed
        self.delay = _validate_non_negative_float(delay, "delay")

    def sample(self, num_samples: int) -> list[float]:
        _validate_num_samples(num_samples)
        return [self.delay] * num_samples


class UniformDelaySampler(DelaySampler):
    def __init__(
        self,
        min_delay: float,
        max_delay: float,
        *,
        seed: int | None = None,
    ):
        self.min_delay = _validate_non_negative_float(min_delay, "min_delay")
        self.max_delay = _validate_non_negative_float(max_delay, "max_delay")
        if self.min_delay > self.max_delay:
            raise ValueError(
                "min_delay must be <= max_delay, "
                f"got {self.min_delay} > {self.max_delay}"
            )
        import random

        self._rng = random.Random(seed)

    def sample(self, num_samples: int) -> list[float]:
        _validate_num_samples(num_samples)
        return [
            max(0.0, self._rng.uniform(self.min_delay, self.max_delay))
            for _ in range(num_samples)
        ]


class ExponentialDelaySampler(DelaySampler):
    def __init__(self, rate: float, *, seed: int | None = None):
        self.rate = _validate_positive_float(rate, "rate")
        import random

        self._rng = random.Random(seed)

    def sample(self, num_samples: int) -> list[float]:
        _validate_num_samples(num_samples)
        return [self._rng.expovariate(self.rate) for _ in range(num_samples)]


class GaussianDelaySampler(DelaySampler):
    def __init__(self, mean: float, stddev: float, *, seed: int | None = None):
        self.mean = _validate_non_negative_float(mean, "mean")
        self.stddev = _validate_non_negative_float(stddev, "stddev")
        import random

        self._rng = random.Random(seed)

    def sample(self, num_samples: int) -> list[float]:
        _validate_num_samples(num_samples)
        return [
            max(0.0, self._rng.gauss(self.mean, self.stddev))
            for _ in range(num_samples)
        ]
