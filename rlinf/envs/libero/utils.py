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

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
from typing import Union

import libero.libero.benchmark as benchmark
import numpy as np


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """
    Extracts image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment

    Returns:
        Preprocessed image as numpy array
    """
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(
    obs: dict[str, np.ndarray], resize_size: Union[int, tuple[int, int]] = 224
) -> np.ndarray:
    """
    Extracts wrist camera image from observations and preprocesses it.

    Args:
        obs: Observation dictionary from LIBERO environment
        resize_size: Target size for resizing

    Returns:
        Preprocessed wrist camera image as numpy array
    """
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def get_benchmark_overridden(benchmark_name) -> benchmark.Benchmark:
    """
    Return the Benchmark class for a given name.
    For "libero_130": return a dynamically aggregated class from all suites.
    For others: delegate to the original LIBERO get_benchmark.

    Args:
        benchmark_name: Name of the benchmark to get

    Returns:
        Benchmark class
    """
    name = str(benchmark_name).lower()
    if name != "libero_130":
        return benchmark.get_benchmark(benchmark_name)

    libreo_cls = benchmark.BENCHMARK_MAPPING.get("libero_130", None)
    if libreo_cls is not None:
        return libreo_cls

    # Build aggregated task map once, preserving order and de-duplicating by task name
    aggregated_task_map: dict[str, benchmark.Task] = {}
    for suite_name in getattr(benchmark, "libero_suites", []):
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(benchmark.Benchmark):
        def __init__(self, task_order_index=0):
            super().__init__(task_order_index=task_order_index)
            self.name = "libero_130"
            self._make_benchmark()

        def _make_benchmark(self):
            tasks = list(aggregated_task_map.values())
            self.tasks = tasks
            self.n_tasks = len(self.tasks)

    # Register for discoverability/help
    benchmark.BENCHMARK_MAPPING["libero_130"] = LIBERO_ALL
    return LIBERO_ALL
