# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from typing import TYPE_CHECKING, Union

import numpy as np
import torch

from rlinf.utils.logging import get_logger

if TYPE_CHECKING:
    from libero.libero.benchmark import Benchmark


def get_libero_type() -> str:
    """
    Returns the type of LIBERO, which can be "standard", "pro", or "plus".
    """
    return os.environ.get("LIBERO_TYPE", "standard").lower()


libero_type = get_libero_type()

if libero_type == "pro":
    try:
        import liberopro.liberopro.benchmark as benchmark
        from liberopro.liberopro.benchmark import Benchmark
    except ImportError:
        print(
            "[Utils] Warning: LIBERO_TYPE=pro but 'liberopro' not found. Falling back to 'libero'."
        )
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

elif libero_type == "plus":
    try:
        import liberoplus.liberoplus.benchmark as benchmark
        from liberoplus.liberoplus.benchmark import Benchmark
    except ImportError:
        print(
            "[Utils] Warning: LIBERO_TYPE=plus but 'liberoplus' not found. Falling back to 'libero'."
        )
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark

else:
    try:
        import libero.libero.benchmark as benchmark
        from libero.libero.benchmark import Benchmark
    except ImportError:
        try:
            import liberopro.liberopro.benchmark as benchmark
            from liberopro.liberopro.benchmark import Benchmark
        except ImportError:
            try:
                import liberoplus.liberoplus.benchmark as benchmark
                from liberoplus.liberoplus.benchmark import Benchmark
            except ImportError:
                raise ImportError(
                    "No valid LIBERO package (libero, liberopro, or liberoplus) found."
                )


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


def get_benchmark_overridden(benchmark_name) -> Benchmark:
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

    libero_cls = benchmark.BENCHMARK_MAPPING.get("libero_130", None)
    if libero_cls is not None:
        return libero_cls

    # Build aggregated task map once, preserving order and de-duplicating by task name
    aggregated_task_map: dict[str, benchmark.Task] = {}
    suites = getattr(benchmark, "libero_suites", [])
    for suite_name in suites:
        suite_map = benchmark.task_maps.get(suite_name, {})
        for task_name, task in suite_map.items():
            if task_name not in aggregated_task_map:
                aggregated_task_map[task_name] = task

    class LIBERO_ALL(Benchmark):
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


def build_interleaved_eval_reset_state_ids(
    trial_id_bins: list[int],
    cumsum_trial_id_bins: np.ndarray,
) -> np.ndarray:
    """Order (task0, trial0), (task1, trial0), ... for even parallel coverage."""
    interleaved = []
    num_tasks = len(trial_id_bins)
    max_trials = max(trial_id_bins) if trial_id_bins else 0
    for trial in range(max_trials):
        for task_id in range(num_tasks):
            if trial < trial_id_bins[task_id]:
                start = cumsum_trial_id_bins[task_id - 1] if task_id > 0 else 0
                interleaved.append(start + trial)
    return np.array(interleaved, dtype=np.int64)


def distribute_reset_state_ids_round_robin(
    reset_state_ids: np.ndarray,
    total_num_processes: int,
) -> np.ndarray:
    """Assign each reset state to exactly one rank (round-robin)."""
    n_procs = total_num_processes
    n_states = len(reset_state_ids)
    per_rank_counts = np.bincount(np.arange(n_states) % n_procs, minlength=n_procs)
    max_per_rank = int(per_rank_counts.max())
    distributed = np.full((n_procs, max_per_rank), -1, dtype=np.int64)
    counters = np.zeros(n_procs, dtype=int)
    for i, state_id in enumerate(reset_state_ids):
        rank = int(i % n_procs)
        distributed[rank, counters[rank]] = state_id
        counters[rank] += 1
    return distributed


def record_completed_episode_task_stats(
    env_idx: np.ndarray,
    final_info: dict,
    task_ids: np.ndarray,
    trial_ids: np.ndarray,
    num_envs: int,
    eval_seen_trials: set[tuple[int, int]],
    task_success_stats: dict[int, dict[str, int]],
    logger=None,
) -> np.ndarray:
    """Record per-task eval stats and return which envs count toward metrics.

    In eval mode each (task_id, trial_id) is counted at most once. Duplicate
    completions from auto_reset cycling are excluded so aggregated eval metrics
    match the benchmark suite size instead of counting every episode termination.
    """
    logger = logger or get_logger()
    count_mask = np.zeros(num_envs, dtype=bool)

    episode = final_info.get("episode")
    if not episode or "success_once" not in episode:
        return count_mask

    success = episode["success_once"]
    if isinstance(success, torch.Tensor):
        success = success.cpu().numpy()
    else:
        success = np.asarray(success)

    for eid in env_idx:
        tid = int(task_ids[eid])
        trial_id = int(trial_ids[eid])
        ok = bool(success[eid])
        trial_key = (tid, trial_id)
        if trial_key in eval_seen_trials:
            logger.warning(
                f"[libero eval] duplicate episode skipped: "
                f"task_id={tid}, trial_id={trial_id}"
            )
            continue
        eval_seen_trials.add(trial_key)
        count_mask[eid] = True
        if tid not in task_success_stats:
            task_success_stats[tid] = {"success": 0, "total": 0}
        task_success_stats[tid]["total"] += 1
        task_success_stats[tid]["success"] += int(ok)
        logger.info(f"[libero eval] task_id={tid}, trial_id={trial_id}, success={ok}")
    return count_mask
