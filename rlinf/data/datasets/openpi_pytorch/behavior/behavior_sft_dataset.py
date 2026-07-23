# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import bisect
import dataclasses
import logging
import os
import random
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import snapshot_download
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import (
    CODEBASE_VERSION,
    LeRobotDataset,
    LeRobotDatasetMetadata,
)
from lerobot.common.datasets.utils import (
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    STATS_PATH,
    TASKS_PATH,
    backward_compatible_episodes_stats,
    cast_stats_to_numpy,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    get_delta_indices,
    get_episode_data_index,
    get_safe_version,
    is_valid_version,
    load_info,
    load_json,
    load_jsonlines,
)
from lerobot.common.datasets.video_utils import get_safe_default_codec

# Transform base for PromptFromLeRobotItem — reuse openpi's (the openpi_pytorch
# data path is built on openpi.transforms throughout).
from openpi.transforms import DataTransformFn
from torch.utils.data import Dataset, get_worker_info

logger = logging.getLogger("BehaviorSftDataset")

# ---------------------------------------------------------------------------
# Inlined constants (from omnigibson.learning.utils.eval_utils)
# ---------------------------------------------------------------------------

ROBOT_CAMERA_NAMES = {
    "R1Pro": {
        "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0",
        "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0",
        "head": "robot_r1::robot_r1:zed_link:Camera:0",
    },
}

TASK_NAMES_TO_INDICES = {
    # B10
    "turning_on_radio": 0,
    "picking_up_trash": 1,
    "putting_away_Halloween_decorations": 2,
    "cleaning_up_plates_and_food": 3,
    "can_meat": 4,
    "setting_mousetraps": 5,
    "hiding_Easter_eggs": 6,
    "picking_up_toys": 7,
    "rearranging_kitchen_furniture": 8,
    "putting_up_Christmas_decorations_inside": 9,
    # B20
    "set_up_a_coffee_station_in_your_kitchen": 10,
    "putting_dishes_away_after_cleaning": 11,
    "preparing_lunch_box": 12,
    "loading_the_car": 13,
    "carrying_in_groceries": 14,
    "bringing_in_wood": 15,
    "moving_boxes_to_storage": 16,
    "bringing_water": 17,
    "tidying_bedroom": 18,
    "outfit_a_basic_toolbox": 19,
    # B30
    "sorting_vegetables": 20,
    "collecting_childrens_toys": 21,
    "putting_shoes_on_rack": 22,
    "boxing_books_up_for_storage": 23,
    "storing_food": 24,
    "clearing_food_from_table_into_fridge": 25,
    "assembling_gift_baskets": 26,
    "sorting_household_items": 27,
    "getting_organized_for_work": 28,
    "clean_up_your_desk": 29,
    # B40
    "setting_the_fire": 30,
    "clean_boxing_gloves": 31,
    "wash_a_baseball_cap": 32,
    "wash_dog_toys": 33,
    "hanging_pictures": 34,
    "attach_a_camera_to_a_tripod": 35,
    "clean_a_patio": 36,
    "clean_a_trumpet": 37,
    "spraying_for_bugs": 38,
    "spraying_fruit_trees": 39,
    # B50
    "make_microwave_popcorn": 40,
    "cook_cabbage": 41,
    "chop_an_onion": 42,
    "slicing_vegetables": 43,
    "chopping_wood": 44,
    "cook_hot_dogs": 45,
    "cook_bacon": 46,
    "freeze_pies": 47,
    "canning_food": 48,
    "make_pizza": 49,
}
TASK_INDICES_TO_NAMES = {v: k for k, v in TASK_NAMES_TO_INDICES.items()}

ANNOTATIONS_PATH = "annotations"
ORCHESTRATORS_PATH = "orchestrators"


# ---------------------------------------------------------------------------
# Lazy OmniGibson utility access
# ---------------------------------------------------------------------------


def _install_lerobot_compat_shim():
    """Alias ``lerobot.datasets.compute_stats`` to the pinned LeRobot's
    ``lerobot.common.datasets.compute_stats`` so OmniGibson's ``lerobot_utils`` (which
    imports ``_assert_type_and_shape`` from the new LeRobot layout) loads unchanged.

    This imports only LeRobot — never OmniGibson — so running it at module import does
    not regress the lazy-OmniGibson contract. It must be installed at import time
    (below) because spawned (``num_workers>0``) DataLoader workers unpickle this
    dataset's cached OmniGibson helper references, which re-imports
    ``omnigibson.learning.utils.lerobot_utils`` before any method runs; without the
    shim already in place that import raises ``ModuleNotFoundError: lerobot.datasets``.
    """
    import sys

    if "lerobot.datasets.compute_stats" in sys.modules:
        return
    try:
        import lerobot.datasets.compute_stats  # noqa: F401
    except ModuleNotFoundError:
        import types

        from lerobot.common.datasets.compute_stats import _assert_type_and_shape

        pkg = sys.modules.setdefault(
            "lerobot.datasets", types.ModuleType("lerobot.datasets")
        )
        shim = types.ModuleType("lerobot.datasets.compute_stats")
        shim._assert_type_and_shape = _assert_type_and_shape
        sys.modules["lerobot.datasets.compute_stats"] = shim
        pkg.compute_stats = shim


# Install the LeRobot compat shim at import time (LeRobot-only, no OmniGibson) so
# spawned DataLoader workers can unpickle this dataset's OmniGibson helper references.
_install_lerobot_compat_shim()


def _omnigibson_utils():
    """Lazily import the OmniGibson video/stat utilities used by this dataset.

    Imported on first call (never at module import) so that loading this module
    — e.g. during config validation — does not pull OmniGibson or trigger any
    scene/asset load. Returns ``(hf_transform_to_torch, aggregate_stats,
    decode_video_frames, OBS_LOADER_MAP)``.
    """
    from omnigibson.learning.utils.lerobot_utils import (
        aggregate_stats,
        decode_video_frames,
        hf_transform_to_torch,
    )
    from omnigibson.learning.utils.obs_utils import OBS_LOADER_MAP

    return hf_transform_to_torch, aggregate_stats, decode_video_frames, OBS_LOADER_MAP


# ---------------------------------------------------------------------------
# PromptFromLeRobotItem transform
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PromptFromLeRobotItem(DataTransformFn):
    """Extracts a prompt from the current LeRobot dataset item's ``task`` field."""

    def __call__(self, data: dict) -> dict:
        return {**data, "prompt": data.pop("task")}


# ---------------------------------------------------------------------------
# Orchestrator helpers
# ---------------------------------------------------------------------------


def load_orchestrators_data(episode_path_or_level_0_task, episode_len):
    """Build the per-level (task, start_frame, end_frame) orchestrator table."""
    output_data = defaultdict(list)
    if type(episode_path_or_level_0_task) is str:
        for i in range(4):
            output_data[i] = [
                {
                    "task": episode_path_or_level_0_task,
                    "start_frame": 0,
                    "end_frame": episode_len - 1,
                }
            ]
        return output_data
    episode_path = episode_path_or_level_0_task
    task_annotated_data = load_json(episode_path / "task_annotated.json")
    level_0_task = task_annotated_data["cot_task_description"]
    output_data[0].append(
        {
            "task": level_0_task,
            "start_frame": 0,
            "end_frame": episode_len - 1,
        }
    )
    try:
        num_level1_tasks = len(task_annotated_data["cot_subtask_description_list"])
        for i in range(num_level1_tasks):
            subtask_data = load_json(episode_path / f"subtask_{i}_annotated.json")
            subtask = subtask_data["cot_subtask_description"]
            start_frame, end_frame = (
                subtask_data["start_frame"],
                subtask_data["end_frame"] - 1,
            )
            skill = subtask_data["skill_description"]
            output_data[1].append(
                {"task": skill, "start_frame": start_frame, "end_frame": end_frame}
            )
            output_data[2].append(
                {"task": subtask, "start_frame": start_frame, "end_frame": end_frame}
            )
            for event_data_path in sorted(
                episode_path.glob(f"event_{i}_*_annotated.json")
            ):
                event_data = load_json(event_data_path)
                event_task = event_data["subtask_answer_detailed"]
                start_frame, end_frame = (
                    event_data["start_frame"],
                    event_data["end_frame"] - 1,
                )
                output_data[3].append(
                    {
                        "task": event_task,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                    }
                )
    except Exception as e:
        logger.warning(
            "%s failed to load orchestrators data: %s, falling back to default task.",
            episode_path,
            e,
        )
        for i in range(len(output_data)):
            output_data[i] = output_data[0]
    return output_data


def skill_weight(cur_skill, skill_list: list[str]) -> float:
    """Return the sampling weight for ``cur_skill`` given a ``skill:weight`` list."""
    if "all" in skill_list:
        skill_list = [skill for skill in skill_list if skill != "all"]
        for skill_item in skill_list:
            skill, weight = skill_item.split(":")
            if skill == cur_skill:
                return float(weight)
        return 1.0
    for skill_item in skill_list:
        skill, weight = skill_item.split(":")
        if skill == cur_skill:
            return float(weight)
    return 0.0


# ---------------------------------------------------------------------------
# BehaviorSftDatasetMetadata
# ---------------------------------------------------------------------------


class BehaviorSftDatasetMetadata(LeRobotDatasetMetadata):
    """LeRobot metadata extended with BEHAVIOR task filtering and orchestrators."""

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        tasks: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
    ):
        self.task_name_candidates = (
            set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        )
        self.modalities = set(modalities) if modalities else {"rgb"}
        self.camera_names = (
            set(cameras) if cameras else {"head", "left_wrist", "right_wrist"}
        )
        assert self.modalities.issubset({"rgb", "depth", "seg_instance_id"})
        assert self.camera_names.issubset(ROBOT_CAMERA_NAMES["R1Pro"])

        self.repo_id = repo_id
        self.revision = revision or CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)
            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(
                allow_patterns="meta/**", ignore_patterns="meta/episodes/**"
            )
            self.load_metadata()

    def load_metadata(self):
        """Load info, filtered tasks/episodes, annotations, orchestrators, stats."""
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index, self.task_names = self.load_tasks(
            self.root
        )
        valid_task_indices = [
            idx
            for idx, name in self.task_names.items()
            if name in self.task_name_candidates
        ]
        self.task_names = {self.task_names[idx] for idx in valid_task_indices}
        self.tasks = {idx: self.tasks[idx] for idx in valid_task_indices}
        self.task_to_task_index = {v: k for k, v in self.tasks.items()}

        self.episodes = self.load_episodes(self.root)
        self.annotations = self.load_annotations(self.root)
        self.orchestrators = self.load_orchestrators(self.root)
        import packaging.version

        if self._version < packaging.version.parse("v2.1"):
            self.stats = self.load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(
                self.stats, self.episodes
            )
        else:
            self.episodes_stats = self.load_episodes_stats(self.root)
            _, aggregate_stats, _, _ = _omnigibson_utils()
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

    def load_tasks(self, local_dir: Path):
        """Load the ``tasks.jsonl`` mapping (index -> task text / task name)."""
        tasks = load_jsonlines(local_dir / TASKS_PATH)
        task_names = {
            item["task_index"]: item["task_name"]
            for item in sorted(tasks, key=lambda x: x["task_index"])
        }
        tasks_dict = {
            item["task_index"]: item["task"]
            for item in sorted(tasks, key=lambda x: x["task_index"])
        }
        task_to_task_index = {task: idx for idx, task in tasks_dict.items()}
        return tasks_dict, task_to_task_index, task_names

    def load_episodes(self, local_dir: Path):
        """Load episodes belonging to the selected tasks."""
        episodes = load_jsonlines(local_dir / EPISODES_PATH)
        return {
            item["episode_index"]: item
            for item in sorted(episodes, key=lambda x: x["episode_index"])
            if item["episode_index"] // 1e4 in self.tasks
        }

    def load_stats(self, local_dir: Path):
        """Load aggregate dataset stats (legacy <v2.1 datasets)."""
        if not (local_dir / STATS_PATH).exists():
            return None
        stats = load_json(local_dir / STATS_PATH)
        return cast_stats_to_numpy(stats)

    def load_episodes_stats(self, local_dir: Path):
        """Load per-episode stats for the selected episodes."""
        episodes_stats = load_jsonlines(local_dir / EPISODES_STATS_PATH)
        return {
            item["episode_index"]: cast_stats_to_numpy(item["stats"])
            for item in sorted(episodes_stats, key=lambda x: x["episode_index"])
            if item["episode_index"] in self.episodes
        }

    def load_annotations(self, local_dir: Path):
        """Load per-episode skill annotations (used by skill mode)."""
        annotations_dir = local_dir / ANNOTATIONS_PATH
        if not annotations_dir.exists():
            return {}
        task_list = [
            task_id for task_id in annotations_dir.iterdir() if task_id.is_dir()
        ]
        return {
            int(episode.stem[8:]): load_json(episode)
            for task_id in task_list
            if int(task_id.name[5:]) in self.tasks
            for episode in sorted(task_id.iterdir())
        }

    def load_orchestrators(self, local_dir: Path):
        """Load (or synthesize) per-episode fine-grained task orchestrators."""
        orchestrators_path = local_dir / ORCHESTRATORS_PATH
        orchestrators = {
            episode_key: load_orchestrators_data(
                episode_data["tasks"][0], episode_data["length"]
            )
            for episode_key, episode_data in sorted(self.episodes.items())
        }
        if orchestrators_path.exists():
            for task in self.tasks:
                task_dir = orchestrators_path / f"task-{task:04d}"
                if task_dir.exists():
                    orchestrators.update(
                        {
                            int(episode.stem[8:]): load_orchestrators_data(
                                episode,
                                self.episodes[int(episode.stem[8:])]["length"],
                            )
                            for episode in sorted(task_dir.iterdir())
                        }
                    )
        return orchestrators

    def get_annotation_path(self, ep_index: int) -> Path:
        """Resolve the annotation file path for ``ep_index``."""
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.annotation_path.format(
            episode_chunk=ep_chunk, episode_index=ep_index
        )
        return Path(fpath)

    def get_metainfo_path(self, ep_index: int) -> Path:
        """Resolve the meta-info file path for ``ep_index``."""
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.metainfo_path.format(
            episode_chunk=ep_chunk, episode_index=ep_index
        )
        return Path(fpath)

    @property
    def annotation_path(self) -> str | None:
        """Template path for episode annotation files (from ``info``)."""
        return self.info.get("annotation_path")

    @property
    def metainfo_path(self) -> str | None:
        """Template path for episode meta-info files (from ``info``)."""
        return self.info.get("metainfo_path")

    @property
    def features(self) -> dict[str, dict]:
        """Image features filtered to the selected cameras/modalities."""
        features = {}
        for name in self.info["features"].keys():
            if (
                name.startswith("observation.images.")
                and name.split(".")[-1] in self.camera_names
                and name.split(".")[-2] in self.modalities
            ):
                features[name] = self.info["features"][name]
        return features


# ---------------------------------------------------------------------------
# BehaviorSftDataset
# ---------------------------------------------------------------------------


def partition_chunk_indices(
    num_chunks: int,
    *,
    rank: int,
    world_size: int,
    worker_id: int,
    num_workers: int,
) -> list[int]:
    """Return the chunk indices a single ``(rank, worker)`` pair streams.

    The distributed ``rank`` is folded into the per-worker stride so that every
    ``(rank, worker)`` pair receives a disjoint, non-overlapping set of chunk
    indices whose union over all ranks and workers covers every chunk. This is
    what lets the streaming dataset shard distinct data per rank under
    ``torchrun``/DDP: a ``DistributedSampler`` cannot, because the dataset
    ignores the ``idx`` it is handed and streams chunks internally.
    """
    global_worker_id = rank * num_workers + worker_id
    stride = world_size * num_workers
    return list(range(global_worker_id, num_chunks, stride))


class BehaviorSftDataset(LeRobotDataset):
    """Streaming BEHAVIOR-1K dataset for pi05 SFT (task filtering + chunk streaming).

    The dataset streams contiguous keyframe *chunks* of each episode rather than
    returning the frame at a random ``idx``: ``__getitem__`` ignores ``idx`` and
    advances an internal streaming cursor. Chunks are partitioned across data
    loader workers (and, in this port, across distributed ranks) so that every
    consumer sees a disjoint stream — see :meth:`__getitem__`.

    Direct-task mode (the default) sets the per-frame prompt to the fine-grained
    task text. Skill mode (``skill_labels`` provided) additionally resolves each
    frame to a skill-level label and skips gap frames; the skill-boundary logic
    is carried over from the old pipeline via :meth:`_build_skill_boundaries` /
    :meth:`_get_skill_label` and is intended to be aligned to the reference
    ``openpi-comet`` semantics in a follow-up.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = "pyav",
        batch_encoding_size: int = 1,
        # Custom arguments
        tasks: Iterable[str] | None = None,
        modalities: Iterable[str] | None = None,
        cameras: Iterable[str] | None = None,
        local_only: bool = False,
        check_timestamp_sync: bool = True,
        chunk_streaming_using_keyframe: bool = True,
        shuffle: bool = True,
        seed: int = 42,
        fine_grained_level: int = 0,
        train_rgb_type: str = "regular",
        return_seg_instance: bool = False,
        skill_list: list[str] | None = None,
        skill_labels: dict[int, str] | None = None,
        use_skill: bool = False,
        enable_gap: bool = True,
        allow_left: int = 0,
        allow_right: int = 0,
        dist_rank: int | None = None,
        dist_world_size: int | None = None,
    ):
        import packaging.version

        if skill_list is None:
            skill_list = ["all"]

        Dataset.__init__(self)
        self.repo_id = repo_id
        self.root = (
            Path(os.path.expanduser(str(root))) if root else HF_LEROBOT_HOME / repo_id
        )
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.tolerance_s = tolerance_s
        self.revision = revision or CODEBASE_VERSION
        self.video_backend = video_backend or get_safe_default_codec()
        self.delta_indices = None
        self.batch_encoding_size = batch_encoding_size
        self.episodes_since_last_encoding = 0
        self.return_seg_instance = return_seg_instance
        self.train_rgb_type = train_rgb_type
        self.skill_list = skill_list
        self.skill_labels = skill_labels
        # When `use_skill` is set, the training prompt is the per-frame SKILL text
        # (resolved by the window logic) instead of the main-task text; explicit
        # `skill_labels` are required in that case (the production builder sources
        # them from config), otherwise the constructor raises below.
        self.use_skill = use_skill
        # Skill-mode windowing, aligned with the JAX openpi-comet semantics:
        # `enable_gap` absorbs a true gap into both adjacent skills (so the gap
        # region overlaps and is shared); `allow_left` / `allow_right` are frame
        # counts that extend contiguous skill boundaries outward (also creating
        # overlap). See `_build_skill_boundaries` / `_get_skill_label`.
        self.enable_gap = enable_gap
        self.allow_left = allow_left
        self.allow_right = allow_right
        # Explicit distributed identity captured in the MAIN process. The streaming
        # chunk partition is rank-aware, but DataLoader workers are SPAWNED (fresh
        # interpreters that do NOT inherit ``torch.distributed``), so reading
        # ``dist.get_rank()`` inside a worker returns 0 and every rank would replicate
        # rank 0's partition. Storing rank/world_size here (pickled into the worker)
        # lets ``_select_streaming_chunk`` partition by the correct per-rank id; we fall
        # back to ``torch.distributed`` only when these are not provided.
        self._dist_rank = dist_rank
        self._dist_world_size = dist_world_size
        # Real OmniGibson video/stat utilities, imported lazily here (never at module
        # import) and cached so the streaming hot path and `load_hf_dataset` can reuse
        # them without re-importing. See `_omnigibson_utils`.
        (
            self._hf_transform_to_torch,
            self._aggregate_stats,
            self._decode_video_frames,
            self._obs_loader_map,
        ) = _omnigibson_utils()

        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        self.seed = seed
        if modalities is None:
            modalities = ["rgb"]
        if cameras is None:
            cameras = ["head", "left_wrist", "right_wrist"]
        self.task_names = (
            set(tasks) if tasks is not None else set(TASK_NAMES_TO_INDICES.keys())
        )
        self.task_indices = [TASK_NAMES_TO_INDICES[task] for task in self.task_names]

        self.meta = BehaviorSftDatasetMetadata(
            repo_id=self.repo_id,
            root=self.root,
            revision=self.revision,
            force_cache_sync=force_cache_sync,
            tasks=self.task_names,
            modalities=modalities,
            cameras=cameras,
        )

        all_episodes = load_jsonlines(self.root / EPISODES_PATH)
        epi_by_task = defaultdict(list)
        for item in all_episodes:
            if item["episode_index"] // 1e4 in self.meta.tasks:
                epi_by_task[item["episode_index"] // 1e4].append(item["episode_index"])
        for task_id, ep_indices in epi_by_task.items():
            epi_by_task[task_id] = sorted(ep_indices)
            if episodes is not None:
                epi_by_task[task_id] = [
                    epi_by_task[task_id][i]
                    for i in episodes
                    if i < len(epi_by_task[task_id])
                ]
        self.episodes = sorted([ep for eps in epi_by_task.values() for ep in eps])

        self._chunk_streaming_using_keyframe = chunk_streaming_using_keyframe
        if self._chunk_streaming_using_keyframe:
            self.chunks = self._get_keyframe_chunk_indices()
            if shuffle:
                self.current_streaming_chunk_idx = None
                self.current_streaming_frame_idx = None
            else:
                self.current_streaming_chunk_idx = 0
                self.current_streaming_frame_idx = self.chunks[
                    self.current_streaming_chunk_idx
                ][0]
            self.obs_loaders = {}
            self._should_obs_loaders_reload = True

        self.episode_data_index_pos = {
            ep_idx: i for i, ep_idx in enumerate(self.episodes)
        }
        logger.info("Total episodes: %d", len(self.episodes))

        if self.episodes is not None and self.meta._version >= packaging.version.parse(
            "v2.1"
        ):
            episodes_stats = [
                self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes
            ]
            self.stats = self._aggregate_stats(episodes_stats)

        try:
            if force_cache_sync:
                raise FileNotFoundError
            for fpath in self.get_episodes_file_paths():
                assert (self.root / fpath).is_file(), (
                    f"Missing file: {self.root / fpath}"
                )
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError) as e:
            if local_only:
                raise e
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )

        if check_timestamp_sync:
            timestamps = th.stack(self.hf_dataset["timestamp"]).numpy()
            episode_indices = th.stack(self.hf_dataset["episode_index"]).numpy()
            ep_data_index_np = {
                k: t.numpy() for k, t in self.episode_data_index.items()
            }
            check_timestamps_sync(
                timestamps,
                episode_indices,
                ep_data_index_np,
                self.fps,
                self.tolerance_s,
            )

        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

        self.prepare_task(fine_grained_level)

        # Skill mode requires explicit REFERENCE skill_labels (sourced from config by
        # the builder). Deriving them from this dataset's orchestrators is unsafe —
        # the real 2025-challenge-demos collapses every orchestrator level to the full
        # task text, so derived labels would equal the task prompt — hence fail loudly.
        if self.use_skill and self.skill_labels is None:
            raise ValueError(
                "BehaviorSftDataset(use_skill=True) requires explicit skill_labels "
                "(the reference per-task subtask labels); none were supplied. The "
                "production builder sources them from data.task_subtasks."
            )
        if self.skill_labels is not None:
            self._build_skill_boundaries()

        self.omnigibson_mapping = {
            ep_idx: defaultdict(dict) for ep_idx in self.episodes
        }

    def prepare_task(self, fine_grained_level: int):
        """Pre-compute cumulative subtask end-frames per episode for lookups."""
        self.fine_grained_level = fine_grained_level
        self.task_sizes = {}
        try:
            for ep_id, ep_orch in self.meta.orchestrators.items():
                self.task_sizes[ep_id] = [
                    task_info["end_frame"] for task_info in ep_orch[fine_grained_level]
                ]
        except Exception as e:
            logger.warning(
                "%s failed to calculate episode subtask cumulate: %s", self.repo_id, e
            )

    # --- Skill label support (for VLM SFT) ---------------------------------
    # Ported from the old pipeline. The skill-boundary / gap structure is kept
    # intact here so that a follow-up can align the precise gap-splitting and
    # left/right inclusion semantics to the reference openpi-comet dataset.

    def _build_skill_boundaries(self):
        """Build per-episode *effective* skill windows, aligned with openpi-comet.

        For each skill, the effective window extends a contiguous (no-gap)
        boundary outward by ``allow_left`` / ``allow_right`` frames, and — when
        ``enable_gap`` is set — absorbs a true gap into both adjacent skills so
        the gap region overlaps and is shared. Windows are clamped to the
        episode's valid duration. Frames inside more than one window are resolved
        per-sample by a random choice in :meth:`_get_skill_label`.
        """
        self.skill_start_frames: dict[int, list[int]] = {}
        self.skill_end_frames: dict[int, list[int]] = {}
        for ep_id in self.episodes:
            if ep_id not in self.meta.annotations:
                continue
            annotation = self.meta.annotations[ep_id]
            skills = sorted(
                annotation["skill_annotation"],
                key=lambda s: s["skill_idx"],
            )
            starts = [s["frame_duration"][0] for s in skills]
            ends = [s["frame_duration"][1] for s in skills]
            valid = annotation["meta_data"]["valid_duration"]
            valid_start, valid_end = valid[0], valid[1]
            n = len(skills)
            eff_s = list(starts)
            eff_e = list(ends)
            for i in range(n):
                # Left boundary: absorb a true gap (enable_gap) or extend a
                # contiguous boundary by allow_left frames.
                if i > 0 and ends[i - 1] < starts[i]:
                    if self.enable_gap:
                        eff_s[i] = ends[i - 1]
                else:
                    eff_s[i] = starts[i] - self.allow_left
                # Right boundary: symmetric.
                if i < n - 1 and ends[i] < starts[i + 1]:
                    if self.enable_gap:
                        eff_e[i] = starts[i + 1]
                else:
                    eff_e[i] = ends[i] + self.allow_right
                eff_s[i] = max(eff_s[i], valid_start)
                eff_e[i] = min(eff_e[i], valid_end)
            self.skill_start_frames[ep_id] = eff_s
            self.skill_end_frames[ep_id] = eff_e

    def _is_gap_frame(self, ep_idx: int, frame_index: int) -> bool:
        """Return True if ``frame_index`` falls outside every effective window."""
        start_frames = self.skill_start_frames.get(ep_idx)
        end_frames = self.skill_end_frames.get(ep_idx)
        if start_frames is None or end_frames is None:
            return False
        for start, end in zip(start_frames, end_frames):
            if start <= frame_index < end:
                return False
        return True

    def _get_skill_label(self, item: dict) -> str:
        """Resolve a frame to a skill label, aligned with openpi-comet.

        A frame may fall inside more than one skill's effective window (overlap
        from gap absorption or boundary extension); in that case one candidate
        is chosen at random per sample. A frame inside exactly one window takes
        that skill; a frame outside all windows falls back to the nearest
        preceding skill.
        """
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        start_frames = self.skill_start_frames[ep_idx]
        end_frames = self.skill_end_frames[ep_idx]
        candidates = [
            i
            for i in range(len(start_frames))
            if start_frames[i] <= frame_index < end_frames[i]
        ]
        if len(candidates) == 1:
            return self.skill_labels[candidates[0]]
        if len(candidates) > 1:
            return self.skill_labels[random.choice(candidates)]
        skill_idx = bisect.bisect_right(start_frames, frame_index) - 1
        return self.skill_labels[max(0, skill_idx)]

    # -------------------------------------------------------------------------

    def get_episodes_file_paths(self) -> list[str]:
        """Return the data/meta/video file paths for the selected episodes."""
        episodes = (
            self.episodes
            if self.episodes is not None
            else list(self.meta.episodes.keys())
        )
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        metainfo_path = getattr(self.meta, "metainfo_path", None)
        if metainfo_path:
            fpaths += [str(self.meta.get_metainfo_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        return fpaths

    def download_episodes(self, download_videos: bool = True) -> None:
        """Download only the selected tasks' data (and videos) from the hub."""
        allow_patterns = []
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in self.task_indices:
                allow_patterns.append(f"**/task-{task:04d}/**")
        ignore_patterns = []
        if not download_videos:
            ignore_patterns.append("videos/")
        if set(self.task_indices) != set(TASK_NAMES_TO_INDICES.values()):
            for task in set(TASK_NAMES_TO_INDICES.values()).difference(
                self.task_indices
            ):
                ignore_patterns.append(f"**/task-{task:04d}/**")
        allow_patterns = None if allow_patterns == [] else allow_patterns
        ignore_patterns = None if ignore_patterns == [] else ignore_patterns
        self.pull_from_repo(
            allow_patterns=allow_patterns, ignore_patterns=ignore_patterns
        )

    def pull_from_repo(self, allow_patterns=None, ignore_patterns=None):
        """Snapshot-download the dataset repo into ``self.root``."""
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            max_workers=max(1, os.cpu_count() - 2),
        )

    def load_hf_dataset(self):
        """Load the parquet frames for the selected episodes as a HF dataset."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [
                str(self.root / self.meta.get_data_file_path(ep_idx))
                for ep_idx in self.episodes
            ]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")
        hf_dataset.set_transform(self._hf_transform_to_torch)
        return hf_dataset

    def _select_streaming_chunk(self) -> None:
        """Pick this ``(rank, worker)``'s active chunk set and starting frame.

        Reads the distributed ``rank`` / ``world_size`` and the data-loader
        ``worker_id`` / ``num_workers``, folds them into the keyframe-chunk
        partition (via :func:`partition_chunk_indices`), shuffles the resulting
        chunks with a per-``(rank, worker)`` seed, and sets the streaming cursor
        to the start of the chosen chunk. Folding the rank in is what makes each
        distributed rank stream a disjoint set of chunks (a ``DistributedSampler``
        cannot, since this dataset ignores ``idx``).
        """
        # Prefer the explicit rank/world_size captured at construction (in the main
        # process); spawned DataLoader workers cannot read ``torch.distributed``, so
        # without these every rank would replicate rank 0's chunk partition.
        if self._dist_rank is not None and self._dist_world_size is not None:
            rank, world_size = self._dist_rank, self._dist_world_size
        else:
            rank = (
                dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            )
            world_size = (
                dist.get_world_size()
                if dist.is_available() and dist.is_initialized()
                else 1
            )
        worker_info = get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers
        global_worker_id = rank * num_workers + worker_id
        if not hasattr(self, "_active_chunks") or self._active_chunks is None:
            indices = partition_chunk_indices(
                len(self.chunks),
                rank=rank,
                world_size=world_size,
                worker_id=worker_id,
                num_workers=num_workers,
            )
            worker_chunks = [self.chunks[i] for i in indices]
            rng = np.random.default_rng(self.seed + global_worker_id)
            rng.shuffle(worker_chunks)
            self._active_chunks = worker_chunks
        rng = np.random.default_rng(self.seed + global_worker_id)
        self.current_streaming_chunk_idx = rng.integers(
            0, len(self._active_chunks)
        ).item()
        self.current_streaming_frame_idx = self._active_chunks[
            self.current_streaming_chunk_idx
        ][0]

    def __getitem__(self, idx) -> dict:
        """Return the next streamed frame (the ``idx`` argument is ignored).

        In streaming mode the dataset maintains a per-consumer cursor over a
        disjoint slice of the keyframe chunks. The slice is selected so that:

        * each data loader worker on a rank streams a different set of chunks, and
        * each distributed rank streams a different set of chunks.

        Because the streaming dataset partitions itself here, a
        ``DistributedSampler`` has no effect on it (the sampler only reorders
        ``idx`` values, which are ignored). Folding the distributed rank into the
        chunk stride is therefore required to avoid every rank seeing identical
        data under ``torchrun``/DDP.
        """
        if not self._chunk_streaming_using_keyframe:
            item = super().__getitem__(idx)
            self._set_prompt(item)
            return item

        # Streaming mode
        if self.current_streaming_chunk_idx is None:
            self._select_streaming_chunk()

        if (
            self.current_streaming_frame_idx
            >= self._active_chunks[self.current_streaming_chunk_idx][1]
        ):
            self.current_streaming_chunk_idx += 1
            if self.current_streaming_chunk_idx >= len(self._active_chunks):
                self.current_streaming_chunk_idx = 0
            self.current_streaming_frame_idx = self._active_chunks[
                self.current_streaming_chunk_idx
            ][0]
            self._should_obs_loaders_reload = True

        item = self.hf_dataset[self.current_streaming_frame_idx]
        if "observation.task_info" in item:
            item.pop("observation.task_info")
        ep_idx = item["episode_index"].item()

        if self._should_obs_loaders_reload:
            for loader in self.obs_loaders.values():
                loader.close()
            self.obs_loaders = {}
            self.current_streaming_episode_idx = ep_idx
            for vid_key in self.meta.video_keys:
                kwargs = {}
                task_id = item["task_index"].item()
                if "rgb" in vid_key:
                    kwargs["train_rgb_type"] = self.train_rgb_type
                loader_cls = self._obs_loader_map.get(vid_key.split(".")[2])
                if loader_cls is None:
                    continue
                self.obs_loaders[vid_key] = iter(
                    loader_cls(
                        data_path=self.root,
                        task_id=task_id,
                        camera_id=vid_key.split(".")[-1],
                        demo_id=f"{ep_idx:08d}",
                        start_idx=self._active_chunks[self.current_streaming_chunk_idx][
                            2
                        ],
                        start_idx_is_keyframe=False,
                        batch_size=1,
                        stride=1,
                        **kwargs,
                    )
                )
            self._should_obs_loaders_reload = False

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(
                self.current_streaming_frame_idx, ep_idx
            )
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        task_skill = self._get_current_task_skill(item)
        weight = skill_weight(task_skill, self.skill_list)
        if not random.choices([True, False], weights=[weight, 1 - weight])[0]:
            self.current_streaming_frame_idx += 1
            for key in self.obs_loaders:
                next(self.obs_loaders[key])[0]
            return self.__getitem__(idx)

        # Skip frames that fall in gaps between skill ranges.
        if self.skill_labels is not None and self.enable_gap:
            ep_idx_val = item["episode_index"].item()
            frame_index_val = round(item["timestamp"].item() * self.fps)
            if self._is_gap_frame(ep_idx_val, frame_index_val):
                self.current_streaming_frame_idx += 1
                for key in self.obs_loaders:
                    next(self.obs_loaders[key])[0]
                return self.__getitem__(idx)

        for key in self.obs_loaders:
            item[key] = next(self.obs_loaders[key])[0]

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        self._set_prompt(item)
        self.current_streaming_frame_idx += 1
        return item

    def _get_current_task_skill(self, item: dict) -> str:
        """Look up the level-1 (skill) task text for the current frame."""
        ep_idx = item["episode_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        sub_idx = bisect.bisect_right(
            self.task_sizes[ep_idx], frame_index, hi=len(self.task_sizes[ep_idx]) - 1
        )
        task_skill = self.meta.orchestrators[ep_idx][1][sub_idx]["task"]
        return task_skill

    def _get_fine_grained_task(self, item: dict) -> str:
        """Look up the fine-grained task text (prompt) for the current frame."""
        ep_idx = item["episode_index"].item()
        task_idx = item["task_index"].item()
        frame_index = round(item["timestamp"].item() * self.fps)
        try:
            sub_idx = bisect.bisect_right(
                self.task_sizes[ep_idx],
                frame_index,
                hi=len(self.task_sizes[ep_idx]) - 1,
            )
            task_text = self.meta.orchestrators[ep_idx][self.fine_grained_level][
                sub_idx
            ]["task"]
        except Exception as e:
            logger.warning("%s failed to get subtask %s: %s", self.repo_id, item, e)
            task_text = self.meta.tasks[task_idx]
        return task_text

    def _set_prompt(self, item: dict) -> None:
        """Set the per-frame training-prompt fields on ``item``.

        Always sets ``item["task"]`` (the fine-grained main-task text). In skill mode
        it also resolves the per-frame skill text (``item["skill_label"]``) and, when
        ``use_skill`` is on, makes that skill text the training prompt
        (``item["prompt"]``, which the transform prefers over ``item["task"]``). With
        ``use_skill`` off, no ``prompt`` key is set, so the prompt is the task text.
        """
        item["task"] = self._get_fine_grained_task(item)
        if self.skill_labels is not None:
            item["skill_label"] = self._get_skill_label(item)
            if self.use_skill:
                item["prompt"] = item["skill_label"]

    def _get_query_indices(self, idx: int, ep_idx: int):
        """Compute action-horizon query indices and per-key padding masks."""
        ep_idx_pos = self.episode_data_index_pos[ep_idx]
        ep_start = self.episode_data_index["from"][ep_idx_pos]
        ep_end = self.episode_data_index["to"][ep_idx_pos]
        query_indices = {
            key: [
                max(ep_start.item(), min(ep_end.item() - 1, idx + delta))
                for delta in delta_idx
            ]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": th.BoolTensor(
                [
                    (idx + delta < ep_start.item()) | (idx + delta >= ep_end.item())
                    for delta in delta_idx
                ]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _query_videos(self, query_timestamps, ep_idx):
        """Decode the requested video frames for ``ep_idx``."""
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = self._decode_video_frames(
                video_path, query_ts, self.tolerance_s, self.video_backend
            )
            item[vid_key] = frames.squeeze(0)
        return item

    def _get_keyframe_chunk_indices(self, chunk_size=250):
        """Split each episode into contiguous ``chunk_size``-frame keyframe chunks.

        Returns a list of ``(global_start, global_end, local_start)`` tuples where
        ``global_*`` index into the flat HF dataset and ``local_start`` is the
        in-episode frame offset used to seek the per-camera video loaders.
        """
        episode_lengths = {
            ep_idx: ep_dict["length"] for ep_idx, ep_dict in self.meta.episodes.items()
        }
        episode_lengths = [episode_lengths[ep_idx] for ep_idx in self.episodes]
        chunks = []
        offset = 0
        for L in episode_lengths:
            local_starts = list(range(0, L, chunk_size))
            local_ends = local_starts[1:] + [L]
            for ls, le in zip(local_starts, local_ends):
                chunks.append((offset + ls, offset + le, ls))
            offset += L
        return chunks
