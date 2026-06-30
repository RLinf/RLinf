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

"""Self-contained BEHAVIOR-1K SFT data loader for the PyTorch pi05 path.

Streams the BEHAVIOR dataset (:class:`~.behavior_sft_dataset.BehaviorSftDataset`),
applies the per-sample :class:`BehaviorSftTransform` (state extraction, image
resize/pad, quantile normalization, pi05 discrete-state tokenization), collates
samples into a batched :class:`Observation` plus an actions tensor of shape
``[batch, action_horizon, action_dim]``, and yields ``(Observation, actions)``.

The streaming dataset partitions its keyframe chunks per ``(rank, worker)``
internally (see :meth:`BehaviorSftDataset.__getitem__`), so a
``DistributedSampler`` is intentionally *not* used: a sampler only reorders the
ignored ``idx`` values and would otherwise give every distributed rank identical
data. Every loader parameter is read directly from YAML; only the fixed BEHAVIOR
task-0000 skill-window recipe is hardcoded.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import typing

import numpy as np
import torch

from rlinf.data.datasets.openpi_pytorch.behavior.behavior_sft_dataset import (
    BehaviorSftDataset,
)
from rlinf.data.datasets.openpi_pytorch.behavior.processing import _pad_to_dim
from rlinf.data.lerobot_paths import (
    resolve_lerobot_repo_id,
)
from rlinf.models.embodiment.openpi_pytorch.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_pytorch.policies.behavior_policy import (
    BehaviorInputs,
)
from rlinf.models.embodiment.openpi_pytorch.utils.image_tools import resize_with_pad
from rlinf.models.embodiment.openpi_pytorch.utils.normalize import (
    NormStats,
    load_norm_stats,
    normalize_quantile,
)
from rlinf.models.embodiment.openpi_pytorch.utils.tokenizer import PaligemmaTokenizer

logger = logging.getLogger(__name__)

__all__ = [
    "BehaviorSftDataConfig",
    "BehaviorSftDataLoader",
    "BehaviorSftTransform",
    "build_behavior_sft_dataloader",
    "collate_behavior_sft_items",
    "create_behavior_sft_data_loader",
]

# Camera views resolved by the BEHAVIOR pi05 transform.
_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_IMAGE_SIZE = 224

# Repack mapping: BehaviorInputs key -> raw LeRobot frame key. Mirrors the old
# LeRobotB1KDataConfig RepackTransform (with the BehaviorInputs key names).
_REPACK_KEYS = {
    "observation/image": "observation.images.rgb.head",
    "observation/left_wrist_image": "observation.images.rgb.left_wrist",
    "observation/right_wrist_image": "observation.images.rgb.right_wrist",
    "observation/state": "observation.state",
}


def _repack(frame: dict) -> dict:
    """Map raw LeRobot keys onto the names ``BehaviorInputs`` expects.

    Images arrive as ``(C, H, W)`` float tensors from the streaming dataset;
    ``BehaviorInputs`` (via its ``_parse_image``) handles the channel order and
    the float-to-uint8 conversion, so they are passed through as numpy arrays.
    """
    data: dict = {}
    for dst, src in _REPACK_KEYS.items():
        data[dst] = np.asarray(frame[src])

    actions = frame.get("action")
    if actions is not None:
        data["actions"] = np.asarray(actions)

    prompt = frame.get("prompt", frame.get("task"))
    if prompt is None:
        raise ValueError(
            "BEHAVIOR SFT frame is missing both 'prompt' and 'task'; the streaming "
            "dataset must set the per-frame task text."
        )
    if not isinstance(prompt, str):
        prompt = prompt.item() if hasattr(prompt, "item") else str(prompt)
    data["prompt"] = prompt
    return data


@dataclasses.dataclass
class BehaviorSftTransform:
    """Map a raw BEHAVIOR LeRobot frame to pi05 model inputs + padded actions.

    Reproduces the old chain (repack -> ``BehaviorInputs`` -> quantile-Normalize
    state/actions -> resize images -> pi05 discrete-state tokenize -> pad state and
    actions to ``action_dim``). Images stay ``uint8`` through resize; the final
    ``uint8 -> float[-1, 1]`` conversion happens in :meth:`Observation.from_dict`.

    Args:
        norm_stats: Quantile normalization statistics keyed by ``"state"`` and
            ``"actions"`` (as loaded from the checkpoint ``norm_stats.json``).
        tokenizer_path: Filesystem path to the PaliGemma SentencePiece model.
        action_dim: Model action dimension to pad the state and actions to.
        max_token_len: Maximum tokenized-prompt length.
        image_size: Target square image resolution.
        tokenizer: Optional pre-built tokenizer. A new
            :class:`PaligemmaTokenizer` is created lazily per worker when ``None``
            so the (non-picklable) SentencePiece processor is not shared across
            ``spawn`` workers.
    """

    norm_stats: dict[str, NormStats]
    tokenizer_path: str
    action_dim: int = 32
    max_token_len: int = 200
    image_size: int = _IMAGE_SIZE
    tokenizer: PaligemmaTokenizer | None = None

    def __post_init__(self):
        self._behavior_inputs = BehaviorInputs(
            extract_state_from_proprio=True,
            use_all_wrist_images=True,
        )

    def _get_tokenizer(self) -> PaligemmaTokenizer:
        if self.tokenizer is None:
            self.tokenizer = PaligemmaTokenizer(
                self.tokenizer_path, max_len=self.max_token_len
            )
        return self.tokenizer

    def __call__(self, frame: dict) -> dict:
        """Transform a single raw LeRobot frame into the model-input dict."""
        # Repack LeRobot keys -> BehaviorInputs keys (+ prompt), then run
        # BehaviorInputs (23-dim state extraction, image-key mapping, masks).
        repacked = _repack(frame)
        inputs = self._behavior_inputs(repacked)

        # Resize each camera image to image_size x image_size (uint8 in/out).
        images = {
            key: resize_with_pad(
                np.asarray(inputs["image"][key]), self.image_size, self.image_size
            )
            for key in _IMAGE_KEYS
        }

        # Quantile-normalize the (still 23-dim) state and actions to [-1, 1] BEFORE
        # padding, tokenize the pi05 discrete-state prompt on the normalized state,
        # then zero-pad the state and actions to the model action dimension.
        state = np.asarray(inputs["state"], dtype=np.float32)
        state = normalize_quantile(state, self.norm_stats["state"]).astype(np.float32)
        actions = np.asarray(inputs["actions"], dtype=np.float32)
        actions = normalize_quantile(actions, self.norm_stats["actions"]).astype(
            np.float32
        )
        tokens, token_masks = self._get_tokenizer().tokenize(inputs["prompt"], state)
        state = _pad_to_dim(state, self.action_dim).astype(np.float32)
        actions = _pad_to_dim(actions, self.action_dim).astype(np.float32)

        return {
            "image": images,
            "image_mask": {
                key: np.asarray(inputs["image_mask"][key]) for key in _IMAGE_KEYS
            },
            "state": state,
            "actions": actions,
            "tokenized_prompt": np.asarray(tokens),
            "tokenized_prompt_mask": np.asarray(token_masks),
        }


@dataclasses.dataclass(frozen=True)
class BehaviorSftDataConfig:
    """Metadata describing the BEHAVIOR SFT data pipeline.

    Exposed via :meth:`BehaviorSftDataLoader.data_config` so the SFT worker can
    read the resolved repo id, action dimension, action horizon, and the
    normalization statistics without reaching into the dataset internals.
    """

    repo_id: str
    action_dim: int
    action_horizon: int
    max_token_len: int
    norm_stats: dict[str, NormStats]


class _TransformedStreamingDataset(torch.utils.data.Dataset):
    """Wrap the streaming dataset, applying the per-sample SFT transform.

    The transform holds a (non-picklable) SentencePiece tokenizer; it is built
    lazily inside each ``spawn`` worker on first use, so only the lightweight
    :class:`BehaviorSftTransform` config travels across the process boundary.
    """

    def __init__(self, dataset: BehaviorSftDataset, transform: BehaviorSftTransform):
        self._dataset = dataset
        self._transform = transform

    def __getitem__(self, idx):
        return self._transform(self._dataset[idx])

    def __len__(self) -> int:
        # The streaming dataset ignores `idx` and partitions chunks internally;
        # `len` only drives torch's default index sampler so iteration proceeds.
        return len(self._dataset.hf_dataset)


def collate_behavior_sft_items(
    items: typing.Sequence[typing.Mapping[str, typing.Any]],
) -> tuple[Observation, torch.Tensor]:
    """Collate transformed items into ``(Observation, actions)``.

    Images are stacked as ``uint8`` ``[B, H, W, C]`` tensors and converted to
    ``float32`` in ``[-1, 1]`` by :meth:`Observation.from_dict` (matching the old
    path). State/actions/tokens are stacked into the appropriate torch dtypes;
    the returned actions tensor has shape ``[batch, action_horizon, action_dim]``.
    """
    if not items:
        raise ValueError("Cannot collate an empty BEHAVIOR SFT batch.")

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    batch = {
        "image": images,
        "image_mask": image_masks,
        "state": torch.from_numpy(
            np.stack([np.asarray(item["state"], dtype=np.float32) for item in items])
        ),
        "tokenized_prompt": torch.from_numpy(
            np.stack(
                [np.asarray(item["tokenized_prompt"], dtype=np.int64) for item in items]
            )
        ).long(),
        "tokenized_prompt_mask": torch.from_numpy(
            np.stack(
                [
                    np.asarray(item["tokenized_prompt_mask"], dtype=np.bool_)
                    for item in items
                ]
            )
        ),
    }
    actions = torch.from_numpy(
        np.stack([np.asarray(item["actions"], dtype=np.float32) for item in items])
    )
    return Observation.from_dict(batch), actions


def _worker_init_fn(worker_id: int) -> None:
    """Per-worker init hook (placeholder for worker-local environment setup)."""
    del worker_id


def create_behavior_sft_data_loader(
    *,
    behavior_dataset_root: str,
    assets_dir: str,
    asset_id: str,
    tokenizer_path: str,
    repo_id: str,
    tasks: list[str],
    modalities: list[str],
    action_dim: int,
    action_horizon: int,
    max_token_len: int,
    batch_size: int,
    num_workers: int,
    fine_grained_level: int,
    tolerance_s: float,
    shuffle: bool,
    seed: int,
    skill_labels: dict[int, str] | None,
    use_skill: bool,
    enable_gap: bool,
    allow_left: int,
    allow_right: int,
    dist_rank: int,
    dist_world_size: int,
) -> "BehaviorSftDataLoader":
    """Build the BEHAVIOR-1K SFT data loader yielding ``(Observation, actions)``.

    Args:
        behavior_dataset_root: Local root of the LeRobot BEHAVIOR dataset.
        assets_dir: Directory holding the checkpoint assets (norm stats).
        asset_id: Sub-directory under ``assets_dir`` for the norm stats
            (``{assets_dir}/{asset_id}/norm_stats.json``).
        tokenizer_path: Filesystem path to the PaliGemma SentencePiece model.
        repo_id: LeRobot dataset repo id (used for metadata bookkeeping).
        tasks: BEHAVIOR task names to include.
        modalities: Observation modalities to load (e.g. ``["rgb"]``).
        action_dim: Model action dimension to pad state/actions to.
        action_horizon: Number of future action steps per sample.
        max_token_len: Maximum tokenized-prompt length.
        batch_size: Per-rank batch size.
        num_workers: Number of ``DataLoader`` workers (``> 0`` uses ``spawn``).
        fine_grained_level: Orchestrator level for the prompt task text.
        tolerance_s: Frame-timestamp sync tolerance.
        shuffle: Whether the streaming dataset shuffles its chunk order.
        seed: Base seed for the streaming chunk partition.
        skill_labels: Optional per-skill labels enabling skill mode.
        use_skill: Train on per-frame SKILL text (window-resolved) instead of the
            main-task text; requires explicit ``skill_labels``.
        enable_gap: Skill mode — absorb a true gap into both adjacent skills.
        allow_left: Skill mode — frames to extend a contiguous skill start left.
        allow_right: Skill mode — frames to extend a contiguous skill end right.
        dist_rank: This rank's id, threaded into the per-rank chunk partition.
        dist_world_size: Total ranks, threaded into the per-rank chunk partition.

    Returns:
        A loader whose iteration yields ``(Observation, actions)`` 2-tuples.
    """
    norm_stats = load_norm_stats(assets_dir, asset_id)
    logger.info("Loaded BEHAVIOR norm stats from %s/%s", assets_dir, asset_id)

    dataset = BehaviorSftDataset(
        repo_id=repo_id,
        root=behavior_dataset_root,
        tolerance_s=tolerance_s,
        tasks=tasks or None,
        modalities=modalities or ["rgb"],
        local_only=True,
        delta_timestamps={"action": [t / 30.0 for t in range(action_horizon)]},
        chunk_streaming_using_keyframe=True,
        shuffle=shuffle,
        seed=seed,
        fine_grained_level=fine_grained_level,
        skill_labels=skill_labels,
        use_skill=use_skill,
        enable_gap=enable_gap,
        allow_left=allow_left,
        allow_right=allow_right,
        dist_rank=dist_rank,
        dist_world_size=dist_world_size,
    )

    transform = BehaviorSftTransform(
        norm_stats=norm_stats,
        tokenizer_path=tokenizer_path,
        action_dim=action_dim,
        max_token_len=max_token_len,
    )
    source = _TransformedStreamingDataset(dataset, transform)

    # The streaming dataset partitions chunks per (rank, worker) on its own, so a
    # DistributedSampler is intentionally omitted: it would only reorder the
    # ignored `idx` values and give every distributed rank identical data.
    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None

    generator = torch.Generator()
    generator.manual_seed(seed)

    logger.info(
        "BEHAVIOR SFT data loader: batch_size=%d, num_workers=%d, action_horizon=%d",
        batch_size,
        num_workers,
        action_horizon,
    )

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, source),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=None,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=collate_behavior_sft_items,
        worker_init_fn=_worker_init_fn,
        drop_last=True,
        generator=generator,
    )

    data_config = BehaviorSftDataConfig(
        repo_id=repo_id,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        norm_stats=norm_stats,
    )
    return BehaviorSftDataLoader(torch_loader, data_config)


class BehaviorSftDataLoader:
    """Infinite ``(Observation, actions)`` loop over the BEHAVIOR SFT dataset.

    Re-iterates the underlying ``torch`` ``DataLoader`` forever. Each batch is
    already collated into an :class:`Observation` plus an actions tensor of shape
    ``[batch, action_horizon, action_dim]`` by :func:`collate_behavior_sft_items`.
    """

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: BehaviorSftDataConfig,
    ):
        self._torch_loader = torch_loader
        self._data_config = data_config

    def data_config(self) -> BehaviorSftDataConfig:
        """Return the resolved data-pipeline metadata."""
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        """Expose the underlying ``torch`` ``DataLoader``."""
        return self._torch_loader

    def __iter__(self):
        while True:
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)


def build_behavior_sft_dataloader(
    cfg, world_size, rank, data_paths, eval_dataset=False
):
    """Build the self-contained BEHAVIOR SFT data loader for the SFT worker.

    The streaming dataset partitions chunks per ``(rank, worker)``; ``rank`` /
    ``world_size`` are captured here (in the main process) and threaded into the
    dataset so that SPAWNED DataLoader workers -- which cannot read
    ``torch.distributed`` -- still partition by the correct per-rank id (otherwise
    every rank replicates rank 0's chunks, collapsing the effective batch to one
    rank's micro-batch). Every parameter is read directly from YAML (no hidden
    defaults). Returns ``(loader, loader.data_config())``.
    """
    data_path = resolve_lerobot_repo_id(data_paths)
    if data_path is None:
        raise ValueError("openpi_pytorch BEHAVIOR SFT requires data.train_data_paths.")

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    # Norm stats + tokenizer resolve STRICTLY from YAML (no checkpoint-relative
    # fallback); load_norm_stats rejects a blank assets_dir/asset_id the same way
    # the eval model factory does, so neither path can silently load non-task-0000
    # stats.
    assets_dir = model_cfg.openpi.assets_dir
    asset_id = model_cfg.openpi.asset_id
    tokenizer_path = model_cfg.openpi.paligemma_tokenizer

    # `cfg.data` is the production source of truth for the BEHAVIOR task set and the
    # prompt-source flag. `use_skill: true` trains on the per-frame REFERENCE skill
    # text; `false` trains on the main-task text.
    use_skill = bool(data_cfg.use_skill)
    tasks = list(data_cfg.tasks)
    skill_labels, enable_gap, allow_left, allow_right = None, True, 0, 0
    if use_skill:
        # The skill labels are the REFERENCE per-task subtask list from config (NOT
        # the dataset's collapsed orchestrators, which equal the full task text). The
        # task-0000 local-skill recipe is exactly one task with a configured subtask
        # list and the fixed window recipe below.
        if len(tasks) != 1:
            raise ValueError(
                "openpi_pytorch BEHAVIOR SFT use_skill:true supports exactly one task "
                f"(the task-0000 skill recipe); got data.tasks={tasks}."
            )
        subtask_labels = data_cfg.task_subtasks
        labels = subtask_labels.get(tasks[0]) if subtask_labels else None
        if not labels:
            raise ValueError(
                "openpi_pytorch BEHAVIOR SFT use_skill:true requires the reference "
                f"skill labels at data.task_subtasks.{tasks[0]}; none was configured."
            )
        skill_labels = {i: str(label) for i, label in enumerate(labels)}
        # Fixed reference skill-window recipe (pi05_b1k-task0000_sft_local_skill);
        # intentionally hardcoded so the reference recipe cannot drift via config.
        enable_gap, allow_left, allow_right = True, 100, 100

    loader = create_behavior_sft_data_loader(
        behavior_dataset_root=str(data_cfg.behavior_dataset_root),
        assets_dir=str(assets_dir),
        asset_id=asset_id,
        tokenizer_path=str(tokenizer_path),
        repo_id=str(data_cfg.repo_id),
        tasks=tasks,
        modalities=list(data_cfg.modalities),
        action_dim=int(model_cfg.openpi.model_action_dim),
        action_horizon=int(model_cfg.num_action_chunks),
        max_token_len=int(model_cfg.openpi.max_token_len),
        batch_size=int(cfg.actor.eval_batch_size)
        if eval_dataset
        else int(cfg.actor.micro_batch_size),
        num_workers=int(data_cfg.num_workers),
        fine_grained_level=int(data_cfg.fine_grained_level),
        tolerance_s=float(data_cfg.tolerance_s),
        shuffle=not eval_dataset,
        seed=int(cfg.actor.seed),
        skill_labels=skill_labels,
        use_skill=use_skill,
        enable_gap=enable_gap,
        allow_left=allow_left,
        allow_right=allow_right,
        dist_rank=rank,
        dist_world_size=world_size,
    )
    return loader, loader.data_config()
