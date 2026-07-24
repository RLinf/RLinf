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

"""FSDP SFT worker for the :class:`SteamCriticModel`.

Parallel to :class:`FSDPValueSftWorker` (scalar value regression) but
built around:

* :class:`~rlinf.data.datasets.steam.PairDataset` — yields
  ``(frame_t, frame_{t+k})`` pairs with 1:1 progress/regress balance.
* :class:`~rlinf.data.datasets.steam.mixture.PairMixtureDataset` —
  optional weighted sampling across multiple pair datasets.
* :class:`~rlinf.data.datasets.steam.BinaryPairDataCollator` — runs
  the multi-view :class:`SteamProcessor` twice (once per frame) and
  stacks the per-camera image tensors along a new ``num_frames`` axis.
* The binary forward contract ``model(observation, labels)`` with labels as
  long bin indices in ``[0, num_bins)`` (binary: 0 = regress, 1 = progress).
  Metrics reported per step / eval run: cross-entropy loss, classification
  accuracy, and the distribution of ``P(progress)``.
"""

import logging
import os
from pathlib import Path
from typing import Any

os.environ["LIBAV_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("av").setLevel(logging.ERROR)

import torch  # noqa: E402
from omegaconf import DictConfig, open_dict  # noqa: E402

from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager  # noqa: E402
from rlinf.models import get_model  # noqa: E402
from rlinf.scheduler import Worker  # noqa: E402
from rlinf.utils.distributed import all_reduce_dict  # noqa: E402


class _PairDataLoaderImpl:
    """Thin wrapper so ``SFTRunner`` / ``set_global_step`` can swap epochs."""

    def __init__(self, data_loader: torch.utils.data.DataLoader):
        self._data_loader = data_loader

    def __len__(self) -> int:
        return len(self._data_loader)

    def set_epoch(self, epoch: int) -> None:
        sampler = getattr(self._data_loader, "sampler", None)
        if sampler is not None and hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        dataset = getattr(self._data_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

    def __iter__(self):
        yield from self._data_loader


def _pair_dataset_camera_shapes(dataset) -> dict[str, tuple[int, ...] | None]:
    """Return representative raw camera shapes from the dataset's first sample."""
    sample = dataset[0]
    shapes: dict[str, tuple[int, ...] | None] = {}
    for camera_key, image in sample["image_t"].items():
        shapes[camera_key] = (
            tuple(int(dim) for dim in image.shape) if image is not None else None
        )
    for camera_key in sample["image_mask_t"].keys():
        shapes.setdefault(camera_key, None)
    return shapes


def _validate_train_dataset_shapes(
    named_datasets: list[tuple[str, Any]],
) -> None:
    """Fail fast when a train mixture would combine incompatible image sizes."""
    if len(named_datasets) <= 1:
        return

    reference_name, reference_dataset = named_datasets[0]
    reference_shapes = _pair_dataset_camera_shapes(reference_dataset)
    mismatches: list[str] = []

    for dataset_name, dataset in named_datasets[1:]:
        shapes = _pair_dataset_camera_shapes(dataset)
        differing = {
            camera_key: (reference_shapes.get(camera_key), shapes.get(camera_key))
            for camera_key in sorted(set(reference_shapes) | set(shapes))
            if reference_shapes.get(camera_key) != shapes.get(camera_key)
        }
        if differing:
            mismatches.append(
                f"{dataset_name}: {differing} (reference {reference_name}: "
                f"{reference_shapes})"
            )

    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(
            "Binary value train mixtures require identical raw camera shapes "
            "across all train_data_paths because PairDataset does not resize "
            f"images inside the collator. Incompatible shapes: {mismatch_text}"
        )


def _collect_non_finite_tensor_paths(value: Any, prefix: str) -> list[str]:
    """Return dotted tensor paths whose values contain NaN/Inf."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 0 or torch.isfinite(value.detach()).all():
            return []
        return [prefix]

    if isinstance(value, dict):
        bad_paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            bad_paths.extend(_collect_non_finite_tensor_paths(child, child_prefix))
        return bad_paths

    if isinstance(value, (list, tuple)):
        bad_paths = []
        for idx, child in enumerate(value):
            child_prefix = f"{prefix}[{idx}]"
            bad_paths.extend(_collect_non_finite_tensor_paths(child, child_prefix))
        return bad_paths

    return []


class FSDPSteamSftWorker(FSDPModelManager, Worker):
    """FSDP worker for the STEAM value critic.

    Reads ``data.train_data_paths`` (list of LeRobot dataset entries) and
    ``data.eval_data_paths`` from the Hydra config; builds a
    :class:`PairDataset` per entry and feeds every batch through the binary
    forward ``model(observation=obs, labels=labels)``.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        # 1、解析 tokenizer；2、创建 SteamImageProcessor；3、 创建 SteamProcessor
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        self.torch_platform.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = self.torch_platform.current_device()

        self.data_loader, self.eval_data_loaders = self.build_dataloader()
        self.data_iter = iter(self.data_loader)
        self._gradient_checkpointing_state: bool | None = None

    def init_worker(self):
        self.setup_model_and_optimizer()
        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def _ensure_steam_precision_cfg(self, model_cfg: DictConfig) -> str:
        """Default unset binary-value precision to fp32 for stable FSDP training.

        The model load dtype controls FSDP's master parameter dtype. Loading in
        bf16 makes the master copy bf16 too, which collapses Adam's second
        moment after a couple of steps and produces non-finite ``hidden_states``
        inside the SigLIP/Gemma backbone. Default to fp32 master and let
        ``fsdp_config.mixed_precision.param_dtype`` decide the forward dtype.
        """
        precision = getattr(model_cfg, "precision", None)
        if precision not in (None, "", "null"):
            return str(precision)

        with open_dict(model_cfg):
            model_cfg.precision = "fp32"

        self._logger.warning(
            "[SteamSFT] actor.model.precision was unset; defaulting to fp32 "
            "so FSDP keeps an fp32 master copy. Forward compute dtype is still "
            "controlled by fsdp_config.mixed_precision.param_dtype."
        )
        return str(model_cfg.precision)

    def model_provider_func(self) -> torch.nn.Module:
        self._ensure_steam_precision_cfg(self.cfg.actor.model)
        ensemble_size = int(getattr(self.cfg.actor.model, "ensemble_size", 1))
        if (
            ensemble_size > 1
            and getattr(self.cfg.actor.model, "ensemble_head_seed_base", None) is None
        ):
            with open_dict(self.cfg.actor.model):
                self.cfg.actor.model.ensemble_head_seed_base = int(self.cfg.actor.seed)
        return get_model(self.cfg.actor.model)

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save weights plus lightweight checkpoint-side model assets."""
        super().save_checkpoint(save_path, step)

        if self._rank == 0:
            from rlinf.models.embodiment.value_model.steam import (
                save_steam_checkpoint_assets,
            )

            save_steam_checkpoint_assets(
                save_path=save_path,
                cfg=self.cfg.actor.model,
                processor=getattr(self, "processor", None),
            )

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def build_dataloader(self):
        """Build train + eval dataloaders from ``data.*`` config."""
        try:
            import av

            av.logging.set_level(av.logging.ERROR)
        except (ImportError, AttributeError):
            pass

        from rlinf.data.datasets.steam import (
            BinaryPairDataCollator,
            PairDataset,
        )
        from rlinf.data.datasets.steam.mixture import PairMixtureDataset
        from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
            has_tokenizer_files,
        )
        from rlinf.models.embodiment.value_model.steam.processing import (
            SteamImageProcessor,
            SteamProcessor,
            resolve_vision_image_size,
        )

        data_cfg = self.cfg.get("data", {})
        model_cfg = self.cfg.actor.model

        # --- Tokenizer resolution ---
        tokenizer_path = getattr(model_cfg, "tokenizer_path", None) or getattr(
            model_cfg, "language_repo_id", None
        )
        if tokenizer_path is None or not has_tokenizer_files(Path(tokenizer_path)):
            raise ValueError(
                "No tokenizer found. Set actor.model.tokenizer_path or "
                f"actor.model.language_repo_id. Tried: {tokenizer_path!r}"
            )
        # Processor camera keys must match the dataset's `camera_keys` so that
        # the per-frame image dict emitted by the dataset lines up with the
        # processor's expected slots. Any absent camera → zero-placeholder +
        # mask=False at the processor stage.
        camera_keys_tuple = tuple(
            data_cfg.get(
                "camera_keys",
                (
                    "base_0_rgb",
                    "left_wrist_0_rgb",
                    "right_wrist_0_rgb",
                ),
            )
        )
        image_size = resolve_vision_image_size(
            getattr(model_cfg, "vision_repo_id"),
            revision=getattr(model_cfg, "vision_revision", None),
        )
        image_processor = SteamImageProcessor(
            image_size=image_size,
            image_keys=camera_keys_tuple,
            do_augment=bool(data_cfg.get("do_augment", True)),
        )
        self._logger.info(
            "Binary value image processor uses %sx%s for vision_repo_id=%s",
            image_size[0],
            image_size[1],
            getattr(model_cfg, "vision_repo_id"),
        )
        processor = SteamProcessor(
            image_processor=image_processor,
            tokenizer_name_or_path=tokenizer_path,
            max_token_len=int(getattr(model_cfg, "max_token_len", 200)),
        )
        self.processor = processor
        max_token_len = int(getattr(model_cfg, "max_token_len", 200))
        # num_bins drives both the dataset (label semantics) and the
        # collator (label tensor dtype); passing the same value to both
        # keeps the head / loss / metrics contract consistent.
        num_bins = int(getattr(model_cfg, "num_bins", 2))
        k = int(data_cfg.get("k", 4))
        with open_dict(model_cfg):
            model_cfg.stride_k = k
        if num_bins > 2 and (2 * k) % num_bins != 0:
            raise ValueError(
                "Binary value multi-bin mode requires 2*data.k to be a "
                f"multiple of model.num_bins; got data.k={k}, "
                f"model.num_bins={num_bins} (2*k={2 * k})."
            )
        train_collator = BinaryPairDataCollator(
            processor=processor,
            max_length=max_token_len,
            train=True,
            num_bins=num_bins,
        )
        eval_collator = BinaryPairDataCollator(
            processor=processor,
            max_length=max_token_len,
            train=False,
            num_bins=num_bins,
        )

        # --- DataLoader knobs ---
        pin_memory = bool(data_cfg.get("pin_memory", True))
        train_num_workers = int(data_cfg.get("train_num_workers", 0))
        eval_num_workers = int(data_cfg.get("eval_num_workers", train_num_workers))
        prefetch_factor = data_cfg.get("prefetch_factor", 2)
        persistent_workers = bool(data_cfg.get("persistent_workers", True))

        def _loader_worker_kwargs(num_workers: int) -> dict:
            kwargs: dict[str, Any] = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = persistent_workers
                if prefetch_factor is not None:
                    kwargs["prefetch_factor"] = int(prefetch_factor)
            return kwargs

        # --- Dataset construction ---
        data_root = data_cfg.get("data_root", None)

        def _resolve(path: str) -> str:
            if data_root and not os.path.isabs(path):
                return os.path.join(data_root, path)
            return path

        def _build_pair_dataset(entry: dict) -> PairDataset:
            ds_path = _resolve(entry["dataset_path"])
            if "type" in entry:
                dataset_type = str(entry["type"]).lower()
            elif "dataset_type" in data_cfg:
                dataset_type = str(data_cfg.get("dataset_type")).lower()
            else:
                raise ValueError(
                    "Binary value PairDataset requires an explicit data.dataset_type "
                    f"or train_data_paths[*].type for dataset_path={ds_path!r}."
                )
            if dataset_type not in ("sft", "rollout"):
                raise ValueError(
                    f"Binary value dataset type must be 'sft' or 'rollout', "
                    f"got {dataset_type!r} for dataset_path={ds_path!r}."
                )
            if "only_success" in entry:
                only_success = entry["only_success"]
            elif "only_success" in data_cfg:
                only_success = data_cfg.get("only_success")
            else:
                raise ValueError(
                    "Binary value PairDataset requires an explicit data.only_success "
                    f"or train_data_paths[*].only_success for dataset_path={ds_path!r}."
                )
            # 长度归一化标签相关配置，走默认值：
            # length_scale_enabled = False
            # length_scale_percentile = 90.0
            length_scale_enabled = bool(
                entry.get(
                    "length_scale_enabled",
                    data_cfg.get("length_scale_enabled", False),
                )
            )
            length_scale_percentile = float(
                entry.get(
                    "length_scale_percentile",
                    data_cfg.get("length_scale_percentile", 90.0),
                )
            )
            return PairDataset(
                dataset_path=ds_path,
                camera_keys=tuple(
                    # 读取相机keys
                    data_cfg.get(
                        "camera_keys",
                        (
                            "base_0_rgb",
                            "left_wrist_0_rgb",
                            "right_wrist_0_rgb",
                        ),
                    )
                ),
                k=k,
                dataset_type=dataset_type,
                only_success=only_success,
                min_episode_length=data_cfg.get("min_episode_length", None),
                num_bins=num_bins,
                length_scale_enabled=length_scale_enabled,
                length_scale_percentile=length_scale_percentile,
            )

        balance_dataset_weights = bool(
            data_cfg.get(
                "balance_weights",
                data_cfg.get("balance_dataset_weights", True),
            )
        )

        # --- Train datasets (supports weighted mixture across pair datasets) ---
        train_entries = [
            dict(e)
            for e in data_cfg.get("train_data_paths", [])
            if e.get("dataset_path")
        ]
        if not train_entries:
            raise ValueError(
                "data.train_data_paths must contain at least one entry with 'dataset_path'."
            )
        datasets_with_weights: list[tuple[PairDataset, float]] = []
        named_train_datasets: list[tuple[str, PairDataset]] = []
        for entry in train_entries:
            resolved_path = _resolve(entry["dataset_path"])
            dataset = _build_pair_dataset(entry)
            weight = float(entry.get("weight", 1.0))
            datasets_with_weights.append((dataset, weight))
            named_train_datasets.append((resolved_path, dataset))
            self._logger.info(
                "[SteamSFT] Loaded train dataset: %s "
                "(type=%s, %d samples, weight=%.4f)",
                resolved_path,
                entry.get("type", data_cfg.get("dataset_type")),
                len(dataset),
                weight,
            )

        _validate_train_dataset_shapes(named_train_datasets)

        # Length-scale reference (L_max): pool eligible-episode lengths across
        # every train dataset so one global L_max governs the whole mixture —
        # a fixed frame stride then maps to the same bin regardless of which
        # dataset the episode came from. Applied to eval datasets too below.
        # np.percentile is deterministic, so every rank derives the same value
        # without a collective.
        global_length_scale_reference: float | None = None
        if any(ds.length_scale_enabled for ds, _ in datasets_with_weights):
            percentile = float(data_cfg.get("length_scale_percentile", 90.0))
            global_length_scale_reference = (
                PairDataset.compute_global_length_scale_reference(
                    [ds for ds, _ in datasets_with_weights],
                    percentile,
                )
            )
            for ds, _ in datasets_with_weights:
                ds.set_length_scale_reference(global_length_scale_reference)
            self._logger.info(
                "[SteamSFT] Global length-scale L_max (p%.1f) = %.2f, applied to "
                "%d train dataset(s).",
                percentile,
                global_length_scale_reference,
                len(datasets_with_weights),
            )

        if len(datasets_with_weights) == 1:
            train_dataset: torch.utils.data.Dataset = datasets_with_weights[0][0]
        else:
            train_dataset = PairMixtureDataset(
                datasets=datasets_with_weights,
                mode="train",
                balance_dataset_weights=balance_dataset_weights,
                seed=int(data_cfg.get("seed", 42)),
            )
        self._logger.info(
            "[SteamSFT] Train: %d dataset(s), %d samples total",
            len(datasets_with_weights),
            len(train_dataset),
        )

        # Per-member training treats each ensemble member's training as an
        # independent SGD trajectory: ``actor.micro_batch_size`` and
        # ``actor.global_batch_size`` describe one member's batching.
        # Per global step the dataloader is consumed
        # ``ensemble_size * grad_accum`` times (each member fetches its
        # own ``grad_accum`` micro batches), so the actual sample
        # throughput per rank is ``ensemble_size * grad_accum *
        # micro_batch_size``.
        ensemble_size = max(1, int(getattr(model_cfg, "ensemble_size", 1)))
        train_loader_batch_size = int(self.cfg.actor.micro_batch_size)
        if ensemble_size > 1:
            self._logger.info(
                "[SteamSFT] Per-member dataloader batch_size=%d "
                "(micro_batch_size is interpreted as the per-member micro "
                "batch; each global step consumes ensemble_size=%d × "
                "grad_accum batches from the loader).",
                train_loader_batch_size,
                ensemble_size,
            )

        train_sampler = None
        if torch.distributed.is_initialized():
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                drop_last=True,
            )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=train_loader_batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            drop_last=True,
            collate_fn=train_collator,
            **_loader_worker_kwargs(train_num_workers),
        )

        # --- Eval datasets ---
        eval_entries = data_cfg.get("eval_data_paths", []) or []
        eval_data_loaders: list[tuple[str, _PairDataLoaderImpl]] = []
        for entry in eval_entries:
            entry = dict(entry)
            if not entry.get("dataset_path"):
                continue
            ds = _build_pair_dataset(entry)
            if global_length_scale_reference is not None:
                ds.set_length_scale_reference(global_length_scale_reference)
            name = entry.get("name", Path(_resolve(entry["dataset_path"])).stem)
            eval_sampler = None
            if torch.distributed.is_initialized():
                eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    ds,
                    num_replicas=self._world_size,
                    rank=self._rank,
                    shuffle=False,
                    drop_last=False,
                )
            eval_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.cfg.actor.micro_batch_size,
                shuffle=False,
                sampler=eval_sampler,
                drop_last=False,
                collate_fn=eval_collator,
                **_loader_worker_kwargs(eval_num_workers),
            )
            eval_data_loaders.append((name, _PairDataLoaderImpl(eval_loader)))
            self._logger.info(
                "[SteamSFT] Eval '%s': %d samples",
                name,
                len(ds),
            )

        return _PairDataLoaderImpl(train_loader), eval_data_loaders

    # ------------------------------------------------------------------
    # Training / eval steps
    # ------------------------------------------------------------------

    def _prepare_input(self, batch: dict):
        """Move a collated batch to device and split into model inputs."""

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device, non_blocking=True)
            if isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x

        observation = _to_device(batch["observation"])
        labels = _to_device(batch["labels"])
        return observation, labels

    def _raise_if_non_finite_training_step(
        self,
        result,
        observation: dict[str, Any],
        labels: torch.Tensor,
        micro_batch_idx: int,
        grad_accum: int,
    ) -> None:
        """Fail fast with actionable diagnostics when training tensors blow up."""
        bad_paths: list[str] = []
        bad_paths.extend(_collect_non_finite_tensor_paths(observation, "observation"))
        bad_paths.extend(_collect_non_finite_tensor_paths(labels, "labels"))

        for field_name in (
            "loss",
            "predicted_values",
            "progress_values",
            "logits",
            "probs",
            "hidden_states",
        ):
            field_value = getattr(result, field_name, None)
            bad_paths.extend(
                _collect_non_finite_tensor_paths(field_value, f"result.{field_name}")
            )

        if not bad_paths:
            return

        label_pos_frac = self._label_pos_frac(labels)
        global_step = int(getattr(self, "global_step", 0))
        precision = getattr(self.cfg.actor.model, "precision", None)
        unique_paths = ", ".join(sorted(set(bad_paths)))
        raise FloatingPointError(
            "Non-finite tensor detected during binary value SFT "
            f"(global_step={global_step}, micro_batch={micro_batch_idx + 1}/{grad_accum}, "
            f"precision={precision}, label_pos_frac={label_pos_frac:.4f}). "
            f"Offending tensors: {unique_paths}. "
            "Check precision settings and upstream batch contents."
        )

    def _label_pos_frac(self, labels: torch.Tensor) -> float:
        """Fraction of "progressive" examples in the batch.

        ``labels`` are long bin indices in ``[0, num_bins)``; progressive
        bins are the upper half ``[num_bins // 2, num_bins)`` (see
        ``pair_dataset._signed_stride_to_bin``). For ``num_bins == 2``
        this degenerates to ``labels == 1``.
        """
        num_bins = int(getattr(self.cfg.actor.model, "num_bins", 2))
        positive_mask = labels >= (num_bins // 2)
        return float(positive_mask.to(dtype=torch.float32).mean().item())

    @staticmethod
    def _critic_output_metrics(result, *, include_std: bool) -> dict[str, float]:
        """Flatten a CriticOutput into a scalar metric dict.

        ``include_std`` adds ``signed_progress_std`` (eval only): on the
        training path each ensemble member sees its own shuffled micro
        batch, so within-batch std is shuffle noise, not ensemble
        disagreement — eval covers the spread question on shared inputs.
        """
        metrics: dict[str, float] = {}
        if result.loss is not None:
            metrics["loss"] = float(result.loss.detach().item())
        if result.cat_acc_best is not None:
            metrics["accuracy"] = float(result.cat_acc_best.detach().item())
        if result.cat_acc_neighbor is not None:
            metrics["accuracy_neighbor"] = float(
                result.cat_acc_neighbor.detach().item()
            )
        if result.predicted_values is not None:
            signed_progress = result.predicted_values.detach().float()
            metrics["signed_progress_mean"] = float(signed_progress.mean().item())
            if include_std:
                metrics["signed_progress_std"] = (
                    float(signed_progress.std(unbiased=False).item())
                    if signed_progress.numel() >= 2
                    else 0.0
                )
        if result.progress_values is not None:
            metrics["progress_value_mean"] = float(
                result.progress_values.detach().float().mean().item()
            )
        return metrics

    def _log_training_batch_diagnostics_once(
        self,
        observation: dict[str, Any],
        labels: torch.Tensor,
        result,
    ) -> None:
        """Log one rank-0 batch snapshot to catch empty inputs or bad labels."""
        if getattr(self, "_logged_steam_batch_diagnostics", False):
            return
        self._logged_steam_batch_diagnostics = True
        if int(getattr(self, "_rank", 0)) != 0:
            return

        num_bins = int(getattr(self.cfg.actor.model, "num_bins", 2))
        label_hist = torch.bincount(
            labels.detach().to(dtype=torch.long).cpu(),
            minlength=num_bins,
        ).tolist()

        image_stats = {}
        for camera_key, images in observation.get("images", {}).items():
            images_f = images.detach().float()
            masks = observation.get("image_masks", {}).get(camera_key)
            image_stats[camera_key] = {
                "shape": tuple(int(dim) for dim in images.shape),
                "mean": round(float(images_f.mean().item()), 6),
                "std": round(float(images_f.std(unbiased=False).item()), 6),
                "min": round(float(images_f.min().item()), 6),
                "max": round(float(images_f.max().item()), 6),
                "mask_frac": round(float(masks.float().mean().item()), 6)
                if isinstance(masks, torch.Tensor)
                else None,
            }

        token_mask = observation.get("tokenized_prompt_mask")
        token_mask_frac = (
            round(float(token_mask.float().mean().item()), 6)
            if isinstance(token_mask, torch.Tensor)
            else None
        )
        logits = getattr(result, "logits", None)
        hidden_states = getattr(result, "hidden_states", None)
        self._logger.warning(
            "[SteamSFT][BATCH_DIAG] num_bins=%d "
            "label_min=%d label_max=%d label_hist=%s image_stats=%s "
            "token_mask_frac=%s logits_std=%s hidden_std=%s",
            num_bins,
            int(labels.min().item()),
            int(labels.max().item()),
            label_hist,
            image_stats,
            token_mask_frac,
            round(float(logits.detach().float().std(unbiased=False).item()), 6)
            if isinstance(logits, torch.Tensor)
            else None,
            round(float(hidden_states.detach().float().std(unbiased=False).item()), 6)
            if isinstance(hidden_states, torch.Tensor)
            else None,
        )

    def _fetch_next_batch(self) -> dict:
        """Return the next training micro-batch; rotate epoch on exhaustion."""
        try:
            return next(self.data_iter)
        except StopIteration:
            new_epoch = getattr(self, "_current_epoch", 0) + 1
            self._current_epoch = new_epoch
            self.data_loader.set_epoch(new_epoch)
            self.data_iter = iter(self.data_loader)
            return next(self.data_iter)

    def _backward_one_micro_batch(
        self,
        grad_accum: int,
        micro_idx: int,
        member_idx: int | None,
    ) -> dict[str, float]:
        """Forward + scaled backward on ONE micro batch. Returns metrics dict.

        Pure per-micro-batch work: no optimizer step, no grad clearing —
        those belong to the flow-level caller
        (:meth:`_run_training_members`).

        ``member_idx=None`` → single-model forward.
        ``member_idx=int`` → per-member forward; the ensemble wrapper
        routes the call to that specific member.
        """
        backward_ctx = self.before_micro_batch(
            self.model, is_last_micro_batch=(micro_idx + 1) == grad_accum
        )

        batch = self._fetch_next_batch()
        observation, labels = self._prepare_input(batch)

        forward_kwargs: dict[str, Any] = {}
        if member_idx is not None:
            forward_kwargs["member_idx"] = int(member_idx)

        with self.amp_context:
            result = self.model(
                observation=observation, labels=labels, **forward_kwargs
            )
            loss = result.loss
        if loss is None:
            raise RuntimeError("Binary value model returned no loss during training.")

        self._raise_if_non_finite_training_step(
            result=result,
            observation=observation,
            labels=labels,
            micro_batch_idx=micro_idx,
            grad_accum=grad_accum,
        )
        self._log_training_batch_diagnostics_once(observation, labels, result)

        metrics = self._critic_output_metrics(result, include_std=False)

        scaled_loss = loss / grad_accum
        with backward_ctx:
            self.grad_scaler.scale(scaled_loss).backward()

        return metrics

    @staticmethod
    def _mean_metrics(metric_dicts: list[dict[str, float]]) -> dict[str, float]:
        """Arithmetic-mean a list of per-batch metric dicts."""
        agg: dict[str, list[float]] = {}
        for m in metric_dicts:
            for k, v in m.items():
                agg.setdefault(k, []).append(v)
        return {k: sum(v) / len(v) for k, v in agg.items()}

    @staticmethod
    def _all_reduce_eval_metrics(metrics: dict[str, float]) -> dict[str, float]:
        """Cross-rank mean; only ranks that produced a key participate."""
        if not torch.distributed.is_initialized():
            return metrics

        world_size = torch.distributed.get_world_size()
        key_lists: list[list[str] | None] = [None] * world_size
        torch.distributed.all_gather_object(key_lists, sorted(metrics.keys()))
        union_keys = sorted({k for keys in key_lists for k in (keys or [])})
        if not union_keys:
            return {}

        device = Worker.torch_platform.current_device()
        sum_tensor = torch.zeros(len(union_keys), dtype=torch.float32, device=device)
        count_tensor = torch.zeros(len(union_keys), dtype=torch.float32, device=device)
        key_to_idx = {k: i for i, k in enumerate(union_keys)}
        for key, value in metrics.items():
            idx = key_to_idx[key]
            sum_tensor[idx] = value
            count_tensor[idx] = 1.0

        torch.distributed.all_reduce(sum_tensor, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(count_tensor, op=torch.distributed.ReduceOp.SUM)
        return {
            key: (sum_tensor[idx] / count_tensor[idx]).item()
            for idx, key in enumerate(union_keys)
            if count_tensor[idx] > 0
        }

    def _clear_non_current_member_grads(self, member_idx: int) -> None:
        """Null sibling members' FSDP-materialized phantom grads.

        Ensemble-only. Called after each member's backward in
        :meth:`_run_training_members` (when ``ensemble_size > 1``) and
        before the optimizer step so
        sibling-member parameters that share a FSDP flat-param bucket
        with the currently trained member don't carry residual zero
        grads into the step (with ``use_orig_params=True`` FSDP can
        materialize those automatically).

        Precondition: ``member_idx`` is a valid int (the caller is the
        ensemble training flow, so ``ensemble_size > 1`` by construction).
        No defensive early returns — if this gets called with an invalid
        index it should fail loudly, not silently no-op.
        """
        current_member_tag = f"members.{int(member_idx)}."
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if "members." not in name:
                continue
            if current_member_tag in name:
                continue
            param.grad = None

    def _run_training_members(
        self,
        grad_accum: int,
        ensemble_size: int,
    ) -> dict[str, float]:
        """Run one global training step over all members.

        Single model and ensemble share one flow: each member runs a
        fresh ``grad_accum``-long micro-batch loop, then ``optimizer_step``
        + ``zero_grad``. The single-model case (``ensemble_size == 1``) is
        the degenerate one-member list ``[None]`` — no per-member forward
        routing and no phantom-grad clearing. For ensembles each member
        trains on its own independent random data (bagging), which makes
        the ensemble's prediction variance a meaningful epistemic
        uncertainty signal; total loader batches consumed per global step
        is ``ensemble_size × grad_accum``, while peak activations +
        gradients stay 1× a single member thanks to the sequential
        execution. Per-member metric breakdowns and ``grad_norm_mean`` are
        only emitted when ``ensemble_size > 1``.
        """
        members: list[int | None] = (
            [None] if ensemble_size == 1 else list(range(ensemble_size))
        )
        per_member_metrics: list[list[dict[str, float]]] = []
        per_member_grad_norms: list[float] = []
        last_lr_list: list[float] = []

        for member_idx in members:
            micro_metrics = [
                self._backward_one_micro_batch(
                    grad_accum=grad_accum,
                    micro_idx=micro_idx,
                    member_idx=member_idx,
                )
                for micro_idx in range(grad_accum)
            ]
            if member_idx is not None:
                self._clear_non_current_member_grads(member_idx)
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            per_member_metrics.append(micro_metrics)
            per_member_grad_norms.append(float(grad_norm))
            last_lr_list = lr_list

        flat_metrics = [m for member_list in per_member_metrics for m in member_list]
        train_metrics = self._mean_metrics(flat_metrics)
        if ensemble_size > 1:
            for member_idx, micro_metrics in enumerate(per_member_metrics):
                member_metrics = self._mean_metrics(micro_metrics)
                for metric_name, metric_value in member_metrics.items():
                    train_metrics[f"member{member_idx}/{metric_name}"] = metric_value
                train_metrics[f"member{member_idx}/grad_norm"] = per_member_grad_norms[
                    member_idx
                ]
            train_metrics["grad_norm"] = max(per_member_grad_norms)
            train_metrics["grad_norm_mean"] = sum(per_member_grad_norms) / len(
                per_member_grad_norms
            )
        else:
            train_metrics["grad_norm"] = per_member_grad_norms[0]
        train_metrics["lr"] = float(last_lr_list[0]) if last_lr_list else 0.0
        return train_metrics

    def run_training(self) -> dict[str, float]:
        """Execute one global training step.

        Runs :meth:`_run_training_members` (one flow for both single model
        and ensemble), then all-reduces the metrics and steps the LR
        scheduler. Offload bookends wrap the whole call when enabled.
        """
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            use_grad_ckpt = bool(
                getattr(self.cfg.actor.model, "use_gradient_checkpointing", False)
            ) or bool(
                getattr(self.cfg.actor.fsdp_config, "gradient_checkpointing", False)
            )
            if self._gradient_checkpointing_state is None or (
                self._gradient_checkpointing_state != use_grad_ckpt
            ):
                if use_grad_ckpt and hasattr(
                    self.model, "gradient_checkpointing_enable"
                ):
                    self.model.gradient_checkpointing_enable()
                elif hasattr(self.model, "gradient_checkpointing_disable"):
                    self.model.gradient_checkpointing_disable()
                self._gradient_checkpointing_state = use_grad_ckpt

            micro_bs = self.cfg.actor.micro_batch_size
            global_bs = self.cfg.actor.global_batch_size
            assert global_bs % (micro_bs * self._world_size) == 0, (
                f"global_batch_size={global_bs} must be divisible by "
                f"micro_batch_size * world_size = {micro_bs * self._world_size}"
            )
            grad_accum = global_bs // micro_bs // self._world_size
            ensemble_size = int(getattr(self.cfg.actor.model, "ensemble_size", 1))

            train_metrics = self._run_training_members(grad_accum, ensemble_size)

            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )
            self.lr_scheduler.step()

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
                    self.offload_optimizer()

            return train_metrics

    def _eval_batch(
        self,
        observation: dict[str, Any],
        labels: torch.Tensor,
        ensemble_size: int,
    ) -> dict[str, float]:
        """Eval one batch over all members on the SAME inputs.

        Single model (``ensemble_size == 1``) is the degenerate one-member
        list ``[None]``. Ensemble members share the full eval batch (no
        chunk-slicing), so the training-time "batch divisible by
        ensemble_size" constraint does not apply. Member-wise metrics are
        averaged into one batch metric dict — the same shape regardless of
        ``ensemble_size``, so the outer ``run_eval`` aggregation is
        uniform; per-member breakdowns are only added when
        ``ensemble_size > 1``.
        """
        members: list[int | None] = (
            [None] if ensemble_size == 1 else list(range(ensemble_size))
        )
        member_metrics: list[dict[str, float]] = []
        for member_idx in members:
            forward_kwargs: dict[str, Any] = {}
            if member_idx is not None:
                forward_kwargs["member_idx"] = member_idx
            with self.amp_context:
                result = self.model(
                    observation=observation, labels=labels, **forward_kwargs
                )
            member_metrics.append(self._critic_output_metrics(result, include_std=True))
        eval_metrics = self._mean_metrics(member_metrics)
        if ensemble_size > 1:
            for member_idx, metrics in enumerate(member_metrics):
                for metric_name, metric_value in metrics.items():
                    eval_metrics[f"member{member_idx}/{metric_name}"] = metric_value
        return eval_metrics

    def run_eval(self) -> dict[str, float]:
        """Run eval over every registered eval dataset.

        Each batch goes through :meth:`_eval_batch` (one flow for single
        model and ensemble); per-dataset aggregation is then flattened
        with cross-dataset means via ``<metric>`` keys (no prefix)
        alongside the ``<dataset>/<metric>`` breakdown.
        """
        with self.worker_timer():
            if not self.eval_data_loaders:
                return {}

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)

            self.model.eval()
            ensemble_size = int(getattr(self.cfg.actor.model, "ensemble_size", 1))
            per_dataset: dict[str, dict[str, float]] = {}
            with torch.no_grad():
                for ds_name, loader in self.eval_data_loaders:
                    batch_metrics: list[dict[str, float]] = []
                    for batch in loader:
                        observation, labels = self._prepare_input(batch)
                        metrics = self._eval_batch(observation, labels, ensemble_size)
                        batch_metrics.append(metrics)
                    if not batch_metrics:
                        continue
                    per_dataset[ds_name] = self._mean_metrics(batch_metrics)

            final: dict[str, float] = {}
            if per_dataset:
                for ds_name, metrics in per_dataset.items():
                    for k, v in metrics.items():
                        final[f"{ds_name}/{k}"] = v

                all_keys: set[str] = set()
                for metrics in per_dataset.values():
                    all_keys.update(metrics.keys())
                for k in sorted(all_keys):
                    vals = [m[k] for m in per_dataset.values() if k in m]
                    if vals:
                        final[k] = sum(vals) / len(vals)

            final = self._all_reduce_eval_metrics(final)

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
            return final

    def get_max_steps_per_epoch(self) -> int:
        """Global training steps per epoch (SFTRunner contract).

        A global training step consumes ``grad_accum × ensemble_size``
        loader batches (see :meth:`build_dataloader`'s per-member note):
        the single-model path consumes ``grad_accum`` batches while the
        ensemble path runs the same grad-accum loop once per member.
        Single model (``ensemble_size=1``) reduces to the usual
        ``loader_len // grad_accum`` calculation.
        """
        loader_len = len(self.data_loader)
        if loader_len == 0:
            return 0
        grad_accum = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        ensemble_size = max(1, int(getattr(self.cfg.actor.model, "ensemble_size", 1)))
        batches_per_step = grad_accum * ensemble_size
        return max(1, loader_len // batches_per_step)

    def set_global_step(self, step: int):
        """Update the current epoch so the sampler reshuffles on rollover.

        Epoch rollover — and the resulting ``DistributedSampler.set_epoch``
        reshuffle — fires when ``step`` crosses a multiple of
        :meth:`get_max_steps_per_epoch`.
        """
        self.global_step = step
        steps_per_epoch = self.get_max_steps_per_epoch()
        if steps_per_epoch == 0:
            return
        new_epoch = step // steps_per_epoch
        current = getattr(self, "_current_epoch", -1)
        if current != new_epoch:
            self._current_epoch = new_epoch
            self.data_loader.set_epoch(new_epoch)
            self.data_iter = iter(self.data_loader)


__all__ = ["FSDPSteamSftWorker"]
