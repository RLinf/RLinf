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

"""
FSDP Value Model SFT Worker.

Standalone worker for training vla_lib ValueCriticModel / QCriticModel
via supervised fine-tuning with FSDP.

This module is self-contained and does NOT share code with fsdp_sft_worker.py.
"""

# Suppress libdav1d/ffmpeg verbose logging (must be before any av/video imports)
import json
import logging
import os
from pathlib import Path

os.environ["LIBAV_LOG_LEVEL"] = "quiet"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
logging.getLogger("libav").setLevel(logging.ERROR)
logging.getLogger("av").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

import torch  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

# Import dataset-related classes from rlinf.datasets
from rlinf.datasets import (  # noqa: E402
    ValueDataLoaderImpl,
    ValueMixtureDataset,
)
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager  # noqa: E402
from rlinf.models import get_model  # noqa: E402
from rlinf.scheduler import Worker  # noqa: E402
from rlinf.utils.distributed import all_reduce_dict  # noqa: E402


def _load_return_stats_from_dataset(
    dataset_path: str,
) -> tuple[float | None, float | None]:
    """Load return min/max from dataset's stats.json.

    Args:
        dataset_path: Path to LeRobot dataset

    Returns:
        Tuple of (return_min, return_max), or (None, None) if not found
    """
    stats_path = Path(dataset_path) / "meta" / "stats.json"
    if not stats_path.exists():
        return None, None

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
        return_stats = stats.get("return", {})
        return return_stats.get("min"), return_stats.get("max")
    except (json.JSONDecodeError, KeyError):
        return None, None


class FSDPValueSftWorker(FSDPModelManager, Worker):
    """FSDP worker for value model SFT training.

    Reads ``data.datasets`` list from config (single dataset = list of one).
    Uses global return_min/return_max for normalization across all datasets.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)
        super().__init__(cfg.actor, self._world_size, self._rank)

        self.cfg = cfg
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
        self.device = torch.cuda.current_device()

        # Build data loader
        self.data_loader, self.eval_data_loader, self.data_config = (
            self.build_dataloader()
        )

    # -----------------------------------------------------------------------
    # Worker interface
    # -----------------------------------------------------------------------

    def init_worker(self):
        """Called by runner to initialize model & optimizer."""
        self.setup_model_and_optimizer()

        if self.cfg.actor.get("enable_offload", False):
            self.offload_param_and_grad()
            self.offload_optimizer()

    def model_provider_func(self) -> torch.nn.Module:
        """Load the vla_lib value model."""
        return get_model(self.cfg.actor.model)

    # -----------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------

    def build_dataloader(self):
        """Build dataloader from ``data.datasets`` list.

        Uses ValueDataset which includes:
        - DataConfig transforms (repack, normalize, model-specific like LiberoInputs)
        - Return discretization for value prediction
        - VLM-mode sample formatting
        """
        # Suppress PyAV/libdav1d verbose logging
        try:
            import av

            av.logging.set_level(av.logging.ERROR)
        except (ImportError, AttributeError):
            pass

        from vla_lib.models.vlas.openpi05.data_collator_pi05 import PI05DataCollator
        from vla_lib.models.vlas.openpi05.processing_pi05 import PI05Processor

        from rlinf.datasets.vla_lib import ValueDataset

        data_cfg = self.cfg.get("data", {})
        model_cfg = self.cfg.actor.model
        pin_memory = data_cfg.get("pin_memory", True)
        train_num_workers = int(data_cfg.get("train_num_workers", 0))
        eval_num_workers = int(data_cfg.get("eval_num_workers", train_num_workers))
        prefetch_factor = data_cfg.get("prefetch_factor", 2)
        persistent_workers = bool(data_cfg.get("persistent_workers", True))

        def _loader_worker_kwargs(num_workers: int) -> dict:
            kwargs = {
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                kwargs["persistent_workers"] = persistent_workers
                if prefetch_factor is not None:
                    kwargs["prefetch_factor"] = int(prefetch_factor)
            return kwargs

        # ---- processor & collator ----
        processor = PI05Processor(
            max_token_len=getattr(model_cfg, "max_token_len", 200),
            discrete_state_input=getattr(model_cfg, "discrete_state_input", False),
            exclude_cot_from_kv_cache=getattr(
                model_cfg, "exclude_cot_from_kv_cache", False
            ),
        )
        train_collator = PI05DataCollator(
            processor=processor,
            max_length=getattr(model_cfg, "max_token_len", 200),
            train=True,
        )
        # Use deterministic preprocessing for eval (no image augmentation).
        eval_collator = PI05DataCollator(
            processor=processor,
            max_length=getattr(model_cfg, "max_token_len", 200),
            train=False,
        )
        # ---- shared defaults ----
        data_root = data_cfg.get("data_root", None)
        critic_fwd_mode = getattr(model_cfg, "critic_forward_mode", "expert")
        auto_skip_vlm = critic_fwd_mode == "expert"

        # Transform-related shared config (required parameters)
        robot_type = data_cfg.get("robot_type")
        model_type = data_cfg.get("model_type")
        if robot_type is None:
            raise ValueError("data_cfg.robot_type is required but not provided")
        if model_type is None:
            raise ValueError("data_cfg.model_type is required but not provided")
        norm_stats_dir = data_cfg.get("norm_stats_dir", None)

        shared = {
            "value_prefix": data_cfg.get("value_prefix", "Value: "),
            "include_state": data_cfg.get("include_state", True),
            "skip_vlm_response": data_cfg.get("skip_vlm_response", auto_skip_vlm),
            "action_horizon": data_cfg.get(
                "action_horizon", getattr(model_cfg, "action_horizon", 10)
            ),
            "gamma": data_cfg.get("gamma", 0.99),
            "normalize_to_minus_one_zero": data_cfg.get(
                "normalize_to_minus_one_zero", True
            ),
            "action_dim": data_cfg.get(
                "action_dim", getattr(model_cfg, "action_dim", None)
            ),
            # Transform-related
            "robot_type": robot_type,
            "model_type": model_type,
            "norm_stats_dir": norm_stats_dir,
        }

        # ---- build datasets ----
        datasets_list = data_cfg.get("datasets", [])
        if not datasets_list:
            raise ValueError(
                "data.datasets must be a non-empty list. "
                "Each entry needs: dataset_path."
            )
        train_entries = [
            dict(entry) for entry in datasets_list if entry.get("dataset_path", None)
        ]
        eval_entries = [
            dict(entry)
            for entry in datasets_list
            if entry.get("eval_dataset_path", None)
        ]
        if not train_entries:
            raise ValueError(
                "data.datasets must contain at least one training entry with 'dataset_path'."
            )

        # ---- Compute global return_min/return_max ----
        # Priority: 1) global config  2) compute from all datasets' stats.json
        global_return_min = data_cfg.get("return_min", None)
        global_return_max = data_cfg.get("return_max", None)

        if global_return_min is None or global_return_max is None:
            # Compute from all datasets' stats.json
            all_mins = []
            all_maxs = []
            for entry in train_entries:
                ds_path = entry.get("dataset_path")
                if ds_path is None:
                    continue
                if data_root and not os.path.isabs(ds_path):
                    ds_path = os.path.join(data_root, ds_path)
                ds_min, ds_max = _load_return_stats_from_dataset(ds_path)
                if ds_min is not None:
                    all_mins.append(ds_min)
                if ds_max is not None:
                    all_maxs.append(ds_max)

            if all_mins and all_maxs:
                global_return_min = (
                    min(all_mins) if global_return_min is None else global_return_min
                )
                global_return_max = (
                    max(all_maxs) if global_return_max is None else global_return_max
                )
                logger.info(
                    "[ValueSFT] Computed global return range from stats.json: "
                    f"[{global_return_min}, {global_return_max}]"
                )
            else:
                # Fallback to default values if no stats found
                global_return_min = (
                    global_return_min if global_return_min is not None else -700.0
                )
                global_return_max = (
                    global_return_max if global_return_max is not None else 0.0
                )
                logger.warning(
                    "[ValueSFT] No stats.json found, using default return range: "
                    f"[{global_return_min}, {global_return_max}]"
                )
        else:
            logger.info(
                "[ValueSFT] Using global return range from config: "
                f"[{global_return_min}, {global_return_max}]"
            )

        # ---- common ValueDataset kwargs (shared by train & eval) ----
        common_ds_kwargs = {
            "robot_type": shared["robot_type"],
            "model_type": shared["model_type"],
            "norm_stats_dir": shared["norm_stats_dir"],
            "action_horizon": shared["action_horizon"],
            "gamma": shared["gamma"],
            "return_min": global_return_min,
            "return_max": global_return_max,
            "normalize_to_minus_one_zero": shared["normalize_to_minus_one_zero"],
            "skip_vlm_response": shared["skip_vlm_response"],
            "value_prefix": shared["value_prefix"],
            "include_state": shared["include_state"],
            "action_dim": shared["action_dim"],
        }

        datasets_with_weights = []
        for entry in train_entries:
            ds_path = entry.get("dataset_path")
            if ds_path is None:
                raise ValueError("Each dataset entry must have 'dataset_path'")
            if data_root and not os.path.isabs(ds_path):
                ds_path = os.path.join(data_root, ds_path)

            # Dataset type: "sft" or "rollout"
            ds_type = entry.get("type", "sft")
            if ds_type not in ("sft", "rollout"):
                raise ValueError(
                    f"Dataset type must be 'sft' or 'rollout', got '{ds_type}'"
                )

            weight = entry.get("weight", 1.0)

            # Per-entry overrides on top of common kwargs
            entry_kwargs = {
                **common_ds_kwargs,
                "dataset_path": ds_path,
                "robot_type": entry.get("robot_type", shared["robot_type"]),
                "model_type": entry.get("model_type", shared["model_type"]),
                "norm_stats_dir": entry.get("norm_stats_dir", shared["norm_stats_dir"]),
                "asset_id": entry.get("asset_id", None),
                "action_horizon": entry.get("action_horizon", shared["action_horizon"]),
                "normalize_to_minus_one_zero": entry.get(
                    "normalize_to_minus_one_zero", shared["normalize_to_minus_one_zero"]
                ),
                "skip_vlm_response": entry.get(
                    "skip_vlm_response", shared["skip_vlm_response"]
                ),
                "value_prefix": entry.get("value_prefix", shared["value_prefix"]),
                "include_state": entry.get("include_state", shared["include_state"]),
                "action_dim": entry.get("action_dim", shared["action_dim"]),
                "split": "train",
                "default_prompt": entry.get("default_prompt", None),
                "max_samples": entry.get("max_samples", None),
                "episode_percentage": entry.get("episode_percentage", None),
                "shuffle_episodes": entry.get("shuffle_episodes", False),
                "episode_seed": entry.get("episode_seed", 42),
            }

            ds = ValueDataset(**entry_kwargs)
            datasets_with_weights.append((ds, weight))
            logger.info(
                f"[ValueSFT] Loaded: {ds_path}  (type={ds_type}, {len(ds)} samples, weight={weight})"
            )

        # ---- single vs mixture ----
        if len(datasets_with_weights) == 1:
            dataset = datasets_with_weights[0][0]
        else:
            dataset = ValueMixtureDataset(
                datasets=datasets_with_weights,
                mode="train",
                balance_dataset_weights=data_cfg.get("balance_weights", True),
                seed=data_cfg.get("seed", 42),
            )
        logger.info(f"[ValueSFT] Total samples: {len(dataset)}")

        # ---- DataLoader ----
        sampler = None
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=self._world_size,
                rank=self._rank,
                shuffle=True,
                drop_last=True,
            )

        train_loader_worker_kwargs = _loader_worker_kwargs(train_num_workers)
        torch_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.cfg.actor.micro_batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            drop_last=True,
            collate_fn=train_collator,
            **train_loader_worker_kwargs,
        )
        logger.info(
            "[ValueSFT] Train DataLoader workers: "
            f"num_workers={train_num_workers}, "
            f"prefetch_factor={prefetch_factor if train_num_workers > 0 else 'N/A'}, "
            f"persistent_workers={persistent_workers if train_num_workers > 0 else 'N/A'}, "
            f"pin_memory={pin_memory}"
        )

        data_config = {"model_type": "vla_lib_value_model"}
        train_data_loader = ValueDataLoaderImpl(data_config, torch_loader)

        # ---- optional eval DataLoader ----
        eval_data_loader = None
        # New preferred config style:
        # data.datasets[*] can contain a standalone eval entry with
        # eval_dataset_path / eval_max_samples
        # Backward compatible with legacy top-level:
        # data.eval_dataset_path / data.eval_max_samples
        if len(eval_entries) > 1:
            logger.warning(
                "[ValueSFT] Multiple datasets define eval_dataset_path; "
                "using the first one."
            )
        eval_entry = eval_entries[0] if eval_entries else {}

        eval_dataset_path = eval_entry.get(
            "eval_dataset_path", data_cfg.get("eval_dataset_path", None)
        )
        eval_max_samples = eval_entry.get(
            "eval_max_samples", data_cfg.get("eval_max_samples", None)
        )
        if eval_dataset_path:
            if "return_min" in eval_entry or "return_max" in eval_entry:
                logger.warning(
                    "[ValueSFT] eval entry return_min/return_max are ignored. "
                    "Eval reward/return normalization always uses train global "
                    f"range [{global_return_min}, {global_return_max}]."
                )
            eval_ds_path = eval_dataset_path
            if data_root and not os.path.isabs(eval_ds_path):
                eval_ds_path = os.path.join(data_root, eval_ds_path)

            eval_dataset = ValueDataset(
                dataset_path=eval_ds_path,
                robot_type=eval_entry.get("robot_type", shared["robot_type"]),
                model_type=eval_entry.get("model_type", shared["model_type"]),
                norm_stats_dir=eval_entry.get(
                    "norm_stats_dir", shared["norm_stats_dir"]
                ),
                asset_id=eval_entry.get("asset_id", None),
                action_horizon=eval_entry.get(
                    "action_horizon", shared["action_horizon"]
                ),
                gamma=eval_entry.get("gamma", shared["gamma"]),
                return_min=global_return_min,
                return_max=global_return_max,
                normalize_to_minus_one_zero=eval_entry.get(
                    "normalize_to_minus_one_zero", shared["normalize_to_minus_one_zero"]
                ),
                skip_vlm_response=eval_entry.get(
                    "skip_vlm_response", shared["skip_vlm_response"]
                ),
                value_prefix=eval_entry.get("value_prefix", shared["value_prefix"]),
                include_state=eval_entry.get("include_state", shared["include_state"]),
                split="val",
                action_dim=eval_entry.get("action_dim", shared["action_dim"]),
                default_prompt=eval_entry.get("default_prompt", None),
                max_samples=eval_max_samples,
            )
            eval_sampler = None
            if torch.distributed.is_initialized():
                eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    eval_dataset,
                    num_replicas=self._world_size,
                    rank=self._rank,
                    shuffle=False,
                    drop_last=False,
                )
            eval_loader_worker_kwargs = _loader_worker_kwargs(eval_num_workers)
            eval_torch_loader = torch.utils.data.DataLoader(
                dataset=eval_dataset,
                batch_size=self.cfg.actor.micro_batch_size,
                shuffle=False,
                sampler=eval_sampler,
                drop_last=False,
                collate_fn=eval_collator,
                **eval_loader_worker_kwargs,
            )
            eval_data_loader = ValueDataLoaderImpl(data_config, eval_torch_loader)
            logger.info(
                f"[ValueSFT] Eval dataset loaded: {eval_dataset_path} "
                f"({len(eval_dataset)} samples, eval_max_samples={eval_max_samples}, "
                "norm_range=train_global:"
                f"[{global_return_min}, {global_return_max}])"
            )
            logger.info(
                "[ValueSFT] Eval DataLoader workers: "
                f"num_workers={eval_num_workers}, "
                f"prefetch_factor={prefetch_factor if eval_num_workers > 0 else 'N/A'}, "
                f"persistent_workers={persistent_workers if eval_num_workers > 0 else 'N/A'}, "
                f"pin_memory={pin_memory}"
            )

        return train_data_loader, eval_data_loader, data_config

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def run_training(self) -> dict[str, float]:
        """Execute one training step (may involve gradient accumulation)."""
        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)
                    self.load_optimizer(self.device)

            self.model.train()
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()

            micro_bs = self.cfg.actor.micro_batch_size
            global_bs = self.cfg.actor.global_batch_size
            assert global_bs % (micro_bs * self._world_size) == 0
            grad_accum = global_bs // micro_bs // self._world_size

            all_metrics = []

            for idx in range(grad_accum):
                backward_ctx = self.before_micro_batch(
                    self.model, is_last_micro_batch=(idx + 1) == grad_accum
                )

                batch = next(self.data_iter)
                obs, target_values, actions, extra = self._prepare_input(batch)

                with self.amp_context:
                    fwd_kwargs = {"observation": obs, "target_values": target_values}
                    if actions is not None:
                        fwd_kwargs["actions"] = actions
                    result = self.model(**fwd_kwargs)
                    loss = result.loss

                metrics = {}
                if result.expert_loss is not None:
                    metrics["expert_loss"] = (
                        result.expert_loss.detach().item()
                        if isinstance(result.expert_loss, torch.Tensor)
                        else result.expert_loss
                    )
                if result.language_loss is not None:
                    metrics["language_loss"] = (
                        result.language_loss.detach().item()
                        if isinstance(result.language_loss, torch.Tensor)
                        else result.language_loss
                    )
                if result.predicted_values is not None:
                    metrics["predicted_value_mean"] = (
                        result.predicted_values.detach().mean().item()
                    )
                    metrics["predicted_value_std"] = (
                        result.predicted_values.detach().std().item()
                    )
                if result.language_token_acc is not None:
                    metrics["language_token_acc"] = (
                        result.language_token_acc.detach().item()
                        if isinstance(result.language_token_acc, torch.Tensor)
                        else result.language_token_acc
                    )
                # Categorical loss metrics
                if result.cat_acc_best is not None:
                    metrics["cat_acc_best"] = (
                        result.cat_acc_best.detach().item()
                        if isinstance(result.cat_acc_best, torch.Tensor)
                        else result.cat_acc_best
                    )
                if result.cat_acc_neighbor is not None:
                    metrics["cat_acc_neighbor"] = (
                        result.cat_acc_neighbor.detach().item()
                        if isinstance(result.cat_acc_neighbor, torch.Tensor)
                        else result.cat_acc_neighbor
                    )
                if result.cat_mae is not None:
                    metrics["cat_mae"] = (
                        result.cat_mae.detach().item()
                        if isinstance(result.cat_mae, torch.Tensor)
                        else result.cat_mae
                    )

                scaled_loss = loss / grad_accum
                with backward_ctx:
                    self.grad_scaler.scale(scaled_loss).backward()

                metrics["loss"] = loss.detach().item()
                # Add target value statistics for comparison
                if target_values is not None:
                    metrics["target_value_mean"] = target_values.detach().mean().item()
                    metrics["target_value_std"] = target_values.detach().std().item()
                all_metrics.append(metrics)

            # optimizer step
            grad_norm, lr_list = self.optimizer_step()
            self.optimizer.zero_grad(set_to_none=True)

            # aggregate metrics
            agg = {}
            for m in all_metrics:
                for k, v in m.items():
                    agg.setdefault(k, []).append(v)
            train_metrics = {k: sum(v) / len(v) for k, v in agg.items()}
            train_metrics["grad_norm"] = grad_norm
            train_metrics["lr"] = lr_list[0] if lr_list else 0.0

            train_metrics = all_reduce_dict(
                train_metrics, op=torch.distributed.ReduceOp.AVG
            )

            self.lr_scheduler.step()

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()
                    self.offload_optimizer()

            return train_metrics

    def run_evaluation(self) -> dict[str, float]:
        """Run periodic evaluation on eval_data_loader if provided."""
        if self.eval_data_loader is None:
            return {}

        with self.worker_timer():
            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.load_param_and_grad(self.device)

            self.model.eval()
            all_metrics = []

            with torch.no_grad():
                for batch in self.eval_data_loader:
                    obs, target_values, actions, _ = self._prepare_input(batch)

                    with self.amp_context:
                        fwd_kwargs = {
                            "observation": obs,
                            "target_values": target_values,
                        }
                        if actions is not None:
                            fwd_kwargs["actions"] = actions
                        result = self.model(**fwd_kwargs)
                        loss = result.loss

                    metrics = {}
                    if result.expert_loss is not None:
                        metrics["expert_loss"] = (
                            result.expert_loss.detach().item()
                            if isinstance(result.expert_loss, torch.Tensor)
                            else result.expert_loss
                        )
                    if result.language_loss is not None:
                        metrics["language_loss"] = (
                            result.language_loss.detach().item()
                            if isinstance(result.language_loss, torch.Tensor)
                            else result.language_loss
                        )
                    if result.predicted_values is not None:
                        metrics["predicted_value_mean"] = (
                            result.predicted_values.detach().mean().item()
                        )
                        metrics["predicted_value_std"] = (
                            result.predicted_values.detach().std().item()
                        )
                    if result.language_token_acc is not None:
                        metrics["language_token_acc"] = (
                            result.language_token_acc.detach().item()
                            if isinstance(result.language_token_acc, torch.Tensor)
                            else result.language_token_acc
                        )
                    if result.cat_acc_best is not None:
                        metrics["cat_acc_best"] = (
                            result.cat_acc_best.detach().item()
                            if isinstance(result.cat_acc_best, torch.Tensor)
                            else result.cat_acc_best
                        )
                    if result.cat_acc_neighbor is not None:
                        metrics["cat_acc_neighbor"] = (
                            result.cat_acc_neighbor.detach().item()
                            if isinstance(result.cat_acc_neighbor, torch.Tensor)
                            else result.cat_acc_neighbor
                        )
                    if result.cat_mae is not None:
                        metrics["cat_mae"] = (
                            result.cat_mae.detach().item()
                            if isinstance(result.cat_mae, torch.Tensor)
                            else result.cat_mae
                        )
                    metrics["loss"] = loss.detach().item()
                    if target_values is not None:
                        metrics["target_value_mean"] = (
                            target_values.detach().mean().item()
                        )
                        metrics["target_value_std"] = (
                            target_values.detach().std().item()
                        )
                    all_metrics.append(metrics)

            if not all_metrics:
                return {}

            agg = {}
            for m in all_metrics:
                for k, v in m.items():
                    agg.setdefault(k, []).append(v)
            eval_metrics = {k: sum(v) / len(v) for k, v in agg.items()}
            eval_metrics = all_reduce_dict(
                eval_metrics, op=torch.distributed.ReduceOp.AVG
            )

            if self.cfg.actor.get("enable_offload", False):
                with self.device_lock:
                    self.offload_param_and_grad()

            return eval_metrics

    def _prepare_input(self, batch: dict):
        """Move batch to device and return (observation, target_values, actions, extra)."""

        def _to_device(x):
            if isinstance(x, torch.Tensor):
                return x.to(self.device)
            elif isinstance(x, dict):
                return {k: _to_device(v) for k, v in x.items()}
            return x

        observation = _to_device(batch["observation"])
        target_values = _to_device(batch.get("target_values"))
        actions = _to_device(batch.get("actions"))

        extra = {}
        for key in (
            "next_images",
            "next_states",
            "reward_sum",
            "num_valid_rewards",
            "dones",
        ):
            val = batch.get(key)
            if val is not None:
                extra[key] = _to_device(val)

        return observation, target_values, actions, extra

    # -----------------------------------------------------------------------
    # Utility hooks called by FSDPModelManager / runner
    # -----------------------------------------------------------------------

    def set_global_step(self, step: int):
        self.global_step = step

        loader_len = len(self.data_loader)
        if loader_len == 0:
            return

        # Calculate current epoch
        grad_accum = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )
        steps_per_epoch = max(1, loader_len // grad_accum)
        new_epoch = step // steps_per_epoch

        # Initialize or reset iterator only when epoch changes
        current_epoch = getattr(self, "_current_epoch", -1)
        if current_epoch != new_epoch:
            self._current_epoch = new_epoch
            self.data_loader.set_epoch(new_epoch)
            self.data_iter = iter(self.data_loader)


__all__ = ["FSDPValueSftWorker"]
