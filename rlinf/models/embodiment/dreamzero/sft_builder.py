import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader, IterableDataset

logger = logging.getLogger(__name__)


def _ensure_groot_importable():
    if "groot" in sys.modules:
        return
    dreamzero_root = Path(__file__).resolve().parents[5]
    if str(dreamzero_root) not in sys.path:
        sys.path.insert(0, str(dreamzero_root))


def _normalize_single_data_path(data_paths):
    if data_paths is None:
        return None
    if isinstance(data_paths, str):
        return data_paths
    if isinstance(data_paths, (list, tuple)):
        if len(data_paths) != 1:
            raise ValueError(
                "DreamZero SFT currently expects a single train_data_paths entry "
                f"for data-root override, got {len(data_paths)} entries."
            )
        return data_paths[0]
    raise TypeError(f"Unsupported data_paths type: {type(data_paths)}")


def load_dreamzero_train_cfg(model_cfg: DictConfig, data_paths=None):
    _ensure_groot_importable()

    model_path = model_cfg.get("model_path")
    train_cfg_path = model_cfg.get(
        "train_cfg_path",
        os.path.join(model_path, "experiment_cfg", "conf.yaml"),
    )
    if not os.path.exists(train_cfg_path):
        raise FileNotFoundError(f"DreamZero train_cfg_path does not exist: {train_cfg_path}")

    train_cfg = OmegaConf.load(train_cfg_path)

    data_root_override = _normalize_single_data_path(data_paths)
    if data_root_override is not None:
        with open_dict(train_cfg):
            data_root_key = model_cfg.get("data_root_key")
            if data_root_key is not None:
                if data_root_key not in train_cfg:
                    raise KeyError(
                        f"DreamZero data_root_key `{data_root_key}` not found in {train_cfg_path}"
                    )
                train_cfg[data_root_key] = data_root_override
            else:
                data_root_keys = [key for key in train_cfg.keys() if key.endswith("_data_root")]
                if len(data_root_keys) == 1:
                    train_cfg[data_root_keys[0]] = data_root_override
                elif len(data_root_keys) > 1:
                    raise ValueError(
                        "DreamZero train config contains multiple `*_data_root` keys. "
                        "Please set `actor.model.data_root_key` in RLinf config."
                    )

    if model_cfg.get("tokenizer_path", None) is not None and "tokenizer_path" in train_cfg:
        with open_dict(train_cfg):
            train_cfg.tokenizer_path = model_cfg.tokenizer_path

    return train_cfg


def _reshape_singleton_state_dict_tensors(
    model: torch.nn.Module, state_dict: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """
    Keep old DreamZero checkpoints with scalar singleton params loadable after
    reshaping those params to 1D tensors for FSDP2 compatibility.
    """
    target_state = model.state_dict()
    patched_state_dict = dict(state_dict)

    for key, value in state_dict.items():
        target_value = target_state.get(key)
        if (
            isinstance(value, torch.Tensor)
            and isinstance(target_value, torch.Tensor)
            and value.numel() == 1
            and target_value.numel() == 1
            and value.shape != target_value.shape
        ):
            patched_state_dict[key] = value.reshape(target_value.shape)

    return patched_state_dict


def _log_cuda_memory(stage: str) -> None:
    if not torch.cuda.is_available():
        logger.info("[DreamZero SFT][CUDA MEM] stage=%s cuda_unavailable", stage)
        return

    logger.info(
        "[DreamZero SFT][CUDA MEM] stage=%s rank=%s local_rank=%s visible=%s "
        "current_device=%s allocated=%.2fGiB reserved=%.2fGiB max_allocated=%.2fGiB",
        stage,
        os.environ.get("RANK"),
        os.environ.get("LOCAL_RANK"),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch.cuda.current_device(),
        torch.cuda.memory_allocated() / 1024**3,
        torch.cuda.memory_reserved() / 1024**3,
        torch.cuda.max_memory_allocated() / 1024**3,
    )


def _load_weights(model: torch.nn.Module, checkpoint_dir: str) -> None:
    from safetensors.torch import load_file

    if not checkpoint_dir or not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(
            f"DreamZero pretrained checkpoint does not exist: {checkpoint_dir}"
        )

    safetensors_index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    safetensors_path = os.path.join(checkpoint_dir, "model.safetensors")
    full_weights_path = os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt")
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )

    if os.path.exists(full_weights_path):
        state_dict = torch.load(full_weights_path, map_location="cpu")
        state_dict = _reshape_singleton_state_dict_tensors(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        return

    if os.path.exists(actor_full_weights_path):
        state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        state_dict = _reshape_singleton_state_dict_tensors(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        return

    if os.path.exists(safetensors_index_path):
        with open(safetensors_index_path, "r") as f:
            index = json.load(f)
        for shard_file in sorted(set(index["weight_map"].values())):
            shard_path = os.path.join(checkpoint_dir, shard_file)
            shard_state_dict = load_file(shard_path)
            shard_state_dict = _reshape_singleton_state_dict_tensors(
                model, shard_state_dict
            )
            model.load_state_dict(shard_state_dict, strict=False)
            del shard_state_dict
            gc.collect()
        return

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        state_dict = _reshape_singleton_state_dict_tensors(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
        return

    raise FileNotFoundError(
        f"No DreamZero weights found under {checkpoint_dir}. "
        "Expected one of model.safetensors(.index.json) or model_state_dict/full_weights.pt"
    )


def build_dreamzero_sft_model(model_cfg: DictConfig) -> torch.nn.Module:
    _log_cuda_memory("build_dreamzero_sft_model:start")
    train_cfg = load_dreamzero_train_cfg(model_cfg)
    model = instantiate(train_cfg.model)
    _log_cuda_memory("build_dreamzero_sft_model:after_instantiate")

    pretrained_model_path = model_cfg.get("pretrained_model_path", model_cfg.get("model_path"))
    _load_weights(model, pretrained_model_path)
    _log_cuda_memory("build_dreamzero_sft_model:after_load_weights")

    if (
        hasattr(model, "action_head")
        and hasattr(model.action_head, "inject_lora_after_loading")
        and getattr(getattr(model.action_head, "config", None), "defer_lora_injection", False)
    ):
        model.action_head.inject_lora_after_loading()
        _log_cuda_memory("build_dreamzero_sft_model:after_inject_lora")

    _log_cuda_memory("build_dreamzero_sft_model:return")
    return model


def build_dreamzero_sft_dataloader(cfg, world_size, global_rank, data_paths, eval_dataset=False):
    train_cfg = load_dreamzero_train_cfg(cfg.actor.model, data_paths=data_paths)
    dataset = instantiate(train_cfg.train_dataset)
    data_collator = instantiate(train_cfg.data_collator)

    batch_size = (
        cfg.actor.get("eval_batch_size", 1) if eval_dataset else cfg.actor.micro_batch_size
    )
    num_workers = cfg.data.get(
        "num_workers",
        train_cfg.get(
            "dataloader_num_workers",
            train_cfg.get("training_args", {}).get("dataloader_num_workers", 0),
        ),
    )
    pin_memory = cfg.data.get(
        "pin_memory",
        train_cfg.get(
            "dataloader_pin_memory",
            train_cfg.get("training_args", {}).get("dataloader_pin_memory", False),
        ),
    )
    persistent_workers = cfg.data.get(
        "persistent_workers",
        train_cfg.get(
            "dataloader_persistent_workers",
            train_cfg.get("training_args", {}).get("dataloader_persistent_workers", False),
        ),
        )
    persistent_workers = bool(persistent_workers and num_workers > 0)

    dataloader_kwargs = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers,
        "drop_last": True,
    }

    is_iterable_dataset = isinstance(dataset, IterableDataset)
    if not is_iterable_dataset:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=global_rank,
            shuffle=not eval_dataset,
        )
        dataloader_kwargs["sampler"] = sampler

    data_loader = DataLoader(dataset, **dataloader_kwargs)

    merged_metadata = getattr(dataset, "merged_metadata", None)
    metadata = None
    if merged_metadata is not None:
        metadata = {
            key: value.model_dump(mode="json") for key, value in merged_metadata.items()
        }

    model_config = None
    model_cfg_node = train_cfg.get("model", {}).get("config", None)
    if model_cfg_node is not None:
        model_config = OmegaConf.to_container(model_cfg_node, resolve=True)

    data_config = {
        "dreamzero_train_cfg": train_cfg,
        "dreamzero_metadata": metadata,
        "dreamzero_model_config": model_config,
    }
    return data_loader, data_config
