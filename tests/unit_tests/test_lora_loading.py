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

"""Unit tests for the shared LoRA adapter export / load API."""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch import nn

from rlinf.utils.lora_adapter import (
    _broadcast_rank0_outcome,
    export_lora_adapter,
    export_lora_state_dict,
    full_weights_path,
    load_adapter_onto_model,
    lora_adapter_path,
    resolve_lora_adapter_dir,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.q_proj(inputs)


def _tiny_peft_model() -> nn.Module:
    return get_peft_model(
        TinyModel(),
        LoraConfig(
            r=2,
            lora_alpha=2,
            target_modules=["q_proj"],
            lora_dropout=0.0,
            init_lora_weights="gaussian",
        ),
    )


def test_checkpoint_paths() -> None:
    assert full_weights_path("/tmp/ckpt/actor").endswith(
        "model_state_dict/full_weights.pt"
    )
    assert lora_adapter_path("/tmp/ckpt/actor").endswith("lora_adapter")


def test_save_lora_adapter_preserves_full_weights(tmp_path: Path) -> None:
    actor_dir = tmp_path / "actor"
    weights_dir = actor_dir / "model_state_dict"
    weights_dir.mkdir(parents=True)
    full_weights = weights_dir / "full_weights.pt"
    full_payload = {"base.weight": torch.ones(2, 2)}
    torch.save(full_payload, full_weights)

    peft_model = _tiny_peft_model()
    adapter_dir = lora_adapter_path(str(actor_dir))
    peft_model.save_pretrained(adapter_dir)

    reloaded_full = torch.load(full_weights, map_location="cpu", weights_only=True)
    assert set(reloaded_full) == {"base.weight"}
    assert (Path(adapter_dir) / "adapter_config.json").is_file()

    resolved = resolve_lora_adapter_dir(str(actor_dir))
    assert Path(resolved) == Path(adapter_dir)


def test_resolve_lora_adapter_dir_from_global_step(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "global_step_100" / "actor" / "lora_adapter"
    adapter_dir.mkdir(parents=True)
    _tiny_peft_model().save_pretrained(str(adapter_dir))

    resolved = resolve_lora_adapter_dir(str(tmp_path / "global_step_100"))
    assert Path(resolved) == adapter_dir


def test_load_adapter_onto_model_from_peft_artifact(tmp_path: Path) -> None:
    source = _tiny_peft_model()
    with torch.no_grad():
        source.base_model.model.q_proj.lora_A.default.weight.fill_(1.0)
    adapter_dir = tmp_path / "lora_adapter"
    source.save_pretrained(str(adapter_dir))

    target = TinyModel()
    model = load_adapter_onto_model(target, str(tmp_path), adapter_name="default")
    loaded = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_A" in key
    }
    assert len(loaded) == 1
    torch.testing.assert_close(next(iter(loaded.values())), torch.ones(2, 4))


def test_load_adapter_from_legacy_full_weights(tmp_path: Path) -> None:
    """Legacy full_weights.pt fallback must return the Peft-wrapped model."""
    from peft import get_peft_model_state_dict

    source = _tiny_peft_model()
    with torch.no_grad():
        source.base_model.model.q_proj.lora_A.default.weight.fill_(1.0)
    lora_state = {
        key: value.detach().cpu()
        for key, value in get_peft_model_state_dict(source).items()
    }
    weights_dir = tmp_path / "actor" / "model_state_dict"
    weights_dir.mkdir(parents=True)
    torch.save(lora_state, weights_dir / "full_weights.pt")

    model = load_adapter_onto_model(
        TinyModel(), str(tmp_path / "actor"), adapter_name="default"
    )
    assert hasattr(model, "peft_config")
    loaded = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if "lora_A" in key
    }
    assert len(loaded) == 1
    torch.testing.assert_close(next(iter(loaded.values())), torch.ones(2, 4))


def test_export_lora_adapter_skip_non_peft(tmp_path: Path) -> None:
    warnings: list[str] = []
    wrote = export_lora_adapter(
        TinyModel(),
        str(tmp_path / "actor"),
        rank=0,
        log_warning=warnings.append,
    )
    assert wrote is False
    assert warnings and "peft_config" in warnings[0]


def test_export_lora_state_dict_collects_lora_keys() -> None:
    model = _tiny_peft_model()
    state = export_lora_state_dict(model)
    assert state
    assert any("lora_" in key for key in state)


def _distributed_save_failure_worker(
    rank: int, world_size: int, store_file: str
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        outcome = ["ok", None]
        if rank == 0:
            try:
                raise OSError("disk full")
            except Exception as error:  # noqa: BLE001 - mirror production catch
                outcome = ["error", f"{type(error).__name__}: {error}"]
        outcome = _broadcast_rank0_outcome(outcome, rank=rank)
        kind, detail = outcome
        if kind == "error":
            raise RuntimeError(f"LoRA adapter export failed on rank 0: {detail}")
    finally:
        dist.destroy_process_group()


def test_distributed_rank0_save_failure_raises_on_all_ranks() -> None:
    """Simulated rank-0 I/O failure must propagate instead of hanging other ranks."""
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = os.path.join(tmp_dir, "pg_store")
        ctx = mp.get_context("spawn")
        processes = [
            ctx.Process(
                target=_distributed_save_failure_worker,
                args=(rank, world_size, store_file),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode is not None, "distributed worker hung"
            assert process.exitcode != 0, "expected all ranks to raise on save failure"


def _distributed_export_skip_worker(
    rank: int, world_size: int, store_file: str, save_path: str
) -> None:
    import torch.distributed as dist

    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{store_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        warnings: list[str] = []
        wrote = export_lora_adapter(
            nn.Linear(2, 2),
            save_path,
            rank=rank,
            log_warning=warnings.append,
        )
        assert wrote is False
        if rank == 0:
            assert warnings
            assert "peft_config" in warnings[0]
    finally:
        dist.destroy_process_group()


def test_distributed_export_skip_is_consistent() -> None:
    world_size = 2
    with tempfile.TemporaryDirectory() as tmp_dir:
        store_file = os.path.join(tmp_dir, "pg_store")
        save_path = os.path.join(tmp_dir, "actor")
        os.makedirs(save_path, exist_ok=True)
        ctx = mp.get_context("spawn")
        processes = [
            ctx.Process(
                target=_distributed_export_skip_worker,
                args=(rank, world_size, store_file, save_path),
            )
            for rank in range(world_size)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=30)
            assert process.exitcode == 0, process.exitcode
