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

"""Save training-loader position alongside FSDP model checkpoints."""

import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.utils.utils import get_rng_state, set_rng_state


def _check_rank_errors(error: Exception | None, operation: str) -> None:
    message = f"{type(error).__name__}: {error}" if error is not None else None
    messages = [message]
    if torch.distributed.is_initialized():
        messages = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(messages, message)
    failures = [f"rank {rank}: {text}" for rank, text in enumerate(messages) if text]
    if failures:
        exception = ValueError if operation == "load" else RuntimeError
        raise exception(
            f"SFT data checkpoint {operation} failed: {'; '.join(failures)}"
        )


class SFTDataCheckpointMixin:
    """Coordinate per-rank data snapshots with model and optimizer saves."""

    def _save_checkpoint_assets(self, save_path: str) -> None:
        """Persist task-specific assets before publishing data completion."""

    def _data_checkpoint_settings(self) -> dict[str, Any]:
        data = OmegaConf.to_container(self.cfg.data, resolve=True, enum_to_str=True)
        for key in (
            "eval_data_paths",
            "val_data_paths",
            "eval_num_workers",
            "eval_batch_size",
        ):
            data.pop(key, None)
        settings = {
            "data": data,
            "model": OmegaConf.to_container(
                self.cfg.actor.model, resolve=True, enum_to_str=True
            ),
            "seed": self.cfg.actor.get("seed", 42),
            "global_batch_size": self.cfg.actor.global_batch_size,
            "micro_batch_size": self.cfg.actor.micro_batch_size,
        }
        return json.loads(json.dumps(settings, default=os.fspath, allow_nan=False))

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save a resumable model and the next consumed training batch."""
        path = Path(save_path)
        incomplete = path / "data_state.incomplete"
        rng = get_rng_state()
        try:
            error = None
            settings = None
            try:
                path.mkdir(parents=True, exist_ok=True)
                settings = self._data_checkpoint_settings()
                if self._rank == 0:
                    incomplete.write_text(str(step))
            except Exception as exc:
                error = exc
            _check_rank_errors(error, "save")

            error = None
            temporary = path / f"data_rank_{self._rank}.pt.tmp"
            try:
                super().save_checkpoint(save_path, step)
                self._save_checkpoint_assets(save_path)
                state = {
                    "step": step,
                    "data": self.data_loader.state_dict(),
                    "rng": rng,
                    "data_iter_offset": getattr(self, "_data_iter_offset", None),
                }
                torch.save(state, temporary)
                os.replace(temporary, path / f"data_rank_{self._rank}.pt")
            except Exception as exc:
                error = exc
            finally:
                temporary.unlink(missing_ok=True)
            _check_rank_errors(error, "save")

            error = None
            temporary = path / "data_state.json.tmp"
            if self._rank == 0:
                try:
                    manifest = {
                        "version": 1,
                        "step": step,
                        "world_size": self._world_size,
                        "settings": settings,
                        "rank_files": {
                            f"data_rank_{rank}.pt": hashlib.sha256(
                                (path / f"data_rank_{rank}.pt").read_bytes()
                            ).hexdigest()
                            for rank in range(self._world_size)
                        },
                    }
                    temporary.write_text(json.dumps(manifest))
                    os.replace(temporary, path / "data_state.json")
                    incomplete.unlink()
                except Exception as exc:
                    error = exc
                finally:
                    temporary.unlink(missing_ok=True)
            _check_rank_errors(error, "save")
        finally:
            set_rng_state(rng)

    def _read_data_checkpoint(self, path: Path) -> dict[str, Any] | None:
        """Validate rank snapshots before entering model-load collectives."""
        if (path / "data_state.incomplete").exists():
            raise ValueError(
                f"Found data_state.incomplete in {path}; resume from a completed checkpoint."
            )
        manifest_path = path / "data_state.json"
        state = None
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest["version"] != 1:
                raise ValueError(
                    f"SFT data checkpoint version must be 1, got {manifest['version']}."
                )
            if manifest["world_size"] != self._world_size:
                raise ValueError(
                    "SFT data checkpoint world size differs: "
                    f"saved world_size={manifest['world_size']}; "
                    f"configured world_size={self._world_size}."
                )
            for key, configured in self._data_checkpoint_settings().items():
                saved = manifest["settings"].get(key)
                if saved != configured:
                    raise ValueError(
                        f"SFT data checkpoint {key} differs: configured {configured!r}, saved {saved!r}."
                    )
            for rank in range(self._world_size):
                if not (path / f"data_rank_{rank}.pt").is_file():
                    raise ValueError(
                        f"SFT data checkpoint is missing data_rank_{rank}.pt in {path}."
                    )
            filename = f"data_rank_{self._rank}.pt"
            payload = (path / filename).read_bytes()
            if hashlib.sha256(payload).hexdigest() != manifest["rank_files"][filename]:
                raise ValueError(
                    f"SFT data checkpoint checksum differs for {filename} in {path}."
                )
            state = torch.load(io.BytesIO(payload), weights_only=False)
            if state["step"] != manifest["step"]:
                raise ValueError(
                    f"SFT data checkpoint step differs from the manifest in {path}."
                )
        elif list(path.glob("data_rank_*.pt")):
            raise ValueError(
                f"SFT data checkpoint has rank snapshots but no data_state.json in {path}."
            )
        else:
            self.log_warning(
                f"Checkpoint {path} has no training data state; samples restart from the beginning."
            )
        return state

    def load_checkpoint(self, load_path: str) -> None:
        """Restore data state after loading model, optimizer and scheduler."""
        state = None
        error = None
        try:
            state = self._read_data_checkpoint(Path(load_path))
        except Exception as exc:
            error = exc
        _check_rank_errors(error, "load")
        super().load_checkpoint(load_path)
        if state is not None:
            self.data_loader.load_state_dict(state["data"])
            self._current_epoch = state["data"]["epoch"]
            self._data_epoch = self._current_epoch
            if state["data_iter_offset"] is not None:
                self._data_iter_offset = state["data_iter_offset"]
            self.data_iter = iter(self.data_loader)
            set_rng_state(state["rng"])
