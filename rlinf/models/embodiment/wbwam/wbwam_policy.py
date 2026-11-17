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

"""Evaluation policy using the upstream Native50 inference server."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.sim.humanoidarena.runtime import (
    RuntimeProcess,
    request_json,
    validate_actions,
)
from rlinf.models.embodiment.base_policy import BasePolicy


class WBWAMPolicy(torch.nn.Module, BasePolicy):
    """Keep WB-WAM dependencies isolated in the configured upstream Python."""

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self._process = None
        self._url = None

    def _start(self) -> None:
        config = self.cfg.wbwam
        repo = Path(config.repo_root).expanduser().resolve()
        output = (
            Path(config.output_dir).expanduser().resolve() / f"policy-{uuid4().hex}"
        )
        output.mkdir(parents=True)
        command = [
            str(config.python),
            "-u",
            str(repo / "benchmark/humanoidarena/server.py"),
            "--checkpoint-root",
            str(self.cfg.model_path),
            "--weight-file",
            str(config.weight_file),
            "--instruction-key",
            str(config.instruction_key),
            "--expected-prompt",
            str(config.prompt),
            "--wbwam-root",
            str(repo / "training"),
            "--output",
            str(output),
            "--seed",
            str(config.seed),
            "--port",
            "0",
            "--replan",
            str(self.cfg.num_action_chunks),
            "--denoise-steps",
            str(config.denoise_steps),
            "--inference-mode",
            "idm",
        ]
        env = dict(os.environ)
        env.update(
            {
                "DIFFSYNTH_MODEL_BASE_PATH": str(config.base_models_root),
                "DIFFSYNTH_SKIP_DOWNLOAD": "true",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": str(repo / "training/src"),
            }
        )
        self._process = RuntimeProcess(
            command, output / "server.log", cwd=repo, env=env
        )
        deadline = time.monotonic() + float(config.timeout)
        health_path = output / "server_health.json"
        try:
            while time.monotonic() < deadline:
                self._process.check()
                if health_path.exists():
                    try:
                        health = json.loads(health_path.read_text())
                    except json.JSONDecodeError:
                        continue
                    self._url = f"http://127.0.0.1:{health['port']}"
                    health = request_json(self._url + "/health", timeout=10)
                    expected = {
                        "state_dim": 64,
                        "action_dim": 40,
                        "action_source_hz": 50,
                        "execute_chunk_steps": int(self.cfg.num_action_chunks),
                        "num_inference_steps": int(config.denoise_steps),
                        "inference_mode": "idm",
                        "action_format": "semantic_v3",
                        "instruction_prompt": str(config.prompt),
                    }
                    for key, value in expected.items():
                        if health.get(key) != value:
                            raise ValueError(
                                f"WB-WAM health {key}: expected {value!r}, got {health.get(key)!r}"
                            )
                    return
                time.sleep(0.2)
            raise TimeoutError(
                f"WB-WAM server startup timed out; see {self._process.log_path}"
            )
        except BaseException:
            self.close()
            raise

    def default_forward(self, **kwargs: Any) -> Any:
        """Reject optimization through the upstream inference-only service."""
        raise NotImplementedError("WB-WAM HTTP integration supports evaluation only")

    def predict_action_batch(
        self, env_obs: dict, mode: str = "eval", **kwargs: Any
    ) -> tuple[np.ndarray, dict]:
        """Infer a single native semantic40 chunk without rescaling hand values."""
        del kwargs
        if mode != "eval":
            raise ValueError("WB-WAM HTTP integration requires mode='eval'")
        states = env_obs["states"]
        images = env_obs["main_images"]
        if isinstance(states, torch.Tensor):
            states = states.detach().cpu().numpy()
        if isinstance(images, torch.Tensor):
            images = images.detach().cpu().numpy()
        if np.shape(states) != (1, 64) or not np.isfinite(states).all():
            raise ValueError(f"Expected finite states[1, 64], got {np.shape(states)}")
        if (
            images.dtype != np.uint8
            or images.ndim != 4
            or images.shape[0] != 1
            or images.shape[-1] != 3
        ):
            raise ValueError(
                f"Expected uint8 images[1, H, W, 3], got {images.shape}, {images.dtype}"
            )
        if env_obs["task_descriptions"] != [str(self.cfg.wbwam.prompt)]:
            raise ValueError(
                "Environment prompt must match the task-specific WB-WAM checkpoint"
            )
        if self._process is None:
            self._start()
        self._process.check()
        if bool(env_obs["reset_mask"][0]):
            request_json(self._url + "/reset", {})
        rgb = np.ascontiguousarray(images[0])
        payload = {
            "observation": {
                "state": states[0].tolist(),
                "images": {
                    "front": {
                        "shape": list(rgb.shape),
                        "dtype": "uint8",
                        "data_b64": base64.b64encode(rgb.tobytes()).decode(),
                    }
                },
            }
        }
        result = request_json(
            self._url + "/infer", payload, timeout=float(self.cfg.wbwam.timeout)
        )
        actions = validate_actions(
            np.asarray(result["action_chunk"])[None], int(self.cfg.num_action_chunks)
        )
        return actions, {}

    def close(self) -> None:
        """Release the original inference server and its subprocesses."""
        if self._process is not None:
            self._process.close()
            self._process = None
        self._url = None
