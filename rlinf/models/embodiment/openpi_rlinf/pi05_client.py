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

"""Stage2 rollout-side stand-in for the frozen Stage1 VLA (TIANJI_PI05_SERVICE=1).

``Pi05ServiceFeatureModel`` replaces ``get_model(rollout.rlt_feature_model)``
in the rollout worker. It loads nothing: ``extract_rlt_obs`` draws the flow
noise exactly where the in-process eager sampler draws it (``torch.randn`` on
the rollout device, global generator), sends the VLA observation keys and the
noise to the resident ``pi05_service`` over its Unix socket, and finishes the
RLT observation (proprio, critic_proprio, ref_chunk_tcp) with the same
``_joint_rlt_obs`` code as the in-process wrapper, on the same device.

Before serving it compares the service's identity (model path, weight file,
norm stats, config, inference code hash) with what this tree expects and
refuses a service that differs. A service that stops mid-run raises a
RuntimeError on the next request (no hang, no silent reconnect).
"""

from __future__ import annotations

import os
import socket
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    TIANJI_JOINT_CONFIG,
    OpenPiPytorchEvalActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi05_service import (
    VLA_OBS_KEYS,
    identity_mismatch,
    local_identity,
    recv_msg,
    request,
    send_msg,
    socket_path,
)
from rlinf.utils.logging import get_logger

logger = get_logger()


def _to_wire(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return value


class Pi05ServiceFeatureModel(torch.nn.Module):
    """``rlt_feature_model`` whose VLA forward runs in the resident service."""

    def __init__(self, feature_cfg: Any, device: Any, *, path: Path | None = None, timeout: float = 30.0):
        super().__init__()
        from omegaconf import OmegaConf

        self.feature_cfg = feature_cfg
        self.rollout_device = torch.device(device)
        self.config_name = str(OmegaConf.select(feature_cfg, "openpi.config_name", default=""))
        if self.config_name != TIANJI_JOINT_CONFIG:
            raise ValueError(
                f"TIANJI_PI05_SERVICE supports openpi.config_name={TIANJI_JOINT_CONFIG!r} only, got {self.config_name!r}"
            )
        self.action_horizon = int(feature_cfg.num_action_chunks)
        self.action_dim = int(feature_cfg.openpi.model_action_dim)
        self.path = path or socket_path()
        self.timeout = timeout
        self._sock: socket.socket | None = None

        self.expected_identity = local_identity(feature_cfg)
        try:
            info = request({"op": "info"}, path=self.path, timeout=10.0)
        except OSError as exc:
            raise RuntimeError(
                f"TIANJI_PI05_SERVICE=1 but no pi05 service answers on {self.path} ({exc}); "
                "start it with launcher/tianji_pi05_service.sh ensure"
            ) from exc
        if info.get("error"):
            raise RuntimeError(f"pi05 service failed while {info.get('phase')}: {info['error']}")
        if not info.get("ready"):
            raise RuntimeError(f"pi05 service is not ready ({info.get('phase')}); the launcher must wait for it")
        reasons = identity_mismatch(self.expected_identity, info.get("identity", {}))
        if reasons:
            raise RuntimeError(
                "pi05 service identity differs from this Stage2 tree/config (restart it with "
                "launcher/tianji_pi05_service.sh restart):\n  " + "\n  ".join(reasons)
            )
        self.service_info = info
        self.start_token = info["start_token"]
        self._connect()
        self._write_ready_flag()
        logger.info(
            "pi05_client: using resident service pid %s on %s (%s; weights %s; code %s)",
            info.get("pid"), self.path, info.get("fast_mode"),
            str(info["identity"].get("weights_sha256"))[:12], str(info["identity"].get("code_sha256"))[:12],
        )

    # ------------------------------------------------------------ plumbing

    @property
    def device(self) -> torch.device:
        return self.rollout_device

    def _connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(str(self.path))
        self._sock = sock

    def _write_ready_flag(self) -> None:
        # What load_full_wrapper_weights writes after loading in-process: the
        # launcher opens the HIL panel once this flag is newer than the launch.
        try:
            flag = os.environ.get("RLT_READY_FLAG", "/tmp/rlt_rollout_ready")
            with open(flag, "w") as f:
                f.write(str(self.service_info["identity"].get("weights_file")))
        except OSError:
            pass

    def _call(self, msg: dict) -> dict:
        if self._sock is None:
            raise RuntimeError("pi05 service connection is closed; restart Stage2")
        try:
            send_msg(self._sock, msg)
            reply = recv_msg(self._sock)
        except socket.timeout as exc:
            self._drop()
            raise RuntimeError(f"pi05 service did not answer within {self.timeout:.0f} s; restart Stage2") from exc
        except OSError as exc:
            self._drop()
            raise RuntimeError(f"pi05 service connection lost ({exc}); is the service running? restart Stage2") from exc
        if reply is None:
            self._drop()
            raise RuntimeError("pi05 service closed the connection (service stopped?); restart Stage2")
        if not reply.get("ok"):
            raise RuntimeError(f"pi05 service error: {reply.get('error')}")
        if reply.get("start_token") != self.start_token:
            raise RuntimeError("pi05 service was restarted during this run; restart Stage2")
        return reply

    def _drop(self) -> None:
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            self._sock = None

    # ------------------------------------------------------------ the model

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        batch_size = int(torch.as_tensor(env_obs["joint_states"]).shape[0])
        # The eager sampler's draw (same call, same device, same global generator).
        noise = torch.randn(batch_size, self.action_horizon, self.action_dim, device=self.rollout_device)
        payload = {k: _to_wire(env_obs[k]) for k in VLA_OBS_KEYS if k in env_obs}
        t0 = time.perf_counter()
        reply = self._call({"op": "infer", "env_obs": payload, "noise": noise.cpu().numpy()})
        self.last_roundtrip_ms = (time.perf_counter() - t0) * 1e3
        self.last_service_ms = float(reply.get("service_ms", float("nan")))
        z_rl = torch.from_numpy(np.ascontiguousarray(reply["z_rl"])).to(self.rollout_device)
        ref_chunk = torch.from_numpy(np.ascontiguousarray(reply["ref_chunk"]))
        return OpenPiPytorchEvalActionModel._joint_rlt_obs(self, env_obs, z_rl, ref_chunk)

    def forward(self, *args, **kwargs):  # pragma: no cover - never called
        raise NotImplementedError("Pi05ServiceFeatureModel only provides extract_rlt_obs")
