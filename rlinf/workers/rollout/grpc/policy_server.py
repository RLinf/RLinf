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

"""Fixed-checkpoint policy serving, independent of Ray and robot hardware."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

import grpc
import numpy as np
import torch

from .protocol import (
    OPTIONS,
    SERVICE,
    VERSION,
    pack_message,
    unpack_message,
    validate_actions,
    validate_observations,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from rlinf.models.embodiment.base_policy import BasePolicy


class PolicyServer:
    """Serve a loaded policy on a trusted interface.

    Calls to the model are serialized. The caller owns the model and must load
    it before start(). close() is idempotent; an instance cannot be restarted.
    """

    def __init__(
        self,
        policy: "BasePolicy",
        *,
        action_dim: int,
        num_action_chunks: int,
        policy_id: str,
        host: str = "127.0.0.1",
        port: int = 50051,
    ) -> None:
        if action_dim < 1 or num_action_chunks < 1 or not policy_id:
            raise ValueError("Positive action dimensions and a policy_id are required")
        if not 0 <= port <= 65535:
            raise ValueError("port must be between 0 and 65535")
        self.policy = policy
        self.metadata = {
            "version": VERSION,
            "action_dim": action_dim,
            "num_action_chunks": num_action_chunks,
            "policy_id": policy_id,
        }
        self.host = host
        self.port = port
        self._lock = threading.Lock()
        self._server = None
        self._executor = None
        self._closed = False

    def start(self) -> int:
        """Bind and start the service; return the actual port (0 selects one)."""
        if self._closed or self._server is not None:
            raise RuntimeError("PolicyServer has already been started or closed")
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._server = grpc.server(
            self._executor, options=OPTIONS, maximum_concurrent_rpcs=2
        )
        handlers = {
            name: grpc.unary_unary_rpc_method_handler(
                method,
                request_deserializer=unpack_message,
                response_serializer=pack_message,
            )
            for name, method in (("Health", self._health), ("Predict", self._predict))
        }
        self._server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler(SERVICE, handlers),)
        )
        try:
            self.port = self._server.add_insecure_port(f"{self.host}:{self.port}")
            if not self.port:
                raise RuntimeError("Could not bind policy server")
            self._server.start()
        except BaseException:
            self.close()
            raise
        return self.port

    def _health(self, request: dict, context: grpc.ServicerContext) -> dict:
        return dict(self.metadata)

    def _predict(self, request: dict, context: grpc.ServicerContext) -> dict:
        try:
            if request.get("policy_id") != self.metadata["policy_id"]:
                raise ValueError("policy_id does not match the loaded checkpoint")
            request_id = request["request_id"]
            if not isinstance(request_id, str) or not request_id:
                raise ValueError("request_id must be a nonempty string")
            obs = request["observations"]
            batch = validate_observations(obs)
        except (KeyError, TypeError, ValueError) as exc:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(exc))
        # A timed-out queued request must never become new inference work.
        with self._lock:
            if not context.is_active():
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Request expired")
            try:
                env_obs = {
                    key: torch.from_numpy(value)
                    if isinstance(value, np.ndarray)
                    else value
                    for key, value in obs.items()
                }
                with torch.inference_mode():
                    actions, _ = self.policy.predict_action_batch(
                        env_obs=env_obs, mode="eval"
                    )
                if isinstance(actions, torch.Tensor):
                    actions = actions.detach().cpu().float().numpy()
                validate_actions(
                    actions,
                    batch,
                    self.metadata["num_action_chunks"],
                    self.metadata["action_dim"],
                )
            except Exception:
                logger.exception("Policy inference failed")
                context.abort(
                    grpc.StatusCode.INTERNAL, "Policy inference failed; see server log"
                )
        return {**self.metadata, "request_id": request_id, "actions": actions}

    def wait(self) -> None:
        """Block until the started service terminates."""
        if self._server is None:
            raise RuntimeError("Policy server is not started")
        self._server.wait_for_termination()

    def close(self) -> None:
        """Stop accepting RPCs and release the service's thread pool."""
        if self._closed:
            return
        self._closed = True
        if self._server is not None:
            self._server.stop(grace=0).wait()
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)


def load_policy(
    model_cfg: Any, device: str, ckpt_path: str | None = None
) -> "BasePolicy":
    """Build a policy using the same factory and optional state dict as local eval."""
    from rlinf.models import get_model

    if ckpt_path and not Path(ckpt_path).is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
    policy = get_model(model_cfg)
    if policy is None:
        raise ValueError(f"Unknown model type: {model_cfg.model_type}")
    if ckpt_path and Path(ckpt_path).is_file():
        policy.load_state_dict(
            torch.load(ckpt_path, map_location="cpu", weights_only=True)
        )
    return policy.to(device).eval()


def serve_policy(config: Any, host: str | None = None, ready_pipe: Any = None) -> None:
    """Load a fixed policy and serve it without importing Ray."""
    server = None

    def notify(message: tuple[bool, Any]) -> None:
        if ready_pipe is None:
            return
        try:
            ready_pipe.send(message)
        except (BrokenPipeError, OSError):
            # The Ray owner may have timed out and closed the pipe already.
            pass

    try:
        policy = load_policy(
            config.model, config.server.device, config.get("ckpt_path")
        )
        if config.server.get("enable_torch_compile", False):
            policy.enable_torch_compile(mode=config.server.torch_compile_mode)
        if config.server.get("enable_cuda_graph", False):
            raise ValueError("gRPC serving does not yet support CUDA graph capture")
        server = PolicyServer(
            policy,
            action_dim=config.model.action_dim,
            num_action_chunks=config.model.num_action_chunks,
            policy_id=config.server.policy_id,
            host=host or config.server.host,
            port=config.server.port,
        )
        port = server.start()
        logger.info("Policy service ready on %s:%s", host or config.server.host, port)
        notify((True, port))
        server.wait()
    except BaseException as exc:
        notify((False, repr(exc)))
        raise
    finally:
        if server is not None:
            server.close()
        if ready_pipe is not None:
            ready_pipe.close()
