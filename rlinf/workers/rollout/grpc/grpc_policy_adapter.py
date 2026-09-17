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

"""gRPC policy adapter that implements BasePolicy interface for remote inference."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rlinf.models.embodiment.base_policy import BasePolicy


class GRPCPolicyAdapter(BasePolicy):
    """
    A policy adapter that forwards predict_action_batch calls to a remote
    gRPC policy server (LeRobot protocol).

    This adapter implements the BasePolicy interface, so it can be used as a
    drop-in replacement for local models in MultiStepRolloutWorker.
    """

    def __init__(self, server_address: str, timeout: float = 30.0, device: str = "cuda"):
        """
        Args:
            server_address: gRPC server address (e.g., "localhost:50051")
            timeout: Request timeout in seconds
            device: Device for tensor operations
        """
        super().__init__()
        self.server_address = server_address
        self.timeout = timeout
        self._device = device

        # gRPC client initialization
        self._grpc_channel = None
        self._grpc_stub = None
        self._init_grpc_client()

    def _init_grpc_client(self):
        """Initialize the gRPC client."""
        import grpc

        try:
            from lerobot.common.policies.policy_protocol_pb2_grpc import PolicyStub
        except ImportError as e:
            raise ImportError(
                "LeRobot gRPC protocol not available. "
                "Please install LeRobot with: pip install lerobot"
            ) from e

        self._grpc_channel = grpc.insecure_channel(self.server_address)
        self._grpc_stub = PolicyStub(self._grpc_channel)
        print(f"[GRPCPolicyAdapter] Connected to gRPC server at {self.server_address}")

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Predict action batch via gRPC call.

        Args:
            env_obs: Environment observations
            mode: "train" or "eval" (ignored for gRPC)
            **kwargs: Additional arguments (ignored for gRPC)

        Returns:
            Tuple of (actions, result_dict)
        """
        batch_size = self._infer_batch_size(env_obs)
        actions_list = []

        # Process each observation in the batch
        for i in range(batch_size):
            obs_i = self._extract_single_obs(env_obs, i)
            action = self._predict_single(obs_i)
            actions_list.append(action)

        # Stack actions into batch
        actions = np.stack(actions_list, axis=0)

        # Build result dict (minimal, matching BasePolicy interface)
        result = {
            "forward_inputs": {
                "action": torch.from_numpy(actions).to(self._device),
                "model_action": None,
            },
            "expert_label_flag": False,
        }

        return actions, result

    def _predict_single(self, obs: dict[str, Any]) -> np.ndarray:
        """Call gRPC server for a single observation."""
        import grpc

        request = self._build_grpc_request(obs)

        try:
            response = self._grpc_stub.GetActionChunk(request, timeout=self.timeout)
            action = np.array(response.action.data, dtype=np.float32).reshape(
                response.action.shape.dimensions
            )
            return action
        except grpc.RpcError as e:
            raise RuntimeError(f"gRPC inference failed: {e}") from e

    def _build_grpc_request(self, obs: dict[str, Any]):
        """Build a gRPC ActionChunkRequest from an observation dict."""
        from lerobot.common.policies.policy_protocol_pb2 import (
            ActionChunkRequest,
            Image,
            Observation,
            Tensor,
            TensorShape,
        )

        grpc_obs = Observation()

        # Add images (main_images, wrist_images, extra_view_images)
        for key in ["main_images", "wrist_images", "extra_view_images"]:
            if key in obs and obs[key] is not None:
                img_tensor = obs[key]
                if isinstance(img_tensor, torch.Tensor):
                    img_np = img_tensor.cpu().numpy()
                else:
                    img_np = np.asarray(img_tensor)

                # Convert to uint8 HWC format
                if img_np.dtype != np.uint8:
                    img_np = (img_np * 255).astype(np.uint8)
                if img_np.ndim == 3 and img_np.shape[0] == 3:  # CHW -> HWC
                    img_np = np.transpose(img_np, (1, 2, 0))

                # Create Image message
                img_msg = Image(
                    data=img_np.tobytes(),
                    shape=TensorShape(dimensions=list(img_np.shape)),
                    encoding="rgb8",
                )
                grpc_obs.images[key].CopyFrom(img_msg)

        # Add state
        if "states" in obs and obs["states"] is not None:
            state_tensor = obs["states"]
            if isinstance(state_tensor, torch.Tensor):
                state_np = state_tensor.cpu().numpy().astype(np.float32)
            else:
                state_np = np.asarray(state_tensor, dtype=np.float32)

            grpc_obs.state.CopyFrom(
                Tensor(
                    data=state_np.tobytes(),
                    shape=TensorShape(dimensions=list(state_np.shape)),
                    dtype="float32",
                )
            )

        # Create request
        request = ActionChunkRequest(observation=grpc_obs)

        # Add task description if available
        task = obs.get("task_description") or obs.get("language_instruction")
        if task:
            request.task = task

        return request

    def _infer_batch_size(self, obs: dict[str, Any]) -> int:
        """Infer the batch size from an observation dictionary."""
        for value in obs.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return value.shape[0]
            elif isinstance(value, np.ndarray) and value.ndim > 0:
                return value.shape[0]
            elif isinstance(value, (list, tuple)) and len(value) > 0:
                return len(value)
        return 1

    def _extract_single_obs(self, obs: dict[str, Any], index: int) -> dict:
        """Extract a single observation from a batched observation dict."""
        single_obs = {}
        for key, value in obs.items():
            if isinstance(value, (torch.Tensor, np.ndarray)) and value.ndim > 0:
                single_obs[key] = value[index]
            elif isinstance(value, (list, tuple)):
                single_obs[key] = value[index]
            else:
                single_obs[key] = value
        return single_obs

    def eval(self):
        """No-op for compatibility with BasePolicy interface."""
        return self

    def to(self, device):
        """Update device for tensor operations."""
        self._device = device
        return self

    def __del__(self):
        """Clean up gRPC resources."""
        if self._grpc_channel is not None:
            self._grpc_channel.close()
