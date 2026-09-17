#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0


# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve an RLinf SO101 PI05 checkpoint via LeRobot's official gRPC protocol."""

from __future__ import annotations

import argparse
import json
import logging
import pickle  # nosec: LeRobot's official RPC protocol uses pickle.
import time
from concurrent import futures
from pathlib import Path
from queue import Empty
from typing import Any

import grpc
import jax
import numpy as np
import torch
from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import (
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import receive_bytes_in_chunks

from rlinf.models import get_model
from rlinf.models.embodiment.openpi.policies.so101_policy import SO101_JOINT_NAMES
from rlinf.models.embodiment.openpi.so101_inference import build_so101_model_config


class RLinfSO101PolicyServer(PolicyServer):
    """Official PolicyServer scheduling with an RLinf PI05 inference backend."""

    def __init__(
        self,
        config: PolicyServerConfig,
        checkpoint: Path,
        norm_stats: Path,
        device: str,
        num_steps: int,
    ) -> None:
        super().__init__(config)
        self._checkpoint = checkpoint.resolve()
        self._norm_stats = norm_stats.resolve()
        self._backend_device = torch.device(device)
        self._num_steps = num_steps
        self._backend: Any | None = None
        self.preprocessor: Any | None = None
        self.postprocessor: Any | None = None
        self._policy_fingerprint: tuple[Any, ...] | None = None
        self._loaded_model_key: tuple[str, str, str] | None = None
        self._load_count = 0
        self._last_load_ms: float | None = None
        self._session_id = 0

    @staticmethod
    def _feature_fingerprint(features: Any) -> str:
        return json.dumps(features, sort_keys=True, default=repr, separators=(",", ":"))

    def _build_model_config(self, checkpoint: Path) -> Any:
        return build_so101_model_config(
            checkpoint, self._norm_stats, num_steps=self._num_steps
        )

    def _load_backend(self, checkpoint: Path, device: str) -> float:
        start = time.perf_counter()
        backend = (
            get_model(self._build_model_config(checkpoint))
            .to(torch.device(device))
            .eval()
        )
        self._backend = backend
        self.preprocessor = backend._input_transform  # noqa: SLF001
        self.postprocessor = backend._output_transform  # noqa: SLF001
        self._backend_device = torch.device(device)
        self._last_load_ms = (time.perf_counter() - start) * 1000.0
        self._load_count += 1
        self._loaded_model_key = ("pi05", str(checkpoint.resolve()), str(device))
        return self._last_load_ms

    def load_initial_policy(self) -> None:
        """Load the configured Pi05 policy before accepting gRPC clients."""
        if not (self._checkpoint / "model_state_dict/full_weights.pt").is_file():
            raise FileNotFoundError(
                f"Incomplete RLinf actor checkpoint: {self._checkpoint}"
            )
        if not self._norm_stats.is_file():
            raise FileNotFoundError(f"Missing norm_stats: {self._norm_stats}")
        load_ms = self._load_backend(self._checkpoint, str(self._backend_device))
        self.logger.info(
            "Initial policy load complete load_ms=%.2f load_count=%d checkpoint=%s device=%s",
            load_ms,
            self._load_count,
            self._checkpoint,
            self._backend_device,
        )

    def Ready(self, request, context):  # noqa: N802
        """Reset session state while retaining the already-loaded backend."""
        self._reset_server()
        self.shutdown_event.clear()
        self.last_processed_obs = None
        self.fps_tracker.reset()
        self._session_id += 1
        self.logger.info(
            "Client %s ready; session=%d; retained_policy=%s; queue_reset=true; timestep_reset=true",
            context.peer(),
            self._session_id,
            self._backend is not None,
        )
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Load the RLinf checkpoint while retaining the official RPC handshake."""
        policy_specs = pickle.loads(request.data)  # nosec
        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Expected RemotePolicyConfig, got {type(policy_specs)}")
        if policy_specs.policy_type != "pi05":
            raise ValueError("SO101 RLinf backend requires policy_type='pi05'.")
        if int(policy_specs.actions_per_chunk) != 20:
            raise ValueError(
                "SO101 RLinf PI05 serves exactly 20 actions per chunk; "
                f"got {policy_specs.actions_per_chunk}."
            )
        requested_checkpoint = Path(policy_specs.pretrained_name_or_path)
        checkpoint = (
            requested_checkpoint if requested_checkpoint.is_dir() else self._checkpoint
        )
        if not (checkpoint / "model_state_dict/full_weights.pt").is_file():
            raise FileNotFoundError(f"Incomplete RLinf actor checkpoint: {checkpoint}")
        if not self._norm_stats.is_file():
            raise FileNotFoundError(f"Missing norm_stats: {self._norm_stats}")

        fingerprint = (
            policy_specs.policy_type,
            str(checkpoint.resolve()),
            str(policy_specs.device),
            int(policy_specs.actions_per_chunk),
            self._feature_fingerprint(policy_specs.lerobot_features),
        )
        if self._backend is not None and fingerprint == self._policy_fingerprint:
            self.device = str(policy_specs.device)
            self.policy_type = policy_specs.policy_type
            self.lerobot_features = policy_specs.lerobot_features
            self.actions_per_chunk = 20
            self.logger.info("Reusing policy fingerprint=%s load_ms=0", fingerprint)
            return services_pb2.Empty()

        model_key = (
            policy_specs.policy_type,
            str(checkpoint.resolve()),
            str(policy_specs.device),
        )
        # The configured policy is loaded before the server starts. A first
        # instruction only binds its feature mapping; a changed fingerprint
        # performs a real reload.
        if self._backend is None or not (
            self._policy_fingerprint is None and model_key == self._loaded_model_key
        ):
            start = time.perf_counter()
            self._backend = (
                get_model(self._build_model_config(checkpoint))
                .to(torch.device(policy_specs.device))
                .eval()
            )
            self.preprocessor = self._backend._input_transform  # noqa: SLF001
            self.postprocessor = self._backend._output_transform  # noqa: SLF001
            self._backend_device = torch.device(policy_specs.device)
            self._last_load_ms = (time.perf_counter() - start) * 1000.0
            self._load_count += 1
            self._loaded_model_key = model_key
            self.logger.info(
                "Policy load complete load_ms=%.2f load_count=%d fingerprint=%s",
                self._last_load_ms,
                self._load_count,
                fingerprint,
            )
        else:
            self.logger.info(
                "Using startup-loaded policy load_ms=0.00 load_count=%d fingerprint=%s",
                self._load_count,
                fingerprint,
            )
        self._policy_fingerprint = fingerprint
        self.device = str(policy_specs.device)
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = 20
        self.logger.info(
            "Loaded RLinf PI05 backend checkpoint=%s norm_stats=%s device=%s",
            self._checkpoint,
            self._norm_stats,
            self._backend_device,
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Use official observation handling and classify client cancellation normally."""
        try:
            received = receive_bytes_in_chunks(
                request_iterator, None, self.shutdown_event, self.logger
            )
            if received is None:
                self.logger.info(
                    "Observation stream ended by client %s", context.peer()
                )
                return services_pb2.Empty()
            observation = pickle.loads(received)  # nosec
            self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
            self.logger.debug("Received observation #%s", observation.get_timestep())
            self._enqueue_observation(observation)
            return services_pb2.Empty()
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.CANCELLED:
                self.logger.info(
                    "Observation stream cancelled by client %s", context.peer()
                )
                return services_pb2.Empty()
            self.logger.error("Observation RPC failed: %s", exc)
            raise
        except Exception:
            self.logger.exception("Observation protocol/internal error")
            raise

    def GetActions(self, request, context):  # noqa: N802
        """Run inference while treating a cancelled client stream as normal."""
        try:
            observation = self.observation_queue.get(
                timeout=self.config.obs_queue_timeout
            )
            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(observation.get_timestep())
            action_chunk = self._predict_action_chunk(observation)
            if not context.is_active():
                self.logger.info("Action RPC cancelled by client %s", context.peer())
                return services_pb2.Empty()
            return services_pb2.Actions(data=pickle.dumps(action_chunk))  # nosec
        except Empty:
            return services_pb2.Empty()
        except grpc.RpcError as exc:
            if exc.code() == grpc.StatusCode.CANCELLED:
                self.logger.info("Action stream cancelled by client %s", context.peer())
                return services_pb2.Empty()
            self.logger.error("Action RPC failed: %s", exc)
            raise
        except Exception:
            self.logger.exception("Action model/protocol/internal error")
            raise

    @staticmethod
    def _as_hwc_rgb(value: Any) -> np.ndarray:
        image = (
            value.detach().cpu().numpy()
            if torch.is_tensor(value)
            else np.asarray(value)
        )
        if image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.moveaxis(image, 0, -1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Expected wrist RGB HWC/CHW image, got {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
            image = np.clip(image * scale, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image

    @staticmethod
    def _get_field(observation: dict[str, Any], key: str) -> Any:
        if key in observation:
            return observation[key]
        raise KeyError(f"Missing required SO101 observation field: {key}")

    def _predict_action_chunk(
        self, observation_t: TimedObservation
    ) -> list[TimedAction]:
        """Only adapt model I/O; timestamps/chunks remain LeRobot-owned."""
        if self._backend is None:
            raise RuntimeError("SendPolicyInstructions must complete before inference.")
        raw = observation_t.get_observation()
        state = self._get_field(raw, "observation.state")
        state = (
            state.detach().cpu().numpy()
            if torch.is_tensor(state)
            else np.asarray(state)
        )
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape != (len(SO101_JOINT_NAMES),):
            raise ValueError(f"Expected observation.state [6], got {state.shape}")
        # The local interface is [-1, 1], while PI05 uses [0, 1] for gripper.
        state = state.copy()
        state[-1] = (state[-1] + 1.0) / 2.0
        wrist = self._as_hwc_rgb(self._get_field(raw, "observation.images.wrist"))
        task = str(self._get_field(raw, "task"))
        inputs = self._backend._input_transform(  # noqa: SLF001
            {"observation/image": wrist, "observation/state": state, "prompt": task}
        )
        inputs = jax.tree.map(
            lambda value: torch.from_numpy(np.asarray(value).copy())[None].to(
                self._backend_device
            ),
            inputs,
        )
        from openpi.models import model as openpi_model

        model_observation = openpi_model.Observation.from_dict(inputs)
        with torch.inference_mode():
            sampled = self._backend.sample_actions(
                model_observation, mode="eval", compute_values=False
            )
            output = self._backend.output_transform(
                {"actions": sampled["actions"], "state": model_observation.state}
            )
        actions = np.asarray(output["actions"][0], dtype=np.float32)[:20, :6]
        if actions.shape != (20, 6):
            raise RuntimeError(f"Expected action chunk [20,6], got {actions.shape}")
        actions[:, -1] = 2.0 * actions[:, -1] - 1.0
        actions = np.clip(actions, -1.0, 1.0)
        action_tensors = [torch.from_numpy(action.copy()) for action in actions]
        return self._time_action_chunk(
            observation_t.get_timestamp(), action_tensors, observation_t.get_timestep()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--inference-latency", type=float, default=0.0)
    parser.add_argument("--obs-queue-timeout", type=float, default=2.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    config = PolicyServerConfig(
        host=args.host,
        port=args.port,
        fps=args.fps,
        inference_latency=args.inference_latency,
        obs_queue_timeout=args.obs_queue_timeout,
    )
    policy_server = RLinfSO101PolicyServer(
        config, args.checkpoint, args.norm_stats, args.device, args.num_steps
    )
    policy_server.load_initial_policy()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    policy_server.logger.info(
        "SO101 LeRobot gRPC PolicyServer listening at %s:%d", args.host, args.port
    )
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    main()
