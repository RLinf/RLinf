# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Run the SO-101 inference and intervention path without hardware.

The harness uses zero-valued follower/leader observations and an in-process
gRPC policy that returns zero action chunks. It is intentionally an experiment
tool, not a replacement for the real-world entry points.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import grpc
import numpy as np

from rlinf.envs.real.so101.reach import SO101ReachConfig, SO101ReachEnv
from rlinf.envs.real.wrappers.teleop.intervention import (
    TeleopDevice,
    TeleopIntervention,
    TeleopSample,
)
from rlinf.envs.real.wrappers.teleop.trigger import (
    RELEASE,
    TAKEOVER,
    InterventionTrigger,
)
from rlinf.workers.rollout.grpc.grpc_policy_adapter import GRPCPolicyAdapter
from rlinf.workers.rollout.grpc.protocol import (
    SERVICE,
    VERSION,
    pack_message,
    unpack_message,
)


ACTION_DIM = 6
ACTION_CHUNK = 20
POLICY_ID = "so101-mock-zero"


class _ZeroPolicyServer:
    """Minimal protocol-compatible policy service for local testing."""

    def __init__(self) -> None:
        self.requests = 0

    def health(self, _request: dict[str, Any], _context: grpc.ServicerContext) -> dict[str, Any]:
        return {
            "version": VERSION,
            "action_dim": ACTION_DIM,
            "num_action_chunks": ACTION_CHUNK,
            "policy_id": POLICY_ID,
        }

    def predict(self, request: dict[str, Any], _context: grpc.ServicerContext) -> dict[str, Any]:
        observations = request["observations"]
        batch = int(np.asarray(observations["states"]).shape[0])
        self.requests += 1
        return {
            "version": VERSION,
            "action_dim": ACTION_DIM,
            "num_action_chunks": ACTION_CHUNK,
            "policy_id": POLICY_ID,
            "request_id": request["request_id"],
            "actions": np.zeros((batch, ACTION_CHUNK, ACTION_DIM), dtype=np.float32),
        }


def _start_zero_server() -> tuple[grpc.Server, _ZeroPolicyServer, str]:
    service = _ZeroPolicyServer()
    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    handlers = {
        "Health": grpc.unary_unary_rpc_method_handler(
            service.health,
            request_deserializer=unpack_message,
            response_serializer=pack_message,
        ),
        "Predict": grpc.unary_unary_rpc_method_handler(
            service.predict,
            request_deserializer=unpack_message,
            response_serializer=pack_message,
        ),
    }
    server.add_generic_rpc_handlers(
        (grpc.method_handlers_generic_handler(SERVICE, handlers),)
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, service, f"127.0.0.1:{port}"


@dataclass
class _Trigger(InterventionTrigger):
    events: list[str] = field(default_factory=list)

    def push(self, event: str) -> None:
        self.events.append(event)

    def poll(self) -> list[str]:
        events, self.events = self.events, []
        return events


class _ZeroLeader(TeleopDevice):
    """Leader mock that exposes the hold and takeover lifecycle."""

    def __init__(self) -> None:
        self.active = False
        self.prepared = 0
        self.started = 0
        self.ended = 0
        self.closed = 0

    def read(self, _env: Any, _policy_action: np.ndarray) -> TeleopSample:
        return TeleopSample(np.zeros(ACTION_DIM, dtype=np.float32), self.active)

    def prepare_intervention(self, _env: Any) -> None:
        self.prepared += 1

    def on_intervention_start(self, _env: Any) -> None:
        self.active = True
        self.started += 1

    def on_intervention_end(self, _env: Any) -> None:
        self.active = False
        self.ended += 1

    def get_hold_action(
        self, _env: Any, fallback_action: np.ndarray | None = None
    ) -> np.ndarray:
        del fallback_action
        return np.zeros(ACTION_DIM, dtype=np.float32)

    def close(self) -> None:
        self.closed += 1


def run() -> None:
    server, policy_service, address = _start_zero_server()
    env = SO101ReachEnv(
        {
            "is_dummy": True,
            "zero_dummy_observation": True,
            "reset_on_init": False,
            "step_frequency": 1000.0,
            "max_num_steps": 20,
            "target_joint_qpos": [1.0] * 5,
            "reset_joint_qpos": [0.0] * 5,
            "reset_gripper_position": 0.0,
        }
    )
    leader = _ZeroLeader()
    trigger = _Trigger()
    wrapped = TeleopIntervention(
        env,
        leader,
        mark_flag=True,
        mode="explicit",
        trigger=trigger,
        buffer_seconds=0.0,
    )
    try:
        wrapped.reset()
        observation, _, _, _, _ = wrapped.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert np.asarray(observation["state"]["arm_joint_position"]).sum() == 0.0

        policy = GRPCPolicyAdapter(
            address,
            action_dim=ACTION_DIM,
            num_action_chunks=ACTION_CHUNK,
            policy_id=POLICY_ID,
            timeout=5.0,
        )
        try:
            request = {
                "states": np.zeros((1, ACTION_DIM), dtype=np.float32),
                "main_images": np.zeros((1, 224, 224, 3), dtype=np.uint8),
                "task_descriptions": ["mock SO-101 task"],
            }
            actions, _ = policy.predict_action_batch(request)
            assert tuple(actions.shape) == (1, ACTION_CHUNK, ACTION_DIM)
            assert bool(np.isfinite(actions.numpy()).all())
        finally:
            policy.close()

        trigger.push(TAKEOVER)
        _, _, _, _, info = wrapped.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert info.get("intervene_buffer") is not None
        _, _, _, _, info = wrapped.step(np.zeros(ACTION_DIM, dtype=np.float32))
        assert info.get("intervene_flag") is not None
        trigger.push(RELEASE)
        wrapped.step(np.zeros(ACTION_DIM, dtype=np.float32))
        wrapped.on_action_chunk_begin()
        assert leader.prepared == 1 and leader.started == 1 and leader.ended == 1
        print(
            f"PASS mock SO-101 chain: grpc_requests={policy_service.requests} "
            f"leader=prepared:{leader.prepared},started:{leader.started},ended:{leader.ended}"
        )
    finally:
        wrapped.close()
        env.close()
        server.stop(0).wait()
        assert leader.closed == 1


if __name__ == "__main__":
    run()
