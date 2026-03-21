# Copyright 2026 Ying-Chun Lee
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

"""Unit tests for the local YAM inference entrypoint."""

import numpy as np
import torch

from examples.embodiment import infer_embodied_agent as infer


class _RecordingChannel:
    def __init__(self):
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RecordingStub:
    def __init__(self):
        self.calls = []

    def Reset(self, request, timeout):
        self.calls.append(("reset", timeout, type(request).__name__))

    def EnterZeroTorqueMode(self, request, timeout):
        self.calls.append(("zero_torque", timeout, type(request).__name__))


class _ReturnHomeStub:
    def __init__(self):
        self.requests = []

    def ChunkStep(self, request, timeout):
        self.requests.append((request, timeout))
        action = np.frombuffer(request.actions, dtype=np.float32).reshape(
            request.num_envs, request.chunk_size, request.action_dim
        )[0, 0]
        observation = infer.robot_env_pb2.Observation(
            states=action.astype(np.float32).tobytes(),
            state_shape=[action.size],
            main_image=b"",
            img_height=0,
            img_width=0,
            is_compressed=False,
        )
        step_result = infer.robot_env_pb2.StepResult(
            observation=observation,
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        return infer.robot_env_pb2.ChunkStepResponse(step_results=[step_result])


def _timed_observation(timestep: int, values: list[float], must_go: bool = False):
    return infer.TimedObservation(
        timestamp=float(timestep),
        timestep=timestep,
        obs={"states": torch.tensor([values], dtype=torch.float32)},
        must_go=must_go,
    )


def test_build_arg_parser_defaults_match_local_follower_workflow():
    args = infer.build_arg_parser().parse_args([])

    assert args.config_name == "pi05_yam_follower"
    assert args.show_state_chunk is False
    assert args.return_home is True
    assert args.return_home_steps == 50
    assert args.grpc_timeout == 120.0
    assert args.max_episode_steps == 10000


def test_close_inference_session_resets_then_enters_zero_torque():
    stub = _RecordingStub()
    channel = _RecordingChannel()

    infer.close_inference_session(
        stub=stub,
        channel=channel,
        grpc_timeout=12.5,
        reset_on_exit=True,
        zero_torque_on_exit=True,
    )

    assert stub.calls == [
        ("reset", 12.5, "ResetRequest"),
        ("zero_torque", 12.5, "Empty"),
    ]
    assert channel.closed is True


def test_return_home_uses_single_step_chunk_requests():
    stub = _ReturnHomeStub()
    current_state = np.array([0.0, 0.2, -0.2], dtype=np.float32)
    home_state = np.array([0.3, -0.1, 0.5], dtype=np.float32)

    returned_state = infer.return_home(
        stub=stub,
        current_state=current_state,
        home_state=home_state,
        steps=4,
        grpc_timeout=3.0,
    )

    assert len(stub.requests) == 4
    assert all(request.chunk_size == 1 for request, _ in stub.requests)
    assert all(request.num_envs == 1 for request, _ in stub.requests)
    assert np.allclose(returned_state, home_state)


def test_should_enqueue_observation_respects_predicted_and_similarity_filters():
    previous = _timed_observation(1, [0.0, 0.0, 0.0])

    assert (
        infer.should_enqueue_observation(
            observation=_timed_observation(2, [0.0, 0.0, 0.0], must_go=True),
            last_processed_observation=previous,
            predicted_timesteps=set(),
        )
        is True
    )
    assert (
        infer.should_enqueue_observation(
            observation=_timed_observation(2, [2.0, 0.0, 0.0]),
            last_processed_observation=previous,
            predicted_timesteps={2},
        )
        is False
    )
    assert (
        infer.should_enqueue_observation(
            observation=_timed_observation(3, [0.1, 0.1, 0.1]),
            last_processed_observation=previous,
            predicted_timesteps=set(),
        )
        is False
    )
    assert (
        infer.should_enqueue_observation(
            observation=_timed_observation(4, [2.0, 0.0, 0.0]),
            last_processed_observation=previous,
            predicted_timesteps=set(),
        )
        is True
    )
