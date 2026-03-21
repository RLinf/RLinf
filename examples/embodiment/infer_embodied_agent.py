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

"""Local inference for embodied policy on a real robot via RobotServer.

No Ray, no Beaker — runs entirely on the local machine (or any machine that
can reach the RobotServer gRPC port).

Prerequisites:
    1. RobotServer running (real or dummy):
         bash scripts/start_robot_server.sh --config .../yam_pi05_follower.yaml \
             --use-follower-servers --no-tunnel [--dummy]
    2. Model weights accessible (auto-downloaded from HuggingFace if needed).

Usage::

    python examples/embodiment/infer_embodied_agent.py \
        --model-path thomas0829/folding_towel_pi05 \
        --task-description "fold the towel"

    # With dummy robot server (no real hardware):
    python examples/embodiment/infer_embodied_agent.py \
        --model-path thomas0829/folding_towel_pi05 \
        --server-url localhost:50051 \
        --task-description "fold the towel" \
        --max-episodes 3

"""

import argparse
import copy
import os
import select
import sys
import termios
import threading
import time
import tty
from collections import OrderedDict
from dataclasses import dataclass
from queue import Empty, Queue

import grpc
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.remote.proto import robot_env_pb2, robot_env_pb2_grpc
from rlinf.utils.logging import get_logger

logger = get_logger()


AGGREGATE_FUNCTIONS = {
    "weighted_average": lambda old, new: 0.3 * old + 0.7 * new,
    "latest_only": lambda old, new: new,
    "average": lambda old, new: 0.5 * old + 0.5 * new,
    "conservative": lambda old, new: 0.7 * old + 0.3 * new,
}


class EpisodeStopController:
    """Capture a single-key episode stop request from the local terminal."""

    def __init__(self, stop_key: str = "s"):
        self._stop_key = stop_key.lower()
        self._fd: int | None = None
        self._enabled = False
        self._original_terminal_settings = None
        self._stop_requested = False

    def __enter__(self):
        if not sys.stdin.isatty():
            return self

        try:
            self._fd = sys.stdin.fileno()
            self._original_terminal_settings = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            self._enabled = True
        except (OSError, ValueError, termios.error):
            self._fd = None
            self._enabled = False
            self._original_terminal_settings = None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if (
            self._enabled
            and self._fd is not None
            and self._original_terminal_settings is not None
        ):
            try:
                termios.tcsetattr(
                    self._fd,
                    termios.TCSADRAIN,
                    self._original_terminal_settings,
                )
            except (OSError, ValueError, termios.error):
                pass

    def poll_stop_request(self) -> bool:
        """Return True once the configured stop key is pressed."""
        if self._stop_requested or not self._enabled or self._fd is None:
            return self._stop_requested

        try:
            readable, _, _ = select.select([self._fd], [], [], 0.0)
        except (OSError, ValueError):
            return False
        if not readable:
            return False

        try:
            chars = os.read(self._fd, 1024).decode(errors="ignore")
        except OSError:
            return False

        if self._stop_key in chars.lower():
            self._stop_requested = True
        return self._stop_requested


@dataclass
class TimedObservation:
    """Observation annotated with the async scheduling metadata."""

    timestamp: float
    timestep: int
    obs: dict
    must_go: bool = False


def get_aggregate_function(name: str):
    """Return the named aggregate function used by async chunk inference."""
    if name not in AGGREGATE_FUNCTIONS:
        available = ", ".join(sorted(AGGREGATE_FUNCTIONS))
        raise ValueError(f"Unknown aggregate function '{name}'. Available: {available}")
    return AGGREGATE_FUNCTIONS[name]


def get_openpi_runtime_overrides(
    config_name: str,
    action_dim: int,
    action_chunk: int,
    num_steps: int,
) -> dict:
    """Build runtime overrides for OpenPI inference presets."""
    overrides = {
        "config_name": config_name,
        "num_images_in_input": 1,
        "noise_method": "flow_sde",
        "action_horizon": 50,
        "action_chunk": action_chunk,
        "num_steps": num_steps,
        "train_expert_only": True,
        "action_env_dim": action_dim,
        "add_value_head": False,
        "value_after_vlm": False,
    }

    if config_name == "pi05_yam_follower":
        overrides.update(
            {
                "num_images_in_input": 3,
                "discrete_state_input": True,
            }
        )

    return overrides


def connect_to_server(server_url: str, timeout: float = 30.0):
    """Connect to RobotServer and return (stub, spaces)."""
    max_msg = 16 * 1024 * 1024
    channel = grpc.insecure_channel(
        server_url,
        options=[
            ("grpc.max_send_message_length", max_msg),
            ("grpc.max_receive_message_length", max_msg),
        ],
    )
    stub = robot_env_pb2_grpc.RobotEnvServiceStub(channel)

    deadline = time.monotonic() + timeout
    while True:
        try:
            spaces = stub.GetSpaces(robot_env_pb2.Empty(), timeout=30.0)
            break
        except grpc.RpcError as e:
            if e.code() not in (
                grpc.StatusCode.UNAVAILABLE,
                grpc.StatusCode.DEADLINE_EXCEEDED,
            ):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"Could not connect to RobotServer at {server_url} "
                    f"within {timeout}s"
                )
            logger.info(
                f"Server not ready at {server_url}, retrying... "
                f"({remaining:.0f}s remaining)"
            )
            time.sleep(min(3.0, remaining))

    logger.info(
        f"Connected to RobotServer at {server_url}: "
        f"state_dim={spaces.state_dim}, action_dim={spaces.action_dim}, "
        f"img=({spaces.img_height},{spaces.img_width},{spaces.img_channels})"
    )
    return stub, spaces, channel


def decode_image(data: bytes, h: int, w: int, is_compressed: bool) -> np.ndarray:
    """Decode image bytes to HWC uint8 numpy array."""
    if is_compressed:
        import cv2

        buf = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        img = img[..., ::-1].copy()  # BGR -> RGB
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        return img
    return np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)


def proto_to_obs(proto_obs, h: int, w: int) -> dict:
    """Convert protobuf Observation to dict with tensors."""
    states = np.frombuffer(proto_obs.states, dtype=np.float32).reshape(
        tuple(proto_obs.state_shape)
    )
    img = decode_image(proto_obs.main_image, h, w, proto_obs.is_compressed)
    img = img[np.newaxis, :]  # (H,W,3) -> (1,H,W,3)

    obs = {
        "states": torch.from_numpy(states.copy()).float(),
        # Keep images as uint8 to match the native PI0.5/OpenPI inference path.
        # The downstream image parser interprets floating-point images as already
        # normalized to [0, 1] and rescales them by 255, which corrupts inputs
        # if we pass float images in the raw 0..255 range.
        "main_images": torch.from_numpy(img.copy()),
    }

    # Wrist images
    if proto_obs.wrist_images:
        wrist_imgs = [
            decode_image(wi, h, w, proto_obs.is_compressed)
            for wi in proto_obs.wrist_images
        ]
        obs["wrist_images"] = torch.from_numpy(
            np.stack(wrist_imgs, axis=0)[np.newaxis, :].copy()
        )

    # Extra view images
    if proto_obs.extra_view_images:
        extra_imgs = [
            decode_image(ei, h, w, proto_obs.is_compressed)
            for ei in proto_obs.extra_view_images
        ]
        obs["extra_view_images"] = torch.from_numpy(
            np.stack(extra_imgs, axis=0)[np.newaxis, :].copy()
        )

    return obs


def load_model(
    model_path: str,
    config_name: str,
    action_dim: int,
    action_chunk: int,
    num_steps: int,
):
    """Load the OpenPI model with transforms."""
    from rlinf.models.embodiment.openpi import get_model

    openpi_overrides = get_openpi_runtime_overrides(
        config_name=config_name,
        action_dim=action_dim,
        action_chunk=action_chunk,
        num_steps=num_steps,
    )
    cfg = OmegaConf.create(
        {
            "model_path": model_path,
            "model_type": "openpi",
            "add_value_head": False,
            "num_action_chunks": 10,
            "action_dim": action_dim,
            "num_steps": num_steps,
            "openpi": openpi_overrides,
        }
    )

    logger.info(f"Loading model from {model_path} ...")
    logger.info(f"OpenPI runtime overrides: {openpi_overrides}")
    model = get_model(cfg)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    logger.info("Model loaded.")
    return model


def reset_robot_on_exit(stub, grpc_timeout: float) -> bool:
    """Reset the robot to the environment's initial pose before exiting."""
    logger.info("Resetting robot to its initial pose before exit...")
    try:
        stub.Reset(robot_env_pb2.ResetRequest(), timeout=grpc_timeout)
    except grpc.RpcError as error:
        logger.warning(f"Failed to reset robot on exit: {error}")
        return False

    logger.info("Robot reset completed.")
    return True


def enter_zero_torque_mode_on_exit(stub, grpc_timeout: float) -> bool:
    """Switch the robot to zero-torque / zero-gravity mode before exiting."""
    logger.info("Switching robot to zero-torque mode before exit...")
    try:
        stub.EnterZeroTorqueMode(robot_env_pb2.Empty(), timeout=grpc_timeout)
    except grpc.RpcError as error:
        logger.warning(f"Failed to enter zero-torque mode on exit: {error}")
        return False

    logger.info("Robot is now in zero-torque mode.")
    return True


def close_inference_session(
    stub,
    channel,
    grpc_timeout: float,
    reset_on_exit: bool = True,
    zero_torque_on_exit: bool = False,
) -> None:
    """Clean up the inference session, optionally resetting the robot."""
    if reset_on_exit:
        reset_robot_on_exit(stub, grpc_timeout)
    if zero_torque_on_exit:
        enter_zero_torque_mode_on_exit(stub, grpc_timeout)
    channel.close()


def wait_for_enter(prompt: str) -> None:
    """Wait for Enter, flushing any stale terminal input first."""
    if sys.stdin.isatty():
        try:
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (OSError, ValueError, termios.error):
            pass
    input(prompt)


def preview_chunk_and_wait_for_enter(
    state: np.ndarray, actions: np.ndarray, chunk_size: int
) -> None:
    """Show the current state and full first chunk, then wait for Enter."""
    logger.info(f"  Preview state before auto-run: {state.flatten()}")
    with np.printoptions(precision=6, suppress=True):
        logger.info(
            f"  Preview chunk actions before auto-run:\n{actions[0, :chunk_size, :]}"
        )
    wait_for_enter("Press Enter to start auto-run...")


def clone_obs(obs: dict) -> dict:
    """Clone observation tensors so a predictor thread can use them safely."""
    cloned = {}
    for key, value in obs.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def flatten_state(obs: dict) -> np.ndarray:
    """Return a flat float32 state vector from a decoded observation."""
    states = obs["states"]
    if torch.is_tensor(states):
        states = states.detach().cpu().numpy()
    return np.asarray(states, dtype=np.float32).reshape(-1)


def observations_similar(
    current_observation: TimedObservation,
    previous_observation: TimedObservation,
    atol: float = 1.0,
) -> bool:
    """Match the native async server's cheap joint-space similarity filter."""
    current_state = flatten_state(current_observation.obs)
    previous_state = flatten_state(previous_observation.obs)
    return bool(np.linalg.norm(current_state - previous_state) < atol)


def should_enqueue_observation(
    observation: TimedObservation,
    last_processed_observation: TimedObservation | None,
    predicted_timesteps: set[int],
    atol: float = 1.0,
) -> bool:
    """Mirror the native async policy server's enqueue checks."""
    if observation.must_go or last_processed_observation is None:
        return True
    if observation.timestep in predicted_timesteps:
        return False
    if observations_similar(observation, last_processed_observation, atol=atol):
        return False
    return True


def build_env_obs(obs: dict, task_description: str, device: torch.device) -> dict:
    """Build the model input dict from a decoded RobotServer observation."""
    env_obs = {
        "main_images": obs["main_images"],
        "states": obs["states"],
        "task_descriptions": [task_description],
    }
    if "wrist_images" in obs:
        env_obs["wrist_images"] = obs["wrist_images"]
    if "extra_view_images" in obs:
        env_obs["extra_view_images"] = obs["extra_view_images"]

    for key, value in env_obs.items():
        if torch.is_tensor(value):
            env_obs[key] = value.to(device)
    return env_obs


@torch.no_grad()
def predict_action_chunk(model, obs: dict, task_description: str) -> np.ndarray:
    """Run one local model inference and return a numpy action chunk."""
    device = next(model.parameters()).device
    env_obs = build_env_obs(obs, task_description, device)
    actions, _ = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
    return actions.detach().cpu().numpy()


def log_predicted_chunk(
    prediction_idx: int,
    start_timestep: int,
    actions: np.ndarray,
    chunk_size: int,
) -> None:
    """Log the summary of a predicted chunk."""
    logger.info(
        f"  [Predict {prediction_idx}] Chunk for timestep {start_timestep} "
        f"with {chunk_size} actions. "
        f"First: {actions[0, 0]}, Last: {actions[0, chunk_size - 1]}"
    )


def aggregate_action_queue(
    action_queue: OrderedDict[int, np.ndarray],
    latest_action_timestep: int,
    start_timestep: int,
    actions: np.ndarray,
    chunk_size: int,
    aggregate_fn,
) -> OrderedDict[int, np.ndarray]:
    """Merge a newly predicted chunk into the local action queue."""
    for offset in range(chunk_size):
        timestep = start_timestep + offset
        if timestep <= latest_action_timestep:
            continue

        action = np.asarray(actions[0, offset], dtype=np.float32)
        if timestep in action_queue:
            action_queue[timestep] = np.asarray(
                aggregate_fn(action_queue[timestep], action), dtype=np.float32
            )
        else:
            action_queue[timestep] = action.copy()

    return OrderedDict(sorted(action_queue.items(), key=lambda item: item[0]))


def enqueue_prediction_request(
    request_queue: Queue,
    observation: TimedObservation,
) -> None:
    """Keep only the freshest pending prediction request."""
    try:
        while True:
            request_queue.get_nowait()
    except Empty:
        pass

    queued_observation = TimedObservation(
        timestamp=observation.timestamp,
        timestep=observation.timestep,
        obs=clone_obs(observation.obs),
        must_go=observation.must_go,
    )
    request_queue.put_nowait(queued_observation)


def predictor_worker(
    model,
    task_description: str,
    action_chunk: int,
    request_queue: Queue,
    result_queue: Queue,
    error_queue: Queue,
    stop_event: threading.Event,
) -> None:
    """Background predictor that mimics the original async policy server."""
    while not stop_event.is_set():
        try:
            observation = request_queue.get(timeout=0.1)
        except Empty:
            continue

        try:
            actions = predict_action_chunk(model, observation.obs, task_description)
            chunk_size = min(action_chunk, actions.shape[1])

            try:
                while True:
                    result_queue.get_nowait()
            except Empty:
                pass

            result_queue.put_nowait((observation, actions, chunk_size))
        except Exception as error:  # pragma: no cover - surfaced to main thread
            error_queue.put_nowait(error)
            stop_event.set()
            return


def execute_single_action(
    stub,
    action: np.ndarray,
    grpc_timeout: float,
):
    """Execute exactly one action using ChunkStep with chunk_size=1."""
    chunk_actions = np.asarray(action, dtype=np.float32)[np.newaxis, np.newaxis, :]
    return stub.ChunkStep(
        robot_env_pb2.ChunkStepRequest(
            actions=chunk_actions.tobytes(),
            num_envs=1,
            chunk_size=1,
            action_dim=chunk_actions.shape[2],
        ),
        timeout=grpc_timeout,
    )


def return_home(
    stub,
    current_state: np.ndarray,
    home_state: np.ndarray,
    steps: int,
    grpc_timeout: float,
) -> np.ndarray | None:
    """Interpolate back to the captured home pose like the original follower runtime."""
    if steps < 1:
        return None

    if np.allclose(current_state, home_state):
        logger.info("  Already at home pose; skipping return_home.")
        return current_state.copy()

    logger.info(f"  Returning home over {steps} interpolated steps...")
    interpolation = np.stack(
        [
            (1.0 - alpha) * current_state + alpha * home_state
            for alpha in np.linspace(1.0 / steps, 1.0, steps, dtype=np.float32)
        ],
        axis=0,
    )

    last_state: np.ndarray | None = current_state.copy()
    for action in interpolation:
        # Avoid a giant ChunkStep response here: with raw multi-view images each
        # interpolated step carries a full observation, and return_home is not on
        # the latency-critical policy path.
        resp = execute_single_action(stub, action, grpc_timeout)
        if not resp.step_results:
            continue
        last_obs = resp.step_results[-1].observation
        last_state = (
            np.frombuffer(last_obs.states, dtype=np.float32)
            .reshape(tuple(last_obs.state_shape))
            .flatten()
        )

    return last_state


@torch.no_grad()
def run_episode(
    model,
    stub,
    spaces,
    task_description: str,
    action_chunk: int,
    max_steps: int,
    grpc_timeout: float,
    chunk_size_threshold: float,
    aggregate_fn_name: str,
    show_state_chunk: bool,
    return_home_after_episode: bool,
    return_home_steps: int,
    stop_controller_cls=EpisodeStopController,
):
    """Run one episode and return (total_reward, num_steps, stopped_by_user)."""
    h, w = spaces.img_height, spaces.img_width
    aggregate_fn = get_aggregate_function(aggregate_fn_name)

    # Set task description on server
    stub.SetTaskDescription(
        robot_env_pb2.TaskDescriptionRequest(task_description=task_description),
        timeout=grpc_timeout,
    )

    # Reset
    proto_obs = stub.Reset(robot_env_pb2.ResetRequest(), timeout=grpc_timeout)
    obs = proto_to_obs(proto_obs, h, w)
    home_state = obs["states"].numpy().flatten().copy()

    action_queue: OrderedDict[int, np.ndarray] = OrderedDict()
    action_chunk_size = max(action_chunk, 1)
    prediction_count = 0
    latest_action_timestep = -1
    predicted_timesteps: set[int] = set()
    last_processed_observation: TimedObservation | None = None
    last_requested_observation: TimedObservation | None = None
    must_go = True
    preview_done = not show_state_chunk

    request_queue: Queue = Queue(maxsize=1)
    result_queue: Queue = Queue(maxsize=1)
    error_queue: Queue = Queue(maxsize=1)
    stop_event = threading.Event()
    predictor_thread = threading.Thread(
        target=predictor_worker,
        args=(
            model,
            task_description,
            action_chunk,
            request_queue,
            result_queue,
            error_queue,
            stop_event,
        ),
        daemon=True,
    )
    predictor_thread.start()

    total_reward = 0.0
    steps_taken = 0
    stopped_by_user = False

    def maybe_enqueue_prediction(current_obs: dict) -> bool:
        nonlocal must_go
        nonlocal last_requested_observation

        observation = TimedObservation(
            timestamp=time.time(),
            timestep=max(latest_action_timestep, 0),
            obs=current_obs,
            must_go=must_go and not action_queue,
        )
        if (
            last_requested_observation is not None
            and observation.timestep == last_requested_observation.timestep
            and observations_similar(observation, last_requested_observation)
        ):
            return False
        if not should_enqueue_observation(
            observation=observation,
            last_processed_observation=last_processed_observation,
            predicted_timesteps=predicted_timesteps,
        ):
            return False

        enqueue_prediction_request(
            request_queue=request_queue,
            observation=observation,
        )
        last_requested_observation = observation
        if observation.must_go:
            must_go = False
        return True

    def drain_prediction_results() -> None:
        nonlocal action_queue
        nonlocal action_chunk_size
        nonlocal prediction_count
        nonlocal last_processed_observation
        nonlocal must_go
        nonlocal preview_done
        nonlocal last_requested_observation

        try:
            error = error_queue.get_nowait()
        except Empty:
            error = None
        if error is not None:
            raise RuntimeError(f"Background prediction failed: {error}") from error

        while True:
            try:
                observation, actions, chunk_size = result_queue.get_nowait()
            except Empty:
                break

            prediction_count += 1
            predicted_timesteps.add(observation.timestep)
            last_processed_observation = observation
            if (
                last_requested_observation is not None
                and last_requested_observation.timestep == observation.timestep
            ):
                last_requested_observation = None
            if show_state_chunk:
                if not preview_done:
                    logger.info(f"  Current state: {flatten_state(observation.obs)}")
                log_predicted_chunk(
                    prediction_idx=prediction_count,
                    start_timestep=observation.timestep,
                    actions=actions,
                    chunk_size=chunk_size,
                )
                if not preview_done:
                    preview_chunk_and_wait_for_enter(
                        observation.obs["states"].numpy(), actions, chunk_size
                    )
                    preview_done = True
            action_queue = aggregate_action_queue(
                action_queue=action_queue,
                latest_action_timestep=latest_action_timestep,
                start_timestep=observation.timestep,
                actions=actions,
                chunk_size=chunk_size,
                aggregate_fn=aggregate_fn,
            )
            action_chunk_size = max(action_chunk_size, chunk_size)
            must_go = True

    try:
        with stop_controller_cls() as stop_controller:
            for _ in range(max_steps):
                if stop_controller.poll_stop_request():
                    stopped_by_user = True
                    logger.info("  Stopping current episode and returning home...")
                    break

                drain_prediction_results()

                if not action_queue:
                    maybe_enqueue_prediction(obs)

                while not action_queue:
                    if stop_controller.poll_stop_request():
                        stopped_by_user = True
                        logger.info("  Stopping current episode and returning home...")
                        break
                    drain_prediction_results()
                    maybe_enqueue_prediction(obs)
                    time.sleep(0.001)

                if stopped_by_user:
                    break

                timestep, action = action_queue.popitem(last=False)
                resp = execute_single_action(stub, action, grpc_timeout)
                latest_action_timestep = timestep
                steps_taken += 1

                last_result = resp.step_results[-1]
                obs = proto_to_obs(last_result.observation, h, w)
                total_reward += last_result.reward

                queue_ratio = (
                    len(action_queue) / action_chunk_size
                    if action_chunk_size > 0
                    else 0.0
                )
                if queue_ratio <= chunk_size_threshold:
                    maybe_enqueue_prediction(obs)

                if last_result.terminated or last_result.truncated:
                    logger.info(
                        f"  Episode ended at step {steps_taken}: "
                        f"terminated={last_result.terminated}, "
                        f"truncated={last_result.truncated}"
                    )
                    break
            else:
                logger.info(f"  Episode reached max_steps={max_steps}")
    finally:
        stop_event.set()
        predictor_thread.join(timeout=2)

    if return_home_after_episode:
        current_state = obs["states"].numpy().flatten()
        return_home(
            stub=stub,
            current_state=current_state,
            home_state=home_state,
            steps=return_home_steps,
            grpc_timeout=grpc_timeout,
        )

    return total_reward, steps_taken, stopped_by_user


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for local inference."""
    parser = argparse.ArgumentParser(
        description="Local inference for embodied policy via RobotServer"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="thomas0829/folding_towel_pi05",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="pi05_yam_follower",
        help="OpenPI config name (pi0_libero, pi05_libero, pi05_yam_follower, etc.)",
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="localhost:50051",
        help="RobotServer gRPC address",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default="bimanual manipulation",
        help="Task description prompt for the policy",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=14,
        help="Action dimension (14 for YAM bimanual: 2x7 DOF)",
    )
    parser.add_argument(
        "--action-chunk",
        type=int,
        default=30,
        help="Number of actions to execute per inference call",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=10,
        help="Number of denoising steps",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=10000,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--chunk-size-threshold",
        type=float,
        default=0.0,
        help="When local action_queue_size / action_chunk <= threshold, request a new chunk",
    )
    parser.add_argument(
        "--aggregate-fn-name",
        type=str,
        default="weighted_average",
        choices=sorted(AGGREGATE_FUNCTIONS),
        help="How to merge overlapping future actions in async chunk execution",
    )
    parser.add_argument(
        "--show-state-chunk",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Show current state / predicted chunk previews and wait for Enter "
            "before auto-run starts"
        ),
    )
    parser.add_argument(
        "--return-home",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpolate back to the captured home pose after each episode",
    )
    parser.add_argument(
        "--return-home-steps",
        type=int,
        default=50,
        help="Interpolation steps used when returning home after an episode",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Max episodes to run (0 = run forever until Ctrl+C)",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for RobotServer connection",
    )
    parser.add_argument(
        "--grpc-timeout",
        type=float,
        default=120.0,
        help="Seconds timeout per gRPC call",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    # Connect to robot server
    stub, spaces, channel = connect_to_server(args.server_url, args.connect_timeout)

    # Load model
    model = load_model(
        args.model_path,
        args.config_name,
        args.action_dim,
        args.action_chunk,
        args.num_steps,
    )
    if sys.stdin.isatty():
        logger.info(
            "Press 's' during auto-run to stop the current episode, return home, "
            "then press Enter to start the next episode."
        )

    # Run episodes
    episode = 0
    total_rewards = []
    interrupted_by_user = False
    try:
        while True:
            episode += 1
            if args.max_episodes > 0 and episode > args.max_episodes:
                break

            logger.info(f"=== Episode {episode} ===")
            reward, steps, stopped_by_user = run_episode(
                model=model,
                stub=stub,
                spaces=spaces,
                task_description=args.task_description,
                action_chunk=args.action_chunk,
                max_steps=args.max_episode_steps,
                grpc_timeout=args.grpc_timeout,
                chunk_size_threshold=args.chunk_size_threshold,
                aggregate_fn_name=args.aggregate_fn_name,
                show_state_chunk=args.show_state_chunk,
                return_home_after_episode=args.return_home,
                return_home_steps=args.return_home_steps,
            )
            if stopped_by_user:
                logger.info(f"  Episode stopped by user at step {steps}.")
                wait_for_enter("Press Enter to start the next episode...")
                continue

            total_rewards.append(reward)
            logger.info(f"  Reward: {reward:.4f}, Steps: {steps}")
            logger.info(
                f"  Average reward over {len(total_rewards)} episodes: "
                f"{np.mean(total_rewards):.4f}"
            )
    except KeyboardInterrupt:
        interrupted_by_user = True
        logger.info("\nStopped by user.")
    finally:
        close_inference_session(
            stub=stub,
            channel=channel,
            grpc_timeout=args.grpc_timeout,
            zero_torque_on_exit=interrupted_by_user,
        )
        if total_rewards:
            logger.info(
                f"\nResults: {len(total_rewards)} episodes, "
                f"avg reward: {np.mean(total_rewards):.4f}"
            )


if __name__ == "__main__":
    main()
