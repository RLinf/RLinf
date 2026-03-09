"""RemoteEnv — a gym.Env that proxies to a remote RobotServer over gRPC.

This environment presents the same interface as ``YAMEnv`` but forwards all
calls to a gRPC server (typically running on the robot's local machine and
exposed via a cloudflared tunnel).

Usage in YAML config::

    env_type: remote
    remote_server_url: "https://<tunnel-id>.trycloudflare.com"
"""

from typing import Optional

import gymnasium as gym
import grpc
import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.envs.remote.proto import robot_env_pb2, robot_env_pb2_grpc
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger

_DEFAULT_MAX_MESSAGE_SIZE = 16 * 1024 * 1024


def _decompress_image(data: bytes, height: int, width: int) -> np.ndarray:
    """Decode JPEG bytes to uint8 HWC numpy array."""
    import cv2

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    img = img[..., ::-1].copy()  # BGR → RGB
    if img.shape[:2] != (height, width):
        img = cv2.resize(img, (width, height))
    return img


def _proto_to_obs(proto_obs: robot_env_pb2.Observation) -> dict:
    """Convert a protobuf Observation to a YAMEnv-compatible dict."""
    # States
    state_shape = tuple(proto_obs.state_shape)
    states = np.frombuffer(proto_obs.states, dtype=np.float32).reshape(state_shape)

    # Image
    h, w = proto_obs.img_height, proto_obs.img_width
    if proto_obs.is_compressed:
        img = _decompress_image(proto_obs.main_image, h, w)
    else:
        img = np.frombuffer(proto_obs.main_image, dtype=np.uint8).reshape(h, w, 3)

    # Add batch dim: (H,W,3) → (1,H,W,3)
    img = img[np.newaxis, :]

    obs = {
        "states": to_tensor(states),
        "main_images": to_tensor(img),
        "task_descriptions": [proto_obs.task_description],
    }
    return obs


class RemoteEnv(gym.Env):
    """Gymnasium environment that proxies to a remote RobotServer over gRPC.

    Constructor signature matches ``YAMEnv`` so it can be used as a drop-in
    replacement via ``env_type: remote`` in YAML configs.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg: DictConfig,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info: Optional[WorkerInfo],
    ):
        assert num_envs == 1, (
            f"RemoteEnv supports exactly 1 environment per worker, got {num_envs}."
        )
        self._logger = get_logger()
        self.cfg = cfg
        self.num_envs = num_envs
        self.worker_info = worker_info

        # gRPC connection settings
        server_url = str(cfg.get("remote_server_url", "localhost:50051"))
        max_msg = int(cfg.get("grpc_max_message_size", _DEFAULT_MAX_MESSAGE_SIZE))
        self._timeout = float(cfg.get("grpc_timeout", 30.0))

        self._logger.info(f"[RemoteEnv] Connecting to server at {server_url}")
        channel_options = [
            ("grpc.max_send_message_length", max_msg),
            ("grpc.max_receive_message_length", max_msg),
        ]

        # Use secure channel for HTTPS URLs (e.g. cloudflared quick tunnels),
        # insecure channel for plain host:port addresses.
        if server_url.startswith("https://"):
            # Strip scheme — gRPC wants "host:port", not a URL.
            target = server_url.removeprefix("https://")
            if ":" not in target:
                target += ":443"
            credentials = grpc.ssl_channel_credentials()
            self._channel = grpc.secure_channel(
                target, credentials, options=channel_options
            )
        else:
            target = server_url.removeprefix("http://")
            self._channel = grpc.insecure_channel(
                target, options=channel_options
            )

        self._stub = robot_env_pb2_grpc.RobotEnvServiceStub(self._channel)

        # Fetch space metadata from server
        spaces = self._stub.GetSpaces(
            robot_env_pb2.Empty(), timeout=self._timeout
        )

        self._state_dim = spaces.state_dim
        self._action_dim = spaces.action_dim
        self._img_h = spaces.img_height
        self._img_w = spaces.img_width
        self._img_c = spaces.img_channels
        self._max_episode_steps = spaces.max_episode_steps
        self._control_rate_hz = spaces.control_rate_hz
        self.auto_reset = bool(cfg.get("auto_reset", spaces.auto_reset))
        self.ignore_terminations = bool(
            cfg.get("ignore_terminations", spaces.ignore_terminations)
        )

        # Gym spaces
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._state_dim,),
                    dtype=np.float32,
                ),
                "main_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(self._img_h, self._img_w, self._img_c),
                    dtype=np.uint8,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=spaces.action_low,
            high=spaces.action_high,
            shape=(self._action_dim,),
            dtype=np.float32,
        )

        # State tracking (mirrors YAMEnv)
        self._task_description: str = str(cfg.get("task_description", ""))
        self._is_start = True
        self._num_steps = 0
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        # Metrics
        self._init_metrics()

        self._logger.info(
            f"[RemoteEnv] Connected. state_dim={self._state_dim}, "
            f"action_dim={self._action_dim}, img=({self._img_h},{self._img_w},{self._img_c})"
        )

    # ------------------------------------------------------------------
    # Gym interface
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None, reset_state_ids=None, env_idx=None):
        self._num_steps = 0
        self._elapsed_steps[:] = 0
        self._reset_metrics()
        self._is_start = True

        req = robot_env_pb2.ResetRequest()
        if seed is not None:
            req.seed = seed
        proto_obs = self._stub.Reset(req, timeout=self._timeout)
        obs = _proto_to_obs(proto_obs)
        return obs, {}

    def step(self, actions=None, auto_reset=True):
        """Single step — delegates to chunk_step with chunk_size=1."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        if actions is not None:
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[np.newaxis, :]  # (action_dim,) → (1, action_dim)
            # (num_envs, action_dim) → (num_envs, 1, action_dim)
            chunk_actions = actions[:, np.newaxis, :]
        else:
            chunk_actions = np.zeros(
                (self.num_envs, 1, self._action_dim), dtype=np.float32
            )

        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list = (
            self.chunk_step(chunk_actions)
        )

        obs = obs_list[0]
        reward = chunk_rewards[:, 0].numpy()
        terminated = chunk_terminations[:, 0].bool().numpy()
        truncated = chunk_truncations[:, 0].bool().numpy()
        infos = infos_list[0] if infos_list else {}

        if auto_reset and (np.any(terminated) or np.any(truncated)):
            obs, _ = self.reset()

        return obs, reward, terminated, truncated, infos

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions on the remote server.

        Parameters
        ----------
        chunk_actions : np.ndarray | torch.Tensor
            Shape ``(num_envs, chunk_size, action_dim)``.

        Returns
        -------
        obs_list, chunk_rewards, chunk_terminations, chunk_truncations, infos_list
        """
        if isinstance(chunk_actions, torch.Tensor):
            chunk_actions = chunk_actions.detach().cpu().numpy()
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        num_envs, chunk_size, action_dim = chunk_actions.shape

        req = robot_env_pb2.ChunkStepRequest(
            actions=chunk_actions.tobytes(),
            num_envs=num_envs,
            chunk_size=chunk_size,
            action_dim=action_dim,
        )
        resp = self._stub.ChunkStep(req, timeout=self._timeout * chunk_size)

        obs_list = []
        rewards = []
        terminations = []
        truncations = []
        infos_list = []

        for sr in resp.step_results:
            obs = _proto_to_obs(sr.observation)
            obs_list.append(obs)

            self._elapsed_steps += 1
            self._num_steps += 1

            step_reward = np.array([sr.reward], dtype=np.float32)
            step_term = np.array([sr.terminated], dtype=bool)
            step_trunc = np.array([sr.truncated], dtype=bool)

            infos = self._record_metrics(
                step_reward, step_term, np.zeros_like(step_term), {}
            )
            infos_list.append(infos)
            rewards.append(step_reward)
            terminations.append(step_term)
            truncations.append(step_trunc)

        chunk_rewards = torch.stack(
            [to_tensor(r) if not isinstance(r, torch.Tensor) else r for r in rewards],
            dim=1,
        )
        raw_chunk_terminations = torch.stack(
            [
                to_tensor(t) if not isinstance(t, torch.Tensor) else t
                for t in terminations
            ],
            dim=1,
        )
        raw_chunk_truncations = torch.stack(
            [
                to_tensor(t) if not isinstance(t, torch.Tensor) else t
                for t in truncations
            ],
            dim=1,
        )

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def close(self):
        self._stub.Close(robot_env_pb2.Empty(), timeout=self._timeout)
        self._channel.close()

    # ------------------------------------------------------------------
    # Metrics (mirrors YAMEnv)
    # ------------------------------------------------------------------

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)
        self.prev_step_reward = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0.0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, intervene_current_step, infos):
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self._elapsed_steps.copy(),
            "reward": np.where(
                self._elapsed_steps > 0, self.returns / self._elapsed_steps, 0.0
            ),
            "intervened_once": self.intervened_once,
            "intervened_steps": self.intervened_steps,
            "success_no_intervened": self.success_once.copy() & (~self.intervened_once),
        }
        infos["episode"] = to_tensor(episode_info)
        return infos

    # ------------------------------------------------------------------
    # Properties (mirrors YAMEnv)
    # ------------------------------------------------------------------

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool):
        self._is_start = value

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def total_num_group_envs(self) -> int:
        return np.iinfo(np.uint8).max // 2

    @property
    def task_description(self) -> str:
        return self._task_description

    @task_description.setter
    def task_description(self, value: str) -> None:
        self._task_description = str(value)
        self._stub.SetTaskDescription(
            robot_env_pb2.TaskDescriptionRequest(task_description=self._task_description),
            timeout=self._timeout,
        )

    @property
    def task_descriptions(self) -> list[str]:
        return [self._task_description]
