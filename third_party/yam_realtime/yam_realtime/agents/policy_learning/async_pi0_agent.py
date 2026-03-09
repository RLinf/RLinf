import threading
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import tyro
from dm_env.specs import Array
from openpi_client import action_chunk_broker, image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime.agents import policy_agent as _policy_agent

from yam_realtime.agents.agent import PolicyAgent
from yam_realtime.agents.constants import ActionSpec
from yam_realtime.data.data_utils import recusive_flatten
from yam_realtime.learning.diffusion_policy.policy_network import ModelConfig
from yam_realtime.robots.utils import Rate
from yam_realtime.utils.portal_utils import remote


class AsyncDiffusionAgent(PolicyAgent):
    def __init__(
        self,
        use_joint_state_as_action: bool = False,
        ip: str = "0.0.0.0",
        port: int = 8111,
        action_horizon: int = 25,
        inference_interval_s: float
        | None = None,  # if not None, open a thread to run inference loop at this interval, and use a buffer to smooth the action.
        length_of_smoothed_action_buffer: int = 4,
    ) -> None:
        self.use_joint_state_as_action = use_joint_state_as_action
        print(f"ip: {ip}")
        self._websocket_client_policy = _websocket_client_policy.WebsocketClientPolicy(
            host=ip,
            port=port,
        )

        self.action_horizon = action_horizon
        self.inference_interval_s = inference_interval_s
        self.length_of_smoothed_action_buffer = length_of_smoothed_action_buffer
        self.inference_interval_rate = (
            Rate(1 / inference_interval_s, rate_name="inference_interval")
            if inference_interval_s is not None
            else None
        )
        self.config = ModelConfig(
            action_keys=(
                "action-left-pos",
                "action-right-pos",
                "action-left-vel",
                "action-right-vel",
            ),
            mlp_keys=(
                "left-joint_pos",
                "left-gripper_pos",
                "right-joint_pos",
                "right-gripper_pos",
            ),
            image_keys=(
                "left_camera-images-rgb",
                "right_camera-images-rgb",
                "top_camera-images-rgb",
            ),
        )
        self.action_lock = threading.Lock()
        self.last_actions = None
        self.obs_lock = threading.Lock()
        self._obs = None
        self.action_counter = 0
        if self.inference_interval_s is not None:
            self.action_thread = threading.Thread(target=self._action_loop)
            self.action_thread.start()
        else:
            self._agent = _policy_agent.PolicyAgent(
                policy=action_chunk_broker.ActionChunkBroker(
                    policy=self._websocket_client_policy,
                    action_horizon=self.action_horizon,
                )
            )

    @remote()
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "action_horizon": self.action_horizon,
            "inference_interval_s": self.inference_interval_s,
            "length_of_smoothed_action_buffer": self.length_of_smoothed_action_buffer,
            **self._websocket_client_policy.get_server_metadata(),
        }

    def obs_to_model_input(self, obs):
        flat_obs = []
        obs = recusive_flatten(obs)
        for k in self.config.mlp_keys:
            flat_obs.append(obs[k])
        flat_obs = np.concatenate(flat_obs, axis=-1)

        images = {}
        for k in self.config.image_keys:
            img = obs[k]
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, 224, 224))
            img = np.transpose(img, (2, 0, 1))
            images[k] = img

        return {
            "state": flat_obs,
            **images,
        }

    @remote()
    def act(self, obs):
        super_action = super().act(obs)
        a = np.array(self(obs))
        if self.use_joint_state_as_action:
            assert a.shape == (28,)
            left = a[:14]
            right = a[14:]
            left[6] = np.clip(left[6], 0, 1)
            right[6] = np.clip(right[6], 0, 1)
            return {
                "left": {"pos": left[:7], "vel": left[7:]},
                "right": {"pos": right[:7], "vel": right[7:]},
                **super_action,
            }
        else:
            assert a.shape == (14,)
            left = a[:7]
            right = a[7:]
            left[-1] = np.clip(left[-1], 0, 1)
            right[-1] = np.clip(right[-1], 0, 1)
            return {
                "left": {"pos": left},
                "right": {"pos": right},
                **super_action,
            }

    @remote(serialization_needed=True)
    def action_spec(self) -> ActionSpec:
        """Define the action specification."""
        if self.use_joint_state_as_action:
            return {
                "left": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
                "right": {
                    "pos": Array(shape=(7,), dtype=np.float32),
                    "vel": Array(shape=(7,), dtype=np.float32),
                },
            }
        else:
            return {
                "left": {"pos": Array(shape=(7,), dtype=np.float32)},
                "right": {"pos": Array(shape=(7,), dtype=np.float32)},
            }

    def _action_loop(self) -> None:
        while True:
            while self._obs is None:
                time.sleep(0.1)

            with self.obs_lock:
                current_obs = self._obs
            with self.action_lock:
                start_inference_action_counter = self.action_counter
            t0 = time.time()
            inferred_action = self._websocket_client_policy.infer(current_obs)["actions"]
            t1 = time.time()
            with self.action_lock:
                complete_inference_action_counter = self.action_counter

                current_action_counter = max(0, complete_inference_action_counter - start_inference_action_counter)
                new_action = inferred_action[current_action_counter : current_action_counter + self.action_horizon, :]

                if self.last_actions is None:
                    self.last_actions = new_action
                    continue

                remaining_actions = self.last_actions[self.action_counter :]
                num_smoothed_actions = min(self.length_of_smoothed_action_buffer, remaining_actions.shape[0])

                weights = np.linspace(1 / num_smoothed_actions, 1, num_smoothed_actions).reshape(-1, 1)

                smoothed_actions = (
                    weights * new_action[:num_smoothed_actions]
                    + (1 - weights) * remaining_actions[:num_smoothed_actions]
                )
                self.last_actions = np.concatenate([smoothed_actions, new_action[num_smoothed_actions:]], axis=0)
                self.action_counter = 0
            assert self.inference_interval_rate is not None
            self.inference_interval_rate.sleep()

    def select_action(self):
        while self.last_actions is None:
            time.sleep(0.1)
        # only compute every self.action_horizon steps
        with self.action_lock:
            action = self.last_actions[self.action_counter]
            if self.action_counter == self.action_horizon - 1:
                print(f"Inference is slower than expected, Repeating action at action_counter: {self.action_counter}")
            else:
                self.action_counter += 1
        return action

    def __call__(self, obs):
        with self.obs_lock:
            self._obs = self.obs_to_model_input(obs)
        if self.inference_interval_s is not None:
            action = self.select_action()
        else:
            action = self._agent.get_action(self._obs)["actions"]
        return action

