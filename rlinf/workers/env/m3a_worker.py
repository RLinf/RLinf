# Copyright 2025 The RLinf Authors.
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
from typing import Any

from omegaconf import DictConfig

from rlinf.data.datasets.android import AndroidWorldDataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Channel, Worker
from rlinf.workers.env.llm_wrapper import LLMWrapper


class AndroidAgentWorker(Worker):
    """Worker that runs Android World M3A agent for evaluation."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize AndroidAgentWorker with Hydra config."""
        super().__init__()
        self._cfg = cfg
        self._llm: LLMWrapper | None = None
        self._dataset: AndroidWorldDataset | None = None
        self._generate_input_channel: Channel | None = None
        self._generate_output_channel: Channel | None = None
        self._envs: dict[int, Any] = {}
        self._num_envs: int = 0
        self._next_env_idx: int = 0

    def init_with_channels(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
    ) -> None:
        """Bind rollout input/output channels used by the M3A agent."""
        self._generate_input_channel = generate_input_channel
        self._generate_output_channel = generate_output_channel

    def init_worker(self) -> None:
        """Initialize envs, dataset and LLM wrapper."""
        hardware_infos = self.hardware_infos
        self.log_info(
            f"Worker rank {self._rank} bound to {len(hardware_infos)} hardware device(s)"
        )

        if not hardware_infos:
            raise ValueError(
                f"Worker rank {self._rank} has no hardware device assigned"
            )

        # Delay imports of android_world until here so that PYTHONPATH can
        # be configured externally (e.g., by eval.sh).
        from android_world.env import env_launcher

        for idx, hw_info in enumerate(hardware_infos):
            hw_info = hw_info  # type: ignore[assignment]
            device_id = hw_info.config.device_id
            adb_path = hw_info.config.adb_path
            grpc_port = int(self._cfg.reward.get("grpc_port", 8556)) + idx
            if ":" in device_id:
                console_port = int(device_id.split(":")[1]) - 1
            else:
                console_port = int(device_id.split("-")[1]) - 1

            self.log_info(
                f"Loading and setting up env for device {device_id} on rank {self._rank}..."
            )
            env = env_launcher.load_and_setup_env(
                console_port=console_port,
                emulator_setup=False,
                freeze_datetime=True,
                adb_path=adb_path,
                grpc_port=grpc_port,
                device_id=device_id,
            )
            self.log_info(
                f"Env loaded and setup for device {device_id} on rank {self._rank}"
            )
            self._envs[idx] = env

        self._num_envs = len(self._envs)
        self._next_env_idx = 0

        # Load dataset.
        tokenizer = hf_tokenizer(self._cfg.data.tokenizer.tokenizer_model)
        self._dataset = AndroidWorldDataset(
            config=self._cfg,
            tokenizer=tokenizer,
            seed=self._cfg.data.get("seed", 42),
        )
        self.log_info(
            f"AndroidWorld dataset loaded: {len(self._dataset)} task instances"
        )

        # Load LLM wrapper.
        if (
            self._generate_input_channel is None
            or self._generate_output_channel is None
        ):
            raise RuntimeError("LLM channels must be initialized before init_worker.")
        self._llm = LLMWrapper(
            generate_input_channel=self._generate_input_channel,
            generate_output_channel=self._generate_output_channel,
        )

    def _get_env(self, device_key: int) -> Any:
        """Return env instance for the given device index."""
        return self._envs[device_key]

    def get_num_devices(self) -> int:
        """Return the number of attached devices."""
        return self._num_envs

    def get_dataset_size(self) -> int:
        """Return the number of tasks in the dataset."""
        if self._dataset is None:
            raise RuntimeError("Dataset is not initialized.")
        return len(self._dataset)

    def _get_next_env(self) -> tuple[Any, dict[str, Any]]:
        """Return the next env and its metadata for scheduling tasks."""
        if self._num_envs == 0:
            raise ValueError(f"Worker rank {self._rank} has no devices available")

        env_idx = self._next_env_idx
        env = self._envs[env_idx]
        hw_info = self.hardware_infos[self._rank]
        grpc_port = int(self._cfg.reward.get("grpc_port", 8556)) + env_idx
        if ":" in hw_info.config.device_id:
            console_port = int(hw_info.config.device_id.split(":")[1]) - 1
        else:
            console_port = int(hw_info.config.device_id.split("-")[1]) - 1
        env_info: dict[str, Any] = {
            "device_id": hw_info.config.device_id,
            "grpc_port": grpc_port,
            "console_port": console_port,
            "adb_path": hw_info.config.adb_path,
        }
        self._next_env_idx = (self._next_env_idx + 1) % self._num_envs
        return env, env_info

    def process_task(
        self,
        task_idx: int,
        reward_worker_group_name: str = "RewardWorkerGroup",
    ) -> Any:
        """Process a single AndroidWorld task.

        Args:
            task_idx: Index of the task in the dataset.
            reward_worker_group_name: Name of the reward worker group.

        Returns:
            Reward returned by the reward worker.
        """
        if self._dataset is None:
            raise RuntimeError("Dataset is not initialized.")
        if self._llm is None:
            raise RuntimeError("LLMWrapper is not initialized.")

        task_item = self._dataset[task_idx]
        task = task_item.answer["task"]
        env, env_info = self._get_next_env()
        device_id = env_info["device_id"]
        self.log_info(
            f"Processing task {task_item.answer['task_name']}, goal: {task.goal} "
            f"on device {device_id} (rank {self._rank})"
        )

        try:
            task.initialize_task(env)
        except Exception as exc:  # pylint: disable=broad-except
            self.log_error(
                f"Failed to initialize task {task_idx} on device {device_id} "
                f"(rank {self._rank}): {type(exc).__name__}: {exc}"
            )
            raise

        agent_result = self._run_agent(env, task)
        env_task_info = {
            "env_info": env_info,
            "agent_result": agent_result,
            "task": task,
        }
        self.send(
            object=env_task_info,
            dst_group_name=reward_worker_group_name,
            dst_rank=self._rank,
        )
        self.log_info(
            f"Sent env task info[{task}] to {reward_worker_group_name}[{self._rank}]"
        )

        reward = self.recv(
            src_group_name=reward_worker_group_name,
            src_rank=self._rank,
        )
        self.log_info(
            f"Received reward from {reward_worker_group_name}[{self._rank}]: {reward}"
        )

        task.tear_down(env)
        self.log_info(f"Torn down env for task {task_idx}")
        return reward

    def _run_agent(self, env: Any, task: Any) -> Any:
        """Run the M3A agent for one task and return the episode result."""
        # Local import to avoid hard dependency at module import time.
        from android_world.agents import m3a  # type: ignore[import]
        from android_world.episode_runner import (  # type: ignore[import]
            EpisodeResult,
            _transpose_lod_to_dol,
        )

        if self._llm is None:
            raise RuntimeError("LLMWrapper is not initialized.")

        agent = m3a.M3A(env, self._llm)
        max_steps = int(getattr(task, "complexity", 1) * 10)
        agent.reset(task.start_on_home_screen)
        agent.set_max_steps(max_steps)
        goal = task.goal
        if hasattr(task, "guidelines"):
            agent.set_task_guidelines(task.guidelines)

        actions_output: list[dict[str, Any]] = []
        for step_n in range(max_steps):
            result = agent.step(goal)
            actions_output.append(result.data)
            self.log_info(
                "Step %d: action=%s\nreasoning=%s\nsummary=%s"
                % (
                    step_n,
                    result.data.get("action_output_json"),
                    result.data.get("action_reason"),
                    result.data.get("summary"),
                )
            )
            if result.done:
                return EpisodeResult(
                    done=result.done,
                    step_data=_transpose_lod_to_dol(actions_output),
                )

        self.log_info(f"Task {task} reached max steps {max_steps} without completing.")
        return EpisodeResult(
            done=False,
            step_data=_transpose_lod_to_dol(actions_output),
        )
