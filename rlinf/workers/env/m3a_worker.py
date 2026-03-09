from omegaconf import DictConfig
from typing import TYPE_CHECKING, Optional, Any
from rlinf.envs.adb.adb_env import ADBEnv
from rlinf.scheduler import Channel, Worker
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.data.datasets.android import AndroidWorldDataset
from rlinf.workers.env.llm_wrapper import LLMWrapper
import sys
android_world_parent = "/mnt/project_rlinf/yingcheng/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

from android_world.env import env_launcher
from android_world.agents import m3a
from android_world.episode_runner import EpisodeResult, _transpose_lod_to_dol

class AndroidAgentWorker(Worker):
    """
    Use the built-in m3a agent from Android World for testing.
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.llm = None
        self.dataset = None
        self.generate_input_channel = None
        self.generate_output_channel = None
        self.envs: dict[int, Any] = {}
    
    def init_with_channels(self, generate_input_channel: Channel, generate_output_channel: Channel):
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel

    def init_worker(self):
        hardware_infos = self.hardware_infos
        self.log_info(f"Worker rank {self._rank} bound to {len(hardware_infos)} hardware device(s)")
        
        if len(hardware_infos) == 0:
            raise ValueError(f"Worker rank {self._rank} has no hardware device assigned")
        
        for idx, hw_info in enumerate(hardware_infos):
            hw_info: "ADBHWInfo"
            device_id = hw_info.config.device_id
            adb_path = hw_info.config.adb_path
            grpc_port = self.cfg.reward.get("grpc_port", 8554) + self._rank
            if ":" in device_id:
                console_port = int(device_id.split(":")[1]) - 1
            else:
                console_port = int(device_id.split("-")[1]) - 1
            with self.worker_timer(f"env_setup_device_{idx}"):
                self.log_info(f"Loading and setting up env for device {device_id} on rank {self._rank}...")
            #1.load and setup env
            env = env_launcher.load_and_setup_env(
                console_port=console_port,
                emulator_setup=False,
                freeze_datetime=True,
                adb_path=adb_path,
                grpc_port=grpc_port,
                device_id=device_id,
            )
            self.log_info(f"Env loaded and setup for device {device_id} on rank {self._rank}")
            self.envs[idx] = env
        self._num_envs = len(self.envs)
        self._next_env_idx = 0
        #2.load dataset
        tokenizer = hf_tokenizer(self.cfg.data.tokenizer.tokenizer_model)
        self.dataset = AndroidWorldDataset(
            config=self.cfg,
            tokenizer=tokenizer,
            seed=self.cfg.data.get("seed", 42),
        )
        self.log_info(f"✓ AndroidWorld dataset loaded: {len(self.dataset)} task instances")

        #3.load llm
        self.llm = LLMWrapper(
            generate_input_channel=self.generate_input_channel, 
            generate_output_channel=self.generate_output_channel
        )

    def _get_env(self, device_key: int):
        """Get the env for the task"""
        return self.envs[device_key]

    def get_num_devices(self) -> int:
        """Get the number of devices"""
        return self._num_envs
    
    def make_device_key(self, worker_rank: int, device_idx: int) -> str:
        """Make a device key for the device"""
        return f"{worker_rank}_{device_idx}"

    def _get_next_env(self):
        """Get the next env for the task, return (env, env_info)"""
        if self._num_envs == 0:
            raise ValueError(f"Worker rank {self._rank} has no devices available")
        
        env_idx = self._next_env_idx
        env = self.envs[env_idx]
        hw_info: "ADBHWInfo" = self.hardware_infos[env_idx]
        grpc_port = self.cfg.reward.get("grpc_port", 8554) + self._rank
        if ":" in hw_info.config.device_id:
            console_port = int(hw_info.config.device_id.split(":")[1]) - 1
        else:
            console_port = int(hw_info.config.device_id.split("-")[1]) - 1
        env_info = {
            "device_id": hw_info.config.device_id,
            "grpc_port": grpc_port,
            "console_port": console_port,
            "adb_path": hw_info.config.adb_path,
        }
        self._next_env_idx = (self._next_env_idx + 1) % self._num_envs
        return env, env_info

    def process_task(self, task_idx: int, reward_worker_group_name: str = "RewardWorkerGroup"):
        """Process a single task from the dataset."""
        task_item = self.dataset[task_idx]
        task = task_item.answer["task"] # get a task instance from the dataset
        env, env_info = self._get_next_env()
        device_id = env_info["device_id"]

        #1.initialize task
        try:
            task.initialize_task(env)
        except Exception as e:
            self.log_error(f"Failed to initialize task {task_idx} on device {device_id} "
                           f"(worker rank {self._rank}): {type(e).__name__}: {e}")
            raise

        #2.run task
        agent_result = self._run_agent(env, task)
        env_info["agent_result"] = agent_result
        env_info["task"] = task

        #3.send env info to reward worker
        self.send(
            object=env_info,
            dst_group_name=reward_worker_group_name,
            dst_rank=self._rank,
        )
        self.log_info(f"Sent env info[{task}] to {reward_worker_group_name}[{self._rank}]")
        #4.get reward from reward worker
        reward = self.recv(
            src_group_name=reward_worker_group_name,
            src_rank=self._rank,
        )
        self.log_info(f"Received reward from {reward_worker_group_name}[{self._rank}], reward: {reward}")
        #5.tear down env
        task.tear_down(env)
        self.log_info(f"Torn down env for task {task_idx}")
        return reward
    
    
    def _run_agent(self, env, task):
        """Run the m3a agent for the task"""
        agent = m3a.M3A(env, self.llm)
        max_steps = int(task.complexity * 10)
        agent.reset(task.start_on_home_screen)
        agent.set_max_steps(max_steps)
        goal = task.goal
        if hasattr(task, 'guidelines'):
            agent.set_task_guidelines(task.guidelines)
        actions_output = []
        
        for step_n in range(max_steps):
            result = agent.step(goal)
            actions_output.append(result.data)
            if result.done:
                return EpisodeResult(done=result.done, step_data=_transpose_lod_to_dol(actions_output))
        self.log_info(f"Task {task} reached max steps {max_steps} without completing.")
        return EpisodeResult(done=False, step_data=_transpose_lod_to_dol(actions_output))
    