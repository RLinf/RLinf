import sys
from omegaconf import DictConfig
from rlinf.algorithms.rewards.android import AndroidReward
from rlinf.scheduler import Worker

# Add android_world to path
android_world_parent = "/mnt/project_rlinf/yingcheng/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

from android_world.env import env_launcher

class AndroidRewardWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.android_reward = AndroidReward(cfg.reward)
    
    def init_worker(self):
        pass

    def _receive_env_info(self, agent_worker_group_name: str = "AndroidAgentWorkerGroup"):
        """Receive env info and task from AgentWorker, reconnect the env for this task."""
        self.log_info(f"Waiting to receive env info and task from {agent_worker_group_name}[{self._rank}]...")
        env_info_and_task = self.recv(src_group_name=agent_worker_group_name, src_rank=self._rank)
        env_info = env_info_and_task["env_info"]
        env = env_launcher.load_and_setup_env(
            console_port=env_info["console_port"],
            emulator_setup=False,
            freeze_datetime=False,
            adb_path=env_info["adb_path"],
            grpc_port=env_info["grpc_port"],
            device_id=env_info["device_id"],
        )
        self.log_info(f"Reconnected env for device {env_info['device_id']}")
        return env, env_info_and_task["task"], env_info_and_task["agent_result"]
    
    def compute_reward(self,  agent_worker_group_name: str = "AndroidAgentWorkerGroup"):
        """Compute reward for the task"""
        env, task, agent_result = self._receive_env_info(agent_worker_group_name)

        reward = self.android_reward.get_reward(env, agent_result, task)

        self.send(
            reward,
            dst_group_name=agent_worker_group_name,
            dst_rank=self._rank,
        )
        self.log_info(f"Sent reward {reward} to {agent_worker_group_name}[{self._rank}]")
        return reward
    
   