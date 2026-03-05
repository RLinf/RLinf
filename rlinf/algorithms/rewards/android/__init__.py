"""Android rewards with android world."""
import sys
from typing import Any

from omegaconf import DictConfig
from qwen3_vl_agent import StepResult

android_world_parent = "/mnt/project_rlinf/yuanqwang/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

from android_world.env import env_launcher
# Type definition for answer from AndroidWorldDataset
# answer 是一个字典，包含以下字段：
# {
#     "task_name": str,           # 任务名称
#     "params": dict,              # 任务参数
#     "instance_seed": int,        # 实例种子
#     "class_name": str,           # 任务类名
#     "task": TaskEval             # AndroidWorld 任务实例对象
# }
AndroidAnswer = dict[str, Any]


class AndroidReward:
    def __init__(self, config: DictConfig):
        self.scale = config.get("reward_scale", 1.0)
        self.device_id = config.get("device_id", "localhost:5555")
        self.grpc_port = config.get("grpc_port", 8554)
        self.adb_path = config.get("adb_path", "adb")

        if ":" in self.device_id:
            self.console_port = int(self.device_id.split(":")[1]) - 1
        else:
            self.console_port = int(self.device_id.split("-")[1]) - 1
        self._env = None

    def get_env(self):
        if self._env is None:
            self._env = env_launcher.load_and_setup_env(
                console_port = self.console_port,
                emulator_setup = False,
                freeze_datetime = True,
                adb_path = self.adb_path,
                grpc_port = self.grpc_port,
                device_id = self.device_id,
            )
        return self._env

    def get_reward(self, env,  result: StepResult, answer: AndroidAnswer):
        print(f"task name and params: {answer['task_name']}, {answer['params']}")
        if not result.finished:
            return 0.0
        else:
          #  env = self.get_env()
            task = answer["task"]
            print(f"task_initialized: {task.initialized}")
            if not task.initialized:
                task.initialized = True

            score = task.is_successful(env)
            return float(score) * self.scale