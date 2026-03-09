"""Android rewards with android world."""
import sys
from omegaconf import DictConfig

android_world_parent = "/mnt/project_rlinf/yingcheng/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)



class AndroidReward:
    def __init__(self, config: DictConfig):
        self.scale = config.get("reward_scale", 1.0)

    def get_reward(self, env,  result,  task):
        if not result.done:
            return 0.0
        else:
            print(f"task_initialized: {task.initialized}")
            if not task.initialized:
                task.initialized = True

            score = task.is_successful(env)
            return float(score) * self.scale