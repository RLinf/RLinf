import sys
import time
from omegaconf import DictConfig
from qwen3_vl_agent import StepResult
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.data.datasets.android import AndroidWorldDataset

from rlinf.algorithms.rewards.android import AndroidAnswer, AndroidReward
from rlinf.scheduler import Worker

# Add android_world to path
android_world_parent = "/mnt/project_rlinf/yingcheng/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

class RewardWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.android_reward = AndroidReward(cfg.reward)
        # 每个任务的单独计时：{task_idx: {metric_name: time}}
        self.per_task_timings: dict[int, dict[str, float]] = {}
    
    def init_worker(self):
        """Initialize worker and prepare to receive env info and dataset.
        
        Note: We don't receive here, but this method can be used for other initialization.
        The actual receiving happens in compute_reward().
        """
        pass
    #必须把task传过来！
        # tokenizer = hf_tokenizer(self.cfg.data.tokenizer.tokenizer_model)
        # self.dataset = AndroidWorldDataset(
        #     config=self.cfg,
        #     tokenizer=tokenizer,
        #     seed=self.cfg.data.get("seed", 42),
        # )
        # self.log_info(f"✓ AndroidWorld dataset loaded: {len(self.dataset)} task instances")
    
    def _receive_env_task(self, agent_worker_group_name: str = "AndroidAgentWorkerGroup", task_idx: int = None):
        """Receive env info and dataset from AgentWorker, then create env.
        
        Args:
            agent_worker_group_name: Name of the AgentWorker group
            task_idx: Task index for per-task timing (if None, will try to get from env_info)
            
        Returns:
            tuple: (env, env_info)
        """
        # Receive env info (device info) instead of env object
        self.log_info(f"Waiting to receive env info from {agent_worker_group_name}[{self._rank}]...")
        recv_start = time.perf_counter()
        env_info = self.recv(src_group_name=agent_worker_group_name, src_rank=self._rank)
        recv_duration = time.perf_counter() - recv_start
        
        # 获取 task_idx（优先使用参数，否则从 env_info 获取）
        if task_idx is None:
            task_idx = env_info.get("task_idx", -1)  # -1 表示未知任务
        
        # 初始化当前任务的计时字典（如果还没有）
        if task_idx not in self.per_task_timings:
            self.per_task_timings[task_idx] = {}
        task_timings = self.per_task_timings[task_idx]
        
        # 细粒度计时：接收 env_info 的时间（同时记录到全局和当前任务）
        self._timer_metrics["recv_env_info"] = (
            self._timer_metrics.get("recv_env_info", 0.0) + recv_duration
        )
        task_timings["recv_env_info"] = recv_duration
        self.log_info(f"Received env info from {agent_worker_group_name}[{self._rank}]")
        
        # Create env using the received device info（计时，同时记录到全局和当前任务）
        from android_world.env import env_launcher
        env_setup_start = time.perf_counter()
        with self.worker_timer("reward_env_setup"):
            self.log_info("Creating env from device info...")
            # Use UIAUTOMATOR mode for reward worker to avoid reconfiguring
            # the AccessibilityForwarder port (which would break the
            # agent_worker's A11Y gRPC stream for subsequent tasks).
            env = env_launcher.load_and_setup_env(
                console_port=env_info["console_port"],
                emulator_setup=False,
                freeze_datetime=False,
                adb_path=env_info["adb_path"],
                grpc_port=env_info["grpc_port"],
                device_id=env_info["device_id"],
            )
            self.log_info(f"Created env for device {env_info['device_id']}")
        env_setup_duration = time.perf_counter() - env_setup_start
        task_timings["reward_env_setup"] = env_setup_duration
        
        return env, env_info
    
    def compute_reward(
        self, 
        agent_worker_group_name: str = "AndroidAgentWorkerGroup",
    ):
        """Compute reward for a task.
        
        Args:
            agent_worker_group_name: Name of the AgentWorker group to receive from and send to
            
        Returns:
            float: Reward value
        """
        env, env_info = self._receive_env_task(agent_worker_group_name)
        task_idx = env_info.get("task_idx", -1)
        task_timings = self.per_task_timings.get(task_idx, {})
        
        task_item= env_info["task_item"]
        # Get task from dataset
        task_instance = {
            "task_name": task_item.answer["task_name"],
            "task_class": task_item.answer.get("class_name"),
            "task_instance": task_item.answer["task"],
            "params": task_item.answer["params"],
            "instance_seed": task_item.answer["instance_seed"],
        }
        
        # Create a mock StepResult for testing
        result = StepResult(
            success=True,
            finished=env_info["agent_result"].done,  # 已完成
            action=None,
            thinking="Task completed successfully",
        )
        
        # Prepare answer dict
        answer: AndroidAnswer = {
            "task_name": task_instance["task_name"],
            "params": task_instance["params"],
            "instance_seed": task_instance["instance_seed"],
            "class_name": task_instance["task_class"] if task_instance["task_class"] else "Unknown",
            "task": task_instance["task_instance"],
        }
        
        # Compute reward（计时，同时记录到全局和当前任务）
        reward_compute_start = time.perf_counter()
        with self.worker_timer("reward_compute"):
            reward = self.android_reward.get_reward(env, result, answer)
        reward_compute_duration = time.perf_counter() - reward_compute_start
        task_timings["reward_compute"] = reward_compute_duration
        self.log_info(f"Computed reward: {reward}")
        
        # 细粒度计时：发送 reward 的时间（同时记录到全局和当前任务）
        send_start = time.perf_counter()
        self.send(
            reward,
            dst_group_name=agent_worker_group_name,
            dst_rank=self._rank,
        )
        send_duration = time.perf_counter() - send_start
        self._timer_metrics["send_reward"] = (
            self._timer_metrics.get("send_reward", 0.0) + send_duration
        )
        task_timings["send_reward"] = send_duration
        return reward

    def compute_reward_loop(
        self,
        agent_worker_group_name: str = "AndroidAgentWorkerGroup",
        num_tasks: int = 1,
    ):
        """连续处理 num_tasks 个 env_info，每次内部调用一次 compute_reward。"""
        rewards = []
        for _ in range(num_tasks):
            r = self.compute_reward(agent_worker_group_name=agent_worker_group_name)
            rewards.append(r)
        return rewards

    def get_timings(self):
        """返回当前 worker 已记录的计时信息（单位：秒）.
        
        Returns:
            dict: 包含 'per_task' 和 'total' 两个键
                - 'per_task': {task_idx: {metric_name: time}} 每个任务的计时
                - 'total': {metric_name: total_time} 所有任务的汇总时间
        """
        return {
            "per_task": dict(self.per_task_timings),
            "total": dict(self._timer_metrics),
        }