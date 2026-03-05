"""InitWorker for Android World - Initializes Android World environments and dataset."""
import sys
from typing import TYPE_CHECKING

from omegaconf import DictConfig

from rlinf.data.datasets.android import AndroidWorldDataset
from rlinf.data.tokenizers import hf_tokenizer
from rlinf.scheduler import Worker

if TYPE_CHECKING:
    from rlinf.scheduler.hardware import ADBHWInfo

# Add android_world to path
android_world_parent = "/mnt/project_rlinf/yuanqwang/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

from android_world.env import env_launcher, json_action

class InitWorker(Worker):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        if not self.hardware_infos:
            raise ValueError(
                "InitWorker requires hardware_infos to bind ADB device(s)."
            )
        self.android_envs = {}
        self.cfg = cfg
        self.dataset = None
    
    def init_worker(self):

        for idx, hw_info in enumerate(self.hardware_infos):
            hw_info: "ADBHWInfo"
            device_id = hw_info.config.device_id
            adb_path = hw_info.config.adb_path

            if ":" in device_id:
                console_port = int(device_id.split(":")[1]) - 1
            else:
                console_port = int(device_id.split("-")[1]) - 1
            
            grpc_port = self.cfg.reward.get("grpc_port", 8554)

            try:
                env = env_launcher.load_and_setup_env(
                    console_port=console_port,
                    emulator_setup=False,
                    freeze_datetime=True,
                    adb_path=adb_path,
                    grpc_port=grpc_port,
                    device_id=device_id,
                )
                self.android_envs[idx] = env
                self.log_info(f"✓ AndroidWorld env created for device {device_id}")
            except Exception as e:
                self.log_error(f"✗ Failed to create env for device {device_id}: {e}")
                raise

        # 2. Load AndroidWorld dataset
        self.log_info("Loading AndroidWorld dataset...")
        try:
            tokenizer = hf_tokenizer(self.cfg.data.tokenizer.tokenizer_model)
            self.dataset = AndroidWorldDataset(
                config=self.cfg,
                tokenizer=tokenizer,
                seed=self.cfg.data.get("seed", 42),
            )
            self.log_info(
                f"✓ AndroidWorld dataset loaded: {len(self.dataset)} task instances"
            )
            task_item = self.dataset[0]
            task = task_item.answer["task"]
            print(f"task: {task}")
            task.initialize_task(env)
            click_action = json_action.JSONAction(
                action_type="click",
                index=5  # 点击索引为 5 的 UI 元素
            )
            env.execute_action(click_action)
        except Exception as e:
            self.log_error(f"✗ Failed to load dataset: {e}")
            raise

    def send_env_task_to_rewardworker(self, reward_worker_group_name: str = "RewardWorkerGroup"):
        """Send env info and dataset to RewardWorker.
        
        Note: We send device info instead of env object because AsyncEnv objects
        are not easily serializable (contain network connections, file handles, etc.).
        RewardWorker will create its own env using the device info.
        
        Args:
            reward_worker_group_name: Name of the RewardWorker group
        """
        self.log_info(f"Sending env info and dataset to {reward_worker_group_name}...")
        
        # Send device info for each env (instead of env object itself)
        for idx, hw_info in enumerate(self.hardware_infos):
            hw_info: "ADBHWInfo"
            env_info = {
                "device_id": hw_info.config.device_id,
                "adb_path": hw_info.config.adb_path,
                "grpc_port": self.cfg.reward.get("grpc_port", 8554),
                "console_port": int(hw_info.config.device_id.split(":")[1]) - 1 if ":" in hw_info.config.device_id else int(hw_info.config.device_id.split("-")[1]) - 1,
            }

            self.log_info(f"Sending env info[{idx}] to {reward_worker_group_name}[{self._rank}]")
            self.send(
                env_info, 
                dst_group_name=reward_worker_group_name, 
                dst_rank=self._rank
            )
            self.log_info(f"Sent env info[{idx}] to {reward_worker_group_name}[{self._rank}]")
        
        # Send dataset (this might still be slow if dataset is large)
        self.log_info(f"Sending dataset to {reward_worker_group_name}...")
        self.send(
            self.dataset[0], 
            dst_group_name=reward_worker_group_name, 
            dst_rank=self._rank
        )
        self.log_info(f"Sent dataset to {reward_worker_group_name}[{self._rank}]")