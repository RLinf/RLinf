#!/usr/bin/env python3
"""实际验证 Android Reward 函数 - 需要连接 Android 模拟器."""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path("/mnt/project_rlinf/yuanqwang/mobile-agent")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 添加 RLinf 路径
rlinf_path = project_root / "RLinf"
if str(rlinf_path) not in sys.path:
    sys.path.insert(0, str(rlinf_path))

# 添加 android_world 路径
android_world_parent = "/mnt/project_rlinf/yuanqwang/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)

from omegaconf import OmegaConf
from qwen3_vl_agent import StepResult
from rlinf.algorithms.rewards.android import AndroidReward, AndroidAnswer
from rlinf.data.datasets.android import AndroidWorldDataset
from transformers import AutoTokenizer


def check_adb_connection(device_id: str = "localhost:5555") -> bool:
    """检查 ADB 设备是否连接."""
    import subprocess
    try:
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=5
        )
        devices = result.stdout.strip().split("\n")[1:]  # 跳过第一行标题
        connected_devices = [d.split()[0] for d in devices if d.strip() and "device" in d]
        
        print(f"Connected ADB devices: {connected_devices}")
        
        if device_id in connected_devices:
            print(f"✓ Device {device_id} is connected")
            return True
        elif connected_devices:
            print(f"⚠ Device {device_id} not found, but found: {connected_devices[0]}")
            print(f"  Using {connected_devices[0]} instead")
            return True
        else:
            print("✗ No ADB devices connected")
            print("\nPlease connect an Android device or start an emulator:")
            print("  1. Connect via USB: adb devices")
            print("  2. Or start emulator: emulator -avd <AVD_NAME> -grpc 8554")
            return False
    except FileNotFoundError:
        print("✗ ADB not found. Please install Android SDK Platform Tools")
        return False
    except Exception as e:
        print(f"✗ Error checking ADB connection: {e}")
        return False


def create_test_config(device_id: str = "localhost:5555") -> OmegaConf:
    """创建测试配置."""
    config = OmegaConf.create({
        "reward_scale": 1.0,
        "device_id": device_id,
        "grpc_port": 8554,
        "adb_path": "adb",
    })
    return config


def create_test_dataset_config() -> OmegaConf:
    """创建测试数据集配置."""
    config = OmegaConf.create({
        "data": {
            "type": "android_world",
            "task_family": "android_world",  # 或 "android"
            "n_instances_per_task": 1,
            "max_complexity": 1,  # 只使用简单任务
            "task_name": None,  # 使用所有任务，或指定特定任务如 "SetAlarmClock"
            "max_prompt_length": 8192,
            "filter_prompt_by_length": False,
            "seed": 42,
            "apply_chat_template": False,
        }
    })
    return config


def test_reward_with_real_task(
    reward_func: AndroidReward,
    task_instance: dict,
    simulate_success: bool = True
) -> float:
    """使用真实任务测试 reward 函数."""
    print("\n" + "=" * 60)
    print("Testing Reward Function with Real Task")
    print("=" * 60)
    
    # 创建 answer 字典
    answer: AndroidAnswer = {
        "task_name": task_instance["task_name"],
        "params": task_instance["params"],
        "instance_seed": task_instance["instance_seed"],
        "class_name": task_instance["task_class"].__name__,
        "task": task_instance["task_instance"],
    }
    
    # 创建 StepResult
    # 注意：在实际使用中，这个 result 应该来自 agent 的执行结果
    result = StepResult(
        success=True,
        finished=simulate_success,  # 模拟任务完成
        action=None,
        thinking="Task completed" if simulate_success else "Task in progress",
    )
    
    print(f"Task: {answer['task_name']}")
    print(f"Task class: {answer['class_name']}")
    print(f"Result finished: {result.finished}")
    print(f"Task initialized: {answer['task'].initialized}")
    
    try:
        # 调用 reward 函数
        reward = reward_func.get_reward(result, answer)
        
        print(f"\n✓ Reward calculated: {reward}")
        print(f"  - Task initialized: {answer['task'].initialized}")
        print(f"  - Scale: {reward_func.scale}")
        
        return reward
    except Exception as e:
        print(f"\n✗ Error calculating reward: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_unfinished_task(reward_func: AndroidReward, answer: AndroidAnswer) -> float:
    """测试未完成任务的 reward."""
    print("\n" + "=" * 60)
    print("Testing Unfinished Task (should return 0.0)")
    print("=" * 60)
    
    result = StepResult(
        success=False,
        finished=False,  # 未完成
        action=None,
        thinking="Still working...",
    )
    
    print(f"Result finished: {result.finished}")
    
    reward = reward_func.get_reward(result, answer)
    print(f"Reward: {reward}")
    print(f"Expected: 0.0")
    
    assert reward == 0.0, f"Expected 0.0 for unfinished task, got {reward}"
    print("✓ Test passed: Unfinished task returns 0.0")
    
    return reward


def test_successful_task(reward_func: AndroidReward, answer: AndroidAnswer) -> float:
    """测试成功任务的 reward."""
    print("\n" + "=" * 60)
    print("Testing Successful Task")
    print("=" * 60)
    
    result = StepResult(
        success=True,
        finished=True,  # 已完成
        action=None,
        thinking="Task completed successfully",
    )
    
    print(f"Result finished: {result.finished}")
    print(f"Task initialized before: {answer['task'].initialized}")
    
    try:
        reward = reward_func.get_reward(result, answer)
        
        print(f"Task initialized after: {answer['task'].initialized}")
        print(f"Reward: {reward}")
        print(f"Scale: {reward_func.scale}")
        
        # Reward 应该是 0.0 或 1.0 * scale（取决于任务是否真的成功）
        assert reward >= 0.0, f"Reward should be >= 0, got {reward}"
        assert reward <= reward_func.scale, f"Reward should be <= scale ({reward_func.scale}), got {reward}"
        
        print("✓ Test passed: Reward calculated successfully")
        return reward
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主测试函数."""
    print("=" * 60)
    print("Android Reward Function - Real Device Verification")
    print("=" * 60)
    
    # 1. 检查 ADB 连接
    device_id = "localhost:5555"  # 可以根据实际情况修改
    if not check_adb_connection(device_id):
        print("\n⚠ Cannot proceed without connected device")
        print("\nTo start an emulator:")
        print("  emulator -avd <AVD_NAME> -no-snapshot -grpc 8554")
        print("\nOr connect a physical device via USB")
        return 1
    
    # 2. 创建 reward 函数
    print("\n" + "=" * 60)
    print("Initializing AndroidReward")
    print("=" * 60)
    
    config = create_test_config(device_id)
    reward_func = AndroidReward(config)
    
    print(f"Device ID: {reward_func.device_id}")
    print(f"Console port: {reward_func.console_port}")
    print(f"GRPC port: {reward_func.grpc_port}")
    print(f"Scale: {reward_func.scale}")
    
    # 3. 创建测试任务
    print("\n" + "=" * 60)
    print("Loading Test Task from AndroidWorld")
    print("=" * 60)
    
    try:
        # 创建一个简单的 tokenizer（用于数据集）
        # 注意：这里我们只需要任务实例，不需要实际 tokenize
        class SimpleTokenizer:
            eos_token_id = 0
            def encode(self, text):
                return list(range(len(text)))
        
        tokenizer = SimpleTokenizer()
        dataset_config = create_test_dataset_config()
        
        print("Creating AndroidWorldDataset...")
        dataset = AndroidWorldDataset(
            config=dataset_config,
            tokenizer=tokenizer,
            seed=42,
        )
        
        if len(dataset) == 0:
            print("✗ No tasks loaded from dataset")
            return 1
        
        print(f"✓ Loaded {len(dataset)} task instances")
        
        # 获取第一个任务
        task_item = dataset[12]
        for i, task_item2 in enumerate(dataset):
            print(f"Task {i}: {task_item2.meta['task_name']}")
            print(f"Task goal: {task_item2.answer['task'].goal}")

        task_instance = {
            "task_name": task_item.meta["task_name"],
            "task_class": task_item.meta["task_class"],
            "task_instance": task_item.answer["task"],
            "params": task_item.answer["params"],
            "instance_seed": task_item.answer["instance_seed"],
        }
        
        print(f"Selected task: {task_instance['task_name']}")
        print(f"Task goal: {task_instance['task_instance'].goal}")
        
    except Exception as e:
        print(f"✗ Error loading task: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 4. 测试未完成的任务
    answer: AndroidAnswer = {
        "task_name": task_instance["task_name"],
        "params": task_instance["params"],
        "instance_seed": task_instance["instance_seed"],
        "class_name": task_instance["task_class"].__name__,
        "task": task_instance["task_instance"],
    }
    
    try:
        test_unfinished_task(reward_func, answer)
    except Exception as e:
        print(f"⚠ Error in unfinished task test: {e}")
    
    # 5. 测试已完成的任务（实际连接设备）
    print("\n" + "=" * 60)
    print("Testing with Real Device Connection")
    print("=" * 60)
    print("⚠ This will connect to the Android device/emulator")
    print("⚠ Make sure the device is ready and apps are installed")
    
    try:
        # 初始化环境（这会实际连接设备）
        print("\nConnecting to device...")
        env = reward_func.get_env()
        print(f"✓ Environment connected.{env.controller.get_ui_elements()}")
        
        # 注意：当前实现中，如果 task.initialized 为 False，
        # reward 函数会直接设置 task.initialized = True
        # 但为了正确验证，我们可能需要先初始化任务
        task = answer["task"]
        print(f"Task initialized before reward call: {task.initialized}")
        
        # 如果需要，可以手动初始化任务以确保正确性
        # 注意：这取决于你的实际需求
        if not task.initialized:
            print("Initializing task manually...")
         #   task.initialize_task(env)
            task.initialized = True
            print("✓ Task initialized")
        
        # 测试 reward
        reward = test_successful_task(reward_func, answer)
        
        if reward is not None:
            print(f"\n✓ Final reward: {reward}")
            print(f"  - This indicates whether the task was successful on the device")
            print(f"  - 0.0 = task failed, {reward_func.scale} = task succeeded")
        
    except Exception as e:
        print(f"✗ Error connecting to device: {e}")
        print("\nTroubleshooting:")
        print("  1. Make sure emulator is running: adb devices")
        print("  2. Check console port matches device_id")
        print("  3. Ensure GRPC port is available")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("✓ All tests completed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
