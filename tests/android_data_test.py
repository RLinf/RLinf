#!/usr/bin/env python3
"""测试 AndroidWorldDataset 是否正常工作."""

import sys

from omegaconf import OmegaConf
import os

# 将 android_world 的父目录添加到路径中
android_world_parent = "/mnt/project_rlinf/yuanqwang/mobile-agent/android_world"
if android_world_parent not in sys.path:
    sys.path.insert(0, android_world_parent)
def test_android_world_dataset():
    """测试 AndroidWorldDataset."""
    
    # 创建测试配置
    config = OmegaConf.create({
        "data": {
            "type": "android_world",
            "task_family": "android",  # 只用 android 家族
            "n_instances_per_task": 2,  # 每个任务 2 个实例
            "max_complexity": 2,  # 只用简单任务
            "task_names": None,  # 使用所有任务
            "max_prompt_length": 51200,
            "filter_prompt_by_length": True,
            "seed": 42,
        }
    })
    
    # 创建一个简单的 tokenizer mock
    class MockTokenizer:
        eos_token_id = 0
        
        def encode(self, text):
            # 简单模拟：每个字符一个 token
            return list(range(len(text)))
    
    tokenizer = MockTokenizer()
    
    # 导入并创建 Dataset
    from rlinf.data.datasets.android import AndroidWorldDataset
    
    print("创建 AndroidWorldDataset...")
    dataset = AndroidWorldDataset(
        config=config,
        tokenizer=tokenizer,
        seed=42,
    )
    
    print(f"\n✓ 成功创建 Dataset，共 {len(dataset)} 个任务实例")
    
    # 打印一些样本
    print("\n前 5 个任务样本:")
    print("-" * 60)
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        print(f"\n[{i}] 任务: {item.meta['task_name']}")
        print(f"    Prompt: {item.prompt_text[:100000]}")
        print(f"    Prompt 长度: {item.length}")
        print(f"    复杂度: {item.meta.get('complexity')}")
    
    print("\n" + "=" * 60)
    print("✓ 测试通过！")


def test_task_registry():
    """测试 TaskRegistry 是否可用."""
    print("\n测试 TaskRegistry...")
    
    from android_world.registry import TaskRegistry
    
    registry = TaskRegistry()
    
    # 测试各个 family
    families = ["android", "android_world"]
    for family in families:
        try:
            tasks = registry.get_registry(family)
            print(f"  {family}: {len(tasks)} 个任务")
        except Exception as e:
            print(f"  {family}: 错误 - {e}")
    
    # 打印一些任务名称
    android_tasks = registry.get_registry("android")
    print(f"\nAndroid 任务列表 (前 10 个):")
    for name in list(android_tasks.keys())[:10]:
        task_class = android_tasks[name]
        complexity = getattr(task_class, 'complexity', '?')
        print(f"  - {name} (complexity={complexity})")


if __name__ == "__main__":
    print("=" * 60)
    print("AndroidWorld Dataset 测试")
    print("=" * 60)
    
    # 测试 TaskRegistry
    test_task_registry()
    
    # 测试 Dataset
    print("\n" + "=" * 60)
    test_android_world_dataset()
