#!/usr/bin/env python3
"""测试 AndroidAgentWorker 和 RewardWorker：跑整个 dataset，两个 worker 并行，结果保存为 JSON."""

import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

project_root = Path("/mnt/project_rlinf/yuanqwang/mobile-agent")
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

rlinf_path = project_root / "RLinf"
if str(rlinf_path) not in sys.path:
    sys.path.insert(0, str(rlinf_path))

import os

current_pythonpath = os.environ.get("PYTHONPATH", "")
pythonpath_parts = []
if current_pythonpath:
    pythonpath_parts.append(current_pythonpath)
pythonpath_parts.append(str(project_root))
pythonpath_parts.append(str(rlinf_path))
os.environ["PYTHONPATH"] = ":".join(pythonpath_parts)

from omegaconf import OmegaConf
from rlinf.scheduler import Cluster
from rlinf.utils.placement import ComponentPlacement
from rlinf.workers.env.agent_worker import AndroidAgentWorker
from rlinf.workers.env.reward_worker import RewardWorker

def create_test_config():
    """创建测试配置：两个 device、两个 worker."""
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH",
        "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
    )

    config_dict = {
        "cluster": {
            "num_nodes": 1,
            "component_placement": {
                "agent_worker": {
                    "node_group": "android_world",
                    "placement": "0",
                },
                "reward_worker": {
                    "node_group": "android_world",
                    "placement": "0",
                },
            },
            "node_groups": [
                {
                    "label": "android_world",
                    "node_ranks": "0",
                    "env_configs": [
                        {
                            "node_ranks": "0",
                            "python_interpreter_path": "/opt/venv/reason/bin/python3",
                        }
                    ],
                    "hardware": {
                        "type": "ADB",
                        "configs": [
                            {"device_id": "localhost:5555", "adb_path": "adb", "node_rank": 0},
                        #    {"device_id": "localhost:5557", "adb_path": "adb", "node_rank": 0},
                        ],
                    },
                }
            ],
        },
        "data": {
            "type": "android",
            "task_family": "android_world",
            "n_instances_per_task": 1,
            "seed": 1234,
            "max_prompt_length": 8192,
            "filter_prompt_by_length": True,
            "tokenizer": {
                "tokenizer_model": tokenizer_path,
                "use_fast": False,
                "trust_remote_code": True,
                "padding_side": "right",
            },
        },
        "reward": {
            "reward_type": "android",
            "reward_scale": 1.0,
            "device_id": "localhost:5555",
            "grpc_port": 8554,
            "adb_path": "adb",
        },
    }
    return OmegaConf.create(config_dict)


def format_time(seconds):
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m}m {s:.1f}s"


def main():
    output_dir = Path("/mnt/project_rlinf/yuanqwang/mobile-agent/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"eval_results_{timestamp}.json"

    print("=" * 60)
    print("Android Agent Full Dataset Evaluation")
    print("=" * 60)

    cfg = create_test_config()

    # --- 启动 Cluster ---
    print("创建 Cluster...")
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    print("✓ Cluster 创建成功\n")

    # --- 启动 Workers（HTTP 模式：不需要 Engine/Channel/Rollout） ---
    print("启动 AndroidAgentWorker (2 workers)...")
    agent_group = AndroidAgentWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("agent_worker"),
        name="AndroidAgentWorkerGroup",
    )
    agent_group.init_worker().wait()
    num_agent_workers = len(agent_group._workers)
    print(f"✓ {num_agent_workers} AndroidAgentWorker 初始化完成\n")

    print("启动 RewardWorker (2 workers)...")
    reward_group = RewardWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("reward_worker"),
        name="RewardWorkerGroup",
    )
    reward_group.init_worker().wait()
    print(f"✓ {len(reward_group._workers)} RewardWorker 初始化完成\n")

    # --- 获取 dataset 大小 ---
    dataset_size_result = agent_group.execute_on(0).get_dataset_size()
    dataset_size = dataset_size_result.wait()
    if isinstance(dataset_size, list):
        dataset_size = dataset_size[0]
    print(f"Dataset 共 {dataset_size} 个任务，使用 {num_agent_workers} 个 worker 并行\n")

    # --- 主循环：按 batch 分配任务给多个 worker ---
    all_results = []
    #total_tasks = dataset_size
    target_tasks = [35]
    all_task_indices = target_tasks
    total_tasks = len(all_task_indices)


    task_ptr = 0
    batch_id = 0
    wall_start = time.perf_counter()

    while task_ptr < total_tasks:
        # batch_size = min(num_agent_workers, total_tasks - task_ptr)
        # batch_tasks = list(range(task_ptr, task_ptr + batch_size))
        batch_size = min(num_agent_workers, total_tasks - task_ptr)
        batch_tasks = all_task_indices[task_ptr : task_ptr + batch_size]

        batch_id += 1
        print(f"\n{'='*60}")
        print(f"Batch {batch_id}: tasks {batch_tasks}")
        print(f"{'='*60}")

        agent_handles = []
        reward_handles = []

        for i, task_idx in enumerate(batch_tasks):
            rank = i
            reward_handle = reward_group.execute_on(rank).compute_reward(
                agent_worker_group_name="AndroidAgentWorkerGroup"
            )
            reward_handles.append(reward_handle)

            agent_handle = agent_group.execute_on(rank).process_task(
                task_idx=task_idx,
                reward_worker_group_name="RewardWorkerGroup",
            )
            agent_handles.append(agent_handle)

        for i, (agent_handle, reward_handle) in enumerate(zip(agent_handles, reward_handles)):
            task_idx = batch_tasks[i]
            try:
                result = agent_handle.wait()
                if isinstance(result, list):
                    result = result[0]
                reward_handle.wait()

                all_results.append(result)
                reward_val = result.get("reward", "N/A")
                steps = result.get("num_steps", "?")
                name = result.get("task_name", "?")
                print(f"  ✓ task {task_idx} ({name}): reward={reward_val}, steps={steps}")
            except Exception as e:
                print(f"  ✗ task {task_idx} failed: {e}")
                traceback.print_exc()
                all_results.append({
                    "task_idx": task_idx,
                    "task_name": "ERROR",
                    "goal": "",
                    "complexity": None,
                    "reward": 0.0,
                    "num_steps": 0,
                    "per_step_timings": [],
                    "task_timings": {},
                    "error": str(e),
                })

        task_ptr += batch_size

    wall_elapsed = time.perf_counter() - wall_start

    # --- 汇总 ---
    total_tasks_run = len(all_results)
    successful = [r for r in all_results if r.get("reward", 0) > 0]
    accuracy = len(successful) / total_tasks_run if total_tasks_run > 0 else 0.0

    summary = {
        "timestamp": timestamp,
        "num_workers": num_agent_workers,
        "total_tasks": total_tasks_run,
        "successful_tasks": len(successful),
        "accuracy": round(accuracy, 4),
        "wall_time_seconds": round(wall_elapsed, 2),
        "wall_time_formatted": format_time(wall_elapsed),
    }

    output_data = {
        "summary": summary,
        "tasks": all_results,
    }

    # --- 写 JSON ---
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'='*60}")
    print(f"结果已保存到: {output_path}")
    print(f"{'='*60}")

    # --- 打印汇总 ---
    print(f"\n总任务数: {total_tasks_run}")
    print(f"成功任务: {len(successful)}")
    print(f"正确率:   {accuracy*100:.1f}%")
    print(f"总用时:   {format_time(wall_elapsed)}")
    print()

    print("各任务结果:")
    print(f"{'idx':>4}  {'task_name':<40}  {'complexity':>10}  {'reward':>7}  {'steps':>5}")
    print("-" * 75)
    for r in all_results:
        idx = r.get("task_idx", "?")
        name = r.get("task_name", "?")[:40]
        cx = r.get("complexity", "-")
        rw = r.get("reward", 0)
        st = r.get("num_steps", 0)
        mark = "✓" if rw > 0 else "✗"
        print(f"{idx:>4}  {name:<40}  {str(cx):>10}  {rw:>7.2f}  {st:>5}  {mark}")


if __name__ == "__main__":
    main()
