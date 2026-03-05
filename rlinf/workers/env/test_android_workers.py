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
from rlinf.scheduler import Channel

import os
from omegaconf import OmegaConf

def create_test_config():
    """
    Args:
        None
    Returns:
        dict: The configuration dictionary for the test.
    """
    tokenizer_path = os.environ.get(
        "TOKENIZER_PATH",
        "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
    )

    config_dict = {
        "cluster": {
            "num_nodes": 1,
            "component_placement": {
                # GPU 放置
                "actor": {"node_group": "gpu", "placement": "0"},
                "rollout": {"node_group": "gpu", "placement": "0"},
                "reward": {"node_group": "gpu", "placement": "all"},
                # Android 相关 worker
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
                    "label": "gpu",
                    "node_ranks": "0",
                },
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
                          #  {"device_id": "localhost:5557", "adb_path": "adb", "node_rank": 0},
                        ],
                    },
                }
            ],
        },
        "runner": {
            "experiment_name": "qwen3-vl-4b-fsdp-2",
            "output_dir": "./logs",
            "seq_length": 28672,
        },
        "actor": {
            "group_name": "ActorGroup",
            "training_backend": "fsdp",
            "enable_offload": False, # 对齐配置 B，追求极致性能时关闭离线
            "model": {
                "model_path": "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
                "model_type": "qwen3_vl",
                "precision": "bf16",
                "enable_memory_saver": False, # 对齐配置 B
            },
        },
        "algorithm": {
            "group_size": 2,
            "sampling_params": {
                "do_sample": True,
                "temperature": 0.6,
                "top_k": 1000000,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "max_new_tokens": 1024, # 根据实际 prompt 长度动态调整
            },
        },
        "rollout": {
            "rollout_backend": "sglang",
            "gpu_memory_utilization": 0.79325, # 核心对齐：配置 B 的显存比例
            "model": {
                "model_path": "/mnt/project_rlinf/hf_models/Qwen3-VL-4B-Instruct",
                "model_type": "qwen3_vl",
                "precision": "bf16",
            },
            "enforce_eager": False, # 捕获 CUDA Graph
            "tensor_parallel_size": 1,
            "max_running_requests": 64, # 对齐高并发配置
            "cuda_graph_max_bs": 256,    # 核心对齐：配置 B 的最大 Batch 图捕获
            "validate_weight": True,
            "sglang": {
                "attention_backend": "triton", # 核心对齐：切换到高性能后端 flashinfer出现乱码  why？
                "sampling_backend": "triton",   # 显式指定
                "grammar_backend": "xgrammar",
                "decode_log_interval": 40,         # 对齐配置 B 的监控频率
                "use_torch_compile": False,
                "torch_compile_max_bs": 32,
                # 针对多模态 Qwen3-VL 补充
                "chunked_prefill_size": 8192,
                "max_prefill_tokens": 16384,
            },
            "detokenize": True,
            "return_logprobs": False,
        },
        "data": {
            "type": "android",
            "task_family": "android_world",
            "n_instances_per_task": 1,
            "seed": 1234,
            "max_prompt_length": 8192,
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
            "reward_weights": {
                "qa_accuracy": 1.0,
            }
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
    channel_a2l = Channel.create("a2l")
    channel_l2a = Channel.create("l2a") 
    from rlinf.utils.placement import ModelParallelComponentPlacement
    from rlinf.workers.rollout.utils import get_rollout_backend_worker
    mpc_placement = ModelParallelComponentPlacement(cfg, cluster)
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_group = rollout_worker_cls.create_group(
        cfg, mpc_placement, weight_reload=None
    ).launch(
        cluster,
        name="RolloutGroup",
        placement_strategy=mpc_placement.get_strategy("rollout"),
    )
    rollout_group.init_worker().wait()

    # 启动 SGLangWorker 的 vl_generate_serverless 循环（在后台）
    llm_handle = rollout_group.vl_generate_serverless(channel_a2l, channel_l2a)

    # --- 启动 Workers ---
    print("启动 AndroidAgentWorker (2 workers)...")
    agent_group = AndroidAgentWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("agent_worker"),
        name="AndroidAgentWorkerGroup",
    )
    agent_group.init_with_channels(channel_a2l, channel_l2a).wait()
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

    # --- 断点续跑：如果结果文件已存在，则读取已完成任务，后续跳过 ---
    existing_results: list = []
    finished_indices: set[int] = set()
    if output_path.exists():
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
            existing_results = prev_data.get("tasks", []) or []
            for item in existing_results:
                idx = item.get("task_idx")
                if isinstance(idx, int):
                    finished_indices.add(idx)
            print(
                f"检测到已有结果文件 {output_path}，"
                f"已完成 {len(finished_indices)} 个任务，将跳过这些任务。"
            )
        except Exception as e:
            print(f"读取已有结果文件失败，将从头开始评估：{e}")
            existing_results = []
            finished_indices = set()

    # --- 主循环：按 worker 分片，每个 worker 顺序跑自己的任务列表（无 batch 短板）---
    # 仅对未完成的任务进行评估
    all_task_indices = [i for i in range(dataset_size) if i not in finished_indices]
    total_tasks = len(all_task_indices)

    from math import ceil

    def split_tasks(indices: list[int], num_shards: int) -> list[list[int]]:
        shard_size = ceil(len(indices) / max(num_shards, 1))
        return [
            indices[i * shard_size : (i + 1) * shard_size]
            for i in range(num_shards)
        ]

    task_shards = split_tasks(all_task_indices, num_agent_workers)
    print("任务分片情况：")
    for rank, shard in enumerate(task_shards):
        print(f"  Worker {rank}: {shard}")

    wall_start = time.perf_counter()
    # 已有结果先加入 all_results，用于断点续跑
    all_results: list = list(existing_results)

    reward_handles = []
    agent_handles = []
    task_shards = [[35]]
    for rank, shard in enumerate(task_shards):
        if not shard:
            continue

        print(f"\n{'='*60}")
        print(f"Worker {rank}: tasks {shard}")
        print(f"{'='*60}")

        rh = reward_group.execute_on(rank).compute_reward_loop(
            agent_worker_group_name="AndroidAgentWorkerGroup",
            num_tasks=len(shard),
        )
        reward_handles.append(rh)

        ah = agent_group.execute_on(rank).run_task_list(
            task_indices=shard,
            reward_worker_group_name="RewardWorkerGroup",
        )
        agent_handles.append(ah)

    # 等所有 worker 把自己的任务串跑完，每完成一批就增量写一次 JSON，防止中途退出丢失所有结果
    for ah in agent_handles:
        shard_results = ah.wait()
        if (
            isinstance(shard_results, list)
            and len(shard_results) == 1
            and isinstance(shard_results[0], list)
        ):
            shard_results = shard_results[0]
        all_results.extend(shard_results)

        # 增量保存当前进度
        cur_elapsed = time.perf_counter() - wall_start
        total_tasks_run = len(all_results)
        successful = [r for r in all_results if r.get("reward", 0) > 0]
        accuracy = (
            len(successful) / total_tasks_run if total_tasks_run > 0 else 0.0
        )
        summary_partial = {
            "timestamp": timestamp,
            "num_workers": num_agent_workers,
            "total_tasks": total_tasks_run,
            "successful_tasks": len(successful),
            "accuracy": round(accuracy, 4),
            "wall_time_seconds": round(cur_elapsed, 2),
            "wall_time_formatted": format_time(cur_elapsed),
        }
        output_data_partial = {
            "summary": summary_partial,
            "tasks": all_results,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                output_data_partial,
                f,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    for rh in reward_handles:
        rh.wait()

    wall_elapsed = time.perf_counter() - wall_start

    # --- 汇总（最终一次）---
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

    # --- 最终写 JSON（覆盖增量版本，保证统计是完整的）---
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
