import json
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import OmegaConf

from rlinf.scheduler import Cluster, Channel
from rlinf.utils.placement import ComponentPlacement, ModelParallelComponentPlacement
from rlinf.utils.utils import output_redirector
from rlinf.workers.env.m3a_worker import AndroidAgentWorker
from rlinf.workers.env.android_reward_worker import AndroidRewardWorker
from rlinf.workers.rollout.utils import get_rollout_backend_worker

CONFIG_DIR = Path(__file__).resolve().parent


def load_resume_state(
    output_path: Path,
    dataset_size: int,
) -> tuple[list, set[int], list[int]]:
    """从已有结果文件恢复断点状态。"""
    existing_results = []
    finished_indices = set()
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
            print(f"读取已有结果文件失败，将从头开始：{e}")
            existing_results = []
            finished_indices = set()
    task_indices_to_run = [
        i for i in range(dataset_size) if i not in finished_indices
    ]
    return existing_results, finished_indices, task_indices_to_run


def save_checkpoint(output_path: Path, all_results: list, summary: dict) -> None:
    """将当前结果与 summary 写入 output_path。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "tasks": all_results},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )


@hydra.main(version_base="1.1", config_path=str(CONFIG_DIR / "config"), config_name="qwen3vl-4b-eval")
@output_redirector
def main(cfg) -> None:
    print(OmegaConf.to_yaml(cfg))

    # 由 __file__ 推导仓库根，与 eval.sh 中 PROJECT_ROOT 一致（eval.sh 已设 PYTHONPATH，此处仅用于写结果目录）
    project_root = (Path(__file__).resolve().parent / ".." / ".." / ".." / "..").resolve()
    print(f"project_root: {project_root}")
    output_dir = project_root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / "eval_results2.json"  # 固定路径，便于断点续跑

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)

    reward_group = AndroidRewardWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("reward_worker"),
        name="RewardWorkerGroup",
    )
    reward_group.init_worker().wait()

    mpc_placement = ModelParallelComponentPlacement(cfg, cluster)
    rollout_worker_cls = get_rollout_backend_worker(cfg)
    rollout_group = rollout_worker_cls.create_group(
        cfg, mpc_placement, weight_reload=None
    ).launch(
        cluster=cluster,
        name=cfg.rollout.get("group_name", "RolloutGroup"),
        placement_strategy=mpc_placement.get_strategy("rollout"),
    )
    rollout_group.init_worker().wait()

    agent_group = AndroidAgentWorker.create_group(cfg).launch(
        cluster=cluster,
        placement_strategy=component_placement.get_strategy("agent_worker"),
        name="AndroidAgentWorkerGroup",
    )
    channel_a2l = Channel.create("a2l")
    channel_l2a = Channel.create("l2a")
    agent_group.init_with_channels(channel_a2l, channel_l2a).wait()
    agent_group.init_worker().wait()


    llm_handle = rollout_group.vl_generate_serverless(channel_a2l, channel_l2a)

    dataset_size_result = agent_group.execute_on(0).get_dataset_size()
    dataset_size = dataset_size_result.wait()
    if isinstance(dataset_size, list):
        dataset_size = dataset_size[0]
    print(f"====================== Dataset 共 {dataset_size} 个任务 ======================\n")

    existing_results, finished_indices, task_indices_to_run = load_resume_state(
        output_path, dataset_size
    )
    all_results = list(existing_results)

    if not task_indices_to_run:
        print("所有任务已完成，无需再跑。")
        return

    for task_idx in task_indices_to_run:
        # 每个 task 先起 reward 的 recv，再跑 agent 的 send/recv，否则 agent send 无人收会卡死
        reward_handle = reward_group.execute_on(0).compute_reward(
            agent_worker_group_name="AndroidAgentWorkerGroup",
        )
        reward = agent_group.execute_on(0).process_task(
            task_idx=task_idx,
            reward_worker_group_name="RewardWorkerGroup",
        ).wait()
        reward_handle.wait()
        all_results.append({
            "task_idx": task_idx,
            "reward": reward,
        })
        total_run = len(all_results)
        print(f"reward: {[r['reward'] for r in all_results]}")
        successful = sum(1 for r in all_results if r.get("reward", 0)[0] > 0)
        acc = successful / total_run if total_run else 0.0
        save_checkpoint(
            output_path,
            all_results,
            {
                "timestamp": timestamp,
                "total_tasks": total_run,
                "successful_tasks": successful,
                "accuracy": round(acc, 4),
            },
        )

    total = len(all_results)
    successful = sum(1 for r in all_results if r.get("reward", 0)[0] > 0)
    accuracy = successful / total if total else 0.0
    save_checkpoint(
        output_path,
        all_results,
        {
            "timestamp": timestamp,
            "total_tasks": total,
            "successful_tasks": successful,
            "accuracy": round(accuracy, 4),
        },
    )
    print(f"结果已保存到 {output_path}")
    print(f"总任务: {total}, 成功: {successful}, 正确率: {accuracy:.2%}")


if __name__ == "__main__":
    main()