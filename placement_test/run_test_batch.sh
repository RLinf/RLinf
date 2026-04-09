#!/bin/bash

# 1. 基础配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_PATH="$SCRIPT_DIR"
export SYNC_FLAG_FILE="$REPO_PATH/task_sync.txt"
export RANK=${RANK:-0}

# 定义要跑的任务列表：格式为 "YAML名称"
TASKS=(
    # "openvla-envnum96-env01-rollout27-actor07-pipelinestage2"
    # "openvla-envnum96-env03-rollout47-actor07-pipelinestage2"
    # "openvla-envnum96-env07-rollout07-actor07-pipelinestage1"

    # "openvla-envnum144-env01-rollout27-actor07-pipelinestage2"
    # "openvla-envnum144-env03-rollout47-actor07-pipelinestage2"
    # "openvla-envnum144-env07-rollout07-actor07-pipelinestage1"

    # "openvla-envnum160-env03-rollout47-actor07-pipelinestage2"
    # "openvla-envnum160-env07-rollout07-actor07-pipelinestage1"

    "openvla-envnum192-env01-rollout27-actor07-pipelinestage2"
    "openvla-envnum192-env03-rollout47-actor07-pipelinestage2"
    "openvla-envnum192-env07-rollout07-actor07-pipelinestage1"

    "openvla-envnum256-env03-rollout47-actor07-pipelinestage2"
    "openvla-envnum256-env07-rollout07-actor07-pipelinestage1"

    "openvla-envnum320-env03-rollout47-actor07-pipelinestage2"
    "openvla-envnum320-env07-rollout07-actor07-pipelinestage1"

    "openvla-envnum384-env01-rollout27-actor07-pipelinestage2"
    "openvla-envnum384-env03-rollout47-actor07-pipelinestage2"
    "openvla-envnum384-env07-rollout07-actor07-pipelinestage1"

    "openvla-envnum512-env03-rollout47-actor07-pipelinestage2"
    "openvla-envnum512-env07-rollout07-actor07-pipelinestage1"
)

# 统一清理函数：确保每轮任务开始前显存和进程完全干净
function super_cleanup() {
    echo "[$(date +%T)] 清理残留进程 (Ray/Python)..."
    command -v ray >/dev/null 2>&1 && ray stop --force
    pkill -9 -u $(whoami) python >/dev/null 2>&1
    pkill -9 -u $(whoami) ray >/dev/null 2>&1
    rm -rf /dev/shm/ray/* 2>/dev/null
    sleep 5
}

# ---------------- RANK 0 (主节点) 逻辑 ----------------
if [ "$RANK" -eq 0 ]; then
    rm -f "$SYNC_FLAG_FILE"
    
    for TASK_STR in "${TASKS[@]}"; do
        read -r YAML_ARG PLATFORM <<< "$TASK_STR"
        
        echo ">>>>>>> 开始执行任务: $YAML_ARG (平台: $PLATFORM) <<<<<<<"
        
        # 1. 清理并发送信号给 Worker
        super_cleanup
        echo "$YAML_ARG" > "$SYNC_FLAG_FILE"
        sleep 2 

        # 2. 启动本节点的 Ray Head
        bash ray_utils/start_ray.sh
        
        # 3. 调用你的单次启动脚本执行训练
        # 假设你的脚本名为 run_single.sh
        bash placement_test/run_placement_test.sh "$YAML_ARG" || echo "Task $YAML_ARG FAILED" >> error_tasks.log
        
        # 4. 任务结束，删除信号通知 Worker 清理
        rm -f "$SYNC_FLAG_FILE"
        super_cleanup
        echo "任务 $YAML_ARG 已完成，等待 10s 进入下一项..."
        sleep 10
    done
    echo "所有 YAML 任务执行完毕！"

# ---------------- WORKER (从节点) 逻辑 ----------------
else
    LAST_YAML=""
    echo "Worker 节点已就绪，等待 Rank 0 信号..."

    while true; do
        if [ ! -f "$SYNC_FLAG_FILE" ]; then
            LAST_YAML="" 
            sleep 5
            continue
        fi

        CURRENT_YAML=$(cat "$SYNC_FLAG_FILE" | tr -d '[:space:]')
        
        if [ -n "$CURRENT_YAML" ] && [ "$CURRENT_YAML" != "$LAST_YAML" ]; then
            echo "[$(date +%T)] 接收到新任务信号: $CURRENT_YAML，正在同步启动..."
            
            super_cleanup
            # 加入主节点创建的集群
            bash ray_utils/start_ray.sh
            
            LAST_YAML="$CURRENT_YAML"
            echo "Worker 已加入 Ray 集群，训练进行中..."
        fi
        sleep 5
    done
fi