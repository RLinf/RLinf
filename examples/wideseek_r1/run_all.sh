#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export RAY_DEBUG=1

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:${REPO_PATH}/examples:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="eval_qwen3_qa"
    # qwen2.5-3b-searchr1_eval
else
    CONFIG_NAME=$1
fi


val_names=(
    bamboogle
    hotpotqa
    nq
    popqa
    triviaqa
    2wikimultihopqa
    musique
)

val_paths=(
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/Bamboogle/test.jsonl
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/HotpotQA_rand1000/test.jsonl
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/NQ_rand1000/test.jsonl
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/PopQA_rand1000/test.jsonl
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/TriviaQA_rand1000/test.jsonl
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/2WikiMultihopQA_rand1000/test.jsonl
    /mnt/project_rlinf/xzxuan/data/Asearcher-test-data/Musique_rand1000/test.jsonl
)


for i in "${!val_names[@]}"; do
    val_name="${val_names[$i]}"
    val_path="${val_paths[$i]}"

    python "${REPO_PATH}/examples/mas/main_eval.py" \
        --config-path "${CONFIG_PATH}/config/" \
        --config-name "eval_qwen3_qa" \
        "cluster.num_nodes=1" \
        "data.val_data_paths=[${val_path}]" \
        "runner.experiment_name=${val_name}" \
        "runner.output_dir=/mnt/project_rlinf/xzxuan/Multiagent/eval/qa" \
        "agentloop.workflow=mas" \
        "rollout.model.model_path=/mnt/project_rlinf/xzxuan/wideseek_model/mas_hybrid_final"
        
done