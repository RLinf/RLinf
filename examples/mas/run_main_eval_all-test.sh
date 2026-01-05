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

val_names=(
    # bamboogle
    nq
    hotpotqa
    # popqa
    # triviaqa
    # 2wikimultihopqa
    # musique
)

val_paths=(
    # /mnt/public/xzxuan/data/ruc-search/bamboogle/test.jsonl
    /mnt/public/xzxuan/data/ruc-search/nq/test.jsonl
    /mnt/public/xzxuan/data/ruc-search/hotpotqa/dev.jsonl
    # /mnt/public/xzxuan/data/ruc-search/popqa/test.jsonl
    # /mnt/public/xzxuan/data/ruc-search/triviaqa/test.jsonl
    # /mnt/public/xzxuan/data/ruc-search/2wikimultihopqa/dev.jsonl
    # /mnt/public/xzxuan/data/ruc-search/musique/dev.jsonl
)


# without train

# python ${REPO_PATH}/examples/mas/main_eval.py \
#     --config-path ${CONFIG_PATH}/config/ \
#     --config-name "mas_eval_qwen2.5_7b_format_yushi" \
#     "runner.experiment_name=7b-yushi" \
#     "data.val_rollout_batch_size=143" \
#     "data.data_size=143" \
#     "runner.output_dir=/mnt/public/xzxuan/Multi-Agent/RLinf/eval/eval_qdrant/old_qdrant"    

for i in "${!val_names[@]}"; do
    val_name="${val_names[$i]}"
    val_path="${val_paths[$i]}"

    python "${REPO_PATH}/examples/mas/main_eval.py" \
        --config-path "${CONFIG_PATH}/config/" \
        --config-name "mas_eval_qwen2.5_7b" \
        "data.val_data_paths=[${val_path}]" \
        "runner.experiment_name=7b-${val_name}" \
        "data.val_rollout_batch_size=1000" \
        "data.data_size=1000" \
        "runner.output_dir=/mnt/public/xzxuan/Multi-Agent/RLinf/eval/eval_qdrant/old_qdrant"    
done




# "tools.search.server_addr=172.27.249.132:8000" \


