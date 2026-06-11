#! /bin/bash
set -x

tabs 4
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export HCCL_INTRA_ROCE_ENABLE=1

CONFIG_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_PATH=$(dirname $(dirname "$CONFIG_PATH"))
MEGATRON_PATH=/opt/Megatron-LM
export PYTHONPATH=${REPO_PATH}:${MEGATRON_PATH}:$PYTHONPATH

#支持NPU单卡多进程
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
#规避ray在device侧调用无法根据is_npu_available接口识别设备可用性
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
#根据当前设备和需要卡数定义
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15

if [ -z "$1" ]; then
    CONFIG_NAME="qwen2.5-1.5b-grpo-megatron"
else
    CONFIG_NAME=$1
fi

# ASCEND_LAUNCH_BLOCKING=1 \
python ${REPO_PATH}/examples/reasoning/main_grpo.py --config-path ${CONFIG_PATH}/config/math/  --config-name $CONFIG_NAME