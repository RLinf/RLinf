
#!/bin/bash
EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
# REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
SRC_FILE="${EMBODIED_PATH}/pre_collect_robotwin2_seed.py"

export ROBOTWIN_PATH="/mnt/public/guozhen/test_robotwin/RLinf_RoboTwin"
export PYTHONPATH=${ROBOTWIN_PATH}:$PYTHONPATH

export ASSETS_PATH="/mnt/public/guozhen/test_robotwin/robotwin_assets"

DATASET_NAME="place_empty_cup"
#start collect seed

python ${SRC_FILE} --tasks $DATASET_NAME --seed-start  100000  --seed-end  200000  --target-count 100  --num-gpus 4  --data-split train
#python pre_collect_robotwin2_seed.py --tasks $DATASET_NAME --seed-start  100000000  --seed-end  100100000  --target-count 160  --num-gpus 8  --data-split eval
# collect seed end 