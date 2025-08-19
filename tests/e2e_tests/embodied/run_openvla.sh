#! /bin/bash
set -x

tabs 4

source /opt/conda/etc/profile.d/conda.sh
conda activate vla
apt update && apt-get install -y libglew-dev libegl1 libvulkan1 vulkan-tools
cp ${REPO_PATH}/tests/e2e_tests/embodied/env_jsons/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
cp ${REPO_PATH}/tests/e2e_tests/embodied/env_jsons/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json
cp ${REPO_PATH}/tests/e2e_tests/embodied/env_jsons/nvidia_layers.json /etc/vulkan/implicit_layer.d/nvidia_layers.json
python -m mani_skill.examples.demo_random_action

export PYTHONPATH=${REPO_PATH}:${REPO_PATH}/megatron:$PYTHONPATH
unset HOME # GitHub action sets HOME to a wrong path (/github/home), breaking simulator

cd /workspace/dataset/repos/openvla-main
pip install -e .
cd -

python ${REPO_PATH}/examples/embodiment/train_embodied_agent.py --config-path ${REPO_PATH}/tests/e2e_tests/embodied --config-name ppo_openvla