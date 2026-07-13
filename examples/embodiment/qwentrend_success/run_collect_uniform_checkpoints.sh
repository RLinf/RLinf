#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_TEMPLATE_EARLY:?Set the step20-120 printf checkpoint template}"
: "${CHECKPOINT_TEMPLATE_LATE:?Set the step140-200 printf checkpoint template}"
: "${OUTPUT_ROOT:?Set the collection output root}"

STEPS=${STEPS:-"0 20 40 60 80 100 120 140 160 180 200"}
NUM_ENVS=${NUM_ENVS:-1024}
SEED=${SEED:-0}
CUDA_DEVICES=${CUDA_DEVICES:-0,1,2,3}
PLACEMENT=${PLACEMENT:-0-3}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}

for step in ${STEPS}; do
  if ((step == 0)); then
    checkpoint=null
  elif ((step <= 120)); then
    checkpoint=$(printf "${CHECKPOINT_TEMPLATE_EARLY}" "${step}")
  else
    checkpoint=$(printf "${CHECKPOINT_TEMPLATE_LATE}" "${step}")
  fi
  run_dir="${OUTPUT_ROOT}/runs/step${step}_seed${SEED}_env${NUM_ENVS}"
  data_dir="${OUTPUT_ROOT}/step${step}"
  ray_dir="/dev/shm/qc_${step}_$$"
  mkdir -p "${run_dir}" "${data_dir}" "${ray_dir}/ray" "${ray_dir}/tmp"
  CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" \
    RAY_TMPDIR="${ray_dir}/ray" \
    RLINF_FORCE_LOCAL_RAY=1 \
    RLINF_RAY_TEMP_DIR="${ray_dir}/ray" \
    TMPDIR="${ray_dir}/tmp" \
    EMBODIED_PATH="${PWD}" \
    "${PYTHON_BIN}" evaluations/eval_embodied_agent.py \
    --config-path ../examples/embodiment/config \
    --config-name maniskill_ppo_mlp_qwentrend_collect \
    runner.only_eval=true \
    runner.ckpt_path="${checkpoint}" \
    runner.logger.log_path="${run_dir}" \
    runner.logger.experiment_name="step${step}_seed${SEED}_env${NUM_ENVS}" \
    cluster.component_placement.env="${PLACEMENT}" \
    cluster.component_placement.rollout="${PLACEMENT}" \
    'rollout.model=${actor.model}' \
    rollout.enable_torch_compile=false \
    rollout.enable_cuda_graph=false \
    env.eval.total_num_envs="${NUM_ENVS}" \
    env.eval.seed="${SEED}" \
    env.eval.wrap_obs_mode=simple \
    env.eval.ignore_terminations=true \
    env.eval.max_episode_steps=50 \
    env.eval.max_steps_per_rollout_epoch=50 \
    env.eval.data_collection.enabled=true \
    env.eval.data_collection.save_dir="${data_dir}" \
    env.eval.data_collection.only_success=false
done
