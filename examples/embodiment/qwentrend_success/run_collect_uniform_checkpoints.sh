#!/usr/bin/env bash
set -euo pipefail

: "${CHECKPOINT_TEMPLATE:?Set a printf template such as /path/step_%d/full_weights.pt}"
: "${OUTPUT_ROOT:?Set the collection output root}"

STEPS=${STEPS:-"0 20 40 60 80 100 120 140 160 180 200"}
NUM_ENVS=${NUM_ENVS:-1024}
SEED=${SEED:-0}

for step in ${STEPS}; do
  checkpoint=$(printf "${CHECKPOINT_TEMPLATE}" "${step}")
  run_dir="${OUTPUT_ROOT}/runs/step${step}_seed${SEED}_env${NUM_ENVS}"
  data_dir="${OUTPUT_ROOT}/step${step}"
  python examples/embodiment/train_embodied_agent.py \
    --config-name maniskill_ppo_mlp_qwentrend_collect \
    runner.only_eval=true \
    runner.ckpt_path="${checkpoint}" \
    runner.logger.log_path="${run_dir}" \
    env.eval.total_num_envs="${NUM_ENVS}" \
    env.eval.seed="${SEED}" \
    env.eval.ignore_terminations=true \
    env.eval.max_episode_steps=50 \
    env.eval.max_steps_per_rollout_epoch=50 \
    env.eval.data_collection.save_dir="${data_dir}" \
    env.eval.data_collection.only_success=false
done
