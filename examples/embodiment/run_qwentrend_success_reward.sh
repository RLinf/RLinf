#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export EMBODIED_PATH=${EMBODIED_PATH:-${SCRIPT_DIR}}

: "${QWEN_MODEL_PATH:?Set the Qwen3-VL base model path}"
: "${QWENTREND_POTENTIAL_CHECKPOINT:?Set the potential Qwen checkpoint}"
: "${QWENTREND_SCALAR_HEAD:?Set the scalar potential head checkpoint}"
: "${QWENTREND_SUCCESS_CHECKPOINT:?Set the selected Qwen success checkpoint}"
: "${POLICY_CHECKPOINT:?Set the starting policy full_weights.pt path}"
: "${PPO_OUTPUT_ROOT:?Set the PPO output directory}"

PLACEMENT=${PLACEMENT:-0-3}
NUM_ENVS=${NUM_ENVS:-128}
MAX_STEPS=${MAX_STEPS:-160}
INFER_BATCH_SIZE=${INFER_BATCH_SIZE:-32}
RESUME_DIR=${RESUME_DIR:-null}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwentrend_success_ppo}
PYTHON_BIN=${PYTHON_BIN:-/opt/venv/openvla/bin/python}
RAY_TMPDIR=${RAY_TMPDIR:-/dev/shm/qppo_$$}
export RAY_TMPDIR

"${PYTHON_BIN}" examples/embodiment/train_embodied_agent.py \
  --config-name maniskill_ppo_mlp_qwentrend_reward \
  cluster.component_placement.actor="${PLACEMENT}" \
  cluster.component_placement.env="${PLACEMENT}" \
  cluster.component_placement.rollout="${PLACEMENT}" \
  cluster.component_placement.reward="${PLACEMENT}" \
  "++cluster.component_placement.reward_server=0-3:0" \
  runner.ckpt_path="${POLICY_CHECKPOINT}" \
  runner.max_steps="${MAX_STEPS}" \
  runner.resume_dir="${RESUME_DIR}" \
  runner.val_check_interval=5 \
  runner.max_epochs="${MAX_STEPS}" \
  runner.save_interval=20 \
  runner.logger.log_path="${PPO_OUTPUT_ROOT}" \
  runner.logger.experiment_name="${EXPERIMENT_NAME}" \
  env.train.total_num_envs="${NUM_ENVS}" \
  env.train.group_size=1 \
  env.train.wrap_obs_mode=simple \
  env.train.reward_mode=only_success \
  env.train.ignore_terminations=true \
  env.train.max_episode_steps=50 \
  env.train.max_steps_per_rollout_epoch=50 \
  env.eval.total_num_envs="${NUM_ENVS}" \
  env.eval.group_size=1 \
  env.eval.wrap_obs_mode=simple \
  env.eval.reward_mode=only_success \
  env.eval.ignore_terminations=true \
  env.eval.max_episode_steps=50 \
  env.eval.max_steps_per_rollout_epoch=50 \
  data.rollout_batch_size="${NUM_ENVS}" \
  actor.micro_batch_size=1600 \
  actor.global_batch_size=6400 \
  "rollout.model=\${actor.model}" \
  rollout.enable_torch_compile=false \
  rollout.enable_cuda_graph=false \
  algorithm.update_epoch=2 \
  algorithm.normalize_advantages=true \
  algorithm.kl_beta=0.01 \
  algorithm.clip_ratio_high=0.1 \
  algorithm.clip_ratio_low=0.1 \
  actor.optim.lr=1.0e-5 \
  actor.optim.value_lr=3.0e-5 \
  reward.use_reward_model=true \
  reward.reward_weight=1.0 \
  reward.env_reward_weight=0.0 \
  reward.reward_mode=history_buffer \
  reward.history_reward_assign=false \
  reward.model.model_path="${QWEN_MODEL_PATH}" \
  reward.model.lora_path="${QWENTREND_POTENTIAL_CHECKPOINT}" \
  reward.model.inference_mode=scalar_head \
  +reward.model.scalar_head_path="${QWENTREND_SCALAR_HEAD}" \
  reward.model.input_builder_name=qwentrend_potential_input_builder \
  '++reward.model.input_builder_params={default_task_description: "Pick up the red cube and place it on the green spot on the table.", include_task: true, num_bins: 10}' \
  reward.model.history_buffers.history_window.history_size=5 \
  reward.model.history_buffers.history_window.min_history_size=5 \
  reward.model.history_buffers.history_window.input_interval=5 \
  reward.model.history_buffers.history_window.input_on_done=true \
  +reward.model.history_buffers.history_window.input_on_done_full_window=true \
  reward.model.infer_micro_batch_size="${INFER_BATCH_SIZE}" \
  +reward.model.potential_scale=1.0 \
  +reward.model.potential_gamma=1.0 \
  +reward.model.potential_ema_alpha=0.5 \
  +reward.model.potential_clip=0.15 \
  +reward.model.success_threshold=0.5 \
  +reward.model.success_confirmation_windows=1 \
  +reward.model.success_bonus=1.0 \
  +reward.model.success_lora_path="${QWENTREND_SUCCESS_CHECKPOINT}" \
  +reward.model.success_input_builder_name=qwentrend_terminal_success_input_builder \
  '++reward.model.success_input_builder_params={default_task_description: "Pick up the red cube and place it on the green spot on the table.", include_task: true}' \
  +reward.model.success_reward_parser_name=qwentrend_binary_digit_reward_parser \
  '++reward.model.success_reward_parser_params={positive_reward: 1.0, negative_reward: 0.0, invalid_reward: 0.0}' \
  reward.model.gt_success_bonus=0.0 \
  "$@"
