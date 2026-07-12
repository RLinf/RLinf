#!/usr/bin/env bash
set -euo pipefail

: "$QWEN_MODEL_PATH"
: "$QWENTREND_SUCCESS_CHECKPOINT"

python examples/embodiment/train_embodied_agent.py \
  --config-name maniskill_ppo_mlp_qwentrend_reward \
  env.train.ignore_terminations=true \
  env.train.max_episode_steps=50 \
  env.train.max_steps_per_rollout_epoch=50 \
  env.eval.ignore_terminations=true \
  env.eval.max_episode_steps=50 \
  env.eval.max_steps_per_rollout_epoch=50 \
  algorithm.update_epoch=2 \
  algorithm.kl_beta=0.01 \
  algorithm.clip_ratio_high=0.1 \
  algorithm.clip_ratio_low=0.1 \
  actor.optim.lr=1.0e-5 \
  actor.optim.value_lr=3.0e-5 \
  reward.history_reward_assign=false \
  reward.env_reward_weight=0.0 \
  reward.model.model_path="$QWEN_MODEL_PATH" \
  reward.model.lora_path="$QWENTREND_SUCCESS_CHECKPOINT" \
  reward.model.gt_success_bonus=0.0 \
  reward.model.input_builder_name=qwentrend_terminal_success_input_builder \
  '++reward.model.input_builder_params={default_task_description: "Pick up the red cube and place it on the green spot on the table.", include_task: true}' \
  reward.model.reward_parser_name=qwentrend_binary_digit_reward_parser \
  '++reward.model.reward_parser_params={positive_reward: 1.0, negative_reward: 0.0, invalid_reward: 0.0}' \
  reward.model.history_buffers.history_window.input_interval=5 \
  reward.model.max_new_tokens=3 \
  reward.model.do_sample=false \
  reward.model.temperature=0.0 \
  "$@"
