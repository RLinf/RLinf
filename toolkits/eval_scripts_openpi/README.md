# OpenPI Evaluation Scripts

This directory contains evaluation scripts for OpenPI models. Compared to the RLinf evaluation script, this script provides more granular metrics, such as success rates across different difficulty levels in MetaWorld or task completion statistics for varying sequence lengths in CALVIN. 

However, this script is slower due to its single-threaded execution, taking approximately 2 hours on a single H100 GPU.

## Environment Setup

**IMPORTANT**: Before running any evaluation script, you must export the required environment variables:

```bash
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH
```

## Evaluation Scripts
### Parameters

All evaluation scripts share these common parameters:
- `--exp_name`: Experiment name for logging
- `--config_name`: Configuration name (e.g., `pi0_libero`, `pi05_calvin`)
- `--pretrained_path`: Path to the pretrained model checkpoint
- `--action_chunk`: Action chunk
- `--num_steps`: Denoise step for flow policy
- `--num_save_videos`: Number of videos to save
- `--video_temp_subsample`: Temporal subsampling rate for videos

**Note:** Ensure that `num_steps` and `action_chunk` during evaluation strictly align with the RL training configuration. While Flow Matching policies in SFT allow for flexible num_steps due to continuous-time training, RL uses a fixed denoise step. Consequently, when evaluating RL-tuned models, `num_steps` must remain consistent with the training config.

For full eval scripts, please refer to `run.sh`.

### LIBERO (`libero_eval.py`)

**Example: pi0 model**
```bash
python toolkits/eval_scripts_openpi/libero_eval.py \
    --exp_name libero_spatial_pi0 \
    --config_name pi0_libero \
    --pretrained_path your_model_path/ \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10
```

**Example: pi05 model**
```bash
python toolkits/eval_scripts_openpi/libero_eval.py \
    --exp_name libero_spatial_pi05 \
    --config_name pi05_libero \
    --pretrained_path your_model_path/ \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10
```

### MetaWorld (`metaworld_eval.py`)

**Example: pi0 model**
```bash
python toolkits/eval_scripts_openpi/metaworld_eval.py \
    --exp_name metaworld_pi0 \
    --config_name pi0_metaworld \
    --pretrained_path your_model_path/ \
    --num_trials_per_task 10 \
    --max_steps 160 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10
```

**Example: pi05 model**
```bash
python toolkits/eval_scripts_openpi/metaworld_eval.py \
    --exp_name metaworld_pi05 \
    --config_name pi05_metaworld \
    --pretrained_path your_model_path/ \
    --num_trials_per_task 10 \
    --max_steps 160 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10
```

### CALVIN (`calvin_eval.py`)

**Example: pi0 model**
```bash
python toolkits/eval_scripts_openpi/calvin_eval.py \
    --exp_name calvin_pi0 \
    --config_name pi0_calvin \
    --pretrained_path your_model_path/ \
    --num_trials 1000 \
    --max_steps 480 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10
```

**Example: pi05 model**
```bash
python toolkits/eval_scripts_openpi/calvin_eval.py \
    --exp_name calvin_pi05 \
    --config_name pi05_calvin \
    --pretrained_path your_model_path/ \
    --num_trials 1000 \
    --max_steps 480 \
    --action_chunk 5 \
    --num_steps 5 \
    --num_save_videos 10 \
    --video_temp_subsample 10
```

### BEHAVIOR (`behavior_eval.py`)

This script evaluates models on the BEHAVIOR benchmark, providing detailed per-instance metrics. It supports both RLinf-trained models and OpenPI-style models.

**Key Features:**
- Single-threaded evaluation with detailed metrics
- Per-instance success rates and episode statistics
- Video recording support
- Compatible with both RLinf and OpenPI policies

**Example: RLinf model (OpenVLA-OFT)**
```bash
# Using the shell script (recommended)
bash toolkits/eval_scripts_openpi/eval_behavior.sh \
    --task_name turning_on_radio \
    --model_path /path/to/model/RLinf-OpenVLAOFT-Behavior/ \
    --policy_type rlinf \
    --action_chunk 32 \
    --max_steps 2000 \
    --num_episodes 1 \
    --log_path ./behavior_eval_results \
    --num_save_videos 10

# Or directly with Python
python toolkits/eval_scripts_openpi/behavior_eval.py \
    --task_name turning_on_radio \
    --pretrained_path /path/to/model/RLinf-OpenVLAOFT-Behavior/ \
    --policy_type rlinf \
    --action_chunk 32 \
    --max_steps 2000 \
    --num_episodes_per_instance 1 \
    --log_path ./behavior_eval_results \
    --num_save_videos 10
```

**Example: OpenPI model (Pi0.5)**
```bash
python toolkits/eval_scripts_openpi/behavior_eval.py \
    --task_name turning_on_radio \
    --pretrained_path /path/to/pi05-turning_on_radio-sft/ \
    --policy_type openpi \
    --config_name pi05_b1k-turning_on_radio_lr2.5e-6_step20k_sft \
    --control_mode receeding_horizon \
    --max_len 32 \
    --action_chunk 5 \
    --max_steps 2000 \
    --log_path ./behavior_eval_results
```

**Available BEHAVIOR Tasks:**
The BEHAVIOR benchmark includes 50 household manipulation tasks. Common tasks include:
- `turning_on_radio` - Turn on a radio
- `opening_packages` - Open packages
- `packing_grocery_items_into_bags` - Pack groceries into bags
- `cleaning_windows` - Clean windows
- `setting_table` - Set a table for dining
- And 45 more...

**Evaluation Options:**
- `--eval_instance_ids`: Specify which test instances to evaluate (default: all 10 instances)
- `--num_episodes_per_instance`: Number of evaluation episodes per instance (default: 1)
- `--num_save_videos`: Number of evaluation videos to save (default: 10)

**For Parallel Evaluation:**
For faster evaluation using RLinf's parallel worker architecture, use the main evaluation script:
```bash
python examples/embodiment/eval_embodied_agent.py \
    --config-name behavior_openvlaoft_eval \
    runner.eval_policy_path=/path/to/model/model.pt
```

