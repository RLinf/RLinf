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

🌟🌟 **Note:** 🌟🌟 Ensure that `num_steps` and `action_chunk` during evaluation strictly align with the RL training configuration. While Flow Matching policies in SFT allow for flexible num_steps due to continuous-time training, RL uses a fixed denoise step. **Consequently, when evaluating RL-tuned models, `num_steps` must remain consistent with the training config.**

### LIBERO (`libero_eval.py`)
For LIBERO, the `num_steps` and `action_chunk` parameters depend on the model and task suite; they correspond to **Denoise step** and **Action replan horizon** in our paper. Please set them in the command below.

![](https://github.com/RLinf/misc/blob/main/pic/hyper_parameters_pi_libero.png)

**Example: pi0 model**
```bash 
python toolkits/eval_scripts_openpi/libero_eval.py \
    --exp_name libero_spatial_pi0 \
    --config_name pi0_libero \
    --pretrained_path your_model_path/ \
    --task_suite_name libero_spatial \
    --num_trials_per_task 50 \
    --action_chunk $action_chunk \
    --num_steps $num_steps \
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
    --action_chunk $action_chunk \
    --num_steps $num_steps \
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

