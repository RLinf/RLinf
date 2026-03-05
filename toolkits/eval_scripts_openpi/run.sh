## --------------- Prepare Environment ---------------
export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

## --------------- LIBERO ---------------
# pi0
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

# pi05
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

## --------------- METAWORLD ---------------
# pi0
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

# pi05
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

## --------------- CALVIN ---------------
# pi0
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

# pi05
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

