export PYTHONPATH=/mnt/mnt/public/fangzhirui/feat_sac/RLinf:$PYTHONPATH
export EMBODIED_PATH=/mnt/mnt/public/fangzhirui/feat_sac/RLinf/examples/embodiment

python -m rlinf.models.reward_model.train_reward_model \
    --positive-dir /mnt/mnt/public/fangzhirui/feat_reward/RLinf/reward_data/positive \
    --negative-dir /mnt/mnt/public/fangzhirui/feat_reward/RLinf/reward_data/negative \
    --output-checkpoint /mnt/mnt/public/fangzhirui/feat_reward/RLinf/toolkits/reward_model/checkpoints/reward_model.pt \
    --backbone resnet10 \
    --image-key base_camera \
    --batch-size 128 \
    --epochs 30 \
    --visualize-positive \
    --vis-output-dir /mnt/mnt/public/fangzhirui/feat_reward/RLinf/examples/embodiment/show