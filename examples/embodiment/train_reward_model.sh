python examples/embodiment/train_reward_model.py \
    --positive-dir /mnt/mnt/public/fangzhirui/feat_sac/RLinf/toolkits/reward_model/data/reward/positive \
    --negative-dir /mnt/mnt/public/fangzhirui/feat_sac/RLinf/toolkits/reward_model/data/reward/negative \
    --output-checkpoint /mnt/mnt/public/fangzhirui/feat_sac/RLinf/toolkits/reward_model/checkpoints/reward_model.pt \
    --backbone resnet10 \
    --image-key base_camera \
    --batch-size 128 \
    --epochs 30