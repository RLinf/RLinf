CKPT_PATH_MG=/mnt/mnt/public/wangxiangyuan/output_1201/rstar2-grpo-1.5b-128-tp2-w-down/checkpoints/global_step_300/actor
CKPT_PATH_HF=/mnt/mnt/public/wangxiangyuan/RLinf_step_300_w_down
CKPT_PATH_ORIGINAL_HF=/mnt/mnt/public/zhuchunyang_rl/hf_models/Qwen2.5-1.5B-Instruct
CKPT_PATH_MF="$CKPT_PATH_HF"_middle_file

TP_SIZE=2
PP_SIZE=1

rm -rf $CKPT_PATH_HF
rm -rf $CKPT_PATH_MF
python -m convert_mg_to_middle_file \
    --load-path $CKPT_PATH_MG \
    --save-path $CKPT_PATH_MF \
    --model 'qwen_2.5_1.5b' \
    --tp-size $TP_SIZE \
    --ep-size 1 \
    --pp-size $PP_SIZE \
    --te-ln-linear-qkv true \
    --te-ln-linear-mlp_fc1 true \
    --te-extra-state-check-none true \
    --use-gpu-num 0 \
    --process-num 16

python -m convert_middle_file_to_hf \
    --load-path $CKPT_PATH_MF \
    --save-path $CKPT_PATH_HF \
    --model 'qwen_2.5_1.5b' \
    --use-gpu-num 0 \
    --process-num 16

rm -rf $CKPT_PATH_MF

# copy other files to new hf folder
rm $CKPT_PATH_HF/*.done
cp $CKPT_PATH_ORIGINAL_HF/*.json $CKPT_PATH_HF
