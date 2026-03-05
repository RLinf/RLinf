CKPT_PATH_MG=$1
CKPT_PATH_HF=$2
CKPT_PATH_ORIGINAL_HF=$3
CKPT_PATH_MF="$CKPT_PATH_HF"_middle_file

TP_SIZE=2
PP_SIZE=1

rm -rf $CKPT_PATH_HF
rm -rf $CKPT_PATH_MF
python -m convert_mg_to_middle_file \
    --load-path $CKPT_PATH_MG \
    --save-path $CKPT_PATH_MF \
    --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
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
    --model 'DeepSeek-R1-Distill-Qwen-1.5B' \
    --use-gpu-num 0 \
    --process-num 16

rm -rf $CKPT_PATH_MF

# copy other files to new hf folder
rm $CKPT_PATH_HF/*.done
shopt -s extglob # !() pattern only works with Bash’s extended globbing.
cp "$CKPT_PATH_ORIGINAL_HF"/!(*model.safetensors.index).json "$CKPT_PATH_HF"
