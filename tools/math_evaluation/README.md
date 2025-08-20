# Model Evaluation
## Acknowledgement
The codebase is adapted from https://github.com/QwenLM/Qwen2.5-Math. 

## Usage
You can evaluate models using the following command: 

```bash
MODEL_NAME_OR_PATH=/model/path # TODO: change to your model path
OUTPUT_DIR=${MODEL_NAME_OR_PATH}/math_eval
SPLIT="test"
NUM_TEST_SAMPLE=-1

DATA_NAME="aime24" # aime24, aime25, gpqa_diamond
PROMPT_TYPE="r1-distilled-qwen"  
# for aime24 and aime25, set PROMPT_TYPE="r1-distilled-qwen";
# for gpqa_diamond, set PROMPT_TYPE="r1-distilled-qwen-gpqa".
TOKENIZERS_PARALLELISM=false \
python3 -u math_eval.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 1.0 \
    --n_sampling 32 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --use_vllm \
    --save_outputs \
    --overwrite \
    --max_tokens_per_call 32768
```

For batch evaluation, check `main_eval.sh`. 

##  Requirements
You can install the required packages with the following command:
```bash
pip install -r requirements.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3
```
