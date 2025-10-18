## Coding RL Offline Version
### Data Preparation
Download training and test datasets from https://modelscope.cn/datasets/paxionfruit/code-fim-v2-python-filtered/


### Model Training 
Before starting training, in addition to configuring the model path and dataset path in qwen2.5-1.5b-grpo-offline.yaml, you also need to specify the LLM service information for reward scoring in run_main_coding_offline_rl.sh
For example:
```shell
export LLMASJUDGE_API_URL=${LLMASJUDGE_API_URL:-"https://cloud.infini-ai.com/maas/v1/chat/completions"}
export LLMASJUDGE_API_KEY=${LLMASJUDGE_API_KEY:-"[your api key]"}
export LLMASJUDGE_MODEL=${LLMASJUDGE_MODEL:-"deepseek-v3.1"}
```

Execute the following command to start training:
```shell
bash examples/coding_online_rl/run_main_coding_offline_rl.sh
```

### Testing
Run the model on the test set once, use the reward calculation prompt to score all results, and use the average score of all samples as the final score on the test set