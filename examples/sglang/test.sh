export CUDA_VISIBLE_DEVICES=2,3
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -m sglang.launch_server \
    --model-path /mnt/public/hf_models/Qwen2.5-3B-Instruct-search \
    --host localhost --log-level info \
    --mem-fraction-static 0.5 \
    --tp 2 \