unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -m sglang.launch_server \
    --model-path /mnt/public/fuxu/instruct_qwen3_30b_a3b/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 --log-level info \
    --context-length 32768 \
    --dp 4 \