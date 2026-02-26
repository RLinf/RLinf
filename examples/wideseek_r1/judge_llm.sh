unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3 -m sglang.launch_server \
    --model-path /mnt/project_rlinf/xzxuan/model/Qwen3-30B-A3B-Instruct-2507 \
    --host 0.0.0.0 --log-level info \
    --context-length 32768 \
    --dp 8 \
    