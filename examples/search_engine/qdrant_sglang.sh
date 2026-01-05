#!/bin/bash

set -ex

WIKI2018_WORK_DIR=/mnt/public/xzxuan/data/Asearch

# Pages file
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl

# Retriever configuration
retriever_name=e5
retriever_path=/mnt/public/xzxuan/model/e5

# Qdrant configuration
qdrant_path=/mnt/public/xzxuan/qdrant_m32_cef512
qdrant_url=http://localhost:6333
qdrant_collection_name=wiki_collection_m32_cef512
qdrant_search_param='{"hnsw_ef":256}'

# SGLang configuration
SGLANG_PORT=${SGLANG_PORT:-30000}
TP_SIZE=${TP_SIZE:-1}  # Tensor Parallel size
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}  # Which GPUs to use
FASTAPI_PORT=${FASTAPI_PORT:-8002}  # FastAPI server port

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

echo "========================================="
echo "Starting SGLang + Qdrant Retrieval Server"
echo "========================================="
echo "TP_SIZE: $TP_SIZE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SGLANG_PORT: $SGLANG_PORT"
echo "FASTAPI_PORT: $FASTAPI_PORT"
echo "Qdrant URL: $qdrant_url"
echo "Collection: $qdrant_collection_name"
echo "========================================="

# Step 1: Start Qdrant server (if not already running)
# echo "[STEP 1] Starting Qdrant server..."
# cd $qdrant_path
# ./qdrant &
# qdrant_pid=$!
# echo "Qdrant started with PID: $qdrant_pid"
# sleep 30

# Step 2: Start SGLang embedding server
# echo "[STEP 2] Starting SGLang embedding server..."


# python3 -m sglang.launch_server \
#     --model-path $retriever_path \
#     --is-embedding \
#     --tp-size $TP_SIZE \
#     --host 0.0.0.0 \
#     --port $SGLANG_PORT \
#     --trust-remote-code &

# python3 -m sglang.launch_server \
#   --model-path "$retriever_path" \
#   --served-model-name "$MODEL_NAME" \
#   --is-embedding \
#   --context-length 256 \
#   --host 0.0.0.0 \
#   --port "$SGLANG_PORT" \
#   --trust-remote-code \
#   --dtype float16 &

# sglang_pid=$!
# echo "SGLang server started with PID: $sglang_pid"
# sleep 60  # Wait for SGLang to initialize

# Step 3: Start FastAPI retrieval server
echo "[STEP 3] Starting FastAPI retrieval server..."
cd /mnt/public/xzxuan/Multi-Agent/RLinf/examples/search_engine

python3 ./local_retrieval_server_qdrant_sglang.py \
    --pages_path $pages_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --qdrant_collection_name $qdrant_collection_name \
    --qdrant_url $qdrant_url \
    --qdrant_search_param "$qdrant_search_param" \
    --port $FASTAPI_PORT \
    --sglang_server_url "http://127.0.0.1:$SGLANG_PORT"

# # Cleanup when FastAPI exits
# echo "[CLEANUP] Stopping services..."
# kill $sglang_pid $qdrant_pid
# wait $sglang_pid $qdrant_pid 2>/dev/null
# echo "All services stopped."
