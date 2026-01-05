#!/bin/bash

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate

set -ex

WIKI2018_WORK_DIR=/mnt/public/xzxuan/data/Asearch

# Note: index_file is not used for Qdrant version, but kept for compatibility
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
# pages_file=/mnt/project_rlinf/zhuchunyang_rl/hf_datasets/Asearch/wiki_webpages.jsonl
# pages_file=/mnt/project_rlinf/zhuchunyang_rl/hf_datasets/Asearch/wiki_webpages_100.jsonl
retriever_name=e5
retriever_path=/mnt/public/xzxuan/model/e5

# Qdrant configuration
# Option 1: Use remote Qdrant server (uncomment the line below and comment out qdrant_path)
# cd /mnt/project_rlinf/zhuchunyang_rl/qdrant/ && qdrant
qdrant_path=/mnt/public/xzxuan/qdrant_m32_cef512
qdrant_url=http://172.27.249.164:6333
qdrant_collection_name=wiki_collection_m32_cef512
qdrant_search_param='{"hnsw_ef":256}'


# cd $qdrant_path
# ./qdrant &
# qdrant_pid=$!
 
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

cd /mnt/public/xzxuan/Multi-Agent/RLinf/examples/search_engine

# Clean up worker counter from previous runs
rm -f /tmp/retrieval_worker_counter.txt

# Use Gunicorn with multiple Uvicorn workers for better concurrency
# Workers = 16 will be distributed across available GPUs
# For example: 16 workers on 8 GPUs = 2 workers per GPU
export PAGES_PATH=$pages_file
export TOPK=3
export RETRIEVER_NAME=$retriever_name
export RETRIEVER_MODEL=$retriever_path
export QDRANT_COLLECTION_NAME=$qdrant_collection_name
export QDRANT_URL=$qdrant_url
export QDRANT_SEARCH_PARAM=$qdrant_search_param
export PORT=8000

# Install gunicorn if not installed: pip install gunicorn
gunicorn local_retrieval_server_qdrant_best:app \
    --workers 16 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 300 \
    --keep-alive 75 \
    --max-requests 10000 \
    --max-requests-jitter 1000 \
    --access-logfile - \
    --error-logfile - \
    --log-level info


echo "wait for qdrant stop"
kill $qdrant_pid
wait $qdrant_pid 2>/dev/null
echo "qdrant is stopped"

