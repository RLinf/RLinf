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
qdrant_path=/mnt/public/xzxuan/qdrant_m32_cef512-cp
qdrant_url=http://localhost:6333
qdrant_collection_name=wiki_collection_m32_cef512
qdrant_search_param='{"hnsw_ef":256}'


# cd $qdrant_path
# ./qdrant &
# qdrant_pid=$!
 
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3  examples/search_engine/local_retrieval_server_qdrant_best_async.py \
                                            --pages_path $pages_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --qdrant_collection_name $qdrant_collection_name \
                                            --qdrant_url $qdrant_url\
                                            --qdrant_search_param $qdrant_search_param\
                                            --port 8000 \


# echo "wait for qdrant stop"
# kill $qdrant_pid
# wait $qdrant_pid 2>/dev/null
# echo "qdrant is stopped"

