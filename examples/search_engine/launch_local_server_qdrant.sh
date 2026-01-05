#!/bin/bash

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate

set -ex

WIKI2018_WORK_DIR=/mnt/public/xzxuan/data/Asearch

corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl

retriever_name=e5
retriever_path=/mnt/public/xzxuan/model/e5

# Qdrant configuration
# Option 1: Use remote Qdrant server (uncomment the line below and comment out qdrant_path)
# cd /mnt/project_rlinf/zhuchunyang_rl/qdrant/ && qdrant
qdrant_url=http://127.0.0.1:6333

# Option 2: Use local Qdrant storage (default)
# qdrant_path=/mnt/public/xzxuan/data/Asearch/qdrant_storage

qdrant_collection_name=wiki_collection

unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

python3  examples/search_engine/local_retrieval_server_qdrant.py --corpus_path $corpus_file \
                                            --pages_path $pages_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --qdrant_collection_name $qdrant_collection_name \
                                            --port 8000 \
                                            --save-address-to /mnt/public/liuweilin/local_server \
                                            ${qdrant_url:+--qdrant_url $qdrant_url} \
                                            # ${qdrant_path:+--qdrant_path $qdrant_path}

