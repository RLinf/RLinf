#!/bin/bash

set -ex

# Launch step 1: set your wiki_dataset pages path
# WIKI2018_DIR=/your/wiki_dataset/path
pages_file=/mnt/public/xzxuan/data/Wiki-2018-Corpus/wiki_webpages.jsonl

# Launch step 2: set your retriever model path
retriever_name=e5
retriever_path=/mnt/public/xzxuan/models/e5-base-v2

# Qdrant configuration
qdrant_url=http://localhost:6333
qdrant_collection_name=wiki_collection_m32_cef512
qdrant_search_param='{"hnsw_ef":256}'

CONFIG_PATH="$( realpath "$( dirname "${BASH_SOURCE[0]}" )"  )"
python3 -u ${CONFIG_PATH}/local_retrieval_server.py \
    --pages_path $pages_file \
    --topk 3 \
    --retriever_name $retriever_name \
    --retriever_model $retriever_path \
    --qdrant_collection_name $qdrant_collection_name \
    --qdrant_url $qdrant_url\
    --qdrant_search_param $qdrant_search_param\
    --port 8000
