#!/bin/bash

set -ex

# Launch step 1: set your wiki_dataset pages path
WIKI2018_DIR=/your/wiki_dataset/path
pages_file=$WIKI2018_DIR/wiki_webpages.jsonl

# Launch step 2: set your retriever model path
retriever_name=e5
retriever_path=/your/retriever/model/path

# Qdrant configuration
qdrant_url=http://localhost:6333
qdrant_collection_name=wiki_collection_m24_cef512
qdrant_search_param='{"hnsw_ef":256}'

python3 -u ./local_retrieval_server_qdrant.py \
                                            --pages_path $pages_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --qdrant_collection_name $qdrant_collection_name \
                                            --qdrant_url $qdrant_url\
                                            --qdrant_search_param $qdrant_search_param\
                                            --port 8000 \


