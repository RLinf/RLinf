#!/bin/bash

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate

set -ex

WIKI2018_WORK_DIR=/path/to/wiki_data

# Note: index_file is not used for Qdrant version, but kept for compatibility
corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl
# corpus_file=/mnt/public/zhuchunyang_rl/hf_datasets/Asearch/wiki_corpus.jsonl
# pages_file=/mnt/public/zhuchunyang_rl/hf_datasets/Asearch/wiki_webpages.jsonl
# corpus_file=/mnt/public/zhuchunyang_rl/hf_datasets/Asearch/wiki_corpus_100.jsonl
# pages_file=/mnt/public/zhuchunyang_rl/hf_datasets/Asearch/wiki_webpages_100.jsonl
retriever_name=e5
retriever_path=/path/to/retriever_model
qdrant_search_param='{"hnsw_ef":128}'
# Qdrant configuration
# Option 1: Use remote Qdrant server (uncomment the line below and comment out qdrant_path)
# cd /mnt/public/zhuchunyang_rl/qdrant/ && qdrant
qdrant_url=http://localhost:6333

# Option 2: Use local Qdrant storage (default)
# qdrant_path=/mnt/public/xzxuan/data/Asearch/qdrant_storage

qdrant_collection_name=wiki_collection

python3  ./local_retrieval_server_qdrant_5.py \
                                            --pages_path $pages_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --qdrant_collection_name $qdrant_collection_name \
                                            --qdrant_url $qdrant_url\
                                            --qdrant_search_param $qdrant_search_param\
                                            --port 8000 \
                                            # ${qdrant_path:+--qdrant_path $qdrant_path}

