#!/bin/bash

# source /root/miniconda3/etc/profile.d/conda.sh
# conda activate

set -ex

WIKI2018_WORK_DIR=/path/to/data/Asearch

# Note: index_file is not used for Qdrant version, but kept for compatibility
corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
pages_file=$WIKI2018_WORK_DIR/wiki_webpages.jsonl

retriever_name=e5
retriever_path=/path/to/model/e5
qdrant_search_param='{"hnsw_ef":128}'

qdrant_url=http://localhost:6333

qdrant_collection_name=wiki_collection

python3  ./local_retrieval_server_qdrant.py \
                                            --pages_path $pages_file \
                                            --topk 3 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --qdrant_collection_name $qdrant_collection_name \
                                            --qdrant_url $qdrant_url\
                                            --qdrant_search_param $qdrant_search_param\
                                            --port 8000 \

