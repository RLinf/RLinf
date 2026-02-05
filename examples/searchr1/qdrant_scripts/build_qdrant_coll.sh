#!/bin/bash

set -ex

# Build step 1: set your wiki_dataset corpus path
WIKI2018_DIR=/your/wiki_dataset/path
corpus_file=$WIKI2018_DIR/wiki_corpus.jsonl

# Build step 2: set your retriever model path
retriever_name=e5
retriever_path=/your/retriever/model/path

# Qdrant configuration
# Build step 3: Install qdrant and set qdrant_path to qdrant dir
qdrant_path=/your/qdrant/path
qdrant_url=http://localhost:6333
qdrant_collection_name=wiki_collection_m24_cef512
hnsw_config='{"m":24,"ef_construct":512}'


"$qdrant_path/qdrant" &
qdrant_pid=$!
sleep 5

python3  ./build_qdrant_coll.py --corpus_path $corpus_file \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --qdrant_collection_name $qdrant_collection_name \
                                            --qdrant_url $qdrant_url\
                                            --hnsw_config $hnsw_config\
                                            --build_parallel 16\

echo "wait for qdrant stop"
kill $qdrant_pid
wait $qdrant_pid 2>/dev/null
echo "qdrant is stopped"
