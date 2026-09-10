#!/bin/bash

set -ex

# Build step 1: set your wiki_dataset corpus path
WIKI2018_DIR="${WIKI2018_DIR:-/path/to/wiki_dataset}"
corpus_file=$WIKI2018_DIR/wiki_corpus.jsonl

# Build step 2: set your retriever model path
retriever_name=e5 # this is for indexing naming
retriever_path="${RETRIEVER_PATH:-/path/to/retriever/model}"

# Qdrant server configuration
qdrant_url="${QDRANT_URL:-http://localhost:6333}"
qdrant_collection_name="${QDRANT_COLLECTION_NAME:-wiki_collection}"
hnsw_config='{"m":32,"ef_construct":512}'

CONFIG_PATH="$( realpath "$( dirname "${BASH_SOURCE[0]}" )"  )"
ASCEND_LAUNCH_BLOCKING="${ASCEND_LAUNCH_BLOCKING:-1}" \
python3 "${CONFIG_PATH}/build_index.py" \
    --device_type npu \
    --corpus_path "${corpus_file}" \
    --retriever_name "${retriever_name}" \
    --retriever_model "${retriever_path}" \
    --qdrant_collection_name "${qdrant_collection_name}" \
    --qdrant_url "${qdrant_url}" \
    --hnsw_config "${hnsw_config}" \
    --build_parallel "${BUILD_PARALLEL:-16}"
