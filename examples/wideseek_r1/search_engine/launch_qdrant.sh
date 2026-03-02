#!/bin/bash

set -e
set -x

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

pages_file=/PATH/TO/wiki_webpages.jsonl  # Path to your pages JSONL file
retriever_name=e5
retriever_path=/PATH/TO/e5-model  # Path to your e5 model directory

pages_file=/mnt/project_rlinf/xzxuan/data/Asearch/wiki_webpages.jsonl
retriever_name=e5
retriever_path=/mnt/project_rlinf/xzxuan/model/e5

qdrant_url=http://172.27.236.111:6333
qdrant_collection_name=wiki_collection_m32_cef512

# hnsw_ef: search accuracy parameter, higher is more accurate but slower
qdrant_search_param='{"hnsw_ef":256}'

server_port=8000

# Avoid proxy interference with local connections
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "Starting Qdrant Retrieval Server..."

if [[ ! -f "${pages_file}" ]]; then
    echo "Error: Pages file not found: ${pages_file}" >&2
    exit 1
fi

if [[ ! -d "${retriever_path}" ]]; then
    echo "Error: Retriever model not found: ${retriever_path}" >&2
    exit 1
fi

echo "Launching FastAPI server on port ${server_port}..."
echo "Pages file: ${pages_file}"
echo "Retriever: ${retriever_name} at ${retriever_path}"
echo "Qdrant URL: ${qdrant_url}"
echo "Collection: ${qdrant_collection_name}"

python examples/wideseek_r1/search_engine/local_retrieval_server_qdrant.py \
    --pages_path "${pages_file}" \
    --topk 3 \
    --retriever_name "${retriever_name}" \
    --retriever_model "${retriever_path}" \
    --qdrant_collection_name "${qdrant_collection_name}" \
    --qdrant_url "${qdrant_url}" \
    --qdrant_search_param "${qdrant_search_param}" \
    --port "${server_port}"
