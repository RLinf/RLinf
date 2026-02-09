#!/bin/bash

save_dir=$WIKI2018_WORK_DIR

corpus_file=$WIKI2018_WORK_DIR/wiki_corpus.jsonl
save_dir=${save_dir}/e5.index
retriever_name=e5 # this is for indexing naming
retriever_model=/path/to/e5-base-v2

# change faiss_type to HNSW32/64/128 for ANN indexing
# change retriever_name to bm25 for BM25 indexing
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 examples/searchr1/index_builder.py \
    --retrieval_method $retriever_name \
    --model_path $retriever_model \
    --corpus_path $corpus_file \
    --save_dir $save_dir \
    --use_fp16 \
    --max_length 256 \
    --batch_size 512 \
    --pooling_method mean \
    --faiss_type Flat \
    --save_embedding