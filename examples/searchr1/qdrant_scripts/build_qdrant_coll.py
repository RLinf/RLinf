
import json
import os
import queue
import warnings
from typing import List, Dict, Optional
import argparse
import time

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import Distance, HnswConfigDiff, VectorParams, PointStruct, CollectionStatus
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets


global_encoder = None
global_client = None
def set_global(retrieval_method, config):
    from multiprocessing import current_process
    process_idx = current_process()._identity[0]

    global global_encoder
    global_encoder = Encoder(
        model_name = retrieval_method,
        model_path = config.retrieval_model_path,
        pooling_method = config.retrieval_pooling_method,
        max_length = config.retrieval_query_max_length,
        # use_fp16 = config.retrieval_use_fp16,
        use_fp16=False,
        device = torch.device(f"cuda:{process_idx % torch.cuda.device_count()}"),
    )

    global global_client
    global_client = QdrantClient(url=config.qdrant_url, prefer_grpc=True, timeout=60)

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=8,
        cache_dir="/mnt/project_rlinf/zhuchunyang_rl/tmp",
    )
    return corpus

def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False, device=torch.device("cuda")):
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model = model.to(device=device)
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, device):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16, device=self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        # processing query for different encoders
        if isinstance(query_list, str):
            query_list = [query_list]

        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        # inputs = {k: v.cuda() for k, v in inputs.items()}
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class QdrantIndexBuilder:
    def __init__(self, config):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url, prefer_grpc=True, timeout=60)
        self.collection_name = config.qdrant_collection_name

    def build(self):
        # Initialize encoder first (needed for building collection)
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        
        # Check if collection exists, if not, build it from corpus
        # TODO: for debug
        self.client.delete_collection(collection_name=self.collection_name)

        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        if self.collection_name in collection_names:
            print(f"Collection '{self.collection_name}' already exists. deleting...")
            self.client.delete_collection(collection_name=self.collection_name)
        else:
            print(f"Collection '{self.collection_name}' not found.")
        print(f"Building collection from corpus...")
        self._build_collection_from_corpus(config.corpus_text_field)
        print(f"Collection '{self.collection_name}' built successfully!")

    @staticmethod
    def encode_and_upsert(config, batch_texts, batch_indices, batch_payload):
        collection_name = config.qdrant_collection_name
        global global_encoder
        global global_client

        points = []
        batch_emb = global_encoder.encode(batch_texts, is_query=False)
        # Create points
        for emb, doc_idx, payload in zip(batch_emb, batch_indices, batch_payload):
            points.append(PointStruct(
                id=doc_idx,
                vector=emb.tolist(),
                payload=payload,
            ))
        global_client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )

    def _build_collection_from_corpus(self, text_field=None):
        """Build Qdrant collection from corpus."""
        # assert False
        corpus_data = load_corpus(self.config.corpus_path)
        corpus_size = len(corpus_data)
        print(f"Corpus size: {corpus_size} documents")
        
        # Get vector dimension by encoding a sample document
        sample_text = "helld, world!"
        encoder = Encoder(
            model_name = self.config.retrieval_method,
            model_path = self.config.retrieval_model_path,
            pooling_method = self.config.retrieval_pooling_method,
            max_length = self.config.retrieval_query_max_length,
            use_fp16 = self.config.retrieval_use_fp16,
            device = torch.device("cuda:1"),
        )
        sample_emb = encoder.encode(sample_text, is_query=False)
        vector_size = sample_emb.shape[1]
        encoder = None
        print(f"Vector dimension: {vector_size}")

        # Create collection
        try:
            hnsw_config = json.loads(self.config.hnsw_config)
            print(f"hnsw_config uses: {hnsw_config}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    **hnsw_config
                )
            )
        except Exception as e:
            # Collection might already exist, check and handle
            collections = self.client.get_collections().collections
            collection_names = [col.name for col in collections]
            if self.collection_name in collection_names:
                print(f"Collection '{self.collection_name}' already exists, skipping creation.")
            else:
                raise e
        
        # Encode and insert documents in batches
        from multiprocessing import Pool
        # pool = Pool(32)
        pool = Pool(self.config.build_parallel, initializer=set_global, initargs=(self.config.retrieval_method, self.config))
        # handles = []
        handles = queue.Queue()

        batch_texts = []
        batch_indices = []
        batch_payload = []
        for idx in tqdm(range(corpus_size), desc='Building collection'):
            doc = corpus_data[idx]
            assert self.config.retrieval_method == "e5"
            text = doc['contents']

            # Skip empty texts
            if not text or len(text.strip()) == 0:
                warnings.warn(f"Document {idx} has empty text, skipping...")
                continue
            batch_texts.append(text)
            batch_indices.append(idx)
            batch_payload.append(doc)
            
            # Process batch when it reaches batch_size
            if len(batch_texts) >= self.batch_size:
                handle = pool.apply_async(QdrantIndexBuilder.encode_and_upsert, (self.config, batch_texts, batch_indices, batch_payload))
                # handles.append(handle)
                handles.put(handle)
                if handles.qsize() >= self.config.build_parallel * 10:
                    handles.get().wait()
                batch_texts = []
                batch_indices = []
                batch_payload = []
        
        # Process remaining items
        if batch_texts:
            handle = pool.apply_async(QdrantIndexBuilder.encode_and_upsert, (self.config, batch_texts, batch_indices, batch_payload))
            # handles.append(handle)
            handles.put(handle)

        while not handles.empty():
            handles.get().wait()
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出

        print(f"wait collection status to be green")
        while self.client.get_collection(self.collection_name).status != CollectionStatus.GREEN:
            time.sleep(1)
        print(f"collection status of '{self.collection_name}' is green now, and infos are {self.client.get_collection(self.collection_name)}")
        print(f"Successfully inserted {corpus_size} documents into collection '{self.collection_name}'")

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        qdrant_url: Optional[str] = None,
        qdrant_collection_name: str = "default_collection",
        corpus_text_field: Optional[str] = None,
        hnsw_config: str = None,
        build_parallel: int = None,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = qdrant_collection_name
        self.corpus_text_field = corpus_text_field
        self.hnsw_config = hnsw_config
        self.build_parallel = build_parallel
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Launch the local qdrant retriever.")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--qdrant_url", type=str, default=None, help="Qdrant server URL (e.g., http://localhost:6333). If not provided, uses local mode.")
    parser.add_argument("--qdrant_collection_name", type=str, default="default_collection", help="Name of the Qdrant collection.")
    parser.add_argument("--corpus_text_field", type=str, default=None, help="Field name in corpus documents containing text to encode. If not specified, will try common field names (text, contents, passage, etc.).")
    parser.add_argument("--hnsw_config", type=str, default="", help="Qdrant hnsw config")
    parser.add_argument("--build_parallel", type=int, default=8, help="Qdrant build thread")

    args = parser.parse_args()


    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        corpus_path=args.corpus_path,
        qdrant_url=args.qdrant_url,
        qdrant_collection_name=args.qdrant_collection_name,
        corpus_text_field=args.corpus_text_field,
        hnsw_config=args.hnsw_config,
        build_parallel=args.build_parallel,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=1024,
        # retrieval_batch_size=64,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    index_builder = QdrantIndexBuilder(config)
    index_builder.build()

