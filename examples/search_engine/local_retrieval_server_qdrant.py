
import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import time

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import Distance, VectorParams, PointStruct
import torch
import threading
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import socket

global_encoder = None
def set_global_encoder(retrieval_method, config):
    from multiprocessing import current_process
    process_idx = current_process()._identity[0]

    global global_encoder
    global_encoder = Encoder(
        model_name = retrieval_method,
        model_path = config.retrieval_model_path,
        pooling_method = config.retrieval_pooling_method,
        max_length = config.retrieval_query_max_length,
        use_fp16 = config.retrieval_use_fp16,
        device = torch.device(f"cuda:{process_idx % torch.cuda.device_count()}"),
    )

def load_corpus(corpus_path: str):
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
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
    # model.cuda()
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

class BaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class DenseRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        # Initialize Qdrant client
        if config.qdrant_url:
            self.client = QdrantClient(url=config.qdrant_url, prefer_grpc=True, timeout=60)
        else:
            # Use local mode if no URL provided
            self.client = QdrantClient(path=config.qdrant_path) if hasattr(config, 'qdrant_path') and config.qdrant_path else QdrantClient(":memory:")
        
        self.collection_name = config.qdrant_collection_name
        
        # Initialize encoder first (needed for building collection)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16,
            device = torch.device("cuda:1"),
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        
        # Check if collection exists, if not, build it from corpus
        # TODO: for debug
        # self.client.delete_collection(collection_name=self.collection_name)

        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        if self.collection_name not in collection_names:
            print(f"Collection '{self.collection_name}' not found. Building collection from corpus...")
            self._build_collection_from_corpus(config.corpus_text_field)
            print(f"Collection '{self.collection_name}' built successfully!")
        else:
            print(f"Collection '{self.collection_name}' already exists. Using existing collection.")
        # if not self.client.collection_exists(self.collection_name):
        #     self.client.create_collection(
        #         collection_name=self.collection_name,
        #         vectors_config=VectorParams(size=100, distance=Distance.COSINE),
        #     )
    
    @staticmethod # TODO: follow /mnt/public/xzxuan/Agent/ASearcher/utils/index_builder.py
    def _extract_text_from_doc(doc, text_field=None):
        """Extract text from a document. Try common field names if text_field is not specified."""
        if text_field:
            if text_field in doc:
                text = doc[text_field]
                if isinstance(text, str):
                    return text
                elif isinstance(text, list):
                    return " ".join(str(t) for t in text)
        
        # Try common field names
        for field in ['text', 'contents', 'passage', 'content', 'body']:
            if field in doc:
                text = doc[field]
                if isinstance(text, str):
                    return text
                elif isinstance(text, list):
                    return " ".join(str(t) for t in text)
        
        # If no text field found, try to concatenate all string values
        text_parts = []
        for key, value in doc.items():
            if isinstance(value, str) and len(value) > 0:
                text_parts.append(value)
        if text_parts:
            return " ".join(text_parts)
        
        # Last resort: convert entire doc to string
        return str(doc)

    @staticmethod
    def encode_and_upsert(config, encoder, batch_texts, batch_indices, batch_payload):
        if config.qdrant_url:
            client = QdrantClient(url=config.qdrant_url, prefer_grpc=True, timeout=60)
        else:
            # Use local mode if no URL provided
            client = QdrantClient(path=config.qdrant_path) if hasattr(config, 'qdrant_path') and config.qdrant_path else QdrantClient(":memory:")
        collection_name = config.qdrant_collection_name
        global global_encoder

        points = []
        batch_emb = global_encoder.encode(batch_texts, is_query=False)
        # batch_emb = encoder.encode(batch_texts, is_query=False)
        # Create points
        for emb, doc_idx, payload in zip(batch_emb, batch_indices, batch_payload):
            points.append(PointStruct(
                id=doc_idx,
                vector=emb.tolist(),
                payload=payload,
            ))
        client.upsert(
            collection_name=collection_name,
            points=points,
        )

    @staticmethod
    def _build_collection_from_corpus_split(idx_start, idx_end, config, encoder, batch_size, corpus_path, text_field):
        points = []
        batch_texts = []
        batch_indices = []
        batch_payload = []
        corpus = load_corpus(corpus_path)
        for idx in tqdm(range(idx_start, idx_end), desc='Building collection'):
            doc = corpus[idx]
            text = DenseRetriever._extract_text_from_doc(doc, text_field)
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                warnings.warn(f"Document {idx} has empty text, skipping...")
                continue
            batch_texts.append(text)
            batch_indices.append(idx)
            batch_payload.append(doc)
            
            # Process batch when it reaches batch_size
            if len(batch_texts) >= batch_size:
                DenseRetriever.encode_and_upsert(config, encoder, batch_texts, batch_indices, batch_payload)
                points = []
                batch_texts = []
                batch_indices = []
                batch_payload = []
                # torch.cuda.empty_cache()
        
        # Process remaining items
        if batch_texts:
            DenseRetriever.encode_and_upsert(config, encoder, batch_texts, batch_indices, batch_payload)

        

    def _build_collection_from_corpus(self, text_field=None):
        """Build Qdrant collection from corpus."""
        # assert False
        self.corpus = load_corpus(self.corpus_path)
        corpus_size = len(self.corpus)
        print(f"Corpus size: {corpus_size} documents")
        
        # Get vector dimension by encoding a sample document
        # sample_doc = self.corpus[0]
        # sample_text = DenseRetriever._extract_text_from_doc(sample_doc, text_field)
        sample_text = "helld, world!"
        sample_emb = self.encoder.encode(sample_text, is_query=False)
        vector_size = sample_emb.shape[1]
        print(f"Vector dimension: {vector_size}")
        
        # Create collection
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
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
        parallel_size = 4
        # pool = Pool(32)
        pool = Pool(parallel_size, initializer=set_global_encoder, initargs=(self.retrieval_method, config))
        handles = []

        # chunk_size = (corpus_size + parallel_size - 1) // parallel_size
        # start_end = [[i, min(corpus_size, i+chunk_size)] for i in range(0, corpus_size, chunk_size)]
        # for i in range(parallel_size):
        #     idx_start = chunk_size * i
        #     idx_end = min(corpus_size, chunk_size * (i + 1))
        #     # print(start_idx, end_idx)
        #     handle = pool.apply_async(DenseRetriever._build_collection_from_corpus_split, (idx_start, idx_end, self.config, None, self.batch_size, self.corpus_path, text_field))
        #     handles.append(handle)

        points = []
        batch_texts = []
        batch_indices = []
        batch_payload = []
        for idx in tqdm(range(corpus_size), desc='Building collection'):
            doc = self.corpus[idx]
            text = DenseRetriever._extract_text_from_doc(doc, text_field)
            # Skip empty texts
            if not text or len(text.strip()) == 0:
                warnings.warn(f"Document {idx} has empty text, skipping...")
                continue
            batch_texts.append(text)
            batch_indices.append(idx)
            batch_payload.append(doc)
            
            # Process batch when it reaches batch_size
            if len(batch_texts) >= self.batch_size:
                # DenseRetriever.encode_and_upsert(self.config, self.encoder, batch_texts, batch_indices, batch_payload)
                handle = pool.apply_async(DenseRetriever.encode_and_upsert, (self.config, self.encoder, batch_texts, batch_indices, batch_payload))
                handles.append(handle)
                points = []
                batch_texts = []
                batch_indices = []
                batch_payload = []
                # torch.cuda.empty_cache()
        
        # Process remaining items
        if batch_texts:
            # DenseRetriever.encode_and_upsert(self.config, self.encoder, batch_texts, batch_indices, batch_payload)
            handle = pool.apply_async(DenseRetriever.encode_and_upsert, (self.config, self.encoder, batch_texts, batch_indices, batch_payload))
            handles.append(handle)

        for handle in handles:
            handle.wait()
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        print(f"Successfully inserted {corpus_size} documents into collection '{self.collection_name}'")

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        query_vector = query_emb[0].tolist()
        
        # Search in Qdrant
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=num
        ).points
        # print(f"zcy_dbg: search_results: {search_results}")

        if len(search_results) < 1:
            if return_score:
                return [], []
            else:
                return []
        
        # # Extract IDs and scores
        # idxs = [result.id for result in search_results]
        # scores = [result.score for result in search_results]
        
        # # Convert IDs to integers if they are stored as integers
        # idxs = [int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else idx for idx in idxs]
        
        # results = load_docs(self.corpus, idxs)
        # if return_score:
        #     return results, scores
        # else:
        #     return results
        payloads = [result.payload for result in search_results]
        scores = [result.score for result in search_results]
        if return_score:
            return payloads, scores
        else:
            return payloads

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if return_score:
            all_payloads, all_scores = [], []
            for query in query_list:
                payloads, scores = self._search(query, num, return_score)
                all_payloads.append(payloads)
                all_scores.append(scores)
            return all_payloads, all_scores
        else:
            all_payloads = []
            for query in query_list:
                payloads = self._search(query, num, return_score)
                all_payloads.append(payloads)
            return all_payloads

        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_emb = self.encoder.encode(query_batch)
            
            batch_results = []
            batch_scores = []
            
            # Search each query in the batch
            for query_emb in batch_emb:
                query_vector = query_emb.tolist()
                search_results = self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=num
                ).points
                # print(f"zcy_dbg: batch search_results: {search_results}")
                
                if len(search_results) > 0:
                    idxs = [result.id for result in search_results]
                    query_scores = [result.score for result in search_results]
                    
                    # Convert IDs to integers if they are stored as integers
                    idxs = [int(idx) if isinstance(idx, (int, str)) and str(idx).isdigit() else idx for idx in idxs]
                    
                    batch_results.append(idxs)
                    batch_scores.append(query_scores)
                else:
                    batch_results.append([])
                    batch_scores.append([])
            
            # Load documents for all queries in batch
            flat_idxs = sum(batch_results, [])
            if flat_idxs:
                batch_docs = load_docs(self.corpus, flat_idxs)
                # Chunk them back per query
                batch_docs_list = []
                idx_offset = 0
                for query_idxs in batch_results:
                    num_docs = len(query_idxs)
                    batch_docs_list.append(batch_docs[idx_offset:idx_offset + num_docs])
                    idx_offset += num_docs
            else:
                batch_docs_list = [[] for _ in batch_results]
            
            results.extend(batch_docs_list)
            scores.extend(batch_scores)
            
            del batch_emb, batch_results, batch_scores, query_batch
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results

def get_retriever(config):
    return DenseRetriever(config)

class PageAccess:
    def __init__(self, pages_path):
        pages = []
        for ff in tqdm(open(pages_path,"r"), desc="PageAccess"):
            pages.append(json.loads(ff))
        self.pages = {page["url"]: page  for page in pages}
    
    def access(self, url):
        # php parsing
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")
        if url not in self.pages:
            return None
        return self.pages[url]

#####################################
# FastAPI server below
#####################################

class Config:
    """
    Minimal config class (simulating your argparse) 
    Replace this with your real arguments or load them dynamically.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25",
        corpus_path: str = "./data/corpus.jsonl",
        dataset_path: str = "./data",
        data_split: str = "train",
        qdrant_url: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        qdrant_collection_name: str = "default_collection",
        corpus_text_field: Optional[str] = None,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.qdrant_url = qdrant_url
        self.qdrant_path = qdrant_path
        self.qdrant_collection_name = qdrant_collection_name
        self.corpus_text_field = corpus_text_field
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size

class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False

class AccessRequest(BaseModel):
    urls: List[str]

app = FastAPI()
threading_lock = threading.Lock()

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """
    Endpoint that accepts queries and performs retrieval.
    Input format:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    time_start = time.time()
    if not request.topk:
        request.topk = config.retrieval_topk  # fallback to default

    # Perform batch retrieval
    with threading_lock:
        if request.return_scores:
            results, scores = retriever.batch_search(
                query_list=request.queries,
                num=request.topk,
                return_score=request.return_scores
            )
        else:
            results = retriever.batch_search(
                query_list=request.queries,
                num=request.topk,
                return_score=request.return_scores
            )
        
    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # If scores are returned, combine them with results
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)
    time_elapse = time.time() - time_start
    print(f"request: {request}, time_elapse: {time_elapse}")
    return {"result": resp}

@app.post("/access")
async def access_endpoint(request: AccessRequest):
    if page_access is None:
        return {"error": "Page access is not available. Pages file was not loaded."}

    resp = []
    with threading_lock:
        for url in request.urls:
            resp.append(page_access.access(url))

    return {"result": resp}


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn")
    parser = argparse.ArgumentParser(description="Launch the local qdrant retriever.")
    parser.add_argument("--index_path", type=str, default="/home/peterjin/mnt/index/wiki-18/e5_Flat.index", help="Corpus indexing file (legacy, not used for Qdrant).")
    parser.add_argument("--corpus_path", type=str, default="/home/peterjin/mnt/data/retrieval-corpus/wiki-18.jsonl", help="Local corpus file.")
    parser.add_argument("--pages_path", type=str, default="xxx", help="Local page file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--qdrant_url", type=str, default=None, help="Qdrant server URL (e.g., http://localhost:6333). If not provided, uses local mode.")
    parser.add_argument("--qdrant_path", type=str, default=None, help="Path for local Qdrant storage (only used if qdrant_url is not provided).")
    parser.add_argument("--qdrant_collection_name", type=str, default="default_collection", help="Name of the Qdrant collection.")
    parser.add_argument("--corpus_text_field", type=str, default=None, help="Field name in corpus documents containing text to encode. If not specified, will try common field names (text, contents, passage, etc.).")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--save-address-to", type=str, help="path to save server address")

    args = parser.parse_args()

    host_name=socket.gethostname()
    host_ip=socket.gethostbyname(socket.gethostname())
    port = args.port

    host_addr = f"{host_ip}:{port}"

    print(f"Server address: {host_addr}")
    
    if args.save_address_to:
        os.makedirs(args.save_address_to, exist_ok=True)
        with open(os.path.join(args.save_address_to, "Host" + host_ip + "_" + "IP" + str(port) + ".txt"), "w") as f:
            f.write(host_addr)

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method = args.retriever_name,  # or "dense"
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        qdrant_url=args.qdrant_url,
        qdrant_path=args.qdrant_path,
        qdrant_collection_name=args.qdrant_collection_name,
        corpus_text_field=args.corpus_text_field,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method="mean",
        retrieval_query_max_length=256,
        retrieval_use_fp16=True,
        retrieval_batch_size=512,
        # retrieval_batch_size=64,
    )

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = get_retriever(config)

    # TODO: test:
    query1 = '介绍一下红牛'
    result1 = retriever.search(query1, 1, return_score=False)
    print(f"test1: query: {query1}, result: {result1}")
    query2 = '介绍一下卢布尔雅那'
    result2 = retriever.search(query2, 2, return_score=True)
    print(f"test2: query: {query2}, result: {result2}")
    query3 = ['介绍一下火星', '介绍一下水星']
    result3 = retriever.batch_search(query3, 3, return_score=True)
    print(f"test3: query: {query3}, result: {result3}")
    print("Retriver is ready.")

    # 3) Load pages
    if os.path.exists(args.pages_path):
        page_access = PageAccess(args.pages_path)
        print("Page Access is ready.")
    else:
        page_access = None
        print(f"Pages file not found at {args.pages_path}, page access will not be available.")

    # 4) Launch the server.
    config = uvicorn.Config(
        app,
        host=host_addr.split(":")[0],
        port=int(host_addr.split(":")[1]),
        log_level="warning",
    )
    # http_server = uvicorn.Server(config)
    # http_server.run()
    uvicorn.run(app, host="0.0.0.0", port=8000)

