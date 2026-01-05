
import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import time
import atexit

from qdrant_client import QdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import Distance, QuantizationSearchParams, SearchParams, VectorParams, PointStruct
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

def get_worker_gpu_assignment(num_gpus: Optional[int] = None):
    """
    Automatically assign GPU to current Gunicorn worker based on worker index.

    For Gunicorn multi-worker setup:
    - Worker 0 -> GPU 0
    - Worker 1 -> GPU 1
    - ...
    - Worker N -> GPU (N % num_gpus)

    Args:
        num_gpus: Number of GPUs to distribute across. If None, auto-detect.

    Returns:
        cuda_device_id: The GPU device ID for this worker (e.g., 0, 1, 2...)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        raise RuntimeError("No CUDA devices available!")

    # Strategy 1: Try to get worker ID from environment variable
    worker_id = int(os.environ.get("GUNICORN_WORKER_ID", "-1"))

    if worker_id == -1:
        # Strategy 2: Use file-based counter with file locking
        # This ensures each worker gets a unique sequential ID
        import fcntl
        counter_file = "/tmp/retrieval_worker_counter.txt"

        try:
            with open(counter_file, "a+") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                try:
                    f.seek(0)
                    content = f.read().strip()
                    worker_id = int(content) if content else 0
                    f.seek(0)
                    f.truncate()
                    f.write(str(worker_id + 1))
                finally:
                    fcntl.flock(f, fcntl.LOCK_UN)
        except Exception as e:
            pid = os.getpid()
            worker_id = pid % 1000  # Use last 3 digits of PID
            print(f"[WARNING] Could not use counter file ({e}), falling back to PID-based assignment")

    # Round-robin assignment
    cuda_device_id = worker_id % num_gpus

    print(f"[INFO] Worker PID: {os.getpid()} | Worker ID: {worker_id} -> Assigned to GPU: cuda:{cuda_device_id} (Total GPUs: {num_gpus})")

    return cuda_device_id

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

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class DenseRetriever(BaseRetriever):
    # Class-level connection pool for Qdrant clients
    _qdrant_client_pool = []
    _pool_lock = threading.Lock()
    _pool_size = 8  # Connection pool size

    @classmethod
    def get_qdrant_client(cls, url, timeout=60):
        """Get a Qdrant client from the connection pool."""
        with cls._pool_lock:
            if not cls._qdrant_client_pool:
                # Initialize connection pool
                print(f"[INFO] Initializing Qdrant connection pool with {cls._pool_size} clients")
                for i in range(cls._pool_size):
                    client = QdrantClient(url=url, prefer_grpc=True, timeout=timeout)
                    cls._qdrant_client_pool.append(client)
                    # Register cleanup on exit
                    if i == 0:
                        atexit.register(cls.close_all_clients)

            # Return a client (simple round-robin)
            client = cls._qdrant_client_pool[0]
            cls._qdrant_client_pool.append(cls._qdrant_client_pool.pop(0))
            return client

    @classmethod
    def close_all_clients(cls):
        """Close all clients in the pool."""
        with cls._pool_lock:
            for client in cls._qdrant_client_pool:
                try:
                    client.close()
                except:
                    pass
            cls._qdrant_client_pool.clear()

    @staticmethod
    def wait_qdrant_load(url, connect_timeout):
        client = QdrantClient(url=url, prefer_grpc=True, timeout=60)
        wait_collection_time = 0
        while True:
            if wait_collection_time >= connect_timeout:
                client.close()
                assert False, f"wait longer than {connect_timeout}s, exit"
            print(f"wait {wait_collection_time}s for qdrant load")
            time.sleep(5)
            wait_collection_time += 5
            try:
                client.info()
                print(f"qdrant loaded and connected")
                client.close()
                break
            except Exception as e:
                pass

    def __init__(self, config: "Config"):
        super().__init__(config)
        # Wait for Qdrant to be ready
        self.wait_qdrant_load(url=config.qdrant_url, connect_timeout=300)

        # Store config for later use
        self.qdrant_url = config.qdrant_url
        self.qdrant_timeout = 60

        self.collection_name = config.qdrant_collection_name

        # Verify collection exists using a temporary client
        temp_client = QdrantClient(url=self.qdrant_url, prefer_grpc=True, timeout=60)
        collections = temp_client.get_collections().collections
        collection_names = [col.name for col in collections]
        temp_client.close()
        assert self.collection_name in collection_names
        
        # Initialize encoder first (needed for building collection)
        # Automatically assign GPU based on worker ID
        assigned_gpu_id = get_worker_gpu_assignment()
        assigned_device = torch.device(f"cuda:{assigned_gpu_id}")

        print(f"[INFO] Initializing encoder on {assigned_device}")

        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16,
            device = assigned_device,
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size
        if config.qdrant_search_quant_param is not None:
            self.search_params = SearchParams(
                **json.loads(config.qdrant_search_param),
                quantization=QuantizationSearchParams(
                    **json.loads(config.qdrant_search_quant_param)
                ),
            )
        else:
            self.search_params = SearchParams(
                **json.loads(config.qdrant_search_param),
            )
        print(f"qdrant search_params: {self.search_params}")

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        query_emb = self.encoder.encode(query)
        query_vector = query_emb[0].tolist()

        # Get a client from the connection pool
        client = self.get_qdrant_client(self.qdrant_url, self.qdrant_timeout)

        # Search in Qdrant
        search_results = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=num,
            search_params=self.search_params,
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
        dataset_path: str = "./data",
        data_split: str = "train",
        qdrant_url: Optional[str] = None,
        qdrant_collection_name: str = "default_collection",
        qdrant_search_param: Optional[str] = None,
        qdrant_search_quant_param: Optional[str] = None,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_search_param = qdrant_search_param
        self.qdrant_search_quant_param = qdrant_search_quant_param
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
    # if not request.topk:
    #     request.topk = config.retrieval_topk  # fallback to default

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
    resp = []
    for url in request.urls:
        resp.append(page_access.access(url))
    
    return {"result": resp}


def initialize_server():
    """Initialize server components. Can be called from main or by Gunicorn worker."""
    global config, retriever, page_access

    # Try to get config from environment variables (for Gunicorn)
    # Otherwise fall back to command-line arguments (for direct python execution)
    if os.environ.get("RETRIEVER_NAME"):
        print("[INFO] Loading configuration from environment variables (Gunicorn mode)")
        config = Config(
            retrieval_method=os.environ.get("RETRIEVER_NAME", "e5"),
            retrieval_topk=int(os.environ.get("TOPK", "3")),
            qdrant_url=os.environ.get("QDRANT_URL"),
            qdrant_collection_name=os.environ.get("QDRANT_COLLECTION_NAME", "default_collection"),
            qdrant_search_param=os.environ.get("QDRANT_SEARCH_PARAM", "{}"),
            qdrant_search_quant_param=os.environ.get("QDRANT_SEARCH_QUANT_PARAM"),
            retrieval_model_path=os.environ.get("RETRIEVER_MODEL", "intfloat/e5-base-v2"),
            retrieval_pooling_method="mean",
            retrieval_query_max_length=256,
            retrieval_use_fp16=True,
            retrieval_batch_size=512,
        )
        pages_path = os.environ.get("PAGES_PATH", "")
    else:
        print("[INFO] Loading configuration from command-line arguments")
        parser = argparse.ArgumentParser(description="Launch the local qdrant retriever.")
        parser.add_argument("--pages_path", type=str, default="xxx", help="Local page file.")
        parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
        parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
        parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
        parser.add_argument("--qdrant_url", type=str, default=None, help="Qdrant server URL (e.g., http://localhost:6333). If not provided, uses local mode.")
        parser.add_argument("--qdrant_collection_name", type=str, default="default_collection", help="Name of the Qdrant collection.")
        parser.add_argument("--qdrant_search_param", type=str, default={}, help="")
        parser.add_argument("--qdrant_search_quant_param", type=str, default=None, help="")
        parser.add_argument("--port", type=int, default=5005)
        parser.add_argument("--save-address-to", type=str, help="path to save server address")
        parser.add_argument("--test_qdrant_load", type=int, default=0)

        args = parser.parse_args()
        pages_path = args.pages_path

        config = Config(
            retrieval_method=args.retriever_name,
            retrieval_topk=args.topk,
            qdrant_url=args.qdrant_url,
            qdrant_collection_name=args.qdrant_collection_name,
            qdrant_search_param=args.qdrant_search_param,
            qdrant_search_quant_param=args.qdrant_search_quant_param,
            retrieval_model_path=args.retriever_model,
            retrieval_pooling_method="mean",
            retrieval_query_max_length=256,
            retrieval_use_fp16=True,
            retrieval_batch_size=512,
        )

    # Instantiate retriever
    retriever = get_retriever(config)

    # Test queries
    query1 = '介绍一下红牛'
    result1 = retriever.search(query1, 1, return_score=False)
    print(f"test1: query: {query1}, result: {result1}")
    query2 = '介绍一下卢布尔雅那'
    result2 = retriever.search(query2, 2, return_score=True)
    print(f"test2: query: {query2}, result: {result2}")
    query3 = ['介绍一下火星', '介绍一下水星']
    result3 = retriever.batch_search(query3, 3, return_score=True)
    print(f"test3: query: {query3}, result: {result3}")
    print("Retriever is ready.")

    # Load pages
    if pages_path and os.path.exists(pages_path):
        page_access = PageAccess(pages_path)
        print("Page Access is ready.")
    else:
        page_access = None
        print("Page Access not configured.")


if __name__ == "__main__":
    from multiprocessing import set_start_method
    set_start_method("spawn")

    parser = argparse.ArgumentParser(description="Launch the local qdrant retriever.")
    parser.add_argument("--pages_path", type=str, default="xxx", help="Local page file.")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages for one query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever model.")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--qdrant_url", type=str, default=None, help="Qdrant server URL (e.g., http://localhost:6333). If not provided, uses local mode.")
    parser.add_argument("--qdrant_collection_name", type=str, default="default_collection", help="Name of the Qdrant collection.")
    parser.add_argument("--qdrant_search_param", type=str, default={}, help="")
    parser.add_argument("--qdrant_search_quant_param", type=str, default=None, help="")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--save-address-to", type=str, help="path to save server address")
    parser.add_argument("--test_qdrant_load", type=int, default=0)

    args = parser.parse_args()

    # Set environment variables for initialize_server to use
    os.environ["PAGES_PATH"] = args.pages_path
    os.environ["TOPK"] = str(args.topk)
    os.environ["RETRIEVER_NAME"] = args.retriever_name
    os.environ["RETRIEVER_MODEL"] = args.retriever_model
    os.environ["QDRANT_URL"] = args.qdrant_url or ""
    os.environ["QDRANT_COLLECTION_NAME"] = args.qdrant_collection_name
    os.environ["QDRANT_SEARCH_PARAM"] = args.qdrant_search_param
    if args.qdrant_search_quant_param:
        os.environ["QDRANT_SEARCH_QUANT_PARAM"] = args.qdrant_search_quant_param

    # Initialize server components
    initialize_server()

    if args.test_qdrant_load != 0:
        print("exit Retriever test.")
        exit(0)

    host_name=socket.gethostname()
    host_ip=socket.gethostbyname(socket.gethostname())
    port = args.port
    host_addr = f"{host_ip}:{port}"

    print(f"Server address: {host_addr}")

    if args.save_address_to:
        os.makedirs(args.save_address_to, exist_ok=True)
        with open(os.path.join(args.save_address_to, "Host" + host_ip + "_" + "IP" + str(port) + ".txt"), "w") as f:
            f.write(host_addr)

    # Launch the server with Uvicorn (single worker mode for direct execution)
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


# Initialize server when loaded by Gunicorn
if os.environ.get("RETRIEVER_NAME"):
    initialize_server()


