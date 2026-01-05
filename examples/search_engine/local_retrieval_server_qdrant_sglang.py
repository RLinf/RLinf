"""
SGLang-based retrieval server using sglang.launch_server command.
Reference: https://blog.csdn.net/gitblog_00644/article/details/151436576
"""

import json
import os
import time
import atexit
import argparse
import asyncio
import aiohttp
from typing import List, Dict, Optional
import threading
import numpy as np
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import QuantizationSearchParams, SearchParams

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import socket
import requests

import os
import time
from typing import List, Union

import numpy as np
import requests
from transformers import AutoTokenizer


class SGLangEmbeddingClient:
    """
    Client for SGLang embedding server launched via sglang.launch_server.
    Uses HTTP /v1/embeddings endpoint.
    Aligns behavior with HF E5 implementation:
      - e5 prefixes
      - max_length=256 truncation
      - (no padding)
      - L2 normalize with torch default eps=1e-12
    """

    def __init__(
        self,
        model_name: str,          # must match --served-model-name (recommended)
        tokenizer_path: str,      # should be the same as HF model_path
        server_url: str = "http://127.0.0.1:30000",
        max_length: int = 256,
    ):
        self.model_name = model_name
        self.server_url = server_url.rstrip("/")
        self.embeddings_endpoint = f"{self.server_url}/v1/embeddings"
        self.max_length = max_length

        # Tokenizer: we do truncation on client side to match HF max_length=256
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            use_fast=True,
            trust_remote_code=True,
        )

        print(f"[INFO] SGLang Client connecting to: {self.server_url}")
        print(f"[INFO] Embeddings endpoint: {self.embeddings_endpoint}")

        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 300):
        print("[INFO] Waiting for SGLang server to be ready...")
        start = time.time()
        while time.time() - start < timeout:
            try:
                r = requests.get(f"{self.server_url}/health", timeout=5)
                if r.status_code == 200:
                    print("[INFO] ✓ SGLang server is ready!")
                    return
            except Exception:
                pass
            time.sleep(2)
        raise TimeoutError(f"SGLang server did not start within {timeout}s")

    def _add_model_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        if "e5" in self.model_name.lower():
            if is_query:
                return [f"query: {t}" for t in texts]
            else:
                return [f"passage: {t}" for t in texts]
        elif "bge" in self.model_name.lower():
            if is_query:
                return [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return texts

    def _texts_to_input_ids(self, texts: List[str]) -> Union[List[int], List[List[int]]]:
        """
        Match HF: tokenizer(..., truncation=True, max_length=256).
        IMPORTANT: do NOT pad here; padding can change pooling if server pooling doesn't mask pads.
        """
        enc = self.tokenizer(
            texts,
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids_batch = enc["input_ids"]  # List[List[int]]

        # SGLang supports passing input_ids directly. :contentReference[oaicite:5]{index=5}
        if len(input_ids_batch) == 1:
            return input_ids_batch[0]        # single: List[int]
        return input_ids_batch              # batch: List[List[int]]

    def encode(self, texts: Union[str, List[str]], is_query: bool = True) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]

        texts = self._add_model_prefix(texts, is_query)

        # Force HF-equivalent truncation behavior via input_ids
        # input_obj = self._texts_to_input_ids(texts)

        if len(texts) == 1:
            texts = texts[0]

        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float16",
        }
        r = requests.post(self.embeddings_endpoint, json=payload, timeout=120)
        r.raise_for_status()
        result = r.json()

        embeddings = np.array([item["embedding"] for item in result["data"]], dtype=np.float32)

        # Match torch.nn.functional.normalize default eps=1e-12
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-12)
        breakpoint()
        return embeddings


class SGLangEmbeddingClient_old:
    """
    Client for SGLang embedding server launched via sglang.launch_server.
    Communicates with the server via HTTP /v1/embeddings endpoint.
    """

    def __init__(
        self,
        model_name: str,
        server_url: str = "http://127.0.0.1:30000",
    ):
        self.model_name = model_name
        self.server_url = server_url
        self.embeddings_endpoint = f"{server_url}/v1/embeddings"

        print(f"[INFO] SGLang Client connecting to: {self.server_url}")
        print(f"[INFO] Embeddings endpoint: {self.embeddings_endpoint}")

        # Wait for server to be ready
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 300):
        """Wait for SGLang server to be ready."""
        print(f"[INFO] Waiting for SGLang server to be ready...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                import requests
                response = requests.get(f"{self.server_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"[INFO] ✓ SGLang server is ready!")
                    return
            except Exception:
                pass
            time.sleep(5)

        raise TimeoutError(f"SGLang server did not start within {timeout}s")

    def _add_model_prefix(self, texts: List[str], is_query: bool) -> List[str]:
        """Add model-specific prefixes for E5/BGE models."""
        if "e5" in self.model_name.lower():
            if is_query:
                return [f"query: {text}" for text in texts]
            else:
                return [f"passage: {text}" for text in texts]
        elif "bge" in self.model_name.lower():
            if is_query:
                return [f"Represent this sentence for searching relevant passages: {text}" for text in texts]
        return texts

    def encode(self, texts: List[str], is_query: bool = True) -> np.ndarray:
        """
        Encode texts to embeddings via HTTP request.

        Args:
            texts: List of texts to encode
            is_query: Whether encoding queries (vs passages)

        Returns:
            np.ndarray: Embeddings of shape (len(texts), embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Add model-specific prefixes
        texts = self._add_model_prefix(texts, is_query)

        # Prepare payload for SGLang /v1/embeddings endpoint
        if len(texts) == 1:
            texts = texts[0]

        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float"
        }
        # Make HTTP request
        try:
            response = requests.post(
                self.embeddings_endpoint,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            result = response.json()

            # Extract embeddings from response
            # Response format: {"data": [{"embedding": [...]}, ...]}
            embeddings = [item["embedding"] for item in result["data"]]
            embeddings_np = np.array(embeddings, dtype=np.float32)

            # Normalize embeddings (important for cosine similarity)
            norms = np.linalg.norm(embeddings_np, axis=1, keepdims=True)
            embeddings_np = embeddings_np / (norms + 1e-8)

            return embeddings_np

        except Exception as e:
            print(f"[ERROR] SGLang encoding failed: {e}")
            raise

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
    @staticmethod
    def wait_qdrant_load(url, connect_timeout):
        client = QdrantClient(url=url, prefer_grpc=True, timeout=60)
        atexit.register(client.close)
        wait_collection_time = 0
        while True:
            if wait_collection_time >= connect_timeout:
                raise TimeoutError(f"Qdrant did not load within {connect_timeout}s")
            print(f"[INFO] Waiting {wait_collection_time}s for Qdrant to load...")
            time.sleep(5)
            wait_collection_time += 5
            try:
                client.info()
                print(f"[INFO] ✓ Qdrant loaded and connected!")
                break
            except Exception:
                pass
        return client

    def __init__(self, config: "Config"):
        super().__init__(config)

        # Connect to Qdrant
        self.client = self.wait_qdrant_load(url=config.qdrant_url, connect_timeout=300)

        self.collection_name = config.qdrant_collection_name
        collections = self.client.get_collections().collections
        collection_names = [col.name for col in collections]
        assert self.collection_name in collection_names, \
            f"Collection '{self.collection_name}' not found. Available: {collection_names}"

        # Initialize SGLang client (connects to running SGLang server)
        print(f"[INFO] Initializing SGLang client...")
        self.encoder = SGLangEmbeddingClient(
            model_name=self.retrieval_method,
            tokenizer_path=config.retrieval_model_path,
            server_url=config.sglang_server_url,
        )

        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

        # Setup Qdrant search parameters
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
        print(f"[INFO] Qdrant search_params: {self.search_params}")

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk

        # Encode query using SGLang client
        query_emb = self.encoder.encode(query, is_query=True)
        query_vector = query_emb[0].tolist()

        # Search in Qdrant
        search_results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=num,
            search_params=self.search_params,
        ).points

        if len(search_results) < 1:
            return ([], []) if return_score else []

        payloads = [result.payload for result in search_results]
        scores = [result.score for result in search_results]

        return (payloads, scores) if return_score else payloads

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        # if return_score:
        #     all_payloads, all_scores = [], []
        #     for query in query_list:
        #         payloads, scores = self._search(query, num, return_score)
        #         all_payloads.append(payloads)
        #         all_scores.append(scores)
        #     return all_payloads, all_scores
        # else:
        #     all_payloads = []
        #     for query in query_list:
        #         payloads = self._search(query, num, return_score)
        #         all_payloads.append(payloads)
        #     return all_payloads        
        

        if num is None:
            num = self.topk

        # Batch encode all queries using SGLang client
        query_embs = self.encoder.encode(query_list, is_query=True)

        all_payloads = []
        all_scores = []

        # Search for each query
        for query_emb in query_embs:
            query_vector = query_emb.tolist()

            search_results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=num,
                search_params=self.search_params,
            ).points

            if len(search_results) > 0:
                payloads = [result.payload for result in search_results]
                scores = [result.score for result in search_results]
                all_payloads.append(payloads)
                all_scores.append(scores)
            else:
                all_payloads.append([])
                all_scores.append([])

        return (all_payloads, all_scores) if return_score else all_payloads


def get_retriever(config):
    return DenseRetriever(config)


class PageAccess:
    def __init__(self, pages_path):
        pages = []
        print(f"[INFO] Loading pages from {pages_path}...")
        for ff in tqdm(open(pages_path, "r"), desc="Loading pages"):
            pages.append(json.loads(ff))
        self.pages = {page["url"]: page for page in pages}
        print(f"[INFO] Loaded {len(self.pages)} pages")

    def access(self, url):
        # PHP parsing
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")
        return self.pages.get(url)


#####################################
# FastAPI Server
#####################################

class Config:
    """Configuration for the retrieval server."""
    def __init__(
        self,
        retrieval_method: str = "e5",
        retrieval_model_path: str = "./model",
        retrieval_topk: int = 10,
        qdrant_url: Optional[str] = None,
        qdrant_collection_name: str = "default_collection",
        qdrant_search_param: Optional[str] = None,
        qdrant_search_quant_param: Optional[str] = None,
        retrieval_query_max_length: int = 256,
        retrieval_batch_size: int = 128,
        sglang_server_url: str = "http://127.0.0.1:30000",
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_topk = retrieval_topk
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_search_param = qdrant_search_param
        self.qdrant_search_quant_param = qdrant_search_quant_param
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_batch_size = retrieval_batch_size
        self.sglang_server_url = sglang_server_url


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


class AccessRequest(BaseModel):
    urls: List[str]


app = FastAPI()

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """Retrieval endpoint."""
    time_start = time.time()

    if not request.topk:
        request.topk = config.retrieval_topk

    # Perform batch retrieval
    if request.return_scores:
        results, scores = retriever.batch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=True
        )
    else:
        results = retriever.batch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=False
        )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)

    time_elapse = time.time() - time_start
    print(f"[INFO] Retrieved {len(request.queries)} queries in {time_elapse:.2f}s")

    return {"result": resp}


@app.post("/access")
async def access_endpoint(request: AccessRequest):
    """Page access endpoint."""
    resp = []
    for url in request.urls:
        resp.append(page_access.access(url))
    return {"result": resp}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SGLang-based retrieval server")
    parser.add_argument("--pages_path", type=str, default="xxx", help="Local page file")
    parser.add_argument("--topk", type=int, default=3, help="Number of retrieved passages")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Retriever model name")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path of the retriever model.")
    parser.add_argument("--qdrant_url", type=str, default=None, help="Qdrant server URL")
    parser.add_argument("--qdrant_collection_name", type=str, default="default_collection", help="Qdrant collection name")
    parser.add_argument("--qdrant_search_param", type=str, default='{}', help="Qdrant search parameters (JSON)")
    parser.add_argument("--qdrant_search_quant_param", type=str, default=None, help="Qdrant quantization parameters")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--sglang_server_url", type=str, default="http://127.0.0.1:30000", help="SGLang server URL")
    parser.add_argument("--save-address-to", type=str, help="Path to save server address")
    parser.add_argument("--test_qdrant_load", type=int, default=0, help="Test mode")

    args = parser.parse_args()

    # Get server address
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(socket.gethostname())
    port = args.port
    host_addr = f"{host_ip}:{port}"

    print(f"[INFO] Server address: {host_addr}")
    print(f"[INFO] SGLang server URL: {args.sglang_server_url}")

    if args.save_address_to:
        os.makedirs(args.save_address_to, exist_ok=True)
        with open(os.path.join(args.save_address_to, f"Host{host_ip}_IP{port}.txt"), "w") as f:
            f.write(host_addr)

    # Build configuration
    config = Config(
        retrieval_method=args.retriever_name,
        retrieval_model_path=args.retriever_model,
        retrieval_topk=args.topk,
        qdrant_url=args.qdrant_url,
        qdrant_collection_name=args.qdrant_collection_name,
        qdrant_search_param=args.qdrant_search_param,
        qdrant_search_quant_param=args.qdrant_search_quant_param,
        retrieval_query_max_length=256,
        retrieval_batch_size=512,
        sglang_server_url=args.sglang_server_url,
    )

    # Initialize retriever (this connects to SGLang server)
    print("[INFO] Initializing retriever...")
    retriever = get_retriever(config)

    # Run test queries
    print("[INFO] Running test queries...")
    # try:
    #     query1 = '介绍一下红牛'
    #     result1 = retriever.search(query1, 1, return_score=False)
    #     print(f"[TEST 1] Query: '{query1}' -> {len(result1)} results")

    #     query2 = '介绍一下卢布尔雅那'
    #     result2, scores2 = retriever.search(query2, 2, return_score=True)
    #     print(f"[TEST 2] Query: '{query2}' -> {len(result2)} results")

    #     query3 = ['介绍一下火星', '介绍一下水星']
    #     result3, scores3 = retriever.batch_search(query3, 3, return_score=True)
    #     print(f"[TEST 3] Batch: {len(query3)} queries -> {len(result3)} results")

    #     print("[INFO] ✓ All test queries passed!")
    # except Exception as e:
    #     print(f"[ERROR] Test queries failed: {e}")
    #     raise

    if args.test_qdrant_load != 0:
        print("[INFO] Test mode: exiting")
        exit(0)

    # Load pages
    # if os.path.exists(args.pages_path):
    #     page_access = PageAccess(args.pages_path)
    # else:
    #     page_access = None
    #     print(f"[WARNING] Pages file not found at {args.pages_path}, page access disabled")
    page_access = None

    # Launch FastAPI server
    print(f"[INFO] Launching FastAPI server on 0.0.0.0:{port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
