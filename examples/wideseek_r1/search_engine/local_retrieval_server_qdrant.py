import json
import os
import warnings
from typing import List, Dict, Optional
import argparse
import time
import atexit

from qdrant_client import AsyncQdrantClient
from qdrant_client.conversions.common_types import QueryResponse
from qdrant_client.models import Distance, QuantizationSearchParams, SearchParams, VectorParams, PointStruct
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import socket
from multiprocessing import set_start_method, Queue
import asyncio
import torch
from concurrent.futures import ProcessPoolExecutor

# ============================================================================
# Model Loading and Pooling Utilities
# ============================================================================

def load_model(model_path: str, use_fp16: bool = False, device=torch.device("cuda")):
    """Load retrieval model from checkpoint.

    Args:
        model_path: Path to the pretrained model
        use_fp16: Whether to use FP16 precision
        device: Target device for the model

    Returns:
        Tuple of (model, tokenizer)
    """
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
    attention_mask=None,
    pooling_method="mean"
):
    """Apply pooling to model outputs.

    Args:
        pooler_output: Pooler output from the model
        last_hidden_state: Last hidden state from the model
        attention_mask: Attention mask
        pooling_method: Pooling method ('mean', 'cls', or 'pooler')

    Returns:
        Pooled representation
    """
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError(f"Pooling method '{pooling_method}' not implemented!")

# ============================================================================
# Encoder Class
# ============================================================================

class Encoder:
    """Text encoder supporting various retrieval models (e5, bge, dpr, T5).

    Attributes:
        model_name: Name of the retrieval model
        model_path: Path to the model checkpoint
        pooling_method: Pooling method to use
        max_length: Maximum sequence length
        use_fp16: Whether to use FP16 precision
        device: Device to run the model on
    """

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
        """Encode text into vectors.

        Args:
            query_list: List of text strings to encode
            is_query: Whether the text is a query (affects prompt formatting)

        Returns:
            Numpy array of embeddings (shape: [batch_size, embedding_dim])
        """
        if isinstance(query_list, str):
            query_list = [query_list]

        # E5 model requires query/passage prefixes
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        # BGE model requires instruction prefix for queries
        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
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


# ============================================================================
# Async Encoder Pool for Multi-GPU Encoding
# ============================================================================

class AsyncEncoderPool:
    """Async encoder pool for multi-GPU parallel encoding.

    Uses ProcessPoolExecutor to distribute encoding tasks across multiple GPUs.
    Each worker process has its own encoder instance on a specific GPU.
    """

    @staticmethod
    def set_global_encoder(init_queue: Queue):
        """Initialize global encoder for worker process.

        Args:
            init_queue: Queue containing encoder initialization parameters
        """
        args = init_queue.get()
        device = args[-1]
        assert "global_encoder" not in globals()
        globals()["global_encoder"] = Encoder(*args)

    @staticmethod
    def global_encode(*args, **kwargs):
        """Encode using global encoder in worker process."""
        assert "global_encoder" in globals()
        encoder: Encoder = globals()["global_encoder"]
        return encoder.encode(*args, **kwargs)

    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16, devices):
        """Initialize encoder pool with multiple GPU workers.

        Args:
            model_name: Name of the retrieval model
            model_path: Path to model checkpoint
            pooling_method: Pooling method to use
            max_length: Maximum sequence length
            use_fp16: Whether to use FP16 precision
            devices: List of GPU devices to use
        """
        init_queue = Queue()
        for device in devices:
            init_queue.put([model_name, model_path, pooling_method, max_length, use_fp16, device])

        self.encoders = ProcessPoolExecutor(
            max_workers=len(devices),
            initializer=AsyncEncoderPool.set_global_encoder,
            initargs=(init_queue,)
        )

    async def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        """Async encode text strings.

        Args:
            query_list: List of text strings to encode
            is_query: Whether the text is a query

        Returns:
            Numpy array of embeddings
        """
        return await loop.run_in_executor(self.encoders, AsyncEncoderPool.global_encode, (query_list, is_query))


# ============================================================================
# Retriever Classes
# ============================================================================

class AsyncBaseRetriever:
    """Base class for async retrievers."""

    def __init__(self, config):
        """Initialize retriever.

        Args:
            config: Configuration object containing retrieval parameters
        """
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

    async def _asearch(self, query: str, num: int, return_score: bool):
        """Internal search method to be implemented by subclasses"""
        raise NotImplementedError

    async def _abatch_search(self, query_list: List[str], num: int, return_score: bool):
        """Internal batch search method to be implemented by subclasses"""
        raise NotImplementedError

    async def asearch(self, query: str, num: int = None, return_score: bool = False):
        """Async search for a single query.

        Args:
            query: Query string
            num: Number of results to return (default: self.topk)
            return_score: Whether to return scores

        Returns:
            List of documents (and scores if return_score=True)
        """
        return await self._asearch(query, num, return_score)

    async def abatch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        """Async batch search for multiple queries.

        Args:
            query_list: List of query strings
            num: Number of results per query (default: self.topk)
            return_score: Whether to return scores

        Returns:
            List of result lists (and scores if return_score=True)
        """
        return await self._abatch_search(query_list, num, return_score)


class AsyncDenseRetriever(AsyncBaseRetriever):
    """Async dense retriever based on Qdrant.

    Performs vector similarity search using dense embeddings.
    Supports multi-GPU parallel encoding and efficient HNSW search.
    """

    @staticmethod
    async def wait_qdrant_load(url, connect_timeout):
        """Wait for Qdrant server to be ready.

        Args:
            url: Qdrant server URL
            connect_timeout: Maximum time to wait (seconds)

        Returns:
            Connected AsyncQdrantClient instance

        Raises:
            AssertionError: If timeout is exceeded
        """
        client = AsyncQdrantClient(url=url, prefer_grpc=True, timeout=60)
        wait_collection_time = 0
        while True:
            if wait_collection_time >= connect_timeout:
                assert False, f"Qdrant connection timeout after {connect_timeout}s"
            print(f"Waiting {wait_collection_time}s for Qdrant to load...")
            time.sleep(5)
            wait_collection_time += 5
            try:
                await client.info()
                print(f"Qdrant loaded and connected successfully")
                break
            except Exception as e:
                pass  # Continue waiting
        return client

    def __init__(self, config: "Config"):
        """Initialize dense retriever.

        Args:
            config: Configuration object
        """
        super().__init__(config)

    async def ainit(self, config: "Config"):
        """Async initialization (load model and connect to Qdrant).

        Args:
            config: Configuration object
        """
        # Connect to Qdrant
        self.client = await self.wait_qdrant_load(url=config.qdrant_url, connect_timeout=300)

        # Verify collection exists
        self.collection_name = config.qdrant_collection_name
        collections = (await self.client.get_collections()).collections
        collection_names = [col.name for col in collections]
        assert self.collection_name in collection_names, \
            f"Collection '{self.collection_name}' not found. Available: {collection_names}"

        # Initialize encoder pool with all available GPUs
        devices = [torch.device(f"cuda:{i}") for i in range(0, torch.cuda.device_count())]
        print(f"Initializing encoder pool with {len(devices)} GPUs: {devices}")
        self.encoder = AsyncEncoderPool(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            devices=devices,
        )

        # Configure search parameters
        self.topk = config.retrieval_topk
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
        print(f"Qdrant search params: {self.search_params}")

    async def _asearch(self, query: str, num: int = None, return_score: bool = False):
        """Search for a single query.

        Args:
            query: Query string
            num: Number of results to return
            return_score: Whether to return similarity scores

        Returns:
            List of document payloads (and scores if return_score=True)
        """
        time_start = time.time()
        if num is None:
            num = self.topk

        # Encode query
        query_emb = await self.encoder.encode(query)
        query_vector = query_emb[0].tolist()
        time_embed = time.time()

        # Search in Qdrant
        search_results = (
            await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=num,
                search_params=self.search_params,
            )
        ).points
        time_search = time.time()

        # Log timing
        time_elapse_search = time_search - time_embed
        time_elapse_embed = time_embed - time_start
        print(f"Search timing - embed: {time_elapse_embed:.3f}s, search: {time_elapse_search:.3f}s")

        # Handle empty results
        if len(search_results) < 1:
            if return_score:
                return [], []
            else:
                return []

        # Extract payloads and scores
        payloads = [result.payload for result in search_results]
        scores = [result.score for result in search_results]
        if return_score:
            return payloads, scores
        else:
            return payloads

    async def _abatch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if return_score:
            all_payloads, all_scores = [], []
            for query in query_list:
                payloads, scores = await self._asearch(query, num, return_score)
                all_payloads.append(payloads)
                all_scores.append(scores)
            return all_payloads, all_scores
        else:
            all_payloads = []
            for query in query_list:
                payloads = await self._asearch(query, num, return_score)
                all_payloads.append(payloads)
            return all_payloads


async def get_retriever(config):
    """Create and initialize async retriever.

    Args:
        config: Configuration object

    Returns:
        Initialized AsyncDenseRetriever instance
    """
    retriever = AsyncDenseRetriever(config)
    await retriever.ainit(config)
    return retriever


# ============================================================================
# Page Access Utility
# ============================================================================

class PageAccess:
    """Page content accessor.

    Loads and provides access to page content by URL.
    Used by the /access endpoint to retrieve full page content.
    """

    def __init__(self, pages_path):
        """Load page data from JSONL file.

        Args:
            pages_path: Path to pages JSONL file
        """
        pages = []
        for line in tqdm(open(pages_path, "r"), desc="Loading pages"):
            pages.append(json.loads(line))
        self.pages = {page["url"]: page for page in pages}
        print(f"Loaded {len(self.pages)} pages")

    def access(self, url):
        """Access page content by URL.

        Args:
            url: Page URL

        Returns:
            Page data dict or None if not found
        """
        # Normalize PHP URL format
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")

        if url not in self.pages:
            return None
        return self.pages[url]

# ============================================================================
# FastAPI Server Configuration and Endpoints
# ============================================================================

class Config:
    """Configuration class containing all server parameters."""

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

class QueryRequest(BaseModel):
    """Query request model for /retrieve endpoint."""
    queries: List[str]  # List of query strings
    topk: Optional[int] = None  # Number of results per query
    return_scores: bool = False  # Whether to return similarity scores


class AccessRequest(BaseModel):
    """Access request model for /access endpoint."""
    urls: List[str]  # List of page URLs to access


# Initialize FastAPI app
app = FastAPI(
    title="Qdrant Retrieval Server",
    description="REST API for document retrieval and page access",
    version="1.0.0"
)

@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    """Retrieval endpoint: search for relevant documents."""
    time_start = time.time()

    # Use default topk if not specified
    if not request.topk:
        request.topk = config.retrieval_topk

    # Perform batch retrieval
    if request.return_scores:
        results, scores = await retriever.abatch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=request.return_scores
        )
    else:
        results = await retriever.abatch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=request.return_scores
        )

    # Format response
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # Combine documents with scores
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            resp.append(single_result)

    time_elapse = time.time() - time_start
    print(f"Retrieve request: {len(request.queries)} queries, topk={request.topk}, time={time_elapse:.3f}s")
    return {"result": resp}

@app.post("/access")
async def access_endpoint(request: AccessRequest):
    """Access endpoint: retrieve full page content by URL."""
    resp = []
    for url in request.urls:
        resp.append(page_access.access(url))

    print(f"Access request: {len(request.urls)} URLs")
    return {"result": resp}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    set_start_method("spawn")  # Required for CUDA in multiprocessing
    loop = asyncio.get_event_loop()

    parser = argparse.ArgumentParser(
        description="Launch Qdrant retrieval server with REST API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python local_retrieval_server_qdrant.py \\
        --pages_path /path/to/pages.jsonl \\
        --retriever_name e5 \\
        --retriever_model /path/to/e5-model \\
        --qdrant_url http://localhost:6333 \\
        --qdrant_collection_name wiki_collection \\
        --qdrant_search_param '{"hnsw_ef":256}' \\
        --port 8000

API Endpoints:
    POST /retrieve - Search for relevant documents
    POST /access - Retrieve full page content by URL
        """
    )

    parser.add_argument(
        "--pages_path",
        type=str,
        required=True,
        help="Path to pages JSONL file for /access endpoint"
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Default number of results to return per query"
    )
    parser.add_argument(
        "--retriever_name",
        type=str,
        default="e5",
        help="Name of the retriever model (e.g., e5, bge, dpr)"
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        required=True,
        help="Path to the retriever model checkpoint"
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL"
    )
    parser.add_argument(
        "--qdrant_collection_name",
        type=str,
        required=True,
        help="Name of the Qdrant collection to use"
    )
    parser.add_argument(
        "--qdrant_search_param",
        type=str,
        default='{"hnsw_ef":128}',
        help='HNSW search parameters as JSON string (e.g., \'{"hnsw_ef":256}\')'
    )
    parser.add_argument(
        "--qdrant_search_quant_param",
        type=str,
        default=None,
        help="Quantization search parameters as JSON string (optional)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--save-address-to",
        type=str,
        default=None,
        help="Directory to save server address file (optional)"
    )

    args = parser.parse_args()

    # Validate required files exist
    if not os.path.exists(args.pages_path):
        print(f"Error: Pages file not found: {args.pages_path}")
        exit(1)

    # Get server address
    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(socket.gethostname())
    port = args.port
    host_addr = f"{host_ip}:{port}"

    print("=" * 80)
    print("Qdrant Retrieval Server")
    print("=" * 80)
    print(f"Server address: {host_addr}")
    print(f"Model: {args.retriever_model}")
    print(f"Collection: {args.qdrant_collection_name}")
    print(f"Qdrant URL: {args.qdrant_url}")
    print("=" * 80)

    # Save server address to file if requested
    if args.save_address_to:
        os.makedirs(args.save_address_to, exist_ok=True)
        address_file = os.path.join(
            args.save_address_to,
            f"Host{host_ip}_Port{port}.txt"
        )
        with open(address_file, "w") as f:
            f.write(host_addr)
        print(f"Server address saved to: {address_file}")

    # Build configuration
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
    )

    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = loop.run_until_complete(get_retriever(config))

    # Run test query to verify retriever is working
    async def test():
        query = 'What is the capital of France?'
        result = await retriever.asearch(query, 1, return_score=False)
        print(f"Test query: {query}")
        print(f"Test result: {result}")
        print("Retriever is ready!")

    loop.run_until_complete(test())

    # Load page access data
    print("\nLoading page data...")
    page_access = PageAccess(args.pages_path)
    print("Page access is ready!")

    # Launch FastAPI server
    print("\n" + "=" * 80)
    print(f"Starting FastAPI server on http://0.0.0.0:{port}")
    print("API Endpoints:")
    print(f"  - POST http://{host_addr}/retrieve")
    print(f"  - POST http://{host_addr}/access")
    print("=" * 80 + "\n")

    server_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=port,
        log_level="warning",
        loop=loop,
    )
    server = uvicorn.Server(server_config)
    loop.run_until_complete(server.serve())
