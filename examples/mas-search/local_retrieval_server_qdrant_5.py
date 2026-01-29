# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import asyncio
import json
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Queue, set_start_method
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    QuantizationSearchParams,
    SearchParams,
)
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_model(model_path: str, use_fp16: bool = False, device=torch.device("cuda")):
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model = model.to(device=device)
    if use_fp16:
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=True, trust_remote_code=True
    )
    return model, tokenizer


def pooling(
    pooler_output, last_hidden_state, attention_mask=None, pooling_method="mean"
):
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(
            ~attention_mask[..., None].bool(), 0.0
        )
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")


class Encoder:
    def __init__(
        self, model_name, model_path, pooling_method, max_length, use_fp16, device
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16
        self.device = device

        self.model, self.tokenizer = load_model(
            model_path=model_path, use_fp16=use_fp16, device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
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
                query_list = [
                    f"Represent this sentence for searching relevant passages: {query}"
                    for query in query_list
                ]

        inputs = self.tokenizer(
            query_list,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(device=self.device) for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs["input_ids"].shape[0], 1), dtype=torch.long
            ).to(inputs["input_ids"].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(
                output.pooler_output,
                output.last_hidden_state,
                inputs["attention_mask"],
                self.pooling_method,
            )
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")

        del inputs, output
        torch.cuda.empty_cache()

        return query_emb


class AsyncEncoderPool:
    @staticmethod
    def set_global_encoder(init_queue: Queue):
        args = init_queue.get()
        assert "global_encoder" not in globals()
        globals()["global_encoder"] = Encoder(*args)

    @staticmethod
    def global_encode(*args, **kwargs):
        assert "global_encoder" in globals()
        encoder: Encoder = globals()["global_encoder"]
        return encoder.encode(*args, **kwargs)

    def __init__(
        self, model_name, model_path, pooling_method, max_length, use_fp16, devices
    ):
        init_queue = Queue()
        for device in devices:
            init_queue.put(
                [model_name, model_path, pooling_method, max_length, use_fp16, device]
            )

        self.encoders = ProcessPoolExecutor(
            max_workers=len(devices),
            initializer=AsyncEncoderPool.set_global_encoder,
            initargs=(init_queue,),
        )

    async def encode(self, query_list: list[str], is_query=True) -> np.ndarray:
        return await loop.run_in_executor(
            self.encoders, AsyncEncoderPool.global_encode, (query_list, is_query)
        )


class AsyncBaseRetriever:
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk

    async def _asearch(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    async def _abatch_search(self, query_list: list[str], num: int, return_score: bool):
        raise NotImplementedError

    async def asearch(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ):
        return await self._asearch(query, num, return_score)

    async def abatch_search(
        self,
        query_list: list[str],
        num: Optional[int] = None,
        return_score: bool = False,
    ):
        return await self._abatch_search(query_list, num, return_score)


class AsyncDenseRetriever(AsyncBaseRetriever):
    @staticmethod
    async def wait_qdrant_load(url, connect_timeout):
        client = AsyncQdrantClient(url=url, prefer_grpc=True, timeout=60)
        # atexit.register(client.close)
        wait_collection_time = 0
        while True:
            if wait_collection_time >= connect_timeout:
                assert False, f"wait longer than {connect_timeout}s, exit"
            print(f"wait {wait_collection_time}s for qdrant load")
            time.sleep(5)
            wait_collection_time += 5
            try:
                await client.info()
                print("qdrant loaded and connected")
                break
            except Exception:
                pass
        return client

    def __init__(self, config: "Config"):
        super().__init__(config)

    async def ainit(self, config: "Config"):
        self.client = await self.wait_qdrant_load(
            url=config.qdrant_url, connect_timeout=300
        )

        self.collection_name = config.qdrant_collection_name
        collections = (await self.client.get_collections()).collections
        collection_names = [col.name for col in collections]
        assert self.collection_name in collection_names

        # Initialize encoder first (needed for building collection)
        devices = [
            torch.device(f"cuda:{i}") for i in range(0, torch.cuda.device_count())
        ]
        self.encoder = AsyncEncoderPool(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            devices=devices,
        )
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
        print(f"qdrant search_params: {self.search_params}")

    async def _asearch(
        self, query: str, num: Optional[int] = None, return_score: bool = False
    ):
        time_start = time.time()
        if num is None:
            num = self.topk
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
        time_elapse_search = time_search - time_embed
        time_elapse_embed = time_embed - time_start
        print(f"time elapse: search: {time_elapse_search}; embed: {time_elapse_embed}")

        if len(search_results) < 1:
            if return_score:
                return [], []
            else:
                return []

        # # Extract IDs and scores
        payloads = [result.payload for result in search_results]
        scores = [result.score for result in search_results]
        if return_score:
            return payloads, scores
        else:
            return payloads

    async def _abatch_search(
        self,
        query_list: list[str],
        num: Optional[int] = None,
        return_score: bool = False,
    ):
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
    retriever = AsyncDenseRetriever(config)
    await retriever.ainit(config)
    return retriever


class PageAccess:
    def __init__(self, pages_path):
        pages = []
        for ff in tqdm(open(pages_path, "r"), desc="PageAccess"):
            pages.append(json.loads(ff))
        self.pages = {page["url"]: page for page in pages}

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
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


class AccessRequest(BaseModel):
    urls: list[str]


app = FastAPI()


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
    if request.return_scores:
        results, scores = await retriever.abatch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=request.return_scores,
        )
    else:
        results = await retriever.abatch_search(
            query_list=request.queries,
            num=request.topk,
            return_score=request.return_scores,
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


if __name__ == "__main__":
    set_start_method("spawn")
    loop = asyncio.get_event_loop()

    parser = argparse.ArgumentParser(description="Launch the local qdrant retriever.")
    parser.add_argument(
        "--pages_path", type=str, default="xxx", help="Local page file."
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of retrieved passages for one query.",
    )
    parser.add_argument(
        "--retriever_name", type=str, default="e5", help="Name of the retriever model."
    )
    parser.add_argument(
        "--retriever_model",
        type=str,
        default="intfloat/e5-base-v2",
        help="Path of the retriever model.",
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default=None,
        help="Qdrant server URL (e.g., http://localhost:6333). If not provided, uses local mode.",
    )
    parser.add_argument(
        "--qdrant_collection_name",
        type=str,
        default="default_collection",
        help="Name of the Qdrant collection.",
    )
    parser.add_argument("--qdrant_search_param", type=str, default={}, help="")
    parser.add_argument("--qdrant_search_quant_param", type=str, default=None, help="")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument(
        "--save-address-to", type=str, help="path to save server address"
    )

    args = parser.parse_args()

    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(socket.gethostname())
    port = args.port

    host_addr = f"{host_ip}:{port}"

    print(f"Server address: {host_addr}")

    if args.save_address_to:
        os.makedirs(args.save_address_to, exist_ok=True)
        with open(
            os.path.join(
                args.save_address_to, "Host" + host_ip + "_" + "IP" + str(port) + ".txt"
            ),
            "w",
        ) as f:
            f.write(host_addr)

    # 1) Build a config (could also parse from arguments).
    #    In real usage, you'd parse your CLI arguments or environment variables.
    config = Config(
        retrieval_method=args.retriever_name,  # or "dense"
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

    # 2) Instantiate a global retriever so it is loaded once and reused.
    retriever = loop.run_until_complete(get_retriever(config))

    # TODO: test:
    async def test():
        query1 = "介绍一下红牛"
        result1 = await retriever.asearch(query1, 1, return_score=False)
        print(f"test1: query: {query1}, result: {result1}")
        query2 = "介绍一下卢布尔雅那"
        result2 = await retriever.asearch(query2, 2, return_score=True)
        print(f"test2: query: {query2}, result: {result2}")
        query3 = ["介绍一下火星", "介绍一下水星"]
        result3 = await retriever.abatch_search(query3, 3, return_score=True)
        print(f"test3: query: {query3}, result: {result3}")
        print("Retriver is ready.")

    loop.run_until_complete(test())

    # 3) Load pages
    if os.path.exists(args.pages_path):
        page_access = PageAccess(args.pages_path)

    print("Page Access is ready.")

    # 4) Launch the server.
    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",
        loop=loop,
    )
    server = uvicorn.Server(config)
    loop.run_until_complete(server.serve())
