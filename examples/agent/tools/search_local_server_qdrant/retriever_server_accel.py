# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hardware-adaptive Qdrant retriever server.

The original version hard-coded CUDA devices. This version can run on:
  - NVIDIA GPU: cuda:0, cuda:1, ...
  - Ascend NPU:  npu:0, npu:1, ...
  - CPU fallback, mainly for local debugging

Examples:
  # Auto-detect Ascend first, then CUDA, then CPU.
  python retriever_server_accel.py \
      --retriever_model /path/to/model \
      --qdrant_collection_name my_collection

  # Force Ascend NPU 0,1,2,3.
  python retriever_server_accel.py \
      --device-backend npu --devices 0,1,2,3 \
      --retriever_model /path/to/model \
      --qdrant_collection_name my_collection

  # Force NVIDIA GPU 0,1.
  python retriever_server_accel.py \
      --device-backend cuda --devices 0,1 \
      --retriever_model /path/to/model \
      --qdrant_collection_name my_collection
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import QuantizationSearchParams, SearchParams
from qdrant_encoder import Encoder
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)

# These globals are intentionally initialized once per encoder worker process.
_GLOBAL_ENCODER: Optional[Encoder] = None
_GLOBAL_ENCODER_DEVICE: Optional[str] = None


@dataclass(frozen=True)
class DeviceSpec:
    """A normalized device descriptor that is safe to pass to child processes."""

    backend: str  # "cuda", "npu", or "cpu"
    index: Optional[int] = None

    @property
    def device_str(self) -> str:
        if self.backend == "cpu":
            return "cpu"
        if self.index is None:
            return self.backend
        return f"{self.backend}:{self.index}"

    def torch_device(self) -> torch.device:
        return torch.device(self.device_str)


def _import_torch_npu() -> bool:
    """Import torch_npu if available. Return whether NPU support is importable."""
    try:
        import torch_npu  # noqa: F401

        return True
    except Exception:
        return False


def _get_npu_module():
    """Return the NPU module exposed by torch_npu, or None if unavailable."""
    try:
        import torch_npu  # type: ignore
    except Exception:
        return None

    # Most recent torch_npu versions expose torch.npu after import torch_npu.
    # Some older code paths also expose torch_npu.npu, so keep both.
    return getattr(torch, "npu", None) or getattr(torch_npu, "npu", None)


def _device_count(backend: str) -> int:
    if backend == "cuda":
        return torch.cuda.device_count() if torch.cuda.is_available() else 0
    if backend == "npu":
        npu_mod = _get_npu_module()
        if npu_mod is None:
            return 0
        try:
            return int(npu_mod.device_count())
        except Exception:
            return 0
    if backend == "cpu":
        return 1
    raise ValueError(f"Unsupported backend: {backend}")


def _backend_available(backend: str) -> bool:
    return _device_count(backend) > 0


def _detect_backend(preferred: str) -> str:
    """
    Detect the accelerator backend.

    In auto mode we prefer Ascend NPU when torch_npu is available, then CUDA,
    then CPU. Use --device-backend cuda/npu to force a specific backend.
    """
    preferred = preferred.lower()
    if preferred != "auto":
        if preferred not in {"cuda", "npu", "cpu"}:
            raise ValueError(f"Unsupported device backend: {preferred}")
        if not _backend_available(preferred):
            raise RuntimeError(
                f"Requested backend '{preferred}' is not available. "
                f"npu_count={_device_count('npu')}, "
                f"cuda_count={_device_count('cuda')}, cpu_count=1"
            )
        return preferred

    if _backend_available("npu"):
        return "npu"
    if _backend_available("cuda"):
        return "cuda"
    return "cpu"


def _normalize_device_token(token: str, backend: str) -> DeviceSpec:
    token = token.strip().lower()
    if not token:
        raise ValueError("Empty device token")
    if token == "cpu":
        return DeviceSpec("cpu", None)

    if ":" in token:
        dev = torch.device(token)
        if dev.type not in {"cuda", "npu", "cpu"}:
            raise ValueError(f"Unsupported device type in token: {token}")
        return DeviceSpec(dev.type, dev.index)

    # A bare integer such as "0" means "<selected_backend>:0".
    if not token.isdigit():
        raise ValueError(
            f"Invalid device token '{token}'. Use '0,1' or explicit 'cuda:0,npu:0'."
        )
    if backend == "cpu":
        return DeviceSpec("cpu", None)
    return DeviceSpec(backend, int(token))


def resolve_devices(
    preferred_backend: str = "auto",
    devices_arg: Optional[str] = None,
    num_encoder_workers: Optional[int] = None,
) -> list[DeviceSpec]:
    """Resolve the actual devices used by encoder worker processes."""
    backend = _detect_backend(preferred_backend)

    if devices_arg:
        devices = [_normalize_device_token(x, backend) for x in devices_arg.split(",")]
    else:
        count = _device_count(backend)
        if backend == "cpu":
            devices = [DeviceSpec("cpu", None)]
        else:
            devices = [DeviceSpec(backend, i) for i in range(count)]

    if not devices:
        raise RuntimeError("No valid devices are resolved for encoder workers.")

    # Validate device indices against visible devices. torch device_count respects
    # CUDA_VISIBLE_DEVICES / ASCEND_RT_VISIBLE_DEVICES-like environment filtering.
    for dev in devices:
        if dev.backend == "cpu":
            continue
        count = _device_count(dev.backend)
        if dev.index is None or dev.index < 0 or dev.index >= count:
            raise RuntimeError(
                f"Device {dev.device_str} is not visible. "
                f"Detected {count} visible {dev.backend} device(s)."
            )

    # By default: one encoder process per visible device. If the caller wants more
    # workers than devices, assign devices round-robin. This is useful for CPU, but
    # on NPU/GPU one worker per card is usually the safer default.
    if num_encoder_workers is not None:
        if num_encoder_workers <= 0:
            raise ValueError("--num-encoder-workers must be positive")
        devices = [devices[i % len(devices)] for i in range(num_encoder_workers)]

    return devices


def set_torch_device(device_spec: DeviceSpec) -> None:
    """Set the default device once inside each worker process."""
    if device_spec.backend == "cpu":
        return

    device = device_spec.torch_device()
    if device_spec.backend == "cuda":
        torch.cuda.set_device(device)
        return

    if device_spec.backend == "npu":
        npu_mod = _get_npu_module()
        if npu_mod is None:
            raise RuntimeError(
                "torch_npu is not importable, but an NPU device was requested."
            )
        # For Ascend it is safer to set the process device before constructing
        # the model/encoder and avoid switching devices later in the same process.
        npu_mod.set_device(device)
        return

    raise ValueError(f"Unsupported backend: {device_spec.backend}")


def _encoder_worker_init(init_queue) -> None:
    """Initializer run exactly once in each ProcessPoolExecutor worker."""
    global _GLOBAL_ENCODER, _GLOBAL_ENCODER_DEVICE

    cfg = init_queue.get()
    device_spec = DeviceSpec(cfg["backend"], cfg["index"])
    set_torch_device(device_spec)

    torch_device = device_spec.torch_device()
    LOGGER.info("Initializing Encoder on %s", torch_device)
    _GLOBAL_ENCODER = Encoder(
        cfg["model_name"],
        cfg["model_path"],
        cfg["pooling_method"],
        cfg["max_length"],
        cfg["use_fp16"],
        torch_device,
    )
    _GLOBAL_ENCODER_DEVICE = device_spec.device_str


def _encoder_worker_encode(query_list: list[str], is_query: bool = True) -> np.ndarray:
    if _GLOBAL_ENCODER is None:
        raise RuntimeError("Encoder worker is not initialized.")
    return _GLOBAL_ENCODER.encode(query_list, is_query)


class AsyncEncoderPool:
    def __init__(
        self,
        model_name: str,
        model_path: str,
        pooling_method: str,
        max_length: int,
        use_fp16: bool,
        devices: list[DeviceSpec],
    ):
        if not devices:
            raise ValueError("AsyncEncoderPool requires at least one device.")

        ctx = mp.get_context("spawn")
        init_queue = ctx.Queue()
        for device in devices:
            init_queue.put(
                {
                    "model_name": model_name,
                    "model_path": model_path,
                    "pooling_method": pooling_method,
                    "max_length": max_length,
                    "use_fp16": use_fp16,
                    "backend": device.backend,
                    "index": device.index,
                }
            )

        self.devices = devices
        self.encoders = ProcessPoolExecutor(
            max_workers=len(devices),
            initializer=_encoder_worker_init,
            initargs=(init_queue,),
            mp_context=ctx,
        )
        LOGGER.info(
            "Encoder pool created with %d worker(s): %s",
            len(devices),
            ", ".join(d.device_str for d in devices),
        )

    async def encode(self, query_list: list[str] | str, is_query: bool = True) -> np.ndarray:
        if isinstance(query_list, str):
            query_list = [query_list]
        else:
            query_list = list(query_list)

        loop = asyncio.get_running_loop()
        fn = partial(_encoder_worker_encode, query_list=query_list, is_query=is_query)
        return await loop.run_in_executor(self.encoders, fn)

    def shutdown(self) -> None:
        self.encoders.shutdown(wait=False, cancel_futures=True)


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
        self, query: str, num: int | None = None, return_score: bool = False
    ):
        return await self._asearch(query, num, return_score)

    async def abatch_search(
        self, query_list: list[str], num: int | None = None, return_score: bool = False
    ):
        return await self._abatch_search(query_list, num, return_score)


def _json_loads_dict(raw: Optional[str], field_name: str) -> dict:
    if raw is None or raw == "":
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} is not valid JSON: {raw}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{field_name} must be a JSON object, got: {type(parsed)}")
    return parsed


class AsyncDenseRetriever(AsyncBaseRetriever):
    @staticmethod
    async def wait_qdrant_load(url: str, connect_timeout: int):
        client = AsyncQdrantClient(url=url, prefer_grpc=True, timeout=60)
        wait_collection_time = 0
        while True:
            if wait_collection_time >= connect_timeout:
                raise TimeoutError(f"wait longer than {connect_timeout}s, exit")
            LOGGER.info("wait %ss for qdrant load", wait_collection_time)
            try:
                await client.info()
                LOGGER.info("qdrant loaded and connected")
                return client
            except Exception:
                await asyncio.sleep(5)
                wait_collection_time += 5

    def __init__(self, config: "Config"):
        super().__init__(config)
        self.client: Optional[AsyncQdrantClient] = None
        self.collection_name: Optional[str] = None
        self.encoder: Optional[AsyncEncoderPool] = None
        self.search_params: Optional[SearchParams] = None

    async def ainit(self, config: "Config"):
        self.client = await self.wait_qdrant_load(
            url=config.qdrant_url, connect_timeout=config.qdrant_connect_timeout
        )

        self.collection_name = config.qdrant_collection_name
        collections = (await self.client.get_collections()).collections
        collection_names = [col.name for col in collections]
        if self.collection_name not in collection_names:
            raise RuntimeError(
                f"no collection! exists: [{collection_names}], need: {self.collection_name}"
            )

        from qdrant_client.http.models import Batch
        from qdrant_client.models import CollectionStatus

        coll_status = (await self.client.get_collection(self.collection_name)).status
        if coll_status != CollectionStatus.GREEN:
            wait_collection_time = 0
            optimize_timeout = config.qdrant_optimize_timeout
            await self.client.upsert(
                collection_name=self.collection_name,
                points=Batch(ids=[], vectors=[]),
            )
            LOGGER.info(
                "Optimizers triggered for collection '%s' with an empty update operation.",
                self.collection_name,
            )
            while coll_status != CollectionStatus.GREEN:
                if wait_collection_time >= optimize_timeout:
                    raise TimeoutError(f"wait longer than {optimize_timeout}s, exit")
                await asyncio.sleep(5)
                wait_collection_time += 5
                coll_status = (await self.client.get_collection(self.collection_name)).status
                LOGGER.info(
                    "wait %ss for qdrant optimize, status=%s",
                    wait_collection_time,
                    coll_status,
                )
            LOGGER.info("qdrant optimized")
        else:
            LOGGER.info("collection status is green now")

        devices = resolve_devices(
            preferred_backend=config.device_backend,
            devices_arg=config.devices,
            num_encoder_workers=config.num_encoder_workers,
        )
        LOGGER.info("Using encoder devices: %s", ", ".join(d.device_str for d in devices))
        self.encoder = AsyncEncoderPool(
            model_name=self.retrieval_method,
            model_path=config.retrieval_model_path,
            pooling_method=config.retrieval_pooling_method,
            max_length=config.retrieval_query_max_length,
            use_fp16=config.retrieval_use_fp16,
            devices=devices,
        )

        self.topk = config.retrieval_topk
        search_kwargs = _json_loads_dict(config.qdrant_search_param, "qdrant_search_param")
        quant_kwargs = _json_loads_dict(
            config.qdrant_search_quant_param, "qdrant_search_quant_param"
        )
        if quant_kwargs:
            self.search_params = SearchParams(
                **search_kwargs,
                quantization=QuantizationSearchParams(**quant_kwargs),
            )
        else:
            self.search_params = SearchParams(**search_kwargs)
        LOGGER.info("qdrant search_params: %s", self.search_params)

    async def _asearch(
        self, query: str, num: int | None = None, return_score: bool = False
    ):
        if self.client is None or self.encoder is None or self.search_params is None:
            raise RuntimeError("Retriever is not initialized.")

        time_start = time.time()
        if num is None:
            num = self.topk

        query_emb = await self.encoder.encode([query], is_query=True)
        query_vector = query_emb[0].tolist()
        time_embed = time.time()

        search_results = (
            await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=num,
                search_params=self.search_params,
            )
        ).points
        time_search = time.time()
        LOGGER.info(
            "time elapse: search: %.6f; embed: %.6f",
            time_search - time_embed,
            time_embed - time_start,
        )

        payloads = [result.payload for result in search_results]
        scores = [result.score for result in search_results]
        if return_score:
            return payloads, scores
        return payloads

    async def _abatch_search(
        self,
        query_list: list[str],
        num: int | None = None,
        return_score: bool = False,
    ):
        # Keep Qdrant requests sequential to preserve the original behavior and
        # avoid changing Qdrant version assumptions. Encoder-side concurrency is
        # handled by the process pool.
        if return_score:
            all_payloads, all_scores = [], []
            for query in query_list:
                payloads, scores = await self._asearch(query, num, return_score=True)
                all_payloads.append(payloads)
                all_scores.append(scores)
            return all_payloads, all_scores

        all_payloads = []
        for query in query_list:
            payloads = await self._asearch(query, num, return_score=False)
            all_payloads.append(payloads)
        return all_payloads

    def shutdown(self) -> None:
        if self.encoder is not None:
            self.encoder.shutdown()


async def get_retriever(config):
    retriever = AsyncDenseRetriever(config)
    await retriever.ainit(config)
    return retriever


class PageAccess:
    def __init__(self, pages_path: str):
        pages = []
        with open(pages_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, desc="Loading pages"):
                pages.append(json.loads(line))
        self.pages = {page["url"]: page for page in pages}

    def access(self, url: str):
        if "index.php/" in url:
            url = url.replace("index.php/", "index.php?title=")
        return self.pages.get(url)


class Config:
    """Minimal config class for local Qdrant retrieval server."""

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
        qdrant_connect_timeout: int = 300,
        qdrant_optimize_timeout: int = 3000,
        retrieval_model_path: str = "./model",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        device_backend: str = "auto",
        devices: Optional[str] = None,
        num_encoder_workers: Optional[int] = None,
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.dataset_path = dataset_path
        self.data_split = data_split
        self.qdrant_url = qdrant_url
        self.qdrant_collection_name = qdrant_collection_name
        self.qdrant_search_param = qdrant_search_param
        self.qdrant_search_quant_param = qdrant_search_quant_param
        self.qdrant_connect_timeout = qdrant_connect_timeout
        self.qdrant_optimize_timeout = qdrant_optimize_timeout
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.device_backend = device_backend
        self.devices = devices
        self.num_encoder_workers = num_encoder_workers


class QueryRequest(BaseModel):
    queries: list[str]
    topk: Optional[int] = None
    return_scores: bool = False


class AccessRequest(BaseModel):
    urls: list[str]


app = FastAPI()
runtime_config: Optional[Config] = None
retriever: Optional[AsyncDenseRetriever] = None
page_access: Optional[PageAccess] = None


@app.post("/retrieve")
async def retrieve_endpoint(request: QueryRequest):
    if runtime_config is None or retriever is None:
        raise HTTPException(status_code=503, detail="Retriever is not initialized.")

    time_start = time.time()
    topk = request.topk or runtime_config.retrieval_topk

    if request.return_scores:
        results, scores = await retriever.abatch_search(
            query_list=request.queries,
            num=topk,
            return_score=True,
        )
    else:
        results = await retriever.abatch_search(
            query_list=request.queries,
            num=topk,
            return_score=False,
        )
        scores = None

    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            combined = [
                {"document": doc, "score": score}
                for doc, score in zip(single_result, scores[i])
            ]
            resp.append(combined)
        else:
            resp.append(single_result)

    LOGGER.info("request: %s, time_elapse: %.6f", request, time.time() - time_start)
    return {"result": resp}


@app.post("/access")
async def access_endpoint(request: AccessRequest):
    if page_access is None:
        raise HTTPException(status_code=503, detail="Page access is not enabled.")
    return {"result": [page_access.access(url) for url in request.urls]}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local qdrant retriever.")
    parser.add_argument("--pages_path", type=str, default=None, help="Local page file.")
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
        required=True,
        help="Path of the retriever model.",
    )
    parser.add_argument(
        "--qdrant_url",
        type=str,
        default="http://localhost:6333",
        help="Qdrant server URL.",
    )
    parser.add_argument(
        "--qdrant_collection_name",
        type=str,
        required=True,
        help="Name of the Qdrant collection.",
    )
    parser.add_argument(
        "--qdrant_search_param",
        type=str,
        default="{}",
        help='HNSW search parameters as JSON string, e.g. \'{"hnsw_ef":256}\'.',
    )
    parser.add_argument(
        "--qdrant_search_quant_param",
        type=str,
        default=None,
        help="Quantization search parameters as JSON string. Optional.",
    )
    parser.add_argument(
        "--qdrant_connect_timeout",
        type=int,
        default=300,
        help="Seconds to wait for Qdrant connection.",
    )
    parser.add_argument(
        "--qdrant_optimize_timeout",
        type=int,
        default=3000,
        help="Seconds to wait for Qdrant collection optimization.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--save-address-to",
        type=str,
        default=None,
        help="Directory to save server address file. Optional.",
    )
    parser.add_argument(
        "--device-backend",
        choices=["auto", "cuda", "npu", "cpu"],
        default="auto",
        help="Encoder device backend. auto prefers npu, then cuda, then cpu.",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "Comma-separated devices. Examples: '0,1,2,3', 'cuda:0,cuda:1', "
            "'npu:0,npu:1'. If omitted, all visible devices of the selected backend are used."
        ),
    )
    parser.add_argument(
        "--num-encoder-workers",
        type=int,
        default=None,
        help=(
            "Number of encoder worker processes. Defaults to one process per selected device. "
            "If larger than the device list, devices are assigned round-robin."
        ),
    )
    parser.add_argument(
        "--retrieval-query-max-length",
        type=int,
        default=256,
        help="Maximum query length for the retriever encoder.",
    )
    parser.add_argument(
        "--retrieval-pooling-method",
        type=str,
        default="mean",
        help="Pooling method passed to qdrant_encoder.Encoder.",
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable fp16 encoder inference. By default fp16 is enabled as in the original script.",
    )
    parser.add_argument(
        "--skip-startup-test",
        action="store_true",
        help="Skip the built-in smoke test queries at startup.",
    )
    return parser


def _save_host_address(save_address_to: Optional[str], host_ip: str, port: int) -> None:
    if not save_address_to:
        return
    os.makedirs(save_address_to, exist_ok=True)
    address_path = os.path.join(save_address_to, f"Host{host_ip}_IP{port}.txt")
    with open(address_path, "w", encoding="utf-8") as f:
        f.write(f"{host_ip}:{port}")


async def _startup_smoke_test() -> None:
    if retriever is None:
        raise RuntimeError("Retriever is not initialized.")
    query1 = "Tell me about Red Bull"
    result1 = await retriever.asearch(query1, 1, return_score=False)
    LOGGER.info("test1: query: %s, result: %s", query1, result1)

    query2 = "Tell me about Ljubljana"
    result2 = await retriever.asearch(query2, 2, return_score=True)
    LOGGER.info("test2: query: %s, result: %s", query2, result2)

    query3 = ["Tell me about Mars", "Tell me about Mercury"]
    result3 = await retriever.abatch_search(query3, 3, return_score=True)
    LOGGER.info("test3: query: %s, result: %s", query3, result3)
    LOGGER.info("Retriever is ready.")


async def main_async(args: argparse.Namespace) -> None:
    global runtime_config, retriever, page_access

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(processName)s] %(name)s: %(message)s",
    )

    # Trigger backend detection early so misconfigured jobs fail before Qdrant init.
    devices = resolve_devices(
        preferred_backend=args.device_backend,
        devices_arg=args.devices,
        num_encoder_workers=args.num_encoder_workers,
    )
    LOGGER.info("Resolved encoder devices: %s", ", ".join(d.device_str for d in devices))

    host_name = socket.gethostname()
    host_ip = socket.gethostbyname(host_name)
    LOGGER.info("Server address: %s:%s", host_ip, args.port)
    _save_host_address(args.save_address_to, host_ip, args.port)

    runtime_config = Config(
        retrieval_method=args.retriever_name,
        retrieval_topk=args.topk,
        qdrant_url=args.qdrant_url,
        qdrant_collection_name=args.qdrant_collection_name,
        qdrant_search_param=args.qdrant_search_param,
        qdrant_search_quant_param=args.qdrant_search_quant_param,
        qdrant_connect_timeout=args.qdrant_connect_timeout,
        qdrant_optimize_timeout=args.qdrant_optimize_timeout,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=args.retrieval_pooling_method,
        retrieval_query_max_length=args.retrieval_query_max_length,
        retrieval_use_fp16=not args.no_fp16,
        device_backend=args.device_backend,
        devices=args.devices,
        num_encoder_workers=args.num_encoder_workers,
    )

    retriever = await get_retriever(runtime_config)

    if not args.skip_startup_test:
        await _startup_smoke_test()
    else:
        LOGGER.info("Startup smoke test skipped.")

    if not args.pages_path:
        LOGGER.info("Page Access is off.")
    elif not os.path.exists(args.pages_path):
        LOGGER.info("Page Access is not loaded because pages_path(%s) does not exist.", args.pages_path)
    else:
        page_access = PageAccess(args.pages_path)
        LOGGER.info("Page Access is ready.")

    server_config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
        loop="asyncio",
    )
    server = uvicorn.Server(server_config)
    LOGGER.info("Server is ready at port %s", args.port)
    try:
        await server.serve()
    finally:
        if retriever is not None:
            retriever.shutdown()


def main() -> None:
    # Explicit spawn is safer for CUDA/NPU runtime initialization in subprocesses.
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = build_arg_parser()
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
