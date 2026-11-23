# Copyright 2026 The RLinf Authors.
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

"""Qdrant embeddings and spawned retrieval workers, without downloaded weights."""

import asyncio
import importlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformers import BertConfig, BertModel, BertTokenizer

RETRIEVER_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples/agent/tools/search_local_server_qdrant"
)


@pytest.fixture(scope="module")
def encoder_module():
    sys.path.insert(0, str(RETRIEVER_DIR))
    try:
        yield importlib.import_module("qdrant_encoder")
    finally:
        sys.path.remove(str(RETRIEVER_DIR))


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("retriever-model")
    vocab = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "[MASK]",
        "query",
        "passage",
        ":",
        "mars",
        "planet",
    ]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    BertTokenizer(str(path / "vocab.txt")).save_pretrained(path)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        BertModel(
            BertConfig(
                vocab_size=len(vocab),
                hidden_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                intermediate_size=32,
            )
        ).save_pretrained(path)
    return str(path)


@pytest.fixture(params=["cpu", "cuda", "npu"])
def device(request, encoder_module):
    name = request.param
    if name == "npu":
        pytest.importorskip("torch_npu")
    if name != "cpu":
        backend = encoder_module.get_device_module(name)
        if not backend.is_available():
            pytest.skip(f"{name} hardware is unavailable")
        return encoder_module.get_devices(name)[0]
    return "cpu"


def test_embeddings_preserve_query_and_passage_contract(
    encoder_module, model_path, device
):
    encoder = encoder_module.Encoder("e5", model_path, "mean", 32, False, device)
    queries = ["mars", "mars planet"]
    query_embeddings = encoder.encode(queries)
    passage_embeddings = encoder.encode(queries, is_query=False)
    assert query_embeddings.shape == (2, 16)
    assert query_embeddings.dtype == np.float32
    assert query_embeddings.flags.c_contiguous
    assert np.isfinite(query_embeddings).all()
    np.testing.assert_allclose(np.linalg.norm(query_embeddings, axis=1), 1, atol=1e-5)
    np.testing.assert_allclose(encoder.encode("mars"), query_embeddings[:1], atol=1e-5)
    assert not np.allclose(query_embeddings, passage_embeddings)


def test_spawned_encoder_preserves_batch_and_query_mode(
    encoder_module, model_path, device
):
    pytest.importorskip("qdrant_client")
    pytest.importorskip("fastapi")
    pytest.importorskip("uvicorn")
    server = importlib.import_module("local_retrieval_server")
    expected_encoder = encoder_module.Encoder(
        "e5", model_path, "mean", 32, False, device
    )
    queries = ["mars", "mars planet"]
    expected = expected_encoder.encode(queries, is_query=False)
    pool = server.AsyncEncoderPool("e5", model_path, "mean", 32, False, [device])
    try:
        actual = asyncio.run(pool.encode(queries, is_query=False))
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        single = asyncio.run(pool.encode("mars"))
        np.testing.assert_allclose(single, expected_encoder.encode("mars"), atol=1e-5)
    finally:
        pool.encoders.shutdown(wait=True)


def test_no_visible_device_fails_before_loading_model(encoder_module, monkeypatch):
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="No cuda devices"):
        encoder_module.get_devices("cuda")


def test_index_and_http_retrieval(model_path, device, tmp_path):
    """Opt in with QDRANT_TEST_URL; each run owns a separate collection."""
    import json
    import os
    import signal
    import socket
    import subprocess
    import time
    import urllib.error
    import urllib.request
    import uuid

    url = os.environ.get("QDRANT_TEST_URL")
    if not url or device == "cpu":
        pytest.skip("Set QDRANT_TEST_URL and provide CUDA or NPU hardware")
    qdrant = pytest.importorskip("qdrant_client")
    client = qdrant.QdrantClient(url=url)
    collection = "rlinf_retriever_test_" + uuid.uuid4().hex
    model = os.environ.get("QDRANT_TEST_MODEL", model_path)
    corpus = [
        {"contents": text, "url": f"https://example.org/{i}"}
        for i, text in enumerate(
            [
                "Mars is a planet in the Solar System.",
                "Mercury is the planet closest to the Sun.",
                "Red Bull is an energy drink.",
                "Ljubljana is the capital of Slovenia.",
            ]
        )
    ]
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text("".join(json.dumps(row) + "\n" for row in corpus))
    common = [
        "--device_type",
        device.split(":")[0],
        "--retriever_model",
        model,
        "--qdrant_url",
        url,
        "--qdrant_collection_name",
        collection,
    ]
    with socket.socket() as port_socket:
        port_socket.bind(("127.0.0.1", 0))
        port = port_socket.getsockname()[1]
    process = None
    try:
        built = subprocess.run(
            [
                sys.executable,
                str(RETRIEVER_DIR / "build_index.py"),
                *common,
                "--corpus_path",
                str(corpus_path),
                "--hnsw_config",
                "{}",
                "--build_parallel",
                "2",
                "--batch_size",
                "2",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert built.returncode == 0, built.stdout + built.stderr
        assert client.count(collection_name=collection, exact=True).count == len(corpus)
        with (tmp_path / "server.log").open("w+") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    str(RETRIEVER_DIR / "local_retrieval_server.py"),
                    *common,
                    "--pages_path",
                    str(corpus_path),
                    "--port",
                    str(port),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            endpoint = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 180
            while True:
                try:
                    with urllib.request.urlopen(endpoint + "/openapi.json", timeout=2):
                        break
                except (urllib.error.URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() > deadline:
                        log.seek(0)
                        pytest.fail(log.read())
                    time.sleep(1)

            def post(path, body):
                request = urllib.request.Request(
                    endpoint + path,
                    data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    return json.load(response)["result"]

            default = post("/retrieve", {"queries": ["Mars", "Mercury"]})
            assert [len(items) for items in default] == [3, 3]
            scored = post(
                "/retrieve", {"queries": ["Mars"], "topk": 2, "return_scores": True}
            )
            assert len(scored[0]) == 2
            assert all(
                np.isfinite(item["score"]) and item["document"] in corpus
                for item in scored[0]
            )
            assert post("/access", {"urls": [corpus[0]["url"], "missing"]}) == [
                corpus[0],
                None,
            ]
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        client.delete_collection(collection_name=collection)
        client.close()
