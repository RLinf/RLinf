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
from types import ModuleType
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def get_device_module(device_type: str) -> ModuleType:
    """Return the torch backend, registering Ascend only when requested."""
    if device_type == "npu":
        import torch_npu  # noqa: F401

    return torch.get_device_module(device_type)


def get_devices(device_type: str) -> list[str]:
    """List visible encoder devices as names safe to send to spawned workers."""
    backend = get_device_module(device_type)
    count = backend.device_count()
    if count == 0:
        raise RuntimeError(f"No {device_type} devices are visible to the retriever")
    return [f"{device_type}:{index}" for index in range(count)]


def load_model(
    model_path: str, use_fp16: bool = False, device=torch.device("cuda")
) -> tuple[Any, Any]:
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
        # Register the backend before constructing an NPU device in each child.
        self.torch_platform = get_device_module(str(device).split(":")[0])
        self.device = torch.device(device)
        if self.device.type != "cpu":
            self.torch_platform.set_device(self.device)

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
        if self.device.type != "cpu":
            self.torch_platform.empty_cache()

        return query_emb
