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

"""Check two image turns against the learner's inputs on a local SGLang server."""

import argparse
import asyncio
import io
import os

import torch
from PIL import Image
from transformers import AutoTokenizer

from rlinf.hybrid_engines.sglang.multimodal.replay import generate_with_capture
from rlinf.scheduler import Cluster
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.utils.multimodal import materialize_multimodal_inputs
from rlinf.workers.rollout.sglang.process import SGLangServerProcess


async def replay_turns(url: str, model: str) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model)
    start, pad, end = tokenizer.convert_tokens_to_ids(
        ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
    )
    ids, spans, images = [], [], []
    history = None
    async with InferenceHTTPClient(url) as client:
        for turn, (size, color) in enumerate(
            [((96, 128), "red"), ((260, 64), "blue")], 1
        ):
            image = io.BytesIO()
            Image.new("RGB", size, color).save(image, format="PNG")
            images.append(image.getvalue())
            ids.extend(
                tokenizer.encode("Describe this image.", add_special_tokens=False)
            )
            ids.extend([start, pad, end])
            spans.append([len(ids) - 2, len(ids) - 1])
            result, row, history = await generate_with_capture(
                client.async_generate,
                ids,
                images,
                {"temperature": 0, "max_new_tokens": 1},
                {"image_spans": spans, "history": history},
                True,
            )
            # Use the expanded input and sampled IDs as the next turn's prefix.
            ids = row["prompt_token_ids"] + result["output_ids"]
            spans = row["image_spans"].copy()
            batch = torch.tensor([ids])
            learner = materialize_multimodal_inputs(
                [row],
                model,
                "cpu",
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                token_space=True,
            )
            torch.testing.assert_close(
                learner["pixel_values"],
                row["pixel_values"].to(torch.bfloat16),
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                learner["image_grid_thw"], row["image_grid_thw"], rtol=0, atol=0
            )
            torch.testing.assert_close(
                learner["multimodal_position_ids"][
                    :, 0, : len(row["prompt_token_ids"])
                ],
                row["mrope_positions"],
                rtol=0,
                atol=0,
            )
            print(
                f"Turn {turn} passed ({len(spans)} images, {len(ids)} tokens).",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Qwen3.5-4B model files")
    model = parser.parse_args().model
    os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"] = (
        "rlinf.hybrid_engines.sglang.multimodal"
    )
    server = SGLangServerProcess(
        {
            "model_path": model,
            "dtype": "bfloat16",
            "tp_size": 1,
            "context_length": 4096,
            "mem_fraction_static": 0.45,
            "disable_cuda_graph": True,
            "disable_radix_cache": True,
            "disable_overlap_schedule": True,
            "attention_backend": "triton",
        },
        Cluster.find_free_port,
        bind_host="127.0.0.1",
        advertise_host="127.0.0.1",
        apply_rlinf_patch=True,
        log_error=print,
    )
    try:
        server.start()
        asyncio.run(replay_turns(server.get_server_url(), model))
    finally:
        server.terminate()
