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

"""Verify two-turn span-aligned image rollout on a local SGLang server."""

import argparse
import asyncio
import io
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor

from rlinf.hybrid_engines.sglang.multimodal import (
    IMAGE_SPANS_PARAM,
    ensure_multimodal_processor_env,
)
from rlinf.scheduler import Cluster
from rlinf.utils.http_client import InferenceHTTPClient
from rlinf.utils.logging import get_logger
from rlinf.utils.multimodal import materialize_multimodal_inputs
from rlinf.workers.agent.agent_loop import MultiModalTurnState
from rlinf.workers.rollout.sglang import sglang_version
from rlinf.workers.rollout.sglang.process import SGLangServerProcess

logger = get_logger()


async def check_turns(url: str, model: str, *, panel: int = 0) -> list[dict]:
    """Run two span-aligned image turns on SGLang and collect training rows."""
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    image_cache: dict = {}
    rows = []
    async with InferenceHTTPClient(url) as client:
        state = MultiModalTurnState(
            processor, [], pixel_dtype=torch.bfloat16, image_cache=image_cache
        )
        for turn, (size, color) in enumerate(
            [((96, 128), "red"), ((260, 64), "blue")], 1
        ):
            buf = io.BytesIO()
            Image.new("RGB", (size[0] + 16 * panel, size[1]), color).save(
                buf, format="PNG"
            )
            state.append_text("Describe this image.")
            if turn == 2:
                # Include literal visual marker tokens in plain text to test span isolation.
                state.append_text("<|vision_start|><|image_pad|><|vision_end|>")
            state.append_image(buf.getvalue())
            prompt_ids = list(state.prompt_ids)
            kwargs = state.generate_kwargs()
            result = await client.async_generate(
                input_ids=prompt_ids,
                image_data=kwargs["image_data"],
                sampling_params={
                    "temperature": 0,
                    "max_new_tokens": 1,
                    "custom_params": {IMAGE_SPANS_PARAM: kwargs["image_spans"]},
                },
                return_logprob=True,
            )
            turn_out = state.record_turn(
                {
                    "output_ids": result["output_ids"],
                    "logprobs": [
                        item[0] for item in result["meta_info"]["output_token_logprobs"]
                    ],
                }
            )
            seq = torch.tensor([turn_out.prompt_ids + turn_out.response_ids])
            materialized = materialize_multimodal_inputs(
                [turn_out.multi_modal_inputs],
                "cpu",
                input_ids=seq,
                attention_mask=torch.ones_like(seq),
                aligned_mode=True,
            )
            assert int((materialized["mm_token_type_ids"] != 0).sum().item()) == sum(
                end - start for start, end in kwargs["image_spans"]
            )
            assert (
                materialized["image_grid_thw"].tolist()
                == turn_out.multi_modal_inputs["image_grid_thw"].tolist()
            )
            rows.append(
                {
                    "prompt_ids": turn_out.prompt_ids,
                    "output_ids": turn_out.response_ids,
                    "logprobs": turn_out.response_logprobs,
                    "multi_modal_inputs": turn_out.multi_modal_inputs,
                }
            )
            logger.info(
                "Turn %d passed (%d images, %d tokens).",
                turn,
                len(kwargs["image_spans"]),
                len(turn_out.prompt_ids),
            )
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to Qwen3.5-4B model files")
    parser.add_argument(
        "--rows", type=Path, help="Save four two-turn episodes for the learner check"
    )
    args = parser.parse_args()
    ensure_multimodal_processor_env(sglang_version)
    server = SGLangServerProcess(
        {
            "model_path": args.model,
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
        log_error=logger.error,
    )
    try:
        server.start()
        if args.rows:
            groups = [
                asyncio.run(check_turns(server.get_server_url(), args.model, panel=i))
                for i in range(4)
            ]
            torch.save(groups, args.rows)
        else:
            asyncio.run(check_turns(server.get_server_url(), args.model))
    finally:
        server.terminate()
