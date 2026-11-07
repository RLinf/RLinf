# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ActiveVLN request, response and weight mapping for native SRT rollout."""

from collections.abc import Iterator
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.models.embodiment.activevln import get_model


class ActiveVLNSGLangEngineAdapter:
    """Share a CPU policy's preprocessing and trajectory layout with the actor."""

    def __init__(self, cfg: DictConfig) -> None:
        model_cfg = (
            cfg.rollout.model if cfg.runner.get("only_eval", False) else cfg.actor.model
        )
        self.model_cfg = OmegaConf.merge(model_cfg, cfg.rollout.model)
        self.model = get_model(self.model_cfg).cpu().eval()

    def get_engine_args(self) -> dict[str, Any]:
        """Return the checkpoint, precision and context length owned by the model."""
        return {
            "model_path": self.model_cfg.model_path,
            "dtype": str(next(self.model.parameters()).dtype).removeprefix("torch."),
            "context_length": self.model.max_context,
        }

    def named_weights(
        self, *, trainable_only: bool
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Map CPU actor parameters to native SRT checkpoint names."""
        return self.model.named_srt_weights(trainable_only=trainable_only)

    def build_request(
        self, env_obs: dict[str, Any], *, sampling_params: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build Engine.generate arguments and retain inputs for actor replay."""
        temperature = (
            float(sampling_params.get("temperature", 1.0))
            if sampling_params.get("do_sample", True)
            else 0.0
        )
        top_p = float(sampling_params.get("top_p", 1.0)) if temperature > 0 else 1.0
        repetition_penalty = float(sampling_params.get("repetition_penalty", 1.0))
        prompts = [
            self.model.prepare_observation(messages, images)
            for messages, images in zip(
                env_obs["messages"], env_obs["images"], strict=True
            )
        ]
        request = {
            "input_ids": [ids[0].tolist() for ids, _, _ in prompts],
            "image_data": [
                [dict(encoded, format="processor_output")] for _, _, encoded in prompts
            ],
            "sampling_params": {
                "temperature": temperature,
                "top_k": -1,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "max_new_tokens": self.model.max_new_tokens,
            },
            "return_logprob": True,
        }
        state = {
            "prompts": prompts,
            "sampling": (
                temperature or 1.0,
                top_p,
                repetition_penalty,
                float(temperature > 0),
            ),
        }
        return request, state

    def parse_response(
        self, outputs: list[dict[str, Any]], state: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Check generated token alignment and pack the actor's replay inputs."""
        prompts = state["prompts"]
        responses, logprobs = [], []
        for (ids, _, _), output in zip(prompts, outputs, strict=True):
            meta = output["meta_info"]
            tokens = [item[1] for item in meta["output_token_logprobs"]]
            if meta["prompt_tokens"] != ids.shape[1]:
                raise ValueError(
                    "SRT prompt length differs from the actor's image/token layout"
                )
            if "output_ids" in output and output["output_ids"] != tokens:
                raise ValueError("SRT output tokens and logprob tokens differ")
            responses.append(torch.tensor(tokens, dtype=torch.long))
            logprobs.append(
                torch.tensor([item[0] for item in meta["output_token_logprobs"]])
            )
        return self.model.pack_rollout(
            prompts, responses, logprobs, sampling=state["sampling"]
        )
