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

"""vLLM prompt-logprob scoring independent of Student weight synchronization."""

from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets

from .backend import TeacherModelInfo, TeacherScoringBackend


class VLLMTeacherBackend(TeacherScoringBackend):
    """Normalize prompt scores, discarding the trigger generation."""

    async def initialize(self) -> TeacherModelInfo:
        import os

        # The owning Worker may already have initialized CUDA before engine launch.
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        info = self.load_model_info()
        options = dict(self.config.inference.get("engine_kwargs", {}))
        options.update(
            model=self.config.model_path,
            tensor_parallel_size=self.config.inference.get("tensor_parallel_size", 1),
            pipeline_parallel_size=self.config.inference.get(
                "pipeline_parallel_size", 1
            ),
            max_model_len=info.max_context_length,
        )
        options.setdefault("max_logprobs", self.model_vocab_size)
        self.engine = AsyncLLM.from_engine_args(AsyncEngineArgs(**options))
        return info

    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        self.validate_request(request)
        if request.selected_token_ids is not None:
            return await self.score_response_tokens(request)
        full = request.topk == -1
        count = self.info.vocab_size if full else request.topk or 0
        output = None
        async for output in self.engine.generate(
            TokensPrompt(prompt_token_ids=list(request.input_ids)),
            SamplingParams(
                temperature=1.0, max_tokens=1, prompt_logprobs=count, detokenize=False
            ),
            request.request_id,
        ):
            pass
        if output is None or output.prompt_logprobs is None:
            raise ValueError(
                f"Teacher request {request.request_id}: engine returned no prompt scores"
            )
        rows = [
            None
            if row is None
            else {int(i): float(value.logprob) for i, value in row.items()}
            for row in output.prompt_logprobs
        ]
        return self.normalize(request, rows)

    async def score_next(
        self, prefix: tuple[int, ...], selected_ids: tuple[int, ...], request_id: str
    ) -> dict[int, float]:
        from vllm import SamplingParams
        from vllm.inputs import TokensPrompt

        # Older supported vLLM versions have no selected-token scoring API.
        count = self.model_vocab_size or self.info.vocab_size
        output = None
        async for output in self.engine.generate(
            TokensPrompt(prompt_token_ids=list(prefix)),
            SamplingParams(
                temperature=1.0, max_tokens=1, logprobs=count, detokenize=False
            ),
            request_id,
        ):
            pass
        if output is None or not output.outputs or not output.outputs[0].logprobs:
            raise ValueError(f"Teacher request {request_id}: missing next-token scores")
        row = output.outputs[0].logprobs[0]
        if any(i not in row for i in selected_ids):
            raise ValueError(
                f"Teacher request {request_id}: missing selected-token scores"
            )
        return {i: float(row[i].logprob) for i in selected_ids}

    async def close(self) -> None:
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.shutdown()
