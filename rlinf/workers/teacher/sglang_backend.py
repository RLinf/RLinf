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

"""SGLang native prompt scoring for a frozen Teacher."""

from rlinf.data.schema.distillation import TeacherScoreRequest, TeacherTargets

from .backend import TeacherModelInfo, TeacherScoringBackend


class SGLangTeacherBackend(TeacherScoringBackend):
    """Use input logprobs rather than generated-token logprobs."""

    async def initialize(self) -> TeacherModelInfo:
        from sglang import Engine
        from sglang.srt.server_args import ServerArgs

        info = self.load_model_info()
        options = dict(self.config.inference.get("engine_kwargs", {}))
        options.update(
            model_path=self.config.model_path,
            tp_size=self.config.inference.get("tensor_parallel_size", 1),
            pp_size=self.config.inference.get("pipeline_parallel_size", 1),
            context_length=info.max_context_length,
        )
        self.supports_position_scores = (
            "selected_token_logprob_vocab_size" in ServerArgs.__dataclass_fields__
        )
        if self.supports_position_scores:
            options["selected_token_logprob_vocab_size"] = info.vocab_size
        self.engine = Engine(**options)
        return info

    async def score(self, request: TeacherScoreRequest) -> TeacherTargets:
        self.validate_request(request)
        if request.selected_token_ids is not None:
            return await self.score_selected(request)
        full = request.topk == -1
        count = self.info.vocab_size if full else request.topk or 0
        output = await self.engine.async_generate(
            input_ids=list(request.input_ids),
            sampling_params={"temperature": 1.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=count,
        )
        meta = output["meta_info"]
        if request.topk is None:
            rows = [
                None if item[0] is None else {int(item[1]): float(item[0])}
                for item in meta["input_token_logprobs"]
            ]
        else:
            rows = [
                None if not row else {int(item[1]): float(item[0]) for item in row}
                for row in meta["input_top_logprobs"]
            ]
        return self.normalize(request, rows)

    async def score_selected(self, request: TeacherScoreRequest) -> TeacherTargets:
        """Score fixed rollout support in one prefill, gathering inside the engine."""
        if not getattr(self, "supports_position_scores", False):
            raise ValueError(
                "Student top-k requires the SGLang per-position scoring patch; "
                "run bash requirements/patches/apply_sglang.sh in a SGLang 0.5.12.post1 environment"
            )
        width = request.topk
        # Prefill row t predicts input token t+1; the final row has no target.
        ids = [i for row in request.selected_token_ids[1:] for i in row] + [0] * width
        output = await self.engine.async_generate(
            input_ids=list(request.input_ids),
            sampling_params={
                "temperature": 1.0,
                "max_new_tokens": 0,
                "custom_params": {"rlinf_token_ids_logprob_k": width},
            },
            return_logprob=True,
            logprob_start_len=0,
            top_logprobs_num=0,
            token_ids_logprob=ids,
        )
        scores = output["meta_info"]["input_token_ids_logprobs"]
        if len(scores) != len(request.input_ids):
            raise ValueError(
                "Teacher prefill scores must match the original input length"
            )
        rows = []
        for position, row in enumerate(scores):
            if not request.response_mask[position] or position == 0:
                rows.append(None)
                continue
            actual_ids = tuple(int(item[1]) for item in row)
            if actual_ids != tuple(request.selected_token_ids[position]):
                raise ValueError(
                    f"Teacher request {request.request_id}: per-position scoring IDs differ at position {position}"
                )
            rows.append({int(item[1]): float(item[0]) for item in row})
        return self.normalize(request, rows)

    async def score_next(
        self, prefix: tuple[int, ...], selected_ids: tuple[int, ...], request_id: str
    ) -> dict[int, float]:
        output = await self.engine.async_generate(
            input_ids=list(prefix),
            # Greedy scoring uses raw log_softmax, avoiding softmax underflow.
            sampling_params={"temperature": 0.0, "max_new_tokens": 1},
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=0,
            token_ids_logprob=list(selected_ids),
        )
        rows = output["meta_info"]["output_token_ids_logprobs"]
        if not rows:
            raise ValueError(f"Teacher request {request_id}: missing next-token scores")
        return {int(item[1]): float(item[0]) for item in rows[0]}

    async def close(self) -> None:
        if self.engine is not None:
            engine, self.engine = self.engine, None
            engine.shutdown()
