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

"""Teacher score contracts and optional real inference-engine comparison."""

import asyncio
import os
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from rlinf.algorithms.distillation.config import TeacherModelConfig
from rlinf.data.schema.distillation import TeacherScoreRequest, teacher_targets_to_batch
from rlinf.workers.teacher.backend import TeacherModelInfo, tokenizer_fingerprint
from rlinf.workers.teacher.client import TeacherClient, TeacherEndpoint
from rlinf.workers.teacher.sglang_backend import SGLangTeacherBackend
from rlinf.workers.teacher.vllm_backend import VLLMTeacherBackend


def request(k=None, selected=None):
    return TeacherScoreRequest(
        "request",
        "sample",
        "group",
        "default",
        (1, 3, 2),
        (False, True, True),
        k,
        "tokenizer",
        selected,
        "step-0" if selected is not None else None,
    )


class TestTeacher(unittest.TestCase):
    def setUp(self):
        # No inference engine is initialized; exercise the real protocol adapter.
        self.backend = VLLMTeacherBackend(TeacherModelConfig("unused"), "default")
        self.backend.info = TeacherModelInfo("model", "tokenizer", 5, 32)
        self.probs = torch.tensor(
            [[0.1, 0.2, 0.3, 0.25, 0.15], [0.3, 0.2, 0.1, 0.15, 0.25]]
        )
        self.rows = [None] + [
            {i: float(value.log()) for i, value in enumerate(row)} for row in self.probs
        ]

    def test_tokenizer_padding_normalization_preserves_identity(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import AutoTokenizer, GPT2Config, PreTrainedTokenizerFast

        from rlinf.models.tokenization import hf_tokenizer

        with tempfile.TemporaryDirectory() as path:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(5)})),
                eos_token="t2",
            )
            tokenizer.save_pretrained(path)
            GPT2Config(vocab_size=8).save_pretrained(path)
            raw = AutoTokenizer.from_pretrained(path)
            actor = hf_tokenizer(path)
            backend = VLLMTeacherBackend(TeacherModelConfig(path), "default")
            info = backend.load_model_info()
            self.assertEqual(tokenizer_fingerprint(raw), info.tokenizer_fingerprint)
            self.assertEqual(tokenizer_fingerprint(actor), info.tokenizer_fingerprint)
            self.assertIsNone(raw.pad_token)
            raw.pad_token = "t3"
            self.assertNotEqual(tokenizer_fingerprint(raw), info.tokenizer_fingerprint)
            raw.pad_token = "t2"
            raw.add_tokens(["new-token"])
            self.assertNotEqual(tokenizer_fingerprint(raw), info.tokenizer_fingerprint)

    def test_rollout_topk_preserves_model_output_ids_and_positions(self):
        from rlinf.workers.teacher.scoring import select_rollout_topk

        rows = [
            [(-0.1, 7, None), (-0.3, 2, None), (-0.2, 1, None)],
            [(-0.1, 7, None), (-0.4, 0, None), (-0.2, 2, None)],
        ]
        self.assertEqual(select_rollout_topk(rows, 2, 2), ((7, 1), (7, 2)))
        vllm_rows = [
            {i: SimpleNamespace(logprob=lp) for lp, i, _ in row} for row in rows
        ]
        self.assertEqual(select_rollout_topk(vllm_rows, 2, 2), ((7, 1), (7, 2)))
        with self.assertRaisesRegex(ValueError, "response length"):
            select_rollout_topk(rows, 3, 2)
        with self.assertRaisesRegex(ValueError, "unique model-output"):
            select_rollout_topk([[(-0.1, 7), (-0.2, 7)]], 1, 2)

    def test_text_metadata_for_flat_and_composite_models(self):
        from omegaconf import OmegaConf
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import (
            GPT2Config,
            PreTrainedTokenizerFast,
            VisionEncoderDecoderConfig,
            ViTConfig,
        )

        from rlinf.workers.teacher.scoring import rollout_topk_count

        text_config = GPT2Config(vocab_size=8, n_positions=64)
        configs = (
            text_config,
            VisionEncoderDecoderConfig.from_encoder_decoder_configs(
                ViTConfig(), text_config
            ),
        )
        with tempfile.TemporaryDirectory() as path:
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(5)})),
                eos_token="t2",
            )
            tokenizer.save_pretrained(path)
            cfg = OmegaConf.create(
                {
                    "rollout": {"model": {"model_path": path}},
                    "distillation": {
                        "enabled": True,
                        "teacher_models": {"default": {"model_path": path}},
                        "distillation_loss": {
                            "loss_mode": "forward_kl_topk",
                            "topk": 2,
                            "topk_source": "student",
                            "use_policy_gradient": False,
                        },
                    },
                }
            )
            for config in configs:
                with self.subTest(model_type=config.model_type):
                    config.save_pretrained(path)
                    for backend_cls in (SGLangTeacherBackend, VLLMTeacherBackend):
                        backend = backend_cls(TeacherModelConfig(path), "default")
                        info = backend.load_model_info()
                        self.assertEqual(info.vocab_size, 8)
                        self.assertEqual(backend.model_vocab_size, 8)
                        self.assertEqual(info.max_context_length, 64)
                    self.assertEqual(rollout_topk_count(cfg), 2)

    def test_sglang_chunked_position_scores_preserve_model_probabilities(self):
        try:
            from sglang.srt.layers.utils.logprob import get_token_ids_logprobs_chunk
            from sglang.srt.server_args import ServerArgs
        except ImportError:
            self.skipTest("SGLang is required")
        if "selected_token_logprob_vocab_size" not in ServerArgs.__dataclass_fields__:
            self.skipTest("SGLang per-position scoring patch is required")
        logits = torch.tensor(
            [
                [0.0, -110.0, -2.0, -3.0, 20.0],
                [0.0, -120.0, -4.0, -5.0, 20.0],
                [0.0, -130.0, -2.0, -4.0, 20.0],
                [0.0, -140.0, -3.0, -4.0, 20.0],
                [0.0, -150.0, -2.0, -5.0, 20.0],
            ]
        )
        ids = [[[1, 4], [3, 1], [2, 0]], [[4, 2], [1, 0]]]
        model_logp = logits.log_softmax(-1)
        values, returned_ids = [], []
        split = get_token_ids_logprobs_chunk(
            model_logp[:2], ids[:1], [3], values, returned_ids, vocab_size=5
        )
        self.assertEqual(split, 2)
        split = get_token_ids_logprobs_chunk(
            model_logp[2:], ids, [3, 2], values, returned_ids, split, vocab_size=5
        )
        self.assertEqual(split, 0)
        self.assertEqual(returned_ids, ids)
        expected = model_logp.gather(-1, torch.tensor(ids[0] + ids[1]))
        torch.testing.assert_close(
            torch.cat([torch.tensor(v) for v in values]), expected
        )

    def test_scoring_budget_includes_intermediate_objects(self):
        backend = VLLMTeacherBackend(
            TeacherModelConfig(
                "unused", inference={"max_payload_bytes": 2 * 1024 * 1024}
            ),
            "default",
        )
        backend.info = TeacherModelInfo("model", "tokenizer", 150000, 32)
        selected = request(2, ((0, 0), (4, 0), (2, 3)))
        with self.assertRaisesRegex(ValueError, "scoring payload"):
            backend.validate_request(selected)
        with self.assertRaisesRegex(ValueError, "scoring payload"):
            backend.validate_request(request(-1))

    def test_topk_scoring_budget_covers_long_native_prefill(self):
        backend = SGLangTeacherBackend(TeacherModelConfig("unused"), "default")
        backend.info = TeacherModelInfo("model", "tokenizer", 248320, 10241)
        backend.validate_request(
            replace(request(64), input_ids=(1,) * 10240, response_mask=(True,) * 10240)
        )

    def test_selected_scores_use_original_prefixes_and_bounded_output(self):
        calls = []
        probabilities = self.probs

        class VLLMEngine:
            async def generate(self, prompt, params, request_id):
                prefix = prompt["prompt_token_ids"]
                calls.append(prefix)
                # Prompt-wide vocabulary scores must never be requested here.
                if hasattr(params, "prompt_logprobs"):
                    raise AssertionError("Expected next-token scoring only")
                assert params.max_tokens == 1
                row = probabilities[len(prefix) - 1]
                yield SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            logprobs=[
                                {
                                    i: SimpleNamespace(logprob=float(value.log()))
                                    for i, value in enumerate(row)
                                }
                            ]
                        )
                    ]
                )

        class SGLangEngine:
            async def async_generate(self, **kwargs):
                prefix = kwargs["input_ids"]
                calls.append(prefix)
                assert kwargs["logprob_start_len"] == 0
                assert kwargs["top_logprobs_num"] == 0
                assert kwargs["sampling_params"]["max_new_tokens"] == 0
                width = kwargs["sampling_params"]["custom_params"][
                    "rlinf_token_ids_logprob_k"
                ]
                ids = kwargs["token_ids_logprob"]
                return {
                    "meta_info": {
                        "input_token_ids_logprobs": [None]
                        + [
                            [
                                (float(row[i].log()), i, None)
                                for i in ids[position * width : (position + 1) * width]
                            ]
                            for position, row in enumerate(probabilities)
                        ]
                    }
                }

        ids = ((0, 0), (4, 0), (2, 3))
        req = request(2, ids)
        modules = {
            "vllm": SimpleNamespace(SamplingParams=SimpleNamespace),
            "vllm.inputs": SimpleNamespace(TokensPrompt=dict),
        }
        for backend_cls, engine in (
            (VLLMTeacherBackend, VLLMEngine()),
            (SGLangTeacherBackend, SGLangEngine()),
        ):
            with (
                self.subTest(backend=backend_cls.__name__),
                patch.dict(sys.modules, modules),
            ):
                calls.clear()
                backend = backend_cls(TeacherModelConfig("unused"), "default")
                backend.info = self.backend.info
                backend.engine = engine
                backend.supports_position_scores = True
                target = asyncio.run(backend.score(req))
                self.assertEqual(
                    calls,
                    [[1], [1, 3]] if backend_cls is VLLMTeacherBackend else [[1, 3, 2]],
                )
                self.assertEqual(target.token_ids[1:].tolist(), [[4, 0], [2, 3]])
                torch.testing.assert_close(
                    target.logprobs[1:],
                    torch.stack((self.probs[0, [4, 0]], self.probs[1, [2, 3]])).log(),
                )
                calls.clear()
                masked = replace(req, response_mask=(False, False, True))
                target = asyncio.run(backend.score(masked))
                self.assertEqual(
                    calls,
                    [[1, 3]] if backend_cls is VLLMTeacherBackend else [[1, 3, 2]],
                )
                self.assertEqual(target.valid_mask.tolist(), [False, False, True])

    def test_sglang_extreme_selected_scores_match_log_softmax(self):
        try:
            from sglang.srt.layers.logits_processor import (
                LogitsMetadata,
                LogitsProcessor,
            )
            from sglang.srt.layers.sampler import Sampler
            from sglang.srt.model_executor.forward_batch_info import ForwardMode
            from sglang.srt.server_args import ServerArgs
        except ImportError:
            self.skipTest("SGLang is required for the real Sampler comparison")

        from rlinf.algorithms.distillation.config import DistillationLossConfig
        from rlinf.algorithms.distillation.fsdp import compute_distribution_loss

        if "selected_token_logprob_vocab_size" not in ServerArgs.__dataclass_fields__:
            self.skipTest("SGLang per-position scoring patch is required")
        processor = object.__new__(LogitsProcessor)
        torch.nn.Module.__init__(processor)
        processor.selected_token_logprob_vocab_size = 5
        logits = torch.tensor([[0.0, -110.0, -220.0, -330.0, -440.0]])
        sampler = object.__new__(Sampler)
        torch.nn.Module.__init__(sampler)
        sampler.use_nan_detection = False
        sampler.rl_on_policy_target = None
        sampler.enable_deterministic = False
        sampler.use_log_softmax_logprob = False
        sampler.use_ascend_backend = False
        sampling = SimpleNamespace(
            has_custom_logit_processor=False,
            is_all_greedy=False,
            need_top_p_sampling=False,
            need_top_k_sampling=False,
            need_min_p_sampling=False,
            temperatures=torch.ones(1, 1),
            sampling_seed=None,
            grammars=None,
            device=torch.device("cpu"),
        )
        rollout = SimpleNamespace(next_token_logits=logits.clone())
        sampler(rollout, sampling, True, [2], [None], torch.tensor([0]))
        self.assertEqual(rollout.next_token_top_logprobs_idx[0].tolist(), [0, 1])
        torch.testing.assert_close(
            rollout.next_token_top_logprobs_val[0], logits.log_softmax(-1)[0, :2]
        )

        class Engine:
            async def async_generate(self, **kwargs):
                width = kwargs["sampling_params"]["custom_params"][
                    "rlinf_token_ids_logprob_k"
                ]
                ids = (
                    torch.tensor(kwargs["token_ids_logprob"])
                    .reshape(-1, width)
                    .tolist()
                )
                metadata = LogitsMetadata(
                    forward_mode=ForwardMode.EXTEND,
                    extend_token_ids_logprob=True,
                    token_ids_logprobs=[ids],
                    extend_logprob_pruned_lens_cpu=[3],
                    extend_input_logprob_token_ids_gpu=torch.tensor([3, 2, 0]),
                )
                output = processor.process_input_logprobs(
                    logits.expand(3, -1), metadata
                )
                return {
                    "meta_info": {
                        "input_token_ids_logprobs": [None]
                        + [
                            [
                                (float(value), int(token), None)
                                for value, token in zip(row, ids)
                            ]
                            for row, ids in zip(
                                output.input_token_ids_logprobs_val[0][:-1],
                                output.input_token_ids_logprobs_idx[0][:-1],
                            )
                        ]
                    }
                }

        backend = SGLangTeacherBackend(TeacherModelConfig("unused"), "default")
        backend.info = self.backend.info
        backend.engine = Engine()
        backend.supports_position_scores = True
        target = asyncio.run(backend.score(request(2, ((0, 0), (1, 4), (2, 3)))))
        expected = logits.log_softmax(-1).expand(2, -1).gather(-1, target.token_ids[1:])
        torch.testing.assert_close(target.logprobs[1:], expected)
        student = torch.zeros(2, 5, requires_grad=True)
        config = DistillationLossConfig(
            loss_mode="reverse_kl_topk",
            topk=2,
            topk_source="student",
            use_policy_gradient=False,
        )
        actual = compute_distribution_loss(
            student, target.token_ids[1:], target.logprobs[1:], config
        ).distribution_losses.sum()
        logq = student.log_softmax(-1).gather(-1, target.token_ids[1:])
        reference = (logq.exp() * (logq - expected)).sum()
        torch.testing.assert_close(actual, reference)
        torch.testing.assert_close(
            torch.autograd.grad(actual, student)[0],
            torch.autograd.grad(reference, student)[0],
        )

    def test_distribution_scores_preserve_model_vocabulary_probabilities(self):
        from rlinf.algorithms.distillation.config import DistillationLossConfig
        from rlinf.algorithms.distillation.fsdp import compute_distribution_loss

        logits = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 12.0],
                [4.0, 3.0, 2.0, 1.0, 0.0, 12.0, 11.0, 10.0],
            ]
        )
        model_logprobs = logits.log_softmax(-1)
        calls = []
        self_test = self

        def row_scores(row, count):
            ids = row.argsort(descending=True)[:count]
            return {int(i): float(row[i]) for i in ids}

        class VLLMEngine:
            async def generate(self, prompt, params, request_id):
                prefix = prompt["prompt_token_ids"]
                calls.append(prefix)
                if hasattr(params, "prompt_logprobs"):
                    self_test.assertEqual(prefix, [1, 3, 2])
                    count = params.prompt_logprobs
                    yield SimpleNamespace(
                        prompt_logprobs=[None]
                        + [
                            {
                                i: SimpleNamespace(logprob=value)
                                for i, value in row_scores(row, count).items()
                            }
                            for row in model_logprobs
                        ]
                    )
                else:
                    self_test.assertEqual(params.logprobs, 8)
                    row = model_logprobs[len(prefix) - 1]
                    yield SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                logprobs=[
                                    {
                                        i: SimpleNamespace(logprob=float(value))
                                        for i, value in enumerate(row)
                                    }
                                ]
                            )
                        ]
                    )

        class SGLangEngine:
            async def async_generate(self, **kwargs):
                calls.append(kwargs["input_ids"])
                self_test.assertEqual(kwargs["input_ids"], [1, 3, 2])
                if kwargs["sampling_params"].get("custom_params"):
                    width = kwargs["sampling_params"]["custom_params"][
                        "rlinf_token_ids_logprob_k"
                    ]
                    ids = kwargs["token_ids_logprob"]
                    return {
                        "meta_info": {
                            "input_token_ids_logprobs": [None]
                            + [
                                [
                                    (float(row[i]), i, None)
                                    for i in ids[
                                        position * width : (position + 1) * width
                                    ]
                                ]
                                for position, row in enumerate(model_logprobs)
                            ]
                        }
                    }
                self_test.assertEqual(kwargs["logprob_start_len"], 0)
                self_test.assertEqual(kwargs["sampling_params"]["max_new_tokens"], 1)
                return {
                    "meta_info": {
                        "input_top_logprobs": [None]
                        + [
                            [
                                (value, i, None)
                                for i, value in row_scores(
                                    row, kwargs["top_logprobs_num"]
                                ).items()
                            ]
                            for row in model_logprobs
                        ]
                    }
                }

        modules = {
            "vllm": SimpleNamespace(SamplingParams=SimpleNamespace),
            "vllm.inputs": SimpleNamespace(TokensPrompt=dict),
        }
        for backend_cls, engine in (
            (VLLMTeacherBackend, VLLMEngine()),
            (SGLangTeacherBackend, SGLangEngine()),
        ):
            backend = backend_cls(TeacherModelConfig("unused"), "default")
            backend.info = replace(self.backend.info, vocab_size=8)
            backend.model_vocab_size = 8
            backend.engine = engine
            backend.supports_position_scores = True
            for req in (request(-1), request(2), request(2, ((0, 0), (7, 5), (6, 0)))):
                with (
                    self.subTest(backend=backend_cls.__name__, request=req),
                    patch.dict(sys.modules, modules),
                ):
                    calls.clear()
                    target = asyncio.run(backend.score(req))
                    ids = target.token_ids[1:] if target.token_ids is not None else None
                    expected = (
                        model_logprobs
                        if ids is None
                        else model_logprobs.gather(-1, ids)
                    )
                    torch.testing.assert_close(target.logprobs[1:], expected)
                    self.assertEqual(
                        calls,
                        [[1], [1, 3]]
                        if backend_cls is VLLMTeacherBackend
                        and req.selected_token_ids is not None
                        else [[1, 3, 2]],
                    )
                    if req.topk != -1:
                        self.assertTrue((target.logprobs[1:].exp().sum(-1) < 1).all())
                    for mode in ("forward_kl_topk_tail", "reverse_kl_topk_tail"):
                        loss = compute_distribution_loss(
                            logits,
                            ids,
                            target.logprobs[1:],
                            DistillationLossConfig(
                                loss_mode=mode,
                                topk=req.topk,
                                topk_source="student",
                                use_policy_gradient=False,
                            ),
                        )
                        torch.testing.assert_close(
                            loss.distribution_losses, torch.zeros(2), atol=2e-6, rtol=0
                        )
            rows = [None] + [
                {i: float(v) for i, v in enumerate(row)} for row in model_logprobs
            ]
            sampled = backend.normalize(request(), rows)
            torch.testing.assert_close(
                sampled.logprobs[1:, 0], model_logprobs[[0, 1], [3, 2]]
            )

    def test_sampled_positions_include_last_token(self):
        target = self.backend.normalize(request(), self.rows)
        torch.testing.assert_close(
            target.logprobs[1:, 0],
            torch.stack((self.probs[0, 3].log(), self.probs[1, 2].log())),
        )
        self.assertEqual(target.valid_mask.tolist(), [False, True, True])
        self.assertEqual(target.token_ids[1:].tolist(), [[3], [2]])

    def test_teacher_topk_discards_extra_sampled_token(self):
        target = self.backend.normalize(request(2), self.rows)
        self.assertEqual(target.token_ids[1:].tolist(), [[2, 3], [0, 4]])
        self.assertLess(target.logprobs[1].exp().sum().item(), 1)

    def test_student_ids_need_not_be_teacher_topk(self):
        ids = ((0, 0), (4, 0), (2, 3))
        target = self.backend.normalize(request(2, ids), self.rows)
        self.assertEqual(target.token_ids[1:].tolist(), [[4, 0], [2, 3]])
        expected = torch.stack((self.probs[0, [4, 0]], self.probs[1, [2, 3]])).log()
        torch.testing.assert_close(target.logprobs[1:], expected)

    def test_full_targets_order_and_response_padding(self):
        target = self.backend.normalize(request(-1), self.rows)
        self.assertIsNone(target.token_ids)
        torch.testing.assert_close(target.logprobs[1:], self.probs.log())
        batch = teacher_targets_to_batch([target], [1], [2], 4, response_only=True)
        self.assertNotIn("teacher_token_ids", batch)
        self.assertEqual(
            batch["teacher_valid_mask"].tolist(), [[True, True, False, False]]
        )
        torch.testing.assert_close(batch["teacher_logprobs"][0, :2], self.probs.log())

    def test_malformed_or_mismatched_response_rejected(self):
        for bad in (
            replace(request(2), topk=0),
            replace(request(2), topk=True),
            replace(request(2), tokenizer_fingerprint="different"),
            request(2, ((0, 0), (1, 1), (2, 3))),
        ):
            with self.subTest(request=bad):
                with self.assertRaises(ValueError):
                    self.backend.normalize(bad, self.rows)
        with self.assertRaises(ValueError):
            self.backend.normalize(request(-1), self.rows[:-1])
        incomplete = [None, {0: -1.0}, self.rows[2]]
        with self.assertRaises(ValueError):
            self.backend.normalize(request(-1), incomplete)


class TestTeacherClient(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web

        self.delay = 0
        self.wrong_model = False
        self.active = 0
        self.peak = 0

        async def handler(http_request):
            payload = await http_request.json()
            selected = payload.get("selected_token_ids")
            self.active += 1
            self.peak = max(self.peak, self.active)
            try:
                await asyncio.sleep(self.delay)
                return web.json_response(
                    {
                        "request_id": payload["request_id"],
                        "teacher_key": "default",
                        "model_fingerprint": "wrong" if self.wrong_model else "model",
                        "target_format": "sparse" if selected else "sampled",
                        "student_policy_version": payload.get("student_policy_version"),
                        "token_ids": selected or [[0], [3], [2]],
                        "logprobs": [
                            [0.0] * len(selected[0]),
                            [-1.0] * len(selected[1]),
                            [-2.0] * len(selected[2]),
                        ]
                        if selected
                        else [[0.0], [-1.0], [-2.0]],
                        "valid_mask": payload["response_mask"],
                    }
                )
            finally:
                self.active -= 1

        app = web.Application()
        app.router.add_post("/score", handler)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await site.start()
        port = self.runner.addresses[0][1]
        self.endpoint = TeacherEndpoint(
            f"http://127.0.0.1:{port}",
            "default",
            TeacherModelInfo("model", "tokenizer", 5, 32),
        )

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_rollout_student_topk_is_scored_before_actor_training(self):
        from rlinf.algorithms.distillation.config import DistillationPlan
        from rlinf.workers.teacher.scoring import RolloutTeacherScorer

        scorer = object.__new__(RolloutTeacherScorer)
        scorer.plan = DistillationPlan(enabled=True, topk=2, need_rollout_topk=True)
        scorer.client = TeacherClient([self.endpoint])
        scorer.fingerprint = "tokenizer"
        scorer.teacher_key = "default"
        scorer.student_policy_version = "step-0"
        try:
            try:
                from rlinf.workers.rollout.sglang.backends import SGLangEngineBackend
                from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker
            except ImportError:
                self.skipTest("SGLang rollout is required")
            from rlinf.data.schema.reasoning_requests import SeqGroupInfo

            class Engine:
                async def async_generate(self, **kwargs):
                    assert kwargs["top_logprobs_num"] == 2
                    assert kwargs["return_logprob"] is True
                    return {
                        "output_ids": [3, 2],
                        "meta_info": {
                            "finish_reason": {"type": "stop"},
                            "output_top_logprobs": [
                                [(-0.1, 4, None), (-0.2, 0, None)],
                                [(-0.1, 2, None), (-0.2, 3, None)],
                            ],
                        },
                    }

            worker = object.__new__(SGLangWorker)
            backend = object.__new__(SGLangEngineBackend)
            backend._engine = Engine()
            worker._backend = backend
            worker._sampling_params = {"max_new_tokens": 2}
            worker._return_logprobs = False
            worker._student_topk_count = 2
            worker._tokenizer = range(5)
            worker._teacher_scorer = scorer
            worker._logger = None
            group = SeqGroupInfo(id=1, input_ids=[1], answer=["x"], group_size=1)
            group = await worker._async_generate_group(group)
            self.assertIsNone(group.teacher_requests)
            target = group.teacher_targets[0]
            self.assertEqual(target.token_ids[1:].tolist(), [[4, 0], [2, 3]])
            self.assertEqual(target.student_policy_version, "step-0")
            turn = SimpleNamespace(
                prompt_ids=[1],
                response_ids=[3, 2],
                response_mask=[True, False],
                response_topk_token_ids=((4, 0), (0, 0)),
                extra_fields={},
            )
            await scorer.score_agent_output(turn)
            self.assertEqual(
                turn.teacher_targets.valid_mask.tolist(), [False, True, False]
            )
            with self.assertRaisesRegex(ValueError, "matching rollout"):
                await scorer.score_sequence([1], [3, 2], "sample")
        finally:
            await scorer.close()

    async def test_matching_bounded_concurrency_and_close(self):
        self.delay = 0.01
        client = TeacherClient([self.endpoint], max_concurrency=2)
        try:
            results = await asyncio.gather(
                *(
                    client.score(replace(request(), request_id=f"r{i}"))
                    for i in range(5)
                )
            )
            self.assertEqual(
                [r.request_id for r in results], [f"r{i}" for i in range(5)]
            )
            self.assertLessEqual(self.peak, 2)
        finally:
            await client.close()
            await client.close()
        with self.assertRaises(RuntimeError):
            await client.score(request())

    async def test_timeout_and_model_mismatch_fail_request(self):
        self.delay = 0.1
        client = TeacherClient([self.endpoint], timeout=0.01)
        try:
            with self.assertRaises(TimeoutError):
                await client.score(request())
        finally:
            await client.close()
        self.delay = 0
        self.wrong_model = True
        client = TeacherClient([self.endpoint])
        try:
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                await client.score(request())
        finally:
            await client.close()


class TestTeacherHTTP(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        from aiohttp import web

        from rlinf.utils.logging import get_logger
        from rlinf.workers.teacher.teacher_worker import TeacherWorker

        class Engine:
            reverse_ids = False

            async def async_generate(self, **kwargs):
                width = kwargs["sampling_params"]["custom_params"][
                    "rlinf_token_ids_logprob_k"
                ]
                ids = kwargs["token_ids_logprob"]
                rows = [None]
                for position in range(len(kwargs["input_ids"]) - 1):
                    selected = ids[position * width : (position + 1) * width]
                    if self.reverse_ids:
                        selected = selected[::-1]
                    rows.append([(-2.0, token, None) for token in selected])
                return {"meta_info": {"input_token_ids_logprobs": rows}}

        backend = SGLangTeacherBackend(TeacherModelConfig("unused"), "default")
        backend.info = TeacherModelInfo("model", "tokenizer", 5, 32)
        backend.engine = self.engine = Engine()
        backend.supports_position_scores = True
        worker = object.__new__(TeacherWorker)
        worker.backend = backend
        worker.teacher_key = "default"
        worker._limit = asyncio.Semaphore(1)
        worker._logger = get_logger()
        worker._stacklevel = 1

        app = web.Application()
        app.router.add_post("/score", worker._handle_score)
        runner = web.AppRunner(app)
        await runner.setup()
        self.addAsyncCleanup(runner.cleanup)
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        endpoint = TeacherEndpoint(
            f"http://127.0.0.1:{runner.addresses[0][1]}", "default", backend.info
        )
        self.client = TeacherClient([endpoint])
        self.addAsyncCleanup(self.client.close)

    async def test_student_selected_ids_survive_http_roundtrip(self):
        result = await self.client.score(request(2, ((0, 0), (4, 0), (2, 3))))
        self.assertEqual(result.token_ids[1:].tolist(), [[4, 0], [2, 3]])
        self.assertEqual(result.valid_mask.tolist(), [False, True, True])
        torch.testing.assert_close(result.logprobs[1:], torch.full((2, 2), -2.0))

    async def test_mismatched_scoring_ids_are_rejected_over_http(self):
        from aiohttp import ClientResponseError

        self.engine.reverse_ids = True
        with self.assertRaises(ClientResponseError) as caught:
            await self.client.score(request(2, ((0, 0), (4, 0), (2, 3))))
        self.assertEqual(caught.exception.status, 400)


@unittest.skipUnless(
    os.environ.get("RLINF_TEST_TEACHER_MODEL"),
    "Set RLINF_TEST_TEACHER_MODEL to enable a real Teacher engine test",
)
class TestTeacherEngine(unittest.TestCase):
    def test_prompt_scores_match_transformers(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        path = os.environ["RLINF_TEST_TEACHER_MODEL"]
        tokenizer = AutoTokenizer.from_pretrained(path)
        engine = os.environ.get("RLINF_TEST_TEACHER_BACKEND", "vllm")
        dtype = torch.float16 if engine == "sglang" else torch.float32
        device = "cuda" if engine == "sglang" else "cpu"
        tolerance = 2e-3 if engine == "sglang" else 2e-4
        model = (
            AutoModelForCausalLM.from_pretrained(path)
            .to(device=device, dtype=dtype)
            .eval()
        )
        tokens = (1, 3, 4, 5, 2)
        with torch.no_grad():
            expected = (
                model(torch.tensor([tokens], device=device))
                .logits[0, :-1]
                .float()
                .log_softmax(-1)
                .cpu()
            )
        del model

        if engine == "sglang":
            from rlinf.workers.teacher.sglang_backend import SGLangTeacherBackend

            backend_cls = SGLangTeacherBackend
            options = {
                "dtype": "float16",
                "disable_cuda_graph": True,
                "disable_piecewise_cuda_graph": True,
                "mem_fraction_static": 0.15,
                "max_running_requests": 1,
                "attention_backend": "triton",
                "sampling_backend": "pytorch",
                "skip_tokenizer_init": True,
                "chunked_prefill_size": 4,
            }
        else:
            backend_cls = VLLMTeacherBackend
            options = {
                "dtype": "float32",
                "enforce_eager": True,
                "gpu_memory_utilization": 0.15,
                "max_num_seqs": 1,
                "enable_prefix_caching": False,
            }
        backend = backend_cls(
            TeacherModelConfig(
                path,
                inference={
                    "backend": engine,
                    "max_model_len": 64,
                    "engine_kwargs": options,
                },
            ),
            "default",
        )

        async def run():
            try:
                await backend.initialize()
                for k in (None, 4, -1):
                    req = TeacherScoreRequest(
                        f"engine-{k}",
                        "sample",
                        "group",
                        "default",
                        tokens,
                        (False, True, True, True, True),
                        k,
                        tokenizer_fingerprint(tokenizer),
                    )
                    target = await backend.score(req)
                    reference = (
                        expected
                        if k == -1
                        else expected.gather(-1, target.token_ids[1:])
                    )
                    torch.testing.assert_close(
                        target.logprobs[1:], reference, atol=tolerance, rtol=2e-4
                    )
                selected = ((0, 0), (7, 8), (9, 10), (11, 12), (13, 14))
                req = TeacherScoreRequest(
                    "engine-selected",
                    "sample",
                    "group",
                    "default",
                    tokens,
                    (False, True, True, True, True),
                    2,
                    tokenizer_fingerprint(tokenizer),
                    selected,
                    "step-0",
                )
                target = await backend.score(req)
                torch.testing.assert_close(
                    target.logprobs[1:],
                    expected.gather(-1, torch.tensor(selected[1:])),
                    atol=tolerance,
                    rtol=2e-4,
                )
            finally:
                await backend.close()

        asyncio.run(run())


def create_tiny_teacher(path, architecture="gpt2"):
    """Save a seeded random GPT-2 and local tokenizer, requiring no model download."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(21)
    tokenizer = Tokenizer(WordLevel({f"t{i}": i for i in range(32)}, unk_token="t0"))
    wrapped = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="t0",
        bos_token="t1",
        eos_token="t2",
        pad_token="t0",
    )
    wrapped.save_pretrained(path)
    if architecture == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        LlamaForCausalLM(
            LlamaConfig(
                vocab_size=32,
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=64,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).save_pretrained(path)
    else:
        GPT2LMHeadModel(
            GPT2Config(
                vocab_size=32,
                n_positions=64,
                n_embd=32,
                n_layer=1,
                n_head=2,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        ).save_pretrained(path)


if __name__ == "__main__":
    unittest.main()
