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

import asyncio
import csv
import json
from pathlib import Path

import pytest

from examples.reasoning.eval_math import (
    _evaluate,
    _grade,
    _has_boxed_answer,
    _load_questions,
    _rollout_layout,
    _score,
    _write_summary,
)
from rlinf.data.datasets.reasoning.math_prompt import format_dapo_math_prompt


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (r"First guess: \boxed{9}. Final answer: \boxed{007}.", True),
        (r"First guess: \boxed{7}. Final answer: \boxed{9}.", False),
        (r"\boxed{7.5}", False),
        ("7", False),
        ("", False),
    ],
)
def test_aime_scores_final_boxed_integer(response, expected):
    assert _score(response, "7", "AIME2025") is expected


def test_math500_scores_equivalent_latex():
    pytest.importorskip("math_verify")
    assert _grade(r"Answer: \boxed{\frac{3}{6}}", r"\frac{1}{2}", "MATH-500") == (
        True,
        True,
    )
    assert not _score(r"\boxed{9}", "7", "MATH-500")


def test_boxed_math_prompt_preserves_the_problem():
    problem = "Consider these cases.\n\nFind all solutions."
    legacy = (
        "Solve the following math problem step by step. "
        "The last line of your response should be of the form Answer: $Answer.\n\n"
        + problem
        + '\n\nRemember to put your answer on its own line after "Answer:".'
    )
    prompt = format_dapo_math_prompt(problem)
    assert format_dapo_math_prompt(legacy) == format_dapo_math_prompt(prompt) == prompt
    assert prompt.count(r"Answer: \boxed{...}") == 1
    assert prompt.split("\n\n")[1:-1] == problem.split("\n\n")


def test_dapo_uses_final_answer_line_and_signed_integers():
    assert _grade("Answer: 4\nAnswer: $-12345$", "-12345", "DAPO-17K") == (True, True)
    assert _grade(r"Answer: \boxed{-12345}", "-12345", "DAPO-17K") == (True, True)
    assert _grade(r"Answer: \boxed{42}", "41", "DAPO-17K") == (False, True)
    assert _grade("Answer: 3.5", "3", "DAPO-17K") == (False, False)


def test_math_final_answer_format():
    assert _has_boxed_answer(r"\boxed{\frac{3}{6}}")
    assert _has_boxed_answer(r"\boxed{\{1,2\}}")
    assert not _has_boxed_answer(r"\boxed{ }")
    assert not _has_boxed_answer(r"\boxed{2} then \boxed{3")


def test_grouped_eval_keeps_tail_and_scores_per_question(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    path = tmp_path / "AIME2026" / "data"
    path.mkdir(parents=True)
    pq.write_table(
        pa.table({"problem": ["A", "B", "C"], "answer": [1, 2, 3]}),
        path / "test.parquet",
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(
            WordLevel({"?": 0, r"\boxed{2}": 1, r"\boxed{3}": 2}, unk_token="?")
        ),
        unk_token="?",
        additional_special_tokens=["</think>"],
        chat_template="{{ messages[0].content }}",
    )

    # Each call represents the four draws for one question, including the tail batch.
    expected_calls = iter([(0, [1, 1, 1, 1]), (1, [1, 2, 2, 2]), (0, [2, 2, 2, 2])])

    class Engine:
        async def async_generate(self, *, input_ids, sampling_params, routed_dp_rank):
            rank, answers = next(expected_calls)
            assert routed_dp_rank == rank
            assert len(input_ids) == len(answers)
            return [
                {
                    "output_ids": [0] * i + [answer],
                    "meta_info": {
                        "completion_tokens": i + 1,
                        "finish_reason": {"type": "length" if i == 3 else "stop"},
                    },
                }
                for i, answer in enumerate(answers)
            ]

    cfg = OmegaConf.create(
        {
            "data": {
                "val_data_paths": [str(path / "test.parquet")],
                "dataset_name": "AIME2026",
                "val_rollout_batch_size": 2,
                "prompt_key": "problem",
                "prompt_suffix": "",
                "answer_key": "answer",
                "data_size": -1,
                "validation_shuffle": False,
                "apply_chat_template": True,
                "apply_chat_template_kwargs": {},
                "filter_prompt_by_length": False,
                "max_prompt_length": 2048,
            },
            "algorithm": {"group_size": 4},
            "runner": {"experiment_name": "test", "max_steps": -1},
            "rollout": {"print_outputs": False},
        }
    )
    output = tmp_path / "scores.jsonl"
    metrics = asyncio.run(
        _evaluate(
            Engine(), tokenizer, cfg, _load_questions(cfg.data), {}, output, dp_size=2
        )
    )
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [sum(sample["correct"] for sample in row["samples"]) for row in records] == [
        0,
        1,
        4,
    ]
    assert metrics["pass@1"] == pytest.approx(5 / 12)
    assert metrics["pass@4"] == pytest.approx(2 / 3)
    assert metrics["first_sample_accuracy"] == pytest.approx(2 / 3)
    assert metrics["total_num_trajectories"] == 12
    assert metrics["avg_prompt_length"] == 1
    assert metrics["avg_response_length"] == 2.5
    assert metrics["avg_total_length"] == 3.5
    assert metrics["total_response_tokens"] == 30
    assert metrics["final_answer_format"] == 1
    assert metrics["length_limit_rate"] == 0.25
    assert records[-1]["prompt_tokens"] == 1
    assert [s["response_tokens"] for s in records[-1]["samples"]] == [1, 2, 3, 4]

    summaries = [{"model": model, **metrics} for model in ("base", "step160")]
    for row in summaries:
        _write_summary(tmp_path, row)
    assert json.loads((tmp_path / "summary.json").read_text()) == summaries
    with (tmp_path / "summary.csv").open() as stream:
        assert [row["model"] for row in csv.DictReader(stream)] == ["base", "step160"]


def test_math_eval_presets_match_opd_rollout():
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "examples/reasoning/config/math"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        training = compose(config_name="qwen3.5-2b-opd-k1-pg-fsdp2-sglang")
        for config, dataset, batch_size in [
            ("eval_qwen3.5-2b", "AIME2025", 30),
            ("eval_qwen3.5-2b-aime2026", "AIME2026", 30),
            ("eval_qwen3.5-2b-math500", "MATH-500", 100),
        ]:
            evaluation = compose(config_name=config)
            assert evaluation.rollout.tensor_parallel_size == 1
            assert _rollout_layout(evaluation, device_count=8) == (0, 4)
            assert evaluation.data.dataset_name == dataset
            assert evaluation.data.val_rollout_batch_size == batch_size
            assert evaluation.algorithm.group_size == 4
            assert evaluation.runner.seq_length == training.runner.seq_length
            assert evaluation.data.max_prompt_length == training.data.max_prompt_length
            assert evaluation.rollout.max_model_len == training.rollout.max_model_len
            for key in (
                "do_sample",
                "temperature",
                "top_k",
                "top_p",
                "repetition_penalty",
                "max_new_tokens",
            ):
                assert (
                    evaluation.algorithm.sampling_params[key]
                    == training.algorithm.sampling_params[key]
                )


def test_math_eval_groups_gpus_by_tp():
    from omegaconf import OmegaConf

    cfg = OmegaConf.create(
        {
            "cluster": {"num_nodes": 1, "component_placement": {"rollout": "2-5"}},
            "rollout": {"tensor_parallel_size": 2, "pipeline_parallel_size": 1},
        }
    )
    assert _rollout_layout(cfg, 8) == (2, 2)
    cfg.rollout.pipeline_parallel_size = 2
    assert _rollout_layout(cfg, 8) == (2, 1)
    cfg.rollout.tensor_parallel_size = 3
    with pytest.raises(ValueError, match="TP=3, PP=2"):
        _rollout_layout(cfg, 8)
