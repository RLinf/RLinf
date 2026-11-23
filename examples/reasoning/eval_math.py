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

"""Evaluate one Hugging Face model on a math validation set using SGLang."""

import asyncio
import csv
import fcntl
import json
import math
import random
import re
from decimal import Decimal
from pathlib import Path

import hydra
import regex
from omegaconf import DictConfig, OmegaConf

from rlinf.scheduler import Worker
from rlinf.scheduler.cluster import parse_rank_config
from rlinf.utils.logging import get_logger
from rlinf.utils.precision import torch_dtype_from_precision


def _has_boxed_answer(final: str) -> bool:
    """Check the last boxed answer for nonempty, balanced LaTeX content."""
    _, marker, tail = final.rpartition(r"\boxed")
    if not marker:
        return False
    box = regex.match(r"\s*(?P<box>\{(?:\\.|[^{}\\]++|(?&box))*\})", tail)
    return bool(box and box["box"][1:-1].strip())


def _rollout_layout(cfg: DictConfig, device_count: int) -> tuple[int, int]:
    """Return the first visible GPU rank and the number of local SGLang replicas."""
    if cfg.cluster.num_nodes != 1:
        raise ValueError(
            f"Local math evaluation expects one node, got {cfg.cluster.num_nodes}"
        )
    if device_count < 1:
        raise ValueError("Expected rollout GPUs on the local node, found none")
    ranks = parse_rank_config(
        cfg.cluster.component_placement.rollout, list(range(device_count)), "GPU"
    )
    if not ranks or ranks != list(range(ranks[0], ranks[-1] + 1)):
        raise ValueError(
            f"Expected consecutive visible GPU ranks for local SGLang rollout, got {ranks}"
        )
    tp = cfg.rollout.tensor_parallel_size
    pp = cfg.rollout.pipeline_parallel_size
    if min(tp, pp) < 1 or len(ranks) % (tp * pp):
        raise ValueError(
            f"Expected positive TP and PP sizes whose product divides {len(ranks)} rollout GPUs, got TP={tp}, PP={pp}"
        )
    return ranks[0], len(ranks) // (tp * pp)


def _load_questions(data: DictConfig) -> list[dict]:
    rows = []
    for filename in data.val_data_paths:
        path = Path(filename)
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            rows.extend(
                pq.read_table(
                    path, columns=[data.prompt_key, data.answer_key]
                ).to_pylist()
            )
        elif path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as stream:
                rows.extend(json.loads(line) for line in stream if line.strip())
        else:
            raise ValueError(f"Expected a Parquet or JSONL validation file, got {path}")
    if not rows:
        raise ValueError(f"No questions in {list(data.val_data_paths)}")
    if data.data_size is not None and data.data_size != -1:
        if data.data_size < 1:
            raise ValueError(
                f"Expected positive data_size or -1 for all rows, got {data.data_size}"
            )
        rows = rows[: data.data_size]
    if data.validation_shuffle:
        random.Random(data.seed).shuffle(rows)
    if data.dataset_name == "DAPO-17K":
        for row in rows:
            messages = row[data.prompt_key]
            if len(messages) != 1 or messages[0]["role"] != "user":
                raise ValueError("Expected a single user message in a DAPO prompt")
            row[data.prompt_key] = messages[0]["content"]
            row[data.answer_key] = row[data.answer_key]["ground_truth"]
    return rows


def _score(final: str, reference: str, benchmark: str) -> bool:
    _, boxed, content = final.rpartition(r"\boxed")
    if not boxed:
        return False
    if benchmark == "MATH-500":
        from math_verify import LatexExtractionConfig, parse, verify

        latex = [LatexExtractionConfig()]
        gold = parse(r"\boxed{" + reference + "}", extraction_config=latex)
        prediction = parse(boxed + content, extraction_config=latex)
        return bool(gold and prediction and verify(gold, prediction))

    match = re.match(r"\s*\{\s*([0-9]{1,3})\s*\}", content)
    return bool(match and int(match[1]) == int(reference))


def _grade(final: str, reference: str, benchmark: str) -> tuple[bool, bool]:
    """Return correctness and compliance with the benchmark's final-answer format."""
    if benchmark == "DAPO-17K":
        last_line = final.rstrip().rpartition("\n")[2].strip()
        match = re.fullmatch(r"Answer:\s*(.+)", last_line, re.IGNORECASE)
        if not match:
            return False, False
        answer = match[1].strip()
        if answer.startswith("$") and answer.endswith("$"):
            answer = answer[1:-1].strip()
        boxed = re.fullmatch(r"\\boxed\{\s*([+-]?\d+)\s*\}", answer)
        if boxed:
            answer = boxed[1]
        if not re.fullmatch(r"[+-]?\d+", answer):
            return False, False
        return Decimal(answer) == Decimal(reference), True
    return _score(final, reference, benchmark), _has_boxed_answer(final)


def _prepare_prompts(rows, tokenizer, data):
    questions = []
    for index, row in enumerate(rows):
        content = row[data.prompt_key] + data.prompt_suffix
        if data.apply_chat_template:
            content = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
                **data.apply_chat_template_kwargs,
            )
        ids = tokenizer.encode(content)
        if len(ids) > data.max_prompt_length:
            if data.filter_prompt_by_length:
                continue
            raise ValueError(
                f"{data.dataset_name} question {index} needs {len(ids)} prompt tokens, limit is {data.max_prompt_length}"
            )
        questions.append((row, ids))
    if len(questions) < len(rows):
        get_logger().warning(
            f"{data.dataset_name}: excluded {len(rows) - len(questions)} of {len(rows)} questions exceeding {data.max_prompt_length} prompt tokens"
        )
    if not questions:
        raise ValueError(f"No questions fit max_prompt_length={data.max_prompt_length}")
    return questions


def _decode_math_response(tokenizer, ids, thinking):
    response = tokenizer.decode(ids, skip_special_tokens=True)
    end = tokenizer.convert_tokens_to_ids("</think>")
    if end in ids:
        final = tokenizer.decode(ids[ids.index(end) + 1 :], skip_special_tokens=True)
    else:
        final = "" if thinking else response
    return response, final


def _pass_rates(
    correct_counts: list[int], group_size: int
) -> tuple[float, float | None]:
    pass_one = sum(correct_counts) / (len(correct_counts) * group_size)
    if group_size < 4:
        return pass_one, None
    draws = math.comb(group_size, 4)
    pass_four = sum(
        1 - math.comb(group_size - hits, 4) / draws for hits in correct_counts
    )
    return pass_one, pass_four / len(correct_counts)


async def _evaluate(engine, tokenizer, cfg, rows, sampling, output, dp_size):
    benchmark = cfg.data.dataset_name
    questions = _prepare_prompts(rows, tokenizer, cfg.data)
    batch_size = cfg.data.val_rollout_batch_size or len(questions)
    if cfg.runner.max_steps != -1:
        questions = questions[: cfg.runner.max_steps * batch_size]
    thinking = cfg.data.apply_chat_template and cfg.data.apply_chat_template_kwargs.get(
        "enable_thinking", False
    )
    group_size = cfg.algorithm.group_size
    hits = []
    first_sample_hits = 0
    prompt_tokens = 0
    response_tokens = 0
    formatted = 0
    length_limited = 0
    with output.open("x", encoding="utf-8") as stream:
        for start in range(0, len(questions), batch_size):
            batch = questions[start : start + batch_size]
            requests = [ids for _, ids in batch for _ in range(group_size)]
            # Replicas share a seed. Keep all draws for a question on one replica.
            shards = min(len(batch), dp_size)
            calls = []
            for rank in range(shards):
                first = len(batch) * rank // shards
                last = len(batch) * (rank + 1) // shards
                calls.append(
                    engine.async_generate(
                        input_ids=requests[first * group_size : last * group_size],
                        sampling_params=sampling,
                        routed_dp_rank=rank if dp_size > 1 else None,
                    )
                )
            outputs = [
                response for shard in await asyncio.gather(*calls) for response in shard
            ]
            if len(outputs) != len(requests):
                raise RuntimeError(
                    f"Expected {len(requests)} responses, received {len(outputs)}"
                )
            for i, (row, prompt_ids) in enumerate(batch):
                samples = []
                for result in outputs[i * group_size : (i + 1) * group_size]:
                    reason = result["meta_info"]["finish_reason"]
                    if reason["type"] == "abort":
                        raise RuntimeError(f"SGLang aborted a request: {reason}")
                    response, final = _decode_math_response(
                        tokenizer,
                        result["output_ids"],
                        thinking,
                    )
                    if cfg.rollout.print_outputs:
                        get_logger().info(
                            f"{benchmark}[{start + i}] sample {len(samples) + 1}: {response}"
                        )
                    response_length = result["meta_info"]["completion_tokens"]
                    correct, valid_format = _grade(
                        final, str(row[cfg.data.answer_key]), benchmark
                    )
                    response_tokens += response_length
                    formatted += valid_format
                    length_limited += reason["type"] == "length"
                    samples.append(
                        {
                            "response": response,
                            "response_tokens": response_length,
                            "final_answer_format": valid_format,
                            "correct": correct,
                            "finish_reason": reason["type"],
                        }
                    )
                hits.append(sum(sample["correct"] for sample in samples))
                first_sample_hits += samples[0]["correct"]
                prompt_tokens += len(prompt_ids) * len(samples)
                stream.write(
                    json.dumps(
                        {
                            "index": start + i,
                            "problem": row[cfg.data.prompt_key],
                            "prompt_tokens": len(prompt_ids),
                            "gold": row[cfg.data.answer_key],
                            "samples": samples,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            stream.flush()
    pass1, pass4 = _pass_rates(hits, group_size)
    trajectories = len(hits) * group_size
    return {
        "benchmark": benchmark,
        "questions": len(questions),
        "group_size": group_size,
        "pass@1": pass1,
        "pass@4": pass4,
        "first_sample_accuracy": first_sample_hits / len(hits),
        "total_num_trajectories": trajectories,
        "avg_prompt_length": prompt_tokens / trajectories,
        "avg_response_length": response_tokens / trajectories,
        "avg_total_length": (prompt_tokens + response_tokens) / trajectories,
        "total_response_tokens": response_tokens,
        "final_answer_format": formatted / trajectories,
        "length_limit_rate": length_limited / trajectories,
    }


def _write_summary(run_dir: Path, metrics: dict) -> None:
    """Publish one model/benchmark result and retain the existing CSV schema."""
    columns = ("model", "benchmark", "questions", "group_size", "pass@1", "pass@4")
    with (run_dir / "summary.csv").open("a", newline="", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        summary = run_dir / "summary.json"
        records = (
            json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else []
        )
        records.append(metrics)
        pending = summary.with_suffix(".json.tmp")
        pending.write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        pending.replace(summary)
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        stream.seek(0, 2)
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(metrics)


@hydra.main(
    version_base="1.1",
    config_path="config/math",
    config_name="eval_qwen3.5-2b",
)
def main(cfg: DictConfig) -> None:
    """Report pass@1 and pass@4 for the configured HF model and validation set."""
    from sglang import Engine
    from transformers import AutoTokenizer

    from rlinf.utils.metric_logger import MetricLogger
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    rollout = cfg.rollout
    if cfg.runner.task_type != "reasoning_eval" or rollout.rollout_backend != "sglang":
        raise ValueError(
            f"Expected reasoning_eval with sglang, got {cfg.runner.task_type} with {rollout.rollout_backend}"
        )
    platform = Worker.torch_platform
    first_gpu, dp_size = _rollout_layout(
        cfg, platform.device_count() if platform else 0
    )
    get_logger().info(
        f"Local SGLang rollout starts at visible GPU {first_gpu}: DP={dp_size}, TP={rollout.tensor_parallel_size}, PP={rollout.pipeline_parallel_size}"
    )
    if cfg.algorithm.group_size < 1 or (
        cfg.algorithm.group_size > 1 and not cfg.algorithm.sampling_params.do_sample
    ):
        raise ValueError(
            "Use a positive group_size and enable sampling for multiple answers per question"
        )
    if cfg.runner.max_steps != -1 and cfg.runner.max_steps < 1:
        raise ValueError(
            f"Expected positive max_steps or -1, got {cfg.runner.max_steps}"
        )
    if (
        cfg.data.val_rollout_batch_size is not None
        and cfg.data.val_rollout_batch_size < 1
    ):
        raise ValueError(
            f"Expected a positive val_rollout_batch_size, got {cfg.data.val_rollout_batch_size}"
        )
    if cfg.data.dataset_name not in {"AIME2025", "AIME2026", "MATH-500", "DAPO-17K"}:
        raise ValueError(f"Unknown math benchmark: {cfg.data.dataset_name}")
    rows = _load_questions(cfg.data)
    sampling = SGLangWorker.get_sampling_param_from_config(
        cfg.algorithm.sampling_params
    )
    sampling["min_new_tokens"] = cfg.algorithm.sampling_params.min_new_tokens
    sampling["stop"] = OmegaConf.to_container(
        cfg.algorithm.sampling_params, resolve=True
    )["stop"]
    if (
        not cfg.data.max_prompt_length + sampling["max_new_tokens"]
        <= cfg.runner.seq_length
        <= rollout.max_model_len
    ):
        raise ValueError(
            "Expected prompt plus generation budgets to fit runner.seq_length and rollout.max_model_len"
        )

    tokenizer = AutoTokenizer.from_pretrained(rollout.model.model_path)
    run_dir = Path(cfg.runner.output_dir)
    model_name = cfg.runner.experiment_name or Path(rollout.model.model_path).name
    output = run_dir / model_name
    output.mkdir(parents=True, exist_ok=True)
    with Engine(
        model_path=rollout.model.model_path,
        skip_tokenizer_init=not sampling["stop"],
        dtype=str(torch_dtype_from_precision(rollout.model.precision)).removeprefix(
            "torch."
        ),
        context_length=rollout.max_model_len,
        random_seed=rollout.seed,
        base_gpu_id=first_gpu,
        dp_size=dp_size,
        tp_size=rollout.tensor_parallel_size,
        pp_size=rollout.pipeline_parallel_size,
        attention_backend=rollout.sglang.attention_backend,
        decode_log_interval=rollout.sglang.decode_log_interval,
        enable_torch_compile=rollout.sglang.use_torch_compile,
        torch_compile_max_bs=min(
            rollout.sglang.torch_compile_max_bs, rollout.max_running_requests
        ),
        mem_fraction_static=rollout.gpu_memory_utilization,
        max_running_requests=rollout.max_running_requests,
        cuda_graph_max_bs=rollout.cuda_graph_max_bs,
        disable_cuda_graph=rollout.enforce_eager,
    ) as engine:
        metrics = engine.loop.run_until_complete(
            _evaluate(
                engine,
                tokenizer,
                cfg,
                rows,
                sampling,
                output / f"{cfg.data.dataset_name}.jsonl",
                dp_size,
            )
        )
    metrics = {"model": model_name, **metrics}
    _write_summary(run_dir, metrics)
    scores = {
        key: metrics[key] for key in ("pass@1", "pass@4") if metrics[key] is not None
    }
    logger_cfg = OmegaConf.merge(cfg, {"runner": {"experiment_name": model_name}})
    logger = MetricLogger(logger_cfg)
    logger.log(
        {f"{cfg.data.dataset_name}/{key}": value for key, value in scores.items()},
        step=0,
    )
    logger.finish()
    report = ", ".join(f"{key}={value:.2%}" for key, value in scores.items())
    get_logger().info(
        f"{model_name} {cfg.data.dataset_name} ({metrics['questions']} questions): {report}"
    )


if __name__ == "__main__":
    main()
