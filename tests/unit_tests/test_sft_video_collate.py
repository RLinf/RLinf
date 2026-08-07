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

from contextlib import nullcontext
from unittest.mock import MagicMock

import torch

from rlinf.data.datasets.common.item import SftDatasetItem
from rlinf.data.datasets.vlm import collate_fn
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker
from rlinf.workers.sft.fsdp_vlm_sft_worker import FSDPVlmSftWorker


def test_sft_collate_supports_qwen_video_inputs():
    items = [
        SftDatasetItem(
            prompt=torch.tensor([1, 2]),
            length=2,
            answer=str(index),
            idx=index,
            attention_mask=torch.ones(2, dtype=torch.long),
            label_mask=torch.tensor([True, False]),
            multi_modal_inputs={
                "pixel_values_videos": torch.full((2, 3), float(index)),
                "video_grid_thw": torch.tensor([[1, 2, 3]]),
            },
        )
        for index in range(2)
    ]

    batch = collate_fn(items)

    assert len(batch["multi_modal_inputs"]["pixel_values_videos"]) == 2
    assert batch["multi_modal_inputs"]["video_grid_thw"].shape == (2, 3)


def test_weighted_answer_loss_applies_class_weights():
    logits = torch.zeros(2, 3, 2)
    logits[0, 0] = torch.tensor([2.0, -2.0])
    logits[1, 0] = torch.tensor([2.0, -2.0])
    labels = torch.tensor([[-100, 0, -100], [-100, 1, -100]])

    worker = object.__new__(FSDPVlmSftWorker)
    loss = worker.weighted_answer_loss(
        logits, labels, ["1", "0"], success_weight=3.0, non_success_weight=1.0
    )

    sample_losses = torch.nn.functional.cross_entropy(
        logits[:, 0], torch.tensor([0, 1]), reduction="none"
    )
    expected = (3.0 * sample_losses[0] + sample_losses[1]) / 4.0
    torch.testing.assert_close(loss, expected)


def test_binary_eval_metrics_are_computed_by_vlm_worker():
    worker = object.__new__(FSDPVlmSftWorker)

    metrics = worker.compute_eval_metrics(
        {
            "correct": 7,
            "total": 10,
            "binary_total": 10,
            "positive_correct": 3,
            "positive_total": 5,
            "negative_correct": 4,
            "negative_total": 5,
        }
    )

    assert metrics["eval_accuracy"] == 0.7
    assert metrics["positive_recall"] == 0.6
    assert metrics["negative_accuracy"] == 0.8
    assert metrics["balanced_accuracy"] == 0.7


def test_compute_eval_metrics_skips_balanced_accuracy_for_non_binary_dataset():
    worker = object.__new__(FSDPVlmSftWorker)

    metrics = worker.compute_eval_metrics(
        {
            "correct": 4,
            "total": 6,
            "binary_total": 4,
            "positive_correct": 1,
            "positive_total": 2,
            "negative_correct": 2,
            "negative_total": 2,
        }
    )

    assert metrics == {"eval_accuracy": 4 / 6}
    assert "balanced_accuracy" not in metrics


class _SizedEvalIter:
    """Iterator that supports ``len()`` like RLinf's eval data loaders."""

    def __init__(self, items):
        self._items = list(items)
        self._index = 0

    def __len__(self):
        return len(self._items)

    def __iter__(self):
        return self

    def __next__(self):
        if self._index >= len(self._items):
            raise StopIteration
        item = self._items[self._index]
        self._index += 1
        return item


class _SizedEvalLoader:
    def __init__(self, items):
        self._items = list(items)

    def __iter__(self):
        return _SizedEvalIter(self._items)


def _eval_batch(answers: list[str]) -> dict:
    batch_size = len(answers)
    return {
        "prompt": torch.zeros(batch_size, 2, dtype=torch.long),
        "answer": answers,
        "attention_mask": torch.ones(batch_size, 2, dtype=torch.long),
        "multi_modal_inputs": {},
    }


def _make_vlm_eval_worker(monkeypatch, batches: list[dict]) -> FSDPVlmSftWorker:
    """Build a VLM SFT worker whose generate path echoes each batch's gold labels."""
    worker = object.__new__(FSDPVlmSftWorker)
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.worker_timer = lambda: nullcontext()
    worker.eval_batch_size = max(len(batch["answer"]) for batch in batches)
    worker.eval_data_loader = _SizedEvalLoader(batches)
    worker.model = MagicMock()
    worker.cfg = type("Cfg", (), {})()
    worker.cfg.actor = type("Actor", (), {})()
    worker.cfg.actor.model = type("Model", (), {"model_type": "qwen3_vl"})()

    class _Tok:
        eos_token_id = 2
        pad_token_id = 0

        def decode(self, _ids, skip_special_tokens=False):
            return ""

    worker.tokenizer = _Tok()

    answer_queues = [list(batch["answer"]) for batch in batches]
    pred_queue: list[str] = []

    def _on_generate(**kwargs):
        answers = answer_queues.pop(0)
        assert len(answers) == kwargs["input_ids"].shape[0]
        pred_queue.extend(answers)
        return torch.zeros(
            kwargs["input_ids"].shape[0], kwargs["input_ids"].shape[1] + 1
        )

    monkeypatch.setattr(
        "rlinf.workers.sft.fsdp_vlm_sft_worker.generate_with_kv_cache",
        _on_generate,
    )
    monkeypatch.setattr(
        "rlinf.workers.sft.fsdp_vlm_sft_worker.vlm_extract_answer",
        lambda _text, _model_type: pred_queue.pop(0),
    )
    monkeypatch.setattr(
        "rlinf.workers.sft.fsdp_sft_worker.all_reduce_dict",
        lambda metrics, op=None: metrics,
    )
    return worker


def test_run_eval_with_0_1_2_labels_across_different_batches(monkeypatch):
    """Regression requested by reviewer: 0/1/2 labels across different batches.

    Batch 0 is purely binary (``0``/``1``); batch 1 contains a non-binary
    potential label (``2``). Both go through the real ``get_eval_model_output``
    path into ``run_eval``. All four samples must count toward accuracy;
    balanced accuracy must not be reported for a non-binary eval set.
    """
    batches = [
        _eval_batch(["0", "1"]),  # binary-only batch
        _eval_batch(["2", "0"]),  # mixed / potential batch
    ]
    worker = _make_vlm_eval_worker(monkeypatch, batches)

    metrics = FSDPSftWorker.run_eval(worker)

    assert metrics["eval_accuracy"] == 1.0
    assert "balanced_accuracy" not in metrics
    assert "positive_recall" not in metrics
