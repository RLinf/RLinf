import torch

from rlinf.data.datasets import sft_collate_fn
from rlinf.data.datasets.item import SftDatasetItem
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

    batch = sft_collate_fn(items)

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
