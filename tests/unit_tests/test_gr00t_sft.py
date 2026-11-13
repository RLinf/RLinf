# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Native GR00T window loss, input shapes and shared SFT training contracts."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from rlinf.data.datasets.gr00t_n1d7.window_dataset import WindowCollator
from rlinf.models.embodiment.gr00t.gr00t_n1d7.sft import (
    native_sft_forward,
    window_loss,
)


def test_window_loss_weights_windows_equally_with_different_valid_action_counts():
    errors = torch.tensor([1.0, 99.0, 3.0, 5.0], requires_grad=True)
    mask = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])
    loss = window_loss(errors.reshape(2, 1, 2) * mask, mask)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert errors.grad.tolist() == [0.5, 0.0, 0.25, 0.25]


@pytest.mark.parametrize("horizon", [16, 40])
def test_random_windows_reduce_over_real_action_values(horizon):
    targets = torch.tensor([1.0, 3.0], requires_grad=True)
    mask = torch.zeros(2, 40, 132)
    mask[:, :horizon, :12] = 1
    masked_errors = targets[:, None, None].expand_as(mask) * mask
    loss = window_loss(masked_errors, mask)
    assert loss.item() == pytest.approx(2.0)
    loss.backward()
    torch.testing.assert_close(targets.grad, torch.tensor([0.5, 0.5]))


@pytest.mark.parametrize("empty_window", [0, 1])
def test_loss_rejects_windows_without_supervised_actions(empty_window):
    mask = torch.ones(2, 1, 1)
    mask[empty_window] = 0
    with pytest.raises(ValueError):
        window_loss(torch.ones(2, 1, 1) * mask, mask)


@pytest.mark.parametrize(
    "loss_shape,mask_shape",
    [((2, 3), (2, 3)), ((2, 3, 4), (2, 3, 1)), ((0, 3, 4), (0, 3, 4))],
)
def test_loss_rejects_empty_or_invalid_action_shapes(loss_shape, mask_shape):
    with pytest.raises(ValueError):
        window_loss(torch.ones(loss_shape), torch.ones(mask_shape))


@pytest.mark.parametrize("horizon", [16, 40])
def test_collated_native_images_reach_forward_without_reshaping(horizon):
    from rlinf.models.embodiment.base_policy import ForwardType

    samples = []
    for index, value in enumerate((1.0, 3.0)):
        action_mask = torch.zeros(40, 132)
        action_mask[:horizon, :12] = 1
        samples.append(
            {
                "action": torch.full((40, 132), value),
                "action_mask": action_mask,
                "pixel_values": torch.arange(24).reshape(6, 4) + index * 24,
                "image_grid_thw": torch.arange(9).reshape(3, 3) + index * 9,
                "input_ids": torch.ones(5, dtype=torch.long),
            }
        )

    native_inputs = {}

    def collate(items):
        assert len(items) == len(samples)
        assert all(item is sample for item, sample in zip(items, samples))
        for key in samples[0]:
            values = [sample[key] for sample in items]
            native_inputs[key] = (
                torch.cat(values)
                if key in {"pixel_values", "image_grid_thw"}
                else torch.stack(values)
            )
        return {"inputs": native_inputs}

    batch = WindowCollator(SimpleNamespace(collator=collate))(samples)
    assert set(batch) == set(native_inputs)
    batch["description"] = "non-tensor metadata must not reach the native model"

    class NativeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.prepare_calls = 0
            self.head_calls = 0

        def prepare_input(self, inputs):
            self.prepare_calls += 1
            assert set(inputs) == {
                "action",
                "action_mask",
                "pixel_values",
                "image_grid_thw",
                "input_ids",
            }
            assert inputs["action"].shape == (2, 40, 132)
            assert inputs["input_ids"].shape == (2, 5)
            assert inputs["pixel_values"].shape == (12, 4)
            assert inputs["image_grid_thw"].shape == (6, 3)
            for key in inputs:
                torch.testing.assert_close(inputs[key], native_inputs[key])
            native = {
                key: value.to(self.weight.device) for key, value in inputs.items()
            }
            return native, native

        def backbone(self, inputs):
            return inputs["input_ids"]

        def action_head(self, features, inputs):
            self.head_calls += 1
            assert features.shape[0] == 2
            return {
                "action_loss": (self.weight * inputs["action"]).square()
                * inputs["action_mask"],
                "action_mask": inputs["action_mask"],
            }

    model = NativeModel()
    output = native_sft_forward(model, ForwardType.SFT, batch)
    assert output["loss"].item() == pytest.approx(5.0)
    assert model.prepare_calls == model.head_calls == 1
    assert model.weight.grad is None
    assert batch["pixel_values"].shape == (12, 4)
    assert batch["image_grid_thw"].shape == (6, 3)
    output["loss"].backward()
    assert model.weight.grad.item() == pytest.approx(10.0)


def test_public_training_accumulates_microbatches_before_one_optimizer_update(
    monkeypatch,
):
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.workers.sft import fsdp_sft_worker
    from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker

    monkeypatch.setattr(fsdp_sft_worker, "all_reduce_dict", lambda values, op: values)

    class TinyNativeHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            self.forward_weights = []

        def forward(self, forward_type, data):
            assert forward_type == ForwardType.SFT
            self.forward_weights.append(self.weight.detach().item())
            error = (self.weight * data["action"]).square() * data["action_mask"]
            return {"loss": window_loss(error, data["action_mask"])}

    sync_calls = []

    @contextmanager
    def context(model, is_last_micro_batch):
        sync_calls.append(is_last_micro_batch)
        yield

    model = TinyNativeHead()
    model.eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    gradients_at_step = []
    scheduler_steps = []

    def optimizer_step():
        gradient = model.weight.grad.detach().item()
        gradients_at_step.append(gradient)
        optimizer.step()
        return abs(gradient), [0.01]

    batch = {
        "action": torch.tensor([1.0, 3.0]).reshape(2, 1, 1),
        "action_mask": torch.ones(2, 1, 1),
    }
    next_batch = {**batch, "action": 2 * batch["action"]}
    worker = SimpleNamespace(
        model=model,
        optimizer=optimizer,
        optimizer_step=optimizer_step,
        gradient_accumulation=2,
        data_iter=iter([batch, next_batch]),
        _data_iter_offset=0,
        global_step=0,
        worker_timer=lambda: nullcontext(),
        amp_context=nullcontext(),
        before_micro_batch=context,
        grad_scaler=torch.amp.GradScaler("cpu", enabled=False),
        lr_scheduler=SimpleNamespace(step=lambda: scheduler_steps.append(True)),
    )

    def train_model_output(data):
        return FSDPVlaSftWorker.get_train_model_output(worker, data)

    worker.get_train_model_output = train_model_output

    metrics = FSDPVlaSftWorker.run_training(worker)

    assert sync_calls == [False, True]
    assert model.forward_weights == [1.0, 1.0]
    assert gradients_at_step == pytest.approx([25.0])
    assert model.weight.item() == pytest.approx(0.75)
    assert model.weight.grad is None
    assert model.training
    assert worker._data_iter_offset == 2
    assert scheduler_steps == [True]
    assert metrics["loss"] == pytest.approx(12.5)
