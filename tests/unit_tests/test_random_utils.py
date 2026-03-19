"""Tests for RNG seeding helpers."""

import random

import numpy as np
import torch

from rlinf.utils.utils import seed_dataloader_worker, seed_everything


def _sample_rng_triplet():
    """Collect one sample from Python, NumPy, and Torch RNGs."""
    return (
        [random.random() for _ in range(3)],
        np.random.rand(3),
        torch.rand(3),
    )


def test_seed_everything_replays_python_numpy_and_torch_rng():
    """Repeated seeding should replay the same RNG streams."""
    seed_everything(1234)
    first_python, first_numpy, first_torch = _sample_rng_triplet()

    seed_everything(1234)
    second_python, second_numpy, second_torch = _sample_rng_triplet()

    assert first_python == second_python
    np.testing.assert_allclose(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)


def test_seed_dataloader_worker_uses_torch_initial_seed(monkeypatch):
    """Worker seeding should derive Python/NumPy seeds from torch.initial_seed."""
    monkeypatch.setattr(torch, "initial_seed", lambda: 2**40 + 17)

    seed_dataloader_worker(worker_id=0)
    first_python, first_numpy, first_torch = _sample_rng_triplet()

    seed_dataloader_worker(worker_id=3)
    second_python, second_numpy, second_torch = _sample_rng_triplet()

    assert first_python == second_python
    np.testing.assert_allclose(first_numpy, second_numpy)
    assert torch.equal(first_torch, second_torch)
