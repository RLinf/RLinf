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

"""Diffusion environment tests."""

import multiprocessing
import sys
import time
from types import ModuleType


def _construct_mock_ocr_scorer(barrier, active, overlap):
    paddleocr = ModuleType("paddleocr")
    levenshtein = ModuleType("Levenshtein")

    class PaddleOCR:
        def __init__(self, **kwargs):
            with active.get_lock():
                active.value += 1
                overlap.value = max(overlap.value, active.value)
            time.sleep(0.2)
            with active.get_lock():
                active.value -= 1

    paddleocr.PaddleOCR = PaddleOCR
    levenshtein.distance = lambda left, right: abs(len(left) - len(right))
    sys.modules["paddleocr"] = paddleocr
    sys.modules["Levenshtein"] = levenshtein

    barrier.wait()
    from rlinf.envs.sim.diffusion.rewards.visual.ocr import OCRScorer

    OCRScorer()


def test_ocr_scorer_serializes_model_initialization(monkeypatch, tmp_path):
    """Concurrent workers do not race while populating the PaddleOCR cache."""
    monkeypatch.setenv("HOME", str(tmp_path))
    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    active = context.Value("i", 0)
    overlap = context.Value("i", 0)
    processes = [
        context.Process(
            target=_construct_mock_ocr_scorer,
            args=(barrier, active, overlap),
        )
        for _ in range(2)
    ]

    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=10)
        assert all(process.exitcode == 0 for process in processes)
        assert overlap.value == 1
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
