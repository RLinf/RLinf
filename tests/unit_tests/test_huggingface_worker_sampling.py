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

import inspect
from types import SimpleNamespace

import torch

from rlinf.config import SupportedModel
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _CapturingPolicy:
    def __init__(self):
        self.kwargs = None

    def predict_action_batch(self, *, env_obs, **kwargs):
        del env_obs
        self.kwargs = kwargs
        return torch.zeros(1, 1, 1), {}


def _make_worker(model_type):
    worker = object.__new__(MultiStepRolloutWorker)
    worker.model_cfg = SimpleNamespace(model_type=model_type.value)
    worker._train_sampling_params = {
        "temperature": 0.3,
        "top_k": 50,
        "do_sample": True,
    }
    worker._eval_sampling_params = {"do_sample": False}
    worker.enable_dagger = False
    worker.algorithm_cfg = {}
    worker.expert_model = None
    worker.hf_model = _CapturingPolicy()
    return worker


def _predict_without_timer(worker, mode="train"):
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    return predict(worker, {}, mode=mode)


def test_pi0_fast_receives_sampling_parameters_and_mode():
    worker = _make_worker(SupportedModel.PI0_FAST)

    _predict_without_timer(worker)

    assert worker.hf_model.kwargs == {
        "temperature": 0.3,
        "top_k": 50,
        "do_sample": True,
        "mode": "train",
    }


def test_existing_action_model_only_receives_mode():
    worker = _make_worker(SupportedModel.OPENPI)

    _predict_without_timer(worker)

    assert worker.hf_model.kwargs == {"mode": "train"}
