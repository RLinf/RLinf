# Copyright 2025 The RLinf Authors.
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

from types import SimpleNamespace

import pytest
import torch


def test_gr00t_fresh_and_precomputed_backbone_outputs_match():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        GR00T_N1_5_ForRLActionPrediction,
    )

    class FakeBackbone:
        def __init__(self):
            self.calls = 0

        def __call__(self, inputs):
            self.calls += 1
            return BatchFeature(data={"backbone_features": inputs["features"] * 2})

    class FakeActionHead:
        action_chunk = 1
        rl_config = SimpleNamespace(joint_logprob=False)
        dtype = torch.float32

        @staticmethod
        def prepare_input(batch):
            return BatchFeature(data=batch)

        def __call__(
            self,
            *,
            backbone_output,
            action_input,
            chains,
            denoise_inds,
            compute_values,
        ):
            del action_input, chains, denoise_inds, compute_values
            features = backbone_output["backbone_features"]
            log_probs = features[:, None, None, :2]
            values = features.mean(dim=-1, keepdim=True)
            return log_probs, values

    class FakePolicy:
        valid_action_dim = 2
        device = torch.device("cpu")
        _prepare_action_head_input = (
            GR00T_N1_5_ForRLActionPrediction._prepare_action_head_input
        )

        def __init__(self):
            self.backbone = FakeBackbone()
            self.action_head = FakeActionHead()

        @staticmethod
        def prepare_input(normalized_input):
            return (
                BatchFeature(data={"features": normalized_input["state"]}),
                BatchFeature(data={}),
            )

    batch_size = 2
    forward_inputs = {
        "state": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        "state_mask": torch.ones(batch_size, 2),
        "eagle_input_ids": torch.ones(batch_size, 1, dtype=torch.int64),
        "eagle_attention_mask": torch.ones(batch_size, 1, dtype=torch.int64),
        "eagle_pixel_values": torch.ones(batch_size, 1, 1),
        "eagle_image_sizes": torch.ones(batch_size, 1, 1),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.int64),
        "chains": torch.zeros(batch_size, 1),
        "denoise_inds": torch.zeros(batch_size, 1, dtype=torch.int64),
    }
    prev_logprobs = torch.zeros(batch_size, 1, 1, 2)

    fresh = GR00T_N1_5_ForRLActionPrediction.default_forward(
        FakePolicy(),
        forward_inputs=forward_inputs,
        prev_logprobs=prev_logprobs,
    )
    raw_backbone = {
        "backbone_features": (forward_inputs["state"] * 2).detach(),
    }

    policy = FakePolicy()
    feature_only_inputs = {
        key: value
        for key, value in forward_inputs.items()
        if not key.startswith("eagle_")
    }
    cached = GR00T_N1_5_ForRLActionPrediction.default_forward(
        policy,
        forward_inputs=feature_only_inputs,
        prev_logprobs=prev_logprobs,
        precomputed_backbone=raw_backbone,
    )

    assert policy.backbone.calls == 0
    assert all(not key.startswith("eagle_") for key in feature_only_inputs)
    assert not raw_backbone["backbone_features"].requires_grad
    for key in ("logprobs", "prev_logprobs", "values"):
        torch.testing.assert_close(cached[key], fresh[key])
