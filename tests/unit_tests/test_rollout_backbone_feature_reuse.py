# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch


def test_rollout_reuse_captures_raw_backbone_output_before_action_head_mutation():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        GR00T_N1_5_ForRLActionPrediction,
    )
    from rlinf.utils.backbone_cache import (
        ROLLOUT_BACKBONE_FEATURE_KEY,
        ROLLOUT_BACKBONE_MASK_KEY,
    )

    raw_features = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
        requires_grad=True,
    )
    raw_mask = torch.ones(2, 2, dtype=torch.bool)

    class FakeBackbone:
        def __call__(self, backbone_inputs):
            del backbone_inputs
            return BatchFeature(
                data={
                    "backbone_features": raw_features,
                    "backbone_attention_mask": raw_mask,
                }
            )

    class MutatingActionHead:
        @staticmethod
        def get_rl_action(backbone_outputs, action_inputs, mode):
            del action_inputs, mode
            # Matches FlowmatchingActionHead.process_backbone_output(): the
            # BatchFeature mapping is updated with action-head-processed data.
            backbone_outputs["backbone_features"] = (
                backbone_outputs["backbone_features"] + 100.0
            )
            batch_size = backbone_outputs["backbone_features"].shape[0]
            return {}, {
                "actions": torch.zeros(batch_size, 1),
                "chains": torch.zeros(batch_size, 1),
                "denoise_inds": torch.zeros(batch_size, 1, dtype=torch.int64),
                "prev_logprobs": torch.zeros(batch_size, 1),
                "prev_values": torch.zeros(batch_size, 1),
            }

    class FakePolicy:
        image_nums = 1
        backbone = FakeBackbone()
        action_head = MutatingActionHead()
        capture_rollout_backbone_output = True

        @staticmethod
        def prepare_input(normalized_input):
            del normalized_input
            return BatchFeature(data={}), BatchFeature(data={})

        @staticmethod
        def validate_data(action_head_outputs, backbone_outputs, is_training):
            del action_head_outputs, backbone_outputs, is_training

    normalized_input = {
        "state": torch.zeros(2, 1),
        "state_mask": torch.ones(2, 1),
        "eagle_input_ids": torch.ones(2, 1, dtype=torch.int64),
        "eagle_attention_mask": torch.ones(2, 1, dtype=torch.int64),
        "eagle_pixel_values": torch.ones(2, 1, 1),
        "eagle_image_sizes": torch.ones(2, 1, 1),
        "embodiment_id": torch.zeros(2, dtype=torch.int64),
    }

    _, result = GR00T_N1_5_ForRLActionPrediction._get_rl_action(
        FakePolicy(), normalized_input
    )
    forward_inputs = result["forward_inputs"]

    torch.testing.assert_close(
        forward_inputs[ROLLOUT_BACKBONE_FEATURE_KEY], raw_features.detach()
    )
    torch.testing.assert_close(forward_inputs[ROLLOUT_BACKBONE_MASK_KEY], raw_mask)
    assert not forward_inputs[ROLLOUT_BACKBONE_FEATURE_KEY].requires_grad


def test_rollout_does_not_retain_backbone_output_when_transport_is_disabled():
    pytest.importorskip("gr00t")
    from transformers.feature_extraction_utils import BatchFeature

    from rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model import (
        GR00T_N1_5_ForRLActionPrediction,
    )
    from rlinf.utils.backbone_cache import (
        ROLLOUT_BACKBONE_FEATURE_KEY,
        ROLLOUT_BACKBONE_MASK_KEY,
    )

    class FakeBackbone:
        @staticmethod
        def __call__(backbone_inputs):
            del backbone_inputs
            return BatchFeature(
                data={
                    "backbone_features": torch.ones(1, 1, 2),
                    "backbone_attention_mask": torch.ones(1, 1, dtype=torch.bool),
                }
            )

    class FakeActionHead:
        @staticmethod
        def get_rl_action(backbone_outputs, action_inputs, mode):
            del backbone_outputs, action_inputs, mode
            return {}, {
                "actions": torch.zeros(1, 1),
                "chains": torch.zeros(1, 1),
                "denoise_inds": torch.zeros(1, 1, dtype=torch.int64),
                "prev_logprobs": torch.zeros(1, 1),
                "prev_values": torch.zeros(1, 1),
            }

    class FakePolicy:
        image_nums = 1
        backbone = FakeBackbone()
        action_head = FakeActionHead()

        @staticmethod
        def prepare_input(normalized_input):
            del normalized_input
            return BatchFeature(data={}), BatchFeature(data={})

        @staticmethod
        def validate_data(action_head_outputs, backbone_outputs, is_training):
            del action_head_outputs, backbone_outputs, is_training

    normalized_input = {
        "state": torch.zeros(1, 1),
        "state_mask": torch.ones(1, 1),
        "eagle_input_ids": torch.ones(1, 1, dtype=torch.int64),
        "eagle_attention_mask": torch.ones(1, 1, dtype=torch.int64),
        "eagle_pixel_values": torch.ones(1, 1, 1),
        "eagle_image_sizes": torch.ones(1, 1, 1),
        "embodiment_id": torch.zeros(1, dtype=torch.int64),
    }

    _, result = GR00T_N1_5_ForRLActionPrediction._get_rl_action(
        FakePolicy(), normalized_input
    )

    assert ROLLOUT_BACKBONE_FEATURE_KEY not in result["forward_inputs"]
    assert ROLLOUT_BACKBONE_MASK_KEY not in result["forward_inputs"]
