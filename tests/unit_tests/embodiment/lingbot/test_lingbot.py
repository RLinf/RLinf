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
# limitations under the License
import pytest

from rlinf.config import SupportedModel, get_supported_model


def test_lingbot_enum_registration():
    """Test if LingBot is correctly registered in the supported models enum."""
    model_type = get_supported_model("lingbot")
    assert model_type == SupportedModel.LINGBOT
    assert model_type.category == "embodied"


def test_lingbot_action_model_import():
    """Test if the LingBot policy wrapper can be imported without syntax/dependency errors."""
    try:
        from rlinf.models.embodiment.lingbot.lingbot_action_model import (
            LingbotForRLActionPrediction,
        )

        assert LingbotForRLActionPrediction is not None
    except ImportError as e:
        pytest.skip(f"Skipping import test due to missing strict dependencies: {e}")
