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

from rlinf.models.embodiment.starvla.dispatch import (
    get_default_forward_handler,
    get_rollout_handler,
)


def test_dispatch_handler_mapping_for_adapter_fast_flowmatching():
    adapter_forward = get_default_forward_handler("adapter")
    fast_forward = get_default_forward_handler("fast")
    flow_forward = get_default_forward_handler("pi")

    adapter_rollout = get_rollout_handler("adapter")
    fast_rollout = get_rollout_handler("fast")
    flow_rollout = get_rollout_handler("pi")

    assert adapter_forward is not None
    assert fast_forward is not None
    assert flow_forward is not None
    assert adapter_rollout is not None
    assert fast_rollout is not None
    assert flow_rollout is not None

    assert adapter_forward.__name__ == "run_default_forward_adapter"
    assert fast_forward.__name__ == "run_default_forward_fast"
    assert flow_forward.__name__ == "run_default_forward_flowmatching"
    assert adapter_rollout.__name__ == "run_rollout_adapter"
    assert fast_rollout.__name__ == "run_rollout_fast"
    assert flow_rollout.__name__ == "run_rollout_flowmatching"


def test_dispatch_unknown_head_returns_none():
    assert get_default_forward_handler("unknown_head") is None
    assert get_rollout_handler("unknown_head") is None
