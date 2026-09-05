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

import numpy as np
import pytest

pytest.importorskip("openpi")

from openpi.models import model as openpi_model

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
from rlinf.models.embodiment.openpi.policies.turtle_policy import (
    TurtleInputs,
    TurtleOutputs,
)


@pytest.mark.parametrize("layout", ["hwc", "chw"])
def test_turtle_inputs_accept_three_hwc_or_chw_cameras(layout: str) -> None:
    height, width = 16, 20
    main = np.full((height, width, 3), 11, dtype=np.uint8)
    extra = np.stack(
        [
            np.full((height, width, 3), 22, dtype=np.uint8),
            np.full((height, width, 3), 33, dtype=np.uint8),
        ]
    )
    if layout == "chw":
        main = main.transpose(2, 0, 1)
        extra = extra.transpose(0, 3, 1, 2)

    inputs = TurtleInputs(model_type=openpi_model.ModelType.PI05)(
        {
            "observation/image": main,
            "observation/extra_view_image": extra,
            "observation/state": np.zeros(6, dtype=np.float32),
            "actions": np.zeros((50, 6), dtype=np.float32),
            "prompt": "press the button",
        }
    )

    assert inputs["image"]["base_0_rgb"].shape == (height, width, 3)
    assert inputs["image"]["left_wrist_0_rgb"].shape == (height, width, 3)
    assert inputs["image"]["right_wrist_0_rgb"].shape == (height, width, 3)
    assert all(bool(mask) for mask in inputs["image_mask"].values())
    assert inputs["state"].shape == (6,)
    assert inputs["actions"].shape == (50, 6)


def test_turtle_inputs_pad_missing_views() -> None:
    base = np.ones((12, 14, 3), dtype=np.uint8)
    inputs = TurtleInputs(model_type=openpi_model.ModelType.PI05)(
        {
            "observation/image": base,
            "observation/state": np.zeros(6, dtype=np.float32),
        }
    )

    for key in ("left_wrist_0_rgb", "right_wrist_0_rgb"):
        np.testing.assert_array_equal(inputs["image"][key], np.zeros_like(base))
        assert not bool(inputs["image_mask"][key])


def test_turtle_outputs_slice_six_dof_actions() -> None:
    outputs = TurtleOutputs()(
        {"actions": np.arange(50 * 32, dtype=np.float32).reshape(50, 32)}
    )
    assert outputs["actions"].shape == (50, 6)


def test_pi05_turtle_reuses_libero_checkpoint_asset_id() -> None:
    turtle = get_openpi_config("pi05_turtle")
    libero = get_openpi_config("pi05_libero")

    assert turtle.model.action_horizon == 50
    assert turtle.data.repo_id == libero.data.repo_id
