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
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi.apxinf_adapter import OpenPIApxInfAdapter


class _FakePolicy:
    action_horizon = 10
    action_dim = 7
    metadata = {"model_action_dim": 32}

    def __init__(self):
        self.calls = []
        self.closed = False

    def infer(self, observation, *, noise=None):
        self.calls.append((observation, noise))
        offset = len(self.calls) * 1000
        actions = np.arange(70, dtype=np.float32).reshape(10, 7) + offset
        return {"actions": actions, "timing": {"model_ms": 1.0}}

    def close(self):
        self.closed = True


def _model_cfg(**apxinf_overrides):
    apxinf = {
        "action_horizon": 10,
        "num_flow_steps": 5,
        "noise_source": "apxinf",
        "seed": 0,
        **apxinf_overrides,
    }
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": "/not/loaded/in/unit/test",
            "num_action_chunks": 5,
            "action_dim": 7,
            "openpi": {
                "config_name": "pi05_libero",
                "num_steps": 5,
                "noise_method": "flow_sde",
                "noise_level": 0.3,
            },
            "apxinf": apxinf,
        }
    )


def _env_obs(batch_size=2):
    main = torch.arange(batch_size * 4 * 5 * 3, dtype=torch.uint8).reshape(
        batch_size, 4, 5, 3
    )
    wrist = main + 1
    return {
        "main_images": main,
        "wrist_images": wrist,
        "extra_view_images": None,
        "states": torch.arange(batch_size * 8, dtype=torch.float32).reshape(
            batch_size, 8
        ),
        "task_descriptions": [f"task {index}" for index in range(batch_size)],
    }


def test_maps_rlinf_batch_without_repeating_env_transforms_and_slices_time():
    policy = _FakePolicy()
    adapter = OpenPIApxInfAdapter(_model_cfg(), "cpu", policy=policy)
    env_obs = _env_obs()

    actions, result = adapter.predict_action_batch(env_obs, mode="eval")

    assert actions.shape == (2, 5, 7)
    assert actions.dtype == torch.float32
    np.testing.assert_array_equal(
        policy.calls[0][0]["observation/image"], env_obs["main_images"][0].numpy()
    )
    np.testing.assert_array_equal(
        policy.calls[1][0]["observation/wrist_image"],
        env_obs["wrist_images"][1].numpy(),
    )
    np.testing.assert_array_equal(
        policy.calls[0][0]["observation/state"], env_obs["states"][0].numpy()
    )
    assert policy.calls[1][0]["prompt"] == "task 1"
    assert len(result["apxinf_timing"]) == 2


def test_explicit_noise_is_split_and_forwarded_exactly():
    policy = _FakePolicy()
    adapter = OpenPIApxInfAdapter(
        _model_cfg(noise_source="observation"), "cpu", policy=policy
    )
    env_obs = _env_obs()
    env_obs["noise"] = torch.arange(2 * 10 * 32, dtype=torch.float32).reshape(2, 10, 32)

    adapter.predict_action_batch(env_obs)

    np.testing.assert_array_equal(policy.calls[0][1], env_obs["noise"][0].numpy())
    np.testing.assert_array_equal(policy.calls[1][1], env_obs["noise"][1].numpy())


def test_torch_noise_is_reproducible_and_has_model_shape():
    obs = _env_obs()
    policy_a = _FakePolicy()
    policy_b = _FakePolicy()
    adapter_a = OpenPIApxInfAdapter(
        _model_cfg(noise_source="torch", seed=7), "cpu", policy=policy_a
    )
    adapter_b = OpenPIApxInfAdapter(
        _model_cfg(noise_source="torch", seed=7), "cpu", policy=policy_b
    )

    adapter_a.predict_action_batch(obs)
    adapter_b.predict_action_batch(obs)

    assert policy_a.calls[0][1].shape == (10, 32)
    np.testing.assert_array_equal(policy_a.calls[0][1], policy_b.calls[0][1])
    np.testing.assert_array_equal(policy_a.calls[1][1], policy_b.calls[1][1])


def test_rejects_mismatched_openpi_and_apxinf_flow_steps():
    with pytest.raises(ValueError, match="must match OpenPI num_steps"):
        OpenPIApxInfAdapter(_model_cfg(num_flow_steps=10), "cpu", policy=_FakePolicy())


def test_rejects_more_than_two_libero_views():
    adapter = OpenPIApxInfAdapter(_model_cfg(), "cpu", policy=_FakePolicy())
    obs = _env_obs()
    obs["extra_view_images"] = torch.zeros_like(obs["main_images"])
    with pytest.raises(ValueError, match="exactly two camera views"):
        adapter.predict_action_batch(obs)


def test_close_delegates_to_policy():
    policy = _FakePolicy()
    adapter = OpenPIApxInfAdapter(_model_cfg(), "cpu", policy=policy)
    adapter.close()
    assert policy.closed
