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


import copy
import io

import torch
from omegaconf import OmegaConf

from rlinf.algorithms.qam import adjoint_targets, qam_update
from rlinf.models import get_model


def small_config():
    return OmegaConf.create(
        {
            "model_type": "qam_policy",
            "precision": "fp32",
            "is_lora": False,
            "load_to_device": False,
            "obs_dim": 3,
            "action_dim": 2,
            "horizon_length": 1,
            "flow_steps": 3,
            "actor_hidden_dims": [8, 8],
            "value_hidden_dims": [8, 8],
            "actor_layer_norm": False,
            "value_layer_norm": True,
            "num_qs": 2,
            "inv_temp": 10.0,
            "best_of_n": 1,
            "rho": 0.5,
            "discount": 0.99,
            "tau": 0.005,
            "action_chunking": True,
        }
    )


def sample_batch():
    return {
        "observations": torch.randn(4, 3),
        "actions": torch.randn(4, 1, 2),
        "next_observations": torch.randn(4, 1, 3),
        "rewards": torch.randn(4, 1),
        "masks": torch.tensor([[1.0], [0.0], [1.0], [1.0]]),
        "valid": torch.ones(4, 1),
    }


def test_detached_targets_old_parameter_ema_and_resume():
    torch.manual_seed(7)
    model = get_model(small_config())
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    generator = torch.Generator().manual_seed(17)
    batch = sample_batch()
    old = copy.deepcopy(model.state_dict())
    targets = adjoint_targets(
        model, batch["observations"], torch.randn(4, 2), torch.randn(3, 4, 2)
    )
    assert all(not x.requires_grad for x in targets)
    info = qam_update(model, optimizer, batch, generator)
    assert all(torch.isfinite(v) for v in info.values())
    assert any(
        not torch.equal(v, old[k])
        for k, v in model.state_dict().items()
        if k.startswith("actor_fast")
    )
    for name, target in (
        ("critic", "target_critic"),
        ("actor_slow", "target_actor_slow"),
    ):
        for k, v in getattr(model, target).state_dict().items():
            expected = old[f"{target}.{k}"] * 0.995 + old[f"{name}.{k}"] * 0.005
            torch.testing.assert_close(v, expected)
    assert all(p.grad is None for p in model.target_critic.parameters())
    assert all(p.grad is None for p in model.target_actor_slow.parameters())
    stream = io.BytesIO()
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "rng": generator.get_state(),
        },
        stream,
    )
    qam_update(model, optimizer, batch, generator)
    expected = copy.deepcopy(model.state_dict())
    stream.seek(0)
    state = torch.load(stream, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    generator.set_state(state["rng"])
    qam_update(model, optimizer, batch, generator)
    for k, v in model.state_dict().items():
        torch.testing.assert_close(v, expected[k], rtol=0, atol=0)
