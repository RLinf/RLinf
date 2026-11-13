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


"""Optional reference oracle; never imported by training.

Set QAM_REFERENCE_ROOT to a checkout of 2726d767, JAX_PLATFORMS=cpu,
CUDA_VISIBLE_DEVICES='' and run this file in an isolated pytest process.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.qam import adjoint_targets, qam_loss, qam_update
from rlinf.models import get_model


@pytest.mark.parametrize("horizon", [1, 2])
def test_reference_losses_gradients_update(horizon):
    root = os.environ.get("QAM_REFERENCE_ROOT")
    if not root:
        pytest.skip("Optional pinned QAM reference checkout is not configured")
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    expected_commit = "2726d767c9a0a7a46d49693f0391f73dc2cf58ac"
    actual_commit = subprocess.check_output(
        ["git", "-C", root, "rev-parse", "HEAD"], text=True
    ).strip()
    assert actual_commit == expected_commit, (
        f"Expected official QAM {expected_commit}, got {actual_commit}"
    )
    assert not subprocess.check_output(
        ["git", "-C", root, "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip(), "QAM reference must have no tracked modifications"
    sys.path.insert(0, root)
    try:
        import jax
        import jax.numpy as jnp
        from agents.qam import QAMAgent, get_config
    finally:
        sys.path.remove(root)

    torch.set_num_threads(1)
    torch.manual_seed(42)
    c = get_config()
    c.horizon_length = horizon
    c.action_chunking = True
    c.actor_hidden_dims = (8, 8)
    c.value_hidden_dims = (8, 8)
    c.num_qs = 2
    c.flow_steps = 3
    c.inv_temp = 10.0
    ref = QAMAgent.create(17, np.zeros(3, np.float32), np.zeros(2, np.float32), c)
    cfg = OmegaConf.create(
        {
            "model_type": "qam_policy",
            "precision": "fp32",
            "is_lora": False,
            "load_to_device": False,
            "obs_dim": 3,
            "action_dim": 2,
            "horizon_length": horizon,
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
    model = get_model(cfg)

    # Explicit orientation mapping, including the ensemble axis and LayerNorm.
    pairs = []
    for name in ("actor_fast", "actor_slow", "target_actor_slow"):
        ref.network.params[f"modules_{name}"]["mlp"]
        module = getattr(model, name)
        for i, layer in enumerate(module.layers):
            pairs += [
                (
                    layer.weight,
                    (f"modules_{name}", "mlp", f"Dense_{i}", "kernel"),
                    True,
                ),
                (layer.bias, (f"modules_{name}", "mlp", f"Dense_{i}", "bias"), False),
            ]
    for name in ("critic", "target_critic"):
        module = getattr(model, name)
        for i, (w, b) in enumerate(zip(module.weights, module.biases)):
            pairs += [
                (w, (f"modules_{name}", "value_net", f"Dense_{i}", "kernel"), False),
                (b, (f"modules_{name}", "value_net", f"Dense_{i}", "bias"), False),
            ]
        for i, (s, b) in enumerate(zip(module.scales, module.offsets)):
            pairs += [
                (s, (f"modules_{name}", "value_net", f"LayerNorm_{i}", "scale"), False),
                (b, (f"modules_{name}", "value_net", f"LayerNorm_{i}", "bias"), False),
            ]

    def lookup(tree, path):
        for key in path:
            tree = tree[key]
        return np.asarray(tree)

    def mapped(tree, path, transpose, shape):
        value = lookup(tree, path)
        return (value.T if transpose else value).reshape(shape)

    with torch.no_grad():
        for param, path, transpose in pairs:
            param.copy_(
                torch.tensor(mapped(ref.network.params, path, transpose, param.shape))
            )
    rng = jax.random.PRNGKey(77)
    _, actor_rng, critic_rng = jax.random.split(rng, 3)
    _, x_rng, t_rng, adj_rng, _ = jax.random.split(actor_rng, 5)
    next_rng, _ = jax.random.split(critic_rng)
    b, a = 4, 2 * horizon
    noise = {
        "next": jax.random.normal(next_rng, (b, 1, a)),
        "flow": jax.random.normal(x_rng, (b, a)),
        "time": jax.random.uniform(t_rng, (b, 1)),
        "initial": jax.random.normal(adj_rng, (b, a)),
        "steps": jnp.stack(
            [jax.random.normal(key, (b, a)) for key in jax.random.split(adj_rng, 3)]
        ),
    }
    nrng = np.random.RandomState(19)
    batch = {
        "observations": nrng.randn(b, 3).astype("float32"),
        "actions": nrng.uniform(-1, 1, (b, horizon, 2)).astype("float32"),
        "next_observations": nrng.randn(b, horizon, 3).astype("float32"),
        "rewards": nrng.randn(b, horizon).astype("float32"),
        "masks": np.ones((b, horizon), "float32"),
        "valid": np.ones((b, horizon), "float32"),
    }
    batch["masks"][1, -1] = 0
    tb = {k: torch.tensor(v) for k, v in batch.items()}
    tn = {k: torch.tensor(np.asarray(v)) for k, v in noise.items()}
    ref_targets = ref.adj_matching(jnp.asarray(batch["observations"]), adj_rng)
    targets = adjoint_targets(model, tb["observations"], tn["initial"], tn["steps"])
    for actual, expected in zip(targets[:3], ref_targets[:3]):
        np.testing.assert_allclose(
            actual.numpy(), np.asarray(expected), atol=2e-4, rtol=2e-4
        )
    (ref_loss, ref_info), grads = jax.value_and_grad(
        ref.total_loss, argnums=1, has_aux=True
    )({k: jnp.asarray(v) for k, v in batch.items()}, ref.network.params, rng)
    loss, info = qam_loss(model, tb, noise=tn)
    np.testing.assert_allclose(float(loss), float(ref_loss), rtol=2e-4, atol=2e-4)
    for native, reference in (
        ("critic_loss", "critic/critic_loss"),
        ("flow_loss", "actor/flow_loss"),
        ("adj_loss", "actor/adj_loss"),
    ):
        np.testing.assert_allclose(
            float(info[native]), float(ref_info[reference]), rtol=2e-4, atol=2e-4
        )
    loss.backward()
    for param, path, transpose in pairs:
        if param.requires_grad:
            np.testing.assert_allclose(
                param.grad.numpy(),
                mapped(grads, path, transpose, param.shape),
                rtol=3e-3,
                atol=3e-4,
            )
    # The reference loss closure uses its old parameters for all detached paths.
    updated, _ = ref.network.apply_loss_fn(lambda p: ref.total_loss(batch, p, rng))
    ref.target_update(updated, "critic")
    ref.target_update(updated, "actor_slow")
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=3e-4
    )
    qam_update(model, optimizer, tb, noise=tn)
    for param, path, transpose in pairs:
        np.testing.assert_allclose(
            param.detach().numpy(),
            mapped(updated.params, path, transpose, param.shape),
            rtol=2e-4,
            atol=2e-5,
        )
