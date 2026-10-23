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

"""Tests for FSDP meshes, wrapping, precision policies, and optimizer dtypes."""

import logging
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch import nn
from torch.distributed.fsdp import (
    FSDPModule,
    MixedPrecision,
    MixedPrecisionPolicy,
    OffloadPolicy,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from rlinf.config import validate_fsdp_cfg
from rlinf.hybrid_engines.fsdp.compat import apply_buffer_dtype_compatibility
from rlinf.hybrid_engines.fsdp.mixed_precision import (
    build_mixed_precision,
    get_mixed_precision_wrap_policy,
)
from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model, create_device_mesh
from rlinf.scheduler import Worker
from rlinf.scheduler.cluster import Cluster

# The timeout a bare init_process_group() installs is backend-specific -- 30
# minutes for NCCL and Gloo, 3636 seconds for HCCL on Ascend -- so no test here
# may hardcode it. What every backend has in common is that the value is not the
# one RLINF_TIMEOUT asked for, and that it is below RLinf's own 180-minute
# default. CONFIGURED_TIMEOUT is an arbitrary value distinguishable from all of
# them.
CONFIGURED_TIMEOUT = timedelta(minutes=97)
RLINF_DEFAULT_TIMEOUT = timedelta(minutes=180)


def free_port() -> str:
    """Reserve an ephemeral port for the rendezvous.

    A fixed port would collide with anything else on a shared CI runner and turn
    every test in this file into an unrelated bind error.

    Returns:
        str: A port that was free a moment ago.
    """
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return str(probe.getsockname()[1])


@pytest.fixture
def single_rank_env(monkeypatch):
    """Give a lone pytest process enough of a rendezvous to build a 1-D mesh.

    Args:
        monkeypatch (pytest.MonkeyPatch): Fixture used to scope the environment
            variables and the device type to this test.

    Yields:
        None: Control returns to the test with the environment in place.
    """
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", free_port())
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    # Worker.torch_device_type is only populated inside a live Worker; the mesh
    # itself does not care which device type it is built over.
    monkeypatch.setattr(Worker, "torch_device_type", "cpu", raising=False)
    try:
        yield
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def group_timeout(group: dist.ProcessGroup) -> timedelta:
    """Read the watchdog timeout a process group was created with.

    Args:
        group (dist.ProcessGroup): The group to inspect.

    Returns:
        timedelta: The timeout carried by the group's backend options. This is
            the same ``_timeout`` attribute ``DeviceMesh`` reads when it forwards
            a timeout to its sub-groups.

    Raises:
        AssertionError: If no backend exposes a timeout. This fails rather than
            skipping, because a skip would leave every assertion in this file
            green on a platform where the timeout cannot be read at all, which
            is exactly when they stop covering anything.
    """
    for device_type in group._device_types:
        options = getattr(group._get_backend(device_type), "options", None)
        timeout = getattr(options, "_timeout", None)
        if timeout is not None:
            return timeout
    raise AssertionError(
        f"no backend of {group} over {list(group._device_types)} exposes a timeout"
    )


def test_mesh_group_uses_the_configured_timeout(single_rank_env, monkeypatch):
    """The mesh's process group carries RLINF_TIMEOUT, not the backend default."""
    monkeypatch.setenv(
        "RLINF_TIMEOUT", str(int(CONFIGURED_TIMEOUT.total_seconds() // 60))
    )

    mesh = create_device_mesh(1)

    assert group_timeout(mesh["fsdp"].get_group()) == CONFIGURED_TIMEOUT


def test_mesh_group_defaults_above_the_torch_watchdog(single_rank_env, monkeypatch):
    """With RLINF_TIMEOUT unset the mesh still gets RLinf's 180-minute default."""
    monkeypatch.delenv("RLINF_TIMEOUT", raising=False)

    mesh = create_device_mesh(1)

    assert group_timeout(mesh["fsdp"].get_group()) == RLINF_DEFAULT_TIMEOUT


def test_existing_process_group_is_left_alone(single_rank_env, monkeypatch, caplog):
    """A default group built by another component keeps its own timeout.

    RLINF_TIMEOUT cannot be applied retroactively, so the only thing left to do
    is say so — otherwise someone who followed the FAQ raises the variable and
    still dies on the backend watchdog with no clue why.
    """
    monkeypatch.setenv(
        "RLINF_TIMEOUT", str(int(CONFIGURED_TIMEOUT.total_seconds() // 60))
    )
    dist.init_process_group(timeout=timedelta(minutes=11))

    with caplog.at_level(logging.WARNING):
        mesh = create_device_mesh(1)

    assert group_timeout(mesh["fsdp"].get_group()) == timedelta(minutes=11)
    assert "RLINF_TIMEOUT" in caplog.text


def test_collective_timeout_matches_the_scheduler_default(monkeypatch):
    """``RLINF_TIMEOUT`` is read with the default the scheduler ships."""
    monkeypatch.delenv("RLINF_TIMEOUT", raising=False)
    assert Cluster.get_collective_timeout() == timedelta(minutes=180)

    monkeypatch.setenv("RLINF_TIMEOUT", "5")
    assert Cluster.get_collective_timeout() == timedelta(minutes=5)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("30m", "integer representing minutes"),
        ("0", "positive number of minutes"),
        ("-5", "positive number of minutes"),
    ],
)
def test_collective_timeout_rejects_unusable_values(monkeypatch, value, message):
    """Bad values fail loudly rather than installing a watchdog nobody wants.

    ``0`` and negatives parse as integers but abort the very first collective,
    so they have to be rejected alongside outright malformed input.
    """
    monkeypatch.setenv("RLINF_TIMEOUT", value)
    with pytest.raises(ValueError, match=message):
        Cluster.get_collective_timeout()


def test_torch_still_installs_the_short_timeout_on_its_own(single_rank_env):
    """Pin the upstream behaviour that makes ``create_device_mesh`` necessary.

    Letting ``init_device_mesh`` build the group leaves it on the backend's own
    watchdog, whatever that happens to be, and ``RLINF_TIMEOUT`` is ignored. The
    assertion is deliberately about what the timeout is *not*: the concrete value
    differs per backend (1800s on NCCL/Gloo, 3636s on Ascend HCCL), so pinning a
    number here would fail on some accelerator without anything being wrong.

    If PyTorch ever starts honouring a longer timeout for the implicitly created
    default group, this test fails and the workaround can be reconsidered.
    """
    assert not dist.is_initialized()
    os.environ["RLINF_TIMEOUT"] = str(int(CONFIGURED_TIMEOUT.total_seconds() // 60))
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", mesh_shape=(1,), mesh_dim_names=["fsdp"])
    finally:
        os.environ.pop("RLINF_TIMEOUT", None)

    group = mesh["fsdp"].get_group()
    assert group is dist.distributed_c10d._get_default_group()
    timeout = group_timeout(group)
    assert timeout != CONFIGURED_TIMEOUT
    assert timeout < RLINF_DEFAULT_TIMEOUT


def test_mesh_dimension_reuses_the_default_group(single_rank_env):
    """The ``fsdp`` dimension is the default group, which is why the fix works.

    ``DeviceMesh`` only hands a mesh dimension its own sub-group when the
    dimension is narrower than the world. For the 1-D mesh RLinf builds, the
    dimension *is* the default group, so setting that group's timeout is enough.
    """
    mesh = create_device_mesh(1)
    assert mesh["fsdp"].get_group() is dist.distributed_c10d._get_default_group()


def test_gradients_reduce_over_the_sharding_dimension(single_rank_env):
    """The norm's process group is the one gradients are sharded over.

    FSDP leaves each rank only its slice of every gradient, so the group has to
    span the sharding dimension. Reducing over nothing reports one rank's shard
    norm rather than the gradient's, and gradient clipping then loosens as the
    job grows; reducing over a replicated dimension counts each shard once per
    replica. Both are silent, so the lookup fails loudly instead of falling back
    when no sharding dimension is present.
    """
    from torch.distributed.device_mesh import init_device_mesh

    from rlinf.hybrid_engines.fsdp.utils import gradient_reduction_group

    sharded = create_device_mesh(1)
    assert gradient_reduction_group(sharded) is sharded["fsdp"].get_group()

    hybrid = init_device_mesh("cpu", (1, 1), mesh_dim_names=["ddp", "fsdp"])
    assert gradient_reduction_group(hybrid) is hybrid["fsdp"].get_group()

    replicated_only = init_device_mesh("cpu", (1,), mesh_dim_names=["ddp"])
    with pytest.raises(KeyError):
        gradient_reduction_group(replicated_only)


class _DomainTable(torch.nn.Module):
    """Reads its embedding weights directly, like cosmos ``DomainAwareLinear``."""

    def __init__(self):
        super().__init__()
        self.table = torch.nn.Embedding(4, 3)

    def forward(self, x):
        return x @ self.table.weight.T


class _Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = _DomainTable()

    def forward(self, x):
        return self.head(x)


class _Policy(torch.nn.Module):
    _no_split_modules = ["_Block"]

    def __init__(self):
        super().__init__()
        self.block = _Block()

    def forward(self, x):
        return self.block(x)


def _shard(policy: torch.nn.Module) -> torch.nn.Module:
    return apply_fsdp2_to_model(
        policy,
        {},
        create_device_mesh(1),
        MixedPrecisionPolicy(),
        OffloadPolicy(),
        reshard_after_forward=True,
    )


def test_embeddings_are_sharded_as_their_own_units_by_default(single_rank_env):
    policy = _shard(_Policy())

    assert isinstance(policy.block.head.table, FSDPModule)


def test_a_model_can_keep_its_embeddings_in_the_enclosing_unit(single_rank_env):
    policy = _Policy()
    policy._fsdp_wrap_embeddings = False
    policy = _shard(policy)

    assert not isinstance(policy.block.head.table, FSDPModule)


@pytest.mark.parametrize("model_type", ["qwen2.5", "openpi"])
@pytest.mark.parametrize("root_cast", [True, False])
@pytest.mark.parametrize("sharding", ["no_shard", "shard_grad_op", "full_shard"])
def test_strategy_passes_wrapper_overrides_and_root_cast_options(
    monkeypatch, root_cast, model_type, sharding
):
    from rlinf.hybrid_engines.fsdp.strategy import fsdp as strategy_module

    fsdp_cfg = config(
        mixed_precision={
            "param_dtype": "bf16",
            "reduce_dtype": "fp32",
            "buffer_dtype": "fp32",
            "cast_root_forward_inputs": root_cast,
        },
        wrap_policy={"module_classes_to_wrap": ["Linear"]},
        sharding_strategy=sharding,
        use_orig_params=True,
    )
    actor = validate_fsdp_cfg(
        OmegaConf.create(
            {
                "fsdp_config": fsdp_cfg,
                "model": {"model_type": model_type, "is_lora": False},
            }
        )
    )
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr(strategy_module, "FSDP", lambda **kwargs: kwargs)
    compatibility_targets = []
    monkeypatch.setattr(
        strategy_module,
        "apply_buffer_dtype_compatibility",
        compatibility_targets.append,
    )
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    kwargs = strategy_module.FSDPStrategy(actor, 1).wrap_model(model, object())
    assert compatibility_targets == [kwargs]
    assert kwargs["sharding_strategy"].name.lower() == sharding
    assert kwargs["mixed_precision"].cast_root_forward_inputs is root_cast
    assert kwargs["mixed_precision"].cast_forward_inputs
    plan = kwargs["auto_wrap_policy"]._run_policy(model, set(), kwargs)
    assert plan[model[0]]["mixed_precision"].param_dtype == torch.bfloat16
    assert plan[model[1]]["mixed_precision"].param_dtype == torch.float32


def config(selector="module_names", targets=None, wrap_mode="individual", **kwargs):
    return OmegaConf.create(
        {
            "mixed_precision_policies": {
                "sensitive": {
                    "param_dtype": "fp32",
                    "reduce_dtype": "fp32",
                    "buffer_dtype": "fp32",
                    "cast_forward_inputs": True,
                }
            },
            "wrappers": {
                "sensitive": {selector: targets or ["1"], "wrap_mode": wrap_mode}
            },
            "mixed_precision_rules": {"sensitive": "sensitive"},
            **kwargs,
        }
    )


@pytest.mark.parametrize("sharding", ["shard_grad_op", "full_shard"])
def test_openpi_sharded_training_requires_fsdp1(sharding):
    actor = OmegaConf.create(
        {
            "model": {"model_type": "openpi"},
            "fsdp_config": {"strategy": "fsdp2", "sharding_strategy": sharding},
        }
    )
    with pytest.raises(AssertionError, match="openpi sharded training requires"):
        validate_fsdp_cfg(actor)


@pytest.mark.parametrize(
    ("selector", "targets"),
    [
        ("module_names", ["1"]),
        ("module_classes", ["torch.nn.modules.linear.Linear"]),
        ("wrap_names", ["sensitive"]),
    ],
)
def test_selectors_create_wrapper_overrides(selector, targets):
    model = nn.Sequential(nn.ReLU(), nn.Linear(2, 2))
    model[1]._fsdp_wrap_name = "sensitive"
    default = MixedPrecision(param_dtype=torch.bfloat16)
    policy = get_mixed_precision_wrap_policy(
        model, config(selector, targets), None, default
    )
    resolved = policy._run_policy(model, set(), {"mixed_precision": default})
    assert set(resolved) == {model[1]}
    assert resolved[model[1]]["mixed_precision"].param_dtype == torch.float32
    assert resolved[model[1]]["mixed_precision"].cast_forward_inputs


def test_nested_wrappers_do_not_inherit_parent_override():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    default = MixedPrecision(param_dtype=torch.bfloat16)

    def base_policy(module, recurse, nonwrapped_numel):
        return lambda_auto_wrap_policy(
            module, recurse, nonwrapped_numel, lambda m: isinstance(m, nn.Linear)
        )

    policy = get_mixed_precision_wrap_policy(
        model, config(targets=["0"]), base_policy, default
    )
    resolved = policy._run_policy(model, set(), {"mixed_precision": default})
    assert resolved[model[0]]["mixed_precision"].param_dtype == torch.float32
    assert resolved[model[0][0]]["mixed_precision"] is default


def test_no_rules_preserves_original_policy():
    sentinel = object()
    assert (
        get_mixed_precision_wrap_policy(
            nn.Linear(2, 2), OmegaConf.create({}), sentinel, MixedPrecision()
        )
        is sentinel
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"wrappers": {"sensitive": {"module_names": ["absent"]}}},
            "matched no",
        ),
        (
            {"mixed_precision_rules": {"sensitive": "typo"}},
            "unknown policy",
        ),
        ({"strategy": "fsdp2"}, "FSDP1"),
        ({"disable": True}, "disable=true"),
        ({"amp_autocast": {"enabled": True}}, "amp_autocast"),
    ],
)
def test_invalid_rules_fail_before_wrapping(changes, message):
    with pytest.raises(ValueError, match=message):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.ReLU(), nn.Linear(2, 2)),
            config(**changes),
            None,
            MixedPrecision(),
        )


def test_overlapping_rules_are_rejected():
    cfg = config()
    cfg.wrappers.overlap = {"module_names": ["*"]}
    with pytest.raises(ValueError, match="Multiple"):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.ReLU(), nn.Linear(2, 2)), cfg, None, MixedPrecision()
        )


def test_batchnorm_override_cannot_silently_lose_to_pytorch():
    with pytest.raises(ValueError, match="BatchNorm"):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.ReLU(), nn.BatchNorm1d(2)),
            config(),
            None,
            MixedPrecision(),
        )


def test_tied_parameters_cannot_cross_wrapper_boundaries():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    model[1].weight = model[0].weight
    with pytest.raises(ValueError, match="Shared parameter"):
        get_mixed_precision_wrap_policy(model, config(), None, MixedPrecision())


def test_explicit_null_keeps_original_dtype_and_torch_defaults():
    mp = build_mixed_precision(OmegaConf.create({"param_dtype": None}))
    assert mp.param_dtype is None
    assert mp.cast_forward_inputs


def test_unknown_runtime_options_are_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        build_mixed_precision(OmegaConf.create({"output_dtype": "fp32"}))


def test_yaml_validation_rejects_fsdp2_overrides():
    cfg = config(strategy="fsdp2", mixed_precision={})
    with pytest.raises(ValueError, match="FSDP1"):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"mixed_precision_policies": {"sensitive": {"param_dtype": "fp32"}}},
            "missing dtype",
        ),
        (
            {"wrappers": {"sensitive": {"module_names": "1"}}},
            "string list",
        ),
        (
            {
                "wrappers": {
                    "sensitive": {"module_names": ["1"], "wrap_names": ["sensitive"]}
                }
            },
            "exactly one",
        ),
        (
            {
                "mixed_precision_policies": {
                    "sensitive": {
                        "param_dtype": "fp32",
                        "reduce_dtype": "fp32",
                        "buffer_dtype": "fp32",
                        "output_dtype": "fp32",
                    }
                }
            },
            "Unsupported",
        ),
    ],
)
def test_invalid_policy_schema_fails_during_yaml_validation(changes, message):
    cfg = config(mixed_precision={}, **changes)
    with pytest.raises(ValueError, match=message):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


@pytest.mark.parametrize("full_precision_eval", [False, True])
def test_actual_fsdp_wrappers_forward_backward_and_buffers(
    tmp_path, monkeypatch, full_precision_eval
):
    """Exercise native FSDP1 hooks rather than only inspecting policy kwargs."""
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if dist.is_initialized():
        pytest.skip("Test requires its own single-rank process group")
    monkeypatch.setenv("FSDP_USE_FULL_PREC_IN_EVAL", str(int(full_precision_eval)))
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )

    class LinearWithBuffer(nn.Linear):
        def __init__(self):
            super().__init__(2, 2)
            self.register_buffer("offset", torch.ones(2))
            self.seen_dtype = None

        def forward(self, x):
            self.seen_dtype = (x.dtype, self.weight.dtype, self.offset.dtype)
            return super().forward(x) + self.offset

    class RootWithBuffer(nn.Sequential):
        def __init__(self):
            super().__init__(LinearWithBuffer(), LinearWithBuffer())
            self.register_buffer("offset", torch.ones(2))

        def forward(self, x):
            return super().forward(x) + self.offset

    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        default = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
        cfg = config()

        def base(module, recurse, nonwrapped_numel):
            return recurse or isinstance(module, LinearWithBuffer)

        def wrap_model():
            model = RootWithBuffer().to(device)
            policy = get_mixed_precision_wrap_policy(model, cfg, base, default)
            wrapped = FSDP(
                model,
                auto_wrap_policy=policy,
                mixed_precision=default,
                device_id=device,
            )
            apply_buffer_dtype_compatibility(wrapped)
            return wrapped

        wrapped = wrap_model()
        assert wrapped._handle is None
        inputs = torch.randn(2, 2, device=device)
        for training in (True, False, True):
            wrapped.train(training)
            wrapped.zero_grad(set_to_none=True)
            wrapped(inputs).sum().backward()
            dtype = (
                torch.float32
                if not training and full_precision_eval
                else torch.bfloat16
            )
            assert wrapped[0].module.seen_dtype == (dtype,) * 3
            assert wrapped[1].module.seen_dtype == (torch.float32,) * 3
            assert wrapped.module.offset.dtype == dtype
            assert all(p.grad.dtype == torch.float32 for p in wrapped.parameters())

        with FSDP.summon_full_params(wrapped):
            assert wrapped[0].module.weight.dtype == torch.float32
            assert wrapped[0].module.offset.dtype == torch.bfloat16

        expected = wrapped(inputs).detach()
        state = wrapped.state_dict()
        assert state["0.offset"].dtype == torch.float32
        assert state["1.offset"].dtype == torch.float32
        assert state["offset"].dtype == torch.float32
        restored = wrap_model()
        restored.load_state_dict(state)
        assert torch.equal(restored(inputs), expected)
        assert restored[0].module.seen_dtype == (torch.bfloat16,) * 3
        assert restored[1].module.seen_dtype == (torch.float32,) * 3
    finally:
        dist.destroy_process_group()


def linear_policy(module, recurse, nonwrapped_numel):
    return recurse or isinstance(module, nn.Linear)


@pytest.mark.parametrize("targets", [["0.0", "0.1"], ["0.*"], ["0"]])
def test_group_creates_one_boundary_and_suppresses_inner_wrappers(targets):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)), nn.Linear(2, 2)
    )
    default = MixedPrecision(param_dtype=torch.bfloat16)
    policy = get_mixed_precision_wrap_policy(
        model, config(targets=targets, wrap_mode="combined"), linear_policy, default
    )
    plan = policy._run_policy(model, set(), {"mixed_precision": default})
    assert set(plan) == {model[0], model[1]}
    assert plan[model[0]]["mixed_precision"].param_dtype == torch.float32
    assert plan[model[1]]["mixed_precision"] is default


def test_selector_without_group_keeps_separate_wrappers():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)))
    policy = get_mixed_precision_wrap_policy(
        model, config(targets=["0.*"]), None, MixedPrecision()
    )
    assert set(policy._run_policy(model, set(), {})) == {model[0][0], model[0][1]}


@pytest.mark.parametrize(
    "kind, message",
    [
        ("coverage", "unselected states"),
        ("dtype", "uniform original"),
        ("grad", "requires_grad"),
        ("nested", "inner wrappers"),
        ("root", "model root"),
    ],
)
def test_invalid_groups_fail_before_wrapping(kind, message):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)),
        nn.Linear(2, 2),
    )
    cfg = config(targets=["0.*"], wrap_mode="combined")
    if kind == "coverage":
        cfg.wrappers.sensitive.module_names = ["0.0", "0.1"]
    elif kind == "dtype":
        model[0][1].bfloat16()
    elif kind == "grad":
        model[0][1].requires_grad_(False)
    elif kind == "nested":
        cfg.wrappers.inner = {"module_names": ["0.0"]}
    else:
        cfg.wrappers.sensitive.module_names = ["0", "1"]
    with pytest.raises(ValueError, match=message):
        get_mixed_precision_wrap_policy(model, cfg, linear_policy, MixedPrecision())


def test_group_allows_mixed_requires_grad_with_orig_params():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)))
    model[0][0].requires_grad_(False)
    policy = get_mixed_precision_wrap_policy(
        model,
        config(targets=["0.*"], wrap_mode="combined", use_orig_params=True),
        linear_policy,
        MixedPrecision(),
    )
    assert set(policy._run_policy(model, set(), {})) == {model[0]}


def test_unbound_wrapper_uses_default_precision():
    model = nn.Sequential(nn.Linear(2, 2))
    default = MixedPrecision(param_dtype=torch.bfloat16)
    cfg = config(targets=["0"], mixed_precision_rules={})
    policy = get_mixed_precision_wrap_policy(model, cfg, None, default)
    assert (
        policy._run_policy(model, set(), {"mixed_precision": default})[model[0]][
            "mixed_precision"
        ]
        is default
    )


def test_every_selector_must_match():
    with pytest.raises(ValueError, match="matched no"):
        get_mixed_precision_wrap_policy(
            nn.Sequential(nn.Linear(2, 2)),
            config(targets=["0", "typo.*"]),
            None,
            MixedPrecision(),
        )


@pytest.mark.parametrize("wrap_mode", ["combined", "subtree"])
def test_actual_group_owns_multiple_modules(tmp_path, wrap_mode):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if dist.is_initialized():
        pytest.skip("Requires its own process group")
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'group-init'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
            nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
            nn.Linear(2, 2),
        ).to(device)
        seen = []
        for child in list(model[0]) + list(model[1]):
            child.register_forward_pre_hook(
                lambda module, args: seen.append((args[0].dtype, module.weight.dtype))
            )
        default = MixedPrecision(param_dtype=torch.bfloat16, cast_forward_inputs=True)
        policy = get_mixed_precision_wrap_policy(
            model,
            config(
                targets=["0.*"] if wrap_mode == "combined" else ["[01]"],
                wrap_mode=wrap_mode,
            ),
            linear_policy,
            default,
        )
        wrapped = FSDP(
            model, auto_wrap_policy=policy, mixed_precision=default, device_id=device
        )
        assert isinstance(wrapped[0], FSDP)
        assert all(isinstance(child, nn.Linear) for child in wrapped[0].module)
        assert wrapped[0]._flat_param.numel() == 12
        if wrap_mode == "subtree":
            assert isinstance(wrapped[1], FSDP)
            assert all(isinstance(child, nn.Linear) for child in wrapped[1].module)
            assert wrapped[1]._flat_param.numel() == 12
        output = wrapped(torch.randn(2, 2, device=device))
        output.float().square().sum().backward()
        second_dtype = torch.float32 if wrap_mode == "subtree" else torch.bfloat16
        assert (
            seen
            == [(torch.float32, torch.float32)] * 2 + [(second_dtype, second_dtype)] * 2
        )
        assert output.dtype == torch.bfloat16
        assert all(p.grad is not None for p in wrapped.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("container", [nn.ModuleList, nn.ModuleDict])
def test_group_requires_executable_parent(container):
    children = [nn.Linear(2, 2), nn.Linear(2, 2)]
    block = container(
        children if container is nn.ModuleList else dict(zip(["a", "b"], children))
    )
    model = nn.Sequential(block)
    with pytest.raises(ValueError, match="forward execution boundary"):
        get_mixed_precision_wrap_policy(
            model, config(targets=["0.*"], wrap_mode="combined"), None, MixedPrecision()
        )


def test_unknown_wrapper_binding_fails_validation():
    with pytest.raises(ValueError, match="Unknown wrapper"):
        validate_fsdp_cfg(
            OmegaConf.create(
                {
                    "fsdp_config": config(
                        mixed_precision={}, mixed_precision_rules={"typo": "sensitive"}
                    )
                }
            )
        )


@pytest.mark.parametrize("select_parent", [False, True])
def test_pruned_policy_cannot_hide_explicit_child_wrapper(select_parent):
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    cfg = config(targets=["0.0"])
    if select_parent:
        cfg.wrappers.parent = {"module_names": ["0"]}

    def base(module, recurse, nonwrapped_numel):
        return module is model

    with pytest.raises(ValueError, match="non-recursing policy"):
        get_mixed_precision_wrap_policy(model, cfg, base, MixedPrecision())


@pytest.mark.parametrize(
    "field", ["wrappers", "mixed_precision_policies", "mixed_precision_rules"]
)
def test_empty_invalid_config_mapping_is_rejected(field):
    cfg = config(mixed_precision={}, **{field: []})
    with pytest.raises(ValueError, match="must be a mapping"):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


@pytest.mark.parametrize("wrap_mode", ["individual", "subtree"])
@pytest.mark.parametrize("wrap_child", [False, True])
def test_wrapper_dtype_validation_excludes_separately_wrapped_children(
    wrap_child, wrap_mode
):
    class MixedDtypeModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.offset = nn.Parameter(torch.ones(2))
            self.child = nn.Linear(2, 2).bfloat16()

        def forward(self, x):
            return self.child(x) + self.offset

    model = nn.Sequential(MixedDtypeModule())
    cfg = config(targets=["0"], wrap_mode=wrap_mode)
    base = linear_policy if wrap_child else None
    if not wrap_child or wrap_mode == "subtree":
        with pytest.raises(ValueError, match="uniform original parameter dtype"):
            get_mixed_precision_wrap_policy(model, cfg, base, MixedPrecision())
    else:
        policy = get_mixed_precision_wrap_policy(model, cfg, base, MixedPrecision())
        plan = policy._run_policy(model, set(), {})
        assert set(plan) == {model[0], model[0].child}


@pytest.mark.parametrize(
    "selector, targets",
    [
        ("module_names", ["[01]"]),
        ("module_classes", ["torch.nn.modules.container.Sequential"]),
        ("wrap_names", ["mlp"]),
    ],
)
def test_subtree_creates_one_complete_wrapper_per_match(selector, targets):
    model = nn.Sequential(
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
        nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)),
        nn.Linear(2, 2),
    )
    model[0]._fsdp_wrap_name = model[1]._fsdp_wrap_name = "mlp"
    default = MixedPrecision(param_dtype=torch.bfloat16)
    cfg = config(selector, targets, wrap_mode="subtree")
    policy = get_mixed_precision_wrap_policy(model, cfg, linear_policy, default)
    plan = policy._run_policy(model, set(), {"mixed_precision": default})
    assert set(plan) == {model[0], model[1], model[2]}
    assert plan[model[0]]["mixed_precision"].param_dtype == torch.float32
    assert plan[model[1]]["mixed_precision"].param_dtype == torch.float32
    assert plan[model[2]]["mixed_precision"] is default


@pytest.mark.parametrize("mode", [True, False, None, "each"])
def test_invalid_wrap_modes_fail_config_validation(mode):
    with pytest.raises(ValueError, match="wrap_mode must be"):
        validate_fsdp_cfg(
            OmegaConf.create(
                {"fsdp_config": config(wrap_mode=mode, mixed_precision={})}
            )
        )


def test_old_group_option_is_rejected():
    cfg = config(mixed_precision={})
    cfg.wrappers.sensitive.group = True
    with pytest.raises(ValueError, match="unsupported options.*group"):
        validate_fsdp_cfg(OmegaConf.create({"fsdp_config": cfg}))


def test_subtree_cannot_select_nested_boundaries():
    model = nn.Sequential(nn.Sequential(nn.Linear(2, 2)))
    with pytest.raises(ValueError, match="inner wrappers"):
        get_mixed_precision_wrap_policy(
            model,
            config(targets=["0", "0.0"], wrap_mode="subtree"),
            linear_policy,
            MixedPrecision(),
        )


@pytest.mark.parametrize("root_keep,child_keep", [(True, False), (False, True)])
def test_adamw_rejects_incompatible_root_or_child_policy(
    tmp_path, root_keep, child_keep
):
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from rlinf.hybrid_engines.fsdp.optim import validate_adam_mixed_precision

    if dist.is_initialized():
        pytest.skip("Test requires its own single-rank process group")
    device = (
        torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    )
    dist.init_process_group(
        "gloo", init_method=f"file://{tmp_path / 'init'}", rank=0, world_size=1
    )
    try:
        model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2)).to(device)

        def policy(keep):
            return MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16 if keep else torch.float32,
                keep_low_precision_grads=keep,
                cast_forward_inputs=True,
            )

        model[1] = FSDP(
            model[1],
            device_id=device,
            use_orig_params=True,
            mixed_precision=policy(child_keep),
        )
        wrapped = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            mixed_precision=policy(root_keep),
        )
        with pytest.raises(ValueError, match="actor.optim.use_fp32_master_params"):
            validate_adam_mixed_precision(wrapped, use_fp32_master_params=False)
    finally:
        dist.destroy_process_group()


def _check_sharded_retained_gradients(rank, init_file):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    from rlinf.hybrid_engines.fsdp.optim import validate_adam_mixed_precision

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        model = FSDP(
            nn.Linear(4, 4).cuda(rank),
            device_id=rank,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                keep_low_precision_grads=True,
            ),
        )
        with pytest.raises(ValueError, match="prepares torch.bfloat16 gradients"):
            validate_adam_mixed_precision(model)
        # Exercise native hooks to verify the dtype this check predicts.
        model(torch.ones(2, 4, device=rank)).float().sum().backward()
        assert {p.grad.dtype for p in model.parameters() if p.grad is not None} == {
            torch.bfloat16
        }
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA GPUs")
def test_sharded_retained_gradient_dtype(tmp_path):
    torch.multiprocessing.spawn(
        _check_sharded_retained_gradients,
        args=(str(tmp_path / "init"),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("embed_precision", [None, "bf16"])
def test_openpi_factory_separates_parameter_and_activation_dtypes(
    tmp_path, monkeypatch, embed_precision
):
    from rlinf.models import get_model
    from rlinf.models.embodiment.openpi.modules import gemma, siglip, utils
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(utils, "_torch_compile_enabled", False)
    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda variant: {
            "width": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_dim": 64,
            "patch_size": (14, 14),
        },
    )
    source = Pi0(
        Pi0Config(
            pi05=True,
            paligemma_variant="dummy",
            action_expert_variant="dummy",
            action_horizon=2,
            max_token_len=4,
        )
    )
    torch.save(source.state_dict(), tmp_path / "full_weights.pt")
    cfg = OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": str(tmp_path),
            "precision": "fp32",
            "is_lora": False,
            "load_to_device": False,
            "action_dim": 32,
            "num_action_chunks": 2,
            "num_steps": 1,
            "openpi": {
                "task": "sft",
                "torch_compile": False,
                "model_action_dim": 32,
                "action_horizon": 2,
                "paligemma_variant": "dummy",
                "action_expert_variant": "dummy",
                "max_token_len": 4,
            },
        }
    )
    if embed_precision is not None:
        cfg.openpi.embed_dtype = embed_precision

    model = get_model(cfg)

    expected = torch.bfloat16 if embed_precision == "bf16" else torch.float32
    assert model.embed_dtype == model.llm.embed_dtype == model.img.dtype_mm == expected
    assert {param.dtype for param in model.parameters()} == {torch.float32}
    torch.testing.assert_close(model.img.stem.weight, source.img.stem.weight)


def test_openpi_sft_uses_activation_checkpointing(monkeypatch):
    from rlinf.models.embodiment.base_policy import ForwardType
    from rlinf.models.embodiment.openpi.modules import gemma, siglip, utils
    from rlinf.models.embodiment.openpi.pi0 import Pi0
    from rlinf.models.embodiment.openpi.pi0_config import Pi0Config

    monkeypatch.setattr(gemma, "PALIGEMMA_VOCAB_SIZE", 32)
    monkeypatch.setattr(utils, "_torch_compile_enabled", False)
    monkeypatch.setattr(
        siglip,
        "_decode_variant",
        lambda variant: {
            "width": 32,
            "depth": 1,
            "num_heads": 4,
            "mlp_dim": 64,
            "patch_size": (14, 14),
        },
    )
    cfg = Pi0Config(
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        action_horizon=2,
        max_token_len=4,
    )
    model = Pi0(cfg).train()
    model.gradient_checkpointing_enable({"use_reentrant": False})
    observation = cfg.fake_obs()
    checkpoint = torch.utils.checkpoint.checkpoint
    calls = []

    def record_checkpoint(function, *args, **kwargs):
        calls.append(function)
        return checkpoint(function, *args, **kwargs)

    monkeypatch.setattr(torch.utils.checkpoint, "checkpoint", record_checkpoint)
    loss = model(forward_type=ForwardType.SFT, data=(observation, cfg.fake_act()))
    loss.backward()

    assert calls
    assert model.img.stem.weight.grad is not None
    assert model.llm.layers[0].attn.q_proj[0].weight.grad is not None
    assert model.action_out_proj.weight.grad is not None
