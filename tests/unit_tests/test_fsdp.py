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

"""Tests for the FSDP device mesh, the process groups derived from it, and the
units FSDP2 shards a model into.

Two properties of that mesh are easy to get wrong and silent when they are, so
they are pinned here: the timeout its collectives run under, and which of its
dimensions a gradient norm reduces over.

The timeout. ``init_device_mesh`` creates the default process group itself when
none exists, using whatever watchdog timeout the backend ships with — 30 minutes
for NCCL/Gloo, about 60 for HCCL, in every case below the 180 minutes RLinf
gives its own groups. A mesh dimension that spans the whole world reuses that
group, so every FSDP collective inherits that timeout, and no environment
variable can raise it. ``create_device_mesh`` therefore creates the group first,
with the same ``RLINF_TIMEOUT`` that RLinf applies to its own inter-worker
groups.

The reduction group. FSDP leaves each rank only its slice of every gradient, so
a norm over one of them is not the gradient's norm. ``gradient_reduction_group``
picks the dimension the shards are spread over, which stays ``fsdp`` even once a
replicated ``ddp`` dimension exists beside it.

The units. A model that reads embedding weights directly sets
``_fsdp_wrap_embeddings = False`` so they are gathered with the enclosing unit.
"""

import logging
import os
import socket
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy

from rlinf.config import validate_fp32_master_adamw_config, validate_fsdp_cfg
from rlinf.hybrid_engines.fsdp.optim import FP32MasterAdamW, build_adamw
from rlinf.hybrid_engines.fsdp.utils import apply_fsdp2_to_model, create_device_mesh
from rlinf.scheduler import Worker
from rlinf.scheduler.cluster import Cluster
from rlinf.utils.utils import warmup_optimizer_state

# The timeout a bare init_process_group() installs is backend-specific -- 30
# minutes for NCCL and Gloo, 3636 seconds for HCCL on Ascend -- so no test here
# may hardcode it. What every backend has in common is that the value is not the
# one RLINF_TIMEOUT asked for, and that it is below RLinf's own 180-minute
# default. CONFIGURED_TIMEOUT is an arbitrary value distinguishable from all of
# them.
CONFIGURED_TIMEOUT = timedelta(minutes=97)
RLINF_DEFAULT_TIMEOUT = timedelta(minutes=180)


@pytest.mark.parametrize("is_lora", [False, True])
def test_fp32_master_full_training_config(is_lora):
    cfg = OmegaConf.create(
        {
            "model": {"model_type": "openpi", "is_lora": is_lora},
            "optim": {"use_fp32_master_params": True},
            "fsdp_config": {
                "strategy": "fsdp",
                "sharding_strategy": "no_shard",
                "mixed_precision": {},
            },
        }
    )
    validated = validate_fsdp_cfg(cfg)
    assert validated.optim.use_fp32_master_params
    assert validated.fsdp_config.use_orig_params


@pytest.mark.parametrize(
    "strategy,sharding", [("fsdp2", "no_shard"), ("fsdp", "full_shard")]
)
def test_fp32_master_rejects_unsupported_sharding(strategy, sharding):
    with pytest.raises(ValueError, match="only FSDP1 training"):
        validate_fp32_master_adamw_config(strategy=strategy, sharding_strategy=sharding)


def test_fp32_master_accumulates_small_updates():
    native = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    master = torch.nn.Parameter(native.detach().clone())
    reference = torch.nn.Parameter(native.detach().float())
    opts = [
        build_adamw([{"params": [native], "lr": 5e-6}], eps=1e-8, weight_decay=0.01),
        build_adamw(
            [{"params": [master], "lr": 5e-6}],
            eps=1e-8,
            weight_decay=0.01,
            use_fp32_master_params=True,
        ),
        torch.optim.AdamW([reference], lr=5e-6, foreach=False),
    ]
    for _ in range(1000):
        for param, opt in zip((native, master, reference), opts):
            param.grad = torch.ones_like(param)
            opt.step()
    assert torch.equal(native, torch.ones_like(native))
    assert not torch.equal(master, native)
    torch.testing.assert_close(master, reference.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(
        opts[1].state[master]["fp32_master_param"], reference, rtol=0, atol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_fp32_master_accumulation_clipping_and_resume(dtype, tmp_path):
    param = torch.nn.Parameter(torch.tensor([0.25, -0.5, 1.0], dtype=dtype))
    reference = torch.nn.Parameter(param.detach().float().clone())
    options = {"lr": 5e-4, "betas": (0.8, 0.95), "eps": 1e-6, "weight_decay": 0.1}
    opt = FP32MasterAdamW([param], **options)
    warmup_optimizer_state(opt)
    ref_opt = torch.optim.AdamW([reference], **options, foreach=False)
    resumed = resumed_opt = None
    for step in range(12):
        opt.zero_grad()
        # Accumulate in the model dtype, as backward does, and give the FP32
        # reference the same effective clipped gradient.
        for microbatch in range(3):
            (param * (step + microbatch + 1) / 3).sum().backward()
        torch.nn.utils.clip_grad_norm_([param], max_norm=0.3)
        reference.grad = param.grad.float().clone()
        if resumed is not None:
            resumed.grad = param.grad.clone()
            resumed_opt.step()
        opt.step()
        ref_opt.step()
        torch.testing.assert_close(param, reference.to(dtype), rtol=0, atol=0)
        if resumed is not None:
            torch.testing.assert_close(param, resumed, rtol=0, atol=0)
            for key, state in opt.state[param].items():
                torch.testing.assert_close(
                    state, resumed_opt.state[resumed][key], rtol=0, atol=0
                )
        if step == 5:
            path = tmp_path / "optimizer.pt"
            torch.save({"param": param.detach(), "optimizer": opt.state_dict()}, path)
            saved = torch.load(path, weights_only=True)
            resumed = torch.nn.Parameter(saved["param"].clone())
            resumed_opt = FP32MasterAdamW([resumed], lr=1.0)
            resumed_opt.load_state_dict(saved["optimizer"])
            for key in ("exp_avg", "exp_avg_sq"):
                assert resumed_opt.state[resumed][key].dtype == torch.float32


@pytest.mark.parametrize("checkpoint_format", ["local_shard", "dcp"])
@pytest.mark.parametrize("use_orig_params", [False, True])
def test_fp32_master_fsdp_checkpoint(
    single_rank_env, tmp_path, checkpoint_format, use_orig_params
):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import StateDictOptions
    from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy

    from rlinf.hybrid_engines.fsdp.strategy.checkpoint import Checkpoint
    from rlinf.hybrid_engines.fsdp.utils import FSDPVersion

    dist.init_process_group("gloo")

    def make_training_state():
        model = FullyShardedDataParallel(
            torch.nn.Linear(3, 2, bias=False).to(dtype=torch.bfloat16),
            device_id=torch.device("cpu"),
            use_orig_params=use_orig_params,
            sharding_strategy=ShardingStrategy.NO_SHARD,
        )
        optimizer = FP32MasterAdamW(model.parameters(), lr=5e-6)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)
        checkpoint = Checkpoint(
            model,
            optimizer,
            scheduler,
            StateDictOptions(full_state_dict=False, cpu_offload=True),
            FSDPVersion.FSDP,
            checkpoint_format,
        )
        return model, optimizer, scheduler, checkpoint

    def update(model, optimizer, scheduler):
        optimizer.zero_grad()
        model(torch.ones(2, 3, dtype=torch.bfloat16)).float().sum().backward()
        optimizer.step()
        scheduler.step()

    model, optimizer, scheduler, checkpoint = make_training_state()
    for _ in range(7):
        update(model, optimizer, scheduler)
    path = tmp_path / "checkpoint"
    if checkpoint_format == "dcp":
        dcp.save({"train": checkpoint}, checkpoint_id=path)
    else:
        torch.save(checkpoint.state_dict(), path)

    restored, restored_opt, restored_scheduler, restored_checkpoint = (
        make_training_state()
    )
    if checkpoint_format == "dcp":
        dcp.load({"train": restored_checkpoint}, checkpoint_id=path)
    else:
        restored_checkpoint.load_state_dict(torch.load(path, weights_only=False))

    # A BF16-only restore can produce identical model weights but lose the
    # accumulated sub-ULP update. Check optimizer state before and after stepping.
    for next_step in (False, True):
        if next_step:
            update(model, optimizer, scheduler)
            update(restored, restored_opt, restored_scheduler)
        assert scheduler.state_dict() == restored_scheduler.state_dict()
        for original, loaded in zip(model.parameters(), restored.parameters()):
            torch.testing.assert_close(original, loaded, rtol=0, atol=0)
            for key, value in optimizer.state[original].items():
                torch.testing.assert_close(
                    value, restored_opt.state[loaded][key], rtol=0, atol=0
                )


def test_fp32_master_rejects_native_low_precision_optimizer_state():
    param = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    native = torch.optim.AdamW([param], lr=5e-6)
    param.grad = torch.ones_like(param)
    native.step()
    master = FP32MasterAdamW([param], lr=5e-6)
    with pytest.raises(ValueError, match="without fp32_master_param"):
        master.load_state_dict(native.state_dict())
    assert not master.state


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
