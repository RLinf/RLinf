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

"""Tests for the collective timeout of the FSDP device mesh.

``init_device_mesh`` creates the default process group itself when none exists,
using whatever watchdog timeout the backend ships with — 30 minutes for
NCCL/Gloo, about 60 for HCCL, in every case below the 180 minutes RLinf gives
its own groups. A mesh dimension that spans the whole world reuses that group,
so every FSDP collective inherits that timeout, and no environment variable can
raise it. ``create_device_mesh`` therefore creates the group first, with the
same ``RLINF_TIMEOUT`` that RLinf applies to its own inter-worker groups.
"""

import logging
import os
import socket
from datetime import timedelta

import pytest
import torch.distributed as dist

from rlinf.hybrid_engines.fsdp.utils import create_device_mesh
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
