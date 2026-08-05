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

"""Tests for the host staging buffers of the no-accelerator-CCL GLOO fallback.

When ``MultiChannelProcessGroup`` cannot form an accelerator collective (mixed
GPU models in one group, or an accelerator whose CCL is unsupported), it falls
back to GLOO and stages accelerator tensors through host memory. GLOO puts a
tensor on the wire by reading its storage linearly, so both staging buffers
must be contiguous regardless of the user tensor's layout.
"""

import pytest
import torch

from rlinf.scheduler import Worker
from rlinf.scheduler.collective.collective_group import CollectiveGroup
from rlinf.scheduler.collective.multi_channel_pg import MultiChannelProcessGroup


def accelerator_is_available():
    """Return whether the Worker accelerator backend is available."""
    return (
        Worker.torch_platform is not None
        and hasattr(Worker.torch_platform, "is_available")
        and Worker.torch_platform.is_available()
    )


pytestmark = pytest.mark.skipif(
    not accelerator_is_available(),
    reason="GLOO host staging only runs for accelerator tensors",
)

ACCEL_DEVICE = Worker.torch_device_type


def make_tensors(layout: str):
    """Build a (source, destination) accelerator tensor pair with a given layout.

    Args:
        layout (str): ``contiguous`` or ``transposed``. A transposed tensor is
            dense but not contiguous, which is the layout that ``empty_like``
            staging used to scramble.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The filled source tensor and a
            zeroed destination tensor with the same shape, dtype and strides.

    """
    src = torch.arange(24, dtype=torch.float32, device=ACCEL_DEVICE).reshape(4, 6)
    dst = torch.zeros(4, 6, dtype=torch.float32, device=ACCEL_DEVICE)
    if layout == "transposed":
        src, dst = src.t(), dst.t()
        assert not src.is_contiguous()
    return src, dst


def fake_gloo_transfer(staged: torch.Tensor, recv_buffer: torch.Tensor):
    """Copy ``staged`` into ``recv_buffer`` the way GLOO moves a tensor.

    GLOO hands the backend a raw pointer plus an element count, so it reads and
    writes storage linearly and ignores strides entirely. Aliasing the storage
    reproduces that faithfully: a staging buffer that kept the user tensor's
    strides gets filled in the wrong order rather than raising.
    """
    flat_recv = torch.empty(0, dtype=recv_buffer.dtype).set_(
        recv_buffer.untyped_storage(), 0, (recv_buffer.numel(),), (1,)
    )
    flat_recv.copy_(staged.reshape(-1))


@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
def test_staged_send_buffer_is_pinned_and_contiguous(layout):
    """The send-side staging buffer is pinned, contiguous and in wire order."""
    src, _ = make_tensors(layout)

    staged = MultiChannelProcessGroup._stage_to_pinned_cpu(src)

    assert staged.is_pinned()
    assert staged.is_contiguous()
    assert torch.equal(staged, src.cpu())


def new_recv_buffer(tensor: torch.Tensor) -> torch.Tensor:
    """Allocate the recv staging buffer the way ``recv`` and ``broadcast`` do.

    Mirrors the inline ``torch.empty(..., pin_memory=True)`` at those two call
    sites. They must not go back to ``torch.empty_like``, which preserves the
    destination's strides; ``test_empty_like_recv_buffer_scrambles`` pins down
    what that would cost.
    """
    return torch.empty(tensor.shape, dtype=tensor.dtype, pin_memory=True)


def test_empty_like_recv_buffer_scrambles():
    """Show why the recv sites allocate with ``empty`` rather than ``empty_like``.

    Whether ``empty_like`` preserves the source strides is backend dependent:
    it does on CUDA and does not on Ascend, which is itself why the call sites
    must not depend on it. Skip where the buffer already comes back contiguous,
    since there is then no corruption to demonstrate.
    """
    src, dst = make_tensors("transposed")
    bad_buffer = torch.empty_like(dst, device="cpu")
    if bad_buffer.is_contiguous():
        pytest.skip("empty_like already yields a contiguous host buffer here")

    pg = object.__new__(MultiChannelProcessGroup)
    pg._no_accel_ccl = True
    staged = MultiChannelProcessGroup._stage_to_pinned_cpu(src)
    fake_gloo_transfer(staged, bad_buffer)
    pg._copy_to_accel_tensor(CollectiveGroup.ACCEL, dst, bad_buffer)
    Worker.torch_platform.synchronize()

    assert not torch.equal(dst, src)


@pytest.mark.parametrize("layout", ["contiguous", "transposed"])
def test_host_staging_round_trip_preserves_values(layout):
    """A tensor survives stage -> wire -> unstage with its layout restored."""
    src, dst = make_tensors(layout)
    # Only _no_accel_ccl is read by _copy_to_accel_tensor; building a real
    # group would need a live two-rank rendezvous.
    pg = object.__new__(MultiChannelProcessGroup)
    pg._no_accel_ccl = True

    staged = MultiChannelProcessGroup._stage_to_pinned_cpu(src)
    recv_buffer = new_recv_buffer(dst)
    assert recv_buffer.is_pinned() and recv_buffer.is_contiguous()
    fake_gloo_transfer(staged, recv_buffer)
    pg._copy_to_accel_tensor(CollectiveGroup.ACCEL, dst, recv_buffer)
    Worker.torch_platform.synchronize()

    assert torch.equal(dst, src)
    assert dst.stride() == src.stride()


def test_stage_to_pinned_cpu_passes_through_host_tensors():
    """A host tensor is not pinned again, but is still made contiguous."""
    host_tensor = torch.arange(24, dtype=torch.float32).reshape(4, 6).t()

    staged = MultiChannelProcessGroup._stage_to_pinned_cpu(host_tensor)

    assert staged.device.type == "cpu"
    assert staged.is_contiguous()
    assert torch.equal(staged, host_tensor)
