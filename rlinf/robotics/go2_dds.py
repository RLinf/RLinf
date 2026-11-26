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

"""Share SDK2's process-wide Go2 channel factory on one network interface."""

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

_lock = threading.Lock()
_interface: str | None = None
_active_channels: set[Literal["state", "video"]] = set()


@contextmanager
def go2_dds_session(
    interface: str, *, channel: Literal["state", "video"] = "state"
) -> Iterator[None]:
    """Reserve a state or video session on the process-wide DDS interface."""
    from unitree_sdk2py.core.channel import ChannelFactory, Qos

    global _interface
    with _lock:
        if channel in _active_channels:
            raise RuntimeError(
                f"Only one Go2 {channel} connection can have exclusive DDS ownership."
            )
        if _interface is None:
            factory = ChannelFactory()
            default_qos = Qos()
            if not factory.Init(0, interface, default_qos):
                raise RuntimeError(f"Go2 SDK2 failed to initialize DDS on {interface}.")
            # A successful SDK2 Init may reuse a different interface.
            # Only the caller that first bound the factory installs its topic QoS.
            if getattr(factory, "_ChannelFactory__qos", None) is not default_qos:
                raise RuntimeError(
                    "Cannot verify the SDK2 DDS interface. Start a fresh process "
                    "and connect the Go2 adapter before other SDK2 clients."
                )
            _interface = interface
        elif interface != _interface:
            raise RuntimeError(
                f"DDS interface is bound to {_interface}, requested {interface}."
            )
        _active_channels.add(channel)
    try:
        yield
    finally:
        with _lock:
            _active_channels.remove(channel)
