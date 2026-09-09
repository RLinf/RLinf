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

"""One holder at a time for a hardware endpoint, across processes on a node.

Parts are placed independently, so an arm and the end effector mounted on it
may open their connections from different processes on the same machine. That
is fine when they address different endpoints, which is the normal case: a
Franka arm speaks to one libfranka port and its hand to another. It goes wrong
when two parts address the *same* endpoint, and the failure that follows tends
to name a socket rather than the mistake.

A claim makes the second one fail immediately, saying who holds the endpoint.
Claims are advisory and cover only parts that take one: they are a diagnostic,
not a permission system.
"""

import os
from typing import Any, Optional

from filelock import FileLock, Timeout

from rlinf.utils.logging import get_logger

#: Where claim files live. One file per endpoint, holding the owner's identity.
_CLAIM_DIR = "/tmp/rlinf-device-claims"


def _claim_path(key: str) -> str:
    """Return the lock file backing one endpoint key."""
    safe = "".join(c if c.isalnum() or c in ".-_" else "_" for c in key)
    return os.path.join(_CLAIM_DIR, f"{safe}.lock")


class DeviceClaim:
    """An exclusive advisory hold on one hardware endpoint.

    Args:
        key: Identity of the endpoint, such as ``"franky-gripper:172.16.0.2"``.
            Two parts that can safely talk to the same box at once must use
            different keys, and two that cannot must use the same one.
        held_by: What to name as the holder when someone else is refused.
    """

    def __init__(self, key: str, held_by: str) -> None:
        self.key = key
        self.held_by = held_by
        self._lock: Optional[FileLock] = None

    def acquire(self) -> None:
        """Take the claim, or say who already holds it.

        Raises:
            RuntimeError: If another part on this machine holds the endpoint.
        """
        os.makedirs(_CLAIM_DIR, exist_ok=True)
        path = _claim_path(self.key)
        lock = FileLock(path)
        try:
            lock.acquire(timeout=0)
        except Timeout:
            raise RuntimeError(
                f"{self.held_by} cannot open {self.key}: another part on this "
                f"machine already holds it ({_holder_of(path)}). One endpoint "
                "serves one part at a time. Place them on the same part, or "
                "give whichever of them is meant to be separate its own "
                "endpoint."
            ) from None
        self._lock = lock
        try:
            with open(path, "w") as claim_file:
                claim_file.write(f"{self.held_by} pid={os.getpid()}\n")
        except OSError:
            # The hold is the lock itself; the note in the file is for humans.
            get_logger().debug("could not record the holder of %s", self.key)

    def release(self) -> None:
        """Drop the claim. Releasing one that was never taken does nothing."""
        if self._lock is None:
            return
        try:
            self._lock.release()
        finally:
            self._lock = None

    def __enter__(self) -> "DeviceClaim":
        """Take the claim for the duration of a block."""
        self.acquire()
        return self

    def __exit__(self, *_exc: Any) -> None:
        """Drop the claim on leaving the block."""
        self.release()


def _holder_of(path: str) -> str:
    """Return the recorded holder of a claim, for the refusal message."""
    try:
        with open(path) as claim_file:
            return claim_file.read().strip() or "holder unknown"
    except OSError:
        return "holder unknown"
