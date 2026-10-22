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

"""Lower the camera near plane of a RoboTwin checkout for SAPIEN 3.0.1.

RoboTwin creates its cameras with a 0.1 m near plane. SAPIEN 3.0.1, which
RLinf installs, clips everything closer than that, so the gripper fingers
disappear from the wrist cameras that the released checkpoints were trained
with. The patch reads the near plane from ``ROBOTWIN_CAMERA_NEAR`` with a
default of 0.01 m; set it to ``0.1`` to restore RoboTwin's value.

The script is idempotent and stops without writing if RoboTwin no longer
contains the code it expects.

Usage:
    python -m rlinf.envs.sim.robotwin.patch_camera_near /path/to/RoboTwin
"""

import os
import sys

CAMERA_FILE = os.path.join("envs", "camera", "camera.py")
_ORIGINAL = "        near, far = 0.1, 100\n"
_PATCHED = (
    '        near, far = float(os.environ.get("ROBOTWIN_CAMERA_NEAR", "0.01")), 100\n'
)


def patch_robotwin(robotwin_path: str) -> bool:
    """Make the camera near plane of a RoboTwin checkout configurable.

    Args:
        robotwin_path: Root of the RoboTwin checkout (``RLinf_support`` branch).

    Returns:
        True if the file was changed, False if it was already patched.

    Raises:
        RuntimeError: If ``envs/camera/camera.py`` does not contain the original
            line exactly once, or does not import ``os``.
    """
    path = os.path.join(robotwin_path, CAMERA_FILE)
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    if _PATCHED in source:
        return False
    if source.count(_ORIGINAL) != 1 or "\nimport os\n" not in f"\n{source}":
        raise RuntimeError(
            f"Expected one '{_ORIGINAL.strip()}' line and 'import os' in {path}. "
            "RoboTwin has changed; review patch_camera_near.py first."
        )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(source.replace(_ORIGINAL, _PATCHED))
    return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: python -m rlinf.envs.sim.robotwin.patch_camera_near /path/to/RoboTwin"
        )
    if patch_robotwin(sys.argv[1]):
        print(f"Patched {CAMERA_FILE}: near plane now from ROBOTWIN_CAMERA_NEAR.")
    else:
        print("RoboTwin camera near plane is already patched.")
