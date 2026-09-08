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

import os
import re


def parse_global_step_from_checkpoint_path(
    checkpoint_path: str | os.PathLike[str],
) -> int:
    """Extract the global step from a checkpoint directory path.

    Args:
        checkpoint_path: Path ending in a ``global_step_<step>`` directory.

    Returns:
        The non-negative global step encoded in the directory name.

    Raises:
        ValueError: If the final directory is not named ``global_step_<step>``.
    """
    checkpoint_dir = os.path.basename(os.path.normpath(checkpoint_path))
    match = re.fullmatch(r"global_step_(\d+)", checkpoint_dir)
    if match is None:
        raise ValueError(
            "Checkpoint path must end with a 'global_step_<step>' directory, "
            f"but got {os.fspath(checkpoint_path)!r}."
        )
    return int(match.group(1))
