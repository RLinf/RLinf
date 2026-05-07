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

# Auto-apply LIBERO reward patch for all Python processes
import sys

# Add repo to path if not already
repo_path = "/workspace/test/RLinf"
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

# Only try once
if "RLINF_REWARD_PATCHED" not in sys.modules.get("builtins", {}).__dict__:
    try:
        from rlinf.models.embodiment.gr00t_1_6.patch_libero_reward import (
            patch_libero_reward,
        )

        patch_libero_reward()
        import builtins

        builtins.RLINF_REWARD_PATCHED = True
    except Exception:
        pass
