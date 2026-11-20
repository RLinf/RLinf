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

"""SGLang subprocess startup."""

import importlib.util
import subprocess
import sys
from textwrap import dedent

import pytest


@pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="requires SGLang")
def test_patched_scheduler_delegates_to_vendor():
    # Applying the engine patch changes module globals throughout the interpreter.
    script = dedent(
        """
        from unittest.mock import Mock, patch
        import sglang.srt.managers.scheduler as vendor

        def check():
            entry = Mock()
            vendor.run_scheduler_process = entry
            from rlinf.hybrid_engines.sglang.common import sgl_engine, sgl_scheduler
            from rlinf.utils.patcher import Patcher

            assert vendor.run_scheduler_process is sgl_scheduler.run_scheduler_process
            with patch.object(Patcher, "apply", side_effect=[None]):
                vendor.run_scheduler_process("request", rank=0)
            entry.assert_called_once_with("request", rank=0)

        check()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)
