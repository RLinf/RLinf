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

from pathlib import Path


def test_openpi_gemma_patch_casts_at_linear_boundaries():
    repo_root = Path(__file__).resolve().parents[2]
    patch_file = (
        repo_root
        / "requirements"
        / "embodied"
        / "patches"
        / "openpi_transformers_4_57_1"
        / "transformers"
        / "models"
        / "gemma"
        / "modeling_gemma.py"
    )
    source = patch_file.read_text()

    assert "proj_dtype = self.q_proj.weight.dtype" in source
    assert "hidden_states = hidden_states.to(dtype=proj_dtype)" in source
    assert "up_proj_dtype = self.up_proj.weight.dtype" in source
    assert "x = x.to(dtype=up_proj_dtype)" in source
    assert "down_proj_dtype = self.down_proj.weight.dtype" in source
    assert "hidden_states = hidden_states.to(dtype=down_proj_dtype)" in source
    assert "o_proj_dtype = self.o_proj.weight.dtype" in source
    assert "attn_output = attn_output.to(dtype=o_proj_dtype)" in source
