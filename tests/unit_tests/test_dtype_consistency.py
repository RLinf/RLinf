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

"""CPU-only tests for weight dtype consistency checks (see issue #1662).

The released RLinf/WideSeek-R1-4b checkpoint stores fp16 weights while its
``config.json`` declares ``bfloat16``, so serving engines that follow the
config dtype by default (vLLM/SGLang/HF Transformers with ``dtype="auto"``)
upcast the weights and lose accuracy. These tests cover the guard that flags
such mismatches from metadata only, without loading any weight tensor.
"""

import json
import struct
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.config import torch_dtype_from_precision
from rlinf.utils.dtype_check import (
    check_weight_dtype_consistency,
    warn_on_weight_dtype_mismatch,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_safetensors_file(path: Path, dtype_str: str = "F16") -> None:
    """Create a minimal valid safetensors file with one small fp tensor."""
    header = {
        "model.weight": {"dtype": dtype_str, "shape": [2], "data_offsets": [0, 4]},
    }
    header_bytes = json.dumps(header).encode("utf-8")
    # The safetensors format pads the header with spaces to a multiple of 8.
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"\x00" * 4)


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    """A model directory that reproduces the issue #1662 mismatch."""
    _write_safetensors_file(tmp_path / "model.safetensors", dtype_str="F16")
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "torch_dtype": "bfloat16",
            }
        )
    )
    return tmp_path


def test_config_declared_dtype_matching_weights_is_silent(model_dir: Path):
    """No warning when config.json agrees with the safetensors weight dtype."""
    (model_dir / "config.json").write_text(json.dumps({"torch_dtype": "float16"}))
    assert check_weight_dtype_consistency(model_dir, serving_dtype=torch.float16) == []


def test_config_bf16_with_fp16_weights_is_flagged(model_dir: Path):
    """The exact issue #1662 shape: bf16 config over fp16 weights."""
    warnings = check_weight_dtype_consistency(model_dir)
    assert len(warnings) == 1
    assert str(torch.bfloat16) in warnings[0]
    assert str(torch.float16) in warnings[0]


def test_serving_dtype_mismatch_is_flagged(model_dir: Path):
    """A serving dtype different from the stored weights is flagged."""
    warnings = check_weight_dtype_consistency(model_dir, serving_dtype=torch.bfloat16)
    assert len(warnings) == 2  # config-vs-weights and serving-vs-weights
    assert any(str(torch.bfloat16) in w for w in warnings)


def test_sharded_checkpoint_is_flagged(tmp_path: Path):
    """Mismatch detection reads dtypes through model.safetensors.index.json."""
    shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
    for shard in shards:
        _write_safetensors_file(tmp_path / shard, dtype_str="F16")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "model.weight": shards[0],
                    "model.layers.0.weight": shards[1],
                },
            }
        )
    )
    (tmp_path / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    warnings = check_weight_dtype_consistency(tmp_path)
    assert len(warnings) == 1
    assert str(torch.float16) in warnings[0]


def test_paths_without_metadata_are_silent(tmp_path: Path):
    """Hub ids, missing directories, and metadata-free dirs never warn."""
    assert check_weight_dtype_consistency("RLinf/WideSeek-R1-4b") == []
    assert check_weight_dtype_consistency(tmp_path / "does-not-exist") == []

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert check_weight_dtype_consistency(empty_dir) == []


def test_warn_helper_never_raises(tmp_path: Path, model_dir: Path):
    """Broken metadata must not break engine initialization."""
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    (broken_dir / "config.json").write_text("{not json")
    warn_on_weight_dtype_mismatch(broken_dir)  # must not raise

    # And on the real mismatch it logs one warning per finding.
    logged: list[str] = []
    warn_on_weight_dtype_mismatch(model_dir, log=logged.append)
    assert len(logged) == 1
    assert str(torch.float16) in logged[0]


@pytest.mark.parametrize(
    "yaml_relpath",
    [
        "examples/agent/wideseek_r1/config/base_eval.yaml",
        "tests/e2e_tests/agent/wideseek/qwen3-eval.yaml",
    ],
)
def test_wideseek_eval_configs_serve_fp16(yaml_relpath: str):
    """The shipped WideSeek-R1 eval configs must serve fp16.

    The released WideSeek-R1-4b weights are stored as fp16, so the rollout
    engines must be handed torch.float16; flipping these to bf16 would
    reintroduce the accuracy drop reported in issue #1662.
    """
    cfg = OmegaConf.load(_REPO_ROOT / yaml_relpath)
    assert torch_dtype_from_precision(cfg.rollout.model.precision) is torch.float16
    assert (
        torch_dtype_from_precision(cfg.rollout_fixed_worker.model.precision)
        is torch.float16
    )
