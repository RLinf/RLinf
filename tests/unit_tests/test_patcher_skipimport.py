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

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

_PATCHER_UNDER_TEST = "_rlinf_utils_patcher_under_test"
_TEST_MODULE_NAMES = (
    "flash_attn",
    "rlinf_test_missing_cuda_dep",
    "rlinf_test_missing_cuda_pkg",
    "rlinf_test_loaded_dep",
)


def _load_patcher_module():
    patcher_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "patcher.py"
    )
    spec = importlib.util.spec_from_file_location(
        _PATCHER_UNDER_TEST,
        patcher_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


Patcher = _load_patcher_module().Patcher


def _remove_test_modules() -> None:
    for root_name in _TEST_MODULE_NAMES:
        for module_name in list(sys.modules):
            if module_name == root_name or module_name.startswith(f"{root_name}."):
                del sys.modules[module_name]


@pytest.fixture(autouse=True)
def clean_patcher_state():
    originals = {
        name: sys.modules[name] for name in _TEST_MODULE_NAMES if name in sys.modules
    }
    _remove_test_modules()
    Patcher.clear()
    yield
    Patcher.clear()
    _remove_test_modules()
    sys.modules.update(originals)


def test_skip_import_registers_stub_immediately():
    result = Patcher.skip_import("rlinf_test_missing_cuda_dep")

    assert result is Patcher
    stub_module = sys.modules["rlinf_test_missing_cuda_dep"]
    assert stub_module.__name__ == "rlinf_test_missing_cuda_dep"
    assert stub_module.some_kernel() is None


def test_skip_import_supports_submodule_imports():
    Patcher.skip_import("rlinf_test_missing_cuda_pkg")

    submodule = importlib.import_module("rlinf_test_missing_cuda_pkg.bert_padding")
    namespace = {}
    exec(
        "from rlinf_test_missing_cuda_pkg.bert_padding import unpad_input",
        namespace,
    )

    assert submodule is sys.modules["rlinf_test_missing_cuda_pkg.bert_padding"]
    assert namespace["unpad_input"]() is None


def test_skip_import_preserves_loaded_modules():
    loaded_module = types.ModuleType("rlinf_test_loaded_dep")
    loaded_module.existing_value = object()
    sys.modules["rlinf_test_loaded_dep"] = loaded_module

    Patcher.skip_import("rlinf_test_loaded_dep")

    assert sys.modules["rlinf_test_loaded_dep"] is loaded_module


def test_clear_stub_import_removes_stub_tree_only():
    Patcher.skip_import("rlinf_test_missing_cuda_dep")
    importlib.import_module("rlinf_test_missing_cuda_dep.bert_padding")

    result = Patcher.clear_stub_import("rlinf_test_missing_cuda_dep")

    assert result is Patcher
    assert "rlinf_test_missing_cuda_dep" not in sys.modules
    assert "rlinf_test_missing_cuda_dep.bert_padding" not in sys.modules


def test_clear_stub_import_preserves_real_modules():
    real_module = types.ModuleType("rlinf_test_loaded_dep")
    real_module.__file__ = "/site-packages/rlinf_test_loaded_dep/__init__.py"
    sys.modules["rlinf_test_loaded_dep"] = real_module

    Patcher.clear_stub_import("rlinf_test_loaded_dep")

    assert sys.modules["rlinf_test_loaded_dep"] is real_module


def test_clear_stub_import_removes_flash_attn_stub_tree_only():
    Patcher.skip_import("flash_attn")
    importlib.import_module("flash_attn.bert_padding")

    Patcher.clear_stub_import("flash_attn")

    assert "flash_attn" not in sys.modules
    assert "flash_attn.bert_padding" not in sys.modules


def test_clear_stub_import_preserves_real_flash_attn_module():
    real_flash_attn = types.ModuleType("flash_attn")
    real_flash_attn.__file__ = "/site-packages/flash_attn/__init__.py"
    sys.modules["flash_attn"] = real_flash_attn

    Patcher.clear_stub_import("flash_attn")

    assert sys.modules["flash_attn"] is real_flash_attn
