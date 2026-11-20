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

"""SGLang request metadata, image processing, and service lifecycle."""

import importlib.metadata
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture
def sglang_args_modules(monkeypatch):
    """Exercise argument assembly without importing the vendor GPU engine."""
    import dataclasses

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.5.12.post1" if name == "sglang" else original_version(name),
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.server_args",
        SimpleNamespace(ServerArgs=dataclasses.make_dataclass("ServerArgs", [])),
    )
    # The backend owns vendor engine/process creation; no backend is launched here.
    monkeypatch.setitem(
        sys.modules,
        "rlinf.workers.rollout.sglang.backends",
        SimpleNamespace(make_backend=None),
    )
    directory = Path(__file__).parents[2] / "rlinf/workers/rollout/sglang"
    modules = []
    for filename in ("__init__", "sglang_worker", "process"):
        name = "rlinf.workers.rollout.sglang"
        if filename != "__init__":
            name += f".{filename}"
        spec = importlib.util.spec_from_file_location(
            name, directory / f"{filename}.py"
        )
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.fixture
def sglang_args_worker():
    from omegaconf import OmegaConf

    rollout = OmegaConf.create(
        {
            "enforce_eager": False,
            "validate_weight": False,
            "validate_weight_first_sync": True,
            "model": {
                "model_path": "/model",
                "model_type": "qwen2.5",
                "precision": "bf16",
                "trust_remote_code": True,
            },
            "sglang": {
                "use_torch_compile": False,
                "torch_compile_max_bs": 4,
                "decode_log_interval": 100,
                "attention_backend": "triton",
                "disable_overlap_schedule": True,
                "chunked_prefill_size": 4096,
            },
            "cuda_graph_max_bs": 4,
            "max_running_requests": 2,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "detokenize": False,
        }
    )
    return SimpleNamespace(
        _cfg=SimpleNamespace(runner=SimpleNamespace(resume_dir=None)),
        _cfg_rollout=rollout,
        _rank=0,
        weight_reload="sync",
        acquire_free_port=lambda: 12345,
        log_on_first_rank=lambda message: None,
    )


def test_sglang_server_args_with_installed_vendor(sglang_args_worker):
    pytest.importorskip("sglang.srt.server_args")
    model = os.environ.get("RLINF_TEST_SGLANG_MODEL")
    if not model:
        pytest.skip("Set RLINF_TEST_SGLANG_MODEL to a local model directory")
    if not torch.cuda.is_available():
        pytest.skip("The native SGLang Engine import requires a CUDA device")
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    worker = sglang_args_worker
    worker._cfg_rollout.model.model_path = model
    worker._cfg_rollout.enforce_eager = True
    args = SGLangWorker._build_server_args(worker)
    graph_limit = args.get("cuda_graph_max_bs", args.get("cuda_graph_max_bs_decode"))
    assert graph_limit == 2
    assert args["chunked_prefill_size"] == 4096
    assert args["load_format"] == "auto"


@pytest.mark.parametrize(
    "args_api,backup,resume_dir",
    [
        ("dataclass-old", None, None),
        ("dataclass-old", False, "/checkpoint"),
        ("dataclass-old", True, "/checkpoint"),
        ("dataclass-new", True, "/checkpoint"),
        ("struct", False, "/checkpoint"),
    ],
)
def test_sglang_server_args_preserve_loading_settings(
    monkeypatch,
    resume_dir,
    backup,
    args_api,
    sglang_args_modules,
    sglang_args_worker,
):
    import dataclasses

    _, sglang_worker, _ = sglang_args_modules

    # Isolate the vendor engine boundary; exercise RLinf's real argument assembly.
    def server_args(**kwargs):
        if args_api == "struct":
            import msgspec

            def resolve_once(self):
                self.chunked_prefill_size //= 2

            cls = msgspec.defstruct(
                "ServerArgs",
                [(key, object) for key in kwargs],
                namespace={
                    "resolve_once": resolve_once,
                    "resolved_dict": lambda self: msgspec.structs.asdict(self),
                },
            )
            return cls(**kwargs)
        cls = dataclasses.make_dataclass(
            "ServerArgs",
            [
                (key, object)
                for key in kwargs
                if args_api != "dataclass-old" or key != "enable_weights_cpu_backup"
            ],
        )
        return cls(**kwargs)

    monkeypatch.setattr(sglang_worker, "ServerArgs", server_args)
    graph_field = (
        "cuda_graph_max_bs"
        if args_api == "dataclass-old"
        else "cuda_graph_max_bs_decode"
    )
    server_args.__struct_fields__ = (graph_field, "chunked_prefill_size")
    if args_api != "dataclass-old":
        server_args.__struct_fields__ += ("enable_weights_cpu_backup",)
    worker = sglang_args_worker
    worker._cfg.runner.resume_dir = resume_dir
    if backup is not None:
        worker._cfg_rollout.sglang.enable_weights_cpu_backup = backup
    if args_api == "dataclass-old" and backup:
        with pytest.raises(
            ValueError, match="does not support enable_weights_cpu_backup"
        ):
            sglang_worker.SGLangWorker._build_server_args(worker)
        return
    args = sglang_worker.SGLangWorker._build_server_args(worker)
    assert args["disable_overlap_schedule"] is True
    if args_api == "dataclass-old":
        assert "enable_weights_cpu_backup" not in args
    else:
        assert args["enable_weights_cpu_backup"] is bool(backup)
    assert args["load_format"] == ("auto" if backup or resume_dir is None else "dummy")
    assert args[graph_field] == 2
    assert args["chunked_prefill_size"] == 4096
