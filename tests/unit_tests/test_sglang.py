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

import asyncio
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import MethodType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="requires SGLang"
)
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


@pytest.fixture
def sglang_args_modules(monkeypatch):
    """Exercise argument assembly without importing the vendor GPU engine."""
    import dataclasses

    original_version = importlib.metadata.version
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda name: "0.5.19" if name == "sglang" else original_version(name),
    )
    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    # Vendor capability checks run against CPU-only stand-ins for its APIs.
    for module_name, class_name, names in (
        ("server_args", "ServerArgs", ["enable_token_space_processor"]),
        (
            "managers.io_struct",
            "GenerateReqInput",
            ["mm_token_expansion_start_len", "return_prompt_token_ids"],
        ),
    ):
        monkeypatch.setitem(
            sys.modules,
            f"sglang.srt.{module_name}",
            SimpleNamespace(
                **{class_name: dataclasses.make_dataclass(class_name, names)}
            ),
        )
    monkeypatch.setitem(
        sys.modules,
        "sglang.srt.multimodal.token_space_mm_process.processor",
        SimpleNamespace(
            TokenSpaceProcessor=SimpleNamespace(
                process=lambda mm_token_expansion_start_len: None
            )
        ),
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


@pytest.mark.parametrize("location", ["top", "meta", "missing"])
def test_serverless_generation_returns_requested_prompt_ids(
    sglang_args_modules, location
):
    _, worker_module, _ = sglang_args_modules
    result = {
        "output_ids": [4],
        "meta_info": {"finish_reason": {"type": "stop"}, "prompt_tokens": 2},
    }
    if location == "top":
        result["prompt_token_ids"] = [1, 3]
    elif location == "meta":
        result["meta_info"]["prompt_token_ids"] = [1, 3]
    worker = SimpleNamespace(
        _is_model_instance_entry=True,
        _weight_version=2,
        _sampling_params={},
        _return_logprobs=False,
        _backend=SimpleNamespace(async_generate=AsyncMock(return_value=result)),
        log_error=Mock(),
    )
    channel = Mock()
    channel.put.return_value.async_wait = AsyncMock()
    asyncio.run(
        worker_module.SGLangWorker.generate_and_send(
            worker, channel, "episode", [1, 3], return_prompt_token_ids=True
        )
    )
    payload = channel.put.call_args.args[0]
    if location == "missing":
        assert "prompt_token_ids" in payload["error"]
    else:
        assert payload["prompt_token_ids"] == [1, 3]
        assert payload["version_start"] == payload["version_end"] == 2


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
    assert not args.get("enable_token_space_processor", False)


@pytest.mark.parametrize(
    "args_api,backup,resume_dir,processor_key,enabled",
    [
        ("dataclass-old", False, None, None, False),
        ("dataclass-new", True, "/checkpoint", None, False),
        ("struct", False, "/checkpoint", None, False),
        ("dataclass-new", False, None, "enable_shared_multi_modal_processor", True),
        ("dataclass-new", False, None, "enable_token_space_processor", True),
        ("struct", False, None, "enable_shared_multi_modal_processor", False),
    ],
)
def test_sglang_server_args_preserve_loading_and_processor_settings(
    monkeypatch,
    resume_dir,
    backup,
    args_api,
    processor_key,
    enabled,
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
            "ServerArgs", [(key, object) for key in kwargs]
        )
        return cls(**kwargs)

    monkeypatch.setattr(sglang_worker, "ServerArgs", server_args)
    graph_field = (
        "cuda_graph_max_bs"
        if args_api == "dataclass-old"
        else "cuda_graph_max_bs_decode"
    )
    server_args.__struct_fields__ = (graph_field, "chunked_prefill_size")
    worker = sglang_args_worker
    worker._cfg.runner.resume_dir = resume_dir
    worker._cfg_rollout.sglang.enable_weights_cpu_backup = backup
    if processor_key is not None:
        worker._cfg_rollout.sglang[processor_key] = enabled
    args = sglang_worker.SGLangWorker._build_server_args(worker)
    assert args["disable_overlap_schedule"] is True
    assert args["enable_weights_cpu_backup"] is bool(backup)
    assert args["load_format"] == ("auto" if backup or resume_dir is None else "dummy")
    assert args[graph_field] == 2
    assert args["chunked_prefill_size"] == 4096
    assert "enable_shared_multi_modal_processor" not in args
    assert args.get("enable_token_space_processor", False) is enabled
    if enabled:
        assert args["disable_fast_image_processor"] is True
        assert args["image_processor_backend"] == "pil"
    else:
        assert "enable_token_space_processor" not in args
        assert "disable_fast_image_processor" not in args
        assert "image_processor_backend" not in args


@pytest.mark.parametrize(
    "settings,enabled,error",
    [
        ({}, False, None),
        ({"enable_shared_multi_modal_processor": True}, True, None),
        ({"enable_token_space_processor": True}, True, None),
        ({"enable_shared_multi_modal_processor": False}, False, None),
        ({"enable_shared_multi_modal_processor": None}, None, "must be a boolean"),
        ({"enable_token_space_processor": "false"}, None, "must be a boolean"),
        (
            {
                "enable_shared_multi_modal_processor": True,
                "enable_token_space_processor": True,
            },
            None,
            "keep only",
        ),
        (
            {
                "enable_shared_multi_modal_processor": False,
                "enable_token_space_processor": True,
            },
            None,
            "keep only",
        ),
    ],
)
def test_shared_multimodal_switch_accepts_legacy_configs(settings, enabled, error):
    from omegaconf import OmegaConf

    from rlinf.config import (
        shared_multi_modal_processor_enabled,
        validate_sglang_multimodal_cfg,
    )

    cfg = OmegaConf.create(settings)
    OmegaConf.set_struct(cfg, True)
    if error:
        with pytest.raises(ValueError, match=error):
            validate_sglang_multimodal_cfg(cfg)
    else:
        validate_sglang_multimodal_cfg(cfg)
        assert shared_multi_modal_processor_enabled(cfg) is enabled
    assert OmegaConf.to_container(cfg) == settings


@pytest.mark.parametrize("release", ["0.5.12.post1", "0.5.19", "0.5.20.dev0"])
def test_sglang_experimental_build_requires_explicit_rollout(
    release, sglang_args_modules
):
    from packaging.version import parse

    package, _, process = sglang_args_modules
    package.sglang_version = parse(release)
    package.require_supported_sglang(token_space=True)
    if release != "0.5.20.dev0":
        package.require_supported_sglang()
        return
    with pytest.raises(ValueError, match="enable_shared_multi_modal_processor=true"):
        package.require_supported_sglang()
    for server_type in ("srt", "embodied"):
        server = process.SGLangServerProcess(
            {"enable_token_space_processor": True},
            acquire_free_port=lambda **kwargs: pytest.fail("allocated a port"),
            server_type=server_type,
        )
        with pytest.raises(ValueError, match="supported release"):
            server.start()
        assert server._server_proc is None


def test_token_space_processor_resolution_preserves_launch_inputs(monkeypatch):
    from rlinf.hybrid_engines.sglang.multimodal.token_space import processor_settings

    class ServerArgs:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def resolve_once(self):
            self.kwargs["tokenizer_path"] = self.kwargs["model_path"]
            self.kwargs["chunked_prefill_size"] //= 2
            self.kwargs["mm_process_config"]["image"] = {"max_pixels": 4096}

        def resolved_dict(self):
            return self.kwargs

    monkeypatch.setitem(
        sys.modules, "sglang.srt.server_args", SimpleNamespace(ServerArgs=ServerArgs)
    )
    raw = {
        "model_path": "/model",
        "model_type": "qwen2.5",
        "tokenizer_path": None,
        "tokenizer_mode": "auto",
        "tokenizer_backend": "huggingface",
        "trust_remote_code": False,
        "revision": None,
        "model_config_parser": "auto",
        "json_model_override_args": "{}",
        "image_processor_backend": "pil",
        "disable_fast_image_processor": True,
        "mm_process_config": {},
        "chunked_prefill_size": 4096,
    }
    settings = processor_settings(raw)
    assert settings["tokenizer_path"] == "/model"
    assert settings["mm_process_config"] == {"image": {"max_pixels": 4096}}
    assert raw["tokenizer_path"] is None
    assert raw["chunked_prefill_size"] == 4096
    assert raw["mm_process_config"] == {}


@pytest.fixture
def qwen_image_processor(monkeypatch):
    """Real SGLang/HF preprocessing; model assets are optional in CPU CI."""
    model_path = os.environ.get("RLINF_TEST_QWEN_VL_MODEL")
    if not model_path:
        pytest.skip("Set RLINF_TEST_QWEN_VL_MODEL to a local Qwen3.5 processor.")
    pytest.importorskip("sglang")
    from transformers import AutoConfig, AutoProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import TokenizedQwenVLProcessor

    hf = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    server = SimpleNamespace(
        mm_process_config={},
        skip_tokenizer_init=True,
        disable_fast_image_processor=True,
        keep_mm_feature_on_device=False,
        tokenizer_worker_num=1,
    )
    # Recent SGLang reads processor settings from a process-local context.
    # Publish real config bags without initializing the GPU engine.
    if importlib.util.find_spec("sglang.srt.runtime_context") is not None:
        from sglang.srt import runtime_context
        from sglang.srt.server_args import ServerArgs

        server = ServerArgs(
            model_path=model_path,
            mm_process_config={},
            skip_tokenizer_init=True,
            disable_fast_image_processor=True,
            mm_preprocess_cache_size_mb=0,
        )
        context = runtime_context.RuntimeContext(runtime_context.ParallelContext())
        context.set_server_args(server)
        monkeypatch.setattr(runtime_context, "_CONTEXT", context)
    processor = TokenizedQwenVLProcessor(config, server, hf, "cpu", skip_mm_pool=True)
    try:
        yield processor, hf, config
    finally:
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()


@pytest.mark.parametrize(
    ("detokenize", "parts"),
    [(False, ["\n", "\n"]), (True, ["\n", "\n"]), (False, ["a", "hello"])],
)
def test_qwen_image_processor_preserves_sampled_tokens(
    qwen_image_processor, detokenize, parts
):
    from PIL import Image
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    processor, hf, config = qwen_image_processor
    processor.server_args.skip_tokenizer_init = not detokenize
    processor.skip_tokenizer_init = not detokenize
    images = [Image.new("RGB", (64, 64), "red"), Image.new("RGB", (96, 64), "blue")]
    separator = "".join(parts)
    text = f"A<|vision_start|><|image_pad|><|vision_end|>{separator}<|im_end|>B<|vision_start|><|image_pad|><|vision_end|>C"
    expected = hf(text=[text], images=images, return_tensors="pt")
    canonical = expected.input_ids[0].tolist()
    at = canonical.index(config.vision_end_token_id) + 1
    separator_size = len(hf.tokenizer.encode(separator, add_special_tokens=False))
    prompt = (
        canonical[:at]
        + [
            token
            for part in parts
            for token in hf.tokenizer.encode(part, add_special_tokens=False)
        ]
        + canonical[at + separator_size :]
    )
    assert hf.tokenizer.decode(prompt) == hf.tokenizer.decode(canonical)
    assert prompt != hf.tokenizer.encode(
        hf.tokenizer.decode(prompt), add_special_tokens=False
    )
    request = SimpleNamespace(
        video_data=None, audio_data=None, rid="token-preservation"
    )
    output = asyncio.run(processor.process_mm_data_async(images, prompt, request))
    assert output.input_ids == prompt
    assert torch.equal(
        torch.cat([item.feature for item in output.mm_items]), expected.pixel_values
    )
    input_ids = torch.tensor([prompt])
    for item in output.mm_items:
        for start, end in item.offsets:
            assert input_ids[0, start : end + 1].eq(config.image_token_id).all()
    # HF's independent position implementation needs only configuration, not weights.
    geometry = SimpleNamespace(config=config)
    geometry.get_vision_position_ids = MethodType(
        Qwen3_5Model.get_vision_position_ids, geometry
    )
    positions, delta = Qwen3_5Model.get_rope_index(
        geometry,
        input_ids,
        (input_ids == config.image_token_id).int(),
        image_grid_thw=expected.image_grid_thw,
    )
    assert torch.equal(output.mrope_positions, positions[:, 0])
    assert torch.equal(output.mrope_position_delta, delta)


@pytest.mark.parametrize("invalid", ["grid", "image_count"])
def test_qwen_image_processor_rejects_inconsistent_images(
    qwen_image_processor, invalid
):
    from PIL import Image

    processor, hf, config = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    prompt = (
        hf(
            text=["A<|vision_start|><|image_pad|><|vision_end|>B"],
            images=images,
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    if invalid == "grid":
        prompt.remove(config.image_token_id)
        message = "processor grid needs"
    else:
        images.append(images[0])
        message = "image spans, but received"
    request = SimpleNamespace(video_data=None, audio_data=None, rid="invalid-image")
    with pytest.raises(ValueError, match=message):
        asyncio.run(processor.process_mm_data_async(images, prompt, request))


def test_qwen_image_processor_keeps_text_request_behavior(qwen_image_processor):
    from PIL import Image

    processor, hf, _ = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    text = "A<|vision_start|><|image_pad|><|vision_end|>\n\nB"
    expected = hf(text=[text], images=images, return_tensors="pt")
    request = SimpleNamespace(video_data=None, audio_data=None, rid="text-request")
    output = asyncio.run(processor.process_mm_data_async(images, text, request))
    assert output.input_ids == expected.input_ids[0].tolist()
    assert torch.equal(output.mm_items[0].feature, expected.pixel_values)


def test_qwen_release_capture_replays_only_real_images(qwen_image_processor):
    from tempfile import TemporaryDirectory
    from uuid import uuid4

    from PIL import Image

    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        finish_capture,
    )

    processor, _, config = qwen_image_processor
    start, pad, end = (
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    )
    images = [Image.new("RGB", (64, 64), color) for color in ("red", "blue")]
    ids, spans, history = [], [], None
    for turn, sampled in enumerate(([pad], [start]), start=1):
        spans = [*spans, [len(ids) + 1, len(ids) + 2]]
        ids = [*ids, start, pad, end]
        with TemporaryDirectory(prefix="rlinf-qwen-") as directory:
            ticket, rid = Path(directory).name, uuid4().hex
            request = SimpleNamespace(
                video_data=None,
                audio_data=None,
                rid=rid,
                sampling_params={
                    "custom_params": {
                        CAPTURE_KEY: {
                            "ticket": ticket,
                            "image_spans": spans,
                            "prefix_length": history["length"] if history else 0,
                        }
                    }
                },
            )
            output = asyncio.run(
                processor.process_mm_data_async(images[:turn], ids, request)
            )
            entry, history = finish_capture(
                capture_file(ticket, rid), sampled, history, None
            )
        assert entry["prompt_token_ids"] == output.input_ids
        assert torch.equal(
            entry["pixel_values"], torch.cat([i.feature.cpu() for i in output.mm_items])
        )
        assert len(entry["image_spans"]) == turn
        assert entry["prompt_token_ids"][: spans[-1][0]] == ids[: spans[-1][0]]
        ids = output.input_ids + sampled + [start, pad, end]
        spans = entry["image_spans"]
