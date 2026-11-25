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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("component_api", [True, False])
def test_sglang_memory_requests_track_split_resume(monkeypatch, component_api):
    native = pytest.importorskip("sglang.srt.managers.scheduler")
    from sglang.srt.managers.io_struct import (
        ReleaseMemoryOccupationReqInput,
        ResumeMemoryOccupationReqInput,
    )
    from sglang.utils import TypeBasedDispatcher

    from rlinf.hybrid_engines.sglang.common.sgl_scheduler import Scheduler

    resident = {"weights", "kv_cache", "cuda_graph"}

    class MemoryManager:
        def release_memory_occupation(self, request):
            resident.difference_update(request.tags or resident.copy())

        def resume_memory_occupation(self, request):
            resident.update(request.tags or ["weights", "kv_cache", "cuda_graph"])

    manager = MemoryManager()

    def initialize(self, *args, **kwargs):
        self.tp_rank = 0
        self.tp_worker = SimpleNamespace(worker=SimpleNamespace())
        if component_api:
            self.weight_updater = manager
            release = manager.release_memory_occupation
            resume = manager.resume_memory_occupation
        else:
            release = self.release_memory_occupation
            resume = self.resume_memory_occupation
        self._request_dispatcher = TypeBasedDispatcher(
            [
                (ReleaseMemoryOccupationReqInput, release),
                (ResumeMemoryOccupationReqInput, resume),
            ]
        )

    monkeypatch.setattr(native.Scheduler, "__init__", initialize)
    if not component_api:
        monkeypatch.setattr(
            native.Scheduler,
            "release_memory_occupation",
            lambda self, request: manager.release_memory_occupation(request),
            raising=False,
        )
        monkeypatch.setattr(
            native.Scheduler,
            "resume_memory_occupation",
            lambda self, request: manager.resume_memory_occupation(request),
            raising=False,
        )
    scheduler = Scheduler(SimpleNamespace(device="cpu"))
    scheduler.weight_reload = "sync"
    scheduler._request_dispatcher(ReleaseMemoryOccupationReqInput())
    assert not resident
    assert scheduler.is_weight_offloaded
    scheduler._request_dispatcher(ResumeMemoryOccupationReqInput(tags=["weights"]))
    assert resident == {"weights"}
    assert not scheduler.is_weight_offloaded
    assert (
        scheduler.offloaded_tags["kv_cache"] and scheduler.offloaded_tags["cuda_graph"]
    )
    scheduler._request_dispatcher(
        ResumeMemoryOccupationReqInput(tags=["kv_cache", "cuda_graph"])
    )
    assert resident == {"weights", "kv_cache", "cuda_graph"}
    assert not any(scheduler.offloaded_tags.values())


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
                "model_type": "qwen",
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


@pytest.mark.parametrize(
    "args_api,backup,resume_dir,processor_key,enabled",
    [
        ("dataclass-old", False, None, None, False),
        ("dataclass-new", True, "/checkpoint", None, False),
        ("struct", False, "/checkpoint", None, False),
        ("dataclass-new", False, None, "enable_shared_multi_modal_processor", True),
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
    monkeypatch.setattr(
        sglang_worker,
        "_server_argument_fields",
        lambda: {
            graph_field,
            "chunked_prefill_size",
            "enable_metrics",
            "enable_cache_report",
            "disable_fast_image_processor",
        },
    )
    worker = sglang_args_worker
    worker._cfg_rollout.sglang.disable_fast_image_processor = True
    worker._cfg_rollout.sglang.enable_metrics = True
    worker._cfg_rollout.sglang.enable_cache_report = True
    worker._cfg.runner.resume_dir = resume_dir
    worker._cfg_rollout.sglang.enable_weights_cpu_backup = backup
    if processor_key is not None:
        worker._cfg_rollout.sglang[processor_key] = enabled
    args = sglang_worker.SGLangWorker._build_server_args(worker)
    assert args["disable_overlap_schedule"] is True
    assert args["disable_fast_image_processor"] is True
    assert args["enable_metrics"] is True
    assert args["enable_cache_report"] is True
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
        assert "image_processor_backend" not in args


@pytest.mark.parametrize(
    "settings,enabled,error",
    [
        ({}, False, None),
        ({"enable_shared_multi_modal_processor": True}, True, None),
        ({"enable_shared_multi_modal_processor": False}, False, None),
        ({"enable_shared_multi_modal_processor": None}, None, "must be a boolean"),
    ],
)
def test_shared_multimodal_switch_validates_explicit_values(settings, enabled, error):
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


def test_serverless_cohort_cleanup_fences_queue(sglang_args_modules):
    from rlinf.scheduler.channel.channel import DEFAULT_KEY

    _, module, _ = sglang_args_modules

    async def serve():
        started = asyncio.Event()
        aborted = asyncio.Event()

        class ChannelService:
            def __init__(self):
                self.queues = {}

            def put(self, item, key=DEFAULT_KEY, **kwargs):
                queue = self.queues.setdefault(key, asyncio.Queue())
                return SimpleNamespace(async_wait=lambda: queue.put(item))

            def get(self, key=DEFAULT_KEY, **kwargs):
                queue = self.queues.setdefault(key, asyncio.Queue())
                return SimpleNamespace(async_wait=queue.get)

        class Backend:
            async def async_generate(self, **kwargs):
                started.set()
                await aborted.wait()
                return {"meta_info": {"finish_reason": {"type": "abort"}}}

            async def abort_generation(self):
                aborted.set()

            async def get_running_state(self):
                return SimpleNamespace(num_running_reqs=0, num_queue_reqs=0)

        worker = SimpleNamespace(
            _is_model_instance_entry=True,
            _cfg_rollout={},
            _rank=0,
            _placement=SimpleNamespace(rollout_tp_size=1),
            _sampling_params={},
            _return_logprobs=True,
            _weight_version=0,
            _backend=Backend(),
            _closed_generation_cohorts=set(),
            _inflight_generates=set(),
            _generation_cohorts={},
            _generation_outputs={},
            _cohort_fences={},
            _dispatch_gate=asyncio.Event(),
            _serverless_stopping=False,
            log_error=lambda message: pytest.fail(message),
        )
        worker.generate_and_send = lambda **kwargs: (
            module.SGLangWorker.generate_and_send(worker, **kwargs)
        )
        worker._dispatch_gate.set()
        incoming, outgoing = ChannelService(), ChannelService()
        await incoming.put(
            {
                "cohort_id": "closed",
                "channel_key": "cancelled-episode",
                "prompt_ids": [1],
            },
            async_op=True,
        ).async_wait()
        server = asyncio.create_task(
            module.SGLangWorker.rollout_serverless(worker, incoming, outgoing)
        )
        await started.wait()
        receipt = await module.SGLangWorker.finish_generation_cohort(worker, "closed")
        assert receipt["queue_fenced"] and receipt["inflight"] == 0
        replies = outgoing.queues["cancelled-episode"]
        assert replies.get_nowait() == {"error": "synchronous_oversample_drop"}
        assert replies.empty()
        await incoming.put(None, async_op=True).async_wait()
        await server
        await asyncio.gather(*worker._inflight_generates)

    asyncio.run(asyncio.wait_for(serve(), timeout=10))


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


def test_sglang_packet_metadata_stays_aligned_with_cumulative_tokens():
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        accumulate_token_metadata,
        decode_generation_result,
    )

    states = {"episode": SimpleNamespace()}
    for version, count, forced in [(1, 3, False), (2, 2, False), (2, 1, True)]:
        packet = SimpleNamespace(
            rids=["episode", "cancelled"],
            customized_info={
                "rollout_weight_versions": [[version] * count, [0]],
                "forced_token_mask": [[forced] * count, [False]],
            },
        )
        accumulate_token_metadata(packet, states)
    assert packet.customized_info["rollout_weight_versions"][0] == [1, 1, 1, 2, 2, 2]
    assert packet.customized_info["forced_token_mask"][0] == [False] * 5 + [True]

    result = decode_generation_result(
        {
            "output_ids": [10, 11, 12, 13, 14, 15],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 8,
                "forced_token_mask": packet.customized_info["forced_token_mask"][0],
                "rollout_weight_versions": packet.customized_info[
                    "rollout_weight_versions"
                ][0],
                "output_token_logprobs": [
                    (-0.5, token, None) for token in range(10, 16)
                ],
            },
        },
        return_logprobs=True,
    )
    assert result["forced_token_positions"] == [5]
    assert (
        len(result["logprobs"])
        == len(result["weight_versions"])
        == len(result["output_ids"])
    )


def test_token_space_http_and_generation_metadata_preserve_prompt_ids(monkeypatch):
    from rlinf.hybrid_engines.sglang.common.token_metadata import (
        decode_generation_result,
    )
    from rlinf.utils.http_client import InferenceHTTPClient

    client = InferenceHTTPClient("http://127.0.0.1:1")

    async def capture_request(path, body):
        assert path == "/generate"
        return body

    monkeypatch.setattr(client, "async_post", capture_request)
    body = asyncio.run(
        client.async_generate(
            input_ids=[1, 9, 9, 2],
            sampling_params={"max_new_tokens": 2},
            return_logprob=True,
            image_data=[b"image"],
            return_prompt_token_ids=True,
            mm_token_expansion_start_len=4,
        )
    )
    assert body["return_prompt_token_ids"] is True
    assert body["mm_token_expansion_start_len"] == 4
    assert body["image_data"] == ["aW1hZ2U="]
    result = decode_generation_result(
        {
            "output_ids": [3],
            "prompt_token_ids": body["input_ids"],
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 4,
                "output_token_logprobs": [(-0.5, 3, None)],
            },
        },
        return_logprobs=True,
    )
    assert result["prompt_token_ids"] == [1, 9, 9, 2]
    assert result["logprobs"] == [-0.5]
    legacy = InferenceHTTPClient._generate_body(None, [1, 2], {}, False)
    assert "return_prompt_token_ids" not in legacy
    assert "mm_token_expansion_start_len" not in legacy


@pytest.mark.parametrize("namespace_package", [False, True])
def test_bundled_cudart_preserves_library_search_paths(
    tmp_path, monkeypatch, namespace_package, sglang_args_modules
):
    _pin_bundled_cudart = sglang_args_modules[2]._pin_bundled_cudart

    # Isolate the installed wheel layout, including namespace-only CUDA wheels.
    nvidia = ModuleType("nvidia")
    runtime = ModuleType("nvidia.cuda_runtime")
    runtime.__path__ = [str(tmp_path)]
    runtime.__file__ = None if namespace_package else str(tmp_path / "__init__.py")
    nvidia.cuda_runtime = runtime
    monkeypatch.setitem(sys.modules, "nvidia", nvidia)
    monkeypatch.setitem(sys.modules, "nvidia.cuda_runtime", runtime)
    library = tmp_path / "lib"
    library.mkdir()
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/existing:{library}:/another")
    _pin_bundled_cudart()
    assert os.environ["LD_LIBRARY_PATH"] == f"{library}:/existing:/another"


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
