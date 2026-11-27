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

"""SGLang external span processor, HF image parity, and service lifecycle."""

import asyncio
import importlib.util
import io
import os
import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from types import SimpleNamespace

import pytest
import torch


def _sglang_test_worker(rollout_overrides: dict | None = None):
    from omegaconf import OmegaConf

    rollout = OmegaConf.create(
        {
            "enforce_eager": False,
            "validate_weight": False,
            "validate_weight_first_sync": True,
            "sync_mtp": False,
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
            },
            "cuda_graph_max_bs": 4,
            "max_running_requests": 2,
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.7,
            "detokenize": False,
            **(rollout_overrides or {}),
        }
    )
    return SimpleNamespace(
        _cfg=SimpleNamespace(runner=SimpleNamespace(resume_dir=None)),
        _cfg_rollout=rollout,
        _rank=0,
        weight_reload="sync",
        acquire_free_port=lambda: 12345,
        log_on_first_rank=lambda message: None,
        log_warning=lambda message: None,
    )


def test_sglang_server_args_with_installed_vendor(monkeypatch):
    pytest.importorskip("sglang.srt.server_args")
    model = os.environ.get("RLINF_TEST_SGLANG_MODEL")
    if not model:
        pytest.skip("Set RLINF_TEST_SGLANG_MODEL to a local model directory")
    if not torch.cuda.is_available():
        pytest.skip("The native SGLang Engine import requires a CUDA device")
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    worker = _sglang_test_worker()
    worker._cfg_rollout.model.model_path = model
    worker._cfg_rollout.enforce_eager = True
    worker._cfg_rollout.sglang.align_multimodal_inputs = True
    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    args = SGLangWorker._build_server_args(worker)
    assert (
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"]
        == "rlinf.hybrid_engines.sglang.multimodal"
    )
    assert args["cuda_graph_max_bs"] == 2 and args["load_format"] == "auto"
    assert "align_multimodal_inputs" not in args

    monkeypatch.setenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", "other.mm.pkg")
    with pytest.raises(ValueError, match="SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"):
        SGLangWorker._build_server_args(worker)


def test_sglang_server_args_forwards_parsers(monkeypatch):
    import dataclasses

    pytest.importorskip("sglang.srt.server_args")
    from rlinf.workers.rollout.sglang import sglang_worker

    @dataclasses.dataclass
    class _FakeServerArgs:
        reasoning_parser: str | None = None
        tool_call_parser: str | None = None
        skip_server_warmup: bool = False

        def __init__(self, **kwargs):
            self.reasoning_parser = kwargs.get("reasoning_parser")
            self.tool_call_parser = kwargs.get("tool_call_parser")
            self.skip_server_warmup = kwargs.get("skip_server_warmup", False)

    monkeypatch.setattr(sglang_worker, "ServerArgs", _FakeServerArgs)
    configured = _sglang_test_worker(
        {
            "sglang": {
                "use_torch_compile": False,
                "torch_compile_max_bs": 4,
                "decode_log_interval": 100,
                "attention_backend": "triton",
                "tool_call_parser": "qwen3_coder",
                "reasoning_parser": "qwen3",
            }
        }
    )
    args = sglang_worker.SGLangWorker._build_server_args(configured)
    assert args["tool_call_parser"] == "qwen3_coder"
    assert args["reasoning_parser"] == "qwen3"

    defaults = sglang_worker.SGLangWorker._build_server_args(_sglang_test_worker())
    assert defaults["tool_call_parser"] is None
    assert defaults["reasoning_parser"] is None


@pytest.mark.parametrize(
    ("env_package", "sglang_ver", "expected_error"),
    [
        (None, "0.5.12.post1", None),
        (None, "0.5.22", None),
        ("rlinf.hybrid_engines.sglang.multimodal", "0.5.12.post1", None),
        ("other.external.pkg", "0.5.12.post1", "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"),
        (None, "0.5.4", "0.5.12.post1"),
    ],
)
def test_ensure_multimodal_processor_env_validates_version_and_package(
    monkeypatch, env_package, sglang_ver, expected_error
):
    from rlinf.hybrid_engines.sglang.multimodal import (
        EXTERNAL_MM_PROCESSOR_PACKAGE,
        ensure_multimodal_processor_env,
    )

    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    if env_package is not None:
        monkeypatch.setenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", env_package)
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            ensure_multimodal_processor_env(sglang_ver)
        return
    ensure_multimodal_processor_env(sglang_ver)
    assert (
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"]
        == EXTERNAL_MM_PROCESSOR_PACKAGE
    )


def test_sglang_server_backend_injects_mm_processor_env():
    pytest.importorskip("sglang")
    from omegaconf import OmegaConf

    from rlinf.workers.rollout.sglang.backends import SGLangServerBackend

    cfg = OmegaConf.create(
        {
            "rollout": {
                "rollout_backend": "sglang",
                "sglang": {"align_multimodal_inputs": True},
            },
            "session_server": {"enable": True, "engine_keep_alive_s": 120},
        }
    )
    backend = SGLangServerBackend(
        server_args={"model_path": "/model"},
        rlinf_ctx=SimpleNamespace(cfg=cfg),
        acquire_free_port=lambda: 12345,
        log_info=lambda msg: None,
        log_error=lambda msg: None,
    )
    assert backend._build_server_extra_env(session_server_enabled=False) == {
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "rlinf.hybrid_engines.sglang.multimodal"
    }
    assert backend._build_server_extra_env(session_server_enabled=True) == {
        "SGLANG_TIMEOUT_KEEP_ALIVE": "120",
        "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE": "rlinf.hybrid_engines.sglang.multimodal",
    }


def test_pin_bundled_cudart_handles_namespace_package(tmp_path, monkeypatch):
    import types

    pytest.importorskip("sglang")
    from rlinf.workers.rollout.sglang.process import _pin_bundled_cudart

    lib_dir = tmp_path / "lib"
    lib_dir.mkdir()
    nvidia_pkg = types.ModuleType("nvidia")
    cuda_rt = types.ModuleType("nvidia.cuda_runtime")
    cuda_rt.__file__ = None
    cuda_rt.__path__ = [str(tmp_path), str(tmp_path / "missing")]
    nvidia_pkg.cuda_runtime = cuda_rt
    monkeypatch.setitem(sys.modules, "nvidia", nvidia_pkg)
    monkeypatch.setitem(sys.modules, "nvidia.cuda_runtime", cuda_rt)
    monkeypatch.setenv("LD_LIBRARY_PATH", f"/existing:{lib_dir}:/tail")

    _pin_bundled_cudart()
    assert os.environ["LD_LIBRARY_PATH"] == f"{lib_dir}:/existing:/tail"


def _make_synthetic_qwen35_model_dir(target_dir: Path) -> str:
    """Create a tiny weight-free Qwen3.5 config + processor dir for CI."""
    from tokenizers import Tokenizer, decoders
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen2VLVideoProcessor,
        Qwen3_5Config,
        Qwen3VLProcessor,
    )

    vocab = {
        "<unk>": 0,
        "<|vision_start|>": 1,
        "<|image_pad|>": 2,
        "<|vision_end|>": 3,
        "<|video_pad|>": 4,
        "A": 5,
        "B": 6,
    }
    tok_obj = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Whitespace()
    tok_obj.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        unk_token="<unk>",
        pad_token="<unk>",
        additional_special_tokens=[
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            "<|video_pad|>",
        ],
    )
    hf_processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=tokenizer,
        video_processor=Qwen2VLVideoProcessor(),
    )
    hf_processor.save_pretrained(target_dir)
    Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        vision_start_token_id=1,
        image_token_id=2,
        vision_end_token_id=3,
        video_token_id=4,
    ).save_pretrained(target_dir)
    return str(target_dir)


@pytest.fixture
def qwen_image_processor(tmp_path, monkeypatch):
    """Real SGLang/HF preprocessing using a local model or tiny CI config."""
    pytest.importorskip("sglang.srt.multimodal.processors.qwen_vl")
    from transformers import AutoConfig, AutoProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import AlignedQwenVLProcessor

    model_path = os.environ.get("RLINF_QWEN35_MODEL") or os.environ.get(
        "RLINF_TEST_QWEN_VL_MODEL"
    )
    if not model_path:
        pytest.importorskip("transformers.models.qwen3_5.configuration_qwen3_5")
        model_path = _make_synthetic_qwen35_model_dir(tmp_path / "qwen35-synth")
    hf = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    server = SimpleNamespace(
        mm_process_config={},
        skip_tokenizer_init=True,
        disable_fast_image_processor=True,
        keep_mm_feature_on_device=False,
        tokenizer_worker_num=1,
    )
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
    processor = AlignedQwenVLProcessor(config, server, hf, "cpu", skip_mm_pool=True)
    try:
        yield processor, hf, config
    finally:
        processor.io_executor.shutdown()
        processor.cpu_executor.shutdown()


def test_aligned_qwen_processor_parity_and_span_isolation(qwen_image_processor):
    from PIL import Image

    from rlinf.hybrid_engines.sglang.multimodal import IMAGE_SPANS_PARAM
    from rlinf.utils.multimodal import process_single_image

    processor, hf, config = qwen_image_processor
    assert processor.gpu_image_decode is False
    start, pad, end = (
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    )

    # 1. Default requests without IMAGE_SPANS_PARAM remain unchanged.
    text = "A<|vision_start|><|image_pad|><|vision_end|>\n\nB"
    buf = io.BytesIO()
    Image.new("RGB", (96, 128), "red").save(buf, format="PNG")
    img_bytes = [buf.getvalue()]
    with Image.open(io.BytesIO(img_bytes[0])) as pil_img:
        expected = hf(text=[text], images=[pil_img.convert("RGB")], return_tensors="pt")
    plain_req = SimpleNamespace(
        video_data=None, audio_data=None, rid="plain", sampling_params={}
    )
    plain_out = asyncio.run(processor.process_mm_data_async(img_bytes, text, plain_req))
    assert plain_out.input_ids == expected.input_ids[0].tolist()

    # 2. Agent-side process_single_image matches SGLang pixels and grid bit-for-bit.
    buf2 = io.BytesIO()
    Image.new("RGB", (260, 64), "blue").save(buf2, format="PNG")
    img_bytes.append(buf2.getvalue())

    ids, spans, agent_pixels, agent_grids = [], [], [], []
    for raw in img_bytes:
        pixels, grid, width = process_single_image(raw, hf)
        agent_pixels.append(pixels)
        agent_grids.append(grid)
        # Inject literal visual markers between real turns to test span isolation.
        ids.extend([start, pad, end])
        img_start = len(ids) + 1
        ids.extend([start, *([pad] * width), end])
        spans.append([img_start, img_start + width])

    span_req = SimpleNamespace(
        video_data=None,
        audio_data=None,
        rid="aligned",
        sampling_params={"custom_params": {IMAGE_SPANS_PARAM: spans}},
    )
    aligned_out = asyncio.run(processor.process_mm_data_async(img_bytes, ids, span_req))
    assert aligned_out.input_ids == ids
    assert [it.offsets for it in aligned_out.mm_items] == [
        [(s, e - 1)] for s, e in spans
    ]
    for it, px, gr in zip(aligned_out.mm_items, agent_pixels, agent_grids, strict=True):
        assert torch.equal(it.feature.cpu(), px)
        assert torch.equal(it.image_grid_thw.cpu(), gr)


def test_aligned_qwen_vl_processor_offline_and_cross_version_structs(monkeypatch):
    msgspec = pytest.importorskip("msgspec")
    pytest.importorskip("sglang.srt.multimodal.processors.qwen_vl")
    import re

    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import (
        IMAGE_SPANS_PARAM,
        AlignedQwenVLProcessor,
    )

    class ProxyFeature:
        """Simulates SGLang 0.5.22+ CudaIpcTensorTransportProxy (not a torch.Tensor)."""

    class StructItem(msgspec.Struct):
        offsets: list[tuple[int, int]]
        feature: object
        image_grid_thw: torch.Tensor
        pad_value: int = 2
        model_specific_data: dict = msgspec.field(default_factory=dict)

    class StructNative(msgspec.Struct):
        input_ids: list[int]
        mm_items: list[StructItem]
        mrope_positions: torch.Tensor | None = None
        mrope_position_delta: torch.Tensor | None = None
        padded_input_ids: list[int] | None = None

        @staticmethod
        def build_padded_input_ids(input_ids, mm_items):
            return [999, len(input_ids), len(mm_items)]

    processor = AlignedQwenVLProcessor.__new__(AlignedQwenVLProcessor)
    processor.vision_start_token_id, processor.IM_TOKEN_ID = 1, 2
    processor.vision_end_token_id = 3
    processor.hf_config = SimpleNamespace(
        vision_config=SimpleNamespace(spatial_merge_size=2)
    )
    processor.model_type = "qwen3_5"
    processor.mm_tokens = SimpleNamespace(
        video_token_id=4,
        image_token="<|image_pad|>",
        image_token_regex=re.compile(r"(?:<\|image_pad\|>)+"),
    )
    processor._tokenizer = SimpleNamespace(
        decode=lambda toks: "".join(
            {1: "<|vision_start|>", 2: "<|image_pad|>", 3: "<|vision_end|>"}.get(
                t, f"t{t}"
            )
            for t in toks
        )
    )

    # Pre-expanded span [1, 5) + literal [1, 2, 3] at [6..9); item.feature is non-Tensor IPC proxy.
    ids = [1, 2, 2, 2, 2, 3, 10, 1, 2, 3, 11]
    cached_meta = {"tag": "shared"}
    native = StructNative(
        input_ids=[1, 2, 2, 2, 2, 3],
        mm_items=[
            StructItem(
                [(1, 4)],
                ProxyFeature(),
                torch.tensor([[1, 4, 4]]),
                model_specific_data=cached_meta,
            )
        ],
    )
    placed = processor._place_images(ids, [(1, 5)], native)
    assert isinstance(placed, StructNative)
    assert placed.input_ids == ids and placed.padded_input_ids == [999, 11, 1]
    assert placed.mm_items[0].offsets == [(1, 4)]
    assert placed.mm_items[0].model_specific_data is not cached_meta
    assert placed.mrope_positions[:, 6:].tolist() == [[4, 5, 6, 7, 8]] * 3

    for bad_ids, bad_spans, match in [
        (ids, [(1, 3)], "patch width"),
        (ids, [(1, 5), (7, 8)], "count mismatch"),
    ]:
        with pytest.raises(ValueError, match=match):
            processor._place_images(bad_ids, bad_spans, native)

    seen_texts = []

    async def fake_parent_async(self, image_data, input_text, request_obj, **kwargs):
        seen_texts.append(input_text)
        return native

    monkeypatch.setattr(
        QwenVLImageProcessor, "process_mm_data_async", fake_parent_async
    )
    req = SimpleNamespace(
        video_data=None,
        audio_data=None,
        rid="offline",
        sampling_params={"custom_params": {IMAGE_SPANS_PARAM: [[1, 5]]}},
    )
    out = asyncio.run(processor.process_mm_data_async([b"img"], ids, req))
    assert seen_texts[-1] == "<|vision_start|><|image_pad|><|vision_end|>"
    assert out.input_ids == ids
    for bad_img, bad_ids, bad_spans, match in [
        ([b"img"], "not-ids", [[1, 5]], "token-ID prompt"),
        ([{"pixel_values": 1}], ids, [[1, 5]], "token-ID prompt"),
        ([b"img1", b"img2"], ids, [[1, 5]], "one explicit image span per image"),
        ([b"img"], ids, [[7, 8]], "does not match"),
    ]:
        bad_req = SimpleNamespace(
            video_data=None,
            audio_data=None,
            rid="bad",
            sampling_params={"custom_params": {IMAGE_SPANS_PARAM: bad_spans}},
        )
        with pytest.raises(ValueError, match=match):
            asyncio.run(processor.process_mm_data_async(bad_img, bad_ids, bad_req))


def test_qwen_vl_module_imports_without_sglang_qwen_vl():
    """Older SGLang builds can still import qwen_vl cleanly."""
    script = dedent(
        """
        import sys
        sys.modules["sglang.srt.multimodal.processors.qwen_vl"] = None
        import rlinf.hybrid_engines.sglang.multimodal.qwen_vl as mod
        assert mod._HAS_QWEN_VL_PROCESSOR is False
        assert not hasattr(mod, "AlignedQwenVLProcessor")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_sglang_worker_generate_and_send_multimodal():
    pytest.importorskip("sglang")
    from omegaconf import OmegaConf

    from rlinf.hybrid_engines.sglang.multimodal import IMAGE_SPANS_PARAM
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    class _OutChannel:
        def __init__(self):
            self.outputs = {}

        def put(self, item: dict, key: str | None = None, async_op: bool = False):
            self.outputs[key] = item

            async def _noop():
                return None

            return SimpleNamespace(async_wait=_noop)

    backend_calls = []

    class _FakeBackend:
        async def async_generate(
            self, input_ids, sampling_params, image_data=None, return_logprob=False
        ):
            backend_calls.append(
                {"image_data": image_data, "sampling_params": sampling_params}
            )
            return {
                "output_ids": [42],
                "meta_info": {
                    "id": "a" * 32,
                    "finish_reason": {"type": "stop"},
                    "output_token_logprobs": [(-0.25, 42)],
                },
            }

    async def scenario():
        worker = SGLangWorker.__new__(SGLangWorker)
        worker._is_model_instance_entry = True
        worker._sampling_params = {"temperature": 0.0}
        worker._weight_version = 3
        worker._return_logprobs = True
        worker._backend = _FakeBackend()
        worker.log_error = lambda msg: None
        worker._cfg_rollout = OmegaConf.create(
            {"sglang": {"align_multimodal_inputs": True}}
        )
        out_ch = _OutChannel()

        # 1. Aligned image request attaches IMAGE_SPANS_PARAM and returns lean token/logprob payload.
        await worker.generate_and_send(
            out_ch, "k_ok", [10, 11, 12], image_data=[b"png"], image_spans=[[1, 2]]
        )
        assert backend_calls[-1]["sampling_params"]["custom_params"] == {
            IMAGE_SPANS_PARAM: [[1, 2]]
        }
        assert out_ch.outputs["k_ok"] == {
            "output_ids": [42],
            "finish_reason": "stop",
            "version_start": 3,
            "version_end": 3,
            "logprobs": [-0.25],
        }

        # 2. Align enabled: empty image_data -> None; raw image_data without spans -> rejected.
        await worker.generate_and_send(out_ch, "k_empty", [1, 2], image_data=[])
        assert backend_calls[-1]["image_data"] is None

        calls_before = len(backend_calls)
        await worker.generate_and_send(out_ch, "k_bad", [1, 2], image_data=[b"raw"])
        assert "image_spans" in out_ch.outputs["k_bad"]["error"]
        assert len(backend_calls) == calls_before

        # 3. Align disabled: raw image_data passes through; image_spans requests fail.
        worker._cfg_rollout.sglang.align_multimodal_inputs = False
        await worker.generate_and_send(out_ch, "k_raw", [1, 2], image_data=[b"raw"])
        assert backend_calls[-1]["image_data"] == [b"raw"]

        await worker.generate_and_send(
            out_ch, "k_off", [10, 11, 12], image_data=[b"png"], image_spans=[[1, 2]]
        )
        assert "align_multimodal_inputs" in out_ch.outputs["k_off"]["error"]

    asyncio.run(scenario())


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="requires SGLang"
)
def test_patched_scheduler_delegates_to_vendor():
    script = dedent(
        """
        import sglang.srt.managers.scheduler as vendor

        def check():
            calls = []
            vendor.run_scheduler_process = lambda *args, **kwargs: calls.append((args, kwargs))
            from rlinf.hybrid_engines.sglang.common import sgl_engine, sgl_scheduler
            from rlinf.utils.patcher import Patcher

            assert vendor.run_scheduler_process is sgl_scheduler.run_scheduler_process
            Patcher.apply = staticmethod(lambda: None)
            vendor.run_scheduler_process("request", rank=0)
            assert calls == [(("request",), {"rank": 0})]

        check()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="requires SGLang"
)
def test_tokenizer_manager_handles_aborted_none_versions_in_mixed_batch(monkeypatch):
    from rlinf.hybrid_engines.sglang.common.tokenizer_manager import TokenizerManager

    seen = []
    monkeypatch.setattr(
        TokenizerManager.__mro__[1],
        "_handle_batch_output",
        lambda self, obj: seen.append(obj),
    )
    tm = object.__new__(TokenizerManager)
    tm.patch_return_output_ids = False
    active_state = SimpleNamespace(rlinf_token_versions=[1])
    tm.rid_to_state = {"ok": active_state, "aborted": None}
    recv_obj = SimpleNamespace(
        rids=["aborted", "ok"],
        customized_info={"rollout_weight_versions": [None, [2, 3]]},
    )
    tm._handle_batch_output(recv_obj)
    assert active_state.rlinf_token_versions == [1, 2, 3]
    assert recv_obj.customized_info["rollout_weight_versions"] == [None, [1, 2, 3]]
    assert seen == [recv_obj]
