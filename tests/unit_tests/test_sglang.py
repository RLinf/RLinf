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

"""SGLang request metadata, image capture, and service lifecycle."""

import asyncio
import importlib.util
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
    worker._cfg_rollout.sglang.capture_multi_modal_inputs = True
    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    args = SGLangWorker._build_server_args(worker)
    assert (
        os.environ["SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"]
        == "rlinf.hybrid_engines.sglang.multimodal"
    )
    assert args["cuda_graph_max_bs"] == 2
    assert args["load_format"] == "auto"
    assert "capture_multi_modal_inputs" not in args

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
        (None, "0.5.19", None),
        ("rlinf.hybrid_engines.sglang.multimodal", "0.5.12.post1", None),
        ("other.external.pkg", "0.5.12.post1", "SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE"),
        (None, "0.5.4", "0.5.12.post1"),
    ],
)
def test_ensure_image_capture_env_validates_version_and_registers_processor(
    monkeypatch, env_package, sglang_ver, expected_error
):
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        EXTERNAL_MM_PROCESSOR_PACKAGE,
        ensure_image_capture_env,
    )

    monkeypatch.delenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", raising=False)
    if env_package is not None:
        monkeypatch.setenv("SGLANG_EXTERNAL_MM_PROCESSOR_PACKAGE", env_package)
    if expected_error is not None:
        with pytest.raises(ValueError, match=expected_error):
            ensure_image_capture_env(sglang_ver)
        return
    ensure_image_capture_env(sglang_ver)
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
                "sglang": {"capture_multi_modal_inputs": True},
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
    pytest.importorskip("sglang")
    from transformers import AutoConfig, AutoProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import TokenizedQwenVLProcessor

    model_path = os.environ.get(
        "RLINF_TEST_QWEN_VL_MODEL"
    ) or _make_synthetic_qwen35_model_dir(tmp_path / "qwen35-synth")
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


def test_qwen_image_processor_keeps_text_request_behavior(qwen_image_processor):
    from PIL import Image

    processor, hf, _ = qwen_image_processor
    images = [Image.new("RGB", (64, 64), "red")]
    text = "A<|vision_start|><|image_pad|><|vision_end|>\n\nB"
    expected = hf(text=[text], images=images, return_tensors="pt")
    request = SimpleNamespace(
        video_data=None, audio_data=None, rid="text-request", sampling_params={}
    )
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


def test_qwen_vl_token_budget_enforces_reserve_before_clamping(qwen_image_processor):
    from tempfile import TemporaryDirectory
    from uuid import uuid4

    from PIL import Image

    from rlinf.hybrid_engines.sglang.multimodal.replay import CAPTURE_KEY

    processor, _, config = qwen_image_processor
    start, pad, end = (
        config.vision_start_token_id,
        config.image_token_id,
        config.vision_end_token_id,
    )
    images = [Image.new("RGB", (64, 64), "red")]
    ids = [start, pad, end]

    def _run(context_length, reserve, sampling_params):
        with TemporaryDirectory(prefix="rlinf-qwen-") as directory:
            ticket, rid = Path(directory).name, uuid4().hex
            request = SimpleNamespace(
                video_data=None,
                audio_data=None,
                rid=rid,
                sampling_params={
                    **sampling_params,
                    "custom_params": {
                        CAPTURE_KEY: {
                            "ticket": ticket,
                            "image_spans": [[1, 2]],
                            "prefix_length": 0,
                            "token_budget": (context_length, reserve),
                        }
                    },
                },
            )
            asyncio.run(processor.process_mm_data_async(images, ids, request))
            return request.sampling_params["max_new_tokens"]

    # First expand once to know exact expanded prompt length.
    unconstrained = _run(10000, 1, {})
    expanded_len = 10000 - unconstrained

    with pytest.raises(ValueError, match="expected at least 16"):
        _run(expanded_len + 8, 16, {"max_new_tokens": 64})

    # requested < reserve <= remaining clamps to requested without raising.
    assert _run(expanded_len + 32, 16, {"max_new_tokens": 8}) == 8
    # Omitted max_new_tokens uses remaining.
    assert _run(expanded_len + 32, 16, {}) == 32


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


def test_qwen_vl_module_imports_without_sglang_05():
    """Older SGLang builds (e.g. 0.4.6.post5 in package-build-test) can import qwen_vl."""
    script = dedent(
        """
        import sys
        sys.modules["sglang.srt.models.qwen3_5"] = None
        import rlinf.hybrid_engines.sglang.multimodal.qwen_vl as mod
        assert mod._HAS_QWEN35_PROCESSOR is False
        assert not hasattr(mod, "TokenizedQwenVLProcessor")
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)


def test_release_capture_checks_sampled_history_and_request_id():
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        generate_with_capture,
        save_capture,
    )

    rid = "a" * 32
    pixels = torch.ones(4, 2)

    async def generate(input_ids, image_data, sampling_params, return_logprob):
        capture = sampling_params["custom_params"][CAPTURE_KEY]
        output = SimpleNamespace(
            input_ids=input_ids,
            mrope_positions=torch.arange(len(input_ids)).repeat(3, 1),
            mrope_position_delta=0,
            mm_items=[
                SimpleNamespace(
                    offsets=[(1, 1)],
                    feature=pixels,
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                )
            ],
        )
        save_capture(capture_file(capture["ticket"], rid), output)
        return {"meta_info": {"id": rid}, "output_ids": [23]}

    def sample(ids, history=None):
        return asyncio.run(
            generate_with_capture(
                generate,
                ids,
                ["image"],
                {},
                {"image_spans": [[1, 2]], "history": history},
                False,
            )
        )

    _, _, history = sample([10, 11, 12])
    _, row, _ = sample([10, 11, 12, 23, 24], history)
    assert row["prompt_token_ids"] == [10, 11, 12, 23, 24]
    with pytest.raises(ValueError, match="sampled token history"):
        sample([10, 11, 12, 99, 24], history)
    pixels.zero_()
    with pytest.raises(ValueError, match="pixels"):
        sample([10, 11, 12, 23, 24], history)
    for bad_ticket, bad_rid in (
        ("../escape", rid),
        ("rlinf-qwen-12345678", "../escape"),
        ("rlinf-qwen-12345678", "z" * 32),
    ):
        with pytest.raises(
            ValueError, match="Invalid SGLang capture ticket or request ID"
        ):
            capture_file(bad_ticket, bad_rid)


@pytest.mark.parametrize(
    "spans,images",
    [
        ([], ["image"]),
        ([[1, 2]], []),
        ([[1, 2]], None),
        ([[1, 2], [3, 4]], ["image"]),
    ],
)
def test_release_capture_rejects_empty_or_mismatched_spans_before_generate(
    spans, images
):
    from rlinf.hybrid_engines.sglang.multimodal.replay import generate_with_capture

    called = 0

    async def generate(**kwargs):
        nonlocal called
        called += 1
        return {}

    with pytest.raises(ValueError, match="non-empty image_spans and images"):
        asyncio.run(
            generate_with_capture(
                generate,
                [10, 11, 12],
                images,
                {},
                {"image_spans": spans, "history": None},
                False,
            )
        )
    assert called == 0


@pytest.mark.parametrize(
    ("mode", "allowed"),
    [
        (0o700, True),
        (0o2700, True),
        (0o755, False),
        (0o4700, False),
        (0o1700, False),
    ],
)
def test_save_capture_allows_setgid_and_rejects_unsafe_modes(
    tmp_path, monkeypatch, mode, allowed
):
    import stat

    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        load_capture,
        save_capture,
    )

    capture_dir = tmp_path / "rlinf-qwen-12345678"
    capture_dir.mkdir(mode=0o700)
    real_fstat = os.fstat
    target_ino = os.stat(capture_dir).st_ino

    def fake_fstat(fd):
        real = real_fstat(fd)
        if real.st_ino != target_ino:
            return real
        return os.stat_result((stat.S_IFDIR | mode, *real[1:]))

    monkeypatch.setattr(os, "fstat", fake_fstat)
    output = SimpleNamespace(
        input_ids=[1, 2, 3],
        mrope_positions=torch.arange(3).repeat(3, 1),
        mrope_position_delta=0,
        mm_items=[
            SimpleNamespace(
                offsets=[(1, 1)],
                feature=torch.ones(4, 2),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
            )
        ],
    )
    target = capture_dir / ("b" * 32 + ".safetensors")
    if allowed:
        save_capture(target, output)
        loaded = load_capture(target)
        assert loaded["prompt_token_ids"] == [1, 2, 3]
    else:
        with pytest.raises(ValueError, match="owner or permissions"):
            save_capture(target, output)
        with pytest.raises(ValueError, match="SGLang did not write image capture file"):
            load_capture(target)


def test_tokenized_qwen_vl_processor_place_images_and_process_mm_data_offline(
    monkeypatch,
):
    import re
    from dataclasses import dataclass
    from tempfile import TemporaryDirectory

    pytest.importorskip("sglang.srt.models.qwen3_5")
    from sglang.srt.multimodal.processors.qwen_vl import QwenVLImageProcessor

    from rlinf.hybrid_engines.sglang.multimodal.qwen_vl import TokenizedQwenVLProcessor
    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        load_capture,
    )

    @dataclass
    class FakeItem:
        offsets: list[tuple[int, int]]
        feature: torch.Tensor
        image_grid_thw: torch.Tensor

    @dataclass
    class FakeNative:
        input_ids: list[int]
        mm_items: list[FakeItem]
        mrope_positions: torch.Tensor | None = None
        mrope_position_delta: torch.Tensor | None = None

    processor = TokenizedQwenVLProcessor.__new__(TokenizedQwenVLProcessor)
    processor.vision_start_token_id = 1
    processor.IM_TOKEN_ID = 2
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

    # Turn 1 (prefix=0): real span [1, 2) expands to 4 pads; literal [1, 2, 3] stays text.
    ids = [1, 2, 3, 10, 1, 2, 3, 11]
    native = FakeNative(
        input_ids=[1, 2, 2, 2, 2, 3],
        mm_items=[
            FakeItem(
                offsets=[(1, 4)],
                feature=torch.ones(16, 4),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
            )
        ],
    )
    placed = processor._place_images(ids, [[1, 2]], prefix=0, native=native)
    assert placed.input_ids == [1, 2, 2, 2, 2, 3, 10, 1, 2, 3, 11]
    assert placed.mm_items[0].offsets == [(1, 4)]
    assert placed.mrope_positions.shape == (3, len(placed.input_ids))
    # Literal [1, 2, 3] at indices 7..9 advances monotonically as plain text.
    assert placed.mrope_positions[:, 6:].tolist() == [[4, 5, 6, 7, 8]] * 3

    # Turn 2 (prefix > 0): keep expanded historical span [1, 5) and expand new [12, 13).
    turn2_ids = [*placed.input_ids, 1, 2, 3, 12]
    native_two = FakeNative(
        input_ids=[1, 2, 2, 2, 2, 3, 1, 2, 2, 2, 2, 3],
        mm_items=[
            FakeItem(
                offsets=[(1, 4)],
                feature=torch.ones(16, 4),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
            ),
            FakeItem(
                offsets=[(7, 10)],
                feature=torch.full((16, 4), 2.0),
                image_grid_thw=torch.tensor([[1, 4, 4]]),
            ),
        ],
    )
    placed_turn2 = processor._place_images(
        turn2_ids, [[1, 5], [12, 13]], prefix=len(placed.input_ids), native=native_two
    )
    assert placed_turn2.mm_items[0].offsets == [(1, 4)]
    assert placed_turn2.mm_items[1].offsets == [(12, 15)]
    assert torch.equal(
        placed_turn2.mrope_positions[:, : len(placed.input_ids)],
        placed.mrope_positions,
    )

    with pytest.raises(ValueError, match="does not match"):
        processor._place_images([1, 9, 3], [[1, 2]], prefix=0, native=native)
    # Historical span straddling prefix or mismatched width must be rejected.
    with pytest.raises(ValueError, match="recorded span"):
        processor._place_images(
            turn2_ids, [[1, 5], [12, 13]], prefix=3, native=native_two
        )
    with pytest.raises(ValueError, match="recorded span"):
        processor._place_images(
            [1, 2, 2, 3, 10, 1, 2, 3], [[1, 3], [6, 7]], prefix=5, native=native_two
        )
    with pytest.raises(ValueError, match="Invalid image history"):
        processor._place_images(ids, [[1, 2]], prefix=len(ids) + 1, native=native)

    # Exercise process_mm_data_async offline (non-capture delegation, capture handoff, errors).
    parent_async_calls = []

    async def fake_parent_async(self, image_data, input_text, request_obj, **kwargs):
        parent_async_calls.append((image_data, input_text))
        if (getattr(request_obj, "sampling_params", None) or {}).get("custom_params"):
            return native
        return "delegated-native"

    monkeypatch.setattr(
        QwenVLImageProcessor, "process_mm_data_async", fake_parent_async
    )
    non_capture_req = SimpleNamespace(
        video_data=None, audio_data=None, rid="plain", sampling_params={}
    )
    assert (
        asyncio.run(processor.process_mm_data_async([b"img"], ids, non_capture_req))
        == "delegated-native"
    )

    with TemporaryDirectory(prefix="rlinf-qwen-") as directory:
        ticket, rid = Path(directory).name, "c" * 32
        capture_req = SimpleNamespace(
            video_data=None,
            audio_data=None,
            rid=rid,
            sampling_params={
                "custom_params": {
                    CAPTURE_KEY: {
                        "ticket": ticket,
                        "image_spans": [[1, 2]],
                        "prefix_length": 0,
                    }
                }
            },
        )
        captured_out = asyncio.run(
            processor.process_mm_data_async([b"img"], ids, capture_req)
        )
        assert (
            parent_async_calls[-1][1] == "<|vision_start|><|image_pad|><|vision_end|>"
        )
        saved = load_capture(capture_file(ticket, rid))
        assert saved["prompt_token_ids"] == captured_out.input_ids
        assert saved["image_spans"] == [[1, 5]]
        assert torch.equal(saved["mrope_positions"], captured_out.mrope_positions)

        with pytest.raises(ValueError, match="token-ID prompt and raw images"):
            asyncio.run(
                processor.process_mm_data_async([b"img"], "not-ids", capture_req)
            )
        with pytest.raises(ValueError, match="token-ID prompt and raw images"):
            asyncio.run(
                processor.process_mm_data_async([{"pixel_values": 1}], ids, capture_req)
            )
        with pytest.raises(ValueError, match="one span per image"):
            asyncio.run(
                processor.process_mm_data_async([b"img1", b"img2"], ids, capture_req)
            )


def test_sglang_worker_rollout_serverless_and_generate_and_send_multimodal():
    pytest.importorskip("sglang")
    import pickle

    from omegaconf import OmegaConf

    from rlinf.hybrid_engines.sglang.multimodal.replay import (
        CAPTURE_KEY,
        capture_file,
        save_capture,
    )
    from rlinf.workers.rollout.sglang.sglang_worker import SGLangWorker

    class _AsyncWork:
        def __init__(self, coro):
            self._coro = coro

        async def async_wait(self):
            return await self._coro

    class _AsyncChannel:
        def __init__(self):
            self._queue: asyncio.Queue = asyncio.Queue()
            self.outputs: dict[str, dict] = {}
            self._events: dict[str, asyncio.Event] = {}

        def enqueue(self, item: dict) -> None:
            self._queue.put_nowait(item)

        def get(self, async_op: bool = False):
            assert async_op
            return _AsyncWork(self._queue.get())

        def put(self, item: dict, key: str | None = None, async_op: bool = False):
            assert async_op

            async def _store():
                self.outputs[key] = item
                self._events.setdefault(key, asyncio.Event()).set()

            return _AsyncWork(_store())

        async def wait_output(self, key: str) -> dict:
            await self._events.setdefault(key, asyncio.Event()).wait()
            return self.outputs[key]

    rid = "c" * 32
    backend_calls = []

    class _FakeBackend:
        async def async_generate(
            self, input_ids, sampling_params, image_data=None, return_logprob=False
        ):
            backend_calls.append(
                {
                    "input_ids": input_ids,
                    "sampling_params": sampling_params,
                    "image_data": image_data,
                    "return_logprob": return_logprob,
                }
            )
            capture = (sampling_params.get("custom_params") or {}).get(CAPTURE_KEY)
            if capture is not None:
                output = SimpleNamespace(
                    input_ids=list(input_ids),
                    mrope_positions=torch.arange(len(input_ids)).repeat(3, 1),
                    mrope_position_delta=0,
                    mm_items=[
                        SimpleNamespace(
                            offsets=[(1, 1)],
                            feature=torch.ones(4, 2),
                            image_grid_thw=torch.tensor([[1, 2, 2]]),
                        )
                    ],
                )
                save_capture(capture_file(capture["ticket"], rid), output)
            return {
                "output_ids": [42],
                "meta_info": {
                    "id": rid,
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
        worker._dispatch_gate = asyncio.Event()
        worker._dispatch_gate.set()
        worker._inflight_generates = set()
        worker.log_error = lambda msg: None
        worker._cfg_rollout = OmegaConf.create(
            {"sglang": {"capture_multi_modal_inputs": True}}
        )

        in_ch, out_ch = _AsyncChannel(), _AsyncChannel()
        loop_task = asyncio.create_task(worker.rollout_serverless(in_ch, out_ch))
        try:
            # 1. Captured multimodal generation through rollout_serverless.
            in_ch.enqueue(
                {
                    "channel_key": "k_capture",
                    "prompt_ids": [10, 11, 12],
                    "image_data": [b"png-bytes"],
                    "capture_images": {"image_spans": [[1, 2]], "history": None},
                }
            )
            captured_out = await out_ch.wait_output("k_capture")
            assert captured_out["output_ids"] == [42]
            assert captured_out["logprobs"] == [-0.25]
            assert captured_out["prompt_token_ids"] == [10, 11, 12]
            assert captured_out["image_history"]["length"] == 4
            # Verify pickle roundtrip does not alias prompt_token_ids with multi_modal_inputs.
            roundtripped = pickle.loads(pickle.dumps(captured_out))
            roundtripped["prompt_token_ids"] += [999]
            assert roundtripped["multi_modal_inputs"]["prompt_token_ids"] == [
                10,
                11,
                12,
            ]

            # 2. Non-capture forwarding: empty image_data normalizes to None, raw passes through.
            in_ch.enqueue(
                {
                    "channel_key": "k_empty_img",
                    "prompt_ids": [1, 2],
                    "image_data": [],
                }
            )
            await out_ch.wait_output("k_empty_img")
            assert backend_calls[-1]["image_data"] is None

            in_ch.enqueue(
                {
                    "channel_key": "k_raw_img",
                    "prompt_ids": [1, 2],
                    "image_data": [b"raw-img"],
                }
            )
            raw_out = await out_ch.wait_output("k_raw_img")
            assert backend_calls[-1]["image_data"] == [b"raw-img"]
            assert "multi_modal_inputs" not in raw_out

            # 3. Disabled capture delivers a clear error payload on output_channel.
            worker._cfg_rollout.sglang.capture_multi_modal_inputs = False
            in_ch.enqueue(
                {
                    "channel_key": "k_disabled",
                    "prompt_ids": [10, 11, 12],
                    "image_data": [b"png-bytes"],
                    "capture_images": {"image_spans": [[1, 2]], "history": None},
                }
            )
            err_out = await out_ch.wait_output("k_disabled")
            assert "capture_multi_modal_inputs" in err_out["error"]
        finally:
            loop_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await loop_task

    asyncio.run(scenario())


@pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None, reason="requires SGLang"
)
def test_patched_scheduler_delegates_to_vendor():
    # Applying the engine patch changes module globals throughout the interpreter.
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
