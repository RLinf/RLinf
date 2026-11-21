# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resident Stage1 VLA (pi05 + RLT token module) for Tianji Stage2.

One process holds the frozen ``rollout.rlt_feature_model`` of the Stage2
config on GPU0 and serves ``OpenPiPytorchEvalActionModel.extract_vla_reference``
(openpi preprocessing, prefix, RLT token, RTC-guided flow sampling,
un-normalization) over a Unix socket in ``$XDG_RUNTIME_DIR``. Stage2's rollout
worker (``pi05_client.py``, enabled by ``TIANJI_PI05_SERVICE=1``) sends the VLA
observation keys and the flow noise it drew itself, and gets ``z_rl`` /
``ref_chunk`` back; the small RL actor stays in Stage2. A Stage2 restart no
longer rebuilds the 4.1B-parameter model.

Weights: ``full_weights.bf16.safetensors`` next to ``full_weights.pt``
(one-time ``launcher/convert_pi05_bf16.py``), read in bounded chunks straight
onto the GPU into a model built on the meta device: no fp32 CPU copy, no page
cache flood, no 13 GB RAM spike.

Identity: the service reports the model path, weight file (size, mtime,
sha256 from the converter's sidecar), norm-stats hash, resolved feature-model
config hash and a hash of the source files on its inference path
(``INFERENCE_CODE_FILES`` + the openpi modules below); Stage2 computes the
same from its own tree and refuses a service that differs.

CLI (``python -m rlinf.models.embodiment.openpi_rlinf.pi05_service``):
  serve     load, capture the TIANJI_FAST_INFER=1 CUDA graphs, self-check
            graph vs eager bitwise, then serve (one request at a time);
  info      JSON state of the running service;
  check     exit 0 = running, ready and identical to this tree/config;
            3 = no service, 4 = mismatch (reasons printed), 5 = still loading,
            6 = failed;
  identity  the identity this tree/config expects;
  wait      block until ready and matching (or --timeout).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import socket
import struct
import sys
import threading
import time
from pathlib import Path
from typing import Any

PROTOCOL = "tianji.pi05_service.v1"
UNIT_NAME = "tianji-pi05"
SOCKET_NAME = "tianji-pi05.sock"
BF16_WEIGHTS_NAME = "full_weights.bf16.safetensors"
SIDECAR_SUFFIX = ".json"
DEFAULT_CONFIG = "tianji2_rlt_stage2_async"

# Observation keys the VLA path reads (eval_action_model: _repack_env_obs,
# _rtc_previous_joints). Everything else of the env observation (qvel,
# tactile, ...) stays in Stage2.
VLA_OBS_KEYS = (
    "main_images",
    "wrist_images",
    "extra_view_images",
    "joint_states",
    "states",
    "task_descriptions",
    "rtc_prev_actions",
    "rtc_prev_valid",
)

RLINF_ROOT = Path(__file__).resolve().parents[4]

# Source files of the inference path, relative to the RLinf root: this module
# and its client, the fast path, the eval wrapper (preprocessing, RTC prefix
# guidance, flow sampling), the Pi0 core, the RLT token module, RTC guidance,
# the Tianji openpi data config (normalization, image/prompt layout) and the
# contract / kinematics helpers it imports.
INFERENCE_CODE_FILES = (
    "rlinf/models/embodiment/openpi_rlinf/pi05_service.py",
    "rlinf/models/embodiment/openpi_rlinf/pi05_client.py",
    "rlinf/models/embodiment/openpi_rlinf/fast_infer.py",
    "rlinf/models/embodiment/openpi_rlinf/eval_action_model.py",
    "rlinf/models/embodiment/openpi_rlinf/openpi_action_model.py",
    "rlinf/models/embodiment/openpi_rlinf/transforms_pipeline.py",
    "rlinf/models/embodiment/openpi_rlinf/__init__.py",
    "rlinf/models/embodiment/openpi_rlinf/utils/rlt_utils.py",
    "rlinf/models/embodiment/openpi_rlinf/utils/model_builders.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/__init__.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/pi0.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/pi0_config.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/model.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/gemma.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/siglip.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/utils.py",
    "rlinf/models/embodiment/openpi_rlinf/pi0_model/lora.py",
    "rlinf/models/embodiment/modules/rlt_token_transformer.py",
    "rlinf/models/embodiment/openpi/rtc_guidance.py",
    "rlinf/models/embodiment/openpi/dataconfig/__init__.py",
    "rlinf/models/embodiment/openpi/dataconfig/tianji_dataconfig.py",
    "rlinf/envs/realworld/tianji/tianji_contract.py",
    "rlinf/envs/realworld/tianji/tianji_kinematics.py",
    "rlinf/utils/ckpt_convertor/openpi/_core.py",
)
# openpi package files on the same path (tokenizer, normalization, image
# resize, transform primitives, model/data config), relative to the installed
# ``openpi`` package directory; located without importing the submodules.
OPENPI_CODE_FILES = (
    "transforms.py",
    "models/tokenizer.py",
    "models/model.py",
    "shared/normalize.py",
    "shared/image_tools.py",
    "training/config.py",
)

IDENTITY_KEYS = (
    "protocol",
    "model_path",
    "weights_file",
    "weights_size",
    "weights_mtime",
    "weights_sha256",
    "norm_stats_path",
    "norm_stats_sha256",
    "model_cfg_sha256",
    "code_sha256",
)

_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "I64": "int64",
    "I32": "int32",
    "I16": "int16",
    "I8": "int8",
    "U8": "uint8",
    "BOOL": "bool",
}


def log(*parts: Any) -> None:
    print(time.strftime("%H:%M:%S"), "pi05_service:", *parts, flush=True)


# ----------------------------------------------------------------- paths


def socket_path() -> Path:
    explicit = os.environ.get("TIANJI_PI05_SOCKET")
    if explicit:
        return Path(explicit)
    runtime = os.environ.get("XDG_RUNTIME_DIR") or f"/run/user/{os.getuid()}"
    return Path(runtime) / SOCKET_NAME


def bf16_weights_path(model_path: Any) -> Path:
    """``full_weights.bf16.safetensors`` beside the checkpoint's full_weights.pt."""
    from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import resolve_full_weights

    path = Path(model_path).expanduser()
    if path.is_file() and path.name.endswith(".safetensors"):
        return path
    full = resolve_full_weights(path)
    if full is None:
        raise FileNotFoundError(f"no full_weights.pt under {path}")
    return full.parent / BF16_WEIGHTS_NAME


def sidecar_path(weights: Path) -> Path:
    return Path(str(weights) + SIDECAR_SUFFIX)


# -------------------------------------------------------------- identity


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def openpi_package_dir() -> Path | None:
    try:
        spec = importlib.util.find_spec("openpi")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return Path(list(spec.submodule_search_locations)[0])


def code_sha256(root: Path = RLINF_ROOT) -> tuple[str, list[str]]:
    """Hash of (relative path, bytes) over INFERENCE_CODE_FILES and the openpi
    files; the file list is returned for the info response."""
    h = hashlib.sha256()
    listed = []
    for rel in INFERENCE_CODE_FILES:
        path = root / rel
        h.update(rel.encode())
        if path.is_file():
            h.update(path.read_bytes())
            listed.append(rel)
        else:
            h.update(b"<missing>")
            listed.append(rel + " (missing)")
    openpi_dir = openpi_package_dir()
    for rel in OPENPI_CODE_FILES:
        name = "openpi/" + rel
        h.update(name.encode())
        path = openpi_dir / rel if openpi_dir is not None else None
        if path is not None and path.is_file():
            h.update(path.read_bytes())
            listed.append(name)
        else:
            h.update(b"<missing>")
            listed.append(name + " (missing)")
    return h.hexdigest(), listed


def model_cfg_sha256(feature_cfg: Any) -> str:
    from omegaconf import OmegaConf

    container = OmegaConf.to_container(feature_cfg, resolve=True)
    return hashlib.sha256(json.dumps(container, sort_keys=True, default=str).encode()).hexdigest()


def read_sidecar(weights: Path) -> dict | None:
    try:
        return json.loads(sidecar_path(weights).read_text())
    except (OSError, ValueError):
        return None


def _norm_stats_path(feature_cfg: Any) -> Path:
    from omegaconf import OmegaConf

    explicit = OmegaConf.select(feature_cfg, "openpi_data.norm_stats_path", default=None)
    if explicit:
        return Path(str(explicit)).expanduser().resolve()
    return Path(str(feature_cfg.model_path)).expanduser().resolve() / "norm_stats.json"


def local_identity(feature_cfg: Any, *, root: Path = RLINF_ROOT) -> dict:
    """What this tree and config expect of the service (no model is built).

    The weight hash comes from the converter's sidecar when its size/mtime
    still match the file; otherwise it is reported as None (mismatch).
    """
    model_path = Path(str(feature_cfg.model_path)).expanduser().resolve()
    weights = bf16_weights_path(model_path)
    stats = _norm_stats_path(feature_cfg)
    identity = {
        "protocol": PROTOCOL,
        "model_path": str(model_path),
        "weights_file": str(weights),
        "weights_size": None,
        "weights_mtime": None,
        "weights_sha256": None,
        "norm_stats_path": str(stats),
        "norm_stats_sha256": sha256_file(stats) if stats.is_file() else None,
        "model_cfg_sha256": model_cfg_sha256(feature_cfg),
    }
    if weights.is_file():
        st = weights.stat()
        identity["weights_size"] = st.st_size
        identity["weights_mtime"] = st.st_mtime
        side = read_sidecar(weights)
        if side and side.get("size") == st.st_size and side.get("mtime") == st.st_mtime:
            identity["weights_sha256"] = side.get("sha256")
    identity["code_sha256"], identity["code_files"] = code_sha256(root)
    return identity


def identity_mismatch(expected: dict, actual: dict) -> list[str]:
    reasons = []
    for key in IDENTITY_KEYS:
        if expected.get(key) != actual.get(key):
            reasons.append(f"{key}: expected {expected.get(key)!r}, service has {actual.get(key)!r}")
    return reasons


def feature_model_config(config_name: str = DEFAULT_CONFIG) -> Any:
    """``rollout.rlt_feature_model`` of the Stage2 Hydra config, resolved in
    this environment (the rollout worker receives the same node)."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf

    directory = RLINF_ROOT / "examples/embodiment/config"
    os.environ.setdefault("EMBODIED_PATH", str(RLINF_ROOT / "examples/embodiment"))
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
        cfg = compose(config_name=config_name)
    node = OmegaConf.select(cfg, "rollout.rlt_feature_model")
    if node is None:
        raise RuntimeError(f"{config_name} has no rollout.rlt_feature_model")
    return OmegaConf.create(OmegaConf.to_container(node, resolve=True))


# --------------------------------------------------------------- protocol


def send_msg(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=5)
    sock.sendall(struct.pack("!Q", len(data)) + data)


def recv_msg(sock: socket.socket) -> Any:
    header = _recv_exact(sock, 8)
    if header is None:
        return None
    (length,) = struct.unpack("!Q", header)
    data = _recv_exact(sock, length)
    if data is None:
        return None
    return pickle.loads(data)


def _recv_exact(sock: socket.socket, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


def request(obj: Any, *, path: Path | None = None, timeout: float = 10.0) -> Any:
    """One request on a fresh connection (info / check / wait)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path or socket_path()))
        send_msg(sock, obj)
        reply = recv_msg(sock)
    if reply is None:
        raise ConnectionError("pi05 service closed the connection")
    return reply


# ----------------------------------------------------------------- loading


def read_safetensors_header(path: Path) -> tuple[dict, int]:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    return header, 8 + n


def load_safetensors_bounded(path: Path, device: str, staging_bytes: int = 256 << 20) -> dict:
    """Read a safetensors file tensor by tensor into tensors on ``device``.

    Bounded memory: one pinned staging buffer (``staging_bytes``) is reused for
    every chunk, and each tensor's pages are dropped from the page cache right
    after they were read (``posix_fadvise(DONTNEED)``), so an 8 GB file does
    not push the rest of the host into swap.
    """
    import numpy as np
    import torch

    header, base = read_safetensors_header(path)
    use_pinned = torch.device(device).type == "cuda"
    staging = torch.empty(staging_bytes, dtype=torch.uint8, pin_memory=use_pinned)
    staging_np = staging.numpy()
    tensors = {}
    fd = os.open(str(path), os.O_RDONLY)
    try:
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = getattr(torch, _SAFETENSORS_DTYPES[meta["dtype"]])
            start, end = meta["data_offsets"]
            out = torch.empty(tuple(meta["shape"]), dtype=dtype, device=device)
            total = end - start
            if total:
                flat = out.view(-1).view(torch.uint8)
                if flat.numel() != total:
                    raise ValueError(f"{name}: {total} bytes for shape {meta['shape']} {meta['dtype']}")
                off = 0
                while off < total:
                    k = min(staging_bytes, total - off)
                    view = memoryview(staging_np)[:k]
                    got = 0
                    while got < k:
                        n = os.preadv(fd, [view[got:]], base + start + off + got)
                        if n <= 0:
                            raise OSError(f"{path}: short read at {base + start + off + got}")
                        got += n
                    flat[off : off + k].copy_(staging[:k])
                    off += k
                try:
                    os.posix_fadvise(fd, base + start, total, os.POSIX_FADV_DONTNEED)
                except OSError:
                    pass
            tensors[name] = out
    finally:
        os.close(fd)
    if use_pinned:
        torch.cuda.synchronize()
    del staging, staging_np, np
    return tensors


def build_wrapper_bf16(feature_cfg: Any, weights: Path, device: str = "cuda:0") -> Any:
    """The eval wrapper of ``get_model`` with its weights read from the bf16
    safetensors straight onto ``device``: the modules are built on the meta
    device (no fp32 CPU initialization) and the loaded tensors are assigned.
    Every parameter and buffer must come from the file (strict)."""
    import torch
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0_config import Pi0Config
    from rlinf.models.embodiment.openpi_rlinf.utils.model_builders import _build_eval_model

    model_cfg = feature_cfg.openpi
    if str(OmegaConf.select(model_cfg, "task", default="")).lower() != "eval":
        raise ValueError("pi05 service serves the eval wrapper only (openpi.task: eval)")
    if str(feature_cfg.precision) != "bf16":
        raise ValueError(f"pi05 service serves precision bf16 only, config has {feature_cfg.precision}")
    pi0_kwargs = {
        "pi05": bool(OmegaConf.select(feature_cfg, "pi05", default=True)),
        "action_horizon": int(feature_cfg.num_action_chunks),
        "action_dim": int(model_cfg.model_action_dim),
        "paligemma_variant": str(model_cfg.paligemma_variant),
        "action_expert_variant": str(model_cfg.action_expert_variant),
        "dtype": "bfloat16",
        "pcd": False,
    }
    discrete_state_input = OmegaConf.select(model_cfg, "discrete_state_input", default=None)
    if discrete_state_input is not None:
        pi0_kwargs["discrete_state_input"] = bool(discrete_state_input)
    max_token_len = OmegaConf.select(model_cfg, "max_token_len", default=None)
    if max_token_len is not None:
        pi0_kwargs["max_token_len"] = int(max_token_len)

    with torch.device("meta"):
        model = Pi0Config(**pi0_kwargs).create().to(torch.bfloat16)
        wrapper = _build_eval_model(
            feature_cfg,
            model_cfg,
            model,
            num_steps=int(feature_cfg.num_steps),
            action_chunk=int(feature_cfg.num_action_chunks),
            action_env_dim=int(feature_cfg.action_dim),
        )
    state = load_safetensors_bounded(weights, device)
    result = wrapper.load_state_dict(state, strict=True, assign=True)
    assert not result.missing_keys and not result.unexpected_keys
    for name, tensor in list(wrapper.named_parameters()) + list(wrapper.named_buffers()):
        if tensor.device.type == "meta":
            raise RuntimeError(f"{name} is not in {weights}")
    wrapper.eval().requires_grad_(False)
    return wrapper


# ------------------------------------------------------------------ server


def _to_torch(value: Any) -> Any:
    import numpy as np
    import torch

    if isinstance(value, np.ndarray) and value.dtype.kind not in "USO":
        return torch.from_numpy(value)
    return value


def selfcheck_frames(path: Path | None, n_random: int = 4) -> list[dict]:
    """Observations for the graph self-check: the shipped real frames, or
    random images and a rest pose. Only the VLA keys (no qvel / tactile)."""
    import numpy as np
    import torch

    from rlinf.envs.realworld.tianji.tianji_kinematics import JOINT_CONTRACT, joint_rl_to_tcp_rl_np

    if path is not None and path.is_file():
        f = np.load(path)
        frames = {k: f[k] for k in f.files}
    else:
        rng = np.random.default_rng(0)
        n = n_random
        frames = {
            "head": rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8),
            "wrist_a": rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8),
            "wrist_b": rng.integers(0, 255, (n, 224, 224, 3), dtype=np.uint8),
            "joint_state": np.tile(np.array([0, -0.5, 0, -1.2, 0, 0.3, 0, 0.0] * 2, np.float32), (n, 1)),
            "prompts": np.array(["self-check prompt"] * n),
        }
    out = []
    for i in range(len(frames["prompts"])):
        js = frames["joint_state"][i].astype(np.float32)
        out.append({
            "main_images": torch.from_numpy(frames["head"][i][None].copy()),
            "extra_view_images": torch.from_numpy(
                np.stack([frames["wrist_a"][i], frames["wrist_b"][i]])[None].copy()
            ),
            "joint_states": torch.from_numpy(js[None]),
            "states": torch.from_numpy(joint_rl_to_tcp_rl_np(JOINT_CONTRACT.encode_clipped(js))[None]),
            "task_descriptions": [str(frames["prompts"][i])],
        })
    return out


class Pi05Service:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.path = Path(args.socket) if args.socket else socket_path()
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.start_token = f"{os.getpid()}-{int(time.time())}"
        self.started = time.time()
        self.ready = False
        self.phase = "starting"
        self.error: str | None = None
        self.identity: dict = {}
        self.fast_mode: str = "eager"
        self.selfcheck: dict | None = None
        self.load_s: float | None = None
        self.build_s: float | None = None
        self.warmup_s: float | None = None
        self.selfcheck_s: float | None = None
        self.requests = 0
        self.last_service_ms: float | None = None
        self.service_ms: list[float] = []
        self.wrapper = None
        self.feature_cfg = None
        self.device = args.device

    # ------------------------------------------------------------ loading

    def load(self) -> None:
        import torch

        try:
            self._set_phase("composing config")
            self.feature_cfg = feature_model_config(self.args.config)
            self._set_phase("hashing identity")
            self.identity = local_identity(self.feature_cfg)
            weights = Path(self.identity["weights_file"])
            if not weights.is_file():
                raise FileNotFoundError(
                    f"{weights} missing: run launcher/convert_pi05_bf16.py {self.identity['model_path']}"
                )
            if self.identity["weights_sha256"] is None:
                raise RuntimeError(
                    f"{sidecar_path(weights)} missing or stale (size/mtime differ): re-run the converter"
                )
            if self.args.verify_hash:
                self._set_phase("verifying weight hash")
                digest = sha256_file(weights)
                if digest != self.identity["weights_sha256"]:
                    raise RuntimeError(
                        f"{weights}: sha256 {digest} differs from its sidecar {self.identity['weights_sha256']}"
                    )
            self._set_phase("loading weights")
            t0 = time.perf_counter()
            torch.cuda.set_device(self.device)
            self.wrapper = build_wrapper_bf16(self.feature_cfg, weights, self.device)
            torch.cuda.synchronize()
            self.load_s = time.perf_counter() - t0
            log(f"loaded {weights} onto {self.device} in {self.load_s:.1f} s "
                f"({torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated)")
            # The rollout's ready flag (the launcher waits for it) is written
            # by the client, never here: the service outlives launches.
            frames_path = Path(self.args.frames) if self.args.frames else None
            if self.args.fast:
                self._set_phase("capturing CUDA graphs")
                os.environ["TIANJI_FAST_INFER"] = "1"
                from rlinf.models.embodiment.openpi_rlinf.fast_infer import warmup

                t0 = time.perf_counter()
                warmup(self.wrapper)
                self.warmup_s = time.perf_counter() - t0
                self.fast_mode = "graph"
                self._set_phase("self-check graph vs eager")
                t0 = time.perf_counter()
                self.selfcheck = self._selfcheck(frames_path)
                self.selfcheck_s = time.perf_counter() - t0
                if not self.selfcheck["ok"]:
                    os.environ["TIANJI_FAST_INFER"] = "0"
                    self.fast_mode = "eager (graph self-check failed: " + self.selfcheck["reason"] + ")"
                    log("GRAPH SELF-CHECK FAILED, serving eager:", self.selfcheck["reason"])
                else:
                    log(f"graph self-check ok on {self.selfcheck['frames']} frames x guided/unguided "
                        f"({self.selfcheck_s:.1f} s); service-side eager p50 "
                        f"{self.selfcheck['eager_ms_p50']:.0f} ms, graph p50 {self.selfcheck['graph_ms_p50']:.0f} ms")
            else:
                os.environ["TIANJI_FAST_INFER"] = "0"
                self.fast_mode = "eager (--fast 0)"
                self._set_phase("eager warm-up")
                self._warm_eager(frames_path)
            with self.state_lock:
                self.ready = True
                self.phase = "serving"
            log(f"ready: {self.fast_mode}; identity code={self.identity['code_sha256'][:12]} "
                f"weights={self.identity['weights_sha256'][:12]} cfg={self.identity['model_cfg_sha256'][:12]}")
        except Exception as exc:  # noqa: BLE001 - reported through info, then exit
            import traceback

            traceback.print_exc()
            with self.state_lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.phase = "failed"
            log("FAILED:", self.error)
            time.sleep(3.0)  # let a waiting launcher read the error
            os._exit(3)

    def _set_phase(self, phase: str) -> None:
        with self.state_lock:
            self.phase = phase
        log(phase)

    def _noise(self, generator=None):
        import torch

        return torch.randn(
            1, self.wrapper.model.action_horizon, self.wrapper.model.action_dim,
            device=self.device, generator=generator,
        )

    def _warm_eager(self, frames_path: Path | None) -> None:
        import torch

        for obs in selfcheck_frames(frames_path)[:1]:
            with torch.no_grad():
                self.wrapper.extract_vla_reference(obs, noise=self._noise())
        torch.cuda.synchronize()

    def _selfcheck(self, frames_path: Path | None) -> dict:
        """Graph vs eager, bitwise, on each frame with and without an RTC
        prefix; the same noise for both. Latency medians of both."""
        import numpy as np
        import torch

        rows = int(os.environ.get("RLT_REF_OFFSET", "24") or 24)
        obs_list = selfcheck_frames(frames_path)
        limit = int(self.args.selfcheck_frames or 0)
        if limit > 0:
            obs_list = obs_list[:limit]
        results, eager_ms, graph_ms = [], [], []
        gen = torch.Generator(device=self.device).manual_seed(1234)

        def run(mode: str, obs, noise):
            os.environ["TIANJI_FAST_INFER"] = mode
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                z, ref = self.wrapper.extract_vla_reference(obs, noise=noise)
            torch.cuda.synchronize()
            return (z.detach().cpu(), torch.as_tensor(ref).detach().cpu()), (time.perf_counter() - t0) * 1e3

        try:
            for i, obs in enumerate(obs_list):
                noise = self._noise(gen)
                (z0, ref0), _ = run("0", obs, noise)
                prev = ref0[0, 12 : 12 + rows].float()
                guided = dict(obs, rtc_prev_actions=prev[None].clone(), rtc_prev_valid=torch.tensor([True]))
                for kind, o in (("unguided", obs), ("guided", guided)):
                    (ze, refe), t_e = run("0", o, noise)
                    (zg, refg), t_g = run("1", o, noise)
                    same = (
                        torch.equal(ze, zg) and torch.equal(refe, refg)
                        and ze.dtype == zg.dtype and refe.dtype == refg.dtype
                    )
                    results.append({"frame": i, "kind": kind, "bitwise": bool(same),
                                    "z_max": float((ze.float() - zg.float()).abs().max()),
                                    "ref_max": float((refe.double() - refg.double()).abs().max())})
                    if i > 0 or len(obs_list) == 1:  # the first frame pays warm-up
                        eager_ms.append(t_e)
                        graph_ms.append(t_g)
        except Exception as exc:  # noqa: BLE001
            os.environ["TIANJI_FAST_INFER"] = "1"
            return {"ok": False, "reason": f"{type(exc).__name__}: {exc}", "rows": results}
        os.environ["TIANJI_FAST_INFER"] = "1"
        ok = bool(results) and all(r["bitwise"] for r in results)
        return {
            "ok": ok,
            "reason": "" if ok else "outputs differ: " + json.dumps([r for r in results if not r["bitwise"]][:3]),
            "frames": len(obs_list),
            "frames_file": str(frames_path) if frames_path and frames_path.is_file() else "random",
            "rows": results,
            "eager_ms_p50": float(np.median(eager_ms)) if eager_ms else None,
            "graph_ms_p50": float(np.median(graph_ms)) if graph_ms else None,
        }

    # ------------------------------------------------------------ serving

    def info(self) -> dict:
        import torch

        with self.state_lock:
            recent = self.service_ms[-200:]
            state = {
                "protocol": PROTOCOL,
                "ready": self.ready,
                "phase": self.phase,
                "error": self.error,
                "pid": os.getpid(),
                "start_token": self.start_token,
                "started": self.started,
                "uptime_s": time.time() - self.started,
                "socket": str(self.path),
                "identity": self.identity,
                "fast_mode": self.fast_mode,
                "graph_selfcheck": (
                    {k: v for k, v in self.selfcheck.items() if k != "rows"} if self.selfcheck else None
                ),
                "load_s": self.load_s,
                "warmup_s": self.warmup_s,
                "selfcheck_s": self.selfcheck_s,
                "requests": self.requests,
                "last_service_ms": self.last_service_ms,
                "service_ms_p50_recent": (sorted(recent)[len(recent) // 2] if recent else None),
            }
        try:
            idx = torch.cuda.current_device()
            free, total = torch.cuda.mem_get_info(idx)
            state["gpu"] = {
                "name": torch.cuda.get_device_name(idx),
                "index": idx,
                "allocated_gib": torch.cuda.memory_allocated(idx) / 2**30,
                "reserved_gib": torch.cuda.memory_reserved(idx) / 2**30,
                "device_used_gib": (total - free) / 2**30,
                "device_total_gib": total / 2**30,
            }
        except Exception as exc:  # noqa: BLE001
            state["gpu"] = {"error": str(exc)}
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith(("VmRSS", "VmHWM", "VmSwap")):
                        key, value = line.split(":", 1)
                        state.setdefault("memory_mib", {})[key] = int(value.split()[0]) // 1024
        except OSError:
            pass
        return state

    def infer(self, env_obs: dict, noise: Any) -> dict:
        import torch

        with self.state_lock:
            if not self.ready:
                raise RuntimeError(f"service not ready ({self.phase}{': ' + self.error if self.error else ''})")
        obs = {k: _to_torch(v) for k, v in env_obs.items() if k in VLA_OBS_KEYS}
        with self.lock:
            torch.cuda.set_device(self.device)
            t0 = time.perf_counter()
            noise_t = _to_torch(noise).to(self.device)
            with torch.no_grad():
                z_rl, ref_chunk = self.wrapper.extract_vla_reference(obs, noise=noise_t)
            z_np = z_rl.detach().cpu().numpy()
            # ref_chunk keeps the eager path's dtype/values (float64, RL frame).
            ref_np = torch.as_tensor(ref_chunk).detach().cpu().numpy()
            ms = (time.perf_counter() - t0) * 1e3
            with self.state_lock:
                self.requests += 1
                self.last_service_ms = ms
                self.service_ms.append(ms)
                if len(self.service_ms) > 2000:
                    del self.service_ms[:1000]
        return {"ok": True, "z_rl": z_np, "ref_chunk": ref_np, "service_ms": ms, "start_token": self.start_token}

    def handle(self, conn: socket.socket) -> None:
        try:
            # No idle timeout: Stage2 keeps one connection open across pauses
            # (waiting for Start can take long); a dead client closes it.
            conn.settimeout(None)
            while True:
                msg = recv_msg(conn)
                if msg is None:
                    return
                op = msg.get("op")
                try:
                    if op == "info":
                        reply = {"ok": True, **self.info()}
                    elif op == "ping":
                        reply = {"ok": True, "start_token": self.start_token}
                    elif op == "infer":
                        reply = self.infer(msg["env_obs"], msg["noise"])
                    else:
                        reply = {"ok": False, "error": f"unknown op {op!r}"}
                except Exception as exc:  # noqa: BLE001 - reported to the client
                    import traceback

                    traceback.print_exc()
                    reply = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "start_token": self.start_token}
                send_msg(conn, reply)
        except (OSError, pickle.PickleError, EOFError) as exc:
            log("connection closed:", repr(exc))
        finally:
            conn.close()

    def serve(self) -> None:
        import signal

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:  # a dead service left its socket behind
                request({"op": "ping"}, path=self.path, timeout=2.0)
                log(f"another service answers on {self.path}; exiting")
                sys.exit(2)
            except OSError:
                self.path.unlink()
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.path))
        os.chmod(self.path, 0o600)
        server.listen(8)
        log(f"listening on {self.path} (pid {os.getpid()})")

        def stop(signum, _frame):
            log(f"signal {signum}: exiting")
            try:
                self.path.unlink()
            except OSError:
                pass
            os._exit(0)

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=self.load, name="pi05-load", daemon=True).start()
        try:
            while True:
                conn, _ = server.accept()
                threading.Thread(target=self.handle, args=(conn,), daemon=True).start()
        finally:
            server.close()
            try:
                self.path.unlink()
            except OSError:
                pass


# --------------------------------------------------------------------- CLI


def check_service(config: str, socket_file: str | None, expected: dict | None = None) -> tuple[int, str]:
    """(code, message): 0 ready+match, 3 no service, 4 mismatch, 5 loading, 6 failed."""
    path = Path(socket_file) if socket_file else socket_path()
    try:
        info = request({"op": "info"}, path=path, timeout=10.0)
    except OSError:
        return 3, f"no pi05 service on {path}"
    if info.get("error"):
        return 6, f"pi05 service failed while {info.get('phase')}: {info['error']}"
    if not info.get("ready"):
        return 5, f"pi05 service loading: {info.get('phase')}"
    if expected is None:
        expected = local_identity(feature_model_config(config))
    reasons = identity_mismatch(expected, info.get("identity", {}))
    if reasons:
        return 4, "pi05 service differs from this Stage2 tree/config:\n" + "\n".join("  " + r for r in reasons)
    return 0, (f"pi05 service ready: {info['fast_mode']}, pid {info['pid']}, "
               f"uptime {info['uptime_s']:.0f} s, requests {info['requests']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "info", "check", "identity", "wait"):
        p = sub.add_parser(name)
        p.add_argument("--config", default=os.environ.get("TIANJI_STAGE2_CONFIG", DEFAULT_CONFIG))
        p.add_argument("--socket", default=os.environ.get("TIANJI_PI05_SOCKET"))
        if name == "serve":
            p.add_argument("--device", default="cuda:0")
            p.add_argument("--fast", type=int, default=1, help="1: CUDA-graph fast path (TIANJI_FAST_INFER=1)")
            p.add_argument("--frames", default=None, help="npz of real frames for the self-check")
            p.add_argument("--selfcheck-frames", type=int, default=0, help="limit (0 = all frames)")
            p.add_argument("--verify-hash", action="store_true", help="sha256 the weights at start (~20 s)")
        if name in ("check", "wait"):
            p.add_argument("--quiet", action="store_true")
        if name == "wait":
            p.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args(argv)

    if args.cmd == "serve":
        Pi05Service(args).serve()
        return 0
    if args.cmd == "identity":
        print(json.dumps(local_identity(feature_model_config(args.config)), indent=1))
        return 0
    if args.cmd == "info":
        path = Path(args.socket) if args.socket else socket_path()
        try:
            print(json.dumps(request({"op": "info"}, path=path, timeout=10.0), indent=1, default=str))
            return 0
        except OSError as exc:
            print(json.dumps({"ready": False, "error": f"no service on {path}: {exc}"}))
            return 3
    if args.cmd == "check":
        rc, message = check_service(args.config, args.socket)
        if rc != 0 or not args.quiet:
            print(message)
        return rc
    if args.cmd == "wait":
        deadline = time.time() + args.timeout
        expected = None
        last = None
        while True:
            rc, message = check_service(args.config, args.socket, expected)
            if rc == 0 and expected is None:
                expected = local_identity(feature_model_config(args.config))
            if message != last and (rc != 0 or not args.quiet):
                print(message, flush=True)  # only when the state changes
                last = message
            if rc in (0, 4, 6):
                return rc
            if time.time() > deadline:
                print(f"pi05 service not ready after {args.timeout:.0f} s")
                return 7
            time.sleep(2.0)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
