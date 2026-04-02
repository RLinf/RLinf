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

"""Launch the official LingBot-VA websocket server in a subprocess."""

from __future__ import annotations

import argparse
import copy
import importlib.machinery
import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn.functional as F


def _install_flash_attn_shim_if_needed(model_path: Path) -> None:
    config_path = model_path / "transformer" / "config.json"
    if not config_path.exists():
        return
    attn_mode = json.loads(config_path.read_text()).get("attn_mode")
    if attn_mode != "torch":
        return
    try:
        import flash_attn  # noqa: F401

        return
    except Exception:
        pass

    def flash_attn_func(
        query,
        key,
        value,
        dropout_p=0.0,
        softmax_scale=None,
        causal=False,
        *args,
        **kwargs,
    ):
        out = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
            dropout_p=dropout_p,
            scale=softmax_scale,
            is_causal=causal,
        )
        return out.transpose(1, 2)

    shim = types.ModuleType("flash_attn")
    shim.flash_attn_func = flash_attn_func
    shim.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
    interface_shim = types.ModuleType("flash_attn_interface")
    interface_shim.flash_attn_func = flash_attn_func
    interface_shim.__spec__ = importlib.machinery.ModuleSpec(
        "flash_attn_interface", loader=None
    )
    sys.modules.setdefault("flash_attn", shim)
    sys.modules.setdefault("flash_attn_interface", interface_shim)


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported LingBot-VA dtype override: {name!r}.")
    return mapping[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--config-name", default="robotwin")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--save-root", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--enable-offload", choices=["true", "false"], default="true")
    parser.add_argument("--param-dtype", default="bf16")
    args = parser.parse_args()

    repo_path = Path(args.repo_path)
    model_path = Path(args.model_path)
    torch_num_threads = int(os.environ.get("LINGBOTVA_TORCH_NUM_THREADS", "32"))
    try:
        torch.set_num_threads(torch_num_threads)
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, min(8, torch_num_threads // 4)))
    except Exception:
        pass
    sys.path.insert(0, str(repo_path))
    sys.path.insert(0, str(repo_path / "wan_va"))

    _install_flash_attn_shim_if_needed(model_path)

    import wan_va.wan_va_server as wan_va_server
    from wan_va.configs import VA_CONFIGS
    from wan_va.utils import init_logger, logger

    init_logger()
    config = copy.deepcopy(VA_CONFIGS[args.config_name])
    config.wan22_pretrained_model_name_or_path = str(model_path)
    config.save_root = args.save_root
    config.host = args.host
    config.port = args.port
    config.enable_offload = args.enable_offload == "true"
    config.param_dtype = _parse_dtype(args.param_dtype)

    VA_CONFIGS[args.config_name] = config
    if hasattr(wan_va_server, "VA_CONFIGS"):
        wan_va_server.VA_CONFIGS[args.config_name] = config

    logger.info("Launching LingBot-VA official websocket server via wan_va_server.run.")
    wan_va_server.run(
        Namespace(
            config_name=args.config_name,
            port=args.port,
            save_root=args.save_root,
        )
    )


if __name__ == "__main__":
    main()
