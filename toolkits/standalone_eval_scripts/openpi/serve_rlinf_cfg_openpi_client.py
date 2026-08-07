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

"""Serve an RLinf OpenPI CFG checkpoint with the openpi-client protocol.

The OpenPI upstream ``scripts/serve_policy.py`` can serve OpenPI-native
checkpoints, but RLinf CFG training saves a ``cfg_model`` checkpoint whose
weights live under ``model_state_dict/full_weights.pt``. This script loads that
RLinf model and exposes the same WebSocket protocol used by ``openpi-client``.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import socket
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.serving import websocket_policy_server

DEFAULT_CHECKPOINT_DIR = (
    "logs/aloha_sandwich_recap/20260707-065405/cfg_rl/"
    "aloha_sandwich_cfg/checkpoints/global_step_1200/actor"
)
DEFAULT_NORM_STATS_SOURCE = (
    "/inspire/qb-ilm/project/robot-reasoning/czxs253130583/yushun/openpi/"
    "checkpoints/torch/pi05_sandwich_new_all_rlinf_base/pi05_sandwich_new_all/"
    "norm_stats.json"
)


def _as_numpy(value: Any) -> np.ndarray:
    """Convert array-like values to NumPy without changing layout."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _ensure_batch(
    value: Any, *, expected_ndim_without_batch: int, name: str
) -> np.ndarray:
    """Return ``value`` with a leading batch dimension."""
    array = _as_numpy(value)
    if array.ndim == expected_ndim_without_batch:
        return array[None, ...]
    if array.ndim == expected_ndim_without_batch + 1:
        return array
    raise ValueError(
        f"{name} must have {expected_ndim_without_batch} or "
        f"{expected_ndim_without_batch + 1} dims, got shape {array.shape}."
    )


def _wrist_pair_from_images(images: dict[str, Any]) -> np.ndarray:
    """Stack ALOHA left/right wrist images into ``[2, ...]``."""
    missing = [
        key for key in ("cam_left_wrist", "cam_right_wrist") if key not in images
    ]
    if missing:
        raise KeyError(f"Missing ALOHA wrist camera(s): {missing}")
    return np.stack(
        [
            _as_numpy(images["cam_left_wrist"]),
            _as_numpy(images["cam_right_wrist"]),
        ],
        axis=0,
    )


def _normalise_observation(obs: dict[str, Any], default_prompt: str) -> dict[str, Any]:
    """Convert openpi-client observations to RLinf ``predict_action_batch`` input."""
    prompt = obs.get("prompt") or default_prompt

    if "images" in obs:
        images = obs["images"]
        if "cam_high" not in images:
            raise KeyError("ALOHA observation images must include 'cam_high'.")
        main_images = _ensure_batch(
            images["cam_high"], expected_ndim_without_batch=3, name="images.cam_high"
        )
        wrist_images = _ensure_batch(
            _wrist_pair_from_images(images),
            expected_ndim_without_batch=4,
            name="stacked wrist images",
        )
        states = _ensure_batch(
            obs["state"], expected_ndim_without_batch=1, name="state"
        )
    elif "observation/image" in obs:
        main_images = _ensure_batch(
            obs["observation/image"],
            expected_ndim_without_batch=3,
            name="observation/image",
        )
        wrist_images = _ensure_batch(
            obs["observation/wrist_image"],
            expected_ndim_without_batch=4,
            name="observation/wrist_image",
        )
        states = _ensure_batch(
            obs["observation/state"],
            expected_ndim_without_batch=1,
            name="observation/state",
        )
    else:
        raise KeyError(
            "Observation must use either ALOHA runtime keys "
            "('images', 'state') or OpenPI keys "
            "('observation/image', 'observation/wrist_image', 'observation/state')."
        )

    if (
        main_images.shape[0] != wrist_images.shape[0]
        or main_images.shape[0] != states.shape[0]
    ):
        raise ValueError(
            "Observation batch sizes do not match: "
            f"main={main_images.shape[0]}, wrist={wrist_images.shape[0]}, "
            f"state={states.shape[0]}."
        )

    return {
        "main_images": main_images,
        "wrist_images": wrist_images,
        "states": states,
        "task_descriptions": [prompt] * states.shape[0],
    }


def _copy_norm_stats_if_needed(
    checkpoint_dir: Path,
    *,
    repo_id: str,
    source: Path | None,
    auto_copy: bool,
) -> None:
    target = checkpoint_dir / repo_id / "norm_stats.json"
    if target.exists():
        return

    if not auto_copy:
        raise FileNotFoundError(
            f"Missing norm stats at {target}. Provide --norm-stats-source "
            "or copy the file before starting the server."
        )

    if source is None:
        raise FileNotFoundError(
            f"Missing norm stats at {target}, and no --norm-stats-source was provided."
        )

    source_file = source / "norm_stats.json" if source.is_dir() else source
    if not source_file.exists():
        raise FileNotFoundError(
            f"Missing norm stats at {target}; source does not exist: {source_file}"
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_file, target)
    logging.info("Copied norm stats from %s to %s", source_file, target)


def _build_model_cfg(args: argparse.Namespace) -> Any:
    return OmegaConf.create(
        {
            "model_path": str(args.checkpoint_dir),
            "model_type": "cfg_model",
            "action_dim": args.action_dim,
            "num_action_chunks": args.action_chunk,
            "openpi_data": {"repo_id": args.repo_id},
            "openpi": {
                "config_name": args.config_name,
                "num_images_in_input": 3,
                "action_env_dim": args.action_dim,
                "action_chunk": args.action_chunk,
                "num_steps": args.num_steps,
                "cfgrl_guidance_scale": args.guidance_scale,
                "unconditional_prob": args.unconditional_prob,
                "train_expert_only": False,
                "guidance_type": args.guidance_type,
                "positive_only_conditional": args.positive_only_conditional,
            },
        }
    )


class RLinfCfgOpenPiPolicy:
    """Small policy wrapper compatible with ``openpi-client`` WebSocket calls."""

    def __init__(self, args: argparse.Namespace):
        from rlinf.models.embodiment.openpi_cfg import get_model

        _copy_norm_stats_if_needed(
            args.checkpoint_dir,
            repo_id=args.repo_id,
            source=args.norm_stats_source,
            auto_copy=args.auto_copy_assets,
        )
        model_cfg = _build_model_cfg(args)
        self._model = get_model(model_cfg)
        self._model.to(args.device)
        self._model.eval()
        self._default_prompt = args.default_prompt
        self.metadata = {
            "model_type": "rlinf_cfg_model",
            "checkpoint_dir": str(args.checkpoint_dir),
            "config_name": args.config_name,
            "repo_id": args.repo_id,
            "action_horizon": args.action_chunk,
            "action_dim": args.action_dim,
            "default_prompt": args.default_prompt,
            "guidance_type": args.guidance_type,
            "guidance_scale": args.guidance_scale,
        }

    @torch.no_grad()
    def infer(self, obs: dict[str, Any], **_: Any) -> dict[str, Any]:
        env_obs = _normalise_observation(obs, self._default_prompt)
        actions, _ = self._model.predict_action_batch(env_obs=env_obs, mode="eval")
        if actions.shape[0] == 1:
            actions = actions[0]
        return {"actions": actions}


def _random_aloha_observation(prompt: str) -> dict[str, Any]:
    return {
        "state": np.zeros((14,), dtype=np.float32),
        "images": {
            "cam_high": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_left_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
            "cam_right_wrist": np.zeros((3, 224, 224), dtype=np.uint8),
        },
        "prompt": prompt,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve an RLinf OpenPI CFG checkpoint through the openpi-client "
            "WebSocket protocol."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT_DIR),
        help="RLinf actor checkpoint directory containing model_state_dict/full_weights.pt.",
    )
    parser.add_argument("--config-name", default="pi05_aloha_robotwin")
    parser.add_argument("--repo-id", default="pi05_sandwich_new_all")
    parser.add_argument("--default-prompt", default="Assemble a sandwich.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument("--action-chunk", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument(
        "--guidance-type",
        choices=("positive", "negative", "no_guide"),
        default="positive",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--unconditional-prob", type=float, default=0.1)
    parser.add_argument(
        "--positive-only-conditional",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--norm-stats-source",
        type=Path,
        default=Path(DEFAULT_NORM_STATS_SOURCE),
        help="norm_stats.json file, or directory containing norm_stats.json.",
    )
    parser.add_argument(
        "--auto-copy-assets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy missing norm_stats.json into the checkpoint directory before loading.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Load the model, run one random ALOHA observation, print action shape, and exit.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )
    args = _parse_args()
    args.checkpoint_dir = args.checkpoint_dir.resolve()

    weights_path = args.checkpoint_dir / "model_state_dict" / "full_weights.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing RLinf full weights: {weights_path}")

    policy = RLinfCfgOpenPiPolicy(args)
    if args.smoke_test:
        result = policy.infer(_random_aloha_observation(args.default_prompt))
        logging.info("Smoke test action shape: %s", result["actions"].shape)
        return

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(
        "Starting openpi-client server on %s:%s (%s)", args.host, args.port, local_ip
    )
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
