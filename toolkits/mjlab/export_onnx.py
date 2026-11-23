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

"""Export an RLinf MLPPolicy checkpoint to a deployment ONNX graph.

The exported graph computes ``actor_mean(backbone(actor_obs_normalizer(obs)))``
so running observation normalization travels inside the ONNX file, matching
the deployment contract expected by ``pollen-robotics/microduck``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.utils.logging import get_logger

logger = get_logger()


class DeterministicActorExport(nn.Module):
    """Deterministic inference graph with observation normalization baked in."""

    def __init__(self, policy: MLPPolicy) -> None:
        super().__init__()
        self.obs_normalizer = policy.actor_obs_normalizer
        self.backbone = policy.backbone
        self.actor_mean = policy.actor_mean

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(self.backbone(self.obs_normalizer(obs)))


def export_mlp_to_onnx(
    policy: MLPPolicy,
    output_path: str | Path,
    opset_version: int = 18,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Export ``policy``'s deterministic actor (with normalizer) to ONNX.

    Args:
        policy: Trained ``MLPPolicy`` instance.
        output_path: Destination ``.onnx`` path.
        opset_version: ONNX opset version passed to ``torch.onnx.export``.
        metadata: Optional string metadata written to ``model.metadata_props``.

    Returns:
        Resolved path to the written ONNX file.
    """
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    export_module = DeterministicActorExport(policy).to("cpu").eval()
    dummy_obs = torch.zeros(1, policy.obs_dim, dtype=torch.float32)

    torch.onnx.export(
        export_module,
        (dummy_obs,),
        str(out),
        input_names=["obs"],
        output_names=["actions"],
        opset_version=opset_version,
        dynamo=False,
    )

    if metadata:
        import onnx

        model = onnx.load(str(out))
        for key, value in metadata.items():
            entry = model.metadata_props.add()
            entry.key = str(key)
            entry.value = str(value)
        onnx.save(model, str(out))

    return out


def _load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in raw and isinstance(raw[key], dict):
                raw = raw[key]
                break
    return {
        k.removeprefix("module.").removeprefix("_fsdp_wrapped_module."): v
        for k, v in raw.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export an RLinf MLPPolicy checkpoint to deployment ONNX."
    )
    parser.add_argument("--ckpt", required=True, help="Path to saved model .pt file.")
    parser.add_argument("--output", default="policy.onnx", help="Output .onnx path.")
    parser.add_argument("--obs-dim", type=int, default=61)
    parser.add_argument("--action-dim", type=int, default=14)
    parser.add_argument(
        "--hidden-sizes",
        type=int,
        nargs="+",
        default=[512, 256, 128],
    )
    parser.add_argument("--activation", default="elu")
    parser.add_argument("--no-obs-norm", action="store_true")
    parser.add_argument("--obs-norm-eps", type=float, default=1e-2)
    parser.add_argument("--task-id", default="Mjlab-Velocity-Flat-MicroDuck")
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args()

    policy = MLPPolicy(
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        num_action_chunks=1,
        add_value_head=False,
        add_q_head=False,
        hidden_sizes=tuple(args.hidden_sizes),
        activation=args.activation,
        obs_normalization=not args.no_obs_norm,
        obs_norm_eps=args.obs_norm_eps,
    )
    state_dict = _load_state_dict(Path(args.ckpt))
    actor_prefixes = ("actor_obs_normalizer.", "backbone.", "actor_mean.")
    expected_keys = {k for k in policy.state_dict() if k.startswith(actor_prefixes)}
    actor_state = {k: v for k, v in state_dict.items() if k in expected_keys}
    missing_keys = sorted(expected_keys - actor_state.keys())
    if missing_keys:
        raise KeyError(f"Checkpoint is missing required actor keys: {missing_keys}.")
    policy.load_state_dict(actor_state, strict=False)

    metadata = {"task_id": args.task_id, "checkpoint": str(args.ckpt)}
    out_path = export_mlp_to_onnx(
        policy, args.output, opset_version=args.opset, metadata=metadata
    )
    logger.info(f"Exported ONNX policy to {out_path}")


if __name__ == "__main__":
    main()
