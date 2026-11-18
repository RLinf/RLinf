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

"""Compare the official BF16 forward and the configured FSDP2 update."""

import argparse
import gc
import json
import math
import os
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.fsdp.fsdp_model_manager import FSDPModelManager
from rlinf.hybrid_engines.fsdp.strategy.base import FSDPStrategyBase
from rlinf.models import get_model
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.logging import get_logger
from rlinf.utils.utils import get_rng_state, set_rng_state


def local_cpu(tensor: torch.Tensor) -> torch.Tensor:
    """Read a single-rank tensor without changing its distribution."""
    if isinstance(tensor, DTensor):
        tensor = tensor.to_local()
    return tensor.detach().cpu().clone()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--backbone-model-path", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("EMBODIED_PATH", str(root / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(root / "examples/sft/config"), version_base="1.1"
    ):
        cfg = compose(config_name="libero_sft_gr00t_n1d7")
    cfg.actor.model.model_path = args.model_path
    cfg.actor.model.backbone_model_path = args.backbone_model_path
    dtype = torch_dtype_from_precision(
        cfg.actor.fsdp_config.mixed_precision.param_dtype
    )
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(cfg.actor.seed)
    batch = torch.load(args.batch, weights_only=False)
    reference_model = get_model(cfg.actor.model).to(device, dtype=dtype).train()
    assert not any(p.requires_grad for p in reference_model.backbone.parameters())
    rng = get_rng_state()
    reference = Gr00tN1d7.forward(reference_model, dict(batch["inputs"]))["loss"]
    reference.backward()
    assert torch.isfinite(reference)
    gradients = {
        n: local_cpu(p.grad).float()
        for n, p in reference_model.named_parameters()
        if p.grad is not None
    }
    # DeepSpeed's BF16 path updates an FP32 master copy of each model parameter.
    masters = {
        n: torch.nn.Parameter(p.detach().float().clone())
        for n, p in reference_model.named_parameters()
        if p.requires_grad
    }
    for name, p in reference_model.named_parameters():
        if name in gradients:
            masters[name].grad = p.grad.detach().float().clone()
    optim = cfg.actor.optim
    reference_optim = torch.optim.AdamW(
        list(masters.values()),
        weight_decay=optim.weight_decay,
        lr=optim.lr,
        betas=(optim.adam_beta1, optim.adam_beta2),
        eps=optim.adam_eps,
        fused=False,
    )
    reference_grad_norm = torch.nn.utils.clip_grad_norm_(
        list(masters.values()), optim.clip_grad
    )
    initial = {n: local_cpu(p) for n, p in masters.items()}
    reference_optim.step()
    expected_update = {n: local_cpu(p) for n, p in masters.items()}
    reference_loss = float(reference.detach())
    reference_norm = float(reference_grad_norm)
    total_parameters = sum(p.numel() for p in reference_model.parameters())
    trainable_parameters = sum(p.numel() for p in masters.values())
    del reference, reference_model, reference_optim, masters, p
    gc.collect()
    torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory() as temp:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"file://{temp}/store",
            rank=0,
            world_size=1,
            device_id=device,
        )
        mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp",))
        strategy = FSDPStrategyBase.create(
            cfg.actor, 1, torch.distributed.group.WORLD, get_logger()
        )
        model = strategy.wrap_model(get_model(cfg.actor.model), mesh)
        model.train()
        manager = SimpleNamespace(
            _cfg=cfg.actor, _logger=get_logger(), store_requires_grad_param_name=[]
        )
        actual_optim = FSDPModelManager.build_optimizer(manager, model)
        set_rng_state(rng)
        amp = (
            torch.autocast("cuda", dtype=torch.bfloat16)
            if cfg.actor.fsdp_config.amp_autocast.enabled
            else nullcontext()
        )
        with amp:
            actual = model(forward_type=ForwardType.SFT, data=batch)["loss"]
        actual.backward()
        torch.testing.assert_close(
            actual.detach().float().cpu(), torch.tensor(reference_loss), rtol=0, atol=0
        )
        max_grad_error = 0.0
        grad_error_sq = grad_ref_sq = grad_found_sq = grad_dot = 0.0
        for name, p in model.named_parameters():
            if name in gradients:
                found = local_cpu(p.grad).float()
                diff = (found - gradients[name]).double()
                grad_error_sq += float(diff.square().sum())
                grad_ref_sq += float(gradients[name].double().square().sum())
                grad_found_sq += float(found.double().square().sum())
                grad_dot += float((found.double() * gradients[name].double()).sum())
                max_grad_error = max(
                    max_grad_error, float((found - gradients[name]).abs().max())
                )
        actual_grad_norm = strategy.clip_grad_norm_(model)
        actual_optim.step()
        max_update_error = 0.0
        update_error_sq = update_ref_sq = 0.0
        active = bf16_mismatches = 0
        for name, p in model.named_parameters():
            if name in expected_update:
                found = local_cpu(p)
                diff = (found - expected_update[name]).double()
                update_error_sq += float(diff.square().sum())
                update_ref_sq += float(
                    (expected_update[name] - initial[name]).double().square().sum()
                )
                if name in gradients:
                    mask = gradients[name] != 0
                    active += int(mask.sum())
                    bf16_mismatches += int(
                        (
                            (found.bfloat16() != expected_update[name].bfloat16())
                            & mask
                        ).sum()
                    )
                max_update_error = max(
                    max_update_error, float((found - expected_update[name]).abs().max())
                )
        result = {
            "official_bf16_loss": reference_loss,
            "fsdp2_loss": float(actual.detach()),
            "official_fp32_master_grad_norm": reference_norm,
            "fsdp2_grad_norm": float(actual_grad_norm),
            "gradient_tensors": len(gradients),
            "max_gradient_abs_error": max_grad_error,
            "max_fp32_master_update_abs_error": max_update_error,
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
            "gradient_relative_l2_error": math.sqrt(grad_error_sq / grad_ref_sq),
            "gradient_cosine_similarity": grad_dot
            / math.sqrt(grad_ref_sq * grad_found_sq),
            "master_update_relative_l2_error": math.sqrt(
                update_error_sq / update_ref_sq
            ),
            "bf16_weight_mismatch_fraction_active": bf16_mismatches / max(1, active),
            "compute_dtype": str(dtype),
            "master_dtype": "torch.float32",
            "world_size": 1,
            "torch_version": torch.__version__,
        }
        eps = torch.finfo(dtype).eps
        result["acceptance"] = {
            "gradient_relative_l2_below_compute_epsilon": result[
                "gradient_relative_l2_error"
            ]
            < eps,
            "gradient_cosine_within_compute_epsilon_squared": result[
                "gradient_cosine_similarity"
            ]
            > 1 - eps**2,
            "bf16_weight_mismatch_fraction_below_compute_epsilon": result[
                "bf16_weight_mismatch_fraction_active"
            ]
            < eps,
        }
        result["passed"] = all(result["acceptance"].values())
        args.output.write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2), flush=True)
        del model, actual_optim, actual, p
        gc.collect()
        torch.cuda.synchronize()
        torch.distributed.destroy_process_group()
        assert result["passed"], result


if __name__ == "__main__":
    main()
