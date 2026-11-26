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
"""Take one optimizer step on saved rollout rows using the agent worker's loss."""

import argparse
import json
import math
from pathlib import Path

import ray
import torch
from hydra import compose, initialize_config_dir
from megatron.core import parallel_state
from megatron.core.distributed import finalize_model_grads

from rlinf.config import validate_cfg
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import AcceleratorUtil, Cluster, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic


def update(worker, rows_file: str, critic: bool) -> dict:
    """Run one actor or critic optimizer update on saved image-replay episodes."""
    # The two turns of an episode stay together on the same data replica.
    groups = torch.load(rows_file, map_location="cpu", weights_only=True)
    dp_rank = parallel_state.get_data_parallel_rank()
    dp_size = parallel_state.get_data_parallel_world_size()
    turns = [turn for group in groups[dp_rank::dp_size] for turn in group]
    batch = DynamicRolloutResult(
        group_id=dp_rank,
        num_sequence=len(turns),
        group_size=1,
        idx_to_traj=[i // 2 for i in range(len(turns))],
        input_ids=[t["entry"]["prompt_token_ids"] + t["output_ids"] for t in turns],
        prompt_lengths=[len(t["entry"]["prompt_token_ids"]) for t in turns],
        response_lengths=[len(t["output_ids"]) for t in turns],
        is_end=[i % 2 == 1 for i in range(len(turns))],
        multi_modal_inputs=[t["entry"] for t in turns],
        rollout_logprobs=[t["logprobs"] for t in turns],
    ).to_actor_batch()
    model = worker.model[0]
    model.train()
    worker.optimizer.zero_grad()
    model.zero_grad_buffer()
    worker.return_loss = False
    forward = worker.get_forward_step_func()
    with torch.no_grad():
        output, collect = forward(iter([batch]), model)
        old = collect(output)
    mask = batch["response_mask"]
    if critic:
        batch["values"] = old
        batch["returns"] = [
            value + selected.float() for value, selected in zip(old, mask, strict=True)
        ]
    else:
        batch["recomputed_logprobs"] = old
        batch["advantages"] = [selected.float() for selected in mask]
        batch["loss_scales"] = [torch.ones_like(row) for row in batch["advantages"]]
    worker.return_loss = True
    output, loss_fn = forward(iter([batch]), model)
    loss, _ = loss_fn(output)
    loss.backward()
    finalize_model_grads(worker.model)
    module = worker.get_model_module_list()[0]
    key = (
        "output_layer.weight"
        if critic
        else "language_model.decoder.layers.0.mlp.linear_fc1.weight"
    )
    parameter = next(
        p for name, p in module.named_parameters() if key in name and p.requires_grad
    )
    before = parameter.detach().clone()
    ok, norm, _, _ = worker.optimizer_step(1)
    changed = (parameter.detach() != before).count_nonzero().item()
    if not (
        ok and changed and torch.isfinite(loss) and math.isfinite(norm) and norm > 0
    ):
        raise RuntimeError(
            f"Optimizer update failed: loss={loss.item()}, grad_norm={norm}, changed={changed}"
        )
    return {
        "rank": torch.distributed.get_rank(),
        "loss": loss.item(),
        "grad_norm": float(norm),
        "changed": changed,
    }


class ReplayActor(MAMegatronActor):
    def update_images(self, rows_file: str) -> dict:
        return update(self, rows_file, False)


class ReplayCritic(MAMegatronCritic):
    def update_images(self, rows_file: str) -> dict:
        return update(self, rows_file, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--role", choices=["actor", "critic"], required=True)
    args = parser.parse_args()
    cards = Worker.torch_platform.device_count()
    if cards not in (2, 8):
        parser.error(
            f"Expected 2 or 8 local GPUs via CUDA_VISIBLE_DEVICES, got {cards}."
        )
    visible = AcceleratorUtil.get_visible_devices(
        AcceleratorUtil.get_accelerator_type()
    )
    devices = sorted(visible) if visible else list(range(cards))
    if devices != list(range(devices[0], devices[-1] + 1)):
        parser.error(f"Expected contiguous local GPU indices, got {devices}.")
    gpu_range = f"{devices[0]}-{devices[-1]}"
    here = Path(__file__).resolve().parent
    with initialize_config_dir(version_base=None, config_dir=str(here / "config")):
        cfg = compose(
            config_name="replay_update",
            overrides=[
                f"rollout.model.model_path={args.model.resolve()}",
                f"cluster.component_placement.actor={gpu_range}",
                f"cluster.component_placement.critic={gpu_range}",
                f"cluster.component_placement.rollout={gpu_range}",
                f"cluster.component_placement.reward={gpu_range}",
            ],
        )
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    try:
        placement = ModelParallelComponentPlacement(cfg, cluster)
        strategy = placement.get_strategy(args.role)
        slots = strategy.get_placement(cluster)
        if len(slots) != cards or {s.local_accelerator_rank for s in slots} != set(
            devices
        ):
            raise RuntimeError(
                f"Ray placement differs from selected GPUs {devices}: got {slots}."
            )
        cls = ReplayCritic if args.role == "critic" else ReplayActor
        group = cls.create_group(cfg, placement).launch(
            cluster,
            name=f"ImageReplay{args.role}",
            placement_strategy=strategy,
        )
        group.init_worker().wait()
        print(
            json.dumps(
                {
                    "role": args.role,
                    "ranks": group.update_images(str(args.rows.resolve())).wait(),
                }
            ),
            flush=True,
        )
    finally:
        ray.shutdown()
