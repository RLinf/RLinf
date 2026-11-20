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

import torch
from megatron.core import parallel_state
from megatron.core.distributed import finalize_model_grads
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.data.schema.reasoning_results import DynamicRolloutResult
from rlinf.scheduler import Cluster
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.actor.ma_megatron_actor_worker import MAMegatronActor
from rlinf.workers.critic.ma_megatron_critic_worker import MAMegatronCritic


def update(worker, rows_file, critic):
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
    return dict(
        rank=worker._rank, loss=loss.item(), grad_norm=float(norm), changed=changed
    )


class ReplayActor(MAMegatronActor):
    def update_images(self, rows_file):
        return update(self, rows_file, False)


class ReplayCritic(MAMegatronCritic):
    def update_images(self, rows_file):
        return update(self, rows_file, True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--role", choices=["dense", "chunked", "critic"], required=True)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    cfg = OmegaConf.merge(
        OmegaConf.load(here.parent / "searchr1/config/train_qwen2.5_ppo.yaml"),
        OmegaConf.load(here / "replay_update.yaml"),
    )
    cfg.rollout.model.model_path = str(args.model.resolve())
    cards = torch.cuda.device_count()
    if cards not in (2, 8):
        parser.error("Select two or eight local GPUs with CUDA_VISIBLE_DEVICES")
    cfg.cluster.component_placement = {"actor,critic,rollout,reward": "all"}
    cfg.data.rollout_batch_size = 4
    cfg.actor.global_batch_size = 8
    cfg.actor.model.lm_head_chunk_size = 256 if args.role == "chunked" else 0
    cfg.critic = OmegaConf.merge(
        cfg.actor,
        {
            "group_name": "CriticGroup",
            "use_critic_model": True,
            "model": {"lm_head_chunk_size": 0},
        },
    )
    cfg = validate_cfg(cfg)
    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = ModelParallelComponentPlacement(cfg, cluster)
    role = "critic" if args.role == "critic" else "actor"
    if placement.get_world_size(role) != cards:
        raise RuntimeError("Use a single-node Ray cluster with only the selected GPUs.")
    cls = ReplayCritic if role == "critic" else ReplayActor
    group = cls.create_group(cfg, placement).launch(
        cluster,
        name=f"ImageReplay{args.role}",
        placement_strategy=placement.get_strategy(role),
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
