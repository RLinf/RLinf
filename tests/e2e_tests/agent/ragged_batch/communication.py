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

"""Replay dense and ragged DP exchange on the same frozen rollout groups."""

import argparse
import gc
import json
import os
import pickle
import time
from pathlib import Path


def main() -> None:
    """Measure DP exchange without running a model or recomputing probabilities."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--seq-length", type=int, default=8192)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    import torch
    import torch.distributed as dist

    from rlinf.algorithms.advantages import compute_grpo_turn_advantages
    from rlinf.algorithms.loss_scales import agent_scale, group_scale, turn_scale
    from rlinf.data.schema.reasoning_results import DynamicRolloutResult as Result
    from rlinf.scheduler import Worker
    from rlinf.utils.data_iter_utils import get_seqlen_balanced_partitions
    from rlinf.utils.distributed import RolloutDataBalance, balance_rollout_groups

    platform = Worker.torch_platform
    platform.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size % args.tp_size == 0, "World size must be divisible by TP size"
    dp_size, dp_rank = world_size // args.tp_size, rank // args.tp_size
    groups = [
        dist.new_group(list(range(tp, world_size, args.tp_size)), backend="nccl")
        for tp in range(args.tp_size)
    ]
    group = groups[rank % args.tp_size]
    source = args.raw_dir / f"step01_rank{dp_rank * args.tp_size:02d}"
    paths = sorted(source.glob("*.pt"))
    assert paths, f"Expected frozen question groups in {source}"
    raw = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    ragged = Result.merge_batches(
        [item.to_ragged_actor_batch() for item in raw], args.group_size
    )
    ragged["recomputed_logprobs"] = [
        torch.zeros(len(row)) for row in ragged["input_ids"]
    ]
    values = compute_grpo_turn_advantages(
        ragged["rewards"], args.group_size, ragged["idx_to_traj"], "trajectory"
    )
    ragged["advantages"] = [
        mask.float() * value for mask, value in zip(ragged["response_mask"], values)
    ]
    ragged["loss_scales"] = [mask.float() for mask in ragged["response_mask"]]
    batch = Result.materialize_ragged_batch(ragged, args.seq_length, 0)
    context = {
        "folding_scale": [],
        "actor_global_batch_size": len(raw) * args.group_size * dp_size // 4,
        "data_parallel_world_size": dp_size,
    }
    for function in (group_scale, agent_scale, turn_scale):
        batch = function(context, batch)
    batch = Result.pack_traj_batch(context, batch)
    batch = {
        key: value
        for key, value in batch.items()
        if key != "idx_to_traj" and not key.startswith("extra:")
    }
    pad = {
        key: value.cpu()
        for key, value in Result.get_batch_pad(args.seq_length, batch.keys()).items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for trial in range(args.trials):
        for mode in ("baseline", "ragged"):
            dist.barrier()
            gc.collect()
            platform.empty_cache()
            platform.synchronize()
            platform.reset_peak_memory_stats()
            before, started = platform.memory_allocated(), time.monotonic()
            result = None
            if mode == "baseline":
                result = RolloutDataBalance.from_rollout_batches_dynamic(
                    batch,
                    dp_size,
                    dp_rank,
                    group,
                    pad,
                    1,
                    get_seqlen_balanced_partitions,
                )
            elif rank % args.tp_size == 0:
                result = balance_rollout_groups(raw, group)
            platform.synchronize()
            records.append(
                {
                    "trial": trial,
                    "mode": mode,
                    "rank": rank,
                    "seconds": time.monotonic() - started,
                    "allocated_before": before,
                    "peak_increment_bytes": platform.max_memory_allocated() - before,
                    "dense_payload_bytes": sum(
                        v.numel() * v.element_size() for v in batch.values()
                    ),
                    "raw_serialized_bytes": len(pickle.dumps(raw, protocol=5)),
                }
            )
            del result
    (args.output / f"rank{rank:02d}.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
