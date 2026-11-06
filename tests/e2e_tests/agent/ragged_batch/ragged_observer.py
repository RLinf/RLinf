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

"""Capture rollout fixtures and phase metrics for paired diagnostic runs."""

import functools
import inspect
import json
import os
import time
from pathlib import Path

ROOT = Path(os.environ["RAGGED_RUN_DIR"])
REGISTERED = False


def append(name, row):
    with (ROOT / "evidence" / name).open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")


def install_actor(cls):
    import torch
    from megatron.core import parallel_state

    import rlinf.workers.actor.ma_megatron_actor_worker as mod

    def memory(self):
        p = self.torch_platform
        return {
            "allocated": p.memory_allocated(),
            "reserved": p.memory_reserved(),
            "peak_allocated": p.max_memory_allocated(),
            "peak_reserved": p.max_memory_reserved(),
        }

    def capture(self, result):
        count = (
            self.cfg.data.rollout_batch_size
            // parallel_state.get_data_parallel_world_size()
        )
        index = getattr(self, "capture_count", 0)
        self.capture_count = index + 1
        self.observed_step = index // count + 1
        if self.is_data_io_rank:
            directory = (
                ROOT
                / "evidence"
                / "raw"
                / f"step{index // count + 1:02d}_rank{self._rank:02d}"
            )
            directory.mkdir(parents=True, exist_ok=True)
            torch.save(result, directory / f"{index % count:04d}.pt")

    def timed(name):
        original = getattr(cls, name)

        @functools.wraps(original)
        def run(self, *args, **kwargs):
            if name == "run_inference":
                self.observed_step = getattr(self, "observed_step", 0) + 1
            if name == "run_training" and not self.cfg.algorithm.recompute_logprobs:
                self.observed_step = getattr(self, "observed_step", 0) + 1
            previous = getattr(self, "observed_phase", None)
            self.observed_phase = name
            self.torch_platform.synchronize()
            self.torch_platform.reset_peak_memory_stats()
            start = time.monotonic()
            result = original(self, *args, **kwargs)
            self.torch_platform.synchronize()
            append(
                f"phases_rank{self._rank:02d}.jsonl",
                dict(
                    step=getattr(self, "observed_step", 0),
                    phase=name,
                    seconds=time.monotonic() - start,
                    **memory(self),
                ),
            )
            self.observed_phase = previous
            return result

        setattr(cls, name, run)

    old_get = cls.get_batch

    def get(self, channel):
        self.observed_phase = (
            "run_inference"
            if self.cfg.algorithm.recompute_logprobs and not channel.is_local
            else "run_training"
        )
        batch, result = old_get(self, channel)
        if not channel.is_local:
            capture(self, result)
        return batch, result

    cls.get_batch = get
    old_ragged = cls.get_ragged_batches

    def ragged(self, channel):
        self.observed_phase = (
            "run_inference"
            if self.cfg.algorithm.recompute_logprobs and not channel.is_local
            else "run_training"
        )
        if not channel.is_local:
            self.observed_step = getattr(self, "ragged_seen", 0) + 1
            self.ragged_seen = self.observed_step
        if channel.is_local:
            return old_ragged(self, channel)

        class Proxy:
            is_local = False

            def get(proxy):
                value = channel.get()
                capture(self, value)
                return value

        return old_ragged(self, Proxy())

    cls.get_ragged_batches = ragged
    old_broadcast = cls.broadcast

    def broadcast(self, *args, **kwargs):
        start = time.monotonic()
        ret = old_broadcast(self, *args, **kwargs)
        if getattr(self, "observed_phase", None):
            self.torch_platform.synchronize()
            append(
                f"communication_rank{self._rank:02d}.jsonl",
                {
                    "step": getattr(self, "observed_step", 1),
                    "phase": getattr(self, "observed_phase", "preparation"),
                    "operation": "model_broadcast",
                    "seconds": time.monotonic() - start,
                },
            )
        return ret

    cls.broadcast = broadcast
    old_balance = cls._dp_load_balance_dynamic

    def balance(self, *args, **kwargs):
        self.torch_platform.synchronize()
        self.torch_platform.reset_peak_memory_stats()
        start = time.monotonic()
        before = memory(self)
        ret = old_balance(self, *args, **kwargs)
        self.torch_platform.synchronize()
        append(
            f"communication_rank{self._rank:02d}.jsonl",
            {
                "step": getattr(self, "observed_step", 1),
                "phase": getattr(self, "observed_phase", "preparation"),
                "operation": "dp_balance",
                "seconds": time.monotonic() - start,
                "before": before,
                "after": memory(self),
            },
        )
        return ret

    cls._dp_load_balance_dynamic = balance
    old_groups = mod.balance_rollout_groups

    def groups(results, group):
        from rlinf.scheduler import Worker

        p = Worker.torch_platform
        p.synchronize()
        p.reset_peak_memory_stats()
        before = p.memory_allocated()
        start = time.monotonic()
        ret = old_groups(results, group)
        p.synchronize()
        append(
            f"communication_rank{torch.distributed.get_rank():02d}.jsonl",
            {
                "operation": "ragged_dp_balance",
                "seconds": time.monotonic() - start,
                "groups": len(ret),
                "input_tokens": sum(sum(map(len, x.input_ids)) for x in results),
                "allocated_before": before,
                "peak_allocated": p.max_memory_allocated(),
            },
        )
        return ret

    mod.balance_rollout_groups = groups
    for name in ("inference_step", "training_step"):
        timed(name)


def install_driver():
    import rlinf.runners.agent_runner as agent_runner_module
    from rlinf.runners.agent_runner import AgentRunner
    from rlinf.utils.metric_logger import MetricLogger

    original_progress = agent_runner_module.check_progress

    def progress(*args, **kwargs):
        evaluate, save, finished = original_progress(*args, **kwargs)
        return evaluate, save, False

    # The prepared fixture has one epoch; record its final batch before epoch exit.
    agent_runner_module.check_progress = progress
    original = MetricLogger.log

    def log(self, data, step, *args, **kwargs):
        append("metrics.jsonl", {"step": step, "metrics": data})
        return original(self, data, step, *args, **kwargs)

    MetricLogger.log = log
    original_put = AgentRunner._put_batch

    def put(self, batch, *args, **kwargs):
        import torch

        torch.save(batch, ROOT / "evidence" / f"dataset_step{self.global_steps:03d}.pt")
        return original_put(self, batch, *args, **kwargs)

    AgentRunner._put_batch = put


def register():
    global REGISTERED
    if REGISTERED or os.environ.get("GROUP_NAME") != "ActorGroup":
        return
    frame = inspect.currentframe().f_back
    while frame is not None:
        owner = frame.f_locals.get("self")
        if owner is not None and hasattr(owner, "_worker_address"):
            install_actor(type(owner))
            REGISTERED = True
            return
        frame = frame.f_back
