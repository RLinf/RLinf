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

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import torch
from omegaconf import DictConfig

from rlinf.scheduler import Cluster, Worker
from rlinf.workers.supermimic.distributed import DataParallel
from rlinf.workers.supermimic.evaluator import SuperMimicEvaluator
from rlinf.workers.supermimic.trainer import SuperMimicTrainer


def oom_message(cfg: DictConfig, rank: int, world_size: int, phase: str) -> str:
    """Out-of-memory message naming the rank, the phase and its per-rank sizes."""
    world = max(world_size, 1)
    return (
        f"CUDA out of memory on rank {rank} during {phase}. Per-rank sizes: "
        f"num_envs={int(cfg.env.num_envs) // world}, "
        f"wm_window_batch={int(cfg.algorithm.wm_window_batch) // world}, "
        f"imagine_batch={int(cfg.algorithm.imagine_batch) // world}, "
        f"replay_length={int(cfg.env.replay_length)}. The global sizes are split "
        "over the component_placement ranks: add GPUs to keep these batches, or "
        "run a separate smaller config. toolkits/supermimic/profile_sim.py "
        "measures the peak memory of a given size."
    )


@contextlib.contextmanager
def explain_cuda_oom(
    cfg: DictConfig, rank: int, world_size: int, phase: str
) -> Iterator[None]:
    """Re-raise a CUDA OOM with the per-rank sizes that caused it."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as err:
        raise torch.cuda.OutOfMemoryError(
            oom_message(cfg, rank, world_size, phase)
        ) from err


class SuperMimicWorker(Worker):
    """One GPU process: sim, replay, world model, policy, and critic.

    With several ranks each worker owns 1/world of the envs and replay and
    joins a torch.distributed group for the gradient all-reduce.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.trainer: SuperMimicTrainer | None = None
        self.evaluator: SuperMimicEvaluator | None = None
        self.dp: DataParallel | None = None

    def init_worker(self) -> None:
        """Pick the device, join the process group, and build the trainer or evaluator."""
        if self.torch_platform.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.torch_platform.set_device(local_rank)
            device = torch.device(self.torch_device_type, local_rank)
        else:
            device = torch.device("cpu")
        if bool(self.cfg.runner.get("only_eval", False)):
            # Evaluation needs no replay, optimizers or process group.
            self.evaluator = SuperMimicEvaluator(self.cfg, device)
            self.log_info(
                f"SuperMimic evaluator ready on {device}, "
                f"goal_dim={self.cfg.algorithm.goal_dim}, "
                f"clips={self.evaluator.motion.num_clips}"
            )
            return
        world_size = max(int(self._world_size), 1)
        rank = max(int(self._rank), 0)
        self.dp = DataParallel.initialize(
            world_size, rank, device, timeout=Cluster.get_collective_timeout()
        )
        with explain_cuda_oom(self.cfg, rank, world_size, "setup"):
            self.trainer = SuperMimicTrainer(
                self.cfg, device=device, dp=self.dp, platform=self.torch_platform
            )
        import torch.nn as nn

        dynamics = [
            module.out_features
            for module in self.trainer.models.dynamics.net
            if isinstance(module, nn.Linear)
        ]
        self.log_info(
            f"SuperMimic worker rank {rank}/{world_size} ready on {device}, "
            f"goal_dim={self.cfg.algorithm.goal_dim}, "
            f"obs_dim={self.cfg.algorithm.obs_dim}, "
            f"k={self.cfg.algorithm.k}, H={self.cfg.algorithm.H}, "
            f"lr={self.trainer.opt_actor.param_groups[0]['lr']}, "
            f"dynamics_out={dynamics}, "
            f"global_num_envs={self.cfg.env.num_envs}, "
            f"local_num_envs={self.trainer.num_envs}, "
            f"local_wm_window_batch={self.trainer.wm_window_batch}, "
            f"local_imagine_batch={self.trainer.imagine_batch}"
        )

    def _training_oom(self) -> contextlib.AbstractContextManager:
        rank = max(int(self._rank), 0)
        world_size = max(int(self._world_size), 1)
        return explain_cuda_oom(self.cfg, rank, world_size, "training")

    def run_iteration(self) -> dict[str, float]:
        """One training iteration on this rank; see ``SuperMimicTrainer.run_iteration``."""
        assert self.trainer is not None
        with self._training_oom():
            return self.trainer.run_iteration()

    def run_until_updates(self, n_updates: int) -> dict[str, float]:
        """Train this rank until ``n_updates`` global updates."""
        assert self.trainer is not None
        with self._training_oom():
            return self.trainer.run_until_updates(n_updates)

    def evaluate(self) -> dict[str, float]:
        """Evaluation metrics on rank 0; other ranks return an empty dict."""
        if self.evaluator is not None:
            if self._rank > 0:
                return {}
            seed = int(self.cfg.get("eval", {}).get("seed", 0))
            metrics, _ = self.evaluator.evaluate_motion(seed=seed)
            if self.evaluator.velocity:
                velocity, _, _ = self.evaluator.evaluate_velocity(seed=seed)
                metrics.update(velocity)
            return metrics
        assert self.trainer is not None
        return self.trainer.evaluate()

    def save_checkpoint(self, directory: str) -> None:
        """Write this rank's part of the checkpoint to ``directory``."""
        assert self.trainer is not None
        self.trainer.save_checkpoint(directory)

    def load_checkpoint(self, directory: str) -> None:
        """Load weights (and, for training, optimizer and RNG state) from ``directory``."""
        if self.evaluator is not None:
            self.evaluator.load(directory)
            return
        assert self.trainer is not None
        self.trainer.load_checkpoint(directory)

    def shutdown(self) -> None:
        """Release the process group so Ray can reuse the GPU."""
        if self.dp is not None:
            self.dp.destroy()
            self.dp = None
