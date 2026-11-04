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


import copy
import csv
import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.qam import qam_update
from rlinf.data.datasets.ogbench import OGBenchDataset
from rlinf.envs import get_env_cls
from rlinf.models import get_model
from rlinf.scheduler import Worker


class QAMPolicyWorker(Worker):
    """Own model, optimizer and replay; runner owns scheduling and evaluation.

    Ranks are independent experiments, not data-parallel replicas. Colocated
    OGBench collection avoids sending one tiny observation over Ray per env step.
    FSDP/DDP training of one QAM seed is not implemented by this worker.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.global_step = 0
        self.updates = 0

    def init_worker(self):
        self.seed = int(self.cfg.seeds[self._rank])
        torch.set_num_threads(4)
        torch.manual_seed(self.seed)
        self.device = torch.device(
            self.torch_device_type
            if self.torch_platform and self.torch_platform.is_available()
            else "cpu"
        )
        self.generator = torch.Generator(device=self.device).manual_seed(self.seed)
        self.dataset = OGBenchDataset.from_task(
            self.cfg.data.task_name, self.seed, int(self.cfg.runner.online_steps)
        )
        model_cfg = self.cfg.actor.model
        obs_dim = self.dataset.data["observations"].shape[-1]
        action_dim = self.dataset.data["actions"].shape[-1]
        if int(model_cfg.obs_dim) != obs_dim or int(model_cfg.action_dim) != action_dim:
            raise ValueError(
                f"QAM model dimensions ({model_cfg.obs_dim}, {model_cfg.action_dim}) "
                f"do not match OGBench data ({obs_dim}, {action_dim})"
            )
        self.model = get_model(model_cfg).to(self.device)
        self.optimizer = torch.optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.actor.lr,
            eps=1e-8,
        )
        self.eval_model = copy.deepcopy(self.model).eval()
        env_cls = get_env_cls(self.cfg.env.env_type, self.cfg.env)
        self.env = env_cls(self.cfg.env)
        self.eval_env = env_cls(self.cfg.env)
        self.observation = None
        self.action_queue = deque()
        self.run_dir = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / f"seed_{self.seed}"
        )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(OmegaConf.to_container(self.cfg, resolve=True), indent=2)
        )
        self.start_time = time.time()
        self._status("initialized")
        self.log_info(
            f"QAM seed={self.seed}, device={self.device}, dataset={self.dataset.size}"
        )
        return {
            "seed": self.seed,
            "dataset_size": self.dataset.size,
            "device": str(self.device),
            "pid": os.getpid(),
        }

    def _status(self, phase):
        payload = {
            "seed": self.seed,
            "pid": os.getpid(),
            "phase": phase,
            "step": self.global_step,
            "updates": self.updates,
            "updated_unix": time.time(),
        }
        temp = self.run_dir / "status.tmp"
        temp.write_text(json.dumps(payload, indent=2))
        temp.replace(self.run_dir / "status.json")

    def _append(self, filename, row):
        path = self.run_dir / filename
        exists = path.exists()
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)

    def _update(self):
        batch = self.dataset.sample(
            self.cfg.actor.batch_size, self.model.horizon, self.model.cfg.discount
        )
        batch = {k: torch.as_tensor(v, device=self.device) for k, v in batch.items()}
        info = qam_update(self.model, self.optimizer, batch, self.generator)
        self.updates += 1
        return info

    def _online_step(self):
        if self.observation is None:
            self.observation, _ = self.env.reset(seed=self.seed)
        if not self.action_queue:
            obs = torch.as_tensor(
                self.observation[None], device=self.device, dtype=torch.float32
            )
            actions = self.model.sample_actions(obs, generator=self.generator)
            self.action_queue.extend(
                actions.cpu().numpy().reshape(self.model.horizon, -1)
            )
        action = self.action_queue.popleft()
        nxt, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.dataset.add(
            {
                "observations": self.observation,
                "actions": action,
                "rewards": reward,
                "next_observations": nxt,
                "masks": 1.0 - float(terminated),
                "terminals": float(done),
            }
        )
        self.observation = nxt
        if done:
            self.observation, _ = self.env.reset()
            self.action_queue.clear()

    @Worker.timer("run_training")
    def run_training(self):
        stop = min(
            self.global_step + int(self.cfg.actor.updates_per_call),
            int(self.cfg.runner.max_steps),
        )
        # Never cross an evaluation, save, or offline/online boundary in one call.
        for interval in (
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            self.cfg.runner.offline_steps,
        ):
            if interval > 0:
                stop = min(stop, (self.global_step // interval + 1) * interval)
        sums, count = {}, 0
        phase = "offline"
        for step in range(self.global_step + 1, stop + 1):
            phase = "offline" if step <= self.cfg.runner.offline_steps else "online"
            info = None
            if phase == "offline":
                info = self._update()
            else:
                self._online_step()
                online_step = step - self.cfg.runner.offline_steps
                if online_step >= self.cfg.runner.start_training:
                    info = self._update()
            if info is not None:
                count += 1
                for k, v in info.items():
                    sums[k] = sums.get(k, 0) + v
            self.global_step = step
        metrics = {k: float(v / count) for k, v in sums.items()} if count else {}
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError(f"Non-finite QAM metrics: {metrics}")
        self._append(
            "train.csv",
            dict(
                step=self.global_step,
                phase=phase,
                updates=self.updates,
                elapsed_seconds=time.time() - self.start_time,
                **{
                    k: metrics.get(k, 0.0)
                    for k in (
                        "critic_loss",
                        "flow_loss",
                        "adj_loss",
                        "q_mean",
                        "adj_max",
                        "total_loss",
                        "grad_norm",
                    )
                },
            ),
        )
        self._status(phase)
        return {**metrics, "__global_step": self.global_step}

    def set_global_step(self, step):
        if self.global_step != int(step):
            raise ValueError(
                f"Runner step {step} differs from worker step {self.global_step}"
            )

    def get_policy_state_dict(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def prepare_evaluation(self):
        self.eval_model.load_state_dict(self.model.state_dict())

    def evaluate(self):
        self._status("evaluating")
        # Evaluation has an isolated RNG stream, so changing eval frequency does
        # not perturb the learner or replay sampler.
        devices = (
            [self.torch_platform.current_device()] if self.device.type != "cpu" else []
        )
        with torch.random.fork_rng(devices=devices, device_type=self.device.type):
            torch.manual_seed(self.seed + self.global_step + 1000000)
            result = self.eval_env.evaluate(
                self.eval_model, self.cfg.runner.eval_episodes, self.seed + 1000000
            )
        self._append("eval.csv", dict(seed=self.seed, step=self.global_step, **result))
        self._status(
            "offline" if self.global_step <= self.cfg.runner.offline_steps else "online"
        )
        self.log_info(
            f"seed={self.seed}, step={self.global_step}, success={result['success']}"
        )
        return result

    def save_checkpoint(self, path, global_step):
        if int(global_step) != self.global_step:
            raise ValueError("Checkpoint step must match worker step")
        path = Path(path) / f"seed_{self.seed}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.get_policy_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "generator": self.generator.get_state(),
            "replay": self.dataset.state_dict(),
            "step": self.global_step,
            "updates": self.updates,
            "seed": self.seed,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
        }
        # Online state includes simulator and action queue for exact continuation.
        if self.observation is not None:
            payload["online_env"] = self.env.state_dict()
            payload["observation"] = self.observation
            payload["action_queue"] = list(self.action_queue)
        temp = path.with_suffix(".tmp")
        torch.save(payload, temp)
        temp.replace(path)
        return str(path)

    def load_checkpoint(self, path):
        path = Path(path) / f"seed_{self.seed}.pt"
        state = torch.load(path, map_location=self.device, weights_only=False)
        if state["seed"] != self.seed:
            raise ValueError("Checkpoint seed mismatch")
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.generator.set_state(state["generator"].cpu())
        self.dataset.load_state_dict(state["replay"])
        self.global_step, self.updates = state["step"], state["updates"]
        if "online_env" in state:
            self.env.load_state_dict(state["online_env"])
            self.observation = state["observation"]
            self.action_queue = deque(state["action_queue"])
        self._status("restored")
        return self.global_step

    def close(self):
        self.env.close()
        self.eval_env.close()
        self._status("completed")
