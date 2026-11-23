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


import json
import subprocess
from multiprocessing.connection import Pipe
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.1", config_path=".", config_name="rmbench_openpi_pi05_eval")
def main(cfg: DictConfig) -> None:
    """Load the RLinf policy and serve the local RMBench evaluation process."""
    rmbench_path = Path(cfg.rmbench_path).expanduser().resolve()
    for relative_path in ("assets/embodiments/aloha-agilex", "assets/objects"):
        if not (rmbench_path / relative_path).is_dir():
            raise FileNotFoundError(
                f"Missing RMBench assets at {rmbench_path / relative_path}. "
                "Run bash scripts/_download_assets.sh from the RMBench checkout."
            )

    model_cfg = cfg.rollout.model
    log_dir = Path(cfg.runner.logger.log_path).resolve()
    sim_cfg = {
        "task_name": cfg.task_name,
        "episodes": cfg.episodes,
        "seed": cfg.seed,
        "instruction_type": cfg.instruction_type,
        "save_video": cfg.save_video,
        "log_dir": str(log_dir),
        "short_memory": (
            OmegaConf.to_container(cfg.short_memory, resolve=True)
            if OmegaConf.select(model_cfg, "openpi.short_memory", default=False)
            else None
        ),
    }

    import torch
    from torch.utils._pytree import tree_map

    from rlinf.models import get_model

    model = get_model(model_cfg).to(cfg.device).eval()
    policy_conn, sim_conn = Pipe()
    with policy_conn, sim_conn:
        command = [
            str(Path(cfg.sim_python).expanduser().absolute()),
            "-m",
            "rlinf.envs.sim.rmbench.sim_worker",
            str(sim_conn.fileno()),
            json.dumps(sim_cfg),
        ]
        with (
            subprocess.Popen(
                command,
                cwd=rmbench_path,
                pass_fds=(sim_conn.fileno(),),
            ) as process,
            policy_conn,
        ):
            sim_conn.close()
            rng = torch.Generator(device=cfg.device)
            active_seed = None
            while process.poll() is None:
                if not policy_conn.poll(1):
                    continue
                try:
                    seed, obs = policy_conn.recv()
                except EOFError:
                    break
                if seed != active_seed:
                    rng.manual_seed(seed)
                    active_seed = seed
                obs = tree_map(
                    lambda value: torch.from_numpy(value)
                    if isinstance(value, np.ndarray)
                    else value,
                    obs,
                )
                actions, _ = model.predict_action_batch(obs, rng=rng)
                policy_conn.send(actions[0].cpu().numpy())
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)


if __name__ == "__main__":
    main()
