# Copyright 2025 The RLinf Authors.
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
import signal
import time

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from rlinf.config import validate_cfg
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv
from rlinf.scheduler import Cluster, Worker
from rlinf.utils.placement import HybridComponentPlacement

mp.set_start_method("spawn", force=True)


class Timeout:
    def __init__(self, seconds):
        self.seconds = seconds
        self.old_handler = None

    def __enter__(self):
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Operation timed out after {self.seconds} seconds")

        self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(self.seconds)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        signal.alarm(0)
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)
        return False


class RobWorker(Worker):
    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg

    def init_worker(self):
        # self.env = RoboTwinEnv(
        #     cfg=self.cfg.env.eval,
        #     seed_offset=self._rank,
        # )
        pass

    def collect_robotwin_seeds(self, start_seed, end_seed, target_num):
        total_seeds = end_seed - start_seed + 1
        seeds_per_rank = total_seeds // self._world_size

        ranked_start_seed = start_seed + self._rank * seeds_per_rank
        ranked_end_seed = ranked_start_seed + seeds_per_rank - 1
        target_num_per_rank = target_num // self._world_size

        successs_seeds = []
        unsuccesss_seeds = []

        pbar = tqdm(
            range(target_num_per_rank),
            desc=f"[Worker {self._rank}] Collecting robotwin2 seeds",
        )
        for s in range(ranked_start_seed, ranked_end_seed, self.cfg.env.eval.num_envs):
            if len(successs_seeds) >= target_num_per_rank:
                break

            seeds = list(range(s, min(s + self.cfg.env.eval.num_envs, ranked_end_seed)))
            t1 = time.time()
            resutls = None
            env = None
            try:
                env = RoboTwinEnv(
                    cfg=self.cfg.env.eval,
                    seed_offset=self._rank,
                )
                t2 = time.time()
                print(
                    f"RobWorker envinit rank={self._rank} cost time={t2 - t1}",
                    flush=True,
                )
                with Timeout(seconds=360):
                    resutls = env.check_seeds(seeds=seeds)
                t3 = time.time()
                print(
                    f"RobWorker check_seeds rank={self._rank} cost time={t3 - t2}",
                    flush=True,
                )
            except TimeoutError:
                print(f"RobWorker timeout rank={self._rank}", flush=True)
            except Exception:
                pass

            check_status = [False] * self.cfg.env.eval.num_envs
            cost_time = [0] * self.cfg.env.eval.num_envs
            if resutls is not None:
                for i, res in enumerate(resutls):
                    if res is not None:
                        check_status[i] = res["status"]
                        cost_time[i] = res["cost_time"]

            t4 = time.time()
            print(
                f"RobWorker envinit + check_seeds rank={self._rank}, cost time={t4 - t1}, {seeds=}, {check_status=}",
                flush=True,
            )
            if env is not None:
                try:
                    with Timeout(seconds=30):
                        env.close()
                except TimeoutError:
                    print(
                        f"RobWorker env.close() timeout rank={self._rank}", flush=True
                    )
                except Exception:
                    pass
            env = None
            # gc.collect()
            # torch.cuda.empty_cache()

            count = 0
            for status, cost, seed in zip(check_status, cost_time, seeds):
                if status and cost < 30:
                    count += 1
                    successs_seeds.append(seed)
                else:
                    unsuccesss_seeds.append(seed)
            if count > 0:
                pbar.update(count)

        collected_seeds = {
            "successs_seeds": successs_seeds,
            "unsuccesss_seeds": unsuccesss_seeds,
        }

        return collected_seeds

    def check_seeds(self, seeds):
        return self.env.check_seeds(seeds=seeds)


@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))
    cfg.runner.only_eval = True

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = RobWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    start_seed = 10000
    end_seed = 20000
    target_num = 128

    collected_seeds_list = env_group.collect_robotwin_seeds(
        start_seed=start_seed, end_seed=end_seed, target_num=target_num
    ).wait()

    successs_seeds = []
    unsuccesss_seeds = []
    for collected_seeds in collected_seeds_list:
        successs_seeds.extend(collected_seeds["successs_seeds"])
        unsuccesss_seeds.extend(collected_seeds["unsuccesss_seeds"])

    task_name = cfg.env.eval.task_config.task_name
    collected_task_seeds = {
        "successs_seeds": successs_seeds,
        "unsuccesss_seeds": unsuccesss_seeds,
    }

    seeds_info = {
        task_name: collected_task_seeds,
    }

    with open(f"robotwin2_train_seeds_{task_name}_thread_test.json", "w") as f:
        json.dump(seeds_info, f, indent=4)

    print(
        f"✓ Successfully collected {len(successs_seeds)} seeds, {len(unsuccesss_seeds)} seeds failed"
    )


if __name__ == "__main__":
    main()
