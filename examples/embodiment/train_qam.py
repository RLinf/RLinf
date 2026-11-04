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


import hydra

from rlinf.runners.qam_runner import QAMRunner
from rlinf.scheduler import Cluster, PackedPlacementStrategy
from rlinf.workers.actor.qam_policy_worker import QAMPolicyWorker


def launch_qam(cfg):
    """Launch independent seeds and return the initialized runner and group."""
    if cfg.algorithm.loss_type != "qam":
        raise ValueError("train_qam requires algorithm.loss_type=qam")
    if cfg.gpu_end - cfg.gpu_start + 1 != len(cfg.seeds):
        raise ValueError("QAM requires one accelerator per independent seed")
    if cfg.runner.max_steps != cfg.runner.offline_steps + cfg.runner.online_steps:
        raise ValueError("max_steps must equal offline_steps + online_steps")
    cluster = Cluster(num_nodes=1)
    group = QAMPolicyWorker.create_group(cfg).launch(
        cluster=cluster,
        name=cfg.runner.logger.experiment_name,
        placement_strategy=PackedPlacementStrategy(cfg.gpu_start, cfg.gpu_end),
    )
    runner = QAMRunner(cfg, group, None, None)
    runner.init_workers()
    return runner, group


@hydra.main(version_base="1.1", config_path="config", config_name="ogbench_qam_antmaze")
def main(cfg):
    runner, group = launch_qam(cfg)
    try:
        runner.run()
    finally:
        group.close().wait()


if __name__ == "__main__":
    main()
