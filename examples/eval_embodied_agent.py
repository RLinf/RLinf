import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import EmbodiedComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MutilStepRolloutWorker

mp.set_start_method("spawn", force=True)

@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:

    cfg = validate_cfg(cfg)
    cfg.runner.only_eval = True

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes, num_gpus_per_node=cfg.cluster.num_gpus_per_node)
    component_placement = EmbodiedComponentPlacement(cfg)

    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MutilStepRolloutWorker(cfg).create_group(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker(cfg).create_group(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    rollout_group.init_worker().wait()
    env_group.init_worker().wait()

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.run()

if __name__ == "__main__":
    main()
