import hydra
import torch.multiprocessing as mp

from rlinf.config import validate_cfg
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import EmbodiedComponentPlacement
from rlinf.workers.actor.fsdp_actor_worker import FSDPActor
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MutilStepRolloutWorker

mp.set_start_method("spawn", force=True)

@hydra.main(
    version_base="1.1", config_path="config", config_name="maniskill_ppo_openvlaoft"
)
def main(cfg) -> None:

    cfg = validate_cfg(cfg)

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes, num_gpus_per_node=cfg.cluster.num_gpus_per_node)
    component_placement = EmbodiedComponentPlacement(cfg)

    # Create actor worker group
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPActor.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    # Create rollout worker group
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = MutilStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = EnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()

if __name__ == "__main__":
    main()
