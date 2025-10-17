import json

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.runners.test_mcp_runner import MCPTestRunner
from rlinf.workers.mcp.filesystem_worker import MCPFilesystemClientWorker

mp.set_start_method("spawn", force=True)

@hydra.main(version_base="1.1")
def main(cfg) -> None:
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(num_nodes=cfg.cluster.num_nodes)

    client_placement = NodePlacementStrategy([0])
    client_group = MCPFilesystemClientWorker.create_group(cfg).launch(
        cluster, name=cfg.client.group_name, placement_strategy=client_placement
    )

    runner = MCPTestRunner(cfg, client_group)
    runner.init_workers()
    runner.run()
    runner.cleanup()

if __name__ == "__main__":
    main()