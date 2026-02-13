import sys
import os
import socket
from typing import Any, Optional, cast
from datetime import datetime
import uuid

import hydra
import agentlightning as agl
from datasets import Dataset as HuggingFaceDataset
from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var, resolve_str_env_var

from rlinf.utils.utils import output_redirector

entrypoint_path = os.path.join(os.path.dirname(__file__), '..', 'entrypoint')
if entrypoint_path not in sys.path:
    sys.path.insert(0, entrypoint_path)
from interface_algorithm import RLinf
from calc_agent import MathProblem, calc_agent


def _find_available_port() -> int:
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def train(cfg: Any):
    train_file = cfg.data.train_data_paths[0]
    val_file = cfg.data.val_data_paths[0]

    n_runners = cfg.agentlightning.n_runners

    train_dataset = cast(agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet(train_file).to_list())
    val_dataset = cast(agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet(val_file).to_list())

    eval_mode = cfg.get("eval", False)
    eval_checkpoint_dir = cfg.get("eval_checkpoint_dir", None)

    algorithm = RLinf(config=cfg, eval=eval_mode, eval_checkpoint_dir=eval_checkpoint_dir)
    store = None
    llm_proxy = None

    trainer_kwargs = {"algorithm": algorithm, "n_runners": n_runners, "store": store, "llm_proxy": llm_proxy}
    trainer_kwargs["port"] = _find_available_port()

    trainer = agl.Trainer(**trainer_kwargs)

    trainer.fit(calc_agent, train_dataset, val_dataset=val_dataset)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    agl.setup_logging("INFO")
    train(cfg)


if __name__ == "__main__":
    main()
