import socket
from typing import Any, Optional, cast
from datetime import datetime
import uuid

import hydra
import agentlightning as agl
from datasets import Dataset as HuggingFaceDataset
from agentlightning.env_var import LightningEnvVar, resolve_bool_env_var, resolve_str_env_var

from rlinf.utils.utils import output_redirector
from rlinf.entrypoint.agentlightning.algorithm import RlinfAlgorithm
from calc_agent import MathProblem, calc_agent


def _find_available_port() -> int:
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def train(cfg: Any):
    train_data_paths = cfg.data.get("train_data_paths", None)
    val_data_paths = cfg.data.get("val_data_paths", None)
    assert train_data_paths, "cfg.data.train_data_paths is required and cannot be empty."
    assert val_data_paths, "cfg.data.val_data_paths is required and cannot be empty."

    train_file = train_data_paths[0]
    val_file = val_data_paths[0]
    assert str(train_file).endswith(".parquet"), (
        f"Only parquet files are supported for train_data_paths, got: {train_file}"
    )
    assert str(val_file).endswith(".parquet"), (
        f"Only parquet files are supported for val_data_paths, got: {val_file}"
    )

    n_runners = cfg.agentlightning.n_runners

    train_dataset = cast(agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet(train_file).to_list())
    val_dataset = cast(agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet(val_file).to_list())

    eval_mode = cfg.get("eval", False)
    eval_checkpoint_dir = cfg.get("eval_checkpoint_dir", None)

    algorithm = RlinfAlgorithm(config=cfg, eval=eval_mode, eval_checkpoint_dir=eval_checkpoint_dir)
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
