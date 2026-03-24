"""Spider text-to-SQL：RLinf + AgentLightning + Hydra（与 calc_x/main.py 同构）。"""

from __future__ import annotations

import socket
from typing import Any, Dict, Optional, cast

import hydra
import agentlightning as agl
from datasets import Dataset as HuggingFaceDataset

from rlinf.utils.utils import output_redirector
from rlinf.entrypoint.agentlightning.algorithm import RlinfAlgorithm

from sql_agent import LitSQLAgent


def _find_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def train(cfg: Any) -> None:
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

    train_dataset = cast(
        agl.Dataset[Dict[str, Any]],
        HuggingFaceDataset.from_parquet(train_file).to_list(),
    )
    val_dataset = cast(
        agl.Dataset[Dict[str, Any]],
        HuggingFaceDataset.from_parquet(val_file).to_list(),
    )

    eval_mode = cfg.get("eval", False)
    eval_checkpoint_dir: Optional[str] = cfg.get("eval_checkpoint_dir", None)

    algorithm = RlinfAlgorithm(config=cfg, eval=eval_mode, eval_checkpoint_dir=eval_checkpoint_dir)
    trainer = agl.Trainer(
        algorithm=algorithm,
        n_runners=n_runners,
        store=None,
        llm_proxy=None,
        port=_find_available_port(),
    )

    agent = LitSQLAgent()
    trainer.fit(agent, train_dataset, val_dataset=val_dataset)


@hydra.main(version_base="1.1")
@output_redirector
def main(cfg) -> None:
    agl.setup_logging("INFO")
    train(cfg)


if __name__ == "__main__":
    main()
