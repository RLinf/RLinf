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

import socket
from typing import Any, cast

import agentlightning as agl
import hydra
from calc_agent import MathProblem, calc_agent
from datasets import Dataset as HuggingFaceDataset

from rlinf.agents.agentlightning.algorithm import RLinf
from rlinf.utils.utils import output_redirector


def _find_available_port() -> int:
    """Find an available port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def train(cfg: Any):
    train_data_paths = cfg.data.get("train_data_paths", None)
    val_data_paths = cfg.data.get("val_data_paths", None)
    assert train_data_paths, (
        "cfg.data.train_data_paths is required and cannot be empty."
    )
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
        agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet(train_file).to_list()
    )
    val_dataset = cast(
        agl.Dataset[MathProblem], HuggingFaceDataset.from_parquet(val_file).to_list()
    )

    eval_mode = cfg.get("eval", False)

    algorithm = RLinf(config=cfg, eval=eval_mode)

    trainer_kwargs = {
        "algorithm": algorithm,
        "n_runners": n_runners,
        "store": None,  # store=None -> InMemoryLightningStore by default;
        "llm_proxy": None,  # llm_proxy=None stays unset (no default LLMProxy).
    }
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
